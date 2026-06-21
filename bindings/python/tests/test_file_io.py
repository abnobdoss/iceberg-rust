# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import threading

import pytest

from pyiceberg_core.file_io import FileIO, InputFile, OutputFile


def local_fio() -> FileIO:
    return FileIO.from_props({})


def file_uri(path) -> str:
    # Path.as_uri() builds a portable file:// URI (correct on Windows too), unlike a
    # hand-assembled f"file://{path}".
    return path.as_uri()


def test_from_props_returns_independent_handles():
    a = FileIO.from_props({})
    b = FileIO.from_props({"s3.region": "us-east-1"})
    assert isinstance(a, FileIO)
    assert isinstance(b, FileIO)
    assert a is not b


@pytest.mark.parametrize(
    "key,value",
    [
        ("access-key-id", "AKIAIOSFODNN7EXAMPLE"),
        ("s3.access-key-id", "AKIAIOSFODNN7EXAMPLE"),
        ("s3.secret-access-key", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        ("s3.key", "plain-s3-key"),
        ("adls.key", "plain-adls-key"),
        (
            "adls.connection-string",
            "AccountKey=c2VjcmV0a2V5;EndpointSuffix=core.windows.net",
        ),
        ("gcs.service.account.private.key", "BEGIN PRIVATE KEY"),
        ("token", "mytoken123"),
        ("password", "hunter2"),
        ("credential", "cred123"),
        ("gcs.credentials", "cred456"),
        ("passphrase", "ssh-passphrase"),
    ],
)
def test_repr_never_prints_values(key, value):
    # The repr lists property keys but never their values, so no secret can leak
    # regardless of the key name (a denylist would miss e.g. adls.connection-string).
    r = repr(FileIO.from_props({key: value, "s3.region": "us-east-1"}))
    assert value not in r
    assert key in r
    assert "s3.region" in r


def test_repr_lists_keys_not_values():
    r = repr(FileIO.from_props({"warehouse": "s3://bucket", "s3.region": "us-east-1"}))
    assert r == 'FileIO(keys=["s3.region", "warehouse"])'
    assert "s3://bucket" not in r


def test_repr_empty_props():
    assert repr(FileIO.from_props({})) == "FileIO()"


def test_new_file_handles_expose_location(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "data.bin")

    inp = fio.new_input(uri)
    out = fio.new_output(uri)

    assert isinstance(inp, InputFile)
    assert isinstance(out, OutputFile)
    assert inp.location() == uri
    assert out.location() == uri


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"x",
        b"\x00\x01\x02iceberg\xff",
        bytes(range(256)),
    ],
)
def test_write_read_exists_and_size_round_trip(tmp_path, payload):
    fio = local_fio()
    path = tmp_path / "round-trip.bin"
    uri = file_uri(path)

    assert fio.exists(uri) is False
    assert fio.new_input(uri).exists() is False

    fio.new_output(uri).write(payload)

    assert fio.exists(uri) is True
    assert fio.new_input(uri).exists() is True
    assert fio.new_input(uri).read() == payload
    assert fio.new_input(uri).size() == len(payload)
    assert path.read_bytes() == payload


def test_write_replaces_existing_file(tmp_path):
    path = tmp_path / "overwrite.txt"
    out = local_fio().new_output(file_uri(path))

    out.write(b"first")
    out.write(b"second")

    assert path.read_bytes() == b"second"


def test_delete_removes_file(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "delete-me.txt")

    fio.new_output(uri).write(b"bye")
    fio.delete(uri)

    assert fio.exists(uri) is False


def test_delete_missing_file_is_idempotent(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "already-gone.txt")

    fio.delete(uri)

    assert fio.exists(uri) is False


@pytest.mark.parametrize("scheme", ["ftp", "http", "weirdscheme"])
def test_unsupported_scheme_raises_not_implemented(scheme):
    with pytest.raises(NotImplementedError) as exc:
        local_fio().new_input(f"{scheme}://host/path")
    assert type(exc.value) is NotImplementedError


@pytest.mark.parametrize("bad_uri", ["not a uri at all", "://bad"])
def test_malformed_uri_raises_value_error(bad_uri):
    with pytest.raises(ValueError) as exc:
        local_fio().new_input(bad_uri)
    assert type(exc.value) is ValueError


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="relies on POSIX directory permission bits",
)
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses filesystem permission bits",
)
def test_permission_denied_raises_os_error(tmp_path):
    # Permission denied is also flattened to Unexpected -> exactly OSError.
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        with pytest.raises(OSError) as exc:
            local_fio().new_output(file_uri(locked / "x.txt")).write(b"hi")
        assert type(exc.value) is OSError
    finally:
        locked.chmod(0o755)


def test_handle_is_reusable_across_multiple_opens(tmp_path):
    fio = local_fio()

    for i in range(3):
        fio.new_output(file_uri(tmp_path / f"f{i}")).write(b"x")

    for i in range(3):
        assert fio.new_input(file_uri(tmp_path / f"f{i}")).read() == b"x"


def test_handle_supports_threaded_io(tmp_path):
    fio = local_fio()
    errors = []
    lock = threading.Lock()

    def worker(i):
        try:
            payload = bytes([i]) * 4096
            uri = file_uri(tmp_path / f"threaded-{i}.bin")
            fio.new_output(uri).write(payload)
            assert fio.new_input(uri).read() == payload
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


def test_file_handle_repr_names_type_and_location(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "repr.txt")
    input_repr = repr(fio.new_input(uri))
    output_repr = repr(fio.new_output(uri))

    assert "InputFile" in input_repr
    assert uri in input_repr
    assert "OutputFile" in output_repr
    assert uri in output_repr


# ---------------------------------------------------------------------------
# Exhaustive error-matrix / edge coverage (every works-vs-breaks path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["read", "size"])
def test_missing_input_is_plain_oserror_not_filenotfound(tmp_path, method):
    # opendal NotFound is flattened to Unexpected, so we must NOT get the
    # FileNotFoundError subclass (the "correct" POSIX errno mapping).
    inp = local_fio().new_input(file_uri(tmp_path / "missing.txt"))
    with pytest.raises(OSError) as exc:
        getattr(inp, method)()
    assert not isinstance(exc.value, FileNotFoundError)
    assert "NotFound" in str(exc.value)


@pytest.mark.parametrize("op", ["new_input", "new_output", "exists"])
@pytest.mark.parametrize("path", ["relative/path.txt", "/abs/no/scheme.bin", ""])
def test_scheme_less_path_raises_value_error(path, op):
    with pytest.raises(ValueError) as exc:
        getattr(local_fio(), op)(path)
    assert type(exc.value) is ValueError
    assert "relative URL without a base" in str(exc.value)


def test_unicode_path_round_trips(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "ünïcödé-文件-🧊.bin")
    fio.new_output(uri).write(b"snowman")
    assert fio.new_input(uri).read() == b"snowman"
    assert fio.new_input(uri).location() == uri


def test_large_payload_round_trips(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "big.bin")
    payload = bytes(range(256)) * 8192  # 2 MiB exercises the chunked write/read path
    fio.new_output(uri).write(payload)
    assert fio.new_input(uri).read() == payload
    assert fio.new_input(uri).size() == len(payload)


@pytest.mark.parametrize(
    "key", ["has space", "has,comma", 'has"quote', "tab\there", "k" * 4096]
)
def test_repr_quotes_awkward_keys_without_leaking_value(key):
    r = repr(FileIO.from_props({key: "secret-value"}))
    assert "secret-value" not in r
    assert r.startswith("FileIO(keys=[")


def test_directory_path_read_write_raise_os_error(tmp_path):
    # Reading or writing a directory path is an I/O error (OSError), not a panic or a
    # wrong exception type. Directory metadata/delete semantics are backend-specific and
    # intentionally not pinned here.
    fio = local_fio()
    d = tmp_path / "subdir"
    d.mkdir()
    duri = file_uri(d)
    with pytest.raises(OSError):
        fio.new_input(duri).read()
    with pytest.raises(OSError):
        fio.new_output(duri).write(b"x")


def test_from_props_rejects_non_string_values():
    with pytest.raises(TypeError):
        FileIO.from_props({"k": 123})


def test_from_props_rejects_non_dict():
    with pytest.raises(TypeError):
        FileIO.from_props(["a", "b"])


def test_shared_handle_concurrent_reads_same_file(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "concurrent.bin")
    payload = bytes(range(256)) * 100
    fio.new_output(uri).write(payload)

    errors = []
    lock = threading.Lock()

    def reader():
        try:
            assert fio.new_input(uri).read() == payload
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
