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

import threading

import pytest

from pyiceberg_core.file_io import FileIO, InputFile, OutputFile


def local_fio() -> FileIO:
    return FileIO.from_props({})


def file_uri(path) -> str:
    return f"file://{path}"


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
        ("gcs.service.account.private.key", "BEGIN PRIVATE KEY"),
        ("token", "mytoken123"),
        ("password", "hunter2"),
        ("credential", "cred123"),
        ("passphrase", "ssh-passphrase"),
    ],
)
def test_repr_redacts_sensitive_values(key, value):
    r = repr(FileIO.from_props({key: value, "s3.region": "us-east-1"}))
    assert value not in r
    assert "s3.region=us-east-1" in r
    assert f"{key}=<redacted>" in r


def test_repr_shows_plain_values():
    r = repr(FileIO.from_props({"warehouse": "s3://bucket", "s3.region": "us-east-1"}))
    assert "s3.region=us-east-1" in r
    assert "warehouse=s3://bucket" in r


def test_repr_does_not_redact_safe_key_terms():
    r = repr(
        FileIO.from_props(
            {
                "io.key-format": "plain",
                "table.key-delimiter": "|",
                "warehouse.key-prefix": "tables/",
            }
        )
    )
    assert "io.key-format=plain" in r
    assert "table.key-delimiter=|" in r
    assert "warehouse.key-prefix=tables/" in r


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


@pytest.mark.parametrize("method", ["read", "size"])
def test_missing_input_operations_raise_file_not_found(tmp_path, method):
    inp = local_fio().new_input(file_uri(tmp_path / "missing.txt"))

    with pytest.raises(FileNotFoundError):
        getattr(inp, method)()


def test_handle_is_reusable_across_many_opens(tmp_path):
    fio = local_fio()

    for i in range(50):
        fio.new_output(file_uri(tmp_path / f"f{i}")).write(b"x")

    for i in range(50):
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
