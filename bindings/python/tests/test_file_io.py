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

"""pytest tests for the PyFileIO / PyInputFile / PyOutputFile binding.

All tests use the local filesystem via ``file://`` URIs and ``tmp_path`` for
isolation; no network dependencies.
"""

import pytest

from pyiceberg_core.file_io import FileIO, InputFile, OutputFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def local_fio() -> FileIO:
    """FileIO handle backed by local filesystem (empty props, file:// URIs)."""
    return FileIO.from_props({})


def file_uri(path) -> str:
    """Convert a pathlib.Path to an absolute file:// URI."""
    return f"file://{path}"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_props_empty_dict():
    """Constructing with an empty dict must succeed without raising."""
    fio = FileIO.from_props({})
    assert isinstance(fio, FileIO)


def test_from_props_with_region_only():
    """Supplying only a region property must not raise at construction time.

    FileIO is lazily initialised; actual S3 credential errors only surface on
    first I/O — not at from_props().
    """
    fio = FileIO.from_props({"s3.region": "us-east-1"})
    assert isinstance(fio, FileIO)


def test_from_props_returns_independent_handles():
    """Two calls to from_props must return distinct objects."""
    a = FileIO.from_props({})
    b = FileIO.from_props({})
    assert a is not b


# ---------------------------------------------------------------------------
# __repr__ — credential redaction
# ---------------------------------------------------------------------------

CREDENTIAL_CASES = [
    ("s3.access-key-id", "AKIAIOSFODNN7EXAMPLE"),
    ("s3.secret-access-key", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    ("gcs.service.account.private.key", "BEGIN PRIVATE KEY"),
    ("token", "mytoken123"),
    ("password", "hunter2"),
    ("credential", "cred123"),
    ("passphrase", "ssh-passphrase"),
]


@pytest.mark.parametrize("key,value", CREDENTIAL_CASES)
def test_repr_redacts_sensitive_key(key, value):
    fio = FileIO.from_props({key: value})
    r = repr(fio)
    assert value not in r, f"repr leaked value for key {key!r}: {r!r}"
    assert "<redacted>" in r


def test_repr_shows_non_sensitive_values():
    fio = FileIO.from_props({"s3.region": "us-east-1", "warehouse": "s3://mybucket"})
    r = repr(fio)
    assert "us-east-1" in r
    assert "s3://mybucket" in r


def test_repr_empty_props():
    fio = FileIO.from_props({})
    assert repr(fio) == "FileIO()"


def test_repr_mixed_sensitive_and_plain():
    fio = FileIO.from_props(
        {
            "s3.region": "eu-west-1",
            "s3.access-key-id": "SECRET",
            "warehouse": "s3://bucket",
        }
    )
    r = repr(fio)
    assert "SECRET" not in r
    assert "eu-west-1" in r
    assert "s3://bucket" in r


# ---------------------------------------------------------------------------
# exists / delete via FileIO
# ---------------------------------------------------------------------------


def test_exists_returns_false_for_missing_file(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "nonexistent.txt")
    assert fio.exists(uri) is False


def test_exists_returns_true_after_write(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "hello.txt")
    fio.new_output(uri).write(b"data")
    assert fio.exists(uri) is True


def test_delete_removes_file(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "to_delete.txt")
    fio.new_output(uri).write(b"bye")
    assert fio.exists(uri) is True
    fio.delete(uri)
    assert fio.exists(uri) is False


def test_delete_nonexistent_file_is_noop(tmp_path):
    """Deleting a file that does not exist must not raise."""
    fio = local_fio()
    uri = file_uri(tmp_path / "ghost.txt")
    fio.delete(uri)  # should not raise


# ---------------------------------------------------------------------------
# new_output / OutputFile.write
# ---------------------------------------------------------------------------


def test_new_output_returns_output_file(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "out.txt")
    out = fio.new_output(uri)
    assert isinstance(out, OutputFile)


def test_output_file_location(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "loc.txt")
    out = fio.new_output(uri)
    assert out.location() == uri


def test_output_write_creates_file(tmp_path):
    fio = local_fio()
    path = tmp_path / "created.txt"
    fio.new_output(file_uri(path)).write(b"hello")
    assert path.exists()
    assert path.read_bytes() == b"hello"


def test_output_write_overwrites_existing(tmp_path):
    fio = local_fio()
    path = tmp_path / "overwrite.txt"
    uri = file_uri(path)
    fio.new_output(uri).write(b"first")
    fio.new_output(uri).write(b"second")
    assert path.read_bytes() == b"second"


def test_output_write_empty_bytes(tmp_path):
    fio = local_fio()
    path = tmp_path / "empty.txt"
    fio.new_output(file_uri(path)).write(b"")
    assert path.read_bytes() == b""


# ---------------------------------------------------------------------------
# new_input / InputFile.read / exists / metadata
# ---------------------------------------------------------------------------


def test_new_input_returns_input_file(tmp_path):
    fio = local_fio()
    path = tmp_path / "read.txt"
    path.write_bytes(b"content")
    inp = fio.new_input(file_uri(path))
    assert isinstance(inp, InputFile)


def test_input_file_location(tmp_path):
    fio = local_fio()
    path = tmp_path / "loc.txt"
    path.write_bytes(b"x")
    uri = file_uri(path)
    assert fio.new_input(uri).location() == uri


def test_input_exists_true(tmp_path):
    fio = local_fio()
    path = tmp_path / "present.txt"
    path.write_bytes(b"here")
    assert fio.new_input(file_uri(path)).exists() is True


def test_input_exists_false(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "absent.txt")
    assert fio.new_input(uri).exists() is False


def test_round_trip_write_then_read(tmp_path):
    """Write via OutputFile; read back via InputFile; bytes must match."""
    fio = local_fio()
    uri = file_uri(tmp_path / "round.bin")
    payload = b"\x00\x01\x02iceberg\xff"
    fio.new_output(uri).write(payload)
    assert fio.new_input(uri).read() == payload


def test_input_metadata_size(tmp_path):
    fio = local_fio()
    path = tmp_path / "meta.txt"
    content = b"size-check"
    path.write_bytes(content)
    meta = fio.new_input(file_uri(path)).metadata()
    assert isinstance(meta, dict)
    assert meta["size"] == len(content)


def test_input_repr(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "repr.txt")
    r = repr(fio.new_input(uri))
    assert "InputFile" in r
    assert "repr.txt" in r


def test_output_repr(tmp_path):
    fio = local_fio()
    uri = file_uri(tmp_path / "repr.txt")
    r = repr(fio.new_output(uri))
    assert "OutputFile" in r
    assert "repr.txt" in r
