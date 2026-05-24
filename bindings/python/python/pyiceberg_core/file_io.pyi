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

"""Type stubs for the iceberg-rust FileIO binding.

Primary constructor: ``FileIO.from_props(props)`` where ``props`` is a plain
``dict[str, str]`` of storage-backend configuration properties (the same keys
iceberg-rust's ``FileIOBuilder`` recognises, e.g. ``s3.region``,
``s3.access-key-id``).  For local-filesystem paths use ``file://…`` URIs with
an empty dict.

The ``FileIO`` instance is lazily initialised on first use and cached, so
constructing once and reusing across many file opens amortises the storage
setup cost.

Credential security: ``repr(FileIO(...))`` redacts any property key that
contains ``secret``, ``key``, ``token``, ``password``, ``credential``, or
``passphrase`` (case-insensitive).
"""

class FileIO:
    """Reusable handle to iceberg-rust's ``iceberg::io::FileIO``.

    Backed by ``OpenDalResolvingStorageFactory``, which auto-detects the
    storage scheme from the path (``file://``, ``s3://``, ``gs://``,
    ``abfss://``, …) and caches the per-scheme operator on first use.
    """

    @staticmethod
    def from_props(props: dict[str, str]) -> "FileIO":
        """Construct a ``FileIO`` handle from a dict of storage properties."""
        ...

    def exists(self, path: str) -> bool:
        """Return ``True`` if the file at ``path`` exists in the underlying storage."""
        ...

    def delete(self, path: str) -> None:
        """Delete the file at ``path``.

        Raises ``IOError`` on failure.  Deleting a non-existent file is a no-op.
        """
        ...

    def new_input(self, path: str) -> "InputFile":
        """Open ``path`` for reading and return an ``InputFile`` handle.

        Raises ``IOError`` if the path cannot be resolved.
        """
        ...

    def new_output(self, path: str) -> "OutputFile":
        """Open ``path`` for writing and return an ``OutputFile`` handle.

        Raises ``IOError`` if the path cannot be resolved.
        """
        ...

    def __repr__(self) -> str: ...


class InputFile:
    """Handle for reading a single file.  Obtained via ``FileIO.new_input(path)``."""

    def location(self) -> str:
        """The absolute path this input file was opened on."""
        ...

    def exists(self) -> bool:
        """Return ``True`` if the file exists in the underlying storage."""
        ...

    def read(self) -> bytes:
        """Read the entire file content and return it as ``bytes``.

        Raises ``IOError`` on read failure.
        """
        ...

    def metadata(self) -> dict[str, int]:
        """Return file metadata.  Currently exposes ``size`` (bytes as int)."""
        ...

    def __repr__(self) -> str: ...


class OutputFile:
    """Handle for writing a single file.  Obtained via ``FileIO.new_output(path)``."""

    def location(self) -> str:
        """The absolute path this output file was opened on."""
        ...

    def write(self, data: bytes) -> None:
        """Write ``data`` to the file, replacing any existing content.

        Raises ``IOError`` on write failure.
        """
        ...

    def __repr__(self) -> str: ...
