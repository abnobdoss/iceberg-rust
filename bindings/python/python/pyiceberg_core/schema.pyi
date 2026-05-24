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

"""Type stubs for the iceberg-rust Schema binding.

Design: opaque handle pattern (Design (d) from the schema-binding-design.md recon).
Parse once via Schema.from_json(); reuse the handle across all downstream building
blocks. Internally holds an Arc<Schema> — clone is a reference count bump.

The _capsule() method returns a PyCapsule named b"iceberg_core_schema\\0" for
direct Rust→Rust handoff to sibling bindings (future Predicate.bind_with_schema,
FileScanTask, ArrowReader). Do not expose the capsule to user code.

Field lookup methods return dicts with keys:
  id       (int)  — unique field ID within the table
  name     (str)  — field name
  type     (str)  — spec-JSON encoding of the type, e.g. '"int"' or '{"type":"list",...}'
  required (bool) — whether the field is required (non-nullable)
"""

from typing import Any, Optional


class Schema:
    """Opaque handle around an iceberg-rust spec::Schema.

    Constructed once from spec-JSON via Schema.from_json(); the parsed
    Arc<Schema> is reused for every subsequent call. No re-parsing occurs
    when the handle is passed to downstream bindings.
    """

    @staticmethod
    def from_json(s: str) -> "Schema":
        """Parse an Iceberg spec-JSON string (V1 or V2) into a Schema handle.

        Raises ValueError if the JSON is malformed or violates spec constraints
        (duplicate field IDs, float/double identifier fields, etc.).
        """
        ...

    def schema_id(self) -> int:
        """The schema-id. Defaults to 0 for V1 schemas that omit it."""
        ...

    def highest_field_id(self) -> int:
        """Highest field ID across all fields at every nesting level."""
        ...

    def column_names(self) -> list[str]:
        """Names of top-level fields only (not dotted-path nested names)."""
        ...

    def identifier_field_ids(self) -> list[int]:
        """Identifier field IDs (primary-key / equality-delete keys), sorted ascending."""
        ...

    def find_field_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """Look up a field by its full dotted-path name (case-sensitive).

        Returns a dict with keys ``id``, ``name``, ``type``, ``required``,
        or None if not found. The ``type`` value is the spec-JSON encoding
        of the field type; use ``json.loads(field['type'])`` to parse it.
        """
        ...

    def field_by_id(self, field_id: int) -> dict[str, Any]:
        """Look up a field by its field ID.

        Returns a dict with keys ``id``, ``name``, ``type``, ``required``.
        Raises KeyError if the field ID is not present in this schema.
        """
        ...

    def to_json(self) -> str:
        """Serialize this schema back to Iceberg spec-JSON (V2 format).

        The output is parseable by Schema.from_json() and round-trips
        losslessly for all Iceberg v2 types.
        """
        ...

    def to_arrow_schema(self) -> Any:
        """Export as a PyArrow Schema via the Arrow C Data Interface.

        Field IDs are preserved at all nesting levels via PARQUET:field_id
        metadata. Lossless for all Iceberg v2 types. Costs ~94 µs for a
        30-field schema; prefer _capsule() for internal Rust→Rust handoff.

        Returns a pyarrow.Schema object (typed as Any to avoid a hard
        dependency on pyarrow in this stub).
        """
        ...

    def __arrow_c_schema__(self) -> Any:
        """Arrow PyCapsule Interface — zero-copy schema export.

        Returns a PyCapsule named "arrow_schema" wrapping an FFI_ArrowSchema.
        Any library that recognises the Arrow PyCapsule Interface (PyArrow >= 14,
        Polars, narwhals, etc.) can import this schema without going through
        Python objects.

        Field IDs are preserved via PARQUET:field_id metadata at all nesting levels.
        """
        ...

    def _capsule(self) -> Any:
        """Return a PyCapsule wrapping Arc<Schema> for Rust→Rust handoff.

        The capsule name is b"iceberg_core_schema\\0". Sibling Rust bindings
        must validate this name before dereferencing. Internal API — not
        intended for direct use from Python application code.
        """
        ...

    def __repr__(self) -> str: ...
