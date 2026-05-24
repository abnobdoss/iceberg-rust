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

"""Tests for the pyiceberg_core.schema.Schema binding.

Fixtures are adapted from crates/iceberg/src/spec/schema/mod.rs test helpers
(table_schema_simple, table_schema_nested) to stay self-contained and
consistent with what iceberg-rust itself guarantees.
"""

import json

import pytest

from pyiceberg_core.schema import Schema

# ---------------------------------------------------------------------------
# Fixture JSON strings (spec-compliant V2 format)
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA_JSON = json.dumps(
    {
        "type": "struct",
        "schema-id": 1,
        "fields": [
            {"id": 1, "name": "foo", "required": False, "type": "string"},
            {"id": 2, "name": "bar", "required": True, "type": "int"},
            {"id": 3, "name": "baz", "required": False, "type": "boolean"},
        ],
        "identifier-field-ids": [2],
    }
)

NESTED_SCHEMA_JSON = json.dumps(
    {
        "type": "struct",
        "schema-id": 1,
        "fields": [
            {"id": 1, "name": "foo", "required": False, "type": "string"},
            {"id": 2, "name": "bar", "required": True, "type": "int"},
            {"id": 3, "name": "baz", "required": False, "type": "boolean"},
            {
                "id": 4,
                "name": "qux",
                "required": True,
                "type": {
                    "type": "list",
                    "element-id": 5,
                    "element": "string",
                    "element-required": True,
                },
            },
            {
                "id": 6,
                "name": "quux",
                "required": True,
                "type": {
                    "type": "map",
                    "key-id": 7,
                    "key": "string",
                    "value-id": 8,
                    "value": {
                        "type": "map",
                        "key-id": 9,
                        "key": "string",
                        "value-id": 10,
                        "value": "int",
                        "value-required": True,
                    },
                    "value-required": True,
                },
            },
            {
                "id": 11,
                "name": "location",
                "required": True,
                "type": {
                    "type": "list",
                    "element-id": 12,
                    "element": {
                        "type": "struct",
                        "fields": [
                            {
                                "id": 13,
                                "name": "latitude",
                                "required": False,
                                "type": "float",
                            },
                            {
                                "id": 14,
                                "name": "longitude",
                                "required": False,
                                "type": "float",
                            },
                        ],
                    },
                    "element-required": True,
                },
            },
            {
                "id": 15,
                "name": "person",
                "required": False,
                "type": {
                    "type": "struct",
                    "fields": [
                        {
                            "id": 16,
                            "name": "name",
                            "required": False,
                            "type": "string",
                        },
                        {
                            "id": 17,
                            "name": "age",
                            "required": True,
                            "type": "int",
                        },
                    ],
                },
            },
        ],
        "identifier-field-ids": [2],
    }
)

# V1 schema omits schema-id and identifier-field-ids
V1_SCHEMA_JSON = json.dumps(
    {
        "type": "struct",
        "fields": [
            {"id": 1, "name": "ts", "required": True, "type": "timestamp"},
            {"id": 2, "name": "msg", "required": False, "type": "string"},
        ],
    }
)

# Realistic 15-field schema with various primitive types
REALISTIC_SCHEMA_JSON = json.dumps(
    {
        "type": "struct",
        "schema-id": 42,
        "fields": [
            {"id": 1, "name": "id", "required": True, "type": "long"},
            {"id": 2, "name": "event_time", "required": True, "type": "timestamptz"},
            {"id": 3, "name": "event_date", "required": False, "type": "date"},
            {"id": 4, "name": "user_id", "required": True, "type": "int"},
            {"id": 5, "name": "session_id", "required": False, "type": "string"},
            {"id": 6, "name": "amount", "required": False, "type": "double"},
            {"id": 7, "name": "is_active", "required": False, "type": "boolean"},
            {"id": 8, "name": "payload", "required": False, "type": "binary"},
            {
                "id": 9,
                "name": "tags",
                "required": False,
                "type": {
                    "type": "list",
                    "element-id": 10,
                    "element": "string",
                    "element-required": False,
                },
            },
            {
                "id": 11,
                "name": "metadata",
                "required": False,
                "type": {
                    "type": "map",
                    "key-id": 12,
                    "key": "string",
                    "value-id": 13,
                    "value": "string",
                    "value-required": False,
                },
            },
            {
                "id": 14,
                "name": "address",
                "required": False,
                "type": {
                    "type": "struct",
                    "fields": [
                        {"id": 15, "name": "street", "required": False, "type": "string"},
                        {"id": 16, "name": "city", "required": False, "type": "string"},
                        {"id": 17, "name": "zip", "required": False, "type": "string"},
                    ],
                },
            },
        ],
        "identifier-field-ids": [1, 4],
    }
)


# ---------------------------------------------------------------------------
# Construction and basic attributes
# ---------------------------------------------------------------------------


def test_from_json_simple():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    assert h.schema_id() == 1
    assert h.highest_field_id() == 3


def test_from_json_nested():
    h = Schema.from_json(NESTED_SCHEMA_JSON)
    assert h.schema_id() == 1
    assert h.highest_field_id() == 17


def test_from_json_v1_schema():
    """V1 schemas omit schema-id; iceberg-rust defaults to 0."""
    h = Schema.from_json(V1_SCHEMA_JSON)
    assert h.schema_id() == 0
    assert h.highest_field_id() == 2


def test_from_json_bad_json_raises_value_error():
    with pytest.raises(ValueError, match="Failed to parse schema JSON"):
        Schema.from_json("{not valid json}")


def test_from_json_missing_fields_raises_value_error():
    """A struct JSON with no 'fields' key is rejected by iceberg-rust's serde."""
    bad = json.dumps({"type": "struct"})  # missing "fields"
    with pytest.raises(ValueError, match="Failed to parse schema JSON"):
        Schema.from_json(bad)


def test_from_json_duplicate_field_ids_raises():
    bad = json.dumps(
        {
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {"id": 1, "name": "a", "required": True, "type": "int"},
                {"id": 1, "name": "b", "required": True, "type": "string"},  # dup id
            ],
        }
    )
    with pytest.raises(ValueError):
        Schema.from_json(bad)


# ---------------------------------------------------------------------------
# column_names — top-level only
# ---------------------------------------------------------------------------


def test_column_names_simple():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    assert h.column_names() == ["foo", "bar", "baz"]


def test_column_names_nested_top_level_only():
    """column_names returns top-level field names only, not dotted paths."""
    h = Schema.from_json(NESTED_SCHEMA_JSON)
    names = h.column_names()
    assert names == ["foo", "bar", "baz", "qux", "quux", "location", "person"]
    # nested names must NOT appear here
    assert "location.element.latitude" not in names
    assert "person.name" not in names


def test_column_names_empty_schema():
    empty = json.dumps({"type": "struct", "schema-id": 0, "fields": []})
    h = Schema.from_json(empty)
    assert h.column_names() == []


# ---------------------------------------------------------------------------
# identifier_field_ids
# ---------------------------------------------------------------------------


def test_identifier_field_ids_present():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    assert h.identifier_field_ids() == [2]


def test_identifier_field_ids_multiple():
    h = Schema.from_json(REALISTIC_SCHEMA_JSON)
    assert h.identifier_field_ids() == [1, 4]  # sorted ascending


def test_identifier_field_ids_empty():
    no_ids = json.dumps(
        {
            "type": "struct",
            "schema-id": 1,
            "fields": [{"id": 1, "name": "x", "required": True, "type": "long"}],
            "identifier-field-ids": [],
        }
    )
    h = Schema.from_json(no_ids)
    assert h.identifier_field_ids() == []


# ---------------------------------------------------------------------------
# find_field_by_name
# ---------------------------------------------------------------------------


def test_find_field_by_name_exists():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    f = h.find_field_by_name("bar")
    assert f is not None
    assert f["id"] == 2
    assert f["name"] == "bar"
    assert f["required"] is True
    # type is the spec-JSON encoding; json.loads unwraps it
    assert json.loads(f["type"]) == "int"


def test_find_field_by_name_optional_field():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    f = h.find_field_by_name("foo")
    assert f is not None
    assert f["required"] is False


def test_find_field_by_name_missing_returns_none():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    assert h.find_field_by_name("nonexistent") is None


def test_find_field_by_name_case_sensitive_miss():
    """field_by_name is case-sensitive; 'Bar' != 'bar'."""
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    assert h.find_field_by_name("Bar") is None
    assert h.find_field_by_name("BAR") is None


def test_find_field_by_name_nested_dotted():
    """Dotted-path names resolve nested fields."""
    h = Schema.from_json(NESTED_SCHEMA_JSON)
    f = h.find_field_by_name("person.name")
    assert f is not None
    assert f["id"] == 16
    assert f["name"] == "name"


def test_find_field_by_name_list_type():
    h = Schema.from_json(NESTED_SCHEMA_JSON)
    f = h.find_field_by_name("qux")
    assert f is not None
    type_obj = json.loads(f["type"])
    assert isinstance(type_obj, dict)
    assert type_obj["type"] == "list"


def test_find_field_by_name_map_type():
    h = Schema.from_json(NESTED_SCHEMA_JSON)
    f = h.find_field_by_name("quux")
    assert f is not None
    type_obj = json.loads(f["type"])
    assert type_obj["type"] == "map"


# ---------------------------------------------------------------------------
# field_by_id
# ---------------------------------------------------------------------------


def test_field_by_id_exists():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    f = h.field_by_id(1)
    assert f["id"] == 1
    assert f["name"] == "foo"
    assert f["required"] is False


def test_field_by_id_missing_raises_key_error():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    with pytest.raises(KeyError):
        h.field_by_id(99)


def test_field_by_id_nested_field():
    h = Schema.from_json(NESTED_SCHEMA_JSON)
    f = h.field_by_id(13)  # location.element.latitude
    assert f["name"] == "latitude"


# ---------------------------------------------------------------------------
# to_json round-trip
# ---------------------------------------------------------------------------


def test_to_json_round_trip_simple():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    j = h.to_json()
    parsed = json.loads(j)
    assert parsed["schema-id"] == 1
    assert len(parsed["fields"]) == 3
    names = [f["name"] for f in parsed["fields"]]
    assert names == ["foo", "bar", "baz"]
    assert parsed["identifier-field-ids"] == [2]


def test_to_json_round_trip_nested():
    h = Schema.from_json(NESTED_SCHEMA_JSON)
    j = h.to_json()
    h2 = Schema.from_json(j)
    assert h2.schema_id() == h.schema_id()
    assert h2.column_names() == h.column_names()
    assert h2.highest_field_id() == h.highest_field_id()
    assert h2.identifier_field_ids() == h.identifier_field_ids()


def test_to_json_round_trip_v1_schema():
    """V1 schema round-trips as V2 (schema-id defaults to 0)."""
    h = Schema.from_json(V1_SCHEMA_JSON)
    j = h.to_json()
    parsed = json.loads(j)
    # V2 format always emits schema-id
    assert "schema-id" in parsed
    assert parsed["schema-id"] == 0


def test_to_json_identifier_field_ids_preserved():
    h = Schema.from_json(REALISTIC_SCHEMA_JSON)
    j = h.to_json()
    parsed = json.loads(j)
    assert sorted(parsed.get("identifier-field-ids", [])) == [1, 4]


# ---------------------------------------------------------------------------
# _capsule (internal Rust→Rust handoff)
# ---------------------------------------------------------------------------


def test_capsule_returns_pycapsule():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    cap = h._capsule()
    assert type(cap).__name__ == "PyCapsule"


def test_capsule_can_be_called_repeatedly():
    """Each call returns a fresh PyCapsule wrapping the same Arc<Schema>."""
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    cap1 = h._capsule()
    cap2 = h._capsule()
    assert cap1 is not cap2  # distinct objects
    assert type(cap1).__name__ == type(cap2).__name__ == "PyCapsule"


def test_capsule_handle_survives_original_del():
    """PyCapsule holds a clone of Arc<Schema>; dropping the handle is safe."""
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    cap = h._capsule()
    del h  # original handle gone; Arc refcount drops to 1 (capsule holds it)
    # capsule still valid — no crash
    assert type(cap).__name__ == "PyCapsule"


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_simple():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    r = repr(h)
    assert "Schema(" in r
    assert "schema_id=1" in r
    assert "fields=3" in r


def test_repr_long_schema_truncates():
    h = Schema.from_json(NESTED_SCHEMA_JSON)
    r = repr(h)
    assert "..." in r  # 7 fields > 4, so ellipsis appears


def test_repr_does_not_leak_arc_address():
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    r = repr(h)
    # Should not contain a raw pointer or memory address
    assert "0x" not in r


# ---------------------------------------------------------------------------
# to_arrow_schema (Arrow C Data Interface export)
# ---------------------------------------------------------------------------


def test_to_arrow_schema_returns_pyarrow_schema():
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    arrow_schema = h.to_arrow_schema()
    assert isinstance(arrow_schema, pa.Schema)


def test_to_arrow_schema_field_count():
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    arrow_schema = h.to_arrow_schema()
    assert len(arrow_schema) == 3


def test_to_arrow_schema_field_ids_preserved():
    """PARQUET:field_id metadata must survive the Iceberg→Arrow conversion."""
    pytest.importorskip("pyarrow")

    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    arrow_schema = h.to_arrow_schema()
    for i, arrow_field in enumerate(arrow_schema, start=1):
        meta = arrow_field.metadata or {}
        field_id = meta.get(b"PARQUET:field_id")
        assert field_id is not None, f"Field {arrow_field.name} missing PARQUET:field_id"
        assert int(field_id) == i


# ---------------------------------------------------------------------------
# __arrow_c_schema__ (Arrow PyCapsule Interface)
# ---------------------------------------------------------------------------


def test_arrow_c_schema_returns_capsule():
    """__arrow_c_schema__ must return a PyCapsule named 'arrow_schema'."""
    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    cap = h.__arrow_c_schema__()
    assert type(cap).__name__ == "PyCapsule"


def test_arrow_c_schema_importable_by_pyarrow():
    """PyArrow must be able to import the schema via the PyCapsule interface."""
    pa = pytest.importorskip("pyarrow")

    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    arrow_schema = pa.schema(h)  # pyarrow calls __arrow_c_schema__() under the hood
    assert isinstance(arrow_schema, pa.Schema)
    assert len(arrow_schema) == 3


def test_arrow_c_schema_field_ids_preserved():
    """Field IDs via __arrow_c_schema__ must match those from to_arrow_schema()."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    h = Schema.from_json(SIMPLE_SCHEMA_JSON)
    via_capsule = pa.schema(h)
    via_method = h.to_arrow_schema()
    assert via_capsule.equals(via_method)
