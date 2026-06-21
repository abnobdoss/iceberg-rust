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

import ctypes
import json

import pytest

from pyiceberg_core.schema import Schema


SIMPLE_SCHEMA = {
    "type": "struct",
    "schema-id": 1,
    "fields": [
        {"id": 1, "name": "foo", "required": False, "type": "string"},
        {"id": 2, "name": "bar", "required": True, "type": "int"},
        {"id": 3, "name": "baz", "required": False, "type": "boolean"},
    ],
    "identifier-field-ids": [2],
}

NESTED_SCHEMA = {
    "type": "struct",
    "schema-id": 7,
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
                    {"id": 16, "name": "name", "required": False, "type": "string"},
                    {"id": 17, "name": "age", "required": True, "type": "int"},
                ],
            },
        },
    ],
    "identifier-field-ids": [2],
}

IDENTIFIER_SCHEMA = {
    "type": "struct",
    "schema-id": 42,
    "fields": [
        {"id": 1, "name": "id", "required": True, "type": "long"},
        {"id": 2, "name": "event_time", "required": True, "type": "timestamptz"},
        {"id": 3, "name": "event_date", "required": False, "type": "date"},
        {"id": 4, "name": "user_id", "required": True, "type": "int"},
        {"id": 5, "name": "payload", "required": False, "type": "binary"},
    ],
    "identifier-field-ids": [4, 1],
}

V1_SCHEMA = {
    "type": "struct",
    "fields": [
        {"id": 1, "name": "ts", "required": True, "type": "timestamp"},
        {"id": 2, "name": "msg", "required": False, "type": "string"},
    ],
}

EMPTY_SCHEMA = {
    "type": "struct",
    "schema-id": 0,
    "fields": [],
}


def schema_json(schema: dict) -> str:
    return json.dumps(schema)


def capsule_name(capsule) -> str:
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.argtypes = [ctypes.py_object]
    get_name.restype = ctypes.c_char_p
    return get_name(capsule).decode()


def field_id(arrow_field) -> int:
    return int((arrow_field.metadata or {})[b"PARQUET:field_id"])


@pytest.mark.parametrize(
    "schema,expected_id,highest_id",
    [
        (SIMPLE_SCHEMA, 1, 3),
        (NESTED_SCHEMA, 7, 17),
        (V1_SCHEMA, 0, 2),
    ],
    ids=["simple-v2", "nested-v2", "v1"],
)
def test_from_json_schema_ids(schema, expected_id, highest_id):
    handle = Schema.from_json(schema_json(schema))
    assert handle.schema_id == expected_id
    assert handle.highest_field_id == highest_id


@pytest.mark.parametrize(
    "schema,expected_match",
    [
        ("{not valid json}", "Invalid JSON"),
        ({"type": "struct"}, "'fields' must be a list"),
        (
            {"type": "struct", "schema-id": "bad", "fields": []},
            "'schema-id' must be an integer",
        ),
        (
            {
                "type": "struct",
                "fields": [{"id": 1, "name": "x", "required": True, "type": "bogus"}],
            },
            "Invalid Iceberg schema",
        ),
        (
            {
                "type": "struct",
                "fields": [
                    {"id": 1, "name": "a", "required": True, "type": "int"},
                    {"id": 1, "name": "b", "required": True, "type": "string"},
                ],
            },
            "Invalid Iceberg schema",
        ),
        (
            {
                "type": "struct",
                "fields": [{"id": 1, "name": "a", "required": True, "type": "float"}],
                "identifier-field-ids": [1],
            },
            "Invalid Iceberg schema",
        ),
    ],
    ids=[
        "bad-json",
        "missing-fields",
        "bad-schema-id",
        "bad-type",
        "duplicate-field-id",
        "invalid-identifier-field",
    ],
)
def test_from_json_rejects_malformed_schema_families(schema, expected_match):
    # Malformed JSON and schema-rule violations must be distinguishable by message.
    with pytest.raises(ValueError, match=expected_match):
        Schema.from_json(schema if isinstance(schema, str) else schema_json(schema))


def test_from_json_error_message_is_actionable_for_rule_violations():
    # A schema-rule violation forwards core's actionable message (here, the duplicate id),
    # not just a generic "invalid".
    duplicate = {
        "type": "struct",
        "fields": [
            {"id": 1, "name": "a", "required": True, "type": "int"},
            {"id": 1, "name": "b", "required": True, "type": "string"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate"):
        Schema.from_json(schema_json(duplicate))


def test_column_names_are_top_level_only():
    assert Schema.from_json(schema_json(NESTED_SCHEMA)).column_names == [
        "foo",
        "bar",
        "baz",
        "qux",
        "quux",
        "location",
        "person",
    ]


def test_empty_schema_metadata_and_repr():
    handle = Schema.from_json(schema_json(EMPTY_SCHEMA))

    assert handle.schema_id == 0
    assert handle.highest_field_id == 0
    assert handle.column_names == []
    assert handle.identifier_field_ids == []
    assert "fields=0" in repr(handle)
    assert "columns=[]" in repr(handle)


@pytest.mark.parametrize(
    "schema,expected",
    [
        (SIMPLE_SCHEMA, [2]),
        (IDENTIFIER_SCHEMA, [1, 4]),
        ({**SIMPLE_SCHEMA, "identifier-field-ids": []}, []),
    ],
)
def test_identifier_field_ids(schema, expected):
    assert Schema.from_json(schema_json(schema)).identifier_field_ids == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("bar", {"id": 2, "name": "bar", "type": "int", "required": True}),
        (
            "person.name",
            {"id": 16, "name": "name", "type": "string", "required": False},
        ),
        ("qux", {"id": 4, "name": "qux", "type": {"type": "list"}, "required": True}),
        ("quux", {"id": 6, "name": "quux", "type": {"type": "map"}, "required": True}),
    ],
)
def test_find_field_by_name(name, expected):
    field = Schema.from_json(schema_json(NESTED_SCHEMA)).find_field_by_name(name)
    assert field is not None
    assert {k: field[k] for k in ("id", "name", "required")} == {
        k: expected[k] for k in ("id", "name", "required")
    }
    # "type" is a native Python value: a str for primitives, a dict for complex.
    field_type = field["type"]
    if isinstance(expected["type"], dict):
        assert isinstance(field_type, dict)
        assert field_type["type"] == expected["type"]["type"]
    else:
        assert field_type == expected["type"]


def test_find_field_by_name_is_case_sensitive_and_returns_none_for_missing():
    handle = Schema.from_json(schema_json(SIMPLE_SCHEMA))
    assert handle.find_field_by_name("bar") is not None
    assert handle.find_field_by_name("Bar") is None
    assert handle.find_field_by_name("missing") is None


@pytest.mark.parametrize("field_id_value,name", [(1, "foo"), (13, "latitude")])
def test_field_by_id(field_id_value, name):
    assert (
        Schema.from_json(schema_json(NESTED_SCHEMA)).field_by_id(field_id_value)["name"]
        == name
    )


def test_field_by_id_missing_raises_key_error():
    with pytest.raises(KeyError, match="99"):
        Schema.from_json(schema_json(SIMPLE_SCHEMA)).field_by_id(99)


def test_field_by_id_handles_sparse_ids():
    # Iceberg field IDs need not be contiguous; a gap must still raise KeyError.
    sparse = {
        "type": "struct",
        "schema-id": 1,
        "fields": [
            {"id": 1, "name": "a", "required": True, "type": "int"},
            {"id": 50, "name": "b", "required": True, "type": "int"},
            {"id": 999, "name": "c", "required": True, "type": "int"},
        ],
    }
    handle = Schema.from_json(schema_json(sparse))
    assert handle.field_by_id(50)["name"] == "b"
    assert handle.highest_field_id == 999
    with pytest.raises(KeyError):
        handle.field_by_id(2)


def test_field_metadata_includes_doc_and_defaults():
    # The whole field is exposed, not a hand-picked subset: doc and write-default
    # must survive the round trip.
    schema = {
        "type": "struct",
        "schema-id": 1,
        "fields": [
            {
                "id": 1,
                "name": "x",
                "required": True,
                "type": "int",
                "doc": "the x column",
                "write-default": 7,
            }
        ],
    }
    field = Schema.from_json(schema_json(schema)).field_by_id(1)
    assert field["doc"] == "the x column"
    assert field["write-default"] == 7


@pytest.mark.parametrize(
    "schema", [SIMPLE_SCHEMA, NESTED_SCHEMA, IDENTIFIER_SCHEMA, V1_SCHEMA]
)
def test_to_json_round_trips_semantically(schema):
    handle = Schema.from_json(schema_json(schema))
    reparsed = Schema.from_json(handle.to_json())
    assert reparsed.schema_id == handle.schema_id
    assert reparsed.column_names == handle.column_names
    assert reparsed.highest_field_id == handle.highest_field_id
    assert reparsed.identifier_field_ids == handle.identifier_field_ids
    # Iterate the actual top-level field IDs (Iceberg IDs need not be contiguous);
    # comparing the native "type" value also covers nested structure.
    for field in schema["fields"]:
        fid = field["id"]
        assert reparsed.field_by_id(fid)["type"] == handle.field_by_id(fid)["type"]


def test_arrow_c_schema_capsule_name():
    assert (
        capsule_name(Schema.from_json(schema_json(SIMPLE_SCHEMA)).__arrow_c_schema__())
        == "arrow_schema"
    )


def test_repr_truncates_column_preview_beyond_four_fields():
    # 7 top-level fields -> the first 4 quoted names, then an ellipsis marker.
    r = repr(Schema.from_json(schema_json(NESTED_SCHEMA)))
    assert "schema_id=7" in r
    assert "fields=7" in r
    assert "'foo'" in r and "'qux'" in r  # quoted preview names
    assert "..." in r  # truncated beyond the preview limit
    assert "'person'" not in r  # the 7th field is not shown


def test_repr_preview_boundary_at_four_columns():
    def schema_with(names):
        return schema_json(
            {
                "type": "struct",
                "schema-id": 1,
                "fields": [
                    {"id": i, "name": n, "required": True, "type": "int"}
                    for i, n in enumerate(names, start=1)
                ],
            }
        )

    # Exactly four columns -> no ellipsis; a fifth column -> ellipsis.
    assert "..." not in repr(Schema.from_json(schema_with(["a", "b", "c", "d"])))
    assert "..." in repr(Schema.from_json(schema_with(["a", "b", "c", "d", "e"])))


@pytest.mark.parametrize("schema", [SIMPLE_SCHEMA, V1_SCHEMA], ids=["v2", "v1"])
def test_to_arrow_schema_returns_pyarrow_schema(schema):
    pa = pytest.importorskip("pyarrow")
    arrow_schema = Schema.from_json(schema_json(schema)).to_arrow_schema()
    assert isinstance(arrow_schema, pa.Schema)
    assert [field_id(field) for field in arrow_schema] == [
        field["id"] for field in schema["fields"]
    ]


def test_to_arrow_schema_preserves_nested_field_ids():
    pytest.importorskip("pyarrow")
    arrow_schema = Schema.from_json(schema_json(NESTED_SCHEMA)).to_arrow_schema()

    assert field_id(arrow_schema.field("qux")) == 4
    assert field_id(arrow_schema.field("qux").type.value_field) == 5

    quux_type = arrow_schema.field("quux").type
    assert field_id(arrow_schema.field("quux")) == 6
    assert field_id(quux_type.key_field) == 7
    assert field_id(quux_type.item_field) == 8
    assert field_id(quux_type.item_field.type.key_field) == 9
    assert field_id(quux_type.item_field.type.item_field) == 10

    location_element = arrow_schema.field("location").type.value_field
    assert field_id(arrow_schema.field("location")) == 11
    assert field_id(location_element) == 12
    assert field_id(location_element.type.field("latitude")) == 13
    assert field_id(location_element.type.field("longitude")) == 14


def test_arrow_c_schema_imports_through_pyarrow():
    pa = pytest.importorskip("pyarrow")
    handle = Schema.from_json(schema_json(NESTED_SCHEMA))
    assert pa.schema(handle).equals(handle.to_arrow_schema())


# ---------------------------------------------------------------------------
# Exhaustive primitive / break-mode coverage (every works-vs-breaks path)
# ---------------------------------------------------------------------------

PRIMITIVE_TYPES = [
    "boolean",
    "int",
    "long",
    "float",
    "double",
    "date",
    "time",
    "timestamp",
    "timestamptz",
    "timestamp_ns",
    "timestamptz_ns",
    "string",
    "uuid",
    "binary",
    "decimal(9, 2)",
    "fixed[16]",
]


def _single_field_schema(type_str):
    return schema_json(
        {
            "type": "struct",
            "schema-id": 1,
            "fields": [{"id": 1, "name": "c", "required": False, "type": type_str}],
        }
    )


@pytest.mark.parametrize("type_str", PRIMITIVE_TYPES)
def test_primitive_type_parses_and_round_trips(type_str):
    # Each primitive parses, is exposed as its preserved type string, and survives a
    # JSON round trip.
    handle = Schema.from_json(_single_field_schema(type_str))
    field = handle.field_by_id(1)
    assert field["type"] == type_str
    assert isinstance(field["type"], str)
    assert Schema.from_json(handle.to_json()).field_by_id(1)["type"] == type_str


@pytest.mark.parametrize(
    "raw",
    ["[]", "[1, 2, 3]", "123", '"hello"', "true", "null"],
    ids=["empty-array", "array", "number", "string", "bool", "null"],
)
def test_from_json_non_object_top_level_is_iceberg_schema_error(raw):
    # Non-object JSON is well-formed but not a schema; it must not be mis-bucketed as
    # the 'fields' structural error (which only fires for objects).
    with pytest.raises(ValueError, match="Invalid Iceberg schema") as exc:
        Schema.from_json(raw)
    assert "'fields' must be a list" not in str(exc.value)


@pytest.mark.parametrize("raw", ["", "   ", "\n\t"], ids=["empty", "spaces", "ws"])
def test_from_json_empty_or_whitespace_is_invalid_json(raw):
    with pytest.raises(ValueError, match="Invalid JSON"):
        Schema.from_json(raw)


@pytest.mark.parametrize(
    "fields_value",
    [{}, {"a": 1}, 5, None, "x", True],
    ids=["empty-obj", "obj", "int", "null", "string", "bool"],
)
def test_from_json_fields_must_be_a_list_for_all_non_list_kinds(fields_value):
    with pytest.raises(ValueError, match=r"'fields' must be a list"):
        Schema.from_json(schema_json({"type": "struct", "fields": fields_value}))


@pytest.mark.parametrize(
    "field,ident_ids,expected",
    [
        (
            {"id": 1, "name": "a", "required": False, "type": "int"},
            [1],
            "is an optional field",
        ),
        (
            {"id": 1, "name": "a", "required": True, "type": "float"},
            [1],
            "cannot be a float or double type",
        ),
        (
            {"id": 1, "name": "a", "required": True, "type": "int"},
            [99],
            "field does not exist",
        ),
    ],
    ids=["optional", "float", "missing-id"],
)
def test_identifier_violations_forward_actionable_detail(field, ident_ids, expected):
    schema = {"type": "struct", "fields": [field], "identifier-field-ids": ident_ids}
    with pytest.raises(ValueError, match="Invalid Iceberg schema") as exc:
        Schema.from_json(schema_json(schema))
    assert expected in str(exc.value)


@pytest.mark.parametrize(
    "name,expected_id",
    [
        ("quux", 6),
        ("quux.key", 7),
        ("quux.value", 8),
        ("quux.value.key", 9),
        ("quux.value.value", 10),
        ("location", 11),
        ("location.element.latitude", 13),
        ("person.name", 16),
    ],
)
def test_find_field_by_name_reaches_map_value_and_list_element(name, expected_id):
    field = Schema.from_json(schema_json(NESTED_SCHEMA)).find_field_by_name(name)
    assert field["id"] == expected_id


@pytest.mark.parametrize("bare", ["name", "age", "latitude", "key", "value", "element"])
def test_find_field_by_name_does_not_register_bare_nested_child_names(bare):
    assert Schema.from_json(schema_json(NESTED_SCHEMA)).find_field_by_name(bare) is None


def test_find_field_by_name_returns_full_field_dict_with_defaults():
    schema = {
        "type": "struct",
        "schema-id": 1,
        "fields": [
            {
                "id": 1,
                "name": "x",
                "required": True,
                "type": "int",
                "doc": "the x",
                "write-default": 7,
                "initial-default": 3,
            }
        ],
    }
    handle = Schema.from_json(schema_json(schema))
    by_name = handle.find_field_by_name("x")
    assert by_name == handle.field_by_id(1)
    assert by_name["doc"] == "the x"
    assert by_name["write-default"] == 7
    assert by_name["initial-default"] == 3


def test_field_by_id_missing_key_error_message_is_stable():
    with pytest.raises(KeyError, match="No field with id 99 in schema"):
        Schema.from_json(schema_json(SIMPLE_SCHEMA)).field_by_id(99)


def test_nested_field_type_dict_carries_inner_ids():
    h = Schema.from_json(schema_json(NESTED_SCHEMA))
    list_type = h.field_by_id(4)["type"]
    assert list_type == {
        "type": "list",
        "element-id": 5,
        "element": "string",
        "element-required": True,
    }
    map_type = h.field_by_id(6)["type"]
    assert map_type["type"] == "map"
    assert map_type["key-id"] == 7 and map_type["value-id"] == 8
    struct_type = h.field_by_id(15)["type"]
    assert struct_type["type"] == "struct"
    assert [f["id"] for f in struct_type["fields"]] == [16, 17]


def test_repr_quotes_column_names_python_style():
    schema = {
        "type": "struct",
        "schema-id": 1,
        "fields": [
            {"id": 1, "name": "a, b", "required": True, "type": "int"},
            {"id": 2, "name": "O'Brien", "required": True, "type": "int"},
        ],
    }
    r = repr(Schema.from_json(schema_json(schema)))
    assert "'a, b'" in r  # comma name quoted (Python repr) stays unambiguous
    assert '"O\'Brien"' in r  # apostrophe name flips to double quotes, like Python repr


def test_empty_schema_serializes_to_json_and_arrow():
    h = Schema.from_json(schema_json(EMPTY_SCHEMA))
    assert Schema.from_json(h.to_json()).column_names == []
    pytest.importorskip("pyarrow")
    assert len(h.to_arrow_schema()) == 0
