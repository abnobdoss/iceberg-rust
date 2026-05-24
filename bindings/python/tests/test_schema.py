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
                        {"id": 13, "name": "latitude", "required": False, "type": "float"},
                        {"id": 14, "name": "longitude", "required": False, "type": "float"},
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
    assert handle.schema_id() == expected_id
    assert handle.highest_field_id() == highest_id


@pytest.mark.parametrize(
    "schema",
    [
        "{not valid json}",
        {"type": "struct"},
        {"type": "struct", "schema-id": "bad", "fields": []},
        {"type": "struct", "fields": [{"id": 1, "name": "x", "required": True, "type": "bogus"}]},
        {
            "type": "struct",
            "fields": [
                {"id": 1, "name": "a", "required": True, "type": "int"},
                {"id": 1, "name": "b", "required": True, "type": "string"},
            ],
        },
        {
            "type": "struct",
            "fields": [{"id": 1, "name": "a", "required": True, "type": "float"}],
            "identifier-field-ids": [1],
        },
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
def test_from_json_rejects_malformed_schema_families(schema):
    with pytest.raises(ValueError):
        Schema.from_json(schema if isinstance(schema, str) else schema_json(schema))


def test_column_names_are_top_level_only():
    assert Schema.from_json(schema_json(NESTED_SCHEMA)).column_names() == [
        "foo",
        "bar",
        "baz",
        "qux",
        "quux",
        "location",
        "person",
    ]


@pytest.mark.parametrize(
    "schema,expected",
    [(SIMPLE_SCHEMA, [2]), (IDENTIFIER_SCHEMA, [1, 4]), ({**SIMPLE_SCHEMA, "identifier-field-ids": []}, [])],
)
def test_identifier_field_ids(schema, expected):
    assert Schema.from_json(schema_json(schema)).identifier_field_ids() == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("bar", {"id": 2, "name": "bar", "type": "int", "required": True}),
        ("person.name", {"id": 16, "name": "name", "type": "string", "required": False}),
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
    field_type = json.loads(field["type"])
    if isinstance(expected["type"], dict):
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
    assert Schema.from_json(schema_json(NESTED_SCHEMA)).field_by_id(field_id_value)["name"] == name


def test_field_by_id_missing_raises_key_error():
    with pytest.raises(KeyError):
        Schema.from_json(schema_json(SIMPLE_SCHEMA)).field_by_id(99)


@pytest.mark.parametrize("schema", [SIMPLE_SCHEMA, NESTED_SCHEMA, IDENTIFIER_SCHEMA, V1_SCHEMA])
def test_to_json_round_trips_semantically(schema):
    handle = Schema.from_json(schema_json(schema))
    reparsed = Schema.from_json(handle.to_json())
    assert reparsed.schema_id() == handle.schema_id()
    assert reparsed.column_names() == handle.column_names()
    assert reparsed.highest_field_id() == handle.highest_field_id()
    assert reparsed.identifier_field_ids() == handle.identifier_field_ids()


def test_capsule_names_and_lifetime():
    handle = Schema.from_json(schema_json(SIMPLE_SCHEMA))
    capsule = handle._capsule()
    assert capsule_name(capsule) == "iceberg_core_schema"

    del handle
    assert capsule_name(capsule) == "iceberg_core_schema"


def test_arrow_c_schema_capsule_name():
    assert capsule_name(Schema.from_json(schema_json(SIMPLE_SCHEMA)).__arrow_c_schema__()) == "arrow_schema"


def test_repr_is_stable_enough_for_debugging():
    text = repr(Schema.from_json(schema_json(NESTED_SCHEMA)))
    assert text.startswith("Schema(")
    assert "schema_id=7" in text
    assert "fields=7" in text
    assert "0x" not in text


def test_to_arrow_schema_returns_pyarrow_schema():
    pa = pytest.importorskip("pyarrow")
    arrow_schema = Schema.from_json(schema_json(SIMPLE_SCHEMA)).to_arrow_schema()
    assert isinstance(arrow_schema, pa.Schema)
    assert [field_id(field) for field in arrow_schema] == [1, 2, 3]


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
