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

import pyarrow as pa
import pytest

from pyiceberg_core.schema import Schema


SCHEMA_DICT = {
    "type": "struct",
    "schema-id": 7,
    "identifier-field-ids": [4, 1],
    "fields": [
        {"id": 1, "name": "id", "required": True, "type": "long"},
        {"id": 2, "name": "name", "required": False, "type": "string"},
        {"id": 3, "name": "score", "required": False, "type": "double"},
        {"id": 4, "name": "user_id", "required": True, "type": "int"},
    ],
}

NESTED_SCHEMA_DICT = {
    "type": "struct",
    "schema-id": 8,
    "fields": [
        {"id": 1, "name": "id", "required": True, "type": "long"},
        {
            "id": 2,
            "name": "tags",
            "required": False,
            "type": {
                "type": "list",
                "element-id": 3,
                "element": "string",
                "element-required": True,
            },
        },
        {
            "id": 4,
            "name": "properties",
            "required": False,
            "type": {
                "type": "map",
                "key-id": 5,
                "key": "string",
                "value-id": 6,
                "value": "int",
                "value-required": True,
            },
        },
        {
            "id": 7,
            "name": "person",
            "required": False,
            "type": {
                "type": "struct",
                "fields": [
                    {"id": 8, "name": "first_name", "required": False, "type": "string"},
                    {"id": 9, "name": "age", "required": True, "type": "int"},
                ],
            },
        },
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


def test_schema_from_json_exposes_fields() -> None:
    schema = Schema.from_json(schema_json(SCHEMA_DICT))

    assert schema.schema_id() == 7
    assert schema.highest_field_id() == 4
    assert schema.identifier_field_ids() == [1, 4]
    assert schema.column_names() == ["id", "name", "score", "user_id"]

    field = schema.field_by_id(2)
    assert field["id"] == 2
    assert field["name"] == "name"
    assert json.loads(field["type"]) == "string"
    assert field["required"] is False

    assert schema.find_field_by_name("id")["id"] == 1
    assert schema.find_field_by_name("missing") is None

    with pytest.raises(KeyError, match="No field with id 99"):
        schema.field_by_id(99)


@pytest.mark.parametrize(
    "schema",
    [
        "{not valid json}",
        {"type": "struct"},
        {"type": "struct", "schema-id": "bad", "fields": []},
        {"type": "struct", "fields": [{"id": 1, "name": "x", "required": True, "type": "bogus"}]},
    ],
    ids=["bad-json", "missing-fields", "bad-schema-id", "bad-type"],
)
def test_schema_from_json_rejects_malformed_input(schema) -> None:
    with pytest.raises(ValueError):
        Schema.from_json(schema if isinstance(schema, str) else schema_json(schema))


def test_nested_field_lookup_uses_iceberg_type_json() -> None:
    schema = Schema.from_json(schema_json(NESTED_SCHEMA_DICT))

    tags = schema.find_field_by_name("tags")
    assert tags["id"] == 2
    assert json.loads(tags["type"]) == {
        "type": "list",
        "element-id": 3,
        "element": "string",
        "element-required": True,
    }

    first_name = schema.find_field_by_name("person.first_name")
    assert first_name["id"] == 8
    assert json.loads(first_name["type"]) == "string"

    assert schema.find_field_by_name("person.First_Name") is None


def test_schema_to_json_round_trips() -> None:
    schema = Schema.from_json(schema_json(NESTED_SCHEMA_DICT))
    round_tripped = Schema.from_json(schema.to_json())

    assert round_tripped.schema_id() == schema.schema_id()
    assert round_tripped.column_names() == schema.column_names()
    assert round_tripped.highest_field_id() == schema.highest_field_id()
    assert round_tripped.field_by_id(8) == schema.field_by_id(8)


def test_schema_to_arrow_schema() -> None:
    schema = Schema.from_json(schema_json(SCHEMA_DICT))

    arrow_schema = schema.to_arrow_schema()

    assert isinstance(arrow_schema, pa.Schema)
    assert arrow_schema.names == ["id", "name", "score", "user_id"]
    assert arrow_schema.field("id").type == pa.int64()
    assert arrow_schema.field("id").nullable is False
    assert arrow_schema.field("name").type == pa.string()
    assert arrow_schema.field("name").nullable is True
    assert [field_id(field) for field in arrow_schema] == [1, 2, 3, 4]


def test_to_arrow_schema_preserves_nested_field_ids() -> None:
    arrow_schema = Schema.from_json(schema_json(NESTED_SCHEMA_DICT)).to_arrow_schema()

    assert field_id(arrow_schema.field("tags")) == 2
    assert field_id(arrow_schema.field("tags").type.value_field) == 3

    properties_type = arrow_schema.field("properties").type
    assert field_id(arrow_schema.field("properties")) == 4
    assert field_id(properties_type.key_field) == 5
    assert field_id(properties_type.item_field) == 6

    person_type = arrow_schema.field("person").type
    assert field_id(arrow_schema.field("person")) == 7
    assert field_id(person_type.field("first_name")) == 8
    assert field_id(person_type.field("age")) == 9


def test_arrow_c_schema_capsule_imports_through_pyarrow() -> None:
    schema = Schema.from_json(schema_json(NESTED_SCHEMA_DICT))

    assert capsule_name(schema.__arrow_c_schema__()) == "arrow_schema"
    assert pa.schema(schema).equals(schema.to_arrow_schema())


def test_internal_schema_capsule_name_and_lifetime() -> None:
    schema = Schema.from_json(schema_json(SCHEMA_DICT))
    capsule = schema._capsule()

    assert capsule_name(capsule) == "iceberg_core_schema"

    del schema
    assert capsule_name(capsule) == "iceberg_core_schema"


def test_schema_repr_is_stable_enough_for_debugging() -> None:
    text = repr(Schema.from_json(schema_json(NESTED_SCHEMA_DICT)))

    assert text.startswith("Schema(")
    assert "schema_id=8" in text
    assert "fields=4" in text
    assert "columns=[id, tags, properties, person]" in text
