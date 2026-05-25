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

import json

import pyarrow as pa
import pytest

from pyiceberg_core.expression import Reference
from pyiceberg_core.file_io import FileIO
from pyiceberg_core.scan import ArrowReader, DeleteFile, FileScanTask
from pyiceberg_core.schema import Schema


def schema() -> Schema:
    return Schema.from_json(
        json.dumps(
            {
                "type": "struct",
                "schema-id": 1,
                "fields": [
                    {"id": 1, "name": "id", "required": True, "type": "long"},
                    {"id": 2, "name": "name", "required": False, "type": "string"},
                ],
            }
        )
    )


def id_schema() -> Schema:
    return Schema.from_json(
        json.dumps(
            {
                "type": "struct",
                "schema-id": 1,
                "fields": [
                    {"id": 1, "name": "id", "required": True, "type": "long"},
                ],
            }
        )
    )


def id_file_schema() -> Schema:
    return Schema.from_json(
        json.dumps(
            {
                "type": "struct",
                "schema-id": 1,
                "fields": [
                    {"id": 1, "name": "id", "required": True, "type": "long"},
                    {"id": 2147483646, "name": "_file", "required": True, "type": "string"},
                ],
            }
        )
    )


def test_delete_file_position_properties():
    delete = DeleteFile(
        "s3://bucket/delete.parquet",
        128,
        "position-deletes",
        partition_spec_id=3,
    )

    assert delete.file_path == "s3://bucket/delete.parquet"
    assert delete.file_size_in_bytes == 128
    assert delete.file_type == "position-deletes"
    assert delete.partition_spec_id == 3
    assert delete.equality_ids is None
    assert "position-deletes" in repr(delete)


def test_delete_file_equality_properties():
    delete = DeleteFile(
        "s3://bucket/eq-delete.parquet",
        256,
        "equality-deletes",
        equality_ids=[1, 2],
    )

    assert delete.file_type == "equality-deletes"
    assert delete.equality_ids == [1, 2]


@pytest.mark.parametrize("file_type", ["data", "unknown"])
def test_delete_file_rejects_invalid_content(file_type):
    with pytest.raises(ValueError, match="Unsupported delete file type"):
        DeleteFile("s3://bucket/file.parquet", 1, file_type)


def test_file_scan_task_properties_without_deletes():
    task = FileScanTask(
        schema(),
        "s3://bucket/data.parquet",
        1024,
        [1, 2],
        record_count=10,
    )

    assert task.data_file_path == "s3://bucket/data.parquet"
    assert task.file_size_in_bytes == 1024
    assert task.start == 0
    assert task.length == 1024
    assert task.record_count == 10
    assert task.data_file_format == "parquet"
    assert task.project_field_ids == [1, 2]
    assert task.delete_count == 0
    assert task.has_predicate is False
    assert task.case_sensitive is True


def test_file_scan_task_default_length_uses_remaining_file_size():
    task = FileScanTask(
        schema(),
        "s3://bucket/data.parquet",
        1024,
        [1],
        start=128,
    )

    assert task.start == 128
    assert task.length == 896


def test_file_scan_task_binds_predicate_and_deletes():
    delete = DeleteFile("s3://bucket/delete.parquet", 128, "position-deletes")
    task = FileScanTask(
        schema(),
        "s3://bucket/data.parquet",
        1024,
        [1],
        length=512,
        predicate=Reference("id").gte(5),
        deletes=[delete],
        case_sensitive=True,
    )

    assert task.length == 512
    assert task.delete_count == 1
    assert task.has_predicate is True


def test_file_scan_task_accepts_partition_context_and_name_mapping():
    partition_spec = json.dumps(
        {
            "spec-id": 1,
            "fields": [
                {
                    "source-id": 1,
                    "field-id": 1000,
                    "name": "id",
                    "transform": "identity",
                }
            ],
        }
    )
    name_mapping = json.dumps([{"field-id": 1, "names": ["id", "record_id"]}])

    task = FileScanTask(
        schema(),
        "s3://bucket/data.parquet",
        1024,
        [1],
        partition_data=[7],
        partition_spec=partition_spec,
        name_mapping=name_mapping,
    )

    assert task.has_partition_data is True
    assert task.has_partition_spec is True
    assert task.has_name_mapping is True


def test_file_scan_task_accepts_partition_data_json_array():
    partition_spec = json.dumps(
        {
            "fields": [
                {
                    "source-id": 2,
                    "field-id": 1000,
                    "name": "name",
                    "transform": "identity",
                }
            ]
        }
    )

    task = FileScanTask(
        schema(),
        "s3://bucket/data.parquet",
        1024,
        [2],
        partition_data=json.dumps(["alice"]),
        partition_spec=partition_spec,
    )

    assert task.has_partition_data is True
    assert task.has_partition_spec is True
    assert task.has_name_mapping is False


def test_file_scan_task_rejects_partition_data_without_spec():
    with pytest.raises(ValueError, match="partition_spec is required"):
        FileScanTask(
            schema(),
            "s3://bucket/data.parquet",
            1024,
            [1],
            partition_data=[7],
        )


def test_file_scan_task_rejects_partition_data_length_mismatch():
    partition_spec = json.dumps(
        {
            "fields": [
                {
                    "source-id": 1,
                    "field-id": 1000,
                    "name": "id",
                    "transform": "identity",
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="partition_data length"):
        FileScanTask(
            schema(),
            "s3://bucket/data.parquet",
            1024,
            [1],
            partition_data=[],
            partition_spec=partition_spec,
        )


def test_file_scan_task_rejects_start_beyond_file_size():
    with pytest.raises(ValueError, match="start \\(1025\\).*file_size_in_bytes \\(1024\\)"):
        FileScanTask(
            schema(),
            "s3://bucket/data.parquet",
            1024,
            [1],
            start=1025,
        )


def test_file_scan_task_rejects_length_beyond_file_size():
    with pytest.raises(ValueError, match="start \\(128\\) \\+ length \\(897\\)"):
        FileScanTask(
            schema(),
            "s3://bucket/data.parquet",
            1024,
            [1],
            start=128,
            length=897,
        )


def test_file_scan_task_rejects_start_plus_length_overflow():
    with pytest.raises(ValueError, match="overflows u64"):
        FileScanTask(
            schema(),
            "s3://bucket/data.parquet",
            2**64 - 1,
            [1],
            start=2**64 - 2,
            length=2,
        )


def test_file_scan_task_rejects_unbindable_predicate():
    with pytest.raises(ValueError, match="Field missing not found in schema"):
        FileScanTask(
            schema(),
            "s3://bucket/data.parquet",
            1024,
            [1],
            predicate=Reference("missing").eq(5),
        )


def test_file_scan_task_rejects_non_delete_sequence_item():
    with pytest.raises(TypeError):
        FileScanTask(
            schema(),
            "s3://bucket/data.parquet",
            1024,
            [1],
            deletes=["not a delete file"],
        )


def test_arrow_reader_returns_pyarrow_record_batch_reader_for_empty_task_stream():
    reader = ArrowReader(FileIO.from_props({}))

    batch_reader = reader.read(schema(), [])

    assert isinstance(batch_reader, pa.RecordBatchReader)
    assert batch_reader.schema.names == ["id", "name"]
    with pytest.raises(StopIteration):
        batch_reader.read_next_batch()


def test_arrow_reader_rejects_output_schema_that_does_not_match_task_projection():
    reader = ArrowReader(FileIO.from_props({}))
    task = FileScanTask(
        schema(),
        "s3://bucket/data.parquet",
        1024,
        [1],
    )

    with pytest.raises(ValueError, match="output_schema field ids .* project_field_ids"):
        reader.read(schema(), [task])

    projected_reader = reader.read(id_schema(), [task])
    assert isinstance(projected_reader, pa.RecordBatchReader)


def test_arrow_reader_metadata_projection_no_longer_fails():
    reader = ArrowReader(FileIO.from_props({}))
    task = FileScanTask(
        schema(),
        "s3://bucket/data.parquet",
        1024,
        [1, 2147483646],
    )

    projected_reader = reader.read(id_file_schema(), [task])
    assert isinstance(projected_reader, pa.RecordBatchReader)
    assert projected_reader.schema.names == ["id", "_file"]
    
    file_field = projected_reader.schema.field("_file")
    assert "run_end_encoded" in str(file_field.type) or pa.types.is_run_end_encoded(file_field.type)


def test_arrow_reader_rejects_empty_metadata_projection_without_task_schema():
    reader = ArrowReader(FileIO.from_props({}))

    with pytest.raises(ValueError, match="cannot infer the exact Arrow schema"):
        reader.read(id_file_schema(), [])


def test_arrow_reader_partition_projection_no_longer_fails():
    partition_spec = json.dumps(
        {
            "spec-id": 1,
            "fields": [
                {
                    "source-id": 1,
                    "field-id": 1000,
                    "name": "id",
                    "transform": "identity",
                }
            ],
        }
    )
    reader = ArrowReader(FileIO.from_props({}))
    task = FileScanTask(
        schema(),
        "s3://bucket/data.parquet",
        1024,
        [1],
        partition_data=[7],
        partition_spec=partition_spec,
    )

    projected_reader = reader.read(id_schema(), [task])
    assert isinstance(projected_reader, pa.RecordBatchReader)
    assert projected_reader.schema.names == ["id"]
    
    id_field = projected_reader.schema.field("id")
    assert "run_end_encoded" in str(id_field.type) or pa.types.is_run_end_encoded(id_field.type)


def test_arrow_reader_rejects_tasks_with_different_physical_schemas():
    partition_spec = json.dumps(
        {
            "spec-id": 1,
            "fields": [
                {
                    "source-id": 1,
                    "field-id": 1000,
                    "name": "id",
                    "transform": "identity",
                }
            ],
        }
    )
    reader = ArrowReader(FileIO.from_props({}))
    constant_task = FileScanTask(
        schema(),
        "s3://bucket/partitioned.parquet",
        1024,
        [1],
        partition_data=[7],
        partition_spec=partition_spec,
    )
    plain_task = FileScanTask(
        schema(),
        "s3://bucket/plain.parquet",
        1024,
        [1],
    )

    with pytest.raises(ValueError, match="same Arrow schema"):
        reader.read(id_schema(), [constant_task, plain_task])
