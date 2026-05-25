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


def test_arrow_reader_max_rows_behavior():
    reader = ArrowReader(FileIO.from_props({}))

    # max_rows=0
    batch_reader = reader.read(id_schema(), [], max_rows=0)
    assert isinstance(batch_reader, pa.RecordBatchReader)
    assert batch_reader.schema.names == ["id"]
    with pytest.raises(StopIteration):
        batch_reader.read_next_batch()

    # max_rows=None
    batch_reader_none = reader.read(id_schema(), [], max_rows=None)
    assert isinstance(batch_reader_none, pa.RecordBatchReader)
    assert batch_reader_none.schema.names == ["id"]
    with pytest.raises(StopIteration):
        batch_reader_none.read_next_batch()


def test_arrow_reader_with_real_parquet_and_limits(tmp_path):
    import os

    import pyarrow.parquet as pq

    # Write a small parquet file with 5 rows
    table = pa.table({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
    local_path = str(tmp_path / "data.parquet")
    pq.write_table(table, local_path)
    file_path = "file://" + local_path
    file_size = os.path.getsize(local_path)

    reader = ArrowReader(FileIO.from_props({}))

    # 1. Test reading all
    task = FileScanTask(schema(), file_path, file_size, [1, 2])
    batch_reader = reader.read(schema(), [task])
    res_table = batch_reader.read_all()
    assert len(res_table) == 5
    assert res_table.column("id").to_pylist() == [1, 2, 3, 4, 5]

    # 2. Test max_rows = 3
    task_limit = FileScanTask(schema(), file_path, file_size, [1, 2])
    batch_reader_limit = reader.read(schema(), [task_limit], max_rows=3)
    res_table_limit = batch_reader_limit.read_all()
    assert len(res_table_limit) == 3
    assert res_table_limit.column("id").to_pylist() == [1, 2, 3]

    # 3. Test max_rows = 0
    task_limit_0 = FileScanTask(schema(), file_path, file_size, [1, 2])
    batch_reader_limit_0 = reader.read(schema(), [task_limit_0], max_rows=0)
    res_table_limit_0 = batch_reader_limit_0.read_all()
    assert len(res_table_limit_0) == 0


TABLE_METADATA_JSON = json.dumps(
    {
        "format-version": 2,
        "table-uuid": "fb070e82-2d1f-4ef6-8ab6-c4d12c6ed490",
        "location": "s3://bucket/table",
        "last-sequence-number": 1,
        "last-updated-ms": 1600000000000,
        "last-column-id": 2,
        "schemas": [
            {
                "schema-id": 1,
                "type": "struct",
                "fields": [
                    {"id": 1, "name": "id", "required": True, "type": "long"},
                    {"id": 2, "name": "name", "required": False, "type": "string"},
                ],
            }
        ],
        "current-schema-id": 1,
        "partition-specs": [{"spec-id": 0, "fields": []}],
        "default-spec-id": 0,
        "last-partition-id": 1000,
        "default-sort-order-id": 0,
        "sort-orders": [{"order-id": 0, "fields": []}],
        "properties": {},
        "current-snapshot-id": -1,
        "snapshots": [],
        "snapshot-log": [],
        "metadata-log": [],
    }
)


def test_table_from_metadata_json_validation():
    from pyiceberg_core.scan import Table

    # 1. Test metadata JSON validation error
    with pytest.raises(ValueError, match="Failed to parse metadata JSON"):
        Table.from_metadata_json(FileIO.from_props({}), ["ns", "tbl"], "{invalid_json}")

    # 2. Test invalid identifier (empty sequence)
    with pytest.raises(ValueError, match="Table identifier can't be empty"):
        Table.from_metadata_json(FileIO.from_props({}), [], TABLE_METADATA_JSON)


def test_table_empty_table_planning():
    from pyiceberg_core.scan import Table

    # 3. Test successful parsing and empty table planning
    table = Table.from_metadata_json(
        FileIO.from_props({}),
        ["ns", "tbl"],
        TABLE_METADATA_JSON,
    )

    # 4. Test planning files on empty table returns empty list
    tasks = table.plan_files()
    assert isinstance(tasks, list)
    assert len(tasks) == 0

    # 5. Test selected field planning works (does not raise error)
    tasks_projected = table.plan_files(selected_fields=["id"])
    assert len(tasks_projected) == 0

    tasks_empty = table.plan_files(selected_fields=[])
    assert len(tasks_empty) == 0

    # 6. Test predicate argument acceptance
    from pyiceberg_core.expression import Reference

    tasks_pred = table.plan_files(predicate=Reference("id").eq(5))
    assert len(tasks_pred) == 0


TABLE_METADATA_WITH_SNAPSHOT_JSON = json.dumps(
    {
        "format-version": 2,
        "table-uuid": "fb070e82-2d1f-4ef6-8ab6-c4d12c6ed490",
        "location": "s3://bucket/table",
        "last-sequence-number": 1,
        "last-updated-ms": 1600000000000,
        "last-column-id": 2,
        "schemas": [
            {
                "schema-id": 1,
                "type": "struct",
                "fields": [
                    {"id": 1, "name": "id", "required": True, "type": "long"},
                    {"id": 2, "name": "name", "required": False, "type": "string"},
                ],
            }
        ],
        "current-schema-id": 1,
        "partition-specs": [{"spec-id": 0, "fields": []}],
        "default-spec-id": 0,
        "last-partition-id": 1000,
        "default-sort-order-id": 0,
        "sort-orders": [{"order-id": 0, "fields": []}],
        "properties": {},
        "current-snapshot-id": 1,
        "snapshots": [
            {
                "snapshot-id": 1,
                "timestamp-ms": 1600000000000,
                "summary": {"operation": "append"},
                "manifest-list": "s3://bucket/table/metadata/snap-1.avro",
                "schema-id": 1
            }
        ],
        "snapshot-log": [],
        "metadata-log": [],
    }
)


def test_table_read_arrow():
    from pyiceberg_core.scan import Table
    from pyiceberg_core.expression import Reference

    table = Table.from_metadata_json(
        FileIO.from_props({}),
        ["ns", "tbl"],
        TABLE_METADATA_JSON,
    )

    # 1. Test empty table reader
    reader = table.read_arrow(schema())
    assert isinstance(reader, pa.RecordBatchReader)
    assert reader.schema.names == ["id", "name"]
    with pytest.raises(StopIteration):
        reader.read_next_batch()

    reader_projected = table.read_arrow(id_schema(), selected_fields=["id"])
    assert reader_projected.schema.names == ["id"]

    with pytest.raises(ValueError, match="output_schema columns"):
        table.read_arrow(schema(), selected_fields=["id"])

    # 2. Test max_rows=0
    reader_limit_0 = table.read_arrow(schema(), max_rows=0)
    assert isinstance(reader_limit_0, pa.RecordBatchReader)
    with pytest.raises(StopIteration):
        reader_limit_0.read_next_batch()

    # 3. Test invalid selected field
    table_with_snapshot = Table.from_metadata_json(
        FileIO.from_props({}),
        ["ns", "tbl"],
        TABLE_METADATA_WITH_SNAPSHOT_JSON,
    )
    with pytest.raises(ValueError, match="Column missing not found in table"):
        table_with_snapshot.read_arrow(schema(), selected_fields=["missing"])

    # 4. Test filter binding
    reader_filtered = table.read_arrow(schema(), predicate=Reference("id").eq(5))
    assert isinstance(reader_filtered, pa.RecordBatchReader)

    # 5. Test unbindable filter (column not in schema)
    with pytest.raises(ValueError, match="Field missing not found in schema"):
        table_with_snapshot.plan_files(predicate=Reference("missing").eq(5))
