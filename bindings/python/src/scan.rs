// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::pyarrow::IntoPyArrow;
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use futures::{StreamExt, stream};
use iceberg::arrow::{ArrowReaderBuilder, schema_to_arrow_schema};
use iceberg::arrow::record_batch_transformer::RecordBatchTransformer;
use iceberg::expr::Bind;
use iceberg::scan::{
    ArrowRecordBatchStream, FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream,
};
use iceberg::spec::{
    DataContentType, DataFileFormat, Literal, NameMapping, PartitionSpec, Struct,
    UnboundPartitionSpec,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyFloat, PyInt, PySequence, PyString};
use serde_json::{Number as JsonNumber, Value as JsonValue};

use crate::expression::PyPredicate;
use crate::file_io::PyFileIO;
use crate::runtime::{iceberg_runtime, runtime};
use crate::schema::PySchema;

fn parse_data_file_format(value: &str) -> PyResult<DataFileFormat> {
    match value.to_ascii_lowercase().as_str() {
        "parquet" => Ok(DataFileFormat::Parquet),
        "avro" => Ok(DataFileFormat::Avro),
        "orc" => Ok(DataFileFormat::Orc),
        "puffin" => Ok(DataFileFormat::Puffin),
        other => Err(PyValueError::new_err(format!(
            "Unsupported data file format: {other}"
        ))),
    }
}

fn format_data_file_format(value: DataFileFormat) -> &'static str {
    match value {
        DataFileFormat::Parquet => "parquet",
        DataFileFormat::Avro => "avro",
        DataFileFormat::Orc => "orc",
        DataFileFormat::Puffin => "puffin",
    }
}

fn parse_delete_file_type(value: &str) -> PyResult<DataContentType> {
    match value.to_ascii_lowercase().as_str() {
        "position" | "position-delete" | "position-deletes" | "positional" => {
            Ok(DataContentType::PositionDeletes)
        }
        "equality" | "equality-delete" | "equality-deletes" => Ok(DataContentType::EqualityDeletes),
        other => Err(PyValueError::new_err(format!(
            "Unsupported delete file type: {other}"
        ))),
    }
}

fn format_delete_file_type(value: DataContentType) -> PyResult<&'static str> {
    match value {
        DataContentType::PositionDeletes => Ok("position-deletes"),
        DataContentType::EqualityDeletes => Ok("equality-deletes"),
        DataContentType::Data => Err(PyValueError::new_err(
            "Data content is not valid for a delete file",
        )),
    }
}

fn py_deletes_to_rust(values: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<FileScanTaskDeleteFile>> {
    let Some(values) = values else {
        return Ok(vec![]);
    };

    let seq = values.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("deletes must be a sequence of pyiceberg_core.scan.DeleteFile values")
    })?;
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let item = seq.get_item(i)?;
        let delete = item.extract::<PyRef<'_, PyDeleteFile>>()?;
        out.push(delete.inner.clone());
    }
    Ok(out)
}

fn py_tasks_to_rust(values: &Bound<'_, PyAny>) -> PyResult<Vec<FileScanTask>> {
    let seq = values.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("tasks must be a sequence of pyiceberg_core.scan.FileScanTask values")
    })?;
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let item = seq.get_item(i)?;
        let task = item.extract::<PyRef<'_, PyFileScanTask>>()?;
        out.push(task.inner.clone());
    }
    Ok(out)
}

fn parse_partition_spec(
    value: Option<&str>,
    schema: &PySchema,
) -> PyResult<Option<Arc<PartitionSpec>>> {
    value
        .map(|value| {
            let spec: UnboundPartitionSpec = serde_json::from_str(value).map_err(|e| {
                PyValueError::new_err(format!("Failed to parse partition_spec JSON: {e}"))
            })?;
            spec.bind(schema.inner.clone())
                .map(Arc::new)
                .map_err(crate::error::to_py_err)
        })
        .transpose()
}

fn parse_name_mapping(value: Option<&str>) -> PyResult<Option<Arc<NameMapping>>> {
    value
        .map(|value| {
            serde_json::from_str(value).map(Arc::new).map_err(|e| {
                PyValueError::new_err(format!("Failed to parse name_mapping JSON: {e}"))
            })
        })
        .transpose()
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn py_to_json_value(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if value.is_none() {
        return Ok(JsonValue::Null);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(JsonValue::Bool(value.extract()?));
    }
    if value.is_instance_of::<PyInt>() {
        let v = value.extract::<i64>().map_err(|_| {
            PyValueError::new_err(format!(
                "integer {} exceeds i64 range; partition values require JSON-compatible integers",
                value
                    .str()
                    .and_then(|s| s.extract::<String>())
                    .unwrap_or_else(|_| "<unprintable>".to_string()),
            ))
        })?;
        return Ok(JsonValue::Number(v.into()));
    }
    if value.is_instance_of::<PyFloat>() {
        let v = value.extract::<f64>()?;
        let number = JsonNumber::from_f64(v)
            .ok_or_else(|| PyValueError::new_err("partition float values must be finite"))?;
        return Ok(JsonValue::Number(number));
    }
    if let Ok(v) = value.extract::<String>() {
        return Ok(JsonValue::String(v));
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(JsonValue::String(bytes_to_hex(bytes.as_bytes())));
    }
    Err(PyTypeError::new_err(format!(
        "Cannot convert partition value to Iceberg JSON value: {}",
        value.repr()?.to_str()?
    )))
}

fn partition_values_to_json_array(values: &Bound<'_, PyAny>) -> PyResult<Vec<JsonValue>> {
    if values.is_instance_of::<PyString>() {
        match serde_json::from_str::<JsonValue>(&values.extract::<String>()?) {
            Ok(JsonValue::Array(values)) => return Ok(values),
            Ok(_) => {
                return Err(PyTypeError::new_err(
                    "partition_data JSON string must contain an array",
                ));
            }
            Err(e) => {
                return Err(PyValueError::new_err(format!(
                    "Failed to parse partition_data JSON: {e}"
                )));
            }
        }
    }
    if values.is_instance_of::<PyBytes>() {
        return Err(PyTypeError::new_err(
            "partition_data must be a sequence of values or a JSON array string, not bytes",
        ));
    }

    let seq = values.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("partition_data must be a sequence of values or a JSON array string")
    })?;
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(py_to_json_value(&seq.get_item(i)?)?);
    }
    Ok(out)
}

fn parse_partition_data(
    value: Option<&Bound<'_, PyAny>>,
    partition_spec: Option<&Arc<PartitionSpec>>,
    schema: &PySchema,
) -> PyResult<Option<Struct>> {
    match (value, partition_spec) {
        (None, None) => Ok(None),
        (None, Some(_)) => Ok(None),
        (Some(_), None) => Err(PyValueError::new_err(
            "partition_spec is required when partition_data is provided",
        )),
        (Some(value), Some(partition_spec)) => {
            let values = partition_values_to_json_array(value)?;
            let partition_type = partition_spec
                .partition_type(schema.inner.as_ref())
                .map_err(crate::error::to_py_err)?;
            let fields = partition_type.fields();
            if values.len() != fields.len() {
                return Err(PyValueError::new_err(format!(
                    "partition_data length {} does not match partition_spec field count {}",
                    values.len(),
                    fields.len()
                )));
            }

            let literals = values
                .into_iter()
                .zip(fields.iter())
                .map(|(value, field)| {
                    Literal::try_from_json(value, &field.field_type)
                        .map_err(crate::error::to_py_err)
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(Some(Struct::from_iter(literals)))
        }
    }
}

fn validate_scan_range(file_size_in_bytes: u64, start: u64, length: Option<u64>) -> PyResult<u64> {
    if start > file_size_in_bytes {
        return Err(PyValueError::new_err(format!(
            "start ({start}) must be less than or equal to file_size_in_bytes ({file_size_in_bytes})"
        )));
    }

    let length = length.unwrap_or(file_size_in_bytes - start);
    let end = start.checked_add(length).ok_or_else(|| {
        PyValueError::new_err(format!("start ({start}) + length ({length}) overflows u64"))
    })?;
    if end > file_size_in_bytes {
        return Err(PyValueError::new_err(format!(
            "start ({start}) + length ({length}) must be less than or equal to file_size_in_bytes ({file_size_in_bytes})"
        )));
    }

    Ok(length)
}

fn schema_top_level_field_ids(schema: &PySchema) -> Vec<i32> {
    schema
        .inner
        .as_struct()
        .fields()
        .iter()
        .map(|field| field.id)
        .collect()
}

fn validate_reader_projection(output_schema: &PySchema, tasks: &[FileScanTask]) -> PyResult<()> {
    let output_field_ids = schema_top_level_field_ids(output_schema);
    for task in tasks {
        if task.project_field_ids != output_field_ids {
            return Err(PyValueError::new_err(format!(
                "output_schema field ids {:?} must match task project_field_ids {:?} for {}; pass the projected schema used to build the tasks",
                output_field_ids, task.project_field_ids, task.data_file_path
            )));
        }
    }
    Ok(())
}

#[pyclass(
    name = "DeleteFile",
    module = "pyiceberg_core.scan",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyDeleteFile {
    pub(crate) inner: FileScanTaskDeleteFile,
}

#[pymethods]
impl PyDeleteFile {
    #[new]
    #[pyo3(signature = (
        file_path,
        file_size_in_bytes,
        file_type,
        *,
        partition_spec_id = 0,
        equality_ids = None
    ))]
    fn new(
        file_path: String,
        file_size_in_bytes: u64,
        file_type: &str,
        partition_spec_id: i32,
        equality_ids: Option<Vec<i32>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FileScanTaskDeleteFile {
                file_path,
                file_size_in_bytes,
                file_type: parse_delete_file_type(file_type)?,
                partition_spec_id,
                equality_ids,
            },
        })
    }

    #[getter]
    fn file_path(&self) -> &str {
        &self.inner.file_path
    }

    #[getter]
    fn file_size_in_bytes(&self) -> u64 {
        self.inner.file_size_in_bytes
    }

    #[getter]
    fn file_type(&self) -> PyResult<&'static str> {
        format_delete_file_type(self.inner.file_type)
    }

    #[getter]
    fn partition_spec_id(&self) -> i32 {
        self.inner.partition_spec_id
    }

    #[getter]
    fn equality_ids(&self) -> Option<Vec<i32>> {
        self.inner.equality_ids.clone()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "DeleteFile(file_path={:?}, file_type={}, file_size_in_bytes={})",
            self.inner.file_path,
            format_delete_file_type(self.inner.file_type)?,
            self.inner.file_size_in_bytes
        ))
    }
}

#[pyclass(
    name = "FileScanTask",
    module = "pyiceberg_core.scan",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFileScanTask {
    pub(crate) inner: FileScanTask,
}

#[pymethods]
impl PyFileScanTask {
    #[new]
    #[pyo3(signature = (
        schema,
        data_file_path,
        file_size_in_bytes,
        project_field_ids,
        *,
        start = 0,
        length = None,
        record_count = None,
        data_file_format = "parquet",
        predicate = None,
        deletes = None,
        partition_data = None,
        partition_spec = None,
        name_mapping = None,
        case_sensitive = true
    ))]
    fn new(
        schema: &PySchema,
        data_file_path: String,
        file_size_in_bytes: u64,
        project_field_ids: Vec<i32>,
        start: u64,
        length: Option<u64>,
        record_count: Option<u64>,
        data_file_format: &str,
        predicate: Option<&PyPredicate>,
        deletes: Option<&Bound<'_, PyAny>>,
        partition_data: Option<&Bound<'_, PyAny>>,
        partition_spec: Option<&str>,
        name_mapping: Option<&str>,
        case_sensitive: bool,
    ) -> PyResult<Self> {
        let partition_spec = parse_partition_spec(partition_spec, schema)?;
        let partition = parse_partition_data(partition_data, partition_spec.as_ref(), schema)?;
        let name_mapping = parse_name_mapping(name_mapping)?;
        let length = validate_scan_range(file_size_in_bytes, start, length)?;

        Ok(Self {
            inner: FileScanTask {
                file_size_in_bytes,
                start,
                length,
                record_count,
                data_file_path,
                data_file_format: parse_data_file_format(data_file_format)?,
                schema: schema.inner.clone(),
                project_field_ids,
                predicate: predicate
                    .map(|p| p.inner.clone().bind(schema.inner.clone(), case_sensitive))
                    .transpose()
                    .map_err(crate::error::to_py_err)?,
                deletes: py_deletes_to_rust(deletes)?,
                partition,
                partition_spec,
                name_mapping,
                case_sensitive,
            },
        })
    }

    #[getter]
    fn data_file_path(&self) -> &str {
        &self.inner.data_file_path
    }

    #[getter]
    fn file_size_in_bytes(&self) -> u64 {
        self.inner.file_size_in_bytes
    }

    #[getter]
    fn start(&self) -> u64 {
        self.inner.start
    }

    #[getter]
    fn length(&self) -> u64 {
        self.inner.length
    }

    #[getter]
    fn record_count(&self) -> Option<u64> {
        self.inner.record_count
    }

    #[getter]
    fn data_file_format(&self) -> &'static str {
        format_data_file_format(self.inner.data_file_format)
    }

    #[getter]
    fn project_field_ids(&self) -> Vec<i32> {
        self.inner.project_field_ids.clone()
    }

    #[getter]
    fn delete_count(&self) -> usize {
        self.inner.deletes.len()
    }

    #[getter]
    fn has_predicate(&self) -> bool {
        self.inner.predicate.is_some()
    }

    #[getter]
    fn has_partition_data(&self) -> bool {
        self.inner.partition.is_some()
    }

    #[getter]
    fn has_partition_spec(&self) -> bool {
        self.inner.partition_spec.is_some()
    }

    #[getter]
    fn has_name_mapping(&self) -> bool {
        self.inner.name_mapping.is_some()
    }

    #[getter]
    fn case_sensitive(&self) -> bool {
        self.inner.case_sensitive
    }

    fn __repr__(&self) -> String {
        format!(
            "FileScanTask(data_file_path={:?}, file_size_in_bytes={}, project_field_ids={:?}, deletes={})",
            self.inner.data_file_path,
            self.inner.file_size_in_bytes,
            self.inner.project_field_ids,
            self.inner.deletes.len()
        )
    }
}

struct BlockingArrowRecordBatchReader {
    schema: ArrowSchemaRef,
    stream: ArrowRecordBatchStream,
    remaining_rows: Option<usize>,
}

impl Iterator for BlockingArrowRecordBatchReader {
    type Item = ArrowResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(0) = self.remaining_rows {
            return None;
        }

        let batch = runtime()
            .block_on(self.stream.next())?
            .map_err(|err| ArrowError::ExternalError(Box::new(err)));

        match batch {
            Ok(batch) => {
                if let Some(rem) = self.remaining_rows {
                    if batch.num_rows() <= rem {
                        self.remaining_rows = Some(rem - batch.num_rows());
                        Some(Ok(batch))
                    } else {
                        let sliced = batch.slice(0, rem);
                        self.remaining_rows = Some(0);
                        Some(Ok(sliced))
                    }
                } else {
                    Some(Ok(batch))
                }
            }
            Err(err) => Some(Err(err)),
        }
    }
}

impl RecordBatchReader for BlockingArrowRecordBatchReader {
    fn schema(&self) -> ArrowSchemaRef {
        Arc::clone(&self.schema)
    }
}

#[pyclass(
    name = "ArrowReader",
    module = "pyiceberg_core.scan",
    skip_from_py_object
)]
pub struct PyArrowReader {
    file_io: iceberg::io::FileIO,
    batch_size: Option<usize>,
    data_file_concurrency_limit: Option<usize>,
    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
}

#[pymethods]
impl PyArrowReader {
    #[new]
    #[pyo3(signature = (
        file_io,
        *,
        batch_size = None,
        data_file_concurrency_limit = None,
        row_group_filtering_enabled = true,
        row_selection_enabled = false
    ))]
    fn new(
        file_io: &PyFileIO,
        batch_size: Option<usize>,
        data_file_concurrency_limit: Option<usize>,
        row_group_filtering_enabled: bool,
        row_selection_enabled: bool,
    ) -> Self {
        Self {
            file_io: file_io.inner.clone(),
            batch_size,
            data_file_concurrency_limit,
            row_group_filtering_enabled,
            row_selection_enabled,
        }
    }

    #[pyo3(signature = (output_schema, tasks, *, max_rows = None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        output_schema: &PySchema,
        tasks: &Bound<'_, PyAny>,
        max_rows: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_tasks = py_tasks_to_rust(tasks)?;
        validate_reader_projection(output_schema, &rust_tasks)?;
        let schema = if let Some(first_task) = rust_tasks.first() {
            RecordBatchTransformer::arrow_schema_for_task(first_task).map_err(crate::error::to_py_err)?
        } else {
            Arc::new(
                schema_to_arrow_schema(output_schema.inner.as_ref())
                    .map_err(crate::error::to_py_err)?,
            )
        };
        let task_stream =
            Box::pin(stream::iter(rust_tasks.into_iter().map(Ok))) as FileScanTaskStream;

        let mut builder = ArrowReaderBuilder::new(self.file_io.clone(), iceberg_runtime())
            .with_row_group_filtering_enabled(self.row_group_filtering_enabled)
            .with_row_selection_enabled(self.row_selection_enabled);
        if let Some(batch_size) = self.batch_size {
            builder = builder.with_batch_size(batch_size);
        }
        if let Some(limit) = self.data_file_concurrency_limit {
            builder = builder.with_data_file_concurrency_limit(limit);
        }

        let stream = builder
            .build()
            .read(task_stream)
            .map_err(crate::error::to_py_err)?
            .stream();
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(BlockingArrowRecordBatchReader {
                schema,
                stream,
                remaining_rows: max_rows,
            });

        reader.into_pyarrow(py)
    }
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "scan")?;
    this.add_class::<PyDeleteFile>()?;
    this.add_class::<PyFileScanTask>()?;
    this.add_class::<PyArrowReader>()?;
    m.add_submodule(&this)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pyiceberg_core.scan", this)?;
    Ok(())
}
