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

use iceberg::expr::Bind;
use iceberg::scan::{FileScanTask, FileScanTaskDeleteFile};
use iceberg::spec::{DataContentType, DataFileFormat};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};

use crate::expression::PyPredicate;
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
        case_sensitive: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FileScanTask {
                file_size_in_bytes,
                start,
                length: length.unwrap_or(file_size_in_bytes),
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
                partition: None,
                partition_spec: None,
                name_mapping: None,
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

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "scan")?;
    this.add_class::<PyDeleteFile>()?;
    this.add_class::<PyFileScanTask>()?;
    m.add_submodule(&this)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pyiceberg_core.scan", this)?;
    Ok(())
}
