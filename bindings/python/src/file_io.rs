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

use std::collections::HashMap;
use std::error::Error as _;
use std::sync::Arc;

use iceberg::ErrorKind;
use iceberg::io::{FileIO, FileIOBuilder, InputFile, OutputFile};
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use opendal::{Error as OpenDalError, ErrorKind as OpenDalErrorKind};
use pyo3::exceptions::{
    PyFileNotFoundError, PyIOError, PyNotImplementedError, PyPermissionError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::runtime::runtime;

fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    let normalized = lower.replace(['-', '_'], ".");
    let parts: Vec<&str> = normalized
        .split('.')
        .filter(|part| !part.is_empty())
        .collect();

    if parts.iter().any(|part| {
        matches!(
            *part,
            "secret" | "token" | "password" | "credential" | "credentials" | "passphrase"
        )
    }) {
        return true;
    }

    let dotted = parts.join(".");
    parts.last().is_some_and(|part| *part == "key")
        || [
            "api.key",
            "access.key",
            "access.key.id",
            "private.key",
            "secret.key",
            "secret.access.key",
        ]
        .iter()
        .any(|suffix| has_dotted_suffix(&dotted, suffix))
}

fn has_dotted_suffix(value: &str, suffix: &str) -> bool {
    value == suffix
        || value
            .strip_suffix(suffix)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

fn py_err_from_opendal_source(err: &iceberg::Error, message: &str) -> Option<PyErr> {
    let source = err.source()?;
    let opendal_err = source.downcast_ref::<OpenDalError>()?;

    match opendal_err.kind() {
        OpenDalErrorKind::NotFound => Some(PyFileNotFoundError::new_err(message.to_string())),
        OpenDalErrorKind::PermissionDenied => Some(PyPermissionError::new_err(message.to_string())),
        OpenDalErrorKind::ConfigInvalid => Some(PyValueError::new_err(message.to_string())),
        OpenDalErrorKind::Unsupported => Some(PyNotImplementedError::new_err(message.to_string())),
        _ => None,
    }
}

fn to_py_io_err(err: iceberg::Error) -> PyErr {
    let message = err.to_string();

    if let Some(py_err) = py_err_from_opendal_source(&err, &message) {
        return py_err;
    }

    match err.kind() {
        ErrorKind::DataInvalid | ErrorKind::PreconditionFailed => PyValueError::new_err(message),
        ErrorKind::FeatureUnsupported => PyNotImplementedError::new_err(message),
        ErrorKind::TableNotFound | ErrorKind::NamespaceNotFound => {
            PyFileNotFoundError::new_err(message)
        }
        _ => PyIOError::new_err(message),
    }
}

#[pyclass(
    name = "FileIO",
    module = "pyiceberg_core.file_io",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFileIO {
    inner: FileIO,
    props: HashMap<String, String>,
}

#[pymethods]
impl PyFileIO {
    #[staticmethod]
    fn from_props(props: HashMap<String, String>) -> PyResult<PyFileIO> {
        let factory = Arc::new(OpenDalResolvingStorageFactory::new());
        let file_io = FileIOBuilder::new(factory)
            .with_props(props.clone())
            .build();
        Ok(PyFileIO {
            inner: file_io,
            props,
        })
    }

    fn exists(&self, py: Python<'_>, path: String) -> PyResult<bool> {
        py.detach(|| {
            runtime()
                .block_on(self.inner.exists(&path))
                .map_err(to_py_io_err)
        })
    }

    fn delete(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| {
            runtime()
                .block_on(self.inner.delete(&path))
                .map_err(to_py_io_err)
        })
    }

    fn new_input(&self, path: String) -> PyResult<PyInputFile> {
        let input = self.inner.new_input(&path).map_err(to_py_io_err)?;
        Ok(PyInputFile { inner: input })
    }

    fn new_output(&self, path: String) -> PyResult<PyOutputFile> {
        let output = self.inner.new_output(&path).map_err(to_py_io_err)?;
        Ok(PyOutputFile { inner: output })
    }

    fn __repr__(&self) -> String {
        let mut pairs: Vec<String> = self
            .props
            .iter()
            .map(|(k, v)| {
                let display = if is_sensitive_key(k) {
                    "<redacted>".to_string()
                } else {
                    v.clone()
                };
                format!("{k}={display}")
            })
            .collect();
        pairs.sort();
        if pairs.is_empty() {
            "FileIO()".to_string()
        } else {
            format!("FileIO({})", pairs.join(", "))
        }
    }
}

#[pyclass(name = "InputFile", module = "pyiceberg_core.file_io")]
pub struct PyInputFile {
    inner: InputFile,
}

#[pymethods]
impl PyInputFile {
    fn location(&self) -> &str {
        self.inner.location()
    }

    fn exists(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| {
            runtime()
                .block_on(self.inner.exists())
                .map_err(to_py_io_err)
        })
    }

    fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| runtime().block_on(self.inner.read()).map_err(to_py_io_err))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn size(&self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| {
            runtime()
                .block_on(self.inner.metadata())
                .map_err(to_py_io_err)
                .map(|meta| meta.size)
        })
    }

    fn __repr__(&self) -> String {
        format!("InputFile({:?})", self.inner.location())
    }
}

#[pyclass(name = "OutputFile", module = "pyiceberg_core.file_io")]
pub struct PyOutputFile {
    inner: OutputFile,
}

#[pymethods]
impl PyOutputFile {
    fn location(&self) -> &str {
        self.inner.location()
    }

    fn write(&self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        let data = data.to_vec();
        py.detach(|| {
            runtime()
                .block_on(self.inner.write(data.into()))
                .map_err(to_py_io_err)
        })
    }

    fn __repr__(&self) -> String {
        format!("OutputFile({:?})", self.inner.location())
    }
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "file_io")?;
    this.add_class::<PyFileIO>()?;
    this.add_class::<PyInputFile>()?;
    this.add_class::<PyOutputFile>()?;
    m.add_submodule(&this)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pyiceberg_core.file_io", this)
}
