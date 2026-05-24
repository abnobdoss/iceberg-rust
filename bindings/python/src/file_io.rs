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
use std::sync::Arc;

use iceberg::io::{FileIO, FileIOBuilder, InputFile, OutputFile};
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::runtime::runtime;

fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_lowercase();
    lower.contains("secret")
        || lower.contains("key")
        || lower.contains("token")
        || lower.contains("password")
        || lower.contains("credential")
        || lower.contains("passphrase")
}

#[pyclass(name = "FileIO", module = "pyiceberg_core.file_io", skip_from_py_object)]
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
        let file_io = FileIOBuilder::new(factory).with_props(props.clone()).build();
        Ok(PyFileIO {
            inner: file_io,
            props,
        })
    }

    fn exists(&self, py: Python<'_>, path: String) -> PyResult<bool> {
        py.detach(|| {
            runtime()
                .block_on(self.inner.exists(&path))
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })
    }

    fn delete(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| {
            runtime()
                .block_on(self.inner.delete(&path))
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })
    }

    fn new_input(&self, path: String) -> PyResult<PyInputFile> {
        let input = self
            .inner
            .new_input(&path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(PyInputFile { inner: input })
    }

    fn new_output(&self, path: String) -> PyResult<PyOutputFile> {
        let output = self
            .inner
            .new_output(&path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
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
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })
    }

    fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| {
            runtime()
                .block_on(self.inner.read())
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn size(&self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| {
            runtime()
                .block_on(self.inner.metadata())
                .map_err(|e| PyIOError::new_err(e.to_string()))
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
                .map_err(|e| PyIOError::new_err(e.to_string()))
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
