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

use bytes::Bytes;
use iceberg::io::{FileIO, FileIOBuilder, InputFile, OutputFile};
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::runtime::runtime;

/// Keys whose values must be redacted in __repr__ to avoid leaking credentials.
fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_lowercase();
    lower.contains("secret")
        || lower.contains("key")
        || lower.contains("token")
        || lower.contains("password")
        || lower.contains("credential")
        || lower.contains("passphrase")
}

#[pyclass(name = "FileIO", module = "pyiceberg_core.file_io", from_py_object)]
#[derive(Clone)]
pub struct PyFileIO {
    inner: FileIO,
    /// A copy of the original props used at construction, for __repr__.
    props: HashMap<String, String>,
}

#[pymethods]
impl PyFileIO {
    /// Construct a `FileIO` handle from a dict of storage properties.
    ///
    /// The property keys are the same ones iceberg-rust's `FileIOBuilder` recognizes
    /// (e.g. `s3.region`, `s3.access-key-id`).  For local-filesystem paths use
    /// `file://…` URIs with an empty dict.
    ///
    /// The `FileIO` instance is lazily initialized on first use and cached, so
    /// constructing once and reusing across many file opens amortizes the setup cost.
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

    /// Check whether a file exists at the given path.
    fn exists(&self, path: String) -> PyResult<bool> {
        runtime()
            .block_on(self.inner.exists(&path))
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Delete the file at the given path.
    fn delete(&self, path: String) -> PyResult<()> {
        runtime()
            .block_on(self.inner.delete(&path))
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Open the file at `path` for reading and return a `InputFile` handle.
    fn new_input(&self, path: String) -> PyResult<PyInputFile> {
        let input = self
            .inner
            .new_input(&path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(PyInputFile { inner: input })
    }

    /// Open the file at `path` for writing and return an `OutputFile` handle.
    fn new_output(&self, path: String) -> PyResult<PyOutputFile> {
        let output = self
            .inner
            .new_output(&path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(PyOutputFile { inner: output })
    }

    fn __repr__(&self) -> String {
        // Build a summary of the props, redacting sensitive values.
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
        pairs.sort(); // deterministic output
        if pairs.is_empty() {
            "FileIO()".to_string()
        } else {
            format!("FileIO({})", pairs.join(", "))
        }
    }
}

/// A handle for reading a single file.
///
/// Obtained via `FileIO.new_input(path)`.
#[pyclass(name = "InputFile", module = "pyiceberg_core.file_io")]
pub struct PyInputFile {
    inner: InputFile,
}

#[pymethods]
impl PyInputFile {
    /// The absolute path this input file was opened on.
    fn location(&self) -> &str {
        self.inner.location()
    }

    /// Return `True` if the file exists in the underlying storage.
    fn exists(&self) -> PyResult<bool> {
        runtime()
            .block_on(self.inner.exists())
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Read the entire file content and return it as `bytes`.
    fn read(&self) -> PyResult<Vec<u8>> {
        let bytes = runtime()
            .block_on(self.inner.read())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(bytes.to_vec())
    }

    /// Return a dict with file metadata.  Currently exposes `size` (bytes).
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let meta = runtime()
            .block_on(self.inner.metadata())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let d = PyDict::new(py);
        d.set_item("size", meta.size)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!("InputFile({:?})", self.inner.location())
    }
}

/// A handle for writing a single file.
///
/// Obtained via `FileIO.new_output(path)`.
#[pyclass(name = "OutputFile", module = "pyiceberg_core.file_io")]
pub struct PyOutputFile {
    inner: OutputFile,
}

#[pymethods]
impl PyOutputFile {
    /// The absolute path this output file was opened on.
    fn location(&self) -> &str {
        self.inner.location()
    }

    /// Write `data` to the file, replacing any existing content.
    fn write(&self, data: &[u8]) -> PyResult<()> {
        let bs = Bytes::copy_from_slice(data);
        runtime()
            .block_on(self.inner.write(bs))
            .map_err(|e| PyIOError::new_err(e.to_string()))
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
        .set_item("pyiceberg_core.file_io", this)?;
    Ok(())
}
