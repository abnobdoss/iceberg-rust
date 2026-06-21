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
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::error::to_py_io_err;
use crate::runtime::runtime;

#[pyclass(name = "FileIO", module = "pyiceberg_core.file_io")]
pub struct PyFileIO {
    inner: FileIO,
    // Only the sorted property keys, kept for __repr__. We deliberately do not retain
    // the values: they can be secrets (access keys, tokens) and the repr never needs them.
    prop_keys: Vec<String>,
}

// FileIO exposes a synchronous Python API: each I/O method blocks on the shared Tokio
// runtime while the GIL is released (py.detach). It is meant to be called from ordinary
// Python threads; calling it from inside a running Tokio task (re-entrant block_on) is
// unsupported and may panic — the same runtime contract the other bindings rely on.
#[pymethods]
impl PyFileIO {
    #[staticmethod]
    fn from_props(props: HashMap<String, String>) -> PyFileIO {
        let mut prop_keys: Vec<String> = props.keys().cloned().collect();
        prop_keys.sort_unstable();
        let factory = Arc::new(OpenDalResolvingStorageFactory::new());
        let inner = FileIOBuilder::new(factory).with_props(props).build();
        PyFileIO { inner, prop_keys }
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
        if self.prop_keys.is_empty() {
            "FileIO()".to_string()
        } else {
            // Quote keys so an unusual key (comma/space) stays unambiguous.
            let keys: Vec<String> = self.prop_keys.iter().map(|k| format!("{k:?}")).collect();
            format!("FileIO(keys=[{}])", keys.join(", "))
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
