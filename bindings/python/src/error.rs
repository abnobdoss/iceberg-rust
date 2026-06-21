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

use pyo3::PyErr;
use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyValueError};

/// Convert an iceberg error to a python error
pub fn to_py_err(err: iceberg::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Convert an iceberg error from an I/O operation into the closest Python exception.
///
/// Unlike `to_py_err` (which maps everything to ValueError), I/O failures default to
/// OSError. iceberg-storage-opendal flattens every opendal error to
/// `ErrorKind::Unexpected` (crates/storage/opendal/src/utils.rs), so `err.kind()` on the
/// I/O path carries no NotFound/PermissionDenied granularity; we map only the kinds the
/// storage-resolution layer sets before opendal runs, and route the rest to OSError.
/// Finer-grained typing (FileNotFoundError, PermissionError) must wait for an upstream
/// `from_opendal_error` that preserves `ErrorKind`.
pub fn to_py_io_err(err: iceberg::Error) -> PyErr {
    let message = err.to_string();
    match err.kind() {
        iceberg::ErrorKind::DataInvalid => PyValueError::new_err(message),
        iceberg::ErrorKind::FeatureUnsupported => PyNotImplementedError::new_err(message),
        _ => PyOSError::new_err(message),
    }
}
