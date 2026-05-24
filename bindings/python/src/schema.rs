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

use arrow::ffi::FFI_ArrowSchema;
use iceberg::spec::{NestedField, Schema};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

use crate::error::to_py_err;

pub const SCHEMA_CAPSULE_NAME: &std::ffi::CStr = c"iceberg_core_schema";

#[pyclass(name = "Schema", module = "pyiceberg_core.schema", from_py_object)]
#[derive(Clone)]
pub struct PySchema {
    pub(crate) inner: Arc<Schema>,
}

fn field_to_py(py: Python<'_>, field: &NestedField) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("id", field.id)?;
    d.set_item("name", &field.name)?;
    d.set_item(
        "type",
        serde_json::to_string(field.field_type.as_ref())
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize field type: {e}")))?,
    )?;
    d.set_item("required", field.required)?;
    d.into_py_any(py)
}

#[pymethods]
impl PySchema {
    /// Parse Iceberg schema JSON into an opaque Schema handle.
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<PySchema> {
        let schema: Schema = serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse schema JSON: {e}")))?;
        Ok(PySchema {
            inner: Arc::new(schema),
        })
    }

    fn schema_id(&self) -> i32 {
        self.inner.schema_id()
    }

    fn highest_field_id(&self) -> i32 {
        self.inner.highest_field_id()
    }

    fn column_names(&self) -> Vec<String> {
        self.inner
            .as_struct()
            .fields()
            .iter()
            .map(|f| f.name.clone())
            .collect()
    }

    fn identifier_field_ids(&self) -> Vec<i32> {
        let mut ids: Vec<i32> = self.inner.identifier_field_ids().collect();
        ids.sort_unstable();
        ids
    }

    fn find_field_by_name(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .field_by_name(name)
            .map(|field| field_to_py(py, field))
            .transpose()
    }

    fn field_by_id(&self, py: Python<'_>, field_id: i32) -> PyResult<Py<PyAny>> {
        let field = self
            .inner
            .field_by_id(field_id)
            .ok_or_else(|| PyKeyError::new_err(format!("No field with id {field_id} in schema")))?;
        field_to_py(py, field)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.as_ref())
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize schema: {e}")))
    }

    fn to_arrow_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use arrow::pyarrow::ToPyArrow;
        let arrow_schema =
            iceberg::arrow::schema_to_arrow_schema(self.inner.as_ref()).map_err(to_py_err)?;
        arrow_schema.to_pyarrow(py)
    }

    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let arrow_schema =
            iceberg::arrow::schema_to_arrow_schema(self.inner.as_ref()).map_err(to_py_err)?;
        let c_schema = FFI_ArrowSchema::try_from(&arrow_schema)
            .map_err(|e| PyValueError::new_err(format!("Arrow FFI export failed: {e}")))?;
        let capsule_name = c"arrow_schema".to_owned();
        PyCapsule::new(py, c_schema, Some(capsule_name))
    }

    fn _capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let capsule_name = SCHEMA_CAPSULE_NAME.to_owned();
        PyCapsule::new(py, self.inner.clone(), Some(capsule_name))
    }

    fn __repr__(&self) -> String {
        let top_names: Vec<&str> = self
            .inner
            .as_struct()
            .fields()
            .iter()
            .take(4)
            .map(|f| f.name.as_str())
            .collect();
        let total = self.inner.as_struct().fields().len();
        let preview = if total > 4 {
            format!("[{}, ...]", top_names.join(", "))
        } else {
            format!("[{}]", top_names.join(", "))
        };
        format!(
            "Schema(schema_id={}, fields={}, columns={})",
            self.inner.schema_id(),
            total,
            preview
        )
    }
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "schema")?;
    this.add_class::<PySchema>()?;
    m.add_submodule(&this)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pyiceberg_core.schema", this)?;
    Ok(())
}
