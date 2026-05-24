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
use iceberg::spec::Schema;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

use crate::error::to_py_err;

/// The PyCapsule name used when handing `Arc<Schema>` to sibling Rust bindings.
///
/// Consumers (e.g. PyFileScanTask, Predicate.bind) must validate this name before
/// dereferencing the pointer. The name is `b"iceberg_core_schema\0"`.
pub const SCHEMA_CAPSULE_NAME: &std::ffi::CStr = c"iceberg_core_schema";

/// Opaque handle around an `iceberg::spec::Schema`.
///
/// Parse once from spec JSON via `Schema.from_json()`; reuse across all downstream
/// building blocks (Predicate bind, FileScanTask construction, ArrowReader). Internally
/// holds an `Arc<Schema>` so clone is cheap and the allocation is shared.
#[pyclass(name = "Schema", module = "pyiceberg_core.schema", from_py_object)]
#[derive(Clone)]
pub struct PySchema {
    pub(crate) inner: Arc<Schema>,
}

#[pymethods]
impl PySchema {
    /// Parse an Iceberg spec-JSON string (V1 or V2) into an opaque Schema handle.
    ///
    /// Raises `ValueError` if the JSON is malformed or violates spec invariants
    /// (e.g. duplicate field IDs, float/double identifier fields).
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<PySchema> {
        let schema: Schema = serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse schema JSON: {e}")))?;
        Ok(PySchema {
            inner: Arc::new(schema),
        })
    }

    /// The schema-id as recorded in the spec JSON. Default is 0 for V1 schemas
    /// that omit it.
    fn schema_id(&self) -> i32 {
        self.inner.schema_id()
    }

    /// The highest field ID among all fields at every nesting level.
    fn highest_field_id(&self) -> i32 {
        self.inner.highest_field_id()
    }

    /// Names of top-level fields only (not dotted-path names for nested fields).
    fn column_names(&self) -> Vec<String> {
        self.inner
            .as_struct()
            .fields()
            .iter()
            .map(|f| f.name.clone())
            .collect()
    }

    /// Identifier field IDs (primary-key / equality-delete keys), sorted ascending.
    fn identifier_field_ids(&self) -> Vec<i32> {
        let mut ids: Vec<i32> = self.inner.identifier_field_ids().collect();
        ids.sort_unstable();
        ids
    }

    /// Look up a field by its full dotted-path name (case-sensitive).
    ///
    /// Returns a dict with keys `id`, `name`, `type`, `required`, or `None` if not found.
    /// The `type` value is the spec-JSON representation of the field type (e.g. `"int"`,
    /// `{"type":"list",...}`).
    fn find_field_by_name(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        let Some(field) = self.inner.field_by_name(name) else {
            return Ok(None);
        };
        let type_str = serde_json::to_string(field.field_type.as_ref())
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize field type: {e}")))?;
        let d = PyDict::new(py);
        d.set_item("id", field.id)?;
        d.set_item("name", &field.name)?;
        d.set_item("type", type_str)?;
        d.set_item("required", field.required)?;
        Ok(Some(d.into_py_any(py)?))
    }

    /// Look up a field by its field ID.
    ///
    /// Returns a dict with keys `id`, `name`, `type`, `required`.
    /// Raises `KeyError` if the field ID is not present in this schema.
    fn field_by_id(&self, py: Python<'_>, field_id: i32) -> PyResult<Py<PyAny>> {
        let field = self
            .inner
            .field_by_id(field_id)
            .ok_or_else(|| PyKeyError::new_err(format!("No field with id {field_id} in schema")))?;
        let type_str = serde_json::to_string(field.field_type.as_ref())
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize field type: {e}")))?;
        let d = PyDict::new(py);
        d.set_item("id", field.id)?;
        d.set_item("name", &field.name)?;
        d.set_item("type", type_str)?;
        d.set_item("required", field.required)?;
        d.into_py_any(py)
    }

    /// Serialize this schema back to spec-JSON (V2 format with `schema-id`).
    ///
    /// Useful for round-trip testing and passing schemas to systems that only accept JSON.
    /// The output is parseable by `Schema.from_json()`.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.as_ref())
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize schema: {e}")))
    }

    /// Export the Iceberg schema as a PyArrow Schema via the Arrow C Data Interface.
    ///
    /// Field IDs are preserved at all nesting levels via `PARQUET:field_id` metadata.
    /// Round-trips losslessly for all Iceberg v2 types. This conversion costs ~94 µs
    /// for a 30-field schema; prefer `_capsule()` for internal Rust→Rust handoff.
    fn to_arrow_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use arrow::pyarrow::ToPyArrow;
        let arrow_schema =
            iceberg::arrow::schema_to_arrow_schema(self.inner.as_ref()).map_err(to_py_err)?;
        arrow_schema.to_pyarrow(py)
    }

    /// Arrow PyCapsule Interface — zero-copy schema export.
    ///
    /// Returns a `PyCapsule` named `"arrow_schema"` wrapping an `FFI_ArrowSchema`.
    /// Any library that recognises the Arrow PyCapsule Interface (PyArrow ≥ 14, Polars,
    /// narwhals, etc.) can import this schema without going through Python objects.
    ///
    /// Field IDs are preserved via `PARQUET:field_id` metadata at all nesting levels.
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let arrow_schema =
            iceberg::arrow::schema_to_arrow_schema(self.inner.as_ref()).map_err(to_py_err)?;
        let c_schema = FFI_ArrowSchema::try_from(&arrow_schema)
            .map_err(|e| PyValueError::new_err(format!("Arrow FFI export failed: {e}")))?;
        let capsule_name = c"arrow_schema".to_owned();
        PyCapsule::new(py, c_schema, Some(capsule_name))
    }

    /// Return a `PyCapsule` wrapping `Arc<Schema>` for direct Rust→Rust handoff.
    ///
    /// The capsule name is `b"iceberg_core_schema\0"`. Sibling bindings must validate
    /// this name via `SCHEMA_CAPSULE_NAME` before dereferencing. Usage:
    ///
    ///   ```python
    ///   handle = Schema.from_json(json_str)
    ///   cap = handle._capsule()
    ///   # pass cap to e.g. a future Predicate.bind_with_schema(predicate, cap)
    ///   ```
    fn _capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let capsule_name = SCHEMA_CAPSULE_NAME.to_owned();
        let arc_clone = self.inner.clone();
        PyCapsule::new(py, arc_clone, Some(capsule_name))
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
