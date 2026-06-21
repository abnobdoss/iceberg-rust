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

use arrow::ffi::FFI_ArrowSchema;
use iceberg::spec::{NestedField, Schema};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyString};

use crate::error::to_py_err;

#[pyclass(name = "Schema", module = "pyiceberg_core.schema")]
pub struct PySchema {
    inner: Schema,
}

/// Convert a serde_json value into native Python objects, so callers receive a
/// str/dict/list rather than a JSON-encoded string.
fn json_value_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value;
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => (*b).into_py_any(py),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py_any(py)
            } else if let Some(u) = n.as_u64() {
                u.into_py_any(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py_any(py)
            } else {
                Err(PyValueError::new_err(format!(
                    "JSON number is not representable as a Python int or float: {n}"
                )))
            }
        }
        Value::String(s) => s.as_str().into_py_any(py),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_value_to_py(py, item)?)?;
            }
            list.into_py_any(py)
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, val) in map {
                dict.set_item(key, json_value_to_py(py, val)?)?;
            }
            dict.into_py_any(py)
        }
    }
}

fn field_to_py<'py>(py: Python<'py>, field: &NestedField) -> PyResult<Bound<'py, PyDict>> {
    // Public API contract: the returned dict IS the field's Iceberg JSON object as native
    // Python values — keys id/name/type/required plus optional doc/initial-default/
    // write-default; nested "type" is a str for primitives or a dict for complex types.
    // Round-tripping through serde keeps it identical to Iceberg's on-disk field JSON.
    let value = serde_json::to_value(field)
        .map_err(|e| PyValueError::new_err(format!("Failed to serialize field: {e}")))?;
    let serde_json::Value::Object(map) = value else {
        return Err(PyValueError::new_err(
            "field did not serialize to a JSON object",
        ));
    };
    let dict = PyDict::new(py);
    for (key, val) in &map {
        dict.set_item(key, json_value_to_py(py, val)?)?;
    }
    Ok(dict)
}

#[pymethods]
impl PySchema {
    /// Parse Iceberg schema JSON into a Schema handle.
    ///
    /// Validation policy (intentionally thin; do NOT expand into a second parser):
    /// we do cheap top-level shape checks here, then delegate ALL real validation to
    /// iceberg core's serde + SchemaBuilder. Core already enforces every schema rule
    /// and emits actionable messages we forward verbatim under the "Invalid Iceberg
    /// schema:" prefix -- e.g. "Found duplicate 'field.id' 1" and every
    /// identifier-field-id rule (nonexistent / optional / float-double / non-primitive
    /// / nested-in-optional). Re-checking any of those here would duplicate core and
    /// risk drifting from it.
    ///
    /// One case core reports poorly: a well-formed JSON whose *shape* matches neither
    /// schema variant (unknown type name, non-string field name) collapses to serde's
    /// generic "data did not match any variant of untagged enum SchemaEnum", because
    /// the core Schema is an untagged V1/V2 enum. Improving that text requires
    /// per-field parsing == reimplementing core, so we leave it; fixing it belongs
    /// upstream in core's deserializer, not in this binding.
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<PySchema> {
        // Cheap guard for the few top-level mistakes that are *required by both* schema
        // variants -- so these checks can never reject a valid schema -- giving an
        // actionable message instead of the opaque untagged-enum text. This is the
        // deliberate ceiling: per-field / per-type / identifier validation stays in core.
        if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(s) {
            let structural_error =
                if !matches!(map.get("fields"), Some(serde_json::Value::Array(_))) {
                    Some("'fields' must be a list")
                } else if map
                    .get("schema-id")
                    .is_some_and(|v| !v.is_i64() && !v.is_u64())
                {
                    Some("'schema-id' must be an integer")
                } else if map
                    .get("identifier-field-ids")
                    .is_some_and(|v| !v.is_array())
                {
                    Some("'identifier-field-ids' must be a list")
                } else {
                    None
                };
            if let Some(message) = structural_error {
                return Err(PyValueError::new_err(format!(
                    "Invalid Iceberg schema: {message}"
                )));
            }
        }
        let schema: Schema = serde_json::from_str(s).map_err(|e| {
            use serde_json::error::Category;
            let message = match e.classify() {
                Category::Syntax | Category::Eof => format!("Invalid JSON: {e}"),
                // Data = well-formed JSON that breaks an Iceberg schema rule; Io is
                // unreachable for from_str but bucketed here for an exhaustive match.
                Category::Data | Category::Io => format!("Invalid Iceberg schema: {e}"),
            };
            PyValueError::new_err(message)
        })?;
        Ok(PySchema { inner: schema })
    }

    #[getter]
    fn schema_id(&self) -> i32 {
        self.inner.schema_id()
    }

    #[getter]
    fn highest_field_id(&self) -> i32 {
        self.inner.highest_field_id()
    }

    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.inner
            .as_struct()
            .fields()
            .iter()
            .map(|f| f.name.clone())
            .collect()
    }

    /// Identifier field IDs in ascending order.
    #[getter]
    fn identifier_field_ids(&self) -> Vec<i32> {
        let mut ids: Vec<i32> = self.inner.identifier_field_ids().collect();
        ids.sort_unstable();
        ids
    }

    /// Return field metadata for a case-sensitive field name.
    ///
    /// Accepts either a full dotted name (`parent.child`) or a registered short
    /// name. The returned dict has native Python values for every field attribute
    /// (id, name, required, type, and optional doc/defaults); "type" is a str for
    /// primitives or a nested dict for complex types.
    fn find_field_by_name<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .field_by_name(name)
            .map(|field| field_to_py(py, field))
            .transpose()
    }

    /// Return field metadata for a field ID (see `find_field_by_name` for the dict shape).
    fn field_by_id<'py>(&self, py: Python<'py>, field_id: i32) -> PyResult<Bound<'py, PyDict>> {
        let field = self
            .inner
            .field_by_id(field_id)
            .ok_or_else(|| PyKeyError::new_err(format!("No field with id {field_id} in schema")))?;
        field_to_py(py, field)
    }

    /// Serialize the schema back to Iceberg JSON.
    ///
    /// Note: `identifier-field-ids` order follows the core serializer and is not
    /// guaranteed sorted; use the `identifier_field_ids` property for a sorted view.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize schema: {e}")))
    }

    fn to_arrow_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use arrow::pyarrow::ToPyArrow;
        let arrow_schema =
            iceberg::arrow::schema_to_arrow_schema(&self.inner).map_err(to_py_err)?;
        arrow_schema.to_pyarrow(py)
    }

    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let arrow_schema =
            iceberg::arrow::schema_to_arrow_schema(&self.inner).map_err(to_py_err)?;
        let c_schema = FFI_ArrowSchema::try_from(&arrow_schema)
            .map_err(|e| PyValueError::new_err(format!("Arrow FFI export failed: {e}")))?;
        let capsule_name = c"arrow_schema".to_owned();
        PyCapsule::new(py, c_schema, Some(capsule_name))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Preview the first few column names, quoted via Python's own repr (so quoting
        // matches Python and a name with a comma/space stays unambiguous); longer schemas
        // are truncated with an ellipsis.
        const PREVIEW_LIMIT: usize = 4;
        let fields = self.inner.as_struct().fields();
        let mut preview: Vec<String> = Vec::new();
        for field in fields.iter().take(PREVIEW_LIMIT) {
            preview.push(PyString::new(py, field.name.as_str()).repr()?.to_string());
        }
        let columns = if fields.len() > PREVIEW_LIMIT {
            format!("[{}, ...]", preview.join(", "))
        } else {
            format!("[{}]", preview.join(", "))
        };
        Ok(format!(
            "Schema(schema_id={}, fields={}, columns={})",
            self.inner.schema_id(),
            fields.len(),
            columns
        ))
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
