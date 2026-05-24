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

use iceberg::expr::{Predicate, Reference};
use iceberg::spec::Datum;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyAny, PyBytes, PyDate, PyDateTime, PyFloat, PyInt, PySequence, PyString, PyTime, PyType,
};

use crate::error::to_py_err;

static DECIMAL_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Convert a Python value into an iceberg `Datum`.
///
/// Dispatch mirrors `pyiceberg.expressions.literals.literal()` (isinstance-based).
/// Order is significant — see the comment on each arm.
fn py_to_datum(value: &Bound<'_, PyAny>) -> PyResult<Datum> {
    // bool before int: in Python `True` IS-A `int` (extract::<i64>(True) == 1).
    if let Ok(v) = value.extract::<bool>() {
        return Ok(Datum::bool(v));
    }

    // PyDateTime before PyDate: datetime IS-A date in Python (downcast succeeds for both).
    if let Ok(dt) = value.cast::<PyDateTime>() {
        // tzinfo decides timestamp vs timestamptz at construction.
        // iceberg-rust's bind-time Datum::to() does NOT cross-convert between
        // Timestamp <-> Timestamptz, so unlike pyiceberg (which defers via a single
        // TimestampLiteral) we must pick the correct type at construction time.
        let iso: String = dt.call_method0("isoformat")?.extract()?;
        let has_tz = !dt.getattr("tzinfo")?.is_none();
        return if has_tz {
            Datum::timestamptz_from_str(&iso).map_err(to_py_err)
        } else {
            Datum::timestamp_from_str(&iso).map_err(to_py_err)
        };
    }
    if let Ok(d) = value.cast::<PyDate>() {
        let iso: String = d.call_method0("isoformat")?.extract()?;
        return Datum::date_from_str(&iso).map_err(to_py_err);
    }
    if let Ok(t) = value.cast::<PyTime>() {
        let iso: String = t.call_method0("isoformat")?.extract()?;
        return Datum::time_from_str(&iso).map_err(to_py_err);
    }

    // Decimal: isinstance check against decimal.Decimal so subclasses are accepted
    // and unrelated classes named "Decimal" from other modules are rejected.
    // Note: Datum::decimal_from_str produces a literal at MAX_DECIMAL_PRECISION (38),
    // and iceberg-rust's Datum::to() requires exact precision/scale equality with
    // the column type. So this Datum only binds against decimal(38, S) columns
    // where S matches the literal's scale. Narrower decimal columns will error at
    // bind time until iceberg-rust adds precision normalization.
    let decimal_class = DECIMAL_TYPE.import(value.py(), "decimal", "Decimal")?;
    if value.is_instance(decimal_class)? {
        let s: String = value.str()?.extract()?;
        return Datum::decimal_from_str(&s).map_err(to_py_err);
    }

    // Python int: must check type explicitly, otherwise an int > i64::MAX falls through
    // to the f64 arm and silently becomes a Datum::double, losing precision silently.
    if value.is_instance_of::<PyInt>() {
        return match value.extract::<i64>() {
            Ok(v) => Ok(Datum::long(v)),
            Err(_) => Err(PyValueError::new_err(format!(
                "integer {} exceeds i64 range; iceberg-rust requires [-2^63, 2^63 - 1]",
                value.str()?.extract::<String>()?,
            ))),
        };
    }
    if value.is_instance_of::<PyFloat>() {
        let v = value.extract::<f64>()?;
        // Always f64 (mirrors pyiceberg DoubleLiteral). Note: iceberg-rust's
        // Datum::to() does not currently narrow Double -> Float, so this Datum
        // can only bind against Double columns. Filters against Float columns
        // will error at bind time until iceberg-rust adds the narrowing arm.
        return Ok(Datum::double(v));
    }
    if let Ok(v) = value.extract::<String>() {
        return Ok(Datum::string(v));
    }
    Err(PyTypeError::new_err(format!(
        "Cannot convert Python value to iceberg Datum: {}",
        value.repr()?.to_str()?
    )))
}

/// Convert a Python sequence into a `Vec<Datum>` for `is_in` / `is_not_in`.
/// `Reference::is_in` accepts any `IntoIterator<Item = Datum>` and builds the
/// internal FnvHashSet itself, so the binding does not need an `fnv` dependency.
fn py_seq_to_datum_vec(values: &Bound<'_, PyAny>) -> PyResult<Vec<Datum>> {
    // Reject str and bytes explicitly: both are sequences in Python, so casting
    // to PySequence would succeed and silently iterate over individual characters/bytes.
    if values.is_instance_of::<PyString>() || values.is_instance_of::<PyBytes>() {
        return Err(PyTypeError::new_err(
            "is_in / is_not_in expects a sequence of values (list, tuple, etc.), not a string or bytes",
        ));
    }
    let seq = values.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("is_in / is_not_in expects a sequence of values (list, tuple, etc.)")
    })?;
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let item = seq.get_item(i)?;
        out.push(py_to_datum(&item)?);
    }
    Ok(out)
}

#[pyclass(name = "Predicate", module = "pyiceberg_core.expression", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPredicate {
    pub(crate) inner: Predicate,
}

#[pymethods]
impl PyPredicate {
    /// The trivial predicate that always evaluates to true.
    /// Useful as the identity for `and`-folds over a (possibly empty) list of predicates.
    #[staticmethod]
    fn always_true() -> Self {
        Self {
            inner: Predicate::AlwaysTrue,
        }
    }

    /// The trivial predicate that always evaluates to false.
    /// Useful for empty `is_in` lists or empty `or`-folds.
    #[staticmethod]
    fn always_false() -> Self {
        Self {
            inner: Predicate::AlwaysFalse,
        }
    }

    /// Combine with another predicate using logical AND.
    fn and_(&self, other: &PyPredicate) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().and(other.inner.clone()),
        }
    }

    /// Combine with another predicate using logical OR.
    fn or_(&self, other: &PyPredicate) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().or(other.inner.clone()),
        }
    }

    /// Logical negation. Applies De Morgan's laws via `rewrite_not` so the result is
    /// in a canonical NOT-free form. Note that `negate()` alone does not simplify the
    /// predicate; `rewrite_not` is called here to push NOT inward to the leaves.
    fn negate(&self) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().negate().rewrite_not(),
        }
    }

    fn __and__(&self, other: &PyPredicate) -> PyPredicate {
        self.and_(other)
    }

    fn __or__(&self, other: &PyPredicate) -> PyPredicate {
        self.or_(other)
    }

    fn __invert__(&self) -> PyPredicate {
        self.negate()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pyclass(name = "Reference", module = "pyiceberg_core.expression", skip_from_py_object)]
#[derive(Clone)]
pub struct PyReference {
    inner: Reference,
}

#[pymethods]
impl PyReference {
    #[new]
    fn new(name: String) -> Self {
        Self {
            inner: Reference::new(name),
        }
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn eq(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let datum = py_to_datum(value)?;
        Ok(PyPredicate {
            inner: self.inner.clone().equal_to(datum),
        })
    }

    fn ne(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let datum = py_to_datum(value)?;
        Ok(PyPredicate {
            inner: self.inner.clone().not_equal_to(datum),
        })
    }

    fn lt(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let datum = py_to_datum(value)?;
        Ok(PyPredicate {
            inner: self.inner.clone().less_than(datum),
        })
    }

    fn lte(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let datum = py_to_datum(value)?;
        Ok(PyPredicate {
            inner: self.inner.clone().less_than_or_equal_to(datum),
        })
    }

    fn gt(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let datum = py_to_datum(value)?;
        Ok(PyPredicate {
            inner: self.inner.clone().greater_than(datum),
        })
    }

    fn gte(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let datum = py_to_datum(value)?;
        Ok(PyPredicate {
            inner: self.inner.clone().greater_than_or_equal_to(datum),
        })
    }

    fn is_null(&self) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().is_null(),
        }
    }

    fn is_not_null(&self) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().is_not_null(),
        }
    }

    /// Constructs a SetExpression with the IN operator.
    ///
    /// Empty and single-element sets are NOT simplified at construction time;
    /// bind-time may simplify against the column schema.
    fn is_in(&self, values: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let set = py_seq_to_datum_vec(values)?;
        Ok(PyPredicate {
            inner: self.inner.clone().is_in(set),
        })
    }

    fn is_not_in(&self, values: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let set = py_seq_to_datum_vec(values)?;
        Ok(PyPredicate {
            inner: self.inner.clone().is_not_in(set),
        })
    }

    fn starts_with(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let s: String = value.extract().map_err(|_| {
            PyTypeError::new_err("starts_with expects a string value")
        })?;
        Ok(PyPredicate {
            inner: self.inner.clone().starts_with(Datum::string(s)),
        })
    }

    fn not_starts_with(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let s: String = value.extract().map_err(|_| {
            PyTypeError::new_err("not_starts_with expects a string value")
        })?;
        Ok(PyPredicate {
            inner: self.inner.clone().not_starts_with(Datum::string(s)),
        })
    }

    fn __repr__(&self) -> String {
        format!("Reference({:?})", self.inner.name())
    }
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "expression")?;
    this.add_class::<PyPredicate>()?;
    this.add_class::<PyReference>()?;
    m.add_submodule(&this)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pyiceberg_core.expression", this)?;
    Ok(())
}
