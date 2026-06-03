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

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use iceberg::expr::{Predicate, PredicateOperator, Reference};
use iceberg::spec::Datum;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyAny, PyByteArray, PyBytes, PyDate, PyDateTime, PyFloat, PyInt, PyString, PyTime, PyType,
};

use crate::error::to_py_err;

static DECIMAL_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

fn operator_hash_key(op: PredicateOperator) -> u16 {
    op as u16
}

fn hash_predicate<H: Hasher>(predicate: &Predicate, state: &mut H) {
    match predicate {
        Predicate::AlwaysTrue => {
            0u8.hash(state);
        }
        Predicate::AlwaysFalse => {
            1u8.hash(state);
        }
        Predicate::And(expr) => {
            2u8.hash(state);
            let [left, right] = expr.inputs();
            hash_predicate(left, state);
            hash_predicate(right, state);
        }
        Predicate::Or(expr) => {
            3u8.hash(state);
            let [left, right] = expr.inputs();
            hash_predicate(left, state);
            hash_predicate(right, state);
        }
        Predicate::Not(expr) => {
            4u8.hash(state);
            let [inner] = expr.inputs();
            hash_predicate(inner, state);
        }
        Predicate::Unary(expr) => {
            5u8.hash(state);
            operator_hash_key(expr.op()).hash(state);
            expr.term().name().hash(state);
        }
        Predicate::Binary(expr) => {
            6u8.hash(state);
            operator_hash_key(expr.op()).hash(state);
            expr.term().name().hash(state);
            expr.literal().hash(state);
        }
        Predicate::Set(expr) => {
            7u8.hash(state);
            operator_hash_key(expr.op()).hash(state);
            expr.term().name().hash(state);

            let mut literal_hashes = expr
                .literals()
                .iter()
                .map(|literal| {
                    let mut hasher = DefaultHasher::new();
                    literal.hash(&mut hasher);
                    hasher.finish()
                })
                .collect::<Vec<_>>();
            literal_hashes.sort_unstable();
            literal_hashes.hash(state);
        }
    }
}

fn predicate_hash(predicate: &Predicate) -> isize {
    let mut hasher = DefaultHasher::new();
    hash_predicate(predicate, &mut hasher);
    let hash = hasher.finish() as isize;
    if hash == -1 { -2 } else { hash }
}

fn py_to_datum(value: &Bound<'_, PyAny>) -> PyResult<Datum> {
    // bool before int: in Python `True` IS-A `int` (extract::<i64>(True) == 1).
    if let Ok(v) = value.extract::<bool>() {
        return Ok(Datum::bool(v));
    }

    // PyDateTime before PyDate: datetime IS-A date in Python (downcast succeeds for both).
    if let Ok(dt) = value.cast::<PyDateTime>() {
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
        if !t.getattr("tzinfo")?.is_none() {
            return Err(PyValueError::new_err(
                "timezone-aware datetime.time values are not supported by Iceberg time",
            ));
        }
        let iso: String = t.call_method0("isoformat")?.extract()?;
        return Datum::time_from_str(&iso).map_err(to_py_err);
    }

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
        // Python float maps to f64; iceberg-rust does not yet narrow Double -> Float.
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

fn py_iterable_to_datum_vec(values: &Bound<'_, PyAny>) -> PyResult<Vec<Datum>> {
    // Reject string-like byte containers explicitly: they are iterable in Python,
    // but treating them as collections of characters/integers is surprising here.
    if values.is_instance_of::<PyString>()
        || values.is_instance_of::<PyBytes>()
        || values.is_instance_of::<PyByteArray>()
    {
        return Err(PyTypeError::new_err(
            "is_in / is_not_in expects an iterable of values, not a string or bytes-like object",
        ));
    }
    let iter = values.try_iter().map_err(|_| {
        PyTypeError::new_err("is_in / is_not_in expects an iterable of values")
    })?;
    let mut out = Vec::new();
    for item in iter {
        let item = item?;
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
    #[staticmethod]
    fn always_true() -> Self {
        Self {
            inner: Predicate::AlwaysTrue,
        }
    }

    #[staticmethod]
    fn always_false() -> Self {
        Self {
            inner: Predicate::AlwaysFalse,
        }
    }

    fn and_(&self, other: &PyPredicate) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().and(other.inner.clone()),
        }
    }

    fn or_(&self, other: &PyPredicate) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().or(other.inner.clone()),
        }
    }

    fn negate(&self) -> PyPredicate {
        PyPredicate {
            // iceberg-rust's Not impl applies De Morgan rewriting.
            inner: !self.inner.clone(),
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

    fn __eq__(&self, other: &PyPredicate) -> bool {
        self.inner == other.inner
    }

    fn __ne__(&self, other: &PyPredicate) -> bool {
        self.inner != other.inner
    }

    fn __hash__(&self) -> isize {
        predicate_hash(&self.inner)
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

    fn is_in(&self, values: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let set = py_iterable_to_datum_vec(values)?;
        Ok(PyPredicate {
            inner: self.inner.clone().is_in(set),
        })
    }

    fn is_not_in(&self, values: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let set = py_iterable_to_datum_vec(values)?;
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
