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
    PyAny, PyByteArray, PyBytes, PyDate, PyDateTime, PyDict, PyFloat, PyInt, PyMemoryView,
    PyString, PyTime, PyType,
};

use crate::error::to_py_err;

static DECIMAL_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static UUID_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Convert a Python value into an iceberg `Datum`.
///
/// Dispatch is grouped temporal-then-scalar. Within each group the arm order
/// encodes correctness invariants: `datetime` before `date` (datetime subclasses
/// date), `bool` before `int` (bool subclasses int), and `int` before `float`
/// (a large int must not fall through to the double arm).
fn py_to_datum(value: &Bound<'_, PyAny>) -> PyResult<Datum> {
    // datetime before date: datetime subclasses date (cast::<PyDate> matches both).
    if let Ok(dt) = value.cast::<PyDateTime>() {
        // utcoffset() (not tzinfo) decides timestamp vs timestamptz: Python treats a
        // datetime as aware only when utcoffset() is not None, so a tzinfo whose
        // utcoffset() returns None is naive. iceberg-rust's bind-time Datum::to() does
        // NOT cross-convert Timestamp <-> Timestamptz, so unlike pyiceberg (which
        // defers via a single TimestampLiteral) we must pick the correct type here.
        let iso: String = dt.call_method0("isoformat")?.extract()?;
        let has_tz = !dt.call_method0("utcoffset")?.is_none();
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

    // --- Scalar arms ---

    // bool before int: Python's bool is a subclass of int, so True would otherwise extract as 1.
    if let Ok(v) = value.extract::<bool>() {
        return Ok(Datum::bool(v));
    }

    // Python int -> i64 long. iceberg-rust's integer literals are i64-bounded, so a value
    // beyond i64 raises a clear out-of-range ValueError. (pyiceberg carries unbounded ints
    // and resolves them to an above/below-max sentinel at filter time; matching that needs
    // a core change, so >i64 is the one divergence here.)
    if value.is_instance_of::<PyInt>() {
        return match value.extract::<i64>() {
            Ok(v) => Ok(Datum::long(v)),
            // Do not format the value itself: an int subclass's __str__ could run user
            // code and raise, masking the range error. The constraint alone is actionable.
            Err(_) => Err(PyValueError::new_err(
                "integer is out of range; iceberg-rust requires a value in [-2^63, 2^63 - 1]",
            )),
        };
    }
    if value.is_instance_of::<PyFloat>() {
        let v = value.extract::<f64>()?;
        // Always f64. Note: iceberg-rust's Datum::to() does not currently narrow
        // Double -> Float, so this Datum can only bind against Double columns.
        // Filters against Float columns will error at bind time until iceberg-rust
        // adds the narrowing arm.
        return Ok(Datum::double(v));
    }

    // isinstance (not a name check): subclasses pass, foreign "Decimal" types don't.
    // decimal_from_str yields a precision-38 literal; bind-time Datum::to() re-narrows to the
    // column's precision when scales match, and errors on scale mismatch (no rescaling).
    let decimal_class = DECIMAL_TYPE.import(value.py(), "decimal", "Decimal")?;
    if value.is_instance(decimal_class)? {
        // Reject non-finite decimals: decimal_from_str would otherwise silently coerce
        // NaN/Infinity to a wrong finite literal (NaN -> 0, Infinity -> -1).
        if !value.call_method0("is_finite")?.extract::<bool>()? {
            return Err(PyValueError::new_err(
                "Cannot convert a non-finite Decimal (NaN or Infinity) to an Iceberg decimal",
            ));
        }
        let s: String = value.str()?.extract()?;
        return Datum::decimal_from_str(&s).map_err(to_py_err);
    }

    // uuid.UUID -> Iceberg uuid (isinstance so subclasses are accepted).
    let uuid_class = UUID_TYPE.import(value.py(), "uuid", "UUID")?;
    if value.is_instance(uuid_class)? {
        let s: String = value.str()?.extract()?;
        return Datum::uuid_from_str(&s).map_err(to_py_err);
    }

    if let Ok(v) = value.extract::<String>() {
        return Ok(Datum::string(v));
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(Datum::binary(bytes.as_bytes().to_vec()));
    }
    // Name the offending type via get_type().name() — never value.repr(), which would
    // run a user-defined __repr__ that could raise and mask the real error.
    let type_name = value.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "Cannot convert value of type '{type_name}' to an Iceberg Datum; supported types: \
         bool, int (within i64 range), float, str, bytes, date, time, datetime, \
         decimal.Decimal, uuid.UUID"
    )))
}

/// Extract a Python `str`, or raise a TypeError that names the method and the offending type.
fn extract_string(value: &Bound<'_, PyAny>, method: &str) -> PyResult<String> {
    value.extract().map_err(|_| match value.get_type().name() {
        Ok(type_name) => {
            PyTypeError::new_err(format!("{method} expects a string, got '{type_name}'"))
        }
        Err(err) => err,
    })
}

/// Convert any Python iterable into a `Vec<Datum>` for `is_in` / `is_not_in`.
/// `Reference::is_in` accepts any `IntoIterator<Item = Datum>` and builds the
/// internal FnvHashSet itself, so the binding does not need an `fnv` dependency.
fn py_iter_to_datum_vec(values: &Bound<'_, PyAny>, method: &str) -> PyResult<Vec<Datum>> {
    // Reject str, the byte buffers (bytes/bytearray/memoryview) and dict: all are iterable
    // but iterate in a surprising way (characters, integer byte values, or dict keys) —
    // almost never what the caller meant for a set of literal values.
    if values.is_instance_of::<PyString>()
        || values.is_instance_of::<PyBytes>()
        || values.is_instance_of::<PyByteArray>()
        || values.is_instance_of::<PyMemoryView>()
        || values.is_instance_of::<PyDict>()
    {
        let type_name = values.get_type().name()?;
        return Err(PyTypeError::new_err(format!(
            "{method} expects an iterable of values (list, set, tuple, etc.), not '{type_name}'"
        )));
    }
    // Propagate any per-element conversion error unchanged, preserving its exact type and
    // cause rather than reconstructing it.
    let mut out = Vec::new();
    for item in values.try_iter()? {
        out.push(py_to_datum(&item?)?);
    }
    Ok(out)
}

#[pyclass(name = "Predicate", module = "pyiceberg_core.expression")]
pub struct PyPredicate {
    inner: Predicate,
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

    /// Logical negation via inverse predicates (De Morgan), producing a canonical
    /// NOT-free form.
    fn negate(&self) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().negate(),
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

    fn __bool__(&self) -> PyResult<bool> {
        // Without this, `pred1 and pred2` would silently short-circuit on Python truthiness
        // (returning one operand) instead of combining predicates. Force the explicit
        // combinators.
        Err(PyTypeError::new_err(
            "Predicate has no truth value; combine predicates with & | ~ (or and_/or_/negate), \
             not Python's and/or/not",
        ))
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pyclass(name = "Reference", module = "pyiceberg_core.expression")]
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

    #[getter]
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

    fn is_nan(&self) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().is_nan(),
        }
    }

    fn is_not_nan(&self) -> PyPredicate {
        PyPredicate {
            inner: self.inner.clone().is_not_nan(),
        }
    }

    /// Constructs a SetExpression with the IN operator.
    ///
    /// Empty and single-element sets are NOT simplified at construction time;
    /// bind-time may simplify against the column schema.
    fn is_in(&self, values: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let set = py_iter_to_datum_vec(values, "is_in")?;
        Ok(PyPredicate {
            inner: self.inner.clone().is_in(set),
        })
    }

    fn is_not_in(&self, values: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let set = py_iter_to_datum_vec(values, "is_not_in")?;
        Ok(PyPredicate {
            inner: self.inner.clone().is_not_in(set),
        })
    }

    fn starts_with(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let s = extract_string(value, "starts_with")?;
        Ok(PyPredicate {
            inner: self.inner.clone().starts_with(Datum::string(s)),
        })
    }

    fn not_starts_with(&self, value: &Bound<'_, PyAny>) -> PyResult<PyPredicate> {
        let s = extract_string(value, "not_starts_with")?;
        Ok(PyPredicate {
            inner: self.inner.clone().not_starts_with(Datum::string(s)),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Use Python's own repr of the name so quoting/escaping matches Python, not Rust.
        let name = PyString::new(py, self.inner.name()).repr()?;
        Ok(format!("Reference({name})"))
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
