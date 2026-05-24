# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import math
from datetime import date, datetime, time, timezone
from decimal import Decimal

import pytest
from pyiceberg_core.expression import Predicate, Reference


@pytest.mark.parametrize(
    "value",
    [
        True,
        5,
        1.5,
        "hello",
        date(2026, 5, 24),
        time(10, 30, 0),
        datetime(2026, 5, 24, 10, 30),
        Decimal("3.14"),
    ],
)
def test_eq_accepts_representative_python_values(value):
    # Smoke test for the public construction path:
    # Python value -> Datum -> Reference.eq -> PyPredicate.
    assert repr(Reference("c").eq(value)).startswith("c = ")


@pytest.mark.parametrize(
    "method,expected",
    [
        ("eq", "c = 5"),
        ("ne", "c != 5"),
        ("lt", "c < 5"),
        ("lte", "c <= 5"),
        ("gt", "c > 5"),
        ("gte", "c >= 5"),
    ],
)
def test_comparison_methods_map_to_expected_predicates(method, expected):
    assert repr(getattr(Reference("c"), method)(5)) == expected


def test_reference_name_and_repr():
    ref = Reference("col")
    assert ref.name() == "col"
    assert repr(ref) == 'Reference("col")'


def test_null_predicates():
    assert repr(Reference("c").is_null()) == "c IS NULL"
    assert repr(Reference("c").is_not_null()) == "c IS NOT NULL"


def test_string_prefix_predicates():
    assert repr(Reference("c").starts_with("foo")) == 'c STARTS WITH "foo"'
    assert repr(Reference("c").not_starts_with("foo")) == 'c NOT STARTS WITH "foo"'


@pytest.mark.parametrize("value", [42, 3.14, True, date(2026, 1, 1), None, [1, 2]])
def test_starts_with_rejects_non_string(value):
    with pytest.raises(TypeError, match="starts_with expects a string"):
        Reference("c").starts_with(value)


@pytest.mark.parametrize(
    "method,expected_prefix",
    [("is_in", "c IN ("), ("is_not_in", "c NOT IN (")],
)
def test_set_methods_accept_sequences(method, expected_prefix):
    pred = getattr(Reference("c"), method)([1, 2, 3])
    rendered = repr(pred)
    assert rendered.startswith(expected_prefix)
    for value in ("1", "2", "3"):
        assert value in rendered


@pytest.mark.parametrize("method", ["is_in", "is_not_in"])
@pytest.mark.parametrize("value", ["abc", b"abc"])
def test_set_methods_reject_string_like_sequences(method, value):
    with pytest.raises(TypeError, match="not a string"):
        getattr(Reference("c"), method)(value)


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63), math.inf, -math.inf, math.nan])
def test_i64_and_float_edge_values_construct(value):
    assert repr(Reference("c").eq(value)).startswith("c = ")


@pytest.mark.parametrize("oversize", [2**63, -(2**63) - 1, 2**100, -(2**100)])
def test_int_overflow_outside_i64_raises(oversize):
    with pytest.raises(ValueError, match="exceeds i64 range"):
        Reference("c").eq(oversize)


def test_object_with_float_dunder_does_not_coerce_to_double():
    class IntLike:
        def __float__(self):
            return 5.0

    with pytest.raises(TypeError, match="Cannot convert"):
        Reference("c").eq(IntLike())


@pytest.mark.parametrize("value", [None, [1, 2, 3], {"a": 1}, object()])
def test_unsupported_python_types_raise(value):
    with pytest.raises(TypeError, match="Cannot convert"):
        Reference("c").eq(value)


def test_datetime_tz_aware_marks_as_timestamptz():
    pred = Reference("c").eq(datetime(2026, 5, 24, 10, 30, tzinfo=timezone.utc))
    assert "UTC" in repr(pred)


def test_datetime_naive_is_timestamp_not_timestamptz():
    assert "UTC" not in repr(Reference("c").eq(datetime(2026, 5, 24, 10, 30)))


def test_datetime_microsecond_precision_preserved():
    pred = Reference("c").eq(datetime(2026, 5, 24, 10, 30, 45, 123456))
    assert "123456" in repr(pred)


def test_time_microsecond_precision_preserved():
    pred = Reference("c").eq(time(10, 30, 0, 123456))
    assert "123456" in repr(pred)


@pytest.mark.parametrize(
    "value,expected_lit",
    [
        (Decimal("0"), "0"),
        (Decimal("0.00"), "0.00"),
        (Decimal("1000.0001"), "1000.0001"),
        (Decimal("-3.14"), "-3.14"),
        (Decimal("1E-10"), "1E-10"),
    ],
)
def test_decimal_values_preserve_scale_and_notation(value, expected_lit):
    assert repr(Reference("c").eq(value)) == f"c = {expected_lit}"


def test_always_true_and_false():
    assert repr(Predicate.always_true()) == "TRUE"
    assert repr(Predicate.always_false()) == "FALSE"


@pytest.mark.parametrize("method,operator", [("and_", "&"), ("or_", "|")])
def test_named_composition_matches_operator(method, operator):
    a = Reference("x").eq(1)
    b = Reference("y").lt(10)
    via_method = getattr(a, method)(b)
    via_op = {"&": a & b, "|": a | b}[operator]
    assert repr(via_method) == repr(via_op)


@pytest.mark.parametrize(
    "predicate",
    [
        Reference("c").eq(5),
        Reference("c").lt(5),
        Reference("c").is_null(),
        Reference("c").is_in([1, 2, 3]),
        Reference("c").starts_with("foo"),
    ],
)
def test_negation_applies_to_supported_shapes(predicate):
    assert repr(predicate.negate()) == repr(~predicate)


def test_and_composition_preserves_all_predicates():
    pred = (
        Reference("i").eq(5)
        .and_(Reference("s").eq("x"))
        .and_(Reference("d").eq(date(2026, 1, 1)))
        .and_(Reference("p").eq(Decimal("3.14")))
    )
    rendered = repr(pred)
    assert "i = 5" in rendered
    assert 's = "x"' in rendered
    assert "d = 2026-01-01" in rendered
    assert "p = 3.14" in rendered


def test_large_is_in_constructs_without_error():
    rendered = repr(Reference("c").is_in(list(range(1000))))
    assert rendered.startswith("c IN (")
    for value in ("0", "500", "999"):
        assert value in rendered
