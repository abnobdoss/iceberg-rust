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
import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from fractions import Fraction

import pytest
from pyiceberg_core.expression import Predicate, Reference


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, "true"),
        (False, "false"),
        (5, "5"),
        (2**40, str(2**40)),
        (1.5, "1.5"),
        ("hello", '"hello"'),
        (date(2026, 5, 24), "2026-05-24"),
        (time(10, 30, 0), "10:30:00"),
        (time(10, 30, 0, 123456), "10:30:00.123456"),
        (datetime(2026, 5, 24, 10, 30), "2026-05-24 10:30:00"),
        (
            datetime(2026, 5, 24, 10, 30, tzinfo=timezone.utc),
            "2026-05-24 10:30:00 UTC",
        ),
        (Decimal("3.14"), "3.14"),
    ],
)
def test_eq_dispatches_supported_python_values(value, expected):
    assert repr(Reference("c").eq(value)) == f"c = {expected}"


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
def test_binary_operator_methods(method, expected):
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


@pytest.mark.parametrize("value", [42, 3.14, True, date(2026, 1, 1), None, [1, 2], b"foo"])
def test_starts_with_rejects_non_string(value):
    with pytest.raises(TypeError, match="starts_with expects a string"):
        Reference("c").starts_with(value)


@pytest.mark.parametrize(
    "values_factory",
    [
        lambda: [1, 2, 3],
        lambda: (1, 2, 3),
        lambda: {1, 2, 3},
        lambda: range(1, 4),
        lambda: (value for value in [1, 2, 3]),
    ],
)
def test_set_predicates_accept_iterables(values_factory):
    for method, expected_prefix in (("is_in", "c IN ("), ("is_not_in", "c NOT IN (")):
        pred = getattr(Reference("c"), method)(values_factory())
        rendered = repr(pred)
        assert rendered.startswith(expected_prefix)
        for value in ("1", "2", "3"):
            assert value in rendered


@pytest.mark.parametrize("method", ["is_in", "is_not_in"])
def test_set_predicates_accept_empty_iterables(method):
    assert repr(getattr(Reference("c"), method)([])).startswith("c ")


def test_set_predicates_reject_str_and_bytes():
    for method in (Reference("c").is_in, Reference("c").is_not_in):
        with pytest.raises(TypeError, match="not a string"):
            method("abc")
        with pytest.raises(TypeError, match="not a string"):
            method(b"abc")
        with pytest.raises(TypeError, match="not a string"):
            method(bytearray(b"abc"))


@pytest.mark.parametrize("method", ["is_in", "is_not_in"])
def test_set_predicates_reject_non_iterables(method):
    with pytest.raises(TypeError, match="expects an iterable"):
        getattr(Reference("c"), method)(5)


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63), math.inf, -math.inf, math.nan])
def test_numeric_edge_values_construct(value):
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


@pytest.mark.parametrize(
    "value",
    [
        None,
        [1, 2, 3],
        {"a": 1},
        object(),
        uuid.UUID("12345678-1234-5678-1234-567812345678"),
        b"abc",
        bytearray(b"abc"),
        1 + 2j,
        Fraction(1, 3),
    ],
)
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


def test_datetime_offset_normalizes_to_utc():
    pred = Reference("c").eq(
        datetime(2026, 5, 24, 10, 30, tzinfo=timezone(timedelta(hours=5)))
    )
    assert repr(pred) == "c = 2026-05-24 05:30:00 UTC"


def test_tz_aware_time_rejected():
    with pytest.raises(ValueError, match="timezone-aware datetime.time"):
        Reference("c").eq(time(10, 30, tzinfo=timezone.utc))


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
def test_decimal_display_is_preserved(value, expected_lit):
    assert repr(Reference("c").eq(value)) == f"c = {expected_lit}"


def test_always_true_and_false():
    assert repr(Predicate.always_true()) == "TRUE"
    assert repr(Predicate.always_false()) == "FALSE"


def test_predicate_equality_and_hash():
    first = Reference("c").eq(5)
    same = Reference("c").eq(5)
    different = Reference("c").eq(6)

    assert first == same
    assert first != different
    assert hash(first) == hash(same)
    assert {first: "seen"}[same] == "seen"
    assert len({first, same, different}) == 2


def test_set_predicate_equality_and_hash_ignore_input_order():
    first = Reference("c").is_in([1, 2])
    same = Reference("c").is_in([2, 1])

    assert first == same
    assert hash(first) == hash(same)


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


def test_negation_preserves_explicit_not_node():
    # De Morgan rewriting only changes compound predicates; a single binary predicate wraps in NOT.
    assert repr(~Reference("c").eq(5)) == "NOT (c = 5)"


def test_composition_preserves_all_columns():
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
