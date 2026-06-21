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
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest
from pyiceberg_core.expression import Predicate, Reference


@pytest.mark.parametrize(
    "value",
    [
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
    # Reachability smoke test: every supported Python type constructs an eq predicate.
    assert repr(Reference("c").eq(value)).startswith("c = ")


def test_bool_dispatches_to_boolean_literal_not_int():
    # bool is an int subclass in Python; py_to_datum must check bool before int
    # so True/False render as boolean literals rather than 1/0.
    assert repr(Reference("c").eq(True)) == "c = true"
    assert repr(Reference("c").eq(False)) == "c = false"
    assert repr(Reference("c").eq(1)) == "c = 1"


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
    assert ref.name == "col"
    assert repr(ref) == "Reference('col')"


def test_null_predicates():
    assert repr(Reference("c").is_null()) == "c IS NULL"
    assert repr(Reference("c").is_not_null()) == "c IS NOT NULL"


def test_nan_predicates():
    assert repr(Reference("c").is_nan()) == "c IS NAN"
    assert repr(Reference("c").is_not_nan()) == "c IS NOT NAN"


def test_string_prefix_predicates():
    assert repr(Reference("c").starts_with("foo")) == 'c STARTS WITH "foo"'
    assert repr(Reference("c").not_starts_with("foo")) == 'c NOT STARTS WITH "foo"'


@pytest.mark.parametrize("value", [42, 3.14, True, date(2026, 1, 1), None, [1, 2]])
def test_starts_with_rejects_non_string(value):
    with pytest.raises(TypeError, match="starts_with expects a string"):
        Reference("c").starts_with(value)


def test_starts_with_error_names_offending_type():
    with pytest.raises(TypeError, match="starts_with expects a string, got 'int'"):
        Reference("c").starts_with(42)


@pytest.mark.parametrize(
    "method,expected_prefix",
    [("is_in", "c IN ("), ("is_not_in", "c NOT IN (")],
)
@pytest.mark.parametrize(
    "make_values",
    [
        lambda: [1, 2, 3],
        lambda: (1, 2, 3),
        lambda: {1, 2, 3},
        lambda: (x for x in (1, 2, 3)),
    ],
    ids=["list", "tuple", "set", "generator"],
)
def test_set_methods_accept_any_iterable(method, expected_prefix, make_values):
    pred = getattr(Reference("c"), method)(make_values())
    rendered = repr(pred)
    assert rendered.startswith(expected_prefix)
    for value in ("1", "2", "3"):
        assert value in rendered


def test_set_method_bad_element_preserves_type_error():
    # A bad element's own error propagates unchanged (exact type + message).
    with pytest.raises(TypeError, match="Cannot convert value of type 'object'"):
        Reference("c").is_in([1, object(), 3])


def test_set_method_preserves_value_error_for_out_of_range_int():
    # An out-of-range int is a ValueError as a scalar (eq); it must stay a ValueError
    # inside is_in, not be flattened to TypeError.
    with pytest.raises(ValueError, match="out of range"):
        Reference("c").is_in([2**63])


@pytest.mark.parametrize("method", ["is_in", "is_not_in"])
@pytest.mark.parametrize(
    "value", ["abc", b"abc", bytearray(b"abc"), memoryview(b"abc"), {1: 2}]
)
def test_set_methods_reject_strings_buffers_and_mappings(method, value):
    # str, byte buffers and dict are iterable but rejected, so they do not silently
    # iterate into characters / integer byte values / dict keys.
    with pytest.raises(TypeError, match="expects an iterable"):
        getattr(Reference("c"), method)(value)


def test_i64_boundary_values_stay_long():
    # The explicit i64-range check keeps boundary ints as long rather than
    # letting them fall through to the float arm and silently become doubles.
    assert repr(Reference("c").eq(2**63 - 1)) == "c = 9223372036854775807"
    assert repr(Reference("c").eq(-(2**63))) == "c = -9223372036854775808"


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_float_special_values_construct(value):
    assert repr(Reference("c").eq(value)).startswith("c = ")


@pytest.mark.parametrize("oversize", [2**63, -(2**63) - 1, 2**100, -(2**100)])
def test_int_overflow_outside_i64_raises(oversize):
    with pytest.raises(ValueError, match="out of range"):
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


def test_unsupported_type_error_does_not_call_user_repr():
    # The TypeError must name the type via type(value).__name__ without invoking
    # the value's __repr__, which could raise and mask the conversion error.
    class Hostile:
        def __repr__(self):
            raise RuntimeError("boom")

    with pytest.raises(TypeError, match="Cannot convert value of type 'Hostile'"):
        Reference("c").eq(Hostile())


def test_datetime_tz_aware_marks_as_timestamptz():
    pred = Reference("c").eq(datetime(2026, 5, 24, 10, 30, tzinfo=timezone.utc))
    assert "UTC" in repr(pred)


def test_datetime_naive_is_timestamp_not_timestamptz():
    assert "UTC" not in repr(Reference("c").eq(datetime(2026, 5, 24, 10, 30)))


def test_datetime_with_none_utcoffset_is_naive():
    # A tzinfo whose utcoffset() returns None is naive per Python semantics, so it
    # must construct a timestamp (not timestamptz). Using `tzinfo is not None`
    # would misclassify it and fail to parse the offset-less ISO string.
    from datetime import tzinfo

    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None

        def tzname(self, dt):
            return None

        def dst(self, dt):
            return None

    pred = Reference("c").eq(datetime(2026, 5, 24, 10, 30, tzinfo=NoOffset()))
    assert "UTC" not in repr(pred)


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


@pytest.mark.parametrize("nonfinite", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_decimal_is_rejected(nonfinite):
    # Non-finite decimals would silently coerce to a wrong finite literal; they must raise.
    with pytest.raises(ValueError, match="non-finite Decimal"):
        Reference("c").eq(Decimal(nonfinite))


@pytest.mark.parametrize("nonfinite", ["NaN", "Infinity"])
def test_non_finite_decimal_is_rejected_inside_is_in(nonfinite):
    with pytest.raises(ValueError, match="non-finite Decimal"):
        Reference("c").is_in([Decimal("1.0"), Decimal(nonfinite)])


def test_bytes_literal_builds_binary_predicate():
    assert repr(Reference("c").eq(b"\x00\x01\xff")).startswith("c = ")
    # bytes are valid set elements even though a top-level bytes argument is rejected.
    assert repr(Reference("c").is_in([b"a", b"b"])).startswith("c IN (")


def test_uuid_literal_builds_uuid_predicate():
    import uuid

    u = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert repr(Reference("c").eq(u)).startswith("c = ")
    assert "12345678" in repr(Reference("c").eq(u))


def test_always_true_and_false():
    assert repr(Predicate.always_true()) == "TRUE"
    assert repr(Predicate.always_false()) == "FALSE"


@pytest.mark.parametrize(
    "compose,expected",
    [
        (lambda a, b: a.and_(b), "(x = 1) AND (y < 10)"),
        (lambda a, b: a & b, "(x = 1) AND (y < 10)"),
        (lambda a, b: a.or_(b), "(x = 1) OR (y < 10)"),
        (lambda a, b: a | b, "(x = 1) OR (y < 10)"),
    ],
)
def test_logical_composition_renders_both_operands(compose, expected):
    a = Reference("x").eq(1)
    b = Reference("y").lt(10)
    assert repr(compose(a, b)) == expected


@pytest.mark.parametrize(
    "predicate,expected",
    [
        (Reference("c").eq(5), "c != 5"),
        (Reference("c").ne(5), "c = 5"),
        (Reference("c").lt(5), "c >= 5"),
        (Reference("c").lte(5), "c > 5"),
        (Reference("c").gt(5), "c <= 5"),
        (Reference("c").is_null(), "c IS NOT NULL"),
        (Reference("c").is_not_null(), "c IS NULL"),
        (Reference("c").starts_with("foo"), 'c NOT STARTS WITH "foo"'),
    ],
)
def test_negate_renders_canonical_inverse(predicate, expected):
    assert repr(predicate.negate()) == expected


def test_invert_operator_delegates_to_negate():
    # One smoke test that ~ maps to negate(); the per-case inverses are covered above.
    predicate = Reference("c").eq(5)
    assert repr(~predicate) == repr(predicate.negate()) == "c != 5"


def test_negate_set_predicate_inverts_to_not_in():
    # The IN-set renders in hash order, so assert the operator, not element order.
    assert repr(Reference("c").is_in([1, 2, 3]).negate()).startswith("c NOT IN (")


def test_and_composition_preserves_all_predicates():
    pred = (
        Reference("i")
        .eq(5)
        .and_(Reference("s").eq("x"))
        .and_(Reference("d").eq(date(2026, 1, 1)))
        .and_(Reference("p").eq(Decimal("3.14")))
    )
    rendered = repr(pred)
    assert "i = 5" in rendered
    assert 's = "x"' in rendered
    assert "d = 2026-01-01" in rendered
    assert "p = 3.14" in rendered


def test_str_matches_repr():
    # __str__ and __repr__ both render the predicate text; keep them in lock-step.
    predicate = Reference("c").eq(5)
    assert str(predicate) == repr(predicate) == "c = 5"


# ---------------------------------------------------------------------------
# Exhaustive literal / break-mode coverage (every works-vs-breaks path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (math.inf, "c = inf"),
        (-math.inf, "c = -inf"),
        (math.nan, "c = NaN"),
        (-0.0, "c = -0"),
        (0.0, "c = 0"),
    ],
)
def test_float_special_values_render_exactly(value, expected):
    assert repr(Reference("c").eq(value)) == expected


def test_tz_aware_datetime_is_normalized_to_utc():
    # An aware datetime is converted to UTC wall-clock (10:30 +08:00 -> 02:30 UTC).
    pred = Reference("c").eq(
        datetime(2026, 5, 24, 10, 30, tzinfo=timezone(timedelta(hours=8)))
    )
    assert repr(pred) == "c = 2026-05-24 02:30:00 UTC"


@pytest.mark.parametrize("method", ["is_in", "is_not_in"])
@pytest.mark.parametrize("value", [42, 3.14, None, object()])
def test_set_methods_reject_non_iterable(method, value):
    with pytest.raises(TypeError, match="is not iterable"):
        getattr(Reference("c"), method)(value)


@pytest.mark.parametrize(
    "method,expected", [("is_in", "c IN ()"), ("is_not_in", "c NOT IN ()")]
)
@pytest.mark.parametrize("empty", [[], (), set()], ids=["list", "tuple", "set"])
def test_empty_set_predicate_is_not_simplified(method, expected, empty):
    assert repr(getattr(Reference("c"), method)(empty)) == expected


def test_set_method_none_element_propagates_type_error():
    with pytest.raises(TypeError, match=r"Cannot convert value of type 'NoneType'"):
        Reference("c").is_in([1, None, 3])


@pytest.mark.parametrize("nonfinite", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_decimal_in_is_in_propagates_value_error(nonfinite):
    with pytest.raises(ValueError, match=r"Cannot convert a non-finite Decimal"):
        Reference("c").is_in([Decimal("1.0"), Decimal(nonfinite)])


def test_set_method_passes_through_foreign_iteration_error():
    # An exception raised mid-iteration propagates unchanged (type and message).
    def boom():
        yield 1
        raise RuntimeError("gen boom")

    with pytest.raises(RuntimeError, match="gen boom"):
        Reference("c").is_in(boom())


@pytest.mark.parametrize(
    "value,typename",
    [(bytearray(b"ab"), "bytearray"), (memoryview(b"ab"), "memoryview")],
)
def test_bytearray_and_memoryview_are_not_binary(value, typename):
    with pytest.raises(TypeError, match=f"Cannot convert value of type '{typename}'"):
        Reference("c").eq(value)


def test_predicate_has_no_bool():
    # An expression DSL must NOT silently short-circuit on Python truthiness:
    # `p1 and p2` would otherwise return p2 instead of combining the predicates.
    p = Reference("c").eq(5)
    with pytest.raises(TypeError, match="no truth value"):
        bool(p)
    with pytest.raises(TypeError, match="no truth value"):
        p and Reference("c").eq(6)  # Python `and` invokes __bool__


def test_index_dunder_object_is_not_treated_as_int():
    class IndexLike:
        def __index__(self):
            return 5

    with pytest.raises(TypeError, match="Cannot convert value of type 'IndexLike'"):
        Reference("c").eq(IndexLike())


def test_out_of_range_int_does_not_invoke_user_str_or_repr():
    class EvilInt(int):
        def __str__(self):
            raise RuntimeError("str boom")

        def __repr__(self):
            raise RuntimeError("repr boom")

    with pytest.raises(ValueError, match="out of range"):
        Reference("c").eq(EvilInt(2**63))


@pytest.mark.parametrize(
    "name,expected",
    [
        ("O'Brien", 'Reference("O\'Brien")'),
        ('a"b', "Reference('a\"b')"),
        ("café", "Reference('café')"),
        ("a\nb", "Reference('a\\nb')"),
        ("", "Reference('')"),
    ],
)
def test_reference_repr_uses_python_quoting(name, expected):
    assert repr(Reference(name)) == expected


def test_decimal_exceeding_max_precision_is_rejected():
    big = Decimal("1234567890123456789012345678901234567890")  # 40 digits
    with pytest.raises(ValueError, match="Can't parse decimal"):
        Reference("c").eq(big)


def test_bool_elements_in_set_render_as_boolean_literals():
    rendered = repr(Reference("c").is_in([True, False]))
    assert rendered.startswith("c IN (")
    assert "true" in rendered and "false" in rendered


def test_bytes_elements_render_as_hex():
    rendered = repr(Reference("c").is_in([b"\x00", b"\xff"]))
    assert rendered.startswith("c IN (")
    assert "00" in rendered and "FF" in rendered
