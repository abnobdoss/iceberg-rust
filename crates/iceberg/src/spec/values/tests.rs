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

//! Tests for Iceberg value types

use apache_avro::to_value;
use apache_avro::types::Value;
use ordered_float::OrderedFloat;
use serde_bytes::ByteBuf;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use super::decimal_utils::{decimal_from_i128_with_scale, decimal_new};
use crate::ErrorKind;
use crate::avro::schema_to_avro_schema;
use crate::spec::Schema;
use crate::spec::Type::Primitive;
use crate::spec::datatypes::{ListType, MapType, NestedField, PrimitiveType, StructType, Type};
use crate::spec::values::datum::{INT_MAX, INT_MIN, LONG_MAX, LONG_MIN};
use crate::spec::values::serde::_serde;
use crate::spec::values::{Datum, Literal, Map, PrimitiveLiteral, RawLiteral, Struct};

fn check_json_serde(json: &str, expected_literal: Literal, expected_type: &Type) {
    let raw_json_value = serde_json::from_str::<JsonValue>(json).unwrap();
    let desered_literal = Literal::try_from_json(raw_json_value.clone(), expected_type).unwrap();
    assert_eq!(desered_literal, Some(expected_literal.clone()));

    let expected_json_value: JsonValue = expected_literal.try_into_json(expected_type).unwrap();
    let sered_json = serde_json::to_string(&expected_json_value).unwrap();
    let parsed_json_value = serde_json::from_str::<JsonValue>(&sered_json).unwrap();

    assert_eq!(parsed_json_value, raw_json_value);
}

fn check_avro_bytes_serde(input: Vec<u8>, expected_datum: Datum, expected_type: &PrimitiveType) {
    let raw_schema = r#""bytes""#;
    let schema = apache_avro::Schema::parse_str(raw_schema).unwrap();

    let bytes = ByteBuf::from(input);
    let datum = Datum::try_from_bytes(&bytes, expected_type.clone()).unwrap();
    assert_eq!(datum, expected_datum);

    let mut writer = apache_avro::Writer::new(&schema, Vec::new());
    writer.append_ser(datum.to_bytes().unwrap()).unwrap();
    let encoded = writer.into_inner().unwrap();
    let reader = apache_avro::Reader::with_schema(&schema, &*encoded).unwrap();

    for record in reader {
        let result = apache_avro::from_value::<ByteBuf>(&record.unwrap()).unwrap();
        let desered_datum = Datum::try_from_bytes(&result, expected_type.clone()).unwrap();
        assert_eq!(desered_datum, expected_datum);
    }
}

fn check_convert_with_avro(expected_literal: Literal, expected_type: &Type) {
    let fields = vec![NestedField::required(1, "col", expected_type.clone()).into()];
    let schema = Schema::builder()
        .with_fields(fields.clone())
        .build()
        .unwrap();
    let avro_schema = schema_to_avro_schema("test", &schema).unwrap();
    let struct_type = Type::Struct(StructType::new(fields));
    let struct_literal = Literal::Struct(Struct::from_iter(vec![Some(expected_literal.clone())]));

    let mut writer = apache_avro::Writer::new(&avro_schema, Vec::new());
    let raw_literal = RawLiteral::try_from(struct_literal.clone(), &struct_type).unwrap();
    writer.append_ser(raw_literal).unwrap();
    let encoded = writer.into_inner().unwrap();

    let reader = apache_avro::Reader::new(&*encoded).unwrap();
    for record in reader {
        let result = apache_avro::from_value::<RawLiteral>(&record.unwrap()).unwrap();
        let desered_literal = result.try_into(&struct_type).unwrap().unwrap();
        assert_eq!(desered_literal, struct_literal);
    }
}

fn check_serialize_avro(literal: Literal, ty: &Type, expect_value: Value) {
    let expect_value = Value::Record(vec![("col".to_string(), expect_value)]);

    let fields = vec![NestedField::required(1, "col", ty.clone()).into()];
    let schema = Schema::builder()
        .with_fields(fields.clone())
        .build()
        .unwrap();
    let avro_schema = schema_to_avro_schema("test", &schema).unwrap();
    let struct_type = Type::Struct(StructType::new(fields));
    let struct_literal = Literal::Struct(Struct::from_iter(vec![Some(literal.clone())]));
    let mut writer = apache_avro::Writer::new(&avro_schema, Vec::new());
    let raw_literal = RawLiteral::try_from(struct_literal.clone(), &struct_type).unwrap();
    let value = to_value(raw_literal)
        .unwrap()
        .resolve(&avro_schema)
        .unwrap();
    writer.append_value_ref(&value).unwrap();
    let encoded = writer.into_inner().unwrap();

    let reader = apache_avro::Reader::new(&*encoded).unwrap();
    for record in reader {
        assert_eq!(record.unwrap(), expect_value);
    }
}

#[test]
fn json_boolean() {
    let record = r#"true"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Boolean(true)),
        &Type::Primitive(PrimitiveType::Boolean),
    );
}

#[test]
fn json_int() {
    let record = r#"32"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Int(32)),
        &Type::Primitive(PrimitiveType::Int),
    );
}

#[test]
fn json_long() {
    let record = r#"32"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Long(32)),
        &Type::Primitive(PrimitiveType::Long),
    );
}

#[test]
fn json_float() {
    let record = r#"1.0"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Float(OrderedFloat(1.0))),
        &Type::Primitive(PrimitiveType::Float),
    );
}

#[test]
fn json_double() {
    let record = r#"1.0"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Double(OrderedFloat(1.0))),
        &Type::Primitive(PrimitiveType::Double),
    );
}

#[test]
fn json_date() {
    let record = r#""2017-11-16""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Int(17486)),
        &Type::Primitive(PrimitiveType::Date),
    );
}

#[test]
fn json_time() {
    let record = r#""22:31:08.123456""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Long(81068123456)),
        &Type::Primitive(PrimitiveType::Time),
    );
}

#[test]
fn json_timestamp() {
    let record = r#""2017-11-16T22:31:08.123456""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Long(1510871468123456)),
        &Type::Primitive(PrimitiveType::Timestamp),
    );
}

#[test]
fn json_timestamptz() {
    let record = r#""2017-11-16T22:31:08.123456+00:00""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Long(1510871468123456)),
        &Type::Primitive(PrimitiveType::Timestamptz),
    );
}

#[test]
fn json_string() {
    let record = r#""iceberg""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::String("iceberg".to_string())),
        &Type::Primitive(PrimitiveType::String),
    );
}

#[test]
fn json_uuid() {
    let record = r#""f79c3e09-677c-4bbd-a479-3f349cb785e7""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::UInt128(
            Uuid::parse_str("f79c3e09-677c-4bbd-a479-3f349cb785e7")
                .unwrap()
                .as_u128(),
        )),
        &Type::Primitive(PrimitiveType::Uuid),
    );
}

#[test]
fn json_decimal() {
    let record = r#""14.20""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Int128(1420)),
        &Type::decimal(28, 2).unwrap(),
    );
}

#[test]
fn json_binary() {
    let record = r#""00010fff""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Binary(vec![0, 1, 15, 255])),
        &Type::Primitive(PrimitiveType::Binary),
    );
}

#[test]
fn json_fixed() {
    let record = r#""00010fff""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Binary(vec![0, 1, 15, 255])),
        &Type::Primitive(PrimitiveType::Fixed(4)),
    );
}

#[test]
fn test_should_parse_json_binary_if_hex_uses_uppercase_digits() {
    let result = Literal::try_from_json(
        serde_json::json!("00010FFF"),
        &Type::Primitive(PrimitiveType::Binary),
    )
    .unwrap();

    assert_eq!(
        result,
        Some(Literal::Primitive(PrimitiveLiteral::Binary(vec![
            0, 1, 15, 255
        ])))
    );
}

#[test]
fn test_should_reject_json_binary_if_hex_is_invalid() {
    assert!(
        Literal::try_from_json(
            serde_json::json!("f"),
            &Type::Primitive(PrimitiveType::Binary),
        )
        .is_err()
    );
    assert!(
        Literal::try_from_json(
            serde_json::json!("fg"),
            &Type::Primitive(PrimitiveType::Binary),
        )
        .is_err()
    );
}

#[test]
fn test_should_reject_json_fixed_if_length_does_not_match() {
    assert!(
        Literal::try_from_json(
            serde_json::json!("ff"),
            &Type::Primitive(PrimitiveType::Fixed(2)),
        )
        .is_err()
    );
    assert!(
        Literal::Primitive(PrimitiveLiteral::Binary(vec![255]))
            .try_into_json(&Type::Primitive(PrimitiveType::Fixed(2)))
            .is_err()
    );
}

#[test]
fn json_struct() {
    let record = r#"{"1": 1, "2": "bar", "3": null}"#;

    check_json_serde(
        record,
        Literal::Struct(Struct::from_iter(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::String(
                "bar".to_string(),
            ))),
            None,
        ])),
        &Type::Struct(StructType::new(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "address", Type::Primitive(PrimitiveType::String)).into(),
        ])),
    );
}

#[test]
fn json_list() {
    let record = r#"[1, 2, 3, null]"#;

    check_json_serde(
        record,
        Literal::List(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(3))),
            None,
        ]),
        &Type::List(ListType {
            element_field: NestedField::list_element(0, Type::Primitive(PrimitiveType::Int), true)
                .into(),
        }),
    );
}

#[test]
fn json_map() {
    let record = r#"{ "keys": ["a", "b", "c"], "values": [1, 2, null] }"#;

    check_json_serde(
        record,
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::String("a".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("b".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("c".to_string())),
                None,
            ),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(0, Type::Primitive(PrimitiveType::String))
                .into(),
            value_field: NestedField::map_value_element(
                1,
                Type::Primitive(PrimitiveType::Int),
                true,
            )
            .into(),
        }),
    );
}

#[test]
fn avro_bytes_boolean() {
    let bytes = vec![1u8];

    check_avro_bytes_serde(bytes, Datum::bool(true), &PrimitiveType::Boolean);
}

#[test]
fn avro_bytes_int() {
    let bytes = vec![32u8, 0u8, 0u8, 0u8];

    check_avro_bytes_serde(bytes, Datum::int(32), &PrimitiveType::Int);
}

#[test]
fn avro_bytes_long() {
    let bytes = vec![32u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];

    check_avro_bytes_serde(bytes, Datum::long(32), &PrimitiveType::Long);
}

#[test]
fn avro_bytes_long_from_int() {
    let bytes = vec![32u8, 0u8, 0u8, 0u8];

    check_avro_bytes_serde(bytes, Datum::long(32), &PrimitiveType::Long);
}

#[test]
fn avro_bytes_float() {
    let bytes = vec![0u8, 0u8, 128u8, 63u8];

    check_avro_bytes_serde(bytes, Datum::float(1.0), &PrimitiveType::Float);
}

#[test]
fn avro_bytes_double() {
    let bytes = vec![0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 240u8, 63u8];

    check_avro_bytes_serde(bytes, Datum::double(1.0), &PrimitiveType::Double);
}

#[test]
fn avro_bytes_double_from_float() {
    let bytes = vec![0u8, 0u8, 128u8, 63u8];

    check_avro_bytes_serde(bytes, Datum::double(1.0), &PrimitiveType::Double);
}

#[test]
fn avro_bytes_string() {
    let bytes = vec![105u8, 99u8, 101u8, 98u8, 101u8, 114u8, 103u8];

    check_avro_bytes_serde(bytes, Datum::string("iceberg"), &PrimitiveType::String);
}

#[test]
fn avro_bytes_decimal() {
    // (input_bytes, decimal_num, expect_scale, expect_precision)
    let cases = vec![
        (vec![4u8, 210u8], 1234, 2, 38),
        (vec![251u8, 46u8], -1234, 2, 38),
        (vec![4u8, 210u8], 1234, 3, 38),
        (vec![251u8, 46u8], -1234, 3, 38),
        (vec![42u8], 42, 2, 2),
        (vec![214u8], -42, 2, 2),
    ];

    for (input_bytes, decimal_num, expect_scale, expect_precision) in cases {
        check_avro_bytes_serde(
            input_bytes,
            Datum::decimal_with_precision(decimal_new(decimal_num, expect_scale), expect_precision)
                .unwrap(),
            &PrimitiveType::Decimal {
                precision: expect_precision,
                scale: expect_scale,
            },
        );
    }
}

#[test]
fn avro_bytes_decimal_expect_error() {
    // (decimal_num, expect_scale, expect_precision)
    let cases = vec![(1234, 2, 1)];

    for (decimal_num, expect_scale, expect_precision) in cases {
        let result =
            Datum::decimal_with_precision(decimal_new(decimal_num, expect_scale), expect_precision);
        assert!(result.is_err(), "expect error but got {result:?}");
        assert_eq!(
            result.unwrap_err().kind(),
            ErrorKind::DataInvalid,
            "expect error DataInvalid",
        );
    }
}

fn check_raw_literal_bytes_serde_via_avro(
    input_bytes: Vec<u8>,
    expected_literal: Literal,
    expected_type: &Type,
) {
    use apache_avro::types::Value;

    // Create an Avro bytes value and deserialize it through the RawLiteral path
    let avro_value = Value::Bytes(input_bytes);
    let raw_literal: _serde::RawLiteral = apache_avro::from_value(&avro_value).unwrap();
    let result = raw_literal.try_into(expected_type).unwrap();
    assert_eq!(result, Some(expected_literal));
}

fn check_raw_literal_bytes_error_via_avro(input_bytes: Vec<u8>, expected_type: &Type) {
    use apache_avro::types::Value;

    let avro_value = Value::Bytes(input_bytes);
    let raw_literal: _serde::RawLiteral = apache_avro::from_value(&avro_value).unwrap();
    let result = raw_literal.try_into(expected_type);
    assert!(result.is_err(), "Expected error but got: {result:?}");
}

#[test]
fn test_raw_literal_bytes_binary() {
    let bytes = vec![1u8, 2u8, 3u8, 4u8, 5u8];
    check_raw_literal_bytes_serde_via_avro(
        bytes.clone(),
        Literal::binary(bytes),
        &Type::Primitive(PrimitiveType::Binary),
    );
}

#[test]
fn test_raw_literal_bytes_binary_empty() {
    let bytes = vec![];
    check_raw_literal_bytes_serde_via_avro(
        bytes.clone(),
        Literal::binary(bytes),
        &Type::Primitive(PrimitiveType::Binary),
    );
}

#[test]
fn test_raw_literal_bytes_fixed_correct_length() {
    let bytes = vec![1u8, 2u8, 3u8, 4u8];
    check_raw_literal_bytes_serde_via_avro(
        bytes.clone(),
        Literal::fixed(bytes),
        &Type::Primitive(PrimitiveType::Fixed(4)),
    );
}

#[test]
fn test_raw_literal_bytes_fixed_wrong_length() {
    let bytes = vec![1u8, 2u8, 3u8]; // 3 bytes, but expecting 4
    check_raw_literal_bytes_error_via_avro(bytes, &Type::Primitive(PrimitiveType::Fixed(4)));
}

#[test]
fn test_raw_literal_bytes_fixed_empty_correct_length() {
    let bytes = vec![];
    check_raw_literal_bytes_serde_via_avro(
        bytes.clone(),
        Literal::fixed(bytes),
        &Type::Primitive(PrimitiveType::Fixed(0)),
    );
}

#[test]
fn test_raw_literal_bytes_uuid_correct_length() {
    let uuid_bytes = vec![
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
        0xef,
    ];
    let expected_uuid = u128::from_be_bytes([
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
        0xef,
    ]);
    check_raw_literal_bytes_serde_via_avro(
        uuid_bytes,
        Literal::Primitive(PrimitiveLiteral::UInt128(expected_uuid)),
        &Type::Primitive(PrimitiveType::Uuid),
    );
}

#[test]
fn test_raw_literal_bytes_uuid_wrong_length() {
    let bytes = vec![1u8, 2u8, 3u8]; // 3 bytes, but UUID needs 16
    check_raw_literal_bytes_error_via_avro(bytes, &Type::Primitive(PrimitiveType::Uuid));
}

#[test]
fn test_raw_literal_bytes_decimal_precision_4_scale_2() {
    // Precision 4 requires 2 bytes
    let decimal_bytes = vec![0x04, 0xd2]; // 1234 in 2 bytes
    let expected_decimal = 1234i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 4,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_4_negative() {
    // Precision 4 requires 2 bytes, negative number
    let decimal_bytes = vec![0xfb, 0x2e]; // -1234 in 2 bytes
    let expected_decimal = -1234i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 4,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_9_scale_2() {
    // Precision 9 requires 4 bytes
    let decimal_bytes = vec![0x00, 0x12, 0xd6, 0x87]; // 1234567 in 4 bytes
    let expected_decimal = 1234567i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 9,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_18_scale_2() {
    // Precision 18 requires 8 bytes
    let decimal_bytes = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xd2]; // 1234 in 8 bytes
    let expected_decimal = 1234i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 18,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_38_scale_2() {
    // Precision 38 requires 16 bytes (maximum precision)
    let decimal_bytes = vec![
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,
        0xd2, // 1234 in 16 bytes
    ];
    let expected_decimal = 1234i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 38,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_1_scale_0() {
    // Precision 1 requires 1 byte
    let decimal_bytes = vec![0x07]; // 7 in 1 byte
    let expected_decimal = 7i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 1,
            scale: 0,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_1_negative() {
    // Precision 1 requires 1 byte, negative number
    let decimal_bytes = vec![0xf9]; // -7 in 1 byte (two's complement)
    let expected_decimal = -7i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 1,
            scale: 0,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_wrong_length() {
    // 3 bytes provided, but precision 4 requires 2 bytes
    let bytes = vec![1u8, 2u8, 3u8];
    check_raw_literal_bytes_error_via_avro(
        bytes,
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 4,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_wrong_length_too_few() {
    // 1 byte provided, but precision 9 requires 4 bytes
    let bytes = vec![0x42];
    check_raw_literal_bytes_error_via_avro(
        bytes,
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 9,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_unsupported_type() {
    let bytes = vec![1u8, 2u8, 3u8, 4u8];
    check_raw_literal_bytes_error_via_avro(bytes, &Type::Primitive(PrimitiveType::Int));
}

#[test]
fn avro_convert_test_int() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Int(32)),
        &Type::Primitive(PrimitiveType::Int),
    );
}

#[test]
fn avro_convert_test_long() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(32)),
        &Type::Primitive(PrimitiveType::Long),
    );
}

#[test]
fn avro_convert_test_float() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Float(OrderedFloat(1.0))),
        &Type::Primitive(PrimitiveType::Float),
    );
}

#[test]
fn avro_convert_test_double() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Double(OrderedFloat(1.0))),
        &Type::Primitive(PrimitiveType::Double),
    );
}

#[test]
fn avro_convert_test_string() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::String("iceberg".to_string())),
        &Type::Primitive(PrimitiveType::String),
    );
}

#[test]
fn avro_convert_test_date() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Int(17486)),
        &Type::Primitive(PrimitiveType::Date),
    );
}

#[test]
fn avro_convert_test_time() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(81068123456)),
        &Type::Primitive(PrimitiveType::Time),
    );
}

#[test]
fn avro_convert_test_timestamp() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(1510871468123456)),
        &Type::Primitive(PrimitiveType::Timestamp),
    );
}

#[test]
fn avro_convert_test_timestamptz() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(1510871468123456)),
        &Type::Primitive(PrimitiveType::Timestamptz),
    );
}

#[test]
fn avro_convert_test_list() {
    check_convert_with_avro(
        Literal::List(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(3))),
            None,
        ]),
        &Type::List(ListType {
            element_field: NestedField::list_element(0, Type::Primitive(PrimitiveType::Int), false)
                .into(),
        }),
    );

    check_convert_with_avro(
        Literal::List(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(3))),
        ]),
        &Type::List(ListType {
            element_field: NestedField::list_element(0, Type::Primitive(PrimitiveType::Int), true)
                .into(),
        }),
    );
}

fn check_convert_with_avro_map(expected_literal: Literal, expected_type: &Type) {
    let fields = vec![NestedField::required(1, "col", expected_type.clone()).into()];
    let schema = Schema::builder()
        .with_fields(fields.clone())
        .build()
        .unwrap();
    let avro_schema = schema_to_avro_schema("test", &schema).unwrap();
    let struct_type = Type::Struct(StructType::new(fields));
    let struct_literal = Literal::Struct(Struct::from_iter(vec![Some(expected_literal.clone())]));

    let mut writer = apache_avro::Writer::new(&avro_schema, Vec::new());
    let raw_literal = RawLiteral::try_from(struct_literal.clone(), &struct_type).unwrap();
    writer.append_ser(raw_literal).unwrap();
    let encoded = writer.into_inner().unwrap();

    let reader = apache_avro::Reader::new(&*encoded).unwrap();
    for record in reader {
        let result = apache_avro::from_value::<RawLiteral>(&record.unwrap()).unwrap();
        let desered_literal = result.try_into(&struct_type).unwrap().unwrap();
        match (&desered_literal, &struct_literal) {
            (Literal::Struct(desered), Literal::Struct(expected)) => {
                match (&desered.fields()[0], &expected.fields()[0]) {
                    (Some(Literal::Map(desered)), Some(Literal::Map(expected))) => {
                        assert!(desered.has_same_content(expected))
                    }
                    _ => {
                        unreachable!()
                    }
                }
            }
            _ => {
                panic!("unexpected literal type");
            }
        }
    }
}

#[test]
fn avro_convert_test_map() {
    check_convert_with_avro_map(
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::Int(1)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::Int(2)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(2))),
            ),
            (Literal::Primitive(PrimitiveLiteral::Int(3)), None),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(2, Type::Primitive(PrimitiveType::Int)).into(),
            value_field: NestedField::map_value_element(
                3,
                Type::Primitive(PrimitiveType::Long),
                false,
            )
            .into(),
        }),
    );

    check_convert_with_avro_map(
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::Int(1)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::Int(2)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(2))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::Int(3)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(3))),
            ),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(2, Type::Primitive(PrimitiveType::Int)).into(),
            value_field: NestedField::map_value_element(
                3,
                Type::Primitive(PrimitiveType::Long),
                true,
            )
            .into(),
        }),
    );
}

#[test]
fn avro_convert_test_string_map() {
    check_convert_with_avro_map(
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::String("a".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("b".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("c".to_string())),
                None,
            ),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(2, Type::Primitive(PrimitiveType::String))
                .into(),
            value_field: NestedField::map_value_element(
                3,
                Type::Primitive(PrimitiveType::Int),
                false,
            )
            .into(),
        }),
    );

    check_convert_with_avro_map(
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::String("a".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("b".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("c".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(3))),
            ),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(2, Type::Primitive(PrimitiveType::String))
                .into(),
            value_field: NestedField::map_value_element(
                3,
                Type::Primitive(PrimitiveType::Int),
                true,
            )
            .into(),
        }),
    );
}

#[test]
fn avro_convert_test_record() {
    check_convert_with_avro(
        Literal::Struct(Struct::from_iter(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::String(
                "bar".to_string(),
            ))),
            None,
        ])),
        &Type::Struct(StructType::new(vec![
            NestedField::required(2, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(4, "address", Type::Primitive(PrimitiveType::String)).into(),
        ])),
    );
}

// # TODO:https://github.com/apache/iceberg-rust/issues/86
// rust avro don't support deserialize any bytes representation now:
// - binary
// - decimal
#[test]
fn avro_convert_test_binary_ser() {
    let literal = Literal::Primitive(PrimitiveLiteral::Binary(vec![1, 2, 3, 4, 5]));
    let ty = Type::Primitive(PrimitiveType::Binary);
    let expect_value = Value::Bytes(vec![1, 2, 3, 4, 5]);
    check_serialize_avro(literal, &ty, expect_value);
}

#[test]
fn avro_convert_test_decimal_ser() {
    let literal = Literal::decimal(12345);
    let ty = Type::Primitive(PrimitiveType::Decimal {
        precision: 9,
        scale: 8,
    });
    let expect_value = Value::Decimal(apache_avro::Decimal::from(12345_i128.to_be_bytes()));
    check_serialize_avro(literal, &ty, expect_value);
}

// # TODO:https://github.com/apache/iceberg-rust/issues/86
// rust avro can't support to convert any byte-like type to fixed in avro now.
// - uuid ser/de
// - fixed ser/de

#[test]
fn test_parse_timestamp() {
    let value = Datum::timestamp_from_str("2021-08-01T01:09:00.0899").unwrap();
    assert_eq!(&format!("{value}"), "2021-08-01 01:09:00.089900");

    let value = Datum::timestamp_from_str("2023-01-06T00:00:00").unwrap();
    assert_eq!(&format!("{value}"), "2023-01-06 00:00:00");

    let value = Datum::timestamp_from_str("2021-08-01T01:09:00.0899+0800");
    assert!(value.is_err(), "Parse timestamp with timezone should fail!");

    let value = Datum::timestamp_from_str("dfa");
    assert!(
        value.is_err(),
        "Parse timestamp with invalid input should fail!"
    );
}

#[test]
fn test_parse_timestamptz() {
    let value = Datum::timestamptz_from_str("2021-08-01T09:09:00.0899+0800").unwrap();
    assert_eq!(&format!("{value}"), "2021-08-01 01:09:00.089900 UTC");

    let value = Datum::timestamptz_from_str("2021-08-01T01:09:00.0899");
    assert!(
        value.is_err(),
        "Parse timestamptz without timezone should fail!"
    );

    let value = Datum::timestamptz_from_str("dfa");
    assert!(
        value.is_err(),
        "Parse timestamptz with invalid input should fail!"
    );
}

#[test]
fn test_datum_ser_deser() {
    let test_fn = |datum: Datum| {
        let json = serde_json::to_value(&datum).unwrap();
        let desered_datum: Datum = serde_json::from_value(json).unwrap();
        assert_eq!(datum, desered_datum);
    };
    let datum = Datum::int(1);
    test_fn(datum);
    let datum = Datum::long(1);
    test_fn(datum);

    let datum = Datum::float(1.0);
    test_fn(datum);
    let datum = Datum::float(0_f32);
    test_fn(datum);
    let datum = Datum::float(-0_f32);
    test_fn(datum);
    let datum = Datum::float(f32::MAX);
    test_fn(datum);
    let datum = Datum::float(f32::MIN);
    test_fn(datum);

    // serde_json can't serialize f32::INFINITY, f32::NEG_INFINITY, f32::NAN
    let datum = Datum::float(f32::INFINITY);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());
    let datum = Datum::float(f32::NEG_INFINITY);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());
    let datum = Datum::float(f32::NAN);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());

    let datum = Datum::double(1.0);
    test_fn(datum);
    let datum = Datum::double(f64::MAX);
    test_fn(datum);
    let datum = Datum::double(f64::MIN);
    test_fn(datum);

    // serde_json can't serialize f32::INFINITY, f32::NEG_INFINITY, f32::NAN
    let datum = Datum::double(f64::INFINITY);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());
    let datum = Datum::double(f64::NEG_INFINITY);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());
    let datum = Datum::double(f64::NAN);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());

    let datum = Datum::string("iceberg");
    test_fn(datum);
    let datum = Datum::bool(true);
    test_fn(datum);
    let datum = Datum::date(17486);
    test_fn(datum);
    let datum = Datum::time_from_hms_micro(22, 15, 33, 111).unwrap();
    test_fn(datum);
    let datum = Datum::timestamp_micros(1510871468123456);
    test_fn(datum);
    let datum = Datum::timestamptz_micros(1510871468123456);
    test_fn(datum);
    let datum = Datum::uuid(Uuid::parse_str("f79c3e09-677c-4bbd-a479-3f349cb785e7").unwrap());
    test_fn(datum);
    let datum = Datum::decimal(decimal_new(1420, 0)).unwrap();
    test_fn(datum);
    let datum = Datum::binary(vec![1, 2, 3, 4, 5]);
    test_fn(datum);
    let datum = Datum::fixed(vec![1, 2, 3, 4, 5]);
    test_fn(datum);
}

#[test]
fn test_datum_date_convert_to_int() {
    let datum_date = Datum::date(12345);

    let result = datum_date.to(&Primitive(PrimitiveType::Int)).unwrap();

    let expected = Datum::int(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_int_convert_to_date() {
    let datum_int = Datum::int(12345);

    let result = datum_int.to(&Primitive(PrimitiveType::Date)).unwrap();

    let expected = Datum::date(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_int_convert_to_float() {
    let datum = Datum::int(34);

    let result = datum.to(&Primitive(PrimitiveType::Float)).unwrap();

    assert_eq!(result, Datum::float(34.0f32));
}

#[test]
fn test_datum_int_convert_to_double() {
    let datum = Datum::int(34);

    let result = datum.to(&Primitive(PrimitiveType::Double)).unwrap();

    assert_eq!(result, Datum::double(34.0f64));
}

#[test]
fn test_datum_int_convert_to_decimal() {
    let target_type = Primitive(PrimitiveType::Decimal {
        precision: 4,
        scale: 2,
    });
    let datum = Datum::int(34);

    let result = datum.to(&target_type).unwrap();

    assert_eq!(result.data_type(), &PrimitiveType::Decimal {
        precision: 4,
        scale: 2,
    });
    assert_eq!(result.literal(), &PrimitiveLiteral::Int128(3400));
}

#[test]
fn test_datum_date_convert_to_numeric_is_unsupported() {
    // `date` is physically an `Int` literal; it must not flow through the
    // integer numeric-widening routes. Java's `DateLiteral.to` only supports
    // the `date` target.
    let datum = Datum::date(19_000);
    for target in [
        PrimitiveType::Float,
        PrimitiveType::Double,
        PrimitiveType::Decimal {
            precision: 9,
            scale: 2,
        },
    ] {
        let result = datum.clone().to(&Primitive(target.clone()));
        assert!(
            result.is_err(),
            "{target:?}: expected error, got {result:?}"
        );
        assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
    }
}

#[test]
fn test_datum_timestamp_convert_to_numeric_is_unsupported() {
    // `timestamp` is physically a `Long` literal; it must not flow through the
    // long numeric-widening routes. Java's `TimestampLiteral.to` does not
    // support numeric targets.
    let datum = Datum::timestamp_micros(1_000_000);
    for target in [
        PrimitiveType::Float,
        PrimitiveType::Double,
        PrimitiveType::Decimal {
            precision: 18,
            scale: 0,
        },
    ] {
        let result = datum.clone().to(&Primitive(target.clone()));
        assert!(
            result.is_err(),
            "{target:?}: expected error, got {result:?}"
        );
        assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
    }
}

#[test]
fn test_datum_double_convert_to_decimal_uses_half_up_rounding() {
    // Lock in HALF_UP (away-from-zero) rounding to match Java's
    // `BigDecimal.setScale(scale, RoundingMode.HALF_UP)`. A midpoint such as
    // 0.125 -> scale 2 rounds to 0.13 under HALF_UP but 0.12 under banker's
    // rounding, so this guards against a rounding-mode regression.
    let target_type = Primitive(PrimitiveType::Decimal {
        precision: 4,
        scale: 2,
    });
    let result = Datum::double(0.125f64).to(&target_type).unwrap();
    assert_eq!(result.literal(), &PrimitiveLiteral::Int128(13));

    let result = Datum::double(-0.125f64).to(&target_type).unwrap();
    assert_eq!(result.literal(), &PrimitiveLiteral::Int128(-13));
}

#[test]
fn test_datum_int_convert_to_decimal_rejects_precision_too_narrow() {
    let target_type = Primitive(PrimitiveType::Decimal {
        precision: 2,
        scale: 0,
    });
    let datum = Datum::int(100);

    let result = datum.to(&target_type);

    assert!(result.is_err(), "expect error but got {result:?}");
    assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
}

#[test]
fn test_datum_long_convert_to_date() {
    let datum = Datum::long(12345);

    let result = datum.to(&Primitive(PrimitiveType::Date)).unwrap();

    let expected = Datum::date(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_date_above_max() {
    let datum = Datum::long(INT_MAX as i64 + 1);

    let result = datum.to(&Primitive(PrimitiveType::Date)).unwrap();

    let expected = Datum::new(PrimitiveType::Date, PrimitiveLiteral::AboveMax);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_date_below_min() {
    let datum = Datum::long(INT_MIN as i64 - 1);

    let result = datum.to(&Primitive(PrimitiveType::Date)).unwrap();

    let expected = Datum::new(PrimitiveType::Date, PrimitiveLiteral::BelowMin);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_time_convert_to_date_is_unsupported() {
    // `time` is physically a `Long` literal; it must not be reinterpreted as a
    // day ordinal via the `long -> date` route. Java's `TimeLiteral.to` only
    // supports the `time` target.
    let datum = Datum::time_micros(123_456).unwrap();

    let result = datum.to(&Primitive(PrimitiveType::Date));

    assert!(result.is_err(), "expect error but got {result:?}");
    assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
}

#[test]
fn test_datum_long_convert_to_int() {
    let datum = Datum::long(12345);

    let result = datum.to(&Primitive(PrimitiveType::Int)).unwrap();

    let expected = Datum::int(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_int_above_max() {
    let datum = Datum::long(INT_MAX as i64 + 1);

    let result = datum.to(&Primitive(PrimitiveType::Int)).unwrap();

    let expected = Datum::new(PrimitiveType::Int, PrimitiveLiteral::AboveMax);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_int_below_min() {
    let datum = Datum::long(INT_MIN as i64 - 1);

    let result = datum.to(&Primitive(PrimitiveType::Int)).unwrap();

    let expected = Datum::new(PrimitiveType::Int, PrimitiveLiteral::BelowMin);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_timestamp() {
    let datum = Datum::long(12345);

    let result = datum.to(&Primitive(PrimitiveType::Timestamp)).unwrap();

    let expected = Datum::timestamp_micros(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_timestamptz() {
    let datum = Datum::long(12345);

    let result = datum.to(&Primitive(PrimitiveType::Timestamptz)).unwrap();

    let expected = Datum::timestamptz_micros(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_float() {
    let datum = Datum::long(34);

    let result = datum.to(&Primitive(PrimitiveType::Float)).unwrap();

    assert_eq!(result, Datum::float(34.0f32));
}

#[test]
fn test_datum_long_convert_to_double() {
    let datum = Datum::long(34);

    let result = datum.to(&Primitive(PrimitiveType::Double)).unwrap();

    assert_eq!(result, Datum::double(34.0f64));
}

#[test]
fn test_datum_long_convert_to_decimal() {
    let target_type = Primitive(PrimitiveType::Decimal {
        precision: 4,
        scale: 1,
    });
    let datum = Datum::long(-34);

    let result = datum.to(&target_type).unwrap();

    assert_eq!(result.data_type(), &PrimitiveType::Decimal {
        precision: 4,
        scale: 1,
    });
    assert_eq!(result.literal(), &PrimitiveLiteral::Int128(-340));
}

#[test]
fn test_datum_long_convert_to_decimal_rejects_precision_too_narrow() {
    let target_type = Primitive(PrimitiveType::Decimal {
        precision: 38,
        scale: 20,
    });
    let datum = Datum::long(LONG_MAX);

    let result = datum.to(&target_type);

    assert!(result.is_err(), "expect error but got {result:?}");
    assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
}

#[test]
fn test_datum_double_convert_to_float() {
    let datum = Datum::double(1.25f64);
    let result = datum.to(&Primitive(PrimitiveType::Float)).unwrap();
    let expected = Datum::float(1.25f32);
    assert_eq!(result, expected);

    // In-range but not exactly representable: round to the nearest float,
    // matching Java's `DoubleLiteral.to(FloatType)`.
    let datum_inexact = Datum::double(1.23f64);
    let result_inexact = datum_inexact.to(&Primitive(PrimitiveType::Float)).unwrap();
    assert_eq!(result_inexact, Datum::float(1.23f32));

    let datum_neg_zero = Datum::double(-0.0f64);
    let result_neg_zero = datum_neg_zero.to(&Primitive(PrimitiveType::Float)).unwrap();
    assert_eq!(result_neg_zero, Datum::float(-0.0f32));

    let datum_nan = Datum::double(f64::NAN);
    let result_nan = datum_nan.to(&Primitive(PrimitiveType::Float)).unwrap();
    assert!(result_nan.is_nan());
    assert_eq!(result_nan.data_type(), &PrimitiveType::Float);
}

#[test]
fn test_datum_double_convert_to_float_above_max() {
    // Exact upper boundary: f32::MAX is representable, so it must round to the
    // float and not collapse to AboveMax.
    let datum_max = Datum::double(f32::MAX as f64);
    let result_max = datum_max.to(&Primitive(PrimitiveType::Float)).unwrap();
    assert_eq!(result_max, Datum::float(f32::MAX));

    // Above float range: collapse to the AboveMax sentinel, not an error.
    let datum_overflow = Datum::double(1e40f64);
    let result_overflow = datum_overflow.to(&Primitive(PrimitiveType::Float)).unwrap();
    assert_eq!(
        result_overflow,
        Datum::new(PrimitiveType::Float, PrimitiveLiteral::AboveMax)
    );

    // +Inf is above Float.MAX_VALUE, so Java collapses it to AboveMax rather
    // than a float infinity.
    let datum_inf = Datum::double(f64::INFINITY);
    let result_inf = datum_inf.to(&Primitive(PrimitiveType::Float)).unwrap();
    assert_eq!(
        result_inf,
        Datum::new(PrimitiveType::Float, PrimitiveLiteral::AboveMax)
    );
}

#[test]
fn test_datum_double_convert_to_float_below_min() {
    // Exact lower boundary: f32::MIN is representable, so it must round to the
    // float and not collapse to BelowMin.
    let datum_min = Datum::double(f32::MIN as f64);
    let result_min = datum_min.to(&Primitive(PrimitiveType::Float)).unwrap();
    assert_eq!(result_min, Datum::float(f32::MIN));

    // Below float range: collapse to the BelowMin sentinel.
    let datum_underflow = Datum::double(-1e40f64);
    let result_underflow = datum_underflow
        .to(&Primitive(PrimitiveType::Float))
        .unwrap();
    assert_eq!(
        result_underflow,
        Datum::new(PrimitiveType::Float, PrimitiveLiteral::BelowMin)
    );
}

#[test]
fn test_datum_float_convert_to_double() {
    let datum = Datum::float(1.25f32);

    let result = datum.to(&Primitive(PrimitiveType::Double)).unwrap();

    assert_eq!(result, Datum::double(1.25f64));
}

#[test]
fn test_datum_float_convert_to_decimal() {
    let target_type = Primitive(PrimitiveType::Decimal {
        precision: 3,
        scale: 1,
    });
    let datum = Datum::float(1.25f32);

    let result = datum.to(&target_type).unwrap();

    assert_eq!(result.data_type(), &PrimitiveType::Decimal {
        precision: 3,
        scale: 1,
    });
    assert_eq!(result.literal(), &PrimitiveLiteral::Int128(13));
}

#[test]
fn test_datum_double_convert_to_decimal() {
    let target_type = Primitive(PrimitiveType::Decimal {
        precision: 4,
        scale: 2,
    });
    let datum = Datum::double(-1.255f64);

    let result = datum.to(&target_type).unwrap();

    assert_eq!(result.data_type(), &PrimitiveType::Decimal {
        precision: 4,
        scale: 2,
    });
    assert_eq!(result.literal(), &PrimitiveLiteral::Int128(-126));
}

#[test]
fn test_datum_double_convert_to_decimal_rejects_precision_too_narrow() {
    let target_type = Primitive(PrimitiveType::Decimal {
        precision: 3,
        scale: 1,
    });
    let datum = Datum::double(999.95f64);

    let result = datum.to(&target_type);

    assert!(result.is_err(), "expect error but got {result:?}");
    assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
}

#[test]
fn test_datum_float_and_double_convert_to_decimal_rejects_non_finite() {
    let target_type = Primitive(PrimitiveType::Decimal {
        precision: 9,
        scale: 2,
    });

    let result_float = Datum::float(f32::NAN).to(&target_type);
    assert!(
        result_float.is_err(),
        "expect error but got {result_float:?}"
    );
    assert_eq!(result_float.unwrap_err().kind(), ErrorKind::DataInvalid);

    let result_double = Datum::double(f64::INFINITY).to(&target_type);
    assert!(
        result_double.is_err(),
        "expect error but got {result_double:?}"
    );
    assert_eq!(result_double.unwrap_err().kind(), ErrorKind::DataInvalid);
}

#[test]
fn test_datum_float_and_double_convert_to_int_or_long_is_unsupported() {
    for datum in [Datum::float(1.25f32), Datum::double(1.25f64)] {
        for target_type in [
            Primitive(PrimitiveType::Int),
            Primitive(PrimitiveType::Long),
        ] {
            let result = datum.clone().to(&target_type);
            assert!(result.is_err(), "expect error but got {result:?}");
            assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
        }
    }
}

#[test]
fn test_datum_decimal_convert_to_long() {
    let datum = Datum::decimal(decimal_new(12345, 0)).unwrap();

    let result = datum.to(&Primitive(PrimitiveType::Long)).unwrap();

    let expected = Datum::long(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_decimal_convert_to_long_above_max() {
    let datum = Datum::decimal(decimal_from_i128_with_scale(LONG_MAX as i128 + 1, 0)).unwrap();

    let result = datum.to(&Primitive(PrimitiveType::Long)).unwrap();

    let expected = Datum::new(PrimitiveType::Long, PrimitiveLiteral::AboveMax);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_decimal_convert_to_long_below_min() {
    let datum = Datum::decimal(decimal_from_i128_with_scale(LONG_MIN as i128 - 1, 0)).unwrap();

    let result = datum.to(&Primitive(PrimitiveType::Long)).unwrap();

    let expected = Datum::new(PrimitiveType::Long, PrimitiveLiteral::BelowMin);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_string_convert_to_boolean() {
    let datum = Datum::string("true");

    let result = datum.to(&Primitive(PrimitiveType::Boolean)).unwrap();

    let expected = Datum::bool(true);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_string_convert_to_int() {
    let datum = Datum::string("12345");

    let result = datum.to(&Primitive(PrimitiveType::Int)).unwrap();

    let expected = Datum::int(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_string_convert_to_long() {
    let datum = Datum::string("12345");

    let result = datum.to(&Primitive(PrimitiveType::Long)).unwrap();

    let expected = Datum::long(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_string_convert_to_timestamp() {
    let datum = Datum::string("1925-05-20T19:25:00.000");

    let result = datum.to(&Primitive(PrimitiveType::Timestamp)).unwrap();

    let expected = Datum::timestamp_micros(-1407990900000000);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_string_convert_to_timestamptz() {
    let datum = Datum::string("1925-05-20T19:25:00.000 UTC");

    let result = datum.to(&Primitive(PrimitiveType::Timestamptz)).unwrap();

    let expected = Datum::timestamptz_micros(-1407990900000000);

    assert_eq!(result, expected);
}

#[test]
fn test_iceberg_float_order() {
    // Test float ordering
    let float_values = vec![
        Datum::float(f32::NAN),
        Datum::float(-f32::NAN),
        Datum::float(f32::MAX),
        Datum::float(f32::MIN),
        Datum::float(f32::INFINITY),
        Datum::float(-f32::INFINITY),
        Datum::float(1.0),
        Datum::float(-1.0),
        Datum::float(0.0),
        Datum::float(-0.0),
    ];

    let mut float_sorted = float_values.clone();
    float_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let float_expected = vec![
        Datum::float(-f32::NAN),
        Datum::float(-f32::INFINITY),
        Datum::float(f32::MIN),
        Datum::float(-1.0),
        Datum::float(-0.0),
        Datum::float(0.0),
        Datum::float(1.0),
        Datum::float(f32::MAX),
        Datum::float(f32::INFINITY),
        Datum::float(f32::NAN),
    ];

    assert_eq!(float_sorted, float_expected);

    // Test double ordering
    let double_values = vec![
        Datum::double(f64::NAN),
        Datum::double(-f64::NAN),
        Datum::double(f64::INFINITY),
        Datum::double(-f64::INFINITY),
        Datum::double(f64::MAX),
        Datum::double(f64::MIN),
        Datum::double(1.0),
        Datum::double(-1.0),
        Datum::double(0.0),
        Datum::double(-0.0),
    ];

    let mut double_sorted = double_values.clone();
    double_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let double_expected = vec![
        Datum::double(-f64::NAN),
        Datum::double(-f64::INFINITY),
        Datum::double(f64::MIN),
        Datum::double(-1.0),
        Datum::double(-0.0),
        Datum::double(0.0),
        Datum::double(1.0),
        Datum::double(f64::MAX),
        Datum::double(f64::INFINITY),
        Datum::double(f64::NAN),
    ];

    assert_eq!(double_sorted, double_expected);
}

#[test]
fn test_negative_zero_less_than_positive_zero() {
    {
        let neg_zero = Datum::float(-0.0);
        let pos_zero = Datum::float(0.0);

        assert_eq!(
            neg_zero.partial_cmp(&pos_zero),
            Some(std::cmp::Ordering::Less),
            "IEEE 754 totalOrder requires -0.0 < +0.0 on F32"
        );
    }

    {
        let neg_zero = Datum::double(-0.0);
        let pos_zero = Datum::double(0.0);

        assert_eq!(
            neg_zero.partial_cmp(&pos_zero),
            Some(std::cmp::Ordering::Less),
            "IEEE 754 totalOrder requires -0.0 < +0.0 on F64"
        );
    }
}

/// Test Date deserialization from JSON as number (days since epoch).
///
/// This reproduces the scenario from Iceberg Java's TestAddFilesProcedure where:
/// - Date partition columns have initial_default values in manifests
/// - These values are serialized as days since epoch (e.g., 18628 for 2021-01-01)
/// - The JSON schema includes: {"type":"date","initial-default":18628}
///
/// Prior to this fix, Date values in JSON were only parsed from String format ("2021-01-01"),
/// causing initial_default values to be lost during schema deserialization.
///
/// This test ensures both formats are supported:
/// - String format: "2021-01-01" (used in table metadata)
/// - Number format: 18628 (used in initial-default values from add_files)
///
/// See: Iceberg Java TestAddFilesProcedure.addDataPartitionedByDateToPartitioned()
#[test]
fn test_date_from_json_as_number() {
    use serde_json::json;

    // Test Date as number (days since epoch) - used in initial-default from add_files
    let date_number = json!(18628); // 2021-01-01 is 18628 days since 1970-01-01
    let result =
        Literal::try_from_json(date_number, &Type::Primitive(PrimitiveType::Date)).unwrap();
    assert_eq!(
        result,
        Some(Literal::Primitive(PrimitiveLiteral::Int(18628)))
    );

    // Test Date as string - traditional format
    let date_string = json!("2021-01-01");
    let result =
        Literal::try_from_json(date_string, &Type::Primitive(PrimitiveType::Date)).unwrap();
    assert_eq!(
        result,
        Some(Literal::Primitive(PrimitiveLiteral::Int(18628)))
    );

    // Both formats should produce the same Literal value
}

#[test]
fn test_datum_to_decimal_narrows_precision_when_scale_matches() {
    let target_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 9,
        scale: 2,
    });
    let datum = Datum::decimal_from_str("123.45").unwrap();

    let converted = datum.to(&target_type).unwrap();

    assert_eq!(converted.data_type(), &PrimitiveType::Decimal {
        precision: 9,
        scale: 2,
    });
    assert_eq!(converted.literal(), &PrimitiveLiteral::Int128(12345));
}

#[test]
fn test_datum_to_decimal_widens_precision_when_scale_matches() {
    let target_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 38,
        scale: 2,
    });
    let datum = Datum::decimal_with_precision(decimal_from_i128_with_scale(12345, 2), 9).unwrap();

    let converted = datum.to(&target_type).unwrap();

    assert_eq!(converted.data_type(), &PrimitiveType::Decimal {
        precision: 38,
        scale: 2,
    });
    assert_eq!(converted.literal(), &PrimitiveLiteral::Int128(12345));
}

#[test]
fn test_datum_to_decimal_accepts_zero_mantissa() {
    let target_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 1,
        scale: 0,
    });
    let datum = Datum::decimal_with_precision(decimal_from_i128_with_scale(0, 0), 9).unwrap();

    let converted = datum.to(&target_type).unwrap();

    assert_eq!(converted.data_type(), &PrimitiveType::Decimal {
        precision: 1,
        scale: 0,
    });
    assert_eq!(converted.literal(), &PrimitiveLiteral::Int128(0));
}

#[test]
fn test_datum_to_decimal_accepts_negative_mantissa() {
    let target_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 2,
        scale: 1,
    });
    let datum = Datum::decimal_from_str("-1.5").unwrap();

    let converted = datum.to(&target_type).unwrap();

    assert_eq!(converted.data_type(), &PrimitiveType::Decimal {
        precision: 2,
        scale: 1,
    });
    assert_eq!(converted.literal(), &PrimitiveLiteral::Int128(-15));
}

#[test]
fn test_datum_to_decimal_rejects_precision_too_narrow() {
    let target_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 1,
        scale: 1,
    });
    let datum = Datum::decimal_from_str("1.5").unwrap();

    let result = datum.to(&target_type);

    assert!(result.is_err(), "expect error but got {result:?}");
    assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
}

#[test]
fn test_datum_to_decimal_rejects_value_that_fits_storage_bytes_but_not_precision() {
    let target_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 1,
        scale: 1,
    });
    let datum = Datum::decimal_from_str("4.2").unwrap();

    let result = datum.to(&target_type);

    assert!(result.is_err(), "expect error but got {result:?}");
    assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
}

#[test]
fn test_datum_to_decimal_accepts_single_digit_mantissa_for_precision_one() {
    let target_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 1,
        scale: 1,
    });
    let datum = Datum::decimal_from_str("0.5").unwrap();

    let converted = datum.to(&target_type).unwrap();

    assert_eq!(converted.data_type(), &PrimitiveType::Decimal {
        precision: 1,
        scale: 1,
    });
    assert_eq!(converted.literal(), &PrimitiveLiteral::Int128(5));
}

#[test]
fn test_datum_decimal_with_precision_rejects_value_that_exceeds_digit_precision() {
    let result = Datum::decimal_with_precision(decimal_from_i128_with_scale(42, 2), 1);

    assert!(result.is_err(), "expect error but got {result:?}");
    assert_eq!(result.unwrap_err().kind(), ErrorKind::DataInvalid);
}

#[test]
fn test_datum_decimal_with_precision_accepts_value_that_fits_digit_precision() {
    let datum = Datum::decimal_with_precision(decimal_from_i128_with_scale(5, 1), 1).unwrap();

    assert_eq!(datum.data_type(), &PrimitiveType::Decimal {
        precision: 1,
        scale: 1,
    });
    assert_eq!(datum.literal(), &PrimitiveLiteral::Int128(5));
}

#[test]
fn test_datum_to_decimal_rejects_scale_change() {
    let target_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 9,
        scale: 3,
    });
    let datum = Datum::decimal_from_str("123.45").unwrap();

    let result = datum.to(&target_type);
    assert!(result.is_err(), "expect error but got {result:?}");

    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.to_string()
            .contains("Decimal scale conversion is not supported")
    );
}
