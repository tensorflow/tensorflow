/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/ffi/attribute_map.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/ffi/attribute_map.pb.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::ffi {
namespace {
using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using ::testing::HasSubstr;
using tsl::proto_testing::EqualsProto;
using tsl::proto_testing::ParseTextProtoOrDie;

TEST(ScalarTest, ProtoConversion) {
  EXPECT_THAT(Scalar(true).ToProto(), EqualsProto(R"pb(
                b: 1
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                b: 1
              )pb")),
              IsOkAndHolds(Scalar(true)));

  EXPECT_THAT(Scalar(int8_t{42}).ToProto(), EqualsProto(R"pb(
                i8: 42
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                i8: 42
              )pb")),
              IsOkAndHolds(Scalar(int8_t{42})));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                i8: 128
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Integer value out of range for int8_t")));

  EXPECT_THAT(Scalar(int16_t{42}).ToProto(), EqualsProto(R"pb(
                i16: 42
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                i16: 42
              )pb")),
              IsOkAndHolds(Scalar(int16_t{42})));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                i16: 32768
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Integer value out of range for int16_t")));

  EXPECT_THAT(Scalar(int32_t{42}).ToProto(), EqualsProto(R"pb(
                i32: 42
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                i32: 42
              )pb")),
              IsOkAndHolds(Scalar(int32_t{42})));

  EXPECT_THAT(Scalar(int64_t{42}).ToProto(), EqualsProto(R"pb(
                i64: 42
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                i64: 42
              )pb")),
              IsOkAndHolds(Scalar(int64_t{42})));

  EXPECT_THAT(Scalar(uint8_t{42}).ToProto(), EqualsProto(R"pb(
                u8: 42
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                u8: 42
              )pb")),
              IsOkAndHolds(Scalar(uint8_t{42})));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                u8: 256
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Integer value out of range for uint8_t")));

  EXPECT_THAT(Scalar(uint16_t{42}).ToProto(), EqualsProto(R"pb(
                u16: 42
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                u16: 42
              )pb")),
              IsOkAndHolds(Scalar(uint16_t{42})));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                u16: 65536
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Integer value out of range for uint16_t")));

  EXPECT_THAT(Scalar(uint32_t{42}).ToProto(), EqualsProto(R"pb(
                u32: 42
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                u32: 42
              )pb")),
              IsOkAndHolds(Scalar(uint32_t{42})));

  EXPECT_THAT(Scalar(uint64_t{42}).ToProto(), EqualsProto(R"pb(
                u64: 42
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                u64: 42
              )pb")),
              IsOkAndHolds(Scalar(uint64_t{42})));

  EXPECT_THAT(Scalar(float{42.0f}).ToProto(), EqualsProto(R"pb(
                f32: 42.0
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                f32: 42.0
              )pb")),
              IsOkAndHolds(Scalar(float{42.0f})));

  EXPECT_THAT(Scalar(double{42.0}).ToProto(), EqualsProto(R"pb(
                f64: 42.0
              )pb"));
  EXPECT_THAT(Scalar::FromProto(ParseTextProtoOrDie<ScalarProto>(R"pb(
                f64: 42.0
              )pb")),
              IsOkAndHolds(Scalar(double{42.0})));
}

TEST(ArrayTest, ProtoConversion) {
  EXPECT_THAT(Array(std::vector<int8_t>{42, 43}).ToProto(), EqualsProto(R"pb(
                i8: { values: 42 values: 43 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                i8: { values: 42 values: 43 }
              )pb")),
              IsOkAndHolds(Array(std::vector<int8_t>{42, 43})));

  EXPECT_THAT(Array(std::vector<int16_t>{42, 43}).ToProto(), EqualsProto(R"pb(
                i16: { values: 42 values: 43 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                i16: { values: 42 values: 43 }
              )pb")),
              IsOkAndHolds(Array(std::vector<int16_t>{42, 43})));

  EXPECT_THAT(Array(std::vector<int32_t>{42, 43}).ToProto(), EqualsProto(R"pb(
                i32: { values: 42 values: 43 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                i32: { values: 42 values: 43 }
              )pb")),
              IsOkAndHolds(Array(std::vector<int32_t>{42, 43})));

  EXPECT_THAT(Array(std::vector<int64_t>{42, 43}).ToProto(), EqualsProto(R"pb(
                i64: { values: 42 values: 43 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                i64: { values: 42 values: 43 }
              )pb")),
              IsOkAndHolds(Array(std::vector<int64_t>{42, 43})));

  EXPECT_THAT(Array(std::vector<uint8_t>{42, 43}).ToProto(), EqualsProto(R"pb(
                u8: { values: 42 values: 43 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                u8: { values: 42 values: 43 }
              )pb")),
              IsOkAndHolds(Array(std::vector<uint8_t>{42, 43})));

  EXPECT_THAT(Array(std::vector<uint16_t>{42, 43}).ToProto(), EqualsProto(R"pb(
                u16: { values: 42 values: 43 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                u16: { values: 42 values: 43 }
              )pb")),
              IsOkAndHolds(Array(std::vector<uint16_t>{42, 43})));

  EXPECT_THAT(Array(std::vector<uint32_t>{42, 43}).ToProto(), EqualsProto(R"pb(
                u32: { values: 42 values: 43 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                u32: { values: 42 values: 43 }
              )pb")),
              IsOkAndHolds(Array(std::vector<uint32_t>{42, 43})));

  EXPECT_THAT(Array(std::vector<uint64_t>{42, 43}).ToProto(), EqualsProto(R"pb(
                u64: { values: 42 values: 43 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                u64: { values: 42 values: 43 }
              )pb")),
              IsOkAndHolds(Array(std::vector<uint64_t>{42, 43})));

  EXPECT_THAT(Array(std::vector<float>{42.0f, 43.0f}).ToProto(),
              EqualsProto(R"pb(
                f32: { values: 42.0 values: 43.0 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                f32: { values: 42.0 values: 43.0 }
              )pb")),
              IsOkAndHolds(Array(std::vector<float>{42.0f, 43.0f})));

  EXPECT_THAT(Array(std::vector<double>{42.0, 43.0}).ToProto(),
              EqualsProto(R"pb(
                f64: { values: 42.0 values: 43.0 }
              )pb"));
  EXPECT_THAT(Array::FromProto(ParseTextProtoOrDie<ArrayProto>(R"pb(
                f64: { values: 42.0 values: 43.0 }
              )pb")),
              IsOkAndHolds(Array(std::vector<double>{42.0, 43.0})));
}

TEST(FlatAttributeTest, ProtoConversion) {
  EXPECT_THAT(FlatAttribute(Scalar(true)).ToProto(), EqualsProto(R"pb(
                scalar: { b: 1 }
              )pb"));
  EXPECT_THAT(FlatAttribute::FromProto(ParseTextProtoOrDie<FlatAttributeProto>(
                  R"pb(
                    scalar: { b: 1 }
                  )pb")),
              IsOkAndHolds(FlatAttribute(Scalar(true))));
  EXPECT_THAT(FlatAttribute(Array(std::vector<int8_t>{42, 43})).ToProto(),
              EqualsProto(R"pb(
                array: { i8: { values: 42 values: 43 } }
              )pb"));
  EXPECT_THAT(FlatAttribute::FromProto(ParseTextProtoOrDie<FlatAttributeProto>(
                  R"pb(
                    array: { i8: { values: 42 values: 43 } }
                  )pb")),
              IsOkAndHolds(FlatAttribute(Array(std::vector<int8_t>{42, 43}))));

  EXPECT_THAT(FlatAttribute(std::string("foo")).ToProto(), EqualsProto(R"pb(
                str: "foo"
              )pb"));
  EXPECT_THAT(FlatAttribute::FromProto(ParseTextProtoOrDie<FlatAttributeProto>(
                  R"pb(
                    str: "foo"
                  )pb")),
              IsOkAndHolds(FlatAttribute(std::string("foo"))));
}

TEST(AttributesMapTest, ProtoConversion) {
  AttributesMap attrs = {{std::string("foo"), Attribute(Scalar(true))}};

  EXPECT_THAT(AttributesMap::FromProto(attrs.ToProto()), IsOkAndHolds(attrs));
}

TEST(DictionaryTest, ProtoConversion) {
  AttributesMap attrs = {{std::string("foo"), Attribute(Scalar(true))}};
  AttributesDictionary dict{};
  dict.attrs = std::make_shared<AttributesMap>(attrs);

  EXPECT_THAT(AttributesDictionary::FromProto(dict.ToProto()),
              IsOkAndHolds(dict));
}

TEST(AttributeTest, ProtoConversion) {
  AttributesDictionary dict{};
  dict.attrs = std::make_shared<AttributesMap>(
      AttributesMap{{std::string("foo"), Attribute(Scalar(true))}});
  Attribute attr(std::move(dict));

  EXPECT_THAT(Attribute::FromProto(attr.ToProto()), IsOkAndHolds(attr));
}

TEST(AttributesMapTest, ScalarConstruction) {
  AttributesMap attrs = {
      {"i32", 42},
      {"f32", 42.0f},
      {"f64", 42.0},
      {"b", true},
  };

  EXPECT_EQ(attrs.size(), 4);
  auto i32 = std::get<Scalar>(attrs.at("i32").AsVariant());
  auto f32 = std::get<Scalar>(attrs.at("f32").AsVariant());
  auto f64 = std::get<Scalar>(attrs.at("f64").AsVariant());
  auto b = std::get<Scalar>(attrs.at("b").AsVariant());
  EXPECT_EQ(i32, Scalar(int32_t{42}));
  EXPECT_EQ(f32, Scalar(42.0f));
  EXPECT_EQ(f64, Scalar(42.0));
  EXPECT_EQ(b, Scalar(true));
}

TEST(AttributesMapTest, StringConstruction) {
  AttributesMap attrs = {
      {"str_literal", "hello"},
      {"str_obj", std::string("world")},
  };

  EXPECT_EQ(attrs.size(), 2);
  auto str_literal = std::get<std::string>(attrs.at("str_literal").AsVariant());
  auto str_obj = std::get<std::string>(attrs.at("str_obj").AsVariant());
  EXPECT_EQ(str_literal, "hello");
  EXPECT_EQ(str_obj, "world");
}

TEST(AttributesMapTest, ArrayConstruction) {
  AttributesMap attrs = {
      {"i32_arr", {1, 2, 3}},
      {"f32_arr", {1.0f, 2.0f, 3.0f}},
  };

  EXPECT_EQ(attrs.size(), 2);
  auto i32_arr = std::get<Array>(attrs.at("i32_arr").AsVariant());
  auto f32_arr = std::get<Array>(attrs.at("f32_arr").AsVariant());
  EXPECT_EQ(i32_arr, Array(std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(f32_arr, Array(std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(AttributesMapTest, NestedConstruction) {
  AttributesMap attrs = {
      {"i32", 42},
      {"nested", {{"f32", 1.0f}, {"str", "foo"}}},
  };

  EXPECT_EQ(attrs.size(), 2);
  auto i32 = std::get<Scalar>(attrs.at("i32").AsVariant());
  EXPECT_EQ(i32, Scalar(int32_t{42}));

  auto& dict = std::get<AttributesDictionary>(attrs.at("nested").AsVariant());
  ASSERT_NE(dict.attrs, nullptr);
  EXPECT_EQ(dict.attrs->size(), 2);
  auto f32 = std::get<Scalar>(dict.attrs->at("f32").AsVariant());
  auto str = std::get<std::string>(dict.attrs->at("str").AsVariant());
  EXPECT_EQ(f32, Scalar(1.0f));
  EXPECT_EQ(str, "foo");
}

TEST(AttributesMapTest, DeeplyNestedConstruction) {
  AttributesMap attrs = {
      {"level1", {{"level2", {{"value", 123}}}}},
  };

  auto& l1 = std::get<AttributesDictionary>(attrs.at("level1").AsVariant());
  ASSERT_NE(l1.attrs, nullptr);
  auto& l2 = std::get<AttributesDictionary>(l1.attrs->at("level2").AsVariant());
  ASSERT_NE(l2.attrs, nullptr);
  auto value = std::get<Scalar>(l2.attrs->at("value").AsVariant());
  EXPECT_EQ(value, Scalar(int32_t{123}));
}

TEST(AttributesMapTest, EmptyConstruction) {
  AttributesMap attrs = {};
  EXPECT_TRUE(attrs.empty());
}

TEST(AttributesMapTest, ConstructionProtoRoundTrip) {
  AttributesMap attrs = {
      {"i32", 42},
      {"str", "hello"},
      {"nested", {{"f32", 1.0f}}},
  };

  EXPECT_THAT(AttributesMap::FromProto(attrs.ToProto()), IsOkAndHolds(attrs));
}

}  // namespace
}  // namespace xla::ffi
