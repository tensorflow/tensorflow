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

#include "xla/pjrt/pjrt_common.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/proto/pjrt_value_type.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(PjRtCommonTest, PjRtValueTypeFromProtoDefault) {
  PjRtValueTypeProto proto;
  PjRtValueType value = PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<std::string>(value));
  EXPECT_EQ(std::get<std::string>(value), "");
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoFloat) {
  PjRtValueType value = 1.0f;
  PjRtValueTypeProto proto = PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_float_value());
  EXPECT_EQ(proto.float_value(), 1.0f);
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoInt) {
  PjRtValueType value = 1L;
  PjRtValueTypeProto proto = PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_int_value());
  EXPECT_EQ(proto.int_value(), 1L);
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoString) {
  PjRtValueType value = std::string("test");
  PjRtValueTypeProto proto = PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_string_value());
  EXPECT_EQ(proto.string_value(), "test");
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoBool) {
  PjRtValueType value = true;
  PjRtValueTypeProto proto = PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_bool_value());
  EXPECT_EQ(proto.bool_value(), true);
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoIntVector) {
  PjRtValueType value = std::vector<int64_t>{1, 2, 3};
  PjRtValueTypeProto proto = PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_int_vector());
  EXPECT_THAT(proto.int_vector().values(), ElementsAre(1, 2, 3));
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoFloat) {
  PjRtValueTypeProto proto;
  proto.set_float_value(1.0f);
  PjRtValueType value = PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<float>(value));
  EXPECT_EQ(std::get<float>(value), 1.0f);
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoInt) {
  PjRtValueTypeProto proto;
  proto.set_int_value(1L);
  PjRtValueType value = PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<int64_t>(value));
  EXPECT_EQ(std::get<int64_t>(value), 1L);
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoString) {
  PjRtValueTypeProto proto;
  proto.set_string_value("test");
  PjRtValueType value = PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<std::string>(value));
  EXPECT_EQ(std::get<std::string>(value), "test");
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoBool) {
  PjRtValueTypeProto proto;
  proto.set_bool_value(true);
  PjRtValueType value = PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<bool>(value));
  EXPECT_EQ(std::get<bool>(value), true);
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoIntVector) {
  PjRtValueTypeProto proto;
  proto.mutable_int_vector()->add_values(1);
  proto.mutable_int_vector()->add_values(2);
  proto.mutable_int_vector()->add_values(3);
  PjRtValueType value = PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<std::vector<int64_t>>(value));
  EXPECT_THAT(std::get<std::vector<int64_t>>(value), ElementsAre(1, 2, 3));
}

TEST(PjRtCommonTest, PjRtValueTypeCaseName) {
  EXPECT_EQ(PjRtValueTypeCaseName(PjRtValueTypeProto::kBoolValue), "bool");
  EXPECT_EQ(PjRtValueTypeCaseName(PjRtValueTypeProto::kFloatValue), "float");
  EXPECT_EQ(PjRtValueTypeCaseName(PjRtValueTypeProto::kIntValue), "int64_t");
  EXPECT_EQ(PjRtValueTypeCaseName(PjRtValueTypeProto::kIntVector),
            "std::vector<int64_t>");
  EXPECT_EQ(PjRtValueTypeCaseName(PjRtValueTypeProto::kStringValue), "string");
  EXPECT_EQ(PjRtValueTypeCaseName(PjRtValueTypeProto::VALUE_NOT_SET), "unset");
}

TEST(PjRtCommonTest, PjRtTypeName) {
  EXPECT_EQ(PjRtTypeName<bool>(), "bool");
  EXPECT_EQ(PjRtTypeName<float>(), "float");
  EXPECT_EQ(PjRtTypeName<int64_t>(), "int64_t");
  EXPECT_EQ(PjRtTypeName<std::vector<int64_t>>(), "std::vector<int64_t>");
  EXPECT_EQ(PjRtTypeName<std::string>(), "string");
  EXPECT_EQ(PjRtTypeName<int>(), "unknown");
}

}  // namespace
}  // namespace xla
