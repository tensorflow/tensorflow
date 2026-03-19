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

using testing::ElementsAre;

namespace {

TEST(PjRtCommonTest, PjRtValueTypeFromProtoDefault) {
  xla::PjRtValueTypeProto proto;
  xla::PjRtValueType value = xla::PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<std::string>(value));
  EXPECT_EQ(std::get<std::string>(value), "");
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoFloat) {
  xla::PjRtValueType value = 1.0f;
  xla::PjRtValueTypeProto proto = xla::PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_float_value());
  EXPECT_EQ(proto.float_value(), 1.0f);
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoInt) {
  xla::PjRtValueType value = 1L;
  xla::PjRtValueTypeProto proto = xla::PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_int_value());
  EXPECT_EQ(proto.int_value(), 1L);
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoString) {
  xla::PjRtValueType value = std::string("test");
  xla::PjRtValueTypeProto proto = xla::PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_string_value());
  EXPECT_EQ(proto.string_value(), "test");
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoBool) {
  xla::PjRtValueType value = true;
  xla::PjRtValueTypeProto proto = xla::PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_bool_value());
  EXPECT_EQ(proto.bool_value(), true);
}

TEST(PjRtCommonTest, PjRtValueTypeToProtoIntVector) {
  xla::PjRtValueType value = std::vector<int64_t>{1, 2, 3};
  xla::PjRtValueTypeProto proto = xla::PjRtValueTypeToProto(value);
  ASSERT_TRUE(proto.has_int_vector());
  EXPECT_THAT(proto.int_vector().values(), ElementsAre(1, 2, 3));
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoFloat) {
  xla::PjRtValueTypeProto proto;
  proto.set_float_value(1.0f);
  xla::PjRtValueType value = xla::PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<float>(value));
  EXPECT_EQ(std::get<float>(value), 1.0f);
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoInt) {
  xla::PjRtValueTypeProto proto;
  proto.set_int_value(1L);
  xla::PjRtValueType value = xla::PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<int64_t>(value));
  EXPECT_EQ(std::get<int64_t>(value), 1L);
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoString) {
  xla::PjRtValueTypeProto proto;
  proto.set_string_value("test");
  xla::PjRtValueType value = xla::PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<std::string>(value));
  EXPECT_EQ(std::get<std::string>(value), "test");
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoBool) {
  xla::PjRtValueTypeProto proto;
  proto.set_bool_value(true);
  xla::PjRtValueType value = xla::PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<bool>(value));
  EXPECT_EQ(std::get<bool>(value), true);
}

TEST(PjRtCommonTest, PjRtValueTypeFromProtoIntVector) {
  xla::PjRtValueTypeProto proto;
  proto.mutable_int_vector()->add_values(1);
  proto.mutable_int_vector()->add_values(2);
  proto.mutable_int_vector()->add_values(3);
  xla::PjRtValueType value = xla::PjRtValueTypeFromProto(proto);
  ASSERT_TRUE(std::holds_alternative<std::vector<int64_t>>(value));
  EXPECT_THAT(std::get<std::vector<int64_t>>(value),
              testing::ElementsAre(1, 2, 3));
}

}  // namespace
