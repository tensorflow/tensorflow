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

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsic {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::xla::codegen::intrinsics::Type;

TEST(IntrinsicTest, TypeName) {
  EXPECT_EQ(Type::S(F32).name(), "f32");
  EXPECT_EQ(Type::V(F32, 4).name(), "v4f32");
  EXPECT_EQ(Type::V(S8, 16).name(), "v16i8");
  EXPECT_EQ(Type::V(U8, 2).name(), "v2u8");
}

TEST(IntrinsicTest, TypeElementType) {
  EXPECT_EQ(Type::S(F32).element_type(), F32);
  EXPECT_EQ(Type::V(F32, 4).element_type(), F32);
}

TEST(IntrinsicTest, TypeVectorWidth) {
  EXPECT_EQ(Type::S(F32).vector_width(), std::nullopt);
  EXPECT_EQ(Type::V(F32, 4).vector_width(), 4);
}

TEST(IntrinsicTest, VerifySameWidth) {
  EXPECT_OK(Type::VerifySameWidth(Type::S(F32), Type::S(F32)));
  EXPECT_OK(Type::VerifySameWidth(Type::V(F32, 4), Type::V(F32, 4)));
  EXPECT_THAT(
      Type::VerifySameWidth(Type::S(F32), Type::V(F32, 4)),
      absl_testing::StatusIs(_, HasSubstr("Expected types of the same kind")));
  EXPECT_THAT(Type::VerifySameWidth(Type::V(F32, 2), Type::V(F32, 4)),
              absl_testing::StatusIs(_, HasSubstr("Expected vector types")));
}

TEST(IntrinsicTest, VerifySameWidthAndElementType) {
  EXPECT_OK(Type::VerifySameWidthAndElementType(Type::S(F32), Type::S(F32)));
  EXPECT_OK(
      Type::VerifySameWidthAndElementType(Type::V(F32, 4), Type::V(F32, 4)));
  EXPECT_THAT(
      Type::VerifySameWidthAndElementType(Type::S(F32), Type::V(F32, 4)),
      absl_testing::StatusIs(_, HasSubstr("Expected types of the same kind")));
  EXPECT_THAT(
      Type::VerifySameWidthAndElementType(Type::V(F32, 2), Type::V(F32, 4)),
      absl_testing::StatusIs(_, HasSubstr("Expected vector types")));
}

TEST(IntrinsicTypeTest, FromName) {
  Type f32 = Type::FromName("f32");
  EXPECT_TRUE(f32.is_scalar());
  EXPECT_EQ(f32.element_type(), F32);
  EXPECT_EQ(f32.vector_width(), std::nullopt);
  Type v4s8 = Type::FromName("v4s8");
  EXPECT_TRUE(v4s8.is_vector());
  EXPECT_EQ(v4s8.element_type(), S8);
  EXPECT_EQ(v4s8.vector_width(), 4);
}

}  // namespace
}  // namespace xla::codegen::intrinsic
