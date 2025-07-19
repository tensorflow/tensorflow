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

#include "xla/codegen/math/intrinsic.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::math {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;
using ::xla::codegen::intrinsics::Type;

TEST(IntrinsicTest, TypeName) {
  EXPECT_EQ(Type::S(F32).name(), "f32");
  EXPECT_EQ(Type::V(F32, 4).name(), "v4f32");
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
  TF_EXPECT_OK(Type::VerifySameWidth(Type::S(F32), Type::S(F32)));
  TF_EXPECT_OK(Type::VerifySameWidth(Type::V(F32, 4), Type::V(F32, 4)));
  EXPECT_THAT(Type::VerifySameWidth(Type::S(F32), Type::V(F32, 4)),
              StatusIs(_, HasSubstr("Expected types of the same kind")));
  EXPECT_THAT(Type::VerifySameWidth(Type::V(F32, 2), Type::V(F32, 4)),
              StatusIs(_, HasSubstr("Expected vector types")));
}

TEST(IntrinsicTest, VerifySameWidthAndElementType) {
  TF_EXPECT_OK(Type::VerifySameWidthAndElementType(Type::S(F32), Type::S(F32)));
  TF_EXPECT_OK(
      Type::VerifySameWidthAndElementType(Type::V(F32, 4), Type::V(F32, 4)));
  EXPECT_THAT(
      Type::VerifySameWidthAndElementType(Type::S(F32), Type::V(F32, 4)),
      StatusIs(_, HasSubstr("Expected types of the same kind")));
  EXPECT_THAT(
      Type::VerifySameWidthAndElementType(Type::V(F32, 2), Type::V(F32, 4)),
      StatusIs(_, HasSubstr("Expected vector types")));
}

}  // namespace
}  // namespace xla::codegen::math
