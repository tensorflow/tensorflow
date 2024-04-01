/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/shlo/f16.h"

#include <cstdint>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"

namespace shlo_ref {
namespace {

using ::testing::FloatNear;

using RoundtripTypeList = ::testing::Types<float, double>;

template <class T>
struct RoundtripF16Test : testing::Test {};

TYPED_TEST_SUITE(RoundtripF16Test, RoundtripTypeList);

TYPED_TEST(RoundtripF16Test, RoundtripConversions) {
  for (TypeParam value : {
           -std::numeric_limits<TypeParam>::infinity(),
           std::numeric_limits<TypeParam>::infinity(),
           TypeParam(-1.0),
           TypeParam(-0.5),
           TypeParam(-0.0),
           TypeParam(1.0),
           TypeParam(0.5),
           TypeParam(0.0),
       }) {
    EXPECT_EQ(value, static_cast<TypeParam>(static_cast<F16>(value)));
  }
}

TEST(F16Test, Arithmetic) {
  EXPECT_EQ(static_cast<float>(F16(2) + F16(2)), 4);
  EXPECT_EQ(static_cast<float>(F16(2) + F16(-2)), 0);
  EXPECT_THAT(static_cast<float>(F16(0.33333f) + F16(0.66667f)),
              FloatNear(1.0f, 1e-3));
  EXPECT_EQ(static_cast<float>(F16(2.0f) * F16(-5.5f)), -11.0f);
  EXPECT_THAT(static_cast<float>(F16(1.0f) / F16(3.0f)),
              FloatNear(0.3339f, 1e-3));
  EXPECT_EQ(static_cast<float>(-F16(4096.0f)), -4096.0f);
  EXPECT_EQ(static_cast<float>(-F16(-4096.0f)), 4096.0f);
}

TEST(F16Test, DefaultConstruct) { EXPECT_EQ(static_cast<float>(F16()), 0.0f); }

TEST(F16Test, ImplicitConversionToFloat) {
  EXPECT_EQ((absl::bit_cast<F16, uint16_t>(0x0000)), 0.0f);
  EXPECT_EQ((absl::bit_cast<F16, uint16_t>(0x3C00)), 1.0f);
}

}  // namespace
}  // namespace shlo_ref
