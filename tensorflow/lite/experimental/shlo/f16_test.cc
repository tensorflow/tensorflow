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

TEST(F16Test, ConstructFromArithmeticType) {
  const F16 from_int8(static_cast<int8_t>(1));
  EXPECT_EQ(static_cast<float>(from_int8), 1);
  const F16 from_int16(static_cast<int16_t>(1));
  EXPECT_EQ(static_cast<float>(from_int16), 1);
  const F16 from_int32(static_cast<int32_t>(1));
  EXPECT_EQ(static_cast<float>(from_int32), 1);
  const F16 from_int64(static_cast<int64_t>(1));
  EXPECT_EQ(static_cast<float>(from_int64), 1);
  const F16 from_float(static_cast<float>(1));
  EXPECT_EQ(static_cast<float>(from_float), 1);
  const F16 from_double(static_cast<double>(1));
  EXPECT_EQ(static_cast<float>(from_double), 1);
}

template <class T>
T ImplicitConversion(T v) {
  return v;
}

TEST(F16Test, ConvertToArithmeticType) {
  const F16 ref(-1);
  EXPECT_EQ(ImplicitConversion<int8_t>(ref), -1);
  EXPECT_EQ(ImplicitConversion<int16_t>(ref), -1);
  EXPECT_EQ(ImplicitConversion<int32_t>(ref), -1);
  EXPECT_EQ(ImplicitConversion<int64_t>(ref), -1);
  EXPECT_EQ(ImplicitConversion<float>(ref), -1);
  EXPECT_EQ(ImplicitConversion<double>(ref), -1);
}

TEST(F16Test, ArithmeticOperations) {
  // Every test relies on the equality comparisons working. We test all the 4
  // bit integral values.
  for (int i = -8; i < 8; ++i) {
    for (int j = -8; j < 8; ++j) {
      EXPECT_EQ(F16(i) == F16(j), i == j);
      EXPECT_EQ(F16(i) != F16(j), i != j);
      EXPECT_EQ(F16(i) > F16(j), i > j);
      EXPECT_EQ(F16(i) >= F16(j), i >= j);
      EXPECT_EQ(F16(i) < F16(j), i < j);
      EXPECT_EQ(F16(i) <= F16(j), i <= j);
    }
  }
  F16 val(0);
  EXPECT_EQ(++val, 1);
  EXPECT_EQ(val++, 1);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val--, 2);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(--val, 0);
  EXPECT_EQ(val += F16(1), 1);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(val *= F16(2), 2);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val /= F16(2), 1);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(val -= F16(4), -3);
  EXPECT_EQ(val, -3);
  EXPECT_EQ(val = F16(7), 7);
  EXPECT_EQ(val, 7);
  EXPECT_EQ(+val, 7);
  EXPECT_EQ(-val, -7);
  EXPECT_EQ(static_cast<bool>(val), true);
  EXPECT_EQ(!val, false);
  EXPECT_EQ(val && F16(2), true);
  EXPECT_EQ(val && F16(0), false);
  EXPECT_EQ(val || F16(0), true);
  EXPECT_EQ(F16(0) || F16(0), false);
}

using ArithmeticTypeList =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

template <class T>
struct ArithmeticTypeF16Test : testing::Test {};

TYPED_TEST_SUITE(ArithmeticTypeF16Test, ArithmeticTypeList);

TYPED_TEST(ArithmeticTypeF16Test, InPlaceArithmetic) {
  // Every test relies on the equality comparisons working. We test all the 4
  // bit integral values.
  for (TypeParam i = -8; i < 8; ++i) {
    for (TypeParam j = -8; j < 8; ++j) {
      EXPECT_EQ(F16(i) == j, i == j);
      EXPECT_EQ(i == F16(j), i == j);
      EXPECT_EQ(F16(i) != j, i != j);
      EXPECT_EQ(i != F16(j), i != j);
      EXPECT_EQ(F16(i) > j, i > j);
      EXPECT_EQ(i > F16(j), i > j);
      EXPECT_EQ(F16(i) >= j, i >= j);
      EXPECT_EQ(i >= F16(j), i >= j);
      EXPECT_EQ(F16(i) < j, i < j);
      EXPECT_EQ(i < F16(j), i < j);
      EXPECT_EQ(F16(i) <= j, i <= j);
      EXPECT_EQ(i <= F16(j), i <= j);
    }
  }
  const TypeParam one = TypeParam(1);
  const TypeParam two = TypeParam(2);
  const TypeParam four = TypeParam(4);
  F16 val(0);
  EXPECT_EQ(val += one, 1);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(val *= two, 2);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val /= two, 1);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(val -= four, -3);
  EXPECT_EQ(val, -3);
  const F16 f16_three(3);
  EXPECT_EQ(f16_three + one, 4.);
  EXPECT_EQ(f16_three - one, 2.);
  EXPECT_EQ(f16_three * two, 3. * two);
  EXPECT_EQ(f16_three / two, 3. / two);
}

}  // namespace
}  // namespace shlo_ref
