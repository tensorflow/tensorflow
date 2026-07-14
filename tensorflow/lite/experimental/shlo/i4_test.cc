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

#include "tensorflow/lite/experimental/shlo/i4.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace shlo_ref {
namespace {

TEST(I4Test, ConstructFromArithmeticType) {
  const I4 from_int8(static_cast<int8_t>(1));
  EXPECT_EQ(from_int8.data, 1);
  const I4 from_int16(static_cast<int16_t>(1));
  EXPECT_EQ(from_int16.data, 1);
  const I4 from_int32(static_cast<int32_t>(1));
  EXPECT_EQ(from_int32.data, 1);
  const I4 from_int64(static_cast<int64_t>(1));
  EXPECT_EQ(from_int64.data, 1);
  const I4 from_float(static_cast<float>(1));
  EXPECT_EQ(from_float.data, 1);
  const I4 from_double(static_cast<double>(1));
  EXPECT_EQ(from_double.data, 1);
}

template <class T>
T ImplicitConversion(T v) {
  return v;
}

TEST(I4Test, ConvertToArithmeticType) {
  const I4 ref(-1);
  EXPECT_EQ(ImplicitConversion<int8_t>(ref), -1);
  EXPECT_EQ(ImplicitConversion<int16_t>(ref), -1);
  EXPECT_EQ(ImplicitConversion<int32_t>(ref), -1);
  EXPECT_EQ(ImplicitConversion<int64_t>(ref), -1);
  EXPECT_EQ(ImplicitConversion<float>(ref), -1);
  EXPECT_EQ(ImplicitConversion<double>(ref), -1);
}

TEST(I4Test, Arithmetic) {
  // Every test relies on the equality comparisons working. We test all the 4
  // bit integral values.
  for (int i = -8; i < 8; ++i) {
    for (int j = -8; j < 8; ++j) {
      EXPECT_EQ(I4(i) == I4(j), i == j);
      EXPECT_EQ(I4(i) != I4(j), i != j);
      EXPECT_EQ(I4(i) > I4(j), i > j);
      EXPECT_EQ(I4(i) >= I4(j), i >= j);
      EXPECT_EQ(I4(i) < I4(j), i < j);
      EXPECT_EQ(I4(i) <= I4(j), i <= j);
    }
  }
  I4 val(0);
  EXPECT_EQ(++val, 1);
  EXPECT_EQ(val++, 1);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val--, 2);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(--val, 0);
  EXPECT_EQ(val += I4(1), 1);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(val *= I4(2), 2);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val /= I4(2), 1);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(val -= I4(4), -3);
  EXPECT_EQ(val, -3);
  EXPECT_EQ(val %= I4(2), -1);
  EXPECT_EQ(val, -1);
  EXPECT_EQ(val = I4(7), 7);
  EXPECT_EQ(val, 7);
  EXPECT_EQ(val &= I4(2), 2);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val |= I4(1), 3);
  EXPECT_EQ(val, 3);
  EXPECT_EQ(val ^= I4(7), 4);
  EXPECT_EQ(val, 4);
  EXPECT_EQ(val >>= I4(1), 2);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val <<= I4(1), 4);
  EXPECT_EQ(val, 4);
  EXPECT_EQ(val >>= I4(1), 2);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val <<= I4(1), 4);
  EXPECT_EQ(val, 4);
  EXPECT_EQ(+val, 4);
  EXPECT_EQ(-val, -4);
  EXPECT_EQ(!val, false);
  EXPECT_EQ(~val, ~4);
  EXPECT_EQ(val && I4(2), true);
  EXPECT_EQ(val && I4(0), false);
  EXPECT_EQ(val || I4(0), true);
  EXPECT_EQ(I4(0) || I4(0), false);
}

using IntegralTypeList = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;

using ArithmeticTypeList =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

template <class T>
struct ArithmeticTypeI4Test : testing::Test {};

TYPED_TEST_SUITE(ArithmeticTypeI4Test, ArithmeticTypeList);

TYPED_TEST(ArithmeticTypeI4Test, Arithmetic) {
  // Every test relies on the equality comparisons working. We test all the 4
  // bit integral values.
  for (TypeParam i = -8; i < 8; ++i) {
    for (TypeParam j = -8; j < 8; ++j) {
      EXPECT_EQ(I4(i) == j, i == j);
      EXPECT_EQ(i == I4(j), i == j);
      EXPECT_EQ(I4(i) != j, i != j);
      EXPECT_EQ(i != I4(j), i != j);
      EXPECT_EQ(I4(i) > j, i > j);
      EXPECT_EQ(i > I4(j), i > j);
      EXPECT_EQ(I4(i) >= j, i >= j);
      EXPECT_EQ(i >= I4(j), i >= j);
      EXPECT_EQ(I4(i) < j, i < j);
      EXPECT_EQ(i < I4(j), i < j);
      EXPECT_EQ(I4(i) <= j, i <= j);
      EXPECT_EQ(i <= I4(j), i <= j);
    }
  }
  I4 val(0);
  const TypeParam one = TypeParam(1);
  const TypeParam two = TypeParam(2);
  const TypeParam three = TypeParam(3);
  const TypeParam four = TypeParam(4);
  EXPECT_EQ(val += one, 1);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(val *= two, 2);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val /= two, 1);
  EXPECT_EQ(val, 1);
  EXPECT_EQ(val -= four, -3);
  EXPECT_EQ(val, -3);
  const I4 i4_three(3);
  EXPECT_EQ(i4_three + one, four);
  EXPECT_EQ(i4_three - one, two);
  EXPECT_EQ(i4_three * two, three * two);
  EXPECT_EQ(i4_three / two, three / two);
}

template <class T>
struct IntegralTypeI4Test : testing::Test {};

TYPED_TEST_SUITE(IntegralTypeI4Test, IntegralTypeList);

TYPED_TEST(IntegralTypeI4Test, Arithmetic) {
  const TypeParam minus_one = TypeParam(-1);
  const TypeParam one = TypeParam(1);
  const TypeParam two = TypeParam(2);
  const TypeParam three = TypeParam(3);
  const TypeParam four = TypeParam(4);
  const TypeParam six = TypeParam(6);
  const TypeParam seven = TypeParam(7);
  const I4 i4_three(3);
  EXPECT_EQ(i4_three % two, one);
  EXPECT_EQ(i4_three & two, two);
  EXPECT_EQ(i4_three | four, seven);
  EXPECT_EQ(i4_three ^ four, seven);
  EXPECT_EQ(i4_three << one, six);
  EXPECT_EQ(i4_three >> one, one);
  I4 val(-3);
  EXPECT_EQ(val %= two, minus_one);
  EXPECT_EQ(val, -1);
  EXPECT_EQ(val = I4(7), seven);
  EXPECT_EQ(val, 7);
  EXPECT_EQ(val &= two, two);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val |= one, three);
  EXPECT_EQ(val, 3);
  EXPECT_EQ(val ^= seven, four);
  EXPECT_EQ(val, 4);
  EXPECT_EQ(val >>= one, two);
  EXPECT_EQ(val, 2);
  EXPECT_EQ(val <<= one, four);
  EXPECT_EQ(val, 4);
}

}  // namespace
}  // namespace shlo_ref
