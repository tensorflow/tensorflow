/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/quantization_util.h"

#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Pair;

template <class FloatIn, class IntOut>
void RunSafeCastTests() {
  const IntOut imax = std::numeric_limits<IntOut>::max();
  EXPECT_GT(imax, 0);
  const IntOut imin = std::numeric_limits<IntOut>::min();
  const bool s = std::numeric_limits<IntOut>::is_signed;
  if (s) {
    EXPECT_LT(imin, 0);
  } else {
    EXPECT_EQ(0, imin);
  }

  // Some basic tests.
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(0.0)), 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-0.0)), 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(0.99)), 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(1.0)), 1);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(1.01)), 1);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(1.99)), 1);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(2.0)), 2);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(2.01)), 2);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-0.99)), 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-1.0)), s ? -1 : 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-1.01)), s ? -1 : 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-1.99)), s ? -1 : 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-2.0)), s ? -2 : 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-2.01)), s ? -2 : 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(117.9)), 117);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(118.0)), 118);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(118.1)), 118);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-117.9)), s ? -117 : 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-118.0)), s ? -118 : 0);
  EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-118.1)), s ? -118 : 0);

  // Some edge cases.
  EXPECT_EQ(SafeCast<IntOut>(std::numeric_limits<FloatIn>::max()), imax);
  EXPECT_EQ(SafeCast<IntOut>(std::numeric_limits<FloatIn>::lowest()), imin);
  EXPECT_EQ(SafeCast<IntOut>(std::numeric_limits<FloatIn>::infinity()), imax);
  EXPECT_EQ(SafeCast<IntOut>(-std::numeric_limits<FloatIn>::infinity()), imin);
  EXPECT_EQ(SafeCast<IntOut>(std::numeric_limits<FloatIn>::quiet_NaN()), 0);

  // Some larger numbers.
  if (sizeof(IntOut) >= 4 && sizeof(FloatIn) > 4) {
    EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(0x76543210)), 0x76543210);
  }

  if (sizeof(FloatIn) > sizeof(IntOut)) {
    // Check values near imax.
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) + 0.1)),
              imax);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) + 0.99)),
              imax);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) + 1.0)),
              imax);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) + 1.99)),
              imax);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) + 2.0)),
              imax);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) - 0.1)),
              imax - 1);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) - 0.99)),
              imax - 1);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) - 1.0)),
              imax - 1);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) - 1.01)),
              imax - 2);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) - 1.99)),
              imax - 2);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) - 2.0)),
              imax - 2);
    EXPECT_EQ(SafeCast<IntOut>(
                  static_cast<FloatIn>(static_cast<FloatIn>(imax) - 2.01)),
              imax - 3);
  }

  // Check values considerably larger in magnitude than imin and imax
  EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imax) * 2)),
      imax);
  EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imax) * 20)),
      imax);
  EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imax) * 100)),
      imax);
  EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imin) * 2)),
      imin);
  EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imin) * 20)),
      imin);
  EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imin) * 100)),
      imin);
}

TEST(QuantizationUtilTest, SafeCast) {
  RunSafeCastTests<float, int8_t>();
  RunSafeCastTests<double, int8_t>();
  RunSafeCastTests<float, int16_t>();
  RunSafeCastTests<double, int16_t>();
  RunSafeCastTests<float, int32_t>();
  RunSafeCastTests<double, int32_t>();
  RunSafeCastTests<float, int64_t>();
  RunSafeCastTests<double, int64_t>();
  RunSafeCastTests<float, uint8_t>();
  RunSafeCastTests<double, uint8_t>();
  RunSafeCastTests<float, uint16_t>();
  RunSafeCastTests<double, uint16_t>();
  RunSafeCastTests<float, uint32_t>();
  RunSafeCastTests<double, uint32_t>();
  RunSafeCastTests<float, uint64_t>();
  RunSafeCastTests<double, uint64_t>();
}

// Example taken from http://www.tensorflow.org/performance/quantization
//
//  Quantized | Float
//  --------- | -----
//  0         | -10.0
//  255       | 30.0
//  128       | 10.0
TEST(QuantizationUtilTest, ChooseQuantizationParams) {
  QuantizationParams qp = ChooseQuantizationParams<uint8>(-10.0, 30.0);
  EXPECT_NEAR(qp.scale, 0.156863, 1e-5);
  EXPECT_EQ(qp.zero_point, 64);
}

TEST(QuantizationUtilTest, ChooseQuantizationParamsZeroPointOnMinBoundary) {
  QuantizationParams qp = ChooseQuantizationParams<uint8>(0.0, 30.0);
  EXPECT_NEAR(qp.scale, 0.117647, 1e-5);
  EXPECT_EQ(qp.zero_point, 0);
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(QuantizationUtilTest, ChooseQuantizationParamsZeroNotInRange) {
  // Assumption is that zero is within the range.
  EXPECT_DEATH(ChooseQuantizationParams<uint8>(10.0, 30.0), "");
}

TEST(QuantizationUtilTest, ChooseQuantizationParamsEmptyRangePositive) {
  // Assumption is that zero is within the range.
  EXPECT_DEATH(ChooseQuantizationParams<uint8>(30.0, 30.0), "");
}
#endif  // GTEST_HAS_DEATH_TEST

TEST(QuantizationUtilTest, ChooseQuantizationParamsEmptyRangeZero) {
  QuantizationParams qp = ChooseQuantizationParams<uint8>(0.0, 0.0);
  EXPECT_NEAR(qp.scale, 0.0, 1e-5);
  EXPECT_EQ(qp.zero_point, 0);
}

TEST(QuantizationUtilTest, ChooseQuantizationParamsZeroPointOnMaxBoundary) {
  QuantizationParams qp = ChooseQuantizationParams<uint8>(-10.0, 0.0);
  EXPECT_NEAR(qp.scale, 0.039216, 1e-5);
  EXPECT_EQ(qp.zero_point, 255);
}

TEST(QuantizationUtilTest, IntegerFrExp) {
  int shift;
  int64_t result = IntegerFrExp(0.0, &shift);
  EXPECT_EQ(0, result);
  EXPECT_EQ(0, shift);

  result = IntegerFrExp(1.0, &shift);
  EXPECT_NEAR(0x40000000, result, 1);
  EXPECT_EQ(1, shift);

  result = IntegerFrExp(0.25, &shift);
  EXPECT_NEAR(0x40000000, result, 1);
  EXPECT_EQ(-1, shift);

  result = IntegerFrExp(-1.0, &shift);
  EXPECT_NEAR(-(1 << 30), result, 1);
  EXPECT_EQ(1, shift);

  result = IntegerFrExp(123.45, &shift);
  EXPECT_NEAR(2071147315, result, 1);
  EXPECT_EQ(7, shift);

  result = IntegerFrExp(NAN, &shift);
  EXPECT_NEAR(0, result, 1);
  EXPECT_EQ(0x7fffffff, shift);

  result = IntegerFrExp(INFINITY, &shift);
  EXPECT_NEAR(std::numeric_limits<int64_t>::max(), result, 1);
  EXPECT_EQ(0x7fffffff, shift);

  result = IntegerFrExp(-INFINITY, &shift);
  EXPECT_NEAR(std::numeric_limits<int64_t>::min(), result, 1);
  EXPECT_EQ(0x7fffffff, shift);
}

TEST(QuantizationUtilTest, IntegerFrExpVersusDouble) {
  int shift;
  int32_t result = IntegerFrExp(0.0, &shift);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(shift, 0);

  int double_shift;
  double double_result = std::frexp(0.0, &double_shift);
  EXPECT_EQ(double_result, 0);
  EXPECT_EQ(double_shift, 0);

  result = IntegerFrExp(1.0, &shift);
  EXPECT_NEAR(result, 0x40000000, 1);
  EXPECT_EQ(shift, 1);
  double_result = std::frexp(1.0, &double_shift);
  EXPECT_NEAR(double_result, 0.5, 1e-5);
  EXPECT_EQ(double_shift, 1);

  result = IntegerFrExp(0.25, &shift);
  EXPECT_NEAR(result, 0x40000000, 1);
  EXPECT_EQ(shift, -1);
  double_result = std::frexp(0.25, &double_shift);
  EXPECT_NEAR(double_result, 0.5, 1e-5);
  EXPECT_EQ(double_shift, -1);

  result = IntegerFrExp(-1.0, &shift);
  EXPECT_NEAR(result, -(1 << 30), 1);
  EXPECT_EQ(shift, 1);
  double_result = std::frexp(-1.0, &double_shift);
  EXPECT_NEAR(double_result, -0.5, 1e-5);
  EXPECT_EQ(double_shift, 1);

  result = IntegerFrExp(123.45, &shift);
  EXPECT_NEAR(result, (0.964453 * (1LL << 31)), 1000);
  EXPECT_EQ(shift, 7);
  double_result = std::frexp(123.45, &double_shift);
  EXPECT_NEAR(double_result, 0.964453, 1e-5);
  EXPECT_EQ(double_shift, 7);
}

TEST(QuantizationUtilTest, DoubleFromFractionAndShift) {
  double result = DoubleFromFractionAndShift(0, 0);
  EXPECT_EQ(0, result);

  result = DoubleFromFractionAndShift(0x40000000, 1);
  EXPECT_NEAR(1.0, result, 1e-5);

  result = DoubleFromFractionAndShift(0x40000000, 2);
  EXPECT_NEAR(2.0, result, 1e-5);

  int shift;
  int64_t fraction = IntegerFrExp(3.0, &shift);
  result = DoubleFromFractionAndShift(fraction, shift);
  EXPECT_NEAR(3.0, result, 1e-5);

  fraction = IntegerFrExp(123.45, &shift);
  result = DoubleFromFractionAndShift(fraction, shift);
  EXPECT_NEAR(123.45, result, 1e-5);

  fraction = IntegerFrExp(-23.232323, &shift);
  result = DoubleFromFractionAndShift(fraction, shift);
  EXPECT_NEAR(-23.232323, result, 1e-5);

  fraction = IntegerFrExp(NAN, &shift);
  result = DoubleFromFractionAndShift(fraction, shift);
  EXPECT_TRUE(std::isnan(result));

  fraction = IntegerFrExp(INFINITY, &shift);
  result = DoubleFromFractionAndShift(fraction, shift);
  EXPECT_FALSE(std::isfinite(result));
}

TEST(QuantizationUtilTest, IntegerDoubleMultiply) {
  EXPECT_NEAR(1.0, IntegerDoubleMultiply(1.0, 1.0), 1e-5);
  EXPECT_NEAR(2.0, IntegerDoubleMultiply(1.0, 2.0), 1e-5);
  EXPECT_NEAR(2.0, IntegerDoubleMultiply(2.0, 1.0), 1e-5);
  EXPECT_NEAR(4.0, IntegerDoubleMultiply(2.0, 2.0), 1e-5);
  EXPECT_NEAR(0.5, IntegerDoubleMultiply(1.0, 0.5), 1e-5);
  EXPECT_NEAR(0.25, IntegerDoubleMultiply(0.5, 0.5), 1e-5);
  EXPECT_NEAR(-1.0, IntegerDoubleMultiply(1.0, -1.0), 1e-5);
  EXPECT_NEAR(-1.0, IntegerDoubleMultiply(-1.0, 1.0), 1e-5);
  EXPECT_NEAR(1.0, IntegerDoubleMultiply(-1.0, -1.0), 1e-5);
  EXPECT_NEAR(15000000.0, IntegerDoubleMultiply(3000.0, 5000.0), 1e-5);
  EXPECT_TRUE(std::isnan(IntegerDoubleMultiply(NAN, 5000.0)));
  EXPECT_TRUE(std::isnan(IntegerDoubleMultiply(3000.0, NAN)));
}

TEST(QuantizationUtilTest, IntegerDoubleCompare) {
  EXPECT_EQ(-1, IntegerDoubleCompare(0.0, 1.0));
  EXPECT_EQ(1, IntegerDoubleCompare(1.0, 0.0));
  EXPECT_EQ(0, IntegerDoubleCompare(1.0, 1.0));
  EXPECT_EQ(0, IntegerDoubleCompare(0.0, 0.0));
  EXPECT_EQ(-1, IntegerDoubleCompare(-10.0, 10.0));
  EXPECT_EQ(1, IntegerDoubleCompare(123.45, 10.0));
  EXPECT_EQ(1, IntegerDoubleCompare(NAN, INFINITY));
  EXPECT_EQ(1, IntegerDoubleCompare(INFINITY, NAN));
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(QuantizationUtilTest, ChooseQuantizationParamsInvalidRange) {
  EXPECT_DEATH(ChooseQuantizationParams<uint8>(10.0, -30.0), "");
}

TEST(QuantizationUtilTest, QuantizeMultiplierSmallerThanOneExp) {
  auto quantize = [](double d) {
    int32_t q;
    int s;
    QuantizeMultiplierSmallerThanOneExp(d, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  EXPECT_DEATH(quantize(-0.1), "");
  EXPECT_DEATH(quantize(0.0), "");
  EXPECT_THAT(quantize(0.25), Pair(1073741824, -1));

  // Around 0.5 we can see the change in exponent and how we try hard to
  // void hitting max int32.
  EXPECT_THAT(quantize(0.50 - 5e-9), Pair(2147483627, -1));
  EXPECT_THAT(quantize(0.50 - 1e-10), Pair(1073741824, 0));
  EXPECT_THAT(quantize(0.50), Pair(1073741824, 0));

  EXPECT_THAT(quantize(0.75), Pair(1610612736, 0));
  EXPECT_THAT(quantize(1 - 1e-9), Pair(2147483646, 0));

  // If we get close enough to 1.0 it crashes and dies in one of two ways:
  // Either the shift becomes negative or we trigger the 'less-than-one' CHECK.
  EXPECT_DEATH(quantize(1 - 1e-15), "");
  EXPECT_DEATH(quantize(1 - 1e-17), "");
  EXPECT_DEATH(quantize(1.0), "");
}

TEST(QuantizationUtilTest, QuantizeMultiplierGreaterThanOne) {
  auto quantize = [](double d) {
    int32_t q;
    int s;
    QuantizeMultiplierGreaterThanOne(d, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  // If we are close enough to 1.0 it crashes.
  EXPECT_DEATH(quantize(1 + 1e-16), "");

  EXPECT_THAT(quantize(1 + 1e-11), Pair(1073741824, 1));
  EXPECT_THAT(quantize(1.25), Pair(1342177280, 1));
  EXPECT_THAT(quantize(1.50), Pair(1610612736, 1));
  EXPECT_THAT(quantize(1.75), Pair(1879048192, 1));

  // Around the powers of two we see the change in exponent. Also,
  // we try hard to avoid hitting max int32.
  EXPECT_THAT(quantize(2 - 1e-9), Pair(2147483647, 1));
  EXPECT_THAT(quantize(2 - 1e-11), Pair(1073741824, 2));
  EXPECT_THAT(quantize(2), Pair(1073741824, 2));
}

#ifndef __APPLE__  // Some Apple toolchains don't support std::ldexp
TEST(QuantizationUtilTest, QuantizeMultiplierUnderflow) {
  auto quantize = [](double d) {
    int32_t q;
    int s;
    QuantizeMultiplier(d, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  EXPECT_THAT(quantize(std::ldexp(1.0f, -31)), Pair(1073741824, -30));
  EXPECT_THAT(quantize(std::ldexp(1.0f, -32)), Pair(1073741824, -31));
  EXPECT_THAT(quantize(std::ldexp(0.99f, -32)), Pair(0, 0));
  EXPECT_THAT(quantize(std::ldexp(1.0f, -33)), Pair(0, 0));
}
#endif

TEST(QuantizationUtilTest, GetInvSqrtQuantizedMultiplierExp) {
  auto inv_sqrt = [](std::int32_t input) {
    int32_t output;
    int output_shift;
    GetInvSqrtQuantizedMultiplierExp(input, 1, &output, &output_shift);
    return std::pair<int32_t, int>{output, output_shift};
  };

  const auto kInt32Max = std::numeric_limits<std::int32_t>::max();
  EXPECT_THAT(inv_sqrt(0), Pair(kInt32Max, 0));
  EXPECT_THAT(inv_sqrt(1), Pair(kInt32Max, 0));
  EXPECT_THAT(inv_sqrt(2), Pair(1518498372, 0));
  EXPECT_THAT(inv_sqrt(3), Pair(1239850284, 0));
  EXPECT_THAT(inv_sqrt(4), Pair(1073741828, 0));
  EXPECT_THAT(inv_sqrt(100), Pair(214748363, 0));
  EXPECT_THAT(inv_sqrt(10000), Pair(343597361, 4));
  EXPECT_THAT(inv_sqrt(1000000), Pair(274877901, 7));
  EXPECT_THAT(inv_sqrt(100000000), Pair(219902323, 10));
  EXPECT_THAT(inv_sqrt((1 << 30)), Pair(268435457, 12));
  EXPECT_THAT(inv_sqrt(kInt32Max), Pair(189812531, 12));
}

TEST(QuantizationUtilTest, MultiplyByQuantizedMultiplierInt32) {
  auto quant_and_multiply = [](int32_t x, double multiplier) {
    int32_t quantized_multiplier;
    int shift;
    QuantizeMultiplier(multiplier, &quantized_multiplier, &shift);
    return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
  };

  EXPECT_EQ(quant_and_multiply(0, 0.1), 0);
  EXPECT_EQ(quant_and_multiply(1, 0), 0);
  EXPECT_EQ(quant_and_multiply(10000, 0.00097656), 10);
  EXPECT_EQ(quant_and_multiply(10000, -0.00097656), -10);
  EXPECT_EQ(quant_and_multiply(-10000, 0.00097656), -10);
  EXPECT_EQ(quant_and_multiply(-10000, -0.00097656), 10);
  EXPECT_EQ(quant_and_multiply(std::numeric_limits<int32_t>::min(), 0.00001),
            -21475);
  EXPECT_EQ(quant_and_multiply(std::numeric_limits<int32_t>::min(), -0.00001),
            21475);
  EXPECT_EQ(quant_and_multiply(std::numeric_limits<int32_t>::max(), 0.00001),
            21475);
  EXPECT_EQ(quant_and_multiply(std::numeric_limits<int32_t>::max(), -0.00001),
            -21475);

  // Test with maximum possible x and quantized_multiplier
  const int32_t x = std::numeric_limits<int32_t>::max();
  const int32_t quantized_multiplier = std::numeric_limits<int32_t>::max();
  const int shift = -3;
  const int32_t expected = static_cast<int32_t>(
      TfLiteRound(static_cast<int64_t>(x) * quantized_multiplier /
                  static_cast<double>(1LL << (31 - shift))));
  EXPECT_EQ(MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift),
            expected);
  EXPECT_EQ(MultiplyByQuantizedMultiplier(-x, quantized_multiplier, shift),
            -expected);
}

TEST(QuantizationUtilTest, MultiplyByQuantizedMultiplierInt64) {
  auto quant_and_multiply = [](int64_t x, double multiplier) {
    int32_t quantized_multiplier;
    int shift;
    QuantizeMultiplier(multiplier, &quantized_multiplier, &shift);
    return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
  };

  // Negative multipliers are not supported by the 64-bit
  // MultiplyByQuantizedMultiplier, only use >= 0 multipliers.
  EXPECT_EQ(quant_and_multiply(0, 0.1), 0);
  EXPECT_EQ(quant_and_multiply(1, 0), 0);
  EXPECT_EQ(quant_and_multiply(10000, 0.00097656), 10);
  EXPECT_EQ(quant_and_multiply(-10000, 0.00097656), -10);
  EXPECT_EQ(quant_and_multiply(-(1LL << 47), 0.00001), -1407385600);
  EXPECT_EQ(quant_and_multiply((1LL << 47) - 1, 0.00001), 1407385600);

  // Test with maximum possible x and quantized_multiplier
  const int64_t x = (1LL << 47) - 1;
  const int32_t quantized_multiplier = std::numeric_limits<int32_t>::max();
  const int shift = -31;
  // Expected is around 'x * quantized_multiplier / 2**(31 - shift)' ~= 65536
  // As there is some rounding error, expected is a bit smaller.
  const int32_t expected = 65534;
  EXPECT_EQ(MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift),
            expected);
  EXPECT_EQ(MultiplyByQuantizedMultiplier(-x, quantized_multiplier, shift),
            -expected);
}

TEST(QuantizationUtilTest, PreprocessSoftmaxScaling) {
  auto quantize = [](double beta, double scale, int integer_bits) {
    int32_t q;
    int s;
    PreprocessSoftmaxScaling(beta, scale, integer_bits, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  // If beta * scale is greater than fits in the number of integer bits, the
  // result is move near the maximum. Otherwise they quantize as expected.
  // With 4 integer bits we can represent up to 16.0.
  EXPECT_THAT(quantize(1.0, 16.0, 4), Pair(2147483647, 31));
  EXPECT_THAT(quantize(1.0, 8.0, 4), Pair(1073741824, 31));
  // But with 5 bits we can go further.
  EXPECT_THAT(quantize(2.0, 16.0, 5), Pair(2147483647, 31));
  EXPECT_THAT(quantize(2.0, 8.0, 5), Pair(1073741824, 31));
}
#endif  // GTEST_HAS_DEATH_TEST

TEST(QuantizationUtilTest, CalculateInputRadius) {
  EXPECT_EQ(CalculateInputRadius(4, 27), 15);
  EXPECT_EQ(CalculateInputRadius(3, 27), 14);
  EXPECT_EQ(CalculateInputRadius(3, 28), 7);
  EXPECT_EQ(CalculateInputRadius(4, 2), 503316480);
}

TEST(QuantizationUtilTest, QuantizeMultiplierArray) {
  const std::vector<double> weights = {-4,    -2,   -1,  -0.5, -0.25, -0.125, 0,
                                       0.125, 0.25, 0.5, 1,    2,     4};
  const int size = weights.size();
  std::vector<int32> effective_scale_significand(size);
  std::vector<int> effective_scale_shift(size);
  QuantizeMultiplierArray(weights.data(), size,
                          effective_scale_significand.data(),
                          effective_scale_shift.data());
  const std::vector<int32> expected_effective_scale_significand = {
      -1073741824,  // float scale = -4
      -1073741824,  // float scale = -2
      -1073741824,  // float scale = -1
      -1073741824,  // float scale = -0.5
      -1073741824,  // float scale = -0.25
      -1073741824,  // float scale = -0.125
      0,            // float scale = 0
      1073741824,   // float scale = 0.125
      1073741824,   // float scale = 0.25
      1073741824,   // float scale = 0.5
      1073741824,   // float scale = 1
      1073741824,   // float scale = 2
      1073741824,   // float scale = 4
  };

  const std::vector<int> expected_effective_scale_shift = {
      3,   // float scale = -4
      2,   // float scale = -2
      1,   // float scale = -1
      0,   // float scale = -0.5
      -1,  // float scale = -0.25
      -2,  // float scale = -0.125
      0,   // float scale = 0
      -2,  // float scale = 0.125
      -1,  // float scale = 0.25
      0,   // float scale = 0.5
      1,   // float scale = 1
      2,   // float scale = 2
      3,   // float scale = 4
  };
  EXPECT_THAT(effective_scale_significand,
              ElementsAreArray(expected_effective_scale_significand));
  EXPECT_THAT(effective_scale_shift,
              ElementsAreArray(expected_effective_scale_shift));
}

}  // namespace
}  // namespace tflite
