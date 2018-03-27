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
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace {

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
  RunSafeCastTests<float, int8>();
  RunSafeCastTests<double, int8>();
  RunSafeCastTests<float, int16>();
  RunSafeCastTests<double, int16>();
  RunSafeCastTests<float, int32>();
  RunSafeCastTests<double, int32>();
  RunSafeCastTests<float, int64>();
  RunSafeCastTests<double, int64>();
  RunSafeCastTests<float, uint8>();
  RunSafeCastTests<double, uint8>();
  RunSafeCastTests<float, uint16>();
  RunSafeCastTests<double, uint16>();
  RunSafeCastTests<float, uint32>();
  RunSafeCastTests<double, uint32>();
  RunSafeCastTests<float, uint64>();
  RunSafeCastTests<double, uint64>();
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

TEST(QuantizationUtilTest, ChooseQuantizationParamsZeroNotInRange) {
  // Assumption is that zero is within the range.
  EXPECT_DEATH(ChooseQuantizationParams<uint8>(10.0, 30.0), "");
}

TEST(QuantizationUtilTest, ChooseQuantizationParamsEmptyRangePositive) {
  // Assumption is that zero is within the range.
  EXPECT_DEATH(ChooseQuantizationParams<uint8>(30.0, 30.0), "");
}

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

TEST(QuantizationUtilTest, ChooseQuantizationParamsInvalidRange) {
  EXPECT_DEATH(ChooseQuantizationParams<uint8>(10.0, -30.0), "");
}

TEST(QuantizationUtilTest, QuantizeMultiplierSmallerThanOne) {
  auto quantize = [](double d) {
    int32_t q;
    int s;
    QuantizeMultiplierSmallerThanOne(d, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  EXPECT_DEATH(quantize(-0.1), "");
  EXPECT_DEATH(quantize(0.0), "");
  EXPECT_THAT(quantize(0.25), Pair(1073741824, 1));

  // Around 0.5 we can see the change in exponent and how we try hard to
  // void hitting max int32.
  EXPECT_THAT(quantize(0.50 - 5e-9), Pair(2147483627, 1));
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

TEST(QuantizationUtilTest, CalculateInputRadius) {
  EXPECT_EQ(CalculateInputRadius(4, 27), 15);
  EXPECT_EQ(CalculateInputRadius(3, 27), 14);
  EXPECT_EQ(CalculateInputRadius(3, 28), 7);
  EXPECT_EQ(CalculateInputRadius(4, 2), 503316480);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
