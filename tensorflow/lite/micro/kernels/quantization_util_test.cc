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

#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {

template <class FloatIn, class IntOut>
void RunSafeCastTests() {
  const IntOut imax = std::numeric_limits<IntOut>::max();
  TF_LITE_MICRO_EXPECT_GT(imax, 0);
  const IntOut imin = std::numeric_limits<IntOut>::min();
  const bool s = std::numeric_limits<IntOut>::is_signed;
  if (s) {
    TF_LITE_MICRO_EXPECT_LT(imin, 0);
  } else {
    TF_LITE_MICRO_EXPECT_EQ(0, imin);
  }

  // Some basic tests.
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(0.0)), 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-0.0)), 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(0.99)), 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(1.0)), 1);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(1.01)), 1);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(1.99)), 1);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(2.0)), 2);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(2.01)), 2);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-0.99)), 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-1.0)),
                          s ? -1 : 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-1.01)),
                          s ? -1 : 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-1.99)),
                          s ? -1 : 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-2.0)),
                          s ? -2 : 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-2.01)),
                          s ? -2 : 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(117.9)), 117);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(118.0)), 118);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(118.1)), 118);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-117.9)),
                          s ? -117 : 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-118.0)),
                          s ? -118 : 0);
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(-118.1)),
                          s ? -118 : 0);

  // Some edge cases.
  TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(std::numeric_limits<FloatIn>::max()),
                          imax);
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(std::numeric_limits<FloatIn>::lowest()), imin);
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(std::numeric_limits<FloatIn>::infinity()), imax);
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(-std::numeric_limits<FloatIn>::infinity()), imin);
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(std::numeric_limits<FloatIn>::quiet_NaN()), 0);

  // Some larger numbers.
  if (sizeof(IntOut) >= 4 && sizeof(FloatIn) > 4) {
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(0x76543210)),
                            0x76543210);
  }

  if (sizeof(FloatIn) > sizeof(IntOut)) {
    // Check values near imax.
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) + 0.1)),
                            imax);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) + 0.99)),
                            imax);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) + 1.0)),
                            imax);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) + 1.99)),
                            imax);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) + 2.0)),
                            imax);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) - 0.1)),
                            imax - 1);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) - 0.99)),
                            imax - 1);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) - 1.0)),
                            imax - 1);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) - 1.01)),
                            imax - 2);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) - 1.99)),
                            imax - 2);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) - 2.0)),
                            imax - 2);
    TF_LITE_MICRO_EXPECT_EQ(SafeCast<IntOut>(static_cast<FloatIn>(
                                static_cast<FloatIn>(imax) - 2.01)),
                            imax - 3);
  }

  // Check values considerably larger in magnitude than imin and imax
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imax) * 2)),
      imax);
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imax) * 20)),
      imax);
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imax) * 100)),
      imax);
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imin) * 2)),
      imin);
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imin) * 20)),
      imin);
  TF_LITE_MICRO_EXPECT_EQ(
      SafeCast<IntOut>(static_cast<FloatIn>(static_cast<FloatIn>(imin) * 100)),
      imin);
}

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(QuantizationUtilTest_SafeCast) {
  tflite::RunSafeCastTests<float, int8_t>();
  tflite::RunSafeCastTests<double, int8_t>();
  tflite::RunSafeCastTests<float, int16_t>();
  tflite::RunSafeCastTests<double, int16_t>();
  tflite::RunSafeCastTests<float, int32_t>();
  tflite::RunSafeCastTests<double, int32_t>();
  tflite::RunSafeCastTests<float, int64_t>();
  tflite::RunSafeCastTests<double, int64_t>();
  tflite::RunSafeCastTests<float, uint8_t>();
  tflite::RunSafeCastTests<double, uint8_t>();
  tflite::RunSafeCastTests<float, uint16_t>();
  tflite::RunSafeCastTests<double, uint16_t>();
  tflite::RunSafeCastTests<float, uint32_t>();
  tflite::RunSafeCastTests<double, uint32_t>();
  tflite::RunSafeCastTests<float, uint64_t>();
  tflite::RunSafeCastTests<double, uint64_t>();
}

// Example taken from http://www.tensorflow.org/performance/quantization
//
//  Quantized | Float
//  --------- | -----
//  0         | -10.0
//  255       | 30.0
//  128       | 10.0
TF_LITE_MICRO_TEST(QuantizationUtilTest_ChooseQuantizationParams) {
  tflite::QuantizationParams qp =
      tflite::ChooseQuantizationParams<uint8>(-10.0, 30.0);
  TF_LITE_MICRO_EXPECT_NEAR(qp.scale, 0.156863, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(qp.zero_point, 64);
}

TF_LITE_MICRO_TEST(
    QuantizationUtilTest_ChooseQuantizationParamsZeroPointOnMinBoundary) {
  tflite::QuantizationParams qp =
      tflite::ChooseQuantizationParams<uint8>(0.0, 30.0);
  TF_LITE_MICRO_EXPECT_NEAR(qp.scale, 0.117647, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(qp.zero_point, 0);
}

TF_LITE_MICRO_TEST(
    QuantizationUtilTest_ChooseQuantizationParamsEmptyRangeZero) {
  tflite::QuantizationParams qp =
      tflite::ChooseQuantizationParams<uint8>(0.0, 0.0);
  TF_LITE_MICRO_EXPECT_NEAR(qp.scale, 0.0, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(qp.zero_point, 0);
}

TF_LITE_MICRO_TEST(
    QuantizationUtilTest_ChooseQuantizationParamsZeroPointOnMaxBoundary) {
  tflite::QuantizationParams qp =
      tflite::ChooseQuantizationParams<uint8>(-10.0, 0.0);
  TF_LITE_MICRO_EXPECT_NEAR(qp.scale, 0.039216, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(qp.zero_point, 255);
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_IntegerFrExp) {
  int shift;
  int64_t result = tflite::IntegerFrExp(0.0, &shift);
  TF_LITE_MICRO_EXPECT_EQ(0, result);
  TF_LITE_MICRO_EXPECT_EQ(0, shift);

  result = tflite::IntegerFrExp(1.0, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(0x40000000, result, 1);
  TF_LITE_MICRO_EXPECT_EQ(1, shift);

  result = tflite::IntegerFrExp(0.25, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(0x40000000, result, 1);
  TF_LITE_MICRO_EXPECT_EQ(-1, shift);

  result = tflite::IntegerFrExp(-1.0, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(-(1 << 30), result, 1);
  TF_LITE_MICRO_EXPECT_EQ(1, shift);

  result = tflite::IntegerFrExp(123.45, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(2071147315, result, 1);
  TF_LITE_MICRO_EXPECT_EQ(7, shift);

  result = tflite::IntegerFrExp(NAN, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(0, result, 1);
  TF_LITE_MICRO_EXPECT_EQ(0x7fffffff, shift);

  result = tflite::IntegerFrExp(INFINITY, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(std::numeric_limits<int64_t>::max(), result, 1);
  TF_LITE_MICRO_EXPECT_EQ(0x7fffffff, shift);

  result = tflite::IntegerFrExp(-INFINITY, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(std::numeric_limits<int64_t>::min(), result, 1);
  TF_LITE_MICRO_EXPECT_EQ(0x7fffffff, shift);
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_IntegerFrExpVersusDouble) {
  int shift;
  int32_t result = tflite::IntegerFrExp(0.0, &shift);
  TF_LITE_MICRO_EXPECT_EQ(result, 0);
  TF_LITE_MICRO_EXPECT_EQ(shift, 0);

  int double_shift;
  double double_result = std::frexp(0.0, &double_shift);
  TF_LITE_MICRO_EXPECT_EQ(double_result, 0);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, 0);

  result = tflite::IntegerFrExp(1.0, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(result, 0x40000000, 1);
  TF_LITE_MICRO_EXPECT_EQ(shift, 1);
  double_result = std::frexp(1.0, &double_shift);
  TF_LITE_MICRO_EXPECT_NEAR(double_result, 0.5, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, 1);

  result = tflite::IntegerFrExp(0.25, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(result, 0x40000000, 1);
  TF_LITE_MICRO_EXPECT_EQ(shift, -1);
  double_result = std::frexp(0.25, &double_shift);
  TF_LITE_MICRO_EXPECT_NEAR(double_result, 0.5, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, -1);

  result = tflite::IntegerFrExp(-1.0, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(result, -(1 << 30), 1);
  TF_LITE_MICRO_EXPECT_EQ(shift, 1);
  double_result = std::frexp(-1.0, &double_shift);
  TF_LITE_MICRO_EXPECT_NEAR(double_result, -0.5, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, 1);

  result = tflite::IntegerFrExp(123.45, &shift);
  TF_LITE_MICRO_EXPECT_NEAR(result, (0.964453 * (1LL << 31)), 1000);
  TF_LITE_MICRO_EXPECT_EQ(shift, 7);
  double_result = std::frexp(123.45, &double_shift);
  TF_LITE_MICRO_EXPECT_NEAR(double_result, 0.964453, 1e-5);
  TF_LITE_MICRO_EXPECT_EQ(double_shift, 7);
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_DoubleFromFractionAndShift) {
  double result = tflite::DoubleFromFractionAndShift(0, 0);
  TF_LITE_MICRO_EXPECT_EQ(0, result);

  result = tflite::DoubleFromFractionAndShift(0x40000000, 1);
  TF_LITE_MICRO_EXPECT_NEAR(1.0, result, 1e-5);

  result = tflite::DoubleFromFractionAndShift(0x40000000, 2);
  TF_LITE_MICRO_EXPECT_NEAR(2.0, result, 1e-5);

  int shift;
  int64_t fraction = tflite::IntegerFrExp(3.0, &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_NEAR(3.0, result, 1e-5);

  fraction = tflite::IntegerFrExp(123.45, &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_NEAR(123.45, result, 1e-5);

  fraction = tflite::IntegerFrExp(-23.232323, &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_NEAR(-23.232323, result, 1e-5);

  fraction = tflite::IntegerFrExp(NAN, &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_TRUE(std::isnan(result));

  fraction = tflite::IntegerFrExp(INFINITY, &shift);
  result = tflite::DoubleFromFractionAndShift(fraction, shift);
  TF_LITE_MICRO_EXPECT_FALSE(std::isfinite(result));
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_IntegerDoubleMultiply) {
  TF_LITE_MICRO_EXPECT_NEAR(1.0, tflite::IntegerDoubleMultiply(1.0, 1.0), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(2.0, tflite::IntegerDoubleMultiply(1.0, 2.0), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(2.0, tflite::IntegerDoubleMultiply(2.0, 1.0), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(4.0, tflite::IntegerDoubleMultiply(2.0, 2.0), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(0.5, tflite::IntegerDoubleMultiply(1.0, 0.5), 1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(0.25, tflite::IntegerDoubleMultiply(0.5, 0.5),
                            1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(-1.0, tflite::IntegerDoubleMultiply(1.0, -1.0),
                            1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(-1.0, tflite::IntegerDoubleMultiply(-1.0, 1.0),
                            1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(1.0, tflite::IntegerDoubleMultiply(-1.0, -1.0),
                            1e-5);
  TF_LITE_MICRO_EXPECT_NEAR(
      15000000.0, tflite::IntegerDoubleMultiply(3000.0, 5000.0), 1e-5);
  TF_LITE_MICRO_EXPECT_TRUE(
      std::isnan(tflite::IntegerDoubleMultiply(NAN, 5000.0)));
  TF_LITE_MICRO_EXPECT_TRUE(
      std::isnan(tflite::IntegerDoubleMultiply(3000.0, NAN)));
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_IntegerDoubleCompare) {
  TF_LITE_MICRO_EXPECT_EQ(-1, tflite::IntegerDoubleCompare(0.0, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(1, tflite::IntegerDoubleCompare(1.0, 0.0));
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::IntegerDoubleCompare(1.0, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::IntegerDoubleCompare(0.0, 0.0));
  TF_LITE_MICRO_EXPECT_EQ(-1, tflite::IntegerDoubleCompare(-10.0, 10.0));
  TF_LITE_MICRO_EXPECT_EQ(1, tflite::IntegerDoubleCompare(123.45, 10.0));
  TF_LITE_MICRO_EXPECT_EQ(1, tflite::IntegerDoubleCompare(NAN, INFINITY));
  TF_LITE_MICRO_EXPECT_EQ(1, tflite::IntegerDoubleCompare(INFINITY, NAN));
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_PreprocessSoftmaxScaling) {
  auto quantize = [](double beta, double scale, int integer_bits) {
    int32_t q;
    int s;
    tflite::PreprocessSoftmaxScaling(beta, scale, integer_bits, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  // If beta * scale is greater than fits in the number of integer bits, the
  // result is move near the maximum. Otherwise they quantize as expected.
  // With 4 integer bits we can represent up to 16.0.

  auto r = quantize(1.0, 16.0, 4);
  TF_LITE_MICRO_EXPECT_EQ(r.first, 2147483647);
  TF_LITE_MICRO_EXPECT_EQ(r.second, 31);

  r = quantize(1.0, 8.0, 4);
  TF_LITE_MICRO_EXPECT_EQ(r.first, 1073741824);
  TF_LITE_MICRO_EXPECT_EQ(r.second, 31);

  // But with 5 bits we can go further.
  r = quantize(2.0, 16.0, 5);
  TF_LITE_MICRO_EXPECT_EQ(r.first, 2147483647);
  TF_LITE_MICRO_EXPECT_EQ(r.second, 31);

  r = quantize(2.0, 8.0, 5);
  TF_LITE_MICRO_EXPECT_EQ(r.first, 1073741824);
  TF_LITE_MICRO_EXPECT_EQ(r.second, 31);
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_CalculateInputRadius) {
  TF_LITE_MICRO_EXPECT_EQ(tflite::CalculateInputRadius(4, 27), 15);
  TF_LITE_MICRO_EXPECT_EQ(tflite::CalculateInputRadius(3, 27), 14);
  TF_LITE_MICRO_EXPECT_EQ(tflite::CalculateInputRadius(3, 28), 7);
  TF_LITE_MICRO_EXPECT_EQ(tflite::CalculateInputRadius(4, 2), 503316480);
}

TF_LITE_MICRO_TEST(QuantizationUtilTest_QuantizeMultiplierArray) {
  const double weights[] = {-4,    -2,   -1,  -0.5, -0.25, -0.125, 0,
                            0.125, 0.25, 0.5, 1,    2,     4};

  const int size = 13;
  int32 effective_scale_significand[size];
  int effective_scale_shift[size];
  tflite::QuantizeMultiplierArray(weights, size, effective_scale_significand,
                                  effective_scale_shift);
  const int32 expected_effective_scale_significand[] = {
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

  const int expected_effective_scale_shift[] = {
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

  for (int i = 0; i < size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(effective_scale_significand[i],
                            expected_effective_scale_significand[i]);
    TF_LITE_MICRO_EXPECT_EQ(effective_scale_shift[i],
                            expected_effective_scale_shift[i]);
  }
}

TF_LITE_MICRO_TESTS_END
