/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <limits>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/kernels/mlir_generated/base_binary_ops_test.h"
#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"

namespace tensorflow {
namespace {

// Test fixture `BinaryOpsTest` that sets the TF device is expected by the TEST
// macros below.
class BinaryOpsTest : public BinaryOpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }
};

/// Test `tf.Add`.

template <typename T>
T baseline_add(T lhs, T rhs) {
  return lhs + rhs;
}

GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_add)
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Float, float, float, baseline_add)
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Double, double, double, baseline_add)
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Int64, int64, int64, baseline_add)

/// Test `tf.AddV2`.

GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_add)
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Float, float, float, baseline_add)
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Double, double, double,
                       baseline_add)
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int64, int64, int64, baseline_add)

/// Test `tf.Atan2`.

// Prevent the undefined case (0, 0) with non-zero rhs values.
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/FloatRhsNonZero, float, float, test::DefaultInput<float>(),
    test::DefaultInputNonZero<float>(), std::atan2);
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/DoubleRhsNonZero, double, double,
    test::DefaultInput<double>(), test::DefaultInputNonZero<double>(),
    std::atan2);

// Prevent the undefined case (0, 0) with non-zero lhs values.
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/FloatLhsNonZero, float, float,
    test::DefaultInputNonZero<float>(), test::DefaultInput<float>(),
    std::atan2);
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/DoubleLhsNonZero, double, double,
    test::DefaultInputNonZero<double>(), test::DefaultInput<double>(),
    std::atan2);

// Test some particularly interesting cases.
TEST_F(BinaryOpsTest, Atan2FloatSpecialCases) {
  TestEqualShapes<float, float, float, float>(
      "Atan2", /*shape=*/{20},
      test::InputAsVector<float>({1, 1, 1, 0, -1, -1, -1, 0}),
      test::InputAsVector<float>({1, 0, -1, -1, -1, 0, 1, 1}), std::atan2,
      test::OpsTestConfig().ExpectStrictlyEqual());
}
TEST_F(BinaryOpsTest, Atan2DoubleSpecialCases) {
  TestEqualShapes<double, double, double, double>(
      "Atan2", /*shape=*/{20},
      test::InputAsVector<double>({1, 1, 1, 0, -1, -1, -1, 0}),
      test::InputAsVector<double>({1, 0, -1, -1, -1, 0, 1, 1}), std::atan2,
      test::OpsTestConfig().ExpectStrictlyEqual());
}

/// Test `tf.BitwiseAnd`.

template <typename T>
T baseline_bitwise_and(T lhs, T rhs) {
  return lhs & rhs;
}

GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int8, int8, int8, baseline_bitwise_and)
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int16, int16, int16, baseline_bitwise_and)
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int32, int32, int32, baseline_bitwise_and)
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int64, int64, int64, baseline_bitwise_and)

/// Test `tf.BitwiseOr`.

template <typename T>
T baseline_bitwise_or(T lhs, T rhs) {
  return lhs | rhs;
}

GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int8, int8, int8, baseline_bitwise_or)
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int16, int16, int16, baseline_bitwise_or)
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int32, int32, int32, baseline_bitwise_or)
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int64, int64, int64, baseline_bitwise_or)

/// Test `tf.BitwiseXor`.

template <typename T>
T baseline_bitwise_xor(T lhs, T rhs) {
  return lhs ^ rhs;
}

GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int8, int8, int8, baseline_bitwise_xor)
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int16, int16, int16, baseline_bitwise_xor)
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int32, int32, int32, baseline_bitwise_xor)
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int64, int64, int64, baseline_bitwise_xor)

/// Test `tf.Complex`.

template <typename T>
std::complex<T> baseline_complex(T lhs, T rhs) {
  return std::complex<T>(lhs, rhs);
}

GENERATE_DEFAULT_TESTS_2(Complex,
                         /*test_name=*/C64, float, float, std::complex<float>,
                         std::complex<float>, test::DefaultInput<float>(),
                         test::DefaultInput<float>(), baseline_complex,
                         test::OpsTestConfig().ExpectStrictlyEqual().AddTout())
GENERATE_DEFAULT_TESTS_2(Complex,
                         /*test_name=*/C128, double, double,
                         std::complex<double>, std::complex<double>,
                         test::DefaultInput<double>(),
                         test::DefaultInput<double>(), baseline_complex,
                         test::OpsTestConfig().ExpectStrictlyEqual().AddTout())

/// Test `tf.Div`.

template <typename T>
T baseline_div(T lhs, T rhs) {
  return lhs / rhs;
}

GENERATE_DEFAULT_TESTS(Div,
                       /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_div);
GENERATE_DEFAULT_TESTS(Div,
                       /*test_name=*/Float, float, float, baseline_div);
GENERATE_DEFAULT_TESTS(Div,
                       /*test_name=*/Double, double, double, baseline_div);
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Int16, int16, int16, test::DefaultInput<int16>(),
    test::DefaultInputNonZero<int16>(), baseline_div);
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Int64, int64, int64, test::DefaultInput<int64>(),
    test::DefaultInputNonZero<int64>(), baseline_div);

/// Test `tf.Equal`.

template <typename T>
bool baseline_equal(T lhs, T rhs) {
  return lhs == rhs;
}

GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Half, Eigen::half, bool,
                       baseline_equal)
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Float, float, bool, baseline_equal)
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Double, double, bool,
                       baseline_equal)
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Bool, bool, bool, baseline_equal)
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Int8, int8, bool, baseline_equal)
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Int16, int16, bool, baseline_equal)
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Int64, int64, bool, baseline_equal)

/// Test `tf.FloorDiv`.

template <typename T>
T baseline_floor_div(T lhs, T rhs) {
  return std::floor(lhs / rhs);
}

template <>
Eigen::half baseline_floor_div(Eigen::half lhs, Eigen::half rhs) {
  return static_cast<Eigen::half>(std::floor(static_cast<float>(lhs / rhs)));
}

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/Half, Eigen::half, Eigen::half,
    test::DefaultInput<Eigen::half>(), test::DefaultInputNonZero<Eigen::half>(),
    baseline_floor_div);
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/Float, float, float, test::DefaultInput<float>(),
    test::DefaultInputNonZero<float>(), baseline_floor_div);
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/Double, double, double, test::DefaultInput<double>(),
    test::DefaultInputNonZero<double>(), baseline_floor_div);

/// Test `tf.Greater`.

template <typename T>
bool baseline_greater(T lhs, T rhs) {
  return lhs > rhs;
}

GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Half, Eigen::half, bool,
                       baseline_greater)
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Float, float, bool,
                       baseline_greater)
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Double, double, bool,
                       baseline_greater)
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Int8, int8, bool,
                       baseline_greater)
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Int16, int16, bool,
                       baseline_greater)
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Int64, int64, bool,
                       baseline_greater)

/// Test `tf.GreaterEqual`.

template <typename T>
bool baseline_greater_equal(T lhs, T rhs) {
  return lhs >= rhs;
}

GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Half, Eigen::half, bool,
                       baseline_greater_equal)
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Float, float, bool,
                       baseline_greater_equal)
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Double, double, bool,
                       baseline_greater_equal)
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Int8, int8, bool,
                       baseline_greater_equal)
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Int16, int16, bool,
                       baseline_greater_equal)
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Int64, int64, bool,
                       baseline_greater_equal)

/// Test `tf.LeftShift`.

template <typename T>
T baseline_left_shift(T lhs, T rhs) {
  return lhs << rhs;
}

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int8, int8, int8, test::DefaultInput<int8>(),
    test::DefaultInputLessThanBitwidth<int8>(), baseline_left_shift)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int16, int16, int16, test::DefaultInput<int16>(),
    test::DefaultInputLessThanBitwidth<int16>(), baseline_left_shift)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int32, int32, int32, test::DefaultInput<int32>(),
    test::DefaultInputLessThanBitwidth<int32>(), baseline_left_shift)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int64, int64, int64, test::DefaultInput<int64>(),
    test::DefaultInputLessThanBitwidth<int64>(), baseline_left_shift)

/// Test `tf.Less`.

template <typename T>
bool baseline_less(T lhs, T rhs) {
  return lhs < rhs;
}

GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Half, Eigen::half, bool,
                       baseline_less)
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Float, float, bool, baseline_less)
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Double, double, bool, baseline_less)
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Int8, int8, bool, baseline_less)
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Int16, int16, bool, baseline_less)
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Int64, int64, bool, baseline_less)

/// Test `tf.LessEqual`.

template <typename T>
bool baseline_less_equal(T lhs, T rhs) {
  return lhs <= rhs;
}

GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Half, Eigen::half, bool,
                       baseline_less_equal)
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Float, float, bool,
                       baseline_less_equal)
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Double, double, bool,
                       baseline_less_equal)
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Int8, int8, bool,
                       baseline_less_equal)
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Int16, int16, bool,
                       baseline_less_equal)
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Int64, int64, bool,
                       baseline_less_equal)

/// Test `tf.LogicalAnd`.

bool baseline_logical_and(bool lhs, bool rhs) { return lhs && rhs; }

GENERATE_DEFAULT_TESTS_2(LogicalAnd, /*test_name=*/Bool, /*T=*/bool,
                         /*BaselineT=*/bool, /*OutT=*/bool,
                         /*BaselineOutT=*/bool, test::DefaultInput<bool>(),
                         test::DefaultInput<bool>(), baseline_logical_and,
                         test::OpsTestConfig().ExpectStrictlyEqual().NoT())

/// Test `tf.LogicalOr`.

bool baseline_logical_or(bool lhs, bool rhs) { return lhs || rhs; }

GENERATE_DEFAULT_TESTS_2(LogicalOr, /*test_name=*/Bool, /*T=*/bool,
                         /*BaselineT=*/bool, /*OutT=*/bool,
                         /*BaselineOutT=*/bool, test::DefaultInput<bool>(),
                         test::DefaultInput<bool>(), baseline_logical_or,
                         test::OpsTestConfig().ExpectStrictlyEqual().NoT())

/// Test `tf.Maximum`.

template <typename T>
T baseline_maximum(T lhs, T rhs) {
  if (std::isnan(lhs) || std::isnan(rhs)) {
    return lhs + rhs;
  }
  return std::max(lhs, rhs);
}

GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_maximum)
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/Float, float, float,
                       baseline_maximum)
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/Double, double, double,
                       baseline_maximum)
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/Int64, int64, int64,
                       baseline_maximum)

/// Test `tf.Minmum`.

template <typename T>
T baseline_minimum(T lhs, T rhs) {
  if (std::isnan(lhs) || std::isnan(rhs)) {
    return lhs + rhs;
  }
  return std::min(lhs, rhs);
}

GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_minimum)
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/Float, float, float,
                       baseline_minimum)
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/Double, double, double,
                       baseline_minimum)
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/Int64, int64, int64,
                       baseline_minimum)

/// Test `tf.Mul`.

template <typename T>
T baseline_mul(T lhs, T rhs) {
  return lhs * rhs;
}

GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_mul)
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Float, float, float, baseline_mul)
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Double, double, double, baseline_mul)
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Int8, int8, int8, baseline_mul)
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Int16, int16, int16, baseline_mul)
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Int64, int64, int64, baseline_mul)

/// Test `tf.NotEqual`.

template <typename T>
bool baseline_not_equal(T lhs, T rhs) {
  return lhs != rhs;
}

GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Half, Eigen::half, bool,
                       baseline_not_equal)
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Float, float, bool,
                       baseline_not_equal)
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Double, double, bool,
                       baseline_not_equal)
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Bool, bool, bool,
                       baseline_not_equal)
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Int8, int8, bool,
                       baseline_not_equal)
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Int16, int16, bool,
                       baseline_not_equal)
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Int64, int64, bool,
                       baseline_not_equal)

/// Test `tf.Polygamma`.

template <typename T>
static absl::InlinedVector<T, 10> GetPolygammaValuesX() {
  return test::InputAsVector<T, double>({-3.5, -3.0, -2.4, -2.0, -1.3, -1.0,
                                         -0.2, -0.0, 0.0, 0.1, 1.0, 1.2, 2.0,
                                         2.3, 3.0, 3.4});
}

template <typename T>
static absl::InlinedVector<T, 10> GetPolygammaValuesN() {
  int num_x_values = GetPolygammaValuesX<T>().size();
  auto n_values = {-4.0, -1.0, -0.0, 0.0, 3.0};
  absl::InlinedVector<T, 10> repeated_n_values;
  repeated_n_values.reserve(n_values.size() * num_x_values);
  for (double n : n_values) {
    for (int i = 0; i < num_x_values; i++) {
      repeated_n_values.push_back(n);
    }
  }
  return repeated_n_values;
}

double baseline_polygamma(double n, double x) {
  // Handle poles which have defined limits for odd n.
  if (x <= 0 && x == std::floor(x)) {
    if (static_cast<int>(n) % 2 == 1) {
      return std::numeric_limits<double>::infinity();
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  // Catch other undefined cases.
  if (n < 0 || n != std::floor(n))
    return std::numeric_limits<double>::quiet_NaN();

  // Approximate series for n > 0
  //   polygamma(n, x) = n! sum(k=0,...) (-x - k)^(n + 1)
  constexpr int kN = 1000000;
  if (n > 0) {
    double factorial = 1.0;
    for (int i = 1; i <= n; i++) {
      factorial *= i;
    }
    double sum = 0;
    for (int k = 0; k < kN; k++) {
      sum += 1.0 / std::pow(-x - k, n + 1);
    }
    return factorial * sum;
  }

  // Approximate series for n = 0
  //   polygamma(n, x) = -gamma + sum(k=1,...) (x - 1) / (k * (k + x - 1))
  assert(n == 0);
  constexpr double kGammaE = 0.5772156649015328606065120900824024;
  double sum = -kGammaE;
  double z = x - 1;
  for (int i = 1; i <= kN; i++) {
    sum += z / (i * (i + z));
  }
  return sum;
}

GENERATE_DEFAULT_TESTS_2(Polygamma, /*test_name=*/Float, float, double, float,
                         double, GetPolygammaValuesN<float>(),
                         GetPolygammaValuesX<float>(), baseline_polygamma,
                         test::OpsTestConfig().ATol(1e-11).RTol(1e-2))
GENERATE_DEFAULT_TESTS_2(Polygamma, /*test_name=*/Double, double, double,
                         double, double, GetPolygammaValuesN<double>(),
                         GetPolygammaValuesX<double>(), baseline_polygamma,
                         test::OpsTestConfig().ATol(1e-11).RTol(1e-2))

// Test at the poles.
TEST_F(BinaryOpsTest, PolygammaFloatSpecialCases) {
  TestEqualShapes<float, double, float, double>(
      "Polygamma", /*shape=*/{20},
      test::InputAsVector<float>({0, 1, 2, 3, 4, 5}),
      test::InputAsVector<float>({-3, -3, -2, -2, 0, 0}), baseline_polygamma,
      test::OpsTestConfig().ATol(1e-11).RTol(1e-2));
}
TEST_F(BinaryOpsTest, PolygammaDoubleSpecialCases) {
  TestEqualShapes<double, double, double, double>(
      "Polygamma", /*shape=*/{20},
      test::InputAsVector<double>({0, 1, 2, 3, 4, 5}),
      test::InputAsVector<double>({-3, -3, -2, -2, 0, 0}), baseline_polygamma,
      test::OpsTestConfig().ATol(1e-11).RTol(1e-2));
}

/// Test `tf.Pow`.

template <typename T>
T baseline_pow(T lhs, T rhs) {
  return std::pow(lhs, rhs);
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> PowInput() {
  return test::InputAsVector<T, double>({0.0, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0});
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, int8, int16, int32, int64>::value,
                           bool> = true>
absl::InlinedVector<T, 10> PowInput() {
  return test::InputAsVector<T, double>({-2, -1, -1, 1, 1, 3});
}

template <>
Eigen::half baseline_pow(Eigen::half lhs, Eigen::half rhs) {
  return static_cast<Eigen::half>(
      std::pow(static_cast<float>(lhs), static_cast<float>(rhs)));
}

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(Pow,
                                                  /*test_name=*/Half,
                                                  Eigen::half, Eigen::half,
                                                  PowInput<Eigen::half>(),
                                                  PowInput<Eigen::half>(),
                                                  baseline_pow)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(Pow,
                                                  /*test_name=*/Float, float,
                                                  float, PowInput<float>(),
                                                  PowInput<float>(),
                                                  baseline_pow)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(Pow,
                                                  /*test_name=*/Double, double,
                                                  double, PowInput<double>(),
                                                  PowInput<double>(),
                                                  baseline_pow)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(Pow,
                                                  /*test_name=*/Int64, int64,
                                                  int64, PowInput<int64>(),
                                                  PowInput<int64>(),
                                                  baseline_pow)

/// Test `tf.RealDiv`.

GENERATE_DEFAULT_TESTS(RealDiv,
                       /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_div);
GENERATE_DEFAULT_TESTS(RealDiv,
                       /*test_name=*/Float, float, float, baseline_div);
GENERATE_DEFAULT_TESTS(RealDiv,
                       /*test_name=*/Double, double, double, baseline_div);

/// Test `tf.RightShift`.

template <typename T>
T baseline_right_shift(T lhs, T rhs) {
  return lhs >> rhs;
}

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int8, int8, int8, test::DefaultInput<int8>(),
    test::DefaultInputLessThanBitwidth<int8>(), baseline_right_shift)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int16, int16, int16, test::DefaultInput<int16>(),
    test::DefaultInputLessThanBitwidth<int16>(), baseline_right_shift)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int32, int32, int32, test::DefaultInput<int32>(),
    test::DefaultInputLessThanBitwidth<int32>(), baseline_right_shift)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int64, int64, int64, test::DefaultInput<int64>(),
    test::DefaultInputLessThanBitwidth<int64>(), baseline_right_shift)

/// Test `tf.SquaredDifference`.

template <typename T>
T baseline_squared_difference(T lhs, T rhs) {
  return (lhs - rhs) * (lhs - rhs);
}

GENERATE_DEFAULT_TESTS(SquaredDifference, /*test_name=*/Half, Eigen::half,
                       Eigen::half, baseline_squared_difference)
GENERATE_DEFAULT_TESTS(SquaredDifference, /*test_name=*/Float, float, float,
                       baseline_squared_difference)
GENERATE_DEFAULT_TESTS(SquaredDifference, /*test_name=*/Double, double, double,
                       baseline_squared_difference)
GENERATE_DEFAULT_TESTS(SquaredDifference, /*test_name=*/Int64, int64, int64,
                       baseline_squared_difference)

/// Test `tf.Sub`.

template <typename T>
T baseline_sub(T lhs, T rhs) {
  return lhs - rhs;
}

GENERATE_DEFAULT_TESTS(Sub,
                       /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_sub)
GENERATE_DEFAULT_TESTS(Sub,
                       /*test_name=*/Float, float, float, baseline_sub)
GENERATE_DEFAULT_TESTS(Sub,
                       /*test_name=*/Double, double, double, baseline_sub)
GENERATE_DEFAULT_TESTS(Sub,
                       /*test_name=*/Int64, int64, int64, baseline_sub)

/// Test `tf.TruncateDiv`.

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv,
    /*test_name=*/Int16, int16, int16, test::DefaultInput<int16>(),
    test::DefaultInputNonZero<int16>(), baseline_div);
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv,
    /*test_name=*/Int64, int64, int64, test::DefaultInput<int64>(),
    test::DefaultInputNonZero<int64>(), baseline_div);

/// Test `tf.Zeta`.

template <typename T>
static absl::InlinedVector<T, 10> GetZetaTestDataX() {
  return test::InputAsVector<T, double>(
      {1.,           169.23969873, 105.93557562, 114.43259882, 179.62388639,
       172.80836494, 127.82036549, 163.07586688, 157.31865127, 121.55091407,
       132.49244284, 14.74785056,  61.69721805,  49.37079477,  32.73957728,
       8.63833678,   5.77183618,   7.43098888,   9.68867483,   6.90594844,
       1.10974422,   9.15604525,   5.39278873,   4.82471684,   3.61560063,
       5.95540334});
}

template <typename T>
static absl::InlinedVector<T, 10> GetZetaTestDataQ() {
  return test::InputAsVector<T, double>(
      {0.23672766, 0.92926068, 0.33551547, 0.53241745, 0.39939397, 0.73085145,
       0.91634121, 0.92935301, 0.90518735, 0.93155356, 0.31607971, 3.76257433,
       3.41533379, 3.4542971,  8.07960302, 7.49355634, 0.26524244, 0.11061626,
       0.26367137, 0.17993167, 0.17947252, 0.27949224, 0.20880047, 0.12189132,
       0.18806052, 0.19976058});
}

double baseline_zeta(double x, double q) {
  // Special divergent case.
  if (x == 1.0) return std::numeric_limits<double>::infinity();

  // Handle poles.
  if (q <= 0 && q == std::floor(q)) {
    if (x == std::floor(x) && static_cast<int>(x) % 2 == 0) {
      return std::numeric_limits<double>::infinity();
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  // Catch other undefined cases.
  if (x < 1.0 || (q <= 0 && x != std::floor(x)))
    return std::numeric_limits<double>::quiet_NaN();

  // Cases for which the series does not converge quickly enough.
  auto close_to = [](double a, double b) { return std::abs(a - b) < 0.0001; };
  if (close_to(x, 1.1097) && close_to(q, 0.1794)) return 16.1542;

  // Approximate through its series
  //   zeta(x, q) = sum(k=0,..) 1 / (k + q)^x
  double sum = 0;
  constexpr int kN = 1000000;
  for (int k = 0; k < kN; k++) sum += 1.0 / std::pow(k + q, x);
  return sum;
}

GENERATE_DEFAULT_TESTS_2(Zeta, /*test_name=*/Float, float, double, float,
                         double, GetZetaTestDataX<float>(),
                         GetZetaTestDataQ<float>(), baseline_zeta,
                         test::OpsTestConfig().ATol(1e-11).RTol(1e-2))
GENERATE_DEFAULT_TESTS_2(Zeta, /*test_name=*/Double, double, double, double,
                         double, GetZetaTestDataX<double>(),
                         GetZetaTestDataQ<double>(), baseline_zeta,
                         test::OpsTestConfig().ATol(1e-11).RTol(1e-2))

// Test at the poles.
TEST_F(BinaryOpsTest, ZetaFloatSpecialCases) {
  TestEqualShapes<float, double, float, double>(
      "Zeta", /*shape=*/{20}, test::InputAsVector<float>({1, 2, 3, 4, 5}),
      test::InputAsVector<float>({-3, -2, -1, 0, 1, 2, 3}), baseline_zeta,
      test::OpsTestConfig().ATol(1e-11).RTol(1e-2));
}
TEST_F(BinaryOpsTest, ZetaDoubleSpecialCases) {
  TestEqualShapes<double, double, double, double>(
      "Zeta", /*shape=*/{20}, test::InputAsVector<double>({1, 2, 3, 4, 5}),
      test::InputAsVector<double>({-3, -2, -1, 0, 1, 2, 3}), baseline_zeta,
      test::OpsTestConfig().ATol(1e-11).RTol(1e-2));
}

}  // namespace
}  // namespace tensorflow
