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
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Float, float, float, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Double, double, double, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Int64, int64_t, int64_t, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/UInt8, uint8_t, uint8_t, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Complex64, std::complex<float>,
                       std::complex<float>, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Complex128, std::complex<double>,
                       std::complex<double>, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Int8, int8_t, int8_t, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Add, /*test_name=*/Int16, int16_t, int16_t, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.AddV2`.

GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Float, float, float, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Double, double, double,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int64, int64_t, int64_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Complex64, std::complex<float>,
                       std::complex<float>, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Complex128, std::complex<double>,
                       std::complex<double>, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int8, int8_t, int8_t, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int16, int16_t, int16_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.Atan2`.

Eigen::half baseline_atan2(Eigen::half lhs, Eigen::half rhs) {
  return static_cast<Eigen::half>(
      std::atan2(static_cast<float>(lhs), static_cast<float>(rhs)));
}

// Prevent the undefined case (0, 0) with non-zero rhs values.
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/HalfRhsNonZero, Eigen::half, Eigen::half,
    test::DefaultInput<Eigen::half>(), test::DefaultInputNonZero<Eigen::half>(),
    baseline_atan2, test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/FloatRhsNonZero, float, float, test::DefaultInput<float>(),
    test::DefaultInputNonZero<float>(), std::atan2,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/DoubleRhsNonZero, double, double,
    test::DefaultInput<double>(), test::DefaultInputNonZero<double>(),
    std::atan2, test::OpsTestConfig().ExpectStrictlyEqual())

// Prevent the undefined case (0, 0) with non-zero lhs values.
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/HalfLhsNonZero, Eigen::half, Eigen::half,
    test::DefaultInputNonZero<Eigen::half>(), test::DefaultInput<Eigen::half>(),
    baseline_atan2, test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/FloatLhsNonZero, float, float,
    test::DefaultInputNonZero<float>(), test::DefaultInput<float>(), std::atan2,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Atan2,
    /*test_name=*/DoubleLhsNonZero, double, double,
    test::DefaultInputNonZero<double>(), test::DefaultInput<double>(),
    std::atan2, test::OpsTestConfig().ExpectStrictlyEqual())

// Test some particularly interesting cases.
TEST_F(BinaryOpsTest, Atan2EigenHalfSpecialCases) {
  TestEqualShapes<Eigen::half, float, Eigen::half, float>(
      "Atan2", /*shape=*/{20},
      test::InputAsVector<Eigen::half>({1, 1, 1, 0, -1, -1, -1, 0}),
      test::InputAsVector<Eigen::half>({1, 0, -1, -1, -1, 0, 1, 1}), std::atan2,
      test::OpsTestConfig().ExpectStrictlyEqual());
}
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
                       /*test_name=*/Int8, int8_t, int8_t, baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int16, int16_t, int16_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int32, int32_t, int32_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int64, int64_t, int64_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.BitwiseOr`.

template <typename T>
T baseline_bitwise_or(T lhs, T rhs) {
  return lhs | rhs;
}

GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int8, int8_t, int8_t, baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int16, int16_t, int16_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int32, int32_t, int32_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int64, int64_t, int64_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.BitwiseXor`.

template <typename T>
T baseline_bitwise_xor(T lhs, T rhs) {
  return lhs ^ rhs;
}

GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int8, int8_t, int8_t, baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int16, int16_t, int16_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int32, int32_t, int32_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int64, int64_t, int64_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())

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
                       baseline_div,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Div,
                       /*test_name=*/Float, float, float, baseline_div,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Div,
                       /*test_name=*/Double, double, double, baseline_div,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Int16, int16_t, int16_t, test::DefaultInput<int16_t>(),
    test::DefaultInputNonZero<int16_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Int64, int64_t, int64_t, test::DefaultInput<int64_t>(),
    test::DefaultInputNonZero<int64_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/UInt8, uint8_t, uint8_t, test::DefaultInput<uint8_t>(),
    test::DefaultInputNonZero<uint8_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/UInt16, uint16_t, uint16_t, test::DefaultInput<uint16_t>(),
    test::DefaultInputNonZero<uint16_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Int8, int8_t, int8_t, test::DefaultInput<int8_t>(),
    test::DefaultInputNonZero<int8_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Uint32, uint32_t, uint32_t, test::DefaultInput<uint32_t>(),
    test::DefaultInputNonZero<uint32_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Uint64, uint64_t, uint64_t, test::DefaultInput<uint64_t>(),
    test::DefaultInputNonZero<uint64_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
#endif

// The following tests don't work with Eigen kernels if the Eigen kernels are
// compiled with nvcc.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TEST_F(BinaryOpsTest, DivComplex64SpecialCases) {
  TestEqualShapes<std::complex<float>, std::complex<float>, std::complex<float>,
                  std::complex<float>>(
      "Div", /*shape=*/{67, 63},
      test::NearZeroInfAndNanInput<std::complex<float>>(),
      test::RepeatElements(test::NearZeroInfAndNanInput<std::complex<float>>(),
                           64),
      baseline_div, test::OpsTestConfig());
}

TEST_F(BinaryOpsTest, DivComplex128SpecialCases) {
  TestEqualShapes<std::complex<double>, std::complex<double>,
                  std::complex<double>, std::complex<double>>(
      "Div", /*shape=*/{67, 63},
      test::NearZeroInfAndNanInput<std::complex<double>>(),
      test::RepeatElements(test::NearZeroInfAndNanInput<std::complex<double>>(),
                           64),
      baseline_div, test::OpsTestConfig());
}
#endif

/// Test `tf.TruncatedDiv`

template <typename T>
T baseline_truncate_div(T lhs, T rhs) {
  T res = lhs / rhs;
  if (res < 0) return ceil(res);
  return floor(res);
}

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv, /*test_name=*/Int8, int8_t, int8_t,
    test::DefaultInput<int8_t>(), test::DefaultInputNonZero<int8_t>(),
    baseline_div, test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv, /*test_name=*/Uint32, uint32_t, uint32_t,
    test::DefaultInput<uint32_t>(), test::DefaultInputNonZero<uint32_t>(),
    baseline_div, test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv, /*test_name=*/Uint64, uint64_t, uint64_t,
    test::DefaultInput<uint64_t>(), test::DefaultInputNonZero<uint64_t>(),
    baseline_div, test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv, /*test_name=*/Half, Eigen::half, Eigen::half,
    test::DefaultInput<Eigen::half>(), test::DefaultInputNonZero<Eigen::half>(),
    baseline_truncate_div, test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv, /*test_name=*/Float, float, float, test::DefaultInput<float>(),
    test::DefaultInputNonZero<float>(), baseline_truncate_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv, /*test_name=*/Double, double, double,
    test::DefaultInput<double>(), test::DefaultInputNonZero<double>(),
    baseline_truncate_div, test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.DivNoNan`.

template <typename T>
T baseline_div_no_nan(T lhs, T rhs) {
  return rhs == T(0) ? T(0) : lhs / rhs;
}

GENERATE_DEFAULT_TESTS(DivNoNan,
                       /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_div_no_nan,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(DivNoNan,
                       /*test_name=*/Float, float, float, baseline_div_no_nan,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(DivNoNan,
                       /*test_name=*/Double, double, double,
                       baseline_div_no_nan,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    DivNoNan,
    /*test_name=*/ZeroDenominator, float, float, test::DefaultInput<float>(),
    test::InputAsVector<float>({0}), baseline_div_no_nan,
    test::OpsTestConfig().ExpectStrictlyEqual())

// The following tests don't work with Eigen kernels, the relative/absolute
// precision is too bad (e.g. for input (-18 + 18j) / (1e-6 - 1e-j), Eigen
// kernels return (18000000 + 0.3410605192j), but the imaginary part should be
// close to 0.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS(DivNoNan,
                       /*test_name=*/Complex64, std::complex<float>,
                       std::complex<float>, baseline_div_no_nan,
                       test::OpsTestConfig())
GENERATE_DEFAULT_TESTS(DivNoNan,
                       /*test_name=*/Complex128, std::complex<double>,
                       std::complex<double>, baseline_div_no_nan,
                       test::OpsTestConfig())
#endif

/// Test `tf.Equal`.

template <typename T>
bool baseline_equal(T lhs, T rhs) {
  return lhs == rhs;
}

GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Half, Eigen::half, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Float, float, bool, baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Double, double, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Bool, bool, bool, baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Int8, int8_t, bool, baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Int16, int16_t, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/Int64, int64_t, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/UInt8, uint8_t, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/C64, std::complex<float>, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/C128, std::complex<double>, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())

TEST_F(BinaryOpsTest, EqualUint8_tSpecialCases) {
  TestEqualShapes<uint8_t, uint8_t, bool, bool>(
      "Equal", /*shape=*/{20},
      test::InputAsVector<uint8_t>({255, 1, 0, 0, 1, 255, 0, 255}),
      test::InputAsVector<uint8_t>({1, 255, 0, 1, 0, 0, 255, 255}),
      baseline_equal, test::OpsTestConfig().ExpectStrictlyEqual());
}

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/UInt16, uint16_t, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/UInt32, uint32_t, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Equal, /*test_name=*/UInt64, uint64_t, bool,
                       baseline_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.FloorDiv`.

template <typename T, std::enable_if_t<llvm::is_one_of<T, float, double>::value,
                                       bool> = true>
T baseline_floor_div(T lhs, T rhs) {
  return std::floor(lhs / rhs);
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, Eigen::half>::value, bool> = true>
T baseline_floor_div(T lhs, T rhs) {
  return static_cast<T>(std::floor(static_cast<float>(lhs / rhs)));
}

template <typename T, std::enable_if_t<llvm::is_one_of<T, int8_t, int16_t,
                                                       int32_t, int64_t>::value,
                                       bool> = true>
T baseline_floor_div(T lhs, T rhs) {
  T res = lhs / rhs;
  if (((lhs < 0 && rhs > 0) || (lhs > 0 && rhs < 0)) && lhs % rhs) {
    --res;
  }
  return res;
}

template <typename T,
          std::enable_if_t<
              llvm::is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>::value,
              bool> = true>
T baseline_floor_div(T lhs, T rhs) {
  return lhs / rhs;
}

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/UInt8, uint8_t, uint8_t, test::DefaultInput<uint8_t>(),
    test::DefaultInputNonZero<uint8_t>(), baseline_floor_div,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/UInt16, uint16_t, uint16_t, test::DefaultInput<uint16_t>(),
    test::DefaultInputNonZero<uint16_t>(), baseline_floor_div,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/Int16, int16_t, int16_t, test::DefaultInput<int16_t>(),
    test::DefaultInputNonZero<int16_t>(), baseline_floor_div,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/Int64, int64_t, int64_t, test::DefaultInput<int64_t>(),
    test::DefaultInputNonZero<int64_t>(), baseline_floor_div,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/Half, Eigen::half, Eigen::half,
    test::DefaultInput<Eigen::half>(), test::DefaultInputNonZero<Eigen::half>(),
    baseline_floor_div, test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/Float, float, float, test::DefaultInput<float>(),
    test::DefaultInputNonZero<float>(), baseline_floor_div,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/Double, double, double, test::DefaultInput<double>(),
    test::DefaultInputNonZero<double>(), baseline_floor_div,
    test::OpsTestConfig().ExpectStrictlyEqual());

/// Test the JIT-compiled kernels.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/Int8, int8_t, int8_t, test::DefaultInput<int8_t>(),
    test::DefaultInputNonZero<int8_t>(), baseline_floor_div,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/UInt32, uint32_t, uint32_t, test::DefaultInput<uint32_t>(),
    test::DefaultInputNonZero<uint32_t>(), baseline_floor_div,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorDiv,
    /*test_name=*/UInt64, uint64_t, uint64_t, test::DefaultInput<uint64_t>(),
    test::DefaultInputNonZero<uint64_t>(), baseline_floor_div,
    test::OpsTestConfig().ExpectStrictlyEqual());
#endif

/// Test `tf.FloorMod`.

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
T baseline_floor_mod(T lhs, T rhs) {
  double res = std::fmod(static_cast<double>(lhs), static_cast<double>(rhs));
  if (res != 0.0 && ((res < 0 && rhs > 0) || (res > 0 && rhs < 0))) {
    res += rhs;
  }
  return static_cast<T>(res);
}

template <typename T, std::enable_if_t<llvm::is_one_of<T, int8_t, int16_t,
                                                       int32_t, int64_t>::value,
                                       bool> = true>
T baseline_floor_mod(T lhs, T rhs) {
  T res = lhs % rhs;
  if (res && ((res < 0 && rhs > 0) || (res > 0 && rhs < 0))) {
    res += rhs;
  }
  return res;
}

template <typename T,
          std::enable_if_t<
              llvm::is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>::value,
              bool> = true>
T baseline_floor_mod(T lhs, T rhs) {
  return lhs % rhs;
}

/// Test the JIT-compiled kernels.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/Int8, int8_t, int8_t, test::DefaultInput<int8_t>(),
    test::DefaultInputNonZero<int8_t>(), baseline_floor_mod,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/Int16, int16_t, int16_t, test::DefaultInput<int16_t>(),
    test::DefaultInputNonZero<int16_t>(), baseline_floor_mod,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/Int64, int64_t, int64_t, test::DefaultInput<int64_t>(),
    test::DefaultInputNonZero<int64_t>(), baseline_floor_mod,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/UInt8, uint8_t, uint8_t, test::DefaultInput<uint8_t>(),
    test::DefaultInputNonZero<uint8_t>(), baseline_floor_mod,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/UInt16, uint16_t, uint16_t, test::DefaultInput<uint16_t>(),
    test::DefaultInputNonZero<uint16_t>(), baseline_floor_mod,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/UInt32, uint32_t, uint32_t, test::DefaultInput<uint32_t>(),
    test::DefaultInputNonZero<uint32_t>(), baseline_floor_mod,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/UInt64, uint64_t, uint64_t, test::DefaultInput<uint64_t>(),
    test::DefaultInputNonZero<uint64_t>(), baseline_floor_mod,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/Half, Eigen::half, Eigen::half,
    test::DefaultInput<Eigen::half>(), test::DefaultInputNonZero<Eigen::half>(),
    baseline_floor_mod, test::OpsTestConfig());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/Float, float, float, test::DefaultInput<float>(),
    test::DefaultInputNonZero<float>(), baseline_floor_mod,
    test::OpsTestConfig().ExpectStrictlyEqual());
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    FloorMod,
    /*test_name=*/Double, double, double, test::DefaultInput<double>(),
    test::DefaultInputNonZero<double>(), baseline_floor_mod,
    test::OpsTestConfig().ExpectStrictlyEqual());
#endif

/// Test `tf.Greater`.

template <typename T>
bool baseline_greater(T lhs, T rhs) {
  return lhs > rhs;
}

GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Half, Eigen::half, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Float, float, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Double, double, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Int8, int8_t, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Int16, int16_t, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/Int64, int64_t, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/UInt8, uint8_t, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/UInt16, uint16_t, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/UInt32, uint32_t, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Greater, /*test_name=*/UInt64, uint64_t, bool,
                       baseline_greater,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.GreaterEqual`.

template <typename T>
bool baseline_greater_equal(T lhs, T rhs) {
  return lhs >= rhs;
}

GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Half, Eigen::half, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Float, float, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Double, double, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Int8, int8_t, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Int16, int16_t, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/Int64, int64_t, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/UInt8, uint8_t, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/UInt16, uint16_t, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/UInt32, uint32_t, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(GreaterEqual, /*test_name=*/UInt64, uint64_t, bool,
                       baseline_greater_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.LeftShift`.

template <typename T>
T baseline_left_shift(T lhs, T rhs) {
  return lhs << rhs;
}

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int8, int8_t, int8_t, test::DefaultInput<int8_t>(),
    test::DefaultInputLessThanBitwidth<int8_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt8, uint8_t, uint8_t,
    test::DefaultInput<uint8_t>(),
    test::DefaultInputLessThanBitwidth<uint8_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int16, int16_t, int16_t,
    test::DefaultInput<int16_t>(),
    test::DefaultInputLessThanBitwidth<int16_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt16, uint16_t, uint16_t,
    test::DefaultInput<uint16_t>(),
    test::DefaultInputLessThanBitwidth<uint16_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int32, int32_t, int32_t,
    test::DefaultInput<int32_t>(),
    test::DefaultInputLessThanBitwidth<int32_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt32, uint32_t, uint32_t,
    test::DefaultInput<uint32_t>(),
    test::DefaultInputLessThanBitwidth<uint32_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int64, int64_t, int64_t,
    test::DefaultInput<int64_t>(),
    test::DefaultInputLessThanBitwidth<int64_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt64, uint64_t, uint64_t,
    test::DefaultInput<uint64_t>(),
    test::DefaultInputLessThanBitwidth<uint64_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.Less`.

template <typename T>
bool baseline_less(T lhs, T rhs) {
  return lhs < rhs;
}

GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Half, Eigen::half, bool,
                       baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Float, float, bool, baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Double, double, bool, baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Int8, int8_t, bool, baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Int16, int16_t, bool, baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/Int64, int64_t, bool, baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/UInt8, uint8_t, bool, baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/UInt16, uint16_t, bool,
                       baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/UInt32, uint32_t, bool,
                       baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Less, /*test_name=*/UInt64, uint64_t, bool,
                       baseline_less,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.LessEqual`.

template <typename T>
bool baseline_less_equal(T lhs, T rhs) {
  return lhs <= rhs;
}

GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Half, Eigen::half, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Float, float, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Double, double, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Int8, int8_t, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Int16, int16_t, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/Int64, int64_t, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/UInt8, uint8_t, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/UInt16, uint16_t, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/UInt32, uint32_t, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(LessEqual, /*test_name=*/UInt64, uint64_t, bool,
                       baseline_less_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.LogicalAnd`.

bool baseline_logical_and(bool lhs, bool rhs) { return lhs && rhs; }

GENERATE_DEFAULT_TESTS(LogicalAnd, /*test_name=*/Bool, bool, bool,
                       baseline_logical_and,
                       test::OpsTestConfig().ExpectStrictlyEqual().NoT())

/// Test `tf.LogicalOr`.

bool baseline_logical_or(bool lhs, bool rhs) { return lhs || rhs; }

GENERATE_DEFAULT_TESTS(LogicalOr, /*test_name=*/Bool, bool, bool,
                       baseline_logical_or,
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
                       baseline_maximum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/Float, float, float,
                       baseline_maximum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/Double, double, double,
                       baseline_maximum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/Int64, int64_t, int64_t,
                       baseline_maximum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_maximum,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test the JIT-compiled kernels.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/Int8, int8_t, int8_t,
                       baseline_maximum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_maximum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_maximum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Maximum, /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_maximum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.Minmum`.

template <typename T>
T baseline_minimum(T lhs, T rhs) {
  if (std::isnan(lhs) || std::isnan(rhs)) {
    return lhs + rhs;
  }
  return std::min(lhs, rhs);
}

GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_minimum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/Float, float, float,
                       baseline_minimum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/Double, double, double,
                       baseline_minimum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/Int64, int64_t, int64_t,
                       baseline_minimum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_minimum,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test the JIT-compiled kernels.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/Int8, int8_t, int8_t,
                       baseline_minimum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_minimum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_minimum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Minimum, /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_minimum,
                       test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.Mul`.

template <typename T>
T baseline_mul(T lhs, T rhs) {
  return lhs * rhs;
}

GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Float, float, float, baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Double, double, double, baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Complex64, std::complex<float>,
                       std::complex<float>, baseline_mul,
                       test::OpsTestConfig().RTol(1e-6).ATol(1e-6))
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Complex128, std::complex<double>,
                       std::complex<double>, baseline_mul,
                       test::OpsTestConfig().RTol(1e-6).ATol(1e-6))
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Int8, int8_t, int8_t, baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Int16, int16_t, int16_t, baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/Int64, int64_t, int64_t, baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/UInt8, uint8_t, uint8_t, baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Mul, /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_mul,
                       test::OpsTestConfig().ExpectStrictlyEqual())

// The following tests don't work with Eigen kernels if the Eigen kernels are
// compiled with nvcc.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
TEST_F(BinaryOpsTest, MulComplex64SpecialCases) {
  TestEqualShapes<std::complex<float>, std::complex<float>, std::complex<float>,
                  std::complex<float>>(
      "Mul", /*shape=*/{67, 63},
      test::NearZeroInfAndNanInput<std::complex<float>>(),
      test::RepeatElements(test::NearZeroInfAndNanInput<std::complex<float>>(),
                           64),
      baseline_mul, test::OpsTestConfig());
}

TEST_F(BinaryOpsTest, MulComplex128SpecialCases) {
  TestEqualShapes<std::complex<double>, std::complex<double>,
                  std::complex<double>, std::complex<double>>(
      "Mul", /*shape=*/{67, 63},
      test::NearZeroInfAndNanInput<std::complex<double>>(),
      test::RepeatElements(test::NearZeroInfAndNanInput<std::complex<double>>(),
                           64),
      baseline_mul, test::OpsTestConfig());
}
#endif

/// Test `tf.MulNoNan`.

template <typename T>
T baseline_mul_no_nan(T lhs, T rhs) {
  return rhs == T(0) ? T(0) : lhs * rhs;
}

GENERATE_DEFAULT_TESTS(MulNoNan,
                       /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_mul_no_nan,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(MulNoNan,
                       /*test_name=*/Float, float, float, baseline_mul_no_nan,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(MulNoNan,
                       /*test_name=*/Double, double, double,
                       baseline_mul_no_nan,
                       test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TESTS(MulNoNan,
                       /*test_name=*/Complex64, std::complex<float>,
                       std::complex<float>, baseline_mul_no_nan,
                       test::OpsTestConfig().ATol(1e-6).RTol(1e-6))
GENERATE_DEFAULT_TESTS(MulNoNan,
                       /*test_name=*/Complex128, std::complex<double>,
                       std::complex<double>, baseline_mul_no_nan,
                       test::OpsTestConfig())

/// Test `tf.NextAfter`.

template <typename T>
T baseline_nextafter(T from, T to) {
  T res = std::nextafter(from, to);
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
  // The Eigen GPU kernel returns the next normal value if ftz is set.
  if (!std::isnormal(res)) {
    if (res < 0 && res > -1) {                // NOLINT
      return -std::numeric_limits<T>::min();  // NOLINT
    }
    if (res > 0 && res < 1) {                // NOLINT
      return std::numeric_limits<T>::min();  // NOLINT
    }
  }
#endif
  return res;
}

GENERATE_DEFAULT_TESTS(NextAfter, /*test_name=*/Float, float, float,
                       baseline_nextafter,
                       test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TESTS(NextAfter, /*test_name=*/Double, double, double,
                       std::nextafter,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.NotEqual`.

template <typename T>
bool baseline_not_equal(T lhs, T rhs) {
  return lhs != rhs;
}

GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Half, Eigen::half, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Float, float, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Double, double, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Bool, bool, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Int8, int8_t, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Int16, int16_t, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/Int64, int64_t, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/UInt8, uint8_t, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/C64, std::complex<float>, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/C128, std::complex<double>, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())

TEST_F(BinaryOpsTest, NotEqualUint8_tSpecialCases) {
  TestEqualShapes<uint8_t, uint8_t, bool, bool>(
      "NotEqual", /*shape=*/{20},
      test::InputAsVector<uint8_t>({255, 1, 0, 0, 1, 255, 0, 255}),
      test::InputAsVector<uint8_t>({1, 255, 0, 1, 0, 0, 255, 255}),
      baseline_not_equal, test::OpsTestConfig().ExpectStrictlyEqual());
}

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/UInt16, uint16_t, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/UInt32, uint32_t, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(NotEqual, /*test_name=*/UInt64, uint64_t, bool,
                       baseline_not_equal,
                       test::OpsTestConfig().ExpectStrictlyEqual())
#endif

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
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Polygamma, /*test_name=*/Double, double, double,
    GetPolygammaValuesN<double>(), GetPolygammaValuesX<double>(),
    baseline_polygamma, test::OpsTestConfig().ATol(1e-11).RTol(1e-2))

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

template <typename T, std::enable_if_t<llvm::is_one_of<T, int8_t, int16_t,
                                                       int32_t, int64_t>::value,
                                       bool> = true>
absl::InlinedVector<T, 10> PowInput() {
  return test::InputAsVector<T, double>({-2, -1, -1, 1, 1, 3});
}

template <>
Eigen::half baseline_pow(Eigen::half lhs, Eigen::half rhs) {
  return static_cast<Eigen::half>(
      std::pow(static_cast<float>(lhs), static_cast<float>(rhs)));
}

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Pow,
    /*test_name=*/Half, Eigen::half, Eigen::half, PowInput<Eigen::half>(),
    PowInput<Eigen::half>(), baseline_pow,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Pow,
    /*test_name=*/Float, float, float, PowInput<float>(), PowInput<float>(),
    baseline_pow, test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Pow,
    /*test_name=*/Double, double, double, PowInput<double>(),
    PowInput<double>(), baseline_pow,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Pow,
    /*test_name=*/Int64, int64_t, int64_t, PowInput<int64_t>(),
    PowInput<int64_t>(), baseline_pow,
    test::OpsTestConfig().ExpectStrictlyEqual())

/// Test the JIT-compiled kernels.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Pow, /*test_name=*/Int8, int8_t, int8_t, PowInput<int8_t>(),
    PowInput<int8_t>(), baseline_pow,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Pow, /*test_name=*/Int16, int16_t, int16_t, PowInput<int16_t>(),
    PowInput<int16_t>(), baseline_pow,
    test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.RealDiv`.

GENERATE_DEFAULT_TESTS(RealDiv,
                       /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_div,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(RealDiv,
                       /*test_name=*/Float, float, float, baseline_div,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(RealDiv,
                       /*test_name=*/Double, double, double, baseline_div,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.ReluGrad`.

template <typename T>
T baseline_relu_grad(T lhs, T rhs) {
  return rhs > T(0) ? lhs : 0;
}

// We cannot compare with strictly equal here, because the Eigen based kernel
// returns -0.0 in some cases where it should return 0.0 (it copies the sign
// from gradients when returning 0, but not for 'remainder' elements).
GENERATE_DEFAULT_NO_BROADCASTING_TESTS_2(
    ReluGrad, /*test_name=*/Half, /*T=*/Eigen::half,
    /*BaselineT=*/float, /*OutT=*/Eigen::half,
    /*BaselineOutT=*/float, test::DefaultInput<Eigen::half>(),
    test::DefaultInput<Eigen::half>(), baseline_relu_grad,
    test::OpsTestConfig())
GENERATE_DEFAULT_NO_BROADCASTING_TESTS(ReluGrad,
                                       /*test_name=*/Float, float, float,
                                       baseline_relu_grad);
GENERATE_DEFAULT_NO_BROADCASTING_TESTS(ReluGrad,
                                       /*test_name=*/Double, double, double,
                                       baseline_relu_grad);

/// Test `tf.RightShift`.

template <typename T>
T baseline_right_shift(T lhs, T rhs) {
  return lhs >> rhs;
}

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift, /*test_name=*/Int8, int8_t, int8_t,
    test::DefaultInput<int8_t>(), test::DefaultInputLessThanBitwidth<int8_t>(),
    baseline_right_shift, test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift, /*test_name=*/UInt8, uint8_t, uint8_t,
    test::DefaultInput<uint8_t>(),
    test::DefaultInputLessThanBitwidth<uint8_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int16, int16_t, int16_t, test::DefaultInput<int16_t>(),
    test::DefaultInputLessThanBitwidth<int16_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift, /*test_name=*/UInt16, uint16_t, uint16_t,
    test::DefaultInput<uint16_t>(),
    test::DefaultInputLessThanBitwidth<uint16_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int32, int32_t, int32_t, test::DefaultInput<int32_t>(),
    test::DefaultInputLessThanBitwidth<int32_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift, /*test_name=*/UInt32, uint32_t, uint32_t,
    test::DefaultInput<uint32_t>(),
    test::DefaultInputLessThanBitwidth<uint32_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int64, int64_t, int64_t, test::DefaultInput<int64_t>(),
    test::DefaultInputLessThanBitwidth<int64_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift, /*test_name=*/UInt64, uint64_t, uint64_t,
    test::DefaultInput<uint64_t>(),
    test::DefaultInputLessThanBitwidth<uint64_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.SquaredDifference`.

template <typename T>
T baseline_squared_difference(T lhs, T rhs) {
  return (lhs - rhs) * (lhs - rhs);
}

GENERATE_DEFAULT_TESTS(SquaredDifference, /*test_name=*/Half, Eigen::half,
                       Eigen::half, baseline_squared_difference,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(SquaredDifference, /*test_name=*/Float, float, float,
                       baseline_squared_difference,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(SquaredDifference, /*test_name=*/Double, double, double,
                       baseline_squared_difference,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(SquaredDifference, /*test_name=*/Int64, int64_t, int64_t,
                       baseline_squared_difference,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.Sub`.

template <typename T>
T baseline_sub(T lhs, T rhs) {
  return lhs - rhs;
}

GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/Float, float, float, baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/Double, double, double, baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/Int64, int64_t, int64_t, baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test the JIT-compiled kernel.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/Int8, int8_t, int8_t, baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/Int16, int16_t, int16_t, baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/Uint8, uint8_t, uint8_t, baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Sub, /*test_name=*/Uint16, uint16_t, uint16_t,
                       baseline_sub,
                       test::OpsTestConfig().ExpectStrictlyEqual())
#endif

TEST_F(BinaryOpsTest, SubUint32SpecialCases) {
  TestEqualShapes<uint32_t, uint32_t, uint32_t, uint32_t>(
      "Sub", /*shape=*/{20},
      test::InputAsVector<uint32_t>(
          {std::numeric_limits<uint32_t>::max(), 0u, 0u, 2u}),
      test::InputAsVector<uint32_t>({std::numeric_limits<uint32_t>::max(),
                                     std::numeric_limits<uint32_t>::max(), 1u,
                                     1u}),
      baseline_sub, test::OpsTestConfig().ExpectStrictlyEqual());
}

/// Test `tf.Xlogy`.

template <typename T>
T baseline_xlogy(T x, T y) {
  return x == T(0) ? x : x * std::log(y);
}

GENERATE_DEFAULT_TESTS_2(Xlogy, /*test_name=*/Half, Eigen::half, float,
                         Eigen::half, float, test::DefaultInput<Eigen::half>(),
                         test::DefaultInput<Eigen::half>(), baseline_xlogy,
                         test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Xlogy, /*test_name=*/Float, float, float, baseline_xlogy,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Xlogy, /*test_name=*/Double, double, double,
                       baseline_xlogy,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Xlogy, /*test_name=*/Complex64, std::complex<float>,
                       std::complex<float>, baseline_xlogy,
                       test::OpsTestConfig().ATol(2e-6).RTol(2e-6))
GENERATE_DEFAULT_TESTS(Xlogy, /*test_name=*/Complex128, std::complex<double>,
                       std::complex<double>, baseline_xlogy,
                       test::OpsTestConfig())

/// Test `tf.Xlog1py`.

template <typename T>
T baseline_xlog1py(T x, T y) {
  return x == T(0) ? x : x * std::log1p(y);
}

template <typename T>
std::complex<T> baseline_xlog1py(std::complex<T> x, std::complex<T> y) {
  return x == std::complex<T>(0) ? x : x * std::log(std::complex<T>(1) + y);
}

GENERATE_DEFAULT_TESTS_2(Xlog1py, /*test_name=*/Half, Eigen::half, float,
                         Eigen::half, float, test::DefaultInput<Eigen::half>(),
                         test::DefaultInput<Eigen::half>(), baseline_xlog1py,
                         test::OpsTestConfig().RTol(1e-2))
GENERATE_DEFAULT_TESTS(Xlog1py, /*test_name=*/Float, float, float,
                       baseline_xlog1py, test::OpsTestConfig().RTol(1e-2))
GENERATE_DEFAULT_TESTS(Xlog1py, /*test_name=*/Double, double, double,
                       baseline_xlog1py, test::OpsTestConfig().RTol(1e-2))
GENERATE_DEFAULT_TESTS(Xlog1py, /*test_name=*/Complex64, std::complex<float>,
                       std::complex<float>, baseline_xlog1py,
                       test::OpsTestConfig().ATol(1e-5).RTol(1e-2))
GENERATE_DEFAULT_TESTS(Xlog1py, /*test_name=*/Complex128, std::complex<double>,
                       std::complex<double>, baseline_xlog1py,
                       test::OpsTestConfig().RTol(1e-2))

/// Test `tf.TruncateDiv`.

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv,
    /*test_name=*/Int16, int16_t, int16_t, test::DefaultInput<int16_t>(),
    test::DefaultInputNonZero<int16_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv,
    /*test_name=*/Int64, int64_t, int64_t, test::DefaultInput<int64_t>(),
    test::DefaultInputNonZero<int64_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv,
    /*test_name=*/UInt8, uint8_t, uint8_t, test::DefaultInput<uint8_t>(),
    test::DefaultInputNonZero<uint8_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    TruncateDiv,
    /*test_name=*/UInt16, uint16_t, uint16_t, test::DefaultInput<uint16_t>(),
    test::DefaultInputNonZero<uint16_t>(), baseline_div,
    test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.Xdivy`.

template <typename T>
T baseline_xdivy(T x, T y) {
  return x == T(0) ? x : x / y;
}

GENERATE_DEFAULT_TESTS_2(Xdivy, /*test_name=*/Half, Eigen::half, float,
                         Eigen::half, float, test::DefaultInput<Eigen::half>(),
                         test::DefaultInput<Eigen::half>(), baseline_xdivy,
                         test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Xdivy, /*test_name=*/Float, float, float, baseline_xdivy,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(Xdivy, /*test_name=*/Double, double, double,
                       baseline_xdivy,
                       test::OpsTestConfig().ExpectStrictlyEqual())

// The following tests don't work with Eigen kernels if the Eigen kernels are
// compiled with nvcc.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TESTS(Xdivy, /*test_name=*/Complex64, std::complex<float>,
                       std::complex<float>, baseline_xdivy,
                       test::OpsTestConfig().ATol(1e-11).RTol(1e-2))
GENERATE_DEFAULT_TESTS(Xdivy, /*test_name=*/Complex128, std::complex<double>,
                       std::complex<double>, baseline_xdivy,
                       test::OpsTestConfig().ATol(1e-11).RTol(1e-2))
#endif

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
