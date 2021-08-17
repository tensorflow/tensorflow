/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <complex>

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
    std::unique_ptr<tensorflow::Device> device_cpu(
        tensorflow::DeviceFactory::NewDevice("CPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_CPU, std::move(device_cpu));
  }
};

/// Test `tf.AddV2`.

template <typename T>
T baseline_add(T lhs, T rhs) {
  return lhs + rhs;
}

GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Float, float, float, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Double, double, double,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int8, int8_t, int8_t, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int16, int16_t, int16_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int32, int32_t, int32_t,
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

/// Test `tf.BitwiseAnd`.
template <typename T>
T baseline_bitwise_and(T lhs, T rhs) {
  return lhs & rhs;
}
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int8, int8_t, int8_t, baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int16, int16_t, int16_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int32, int32_t, int32_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int64, int64_t, int64_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt32, uint32_t, uint32_t,
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
                       /*test_name=*/Int16, int16_t, int16_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int32, int32_t, int32_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int64, int64_t, int64_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt32, uint32_t, uint32_t,
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
                       /*test_name=*/Int16, int16_t, int16_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int32, int32_t, int32_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int64, int64_t, int64_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_bitwise_xor,
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
    LeftShift, /*test_name=*/Int16, int16_t, int16_t,
    test::DefaultInput<int16_t>(),
    test::DefaultInputLessThanBitwidth<int16_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int32, int32_t, int32_t,
    test::DefaultInput<int32_t>(),
    test::DefaultInputLessThanBitwidth<int32_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int64, int64_t, int64_t,
    test::DefaultInput<int64_t>(),
    test::DefaultInputLessThanBitwidth<int64_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt8, uint8_t, uint8_t,
    test::DefaultInput<uint8_t>(),
    test::DefaultInputLessThanBitwidth<uint8_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt16, uint16_t, uint16_t,
    test::DefaultInput<uint16_t>(),
    test::DefaultInputLessThanBitwidth<uint16_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt32, uint32_t, uint32_t,
    test::DefaultInput<uint32_t>(),
    test::DefaultInputLessThanBitwidth<uint32_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt64, uint64_t, uint64_t,
    test::DefaultInput<uint64_t>(),
    test::DefaultInputLessThanBitwidth<uint64_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.RightShift`.
template <typename T>
T baseline_right_shift(T lhs, T rhs) {
  return lhs >> rhs;
}
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int8, int8_t, int8_t, test::DefaultInput<int8_t>(),
    test::DefaultInputLessThanBitwidth<int8_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int16, int16_t, int16_t, test::DefaultInput<int16_t>(),
    test::DefaultInputLessThanBitwidth<int16_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int32, int32_t, int32_t, test::DefaultInput<int32_t>(),
    test::DefaultInputLessThanBitwidth<int32_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int64, int64_t, int64_t, test::DefaultInput<int64_t>(),
    test::DefaultInputLessThanBitwidth<int64_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/UInt8, uint8_t, uint8_t, test::DefaultInput<uint8_t>(),
    test::DefaultInputLessThanBitwidth<uint8_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/UInt16, uint16_t, uint16_t, test::DefaultInput<uint16_t>(),
    test::DefaultInputLessThanBitwidth<uint16_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/UInt32, uint32_t, uint32_t, test::DefaultInput<uint32_t>(),
    test::DefaultInputLessThanBitwidth<uint32_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/UInt64, uint64_t, uint64_t, test::DefaultInput<uint64_t>(),
    test::DefaultInputLessThanBitwidth<uint64_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())

}  // namespace
}  // namespace tensorflow
