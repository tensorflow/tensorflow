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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"
#include "tensorflow/core/kernels/mlir_generated/base_unary_ops_test.h"

namespace tensorflow {
namespace {

// Test fixture `UnaryOpsTest` that sets the TF device is expected by the TEST
// macros below.
class UnaryOpsTest : public UnaryOpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_cpu(
        tensorflow::DeviceFactory::NewDevice("CPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_CPU, std::move(device_cpu));
  }
};

/// Test `tf.Abs`.

// TODO(b/179242253): Re-enable buffer reuse.
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Abs, DT_HALF, DT_HALF, DT_HALF, DT_HALF,
    test::NearZeroAndExtremeInput<Eigen::half>(), Eigen::numext::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_FLOAT, DT_FLOAT, test::NearZeroAndExtremeInput<float>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_DOUBLE, DT_DOUBLE, test::NearZeroAndExtremeInput<double>(),
    std::abs, test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Abs, DT_INT8, DT_INT32, DT_INT8, DT_INT32,
    test::NearZeroAndExtremeInput<int8_t>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Abs, DT_INT16, DT_INT32, DT_INT16, DT_INT32,
    test::NearZeroAndExtremeInput<int16_t>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_INT32, DT_INT32, test::NearZeroAndExtremeInput<int32_t>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_INT64, DT_INT64, test::NearZeroAndExtremeInput<int64_t>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

/// Test `tf.Angle`.
template <typename T>
typename T::value_type baseline_angle(T x) {
  return std::arg(x);
}

GENERATE_DEFAULT_TEST(Angle, DT_COMPLEX64, DT_FLOAT, baseline_angle,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

GENERATE_DEFAULT_TEST(Angle, DT_COMPLEX128, DT_DOUBLE, baseline_angle,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

/// Test `tf.Ceil`.
GENERATE_DEFAULT_TEST(Ceil, DT_HALF, DT_HALF, Eigen::numext::ceil,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Ceil, DT_FLOAT, DT_FLOAT, Eigen::numext::ceil,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Ceil, DT_DOUBLE, DT_DOUBLE, Eigen::numext::ceil,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Cos`.
GENERATE_DEFAULT_TEST(Cos, DT_HALF, DT_HALF, Eigen::numext::cos,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Cos, DT_FLOAT, DT_FLOAT, Eigen::numext::cos,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Cos, DT_DOUBLE, DT_DOUBLE, Eigen::numext::cos,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Floor`.
GENERATE_DEFAULT_TEST(Floor, DT_HALF, DT_HALF, Eigen::numext::floor,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Floor, DT_FLOAT, DT_FLOAT, Eigen::numext::floor,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Floor, DT_DOUBLE, DT_DOUBLE, Eigen::numext::floor,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Invert`.
template <typename T>
T baseline_invert(T x) {
  return ~x;
}
GENERATE_DEFAULT_TEST(
    Invert, DT_INT8, DT_INT8, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_INT16, DT_INT16, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_INT32, DT_INT32, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_INT64, DT_INT64, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_UINT8, DT_UINT8, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_UINT16, DT_UINT16, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_UINT32, DT_UINT32, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_UINT64, DT_UINT64, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

/// Test `tf.Rsqrt`.
GENERATE_DEFAULT_TEST(Rsqrt, DT_HALF, DT_HALF, Eigen::numext::rsqrt,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Rsqrt, DT_FLOAT, DT_FLOAT, Eigen::numext::rsqrt,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Rsqrt, DT_DOUBLE, DT_DOUBLE, Eigen::numext::rsqrt,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Sin`.
GENERATE_DEFAULT_TEST(Sin, DT_HALF, DT_HALF, Eigen::numext::sin,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Sin, DT_FLOAT, DT_FLOAT, Eigen::numext::sin,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Sin, DT_DOUBLE, DT_DOUBLE, Eigen::numext::sin,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Sqrt`.
GENERATE_DEFAULT_TEST(Sqrt, DT_HALF, DT_HALF, Eigen::numext::sqrt,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Sqrt, DT_FLOAT, DT_FLOAT, Eigen::numext::sqrt,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Sqrt, DT_DOUBLE, DT_DOUBLE, Eigen::numext::sqrt,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Square`.
template <typename T>
T baseline_square(T a) {
  return a * a;
}

GENERATE_DEFAULT_TEST(Square, DT_HALF, DT_HALF, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Square, DT_FLOAT, DT_FLOAT, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Square, DT_DOUBLE, DT_DOUBLE, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(
    Square, DT_INT32, DT_INT32, baseline_square,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Square, DT_INT64, DT_INT64, baseline_square,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Square, DT_COMPLEX64, DT_COMPLEX64, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Square, DT_COMPLEX128, DT_COMPLEX128, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Tan`.
GENERATE_DEFAULT_TEST(Tan, DT_HALF, DT_HALF, Eigen::numext::tan,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Tan, DT_FLOAT, DT_FLOAT, Eigen::numext::tan,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Tan, DT_DOUBLE, DT_DOUBLE, Eigen::numext::tan,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Tanh`.
GENERATE_DEFAULT_TEST(Tanh, DT_FLOAT, DT_FLOAT, Eigen::numext::tanh,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Tanh, DT_DOUBLE, DT_DOUBLE, Eigen::numext::tanh,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST_2(Tanh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        Eigen::numext::tanh,
                        test::OpsTestConfig().NoBufferReuse())

}  // namespace
}  // namespace tensorflow
