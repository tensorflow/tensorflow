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

#include <initializer_list>
#include <limits>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/mlir_generated/gpu_ops_test_util.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class GpuBinaryOpTest : public OpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }

  template <typename T, typename OutT>
  void SetOpKernel(const std::string& op_name, const TensorShape& lhs_shape,
                   const absl::InlinedVector<T, 10>& lhs_input,
                   const TensorShape& rhs_shape,
                   const absl::InlinedVector<T, 10>& rhs_input, bool add_t,
                   bool add_tout) {
    auto builder = NodeDefBuilder("some_name", op_name)
                       .Input(FakeInput(DataTypeToEnum<T>::v()))
                       .Input(FakeInput(DataTypeToEnum<T>::v()));
    if (add_t) {
      builder.Attr("T", DataTypeToEnum<T>::v());
    }
    if (add_tout) {
      builder.Attr("Tout", DataTypeToEnum<OutT>::v());
    }
    TF_ASSERT_OK(builder.Finalize(node_def()));

    TF_ASSERT_OK(InitOp());
    AddInputFromArray<T>(lhs_shape, lhs_input);
    AddInputFromArray<T>(rhs_shape, rhs_input);
  }

  // Run fully specified tests.

  template <typename T, typename OutT>
  void RunAndExpectResult(const std::string& op_name,
                          const TensorShape& lhs_shape,
                          const absl::InlinedVector<T, 10>& lhs_input,
                          const TensorShape& rhs_shape,
                          const absl::InlinedVector<T, 10>& rhs_input,
                          const TensorShape& expected_shape,
                          const absl::InlinedVector<OutT, 10>& expected_output,
                          const test::GpuOpsTestConfig& config) {
    SetOpKernel<T, OutT>(op_name, lhs_shape, lhs_input, rhs_shape, rhs_input,
                         config.add_t, config.add_tout);
    TF_ASSERT_OK(RunOpKernel());

    // Compare output to expectation.
    Tensor expected_tensor(allocator(), DataTypeToEnum<OutT>::value,
                           expected_shape);
    test::FillValues<OutT>(&expected_tensor, expected_output);
    if (config.expect_strictly_equal) {
      test::ExpectEqual(expected_tensor, *GetOutput(0));
    } else {
      test::ExpectClose(expected_tensor, *GetOutput(0));
    }
  }

  template <typename T, typename OutT>
  void RunAndExpectInvalidArgument(const std::string& op_name,
                                   const TensorShape& lhs_shape,
                                   const absl::InlinedVector<T, 10>& lhs_input,
                                   const TensorShape& rhs_shape,
                                   const absl::InlinedVector<T, 10>& rhs_input,
                                   const test::GpuOpsTestConfig& config) {
    SetOpKernel<T, OutT>(op_name, lhs_shape, lhs_input, rhs_shape, rhs_input,
                         config.add_t, config.add_tout);
    auto status = RunOpKernel();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  }

  // Run common test cases.

  template <typename T, typename OutT>
  void TestIncompatibleShapes(const std::string& op_name,
                              const absl::InlinedVector<T, 10>& lhs_input,
                              const absl::InlinedVector<T, 10>& rhs_input,
                              const test::GpuOpsTestConfig& config) {
    // Prepare incompatibly shaped inputs.
    TensorShape lhs_shape{3};
    TensorShape rhs_shape{2};
    auto repeated_lhs_input =
        test::RepeatInputToMatchShape(lhs_input, lhs_shape.num_elements());
    auto repeated_rhs_input =
        test::RepeatInputToMatchShape(rhs_input, rhs_shape.num_elements());

    RunAndExpectInvalidArgument<T, OutT>(op_name, lhs_shape, repeated_lhs_input,
                                         rhs_shape, repeated_rhs_input, config);
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void TestEqualShapes(const std::string& op_name, const TensorShape& shape,
                       const absl::InlinedVector<T, 10>& lhs_input,
                       const absl::InlinedVector<T, 10>& rhs_input,
                       BaselineOutT (*baseline_callback)(BaselineT, BaselineT),
                       const test::GpuOpsTestConfig& config) {
    // Prepare inputs.
    int input_size = shape.num_elements();
    auto repeated_lhs_input =
        test::RepeatInputToMatchShape(lhs_input, input_size);
    auto repeated_rhs_input =
        test::RepeatInputToMatchShape(rhs_input, input_size);

    // Compute expected results.
    absl::InlinedVector<OutT, 10> expected_output;
    for (auto it_lhs = repeated_lhs_input.begin(),
              it_rhs = repeated_rhs_input.begin(),
              end = repeated_lhs_input.end();
         it_lhs != end; ++it_lhs, ++it_rhs) {
      auto lhs = static_cast<BaselineT>(*it_lhs);
      auto rhs = static_cast<BaselineT>(*it_rhs);
      auto result = static_cast<OutT>(baseline_callback(lhs, rhs));
      expected_output.push_back(result);
    }

    RunAndExpectResult<T, OutT>(op_name, shape, repeated_lhs_input, shape,
                                repeated_rhs_input, shape, expected_output,
                                config);
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void TestOneScalar(const std::string& op_name, T scalar_input,
                     const TensorShape& other_shape,
                     const absl::InlinedVector<T, 10>& other_input,
                     BaselineOutT (*baseline_callback)(BaselineT, BaselineT),
                     const test::GpuOpsTestConfig& config) {
    // Prepare inputs.
    TensorShape scalar_shape{};
    auto repeated_other_input =
        test::RepeatInputToMatchShape(other_input, other_shape.num_elements());

    // Compute expected results.
    absl::InlinedVector<OutT, 10> expected_output;
    for (auto it = repeated_other_input.begin(),
              end = repeated_other_input.end();
         it != end; ++it) {
      auto scalar = static_cast<BaselineT>(scalar_input);
      auto other_value = static_cast<BaselineT>(*it);
      auto result = static_cast<OutT>(baseline_callback(scalar, other_value));
      expected_output.push_back(result);
    }

    auto scalar_input_vector = test::InputAsVector<T>({scalar_input});
    RunAndExpectResult<T, OutT>(op_name, scalar_shape, scalar_input_vector,
                                other_shape, repeated_other_input,
                                /*expected_shape=*/other_shape, expected_output,
                                config);
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void TestBroadcastingExpand(const std::string& op_name,
                              const absl::InlinedVector<T, 10>& lhs_input,
                              const absl::InlinedVector<T, 10>& rhs_input,
                              BaselineOutT (*baseline_callback)(BaselineT,
                                                                BaselineT),
                              const test::GpuOpsTestConfig& config) {
    // Prepare inputs.
    TensorShape lhs_shape{1};
    TensorShape rhs_shape{6};
    auto repeated_lhs_input =
        test::RepeatInputToMatchShape(lhs_input, lhs_shape.num_elements());
    auto repeated_rhs_input =
        test::RepeatInputToMatchShape(rhs_input, rhs_shape.num_elements());

    // Compute expected results.
    std::vector<int> lhs_indices = {0, 0, 0, 0, 0, 0};
    std::vector<int> rhs_indices = {0, 1, 2, 3, 4, 5};
    auto expected_output =
        ComputeExpectedOutput<T, BaselineT, OutT, BaselineOutT>(
            lhs_indices, repeated_lhs_input, rhs_indices, repeated_rhs_input,
            baseline_callback);

    RunAndExpectResult<T, OutT>(
        op_name, lhs_shape, repeated_lhs_input, rhs_shape, repeated_rhs_input,
        /*expected_shape=*/rhs_shape, expected_output, config);
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void TestBroadcastingInDim(const std::string& op_name,
                             const absl::InlinedVector<T, 10>& lhs_input,
                             const absl::InlinedVector<T, 10>& rhs_input,
                             BaselineOutT (*baseline_callback)(BaselineT,
                                                               BaselineT),
                             const test::GpuOpsTestConfig& config) {
    // Prepare inputs.
    TensorShape lhs_shape{3};
    TensorShape rhs_shape{2, 3};
    auto repeated_lhs_input =
        test::RepeatInputToMatchShape(lhs_input, lhs_shape.num_elements());
    auto repeated_rhs_input =
        test::RepeatInputToMatchShape(rhs_input, rhs_shape.num_elements());

    // Compute expected results.
    std::vector<int> lhs_indices = {0, 1, 2, 0, 1, 2};
    std::vector<int> rhs_indices = {0, 1, 2, 3, 4, 5};
    auto expected_output =
        ComputeExpectedOutput<T, BaselineT, OutT, BaselineOutT>(
            lhs_indices, repeated_lhs_input, rhs_indices, repeated_rhs_input,
            baseline_callback);

    RunAndExpectResult<T, OutT>(
        op_name, lhs_shape, repeated_lhs_input, rhs_shape, repeated_rhs_input,
        /*expected_shape=*/rhs_shape, expected_output, config);
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void TestBroadcasting(const std::string& op_name,
                        const absl::InlinedVector<T, 10>& lhs_input,
                        const absl::InlinedVector<T, 10>& rhs_input,
                        BaselineOutT (*baseline_callback)(BaselineT, BaselineT),
                        const test::GpuOpsTestConfig& config) {
    // Prepare inputs.
    TensorShape lhs_shape{2, 1};
    TensorShape rhs_shape{3};
    auto repeated_lhs_input =
        test::RepeatInputToMatchShape(lhs_input, lhs_shape.num_elements());
    auto repeated_rhs_input =
        test::RepeatInputToMatchShape(rhs_input, rhs_shape.num_elements());

    // Compute expected results.
    TensorShape expected_shape{2, 3};
    std::vector<int> lhs_indices = {0, 0, 0, 1, 1, 1};
    std::vector<int> rhs_indices = {0, 1, 2, 0, 1, 2};
    auto expected_output =
        ComputeExpectedOutput<T, BaselineT, OutT, BaselineOutT>(
            lhs_indices, repeated_lhs_input, rhs_indices, repeated_rhs_input,
            baseline_callback);

    RunAndExpectResult<T, OutT>(op_name, lhs_shape, repeated_lhs_input,
                                rhs_shape, repeated_rhs_input, expected_shape,
                                expected_output, config);
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void TestEmptyShapeBroadcasting(const std::string& op_name,
                                  const absl::InlinedVector<T, 10>& lhs_input,
                                  const absl::InlinedVector<T, 10>& rhs_input,
                                  const test::GpuOpsTestConfig& config) {
    // Prepare inputs.
    TensorShape lhs_shape{2, 0, 1};
    TensorShape rhs_shape{2, 0, 5};
    absl::InlinedVector<T, 10> empty_input = {};

    // Define expected result.
    TensorShape expected_shape{2, 0, 5};
    absl::InlinedVector<OutT, 10> expected_output = {};

    RunAndExpectResult<T, OutT>(op_name, lhs_shape, empty_input, rhs_shape,
                                empty_input, expected_shape, expected_output,
                                config);
  }

 private:
  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  absl::InlinedVector<OutT, 10> ComputeExpectedOutput(
      std::vector<int> lhs_indices, absl::InlinedVector<T, 10> lhs_input,
      std::vector<int> rhs_indices, absl::InlinedVector<T, 10> rhs_input,
      BaselineOutT (*baseline_callback)(BaselineT, BaselineT)) {
    absl::InlinedVector<OutT, 10> expected_output;
    for (int i = 0; i < lhs_indices.size(); i++) {
      auto lhs = static_cast<BaselineT>(lhs_input[lhs_indices[i]]);
      auto rhs = static_cast<BaselineT>(rhs_input[rhs_indices[i]]);
      auto result = static_cast<OutT>(baseline_callback(lhs, rhs));
      expected_output.push_back(result);
    }
    return expected_output;
  }
};

// Macros to easily generate common test cases. For specific inputs, please
// define your own test fixtures.

#define GENERATE_DEFAULT_TESTS_2(op_name, test_name, T, BaselineT, OutT,      \
                                 BaselineOutT, lhs_input, rhs_input,          \
                                 baseline_callback, config)                   \
  TEST_F(GpuBinaryOpTest, op_name##EqShapes##test_name) {                     \
    TestEqualShapes<T, BaselineT, OutT, BaselineOutT>(                        \
        #op_name, /*shape=*/test::DefaultInputShape(), lhs_input, rhs_input,  \
        baseline_callback, config);                                           \
  }                                                                           \
                                                                              \
  TEST_F(GpuBinaryOpTest, op_name##OneScalar##test_name) {                    \
    TestOneScalar<T, BaselineT, OutT, BaselineOutT>(                          \
        #op_name, /*scalar_input=*/lhs_input.front(),                         \
        /*other_shape=*/test::DefaultInputShape(), /*other_input=*/rhs_input, \
        baseline_callback, config);                                           \
  }                                                                           \
                                                                              \
  TEST_F(GpuBinaryOpTest, op_name##IncompatibleShapes##test_name) {           \
    TestIncompatibleShapes<T, OutT>(#op_name, lhs_input, rhs_input, config);  \
  }                                                                           \
                                                                              \
  TEST_F(GpuBinaryOpTest, op_name##BroadcastingExpand##test_name) {           \
    TestBroadcastingExpand<T, BaselineT, OutT, BaselineOutT>(                 \
        #op_name, lhs_input, rhs_input, baseline_callback, config);           \
  }                                                                           \
                                                                              \
  TEST_F(GpuBinaryOpTest, op_name##BroadcastingInDim##test_name) {            \
    TestBroadcastingInDim<T, BaselineT, OutT, BaselineOutT>(                  \
        #op_name, lhs_input, rhs_input, baseline_callback, config);           \
  }                                                                           \
                                                                              \
  TEST_F(GpuBinaryOpTest, op_name##Broadcasting##test_name) {                 \
    TestBroadcasting<T, BaselineT, OutT, BaselineOutT>(                       \
        #op_name, lhs_input, rhs_input, baseline_callback, config);           \
  }                                                                           \
                                                                              \
  TEST_F(GpuBinaryOpTest, op_name##EmptyShapeBroadcasting##test_name) {       \
    TestEmptyShapeBroadcasting<T, BaselineT, OutT, BaselineOutT>(             \
        #op_name, lhs_input, rhs_input, config);                              \
  }

#define GENERATE_DEFAULT_TESTS(op_name, test_name, T, OutT, baseline_callback) \
  GENERATE_DEFAULT_TESTS_2(op_name, test_name, T, T, OutT, OutT,               \
                           test::DefaultInput<T>(), test::DefaultInput<T>(),   \
                           baseline_callback,                                  \
                           test::GpuOpsTestConfig().ExpectStrictlyEqual())

#define GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(                  \
    op_name, test_name, T, OutT, lhs_input, rhs_input, baseline_callback)   \
  GENERATE_DEFAULT_TESTS_2(op_name, test_name, T, T, OutT, OutT, lhs_input, \
                           rhs_input, baseline_callback,                    \
                           test::GpuOpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.AddV2`.

template <typename T>
T baseline_add(T lhs, T rhs) {
  return lhs + rhs;
}

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
TEST_F(GpuBinaryOpTest, Atan2FloatSpecialCases) {
  TestEqualShapes<float, float, float, float>(
      "Atan2", /*shape=*/{20},
      test::InputAsVector<float>({1, 1, 1, 0, -1, -1, -1, 0}),
      test::InputAsVector<float>({1, 0, -1, -1, -1, 0, 1, 1}), std::atan2,
      test::GpuOpsTestConfig().ExpectStrictlyEqual());
}
TEST_F(GpuBinaryOpTest, Atan2DoubleSpecialCases) {
  TestEqualShapes<double, double, double, double>(
      "Atan2", /*shape=*/{20},
      test::InputAsVector<double>({1, 1, 1, 0, -1, -1, -1, 0}),
      test::InputAsVector<double>({1, 0, -1, -1, -1, 0, 1, 1}), std::atan2,
      test::GpuOpsTestConfig().ExpectStrictlyEqual());
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

GENERATE_DEFAULT_TESTS_2(
    Complex,
    /*test_name=*/C64, float, float, std::complex<float>, std::complex<float>,
    test::DefaultInput<float>(), test::DefaultInput<float>(), baseline_complex,
    test::GpuOpsTestConfig().ExpectStrictlyEqual().AddTout())
GENERATE_DEFAULT_TESTS_2(
    Complex,
    /*test_name=*/C128, double, double, std::complex<double>,
    std::complex<double>, test::DefaultInput<double>(),
    test::DefaultInput<double>(), baseline_complex,
    test::GpuOpsTestConfig().ExpectStrictlyEqual().AddTout())

/// Test `tf.Div`.

template <typename T>
T baseline_div(T lhs, T rhs) {
  return lhs / rhs;
}

GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Half, Eigen::half, Eigen::half,
    test::DefaultInput<Eigen::half>(), test::DefaultInputNonZero<Eigen::half>(),
    baseline_div);
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Float, float, float, test::DefaultInput<float>(),
    test::DefaultInputNonZero<float>(), baseline_div);
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    Div,
    /*test_name=*/Double, double, double, test::DefaultInput<double>(),
    test::DefaultInputNonZero<double>(), baseline_div);
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
                         test::GpuOpsTestConfig().ExpectStrictlyEqual().NoT())

/// Test `tf.LogicalOr`.

bool baseline_logical_or(bool lhs, bool rhs) { return lhs || rhs; }

GENERATE_DEFAULT_TESTS_2(LogicalOr, /*test_name=*/Bool, /*T=*/bool,
                         /*BaselineT=*/bool, /*OutT=*/bool,
                         /*BaselineOutT=*/bool, test::DefaultInput<bool>(),
                         test::DefaultInput<bool>(), baseline_logical_or,
                         test::GpuOpsTestConfig().ExpectStrictlyEqual().NoT())

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

}  // namespace
}  // end namespace tensorflow
