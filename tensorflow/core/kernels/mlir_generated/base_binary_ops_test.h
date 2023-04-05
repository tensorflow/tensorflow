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

#ifndef TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_BINARY_OPS_TEST_H_
#define TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_BINARY_OPS_TEST_H_

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Base class for `BinaryOpsTest` fixture that has to be defined with a custom
// TF device if you want to use the test macros in this file.
class BinaryOpsTestBase : public OpsTestBase {
 protected:
  // This method should set the TF device, e.g. DEVICE_CPU, DEVICE_GPU.
  void SetUp() override = 0;

  template <typename T, typename OutT>
  void SetOpKernel(const std::string& op_name, const TensorShape& lhs_shape,
                   const absl::InlinedVector<T, 10>& lhs_input,
                   const TensorShape& rhs_shape,
                   const absl::InlinedVector<T, 10>& rhs_input,
                   const test::OpsTestConfig& config) {
    auto builder = NodeDefBuilder("some_name", op_name)
                       .Input(FakeInput(DataTypeToEnum<T>::v()))
                       .Input(FakeInput(DataTypeToEnum<T>::v()));
    if (config.add_t) {
      builder.Attr(config.input_attribute, DataTypeToEnum<T>::v());
    }
    if (config.add_tout) {
      builder.Attr(config.output_attribute, DataTypeToEnum<OutT>::v());
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
                          const test::OpsTestConfig& config) {
    SetOpKernel<T, OutT>(op_name, lhs_shape, lhs_input, rhs_shape, rhs_input,
                         config);
    TF_ASSERT_OK(RunOpKernel());

    // Compare output to expectation.
    Tensor expected_tensor(allocator(), DataTypeToEnum<OutT>::value,
                           expected_shape);
    test::FillValues<OutT>(&expected_tensor, expected_output);
    if (config.expect_strictly_equal) {
      test::ExpectEqual(expected_tensor, *GetOutput(0),
                        config.supress_tolerance ? test::Tolerance::kNone
                                                 : test::Tolerance::kDefault);
    } else {
      test::ExpectClose(expected_tensor, *GetOutput(0), config.atol,
                        config.rtol);
    }
  }

  template <typename T, typename OutT>
  void RunAndExpectInvalidArgument(const std::string& op_name,
                                   const TensorShape& lhs_shape,
                                   const absl::InlinedVector<T, 10>& lhs_input,
                                   const TensorShape& rhs_shape,
                                   const absl::InlinedVector<T, 10>& rhs_input,
                                   const test::OpsTestConfig& config) {
    SetOpKernel<T, OutT>(op_name, lhs_shape, lhs_input, rhs_shape, rhs_input,
                         config);
    auto status = RunOpKernel();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  }

  // Run common test cases.

  template <typename T, typename OutT>
  void TestIncompatibleShapes(const std::string& op_name,
                              const absl::InlinedVector<T, 10>& lhs_input,
                              const absl::InlinedVector<T, 10>& rhs_input,
                              const test::OpsTestConfig& config) {
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
                       const test::OpsTestConfig& config) {
    // Prepare inputs.
    int64_t input_size = shape.num_elements();
    CHECK(lhs_input.size() <= input_size && rhs_input.size() <= input_size &&
          "expect input shape to hold all input values");
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
                     const test::OpsTestConfig& config) {
    // Prepare inputs.
    TensorShape scalar_shape{};
    CHECK(other_input.size() <= other_shape.num_elements() &&
          "expect other input shape to hold all input values");
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
  void TestOneEffectiveScalar(const std::string& op_name, T scalar_input,
                              const TensorShape& other_shape,
                              const absl::InlinedVector<T, 10>& other_input,
                              BaselineOutT (*baseline_callback)(BaselineT,
                                                                BaselineT),
                              const test::OpsTestConfig& config) {
    // Prepare inputs.
    TensorShape effective_scalar_shape{1, 1, 1, 1, 1, 1, 1};
    CHECK(other_input.size() <= other_shape.num_elements() &&
          "expect other input shape to hold all input values");
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
    TensorShape expected_shape = other_shape;
    while (expected_shape.dims() < effective_scalar_shape.dims()) {
      expected_shape.InsertDim(0, 1);
    }
    RunAndExpectResult<T, OutT>(
        op_name, effective_scalar_shape, scalar_input_vector, other_shape,
        repeated_other_input, expected_shape, expected_output, config);
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void TestBroadcastingExpand(const std::string& op_name,
                              const absl::InlinedVector<T, 10>& lhs_input,
                              const absl::InlinedVector<T, 10>& rhs_input,
                              BaselineOutT (*baseline_callback)(BaselineT,
                                                                BaselineT),
                              const test::OpsTestConfig& config) {
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
                             const test::OpsTestConfig& config) {
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
                        const test::OpsTestConfig& config) {
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
  void TestBroadcastingRank6(const std::string& op_name,
                             const absl::InlinedVector<T, 10>& lhs_input,
                             const absl::InlinedVector<T, 10>& rhs_input,
                             BaselineOutT (*baseline_callback)(BaselineT,
                                                               BaselineT),
                             const test::OpsTestConfig& config) {
    // Prepare inputs.
    TensorShape lhs_shape{1, 2, 3, 1, 2, 1};
    TensorShape rhs_shape{1, 1, 1, 2, 3};
    auto repeated_lhs_input =
        test::RepeatInputToMatchShape(lhs_input, lhs_shape.num_elements());
    auto repeated_rhs_input =
        test::RepeatInputToMatchShape(rhs_input, rhs_shape.num_elements());

    // Compute expected results.
    TensorShape expected_shape{1, 2, 3, 1, 2, 3};
    std::vector<int> lhs_indices = {0, 0, 0, 1, 1, 1, 2,  2,  2,  3,  3,  3,
                                    4, 4, 4, 5, 5, 5, 6,  6,  6,  7,  7,  7,
                                    8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11};
    std::vector<int> rhs_indices = {
        0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
    };
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
                                  const test::OpsTestConfig& config) {
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
    for (int64_t i = 0; i < lhs_indices.size(); i++) {
      auto lhs = static_cast<BaselineT>(lhs_input[lhs_indices[i]]);
      auto rhs = static_cast<BaselineT>(rhs_input[rhs_indices[i]]);
      auto result = static_cast<OutT>(baseline_callback(lhs, rhs));
      expected_output.push_back(result);
    }
    return expected_output;
  }
};

// Macros to easily generate common test cases. The macros use `BinaryOpsTest`
// fixture in order to share implementation across GPU and CPU platform tests.
// For specific inputs, please define your own test fixtures.
#define GENERATE_DEFAULT_NO_BROADCASTING_TESTS_2(                            \
    op_name, test_name, T, BaselineT, OutT, BaselineOutT, lhs_input,         \
    rhs_input, baseline_callback, config)                                    \
  TEST_F(BinaryOpsTest, op_name##EqShapes##test_name) {                      \
    TestEqualShapes<T, BaselineT, OutT, BaselineOutT>(                       \
        #op_name, /*shape=*/test::DefaultInputShape(), lhs_input, rhs_input, \
        baseline_callback, config);                                          \
  }                                                                          \
  TEST_F(BinaryOpsTest, op_name##IncompatibleShapes##test_name) {            \
    TestIncompatibleShapes<T, OutT>(#op_name, lhs_input, rhs_input, config); \
  }

#define GENERATE_DEFAULT_TESTS_2(op_name, test_name, T, BaselineT, OutT,      \
                                 BaselineOutT, lhs_input, rhs_input,          \
                                 baseline_callback, config)                   \
                                                                              \
  GENERATE_DEFAULT_NO_BROADCASTING_TESTS_2(                                   \
      op_name, test_name, T, BaselineT, OutT, BaselineOutT, lhs_input,        \
      rhs_input, baseline_callback, config)                                   \
                                                                              \
  TEST_F(BinaryOpsTest, op_name##OneScalar##test_name) {                      \
    TestOneScalar<T, BaselineT, OutT, BaselineOutT>(                          \
        #op_name, /*scalar_input=*/lhs_input.front(),                         \
        /*other_shape=*/test::DefaultInputShape(), /*other_input=*/rhs_input, \
        baseline_callback, config);                                           \
  }                                                                           \
                                                                              \
  TEST_F(BinaryOpsTest, op_name##TestOneEffectiveScalar##test_name) {         \
    TestOneEffectiveScalar<T, BaselineT, OutT, BaselineOutT>(                 \
        #op_name, /*scalar_input=*/lhs_input.front(),                         \
        /*other_shape=*/test::DefaultInputShape(), /*other_input=*/rhs_input, \
        baseline_callback, config);                                           \
  }                                                                           \
                                                                              \
  TEST_F(BinaryOpsTest, op_name##BroadcastingExpand##test_name) {             \
    TestBroadcastingExpand<T, BaselineT, OutT, BaselineOutT>(                 \
        #op_name, lhs_input, rhs_input, baseline_callback, config);           \
  }                                                                           \
                                                                              \
  TEST_F(BinaryOpsTest, op_name##BroadcastingInDim##test_name) {              \
    TestBroadcastingInDim<T, BaselineT, OutT, BaselineOutT>(                  \
        #op_name, lhs_input, rhs_input, baseline_callback, config);           \
  }                                                                           \
                                                                              \
  TEST_F(BinaryOpsTest, op_name##Broadcasting##test_name) {                   \
    TestBroadcasting<T, BaselineT, OutT, BaselineOutT>(                       \
        #op_name, lhs_input, rhs_input, baseline_callback, config);           \
  }                                                                           \
                                                                              \
  TEST_F(BinaryOpsTest, op_name##BroadcastingRank6##test_name) {              \
    TestBroadcastingRank6<T, BaselineT, OutT, BaselineOutT>(                  \
        #op_name, lhs_input, rhs_input, baseline_callback, config);           \
  }                                                                           \
                                                                              \
  TEST_F(BinaryOpsTest, op_name##EmptyShapeBroadcasting##test_name) {         \
    TestEmptyShapeBroadcasting<T, BaselineT, OutT, BaselineOutT>(             \
        #op_name, lhs_input, rhs_input, config);                              \
  }

#define GENERATE_DEFAULT_TESTS(op_name, test_name, T, OutT, baseline_callback, \
                               config)                                         \
  GENERATE_DEFAULT_TESTS_2(op_name, test_name, T, T, OutT, OutT,               \
                           test::DefaultInput<T>(), test::DefaultInput<T>(),   \
                           baseline_callback, config)

#define GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(                  \
    op_name, test_name, T, OutT, lhs_input, rhs_input, baseline_callback,   \
    config)                                                                 \
  GENERATE_DEFAULT_TESTS_2(op_name, test_name, T, T, OutT, OutT, lhs_input, \
                           rhs_input, baseline_callback, config)

#define GENERATE_DEFAULT_NO_BROADCASTING_TESTS(op_name, test_name, T, OutT, \
                                               baseline_callback)           \
  GENERATE_DEFAULT_NO_BROADCASTING_TESTS_2(                                 \
      op_name, test_name, T, T, OutT, OutT, test::DefaultInput<T>(),        \
      test::DefaultInput<T>(), baseline_callback,                           \
      test::OpsTestConfig().ExpectStrictlyEqual())

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_BINARY_OPS_TEST_H_
