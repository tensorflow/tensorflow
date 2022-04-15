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

#ifndef TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_UNARY_OPS_TEST_H_
#define TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_UNARY_OPS_TEST_H_

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_jit_cache.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Base class for `UnaryOpsTest` fixture that has to be defined with a custom TF
// device if you want to use the test macros in this file.
class UnaryOpsTestBase : public OpsTestBase {
 protected:
  // This method should set the TF device, e.g. DEVICE_CPU, DEVICE_GPU.
  void SetUp() override = 0;

  template <typename T, typename OutT>
  void SetOpKernel(const std::string& op_name, const TensorShape& shape,
                   const absl::InlinedVector<T, 10>& input,
                   const test::OpsTestConfig& config) {
    NodeDefBuilder builder("some_name", op_name);
    builder.Input(FakeInput(DataTypeToEnum<T>::v()));
    if (config.add_t) {
      builder.Attr(config.input_attribute, DataTypeToEnum<T>::v());
    }
    if (config.add_tout) {
      builder.Attr(config.output_attribute, DataTypeToEnum<OutT>::v());
    }
    TF_ASSERT_OK(builder.Finalize(node_def()));

    TF_ASSERT_OK(InitOp());
    AddInputFromArray<T>(shape, input);
  }

  template <typename T, typename OutT>
  void RunAndExpectResult(const std::string& op_name, const TensorShape& shape,
                          const absl::InlinedVector<T, 10>& input,
                          const absl::InlinedVector<OutT, 10>& expected_output,
                          const test::OpsTestConfig& config) {
    SetOpKernel<T, OutT>(op_name, shape, input, config);
    TF_ASSERT_OK(RunOpKernel());

    // Assert buffer reuse if expected.
    if (config.expect_buffer_reuse) {
      void* arg_ptr_on_device = context_->input(0).data();
      void* result_ptr_on_device = context_->mutable_output(0)->data();
      ASSERT_EQ(arg_ptr_on_device, result_ptr_on_device);
    }

    // Assert expected results.
    Tensor expected_tensor(allocator(), DataTypeToEnum<OutT>::value, shape);
    test::FillValues<OutT>(&expected_tensor, expected_output);
    if (config.expect_strictly_equal) {
      test::ExpectEqual(expected_tensor, *GetOutput(0),
                        config.supress_tolerance ? test::Tolerance::kNone
                                                 : test::Tolerance::kDefault);
    } else {
      test::ExpectClose(expected_tensor, *GetOutput(0), kAbsoluteTolerance,
                        kRelativeTolerance);
    }

    // For JIT-compiled kernels, expect exactly one entry in the JIT cache for
    // the current test. The cache is not affected by other tests as we always
    // set up a new environment.
    if (config.jit_compilation) {
      ResourceMgr* mgr = context_->resource_manager();
      mlir::kernel_gen::tf_framework::JITCache* cache;
      TF_ASSERT_OK(mgr->Lookup<mlir::kernel_gen::tf_framework::JITCache>(
          mgr->default_container(),
          mlir::kernel_gen::tf_framework::JITCache::kDefaultResourceName,
          &cache));
      core::ScopedUnref cache_ref(cache);
      ASSERT_EQ(cache->Size(), 1);
    }
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineCallback>
  void TestImpl(const std::string& op_name, const TensorShape& shape,
                const absl::InlinedVector<T, 10>& input,
                const BaselineCallback& baseline_callback,
                const test::OpsTestConfig& config) {
    // Prepare inputs and compute expected results.
    CHECK(input.size() <= shape.num_elements());
    auto repeated_input =
        test::RepeatInputToMatchShape(input, shape.num_elements());
    absl::InlinedVector<OutT, 10> expected_output =
        ComputeExpectedOutput<T, BaselineT, OutT>(repeated_input,
                                                  baseline_callback);

    RunAndExpectResult<T, OutT>(op_name, shape, repeated_input, expected_output,
                                config);
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineCallback>
  void Test(const std::string& op_name, const TensorShape& shape,
            const absl::InlinedVector<T, 10>& input,
            const BaselineCallback& baseline_callback,
            const test::OpsTestConfig& config) {
    TestImpl<T, BaselineT, OutT>(op_name, shape, input, baseline_callback,
                                 config);
  }

  // Allow deduction of overloaded function with const ref input.
  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void Test(const std::string& op_name, const TensorShape& shape,
            const absl::InlinedVector<T, 10>& input,
            BaselineOutT (*baseline_callback)(const BaselineT&),
            const test::OpsTestConfig& config) {
    TestImpl<T, BaselineT, OutT>(op_name, shape, input, baseline_callback,
                                 config);
  }

  // Allow deduction of overloaded function with value input.
  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void Test(const std::string& op_name, const TensorShape& shape,
            const absl::InlinedVector<T, 10>& input,
            BaselineOutT (*baseline_callback)(BaselineT),
            const test::OpsTestConfig& config) {
    TestImpl<T, BaselineT, OutT>(op_name, shape, input, baseline_callback,
                                 config);
  }

  template <typename T, typename OutT>
  void TestEmptyShape(const std::string& op_name,
                      const test::OpsTestConfig& config) {
    TensorShape shape{0, 1, 2};
    absl::InlinedVector<T, 10> empty_input = {};
    absl::InlinedVector<OutT, 10> expected_output = {};
    RunAndExpectResult<T, OutT>(op_name, shape, empty_input, expected_output,
                                config);
  }

 private:
  constexpr static double kAbsoluteTolerance = 0.001;
  constexpr static double kRelativeTolerance = 0.001;

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineCallback>
  absl::InlinedVector<OutT, 10> ComputeExpectedOutput(
      absl::InlinedVector<T, 10> input,
      const BaselineCallback& baseline_callback) {
    absl::InlinedVector<OutT, 10> expected_output;
    for (int i = 0; i < input.size(); i++) {
      auto arg = static_cast<BaselineT>(input[i]);
      auto result = static_cast<OutT>(baseline_callback(arg));
      expected_output.push_back(result);
    }
    return expected_output;
  }
};

// Macros to easily generate common test cases. The macros use `UnaryOpsTest`
// fixture in order to share implementation across GPU and CPU platform tests.
// For specific inputs, please define your own test fixtures.
#define GENERATE_DEFAULT_TEST(op_name, InT, OutT, baseline_callback, config) \
  GENERATE_DEFAULT_TEST_2(op_name, InT, InT, OutT, OutT, baseline_callback,  \
                          config)

#define GENERATE_DEFAULT_TEST_2(op_name, InT, BaselineT, OutT, BaselineOutT, \
                                baseline_callback, config)                   \
  GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(                        \
      op_name, InT, BaselineT, OutT, BaselineOutT,                           \
      test::DefaultInput<NativeT>(), baseline_callback, config)

#define GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(        \
    op_name, InT, OutT, input_values, baseline_callback, config) \
  GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(            \
      op_name, InT, InT, OutT, OutT, input_values, baseline_callback, config)

#define GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(                   \
    op_name, InT, BaselineT, OutT, BaselineOutT, input_values,                \
    baseline_callback, config)                                                \
  TEST_F(UnaryOpsTest, op_name##InT##OutT) {                                  \
    using NativeT = EnumToDataType<InT>::Type;                                \
    using NativeBaselineT = EnumToDataType<BaselineT>::Type;                  \
    using NativeOutT = EnumToDataType<OutT>::Type;                            \
    using NativeBaselineOutT = EnumToDataType<BaselineOutT>::Type;            \
    Test<NativeT, NativeBaselineT, NativeOutT, NativeBaselineOutT>(           \
        #op_name, test::DefaultInputShape(), input_values, baseline_callback, \
        config);                                                              \
  }                                                                           \
  TEST_F(UnaryOpsTest, op_name##InT##OutT##EmptyShape) {                      \
    using NativeT = EnumToDataType<InT>::Type;                                \
    using NativeOutT = EnumToDataType<OutT>::Type;                            \
    TestEmptyShape<NativeT, NativeOutT>(#op_name, config);                    \
  }

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_UNARY_OPS_TEST_H_
