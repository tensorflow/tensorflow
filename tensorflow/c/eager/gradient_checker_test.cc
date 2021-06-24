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
#include "tensorflow/c/eager/gradient_checker.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

void CompareNumericalAndManualGradients(
    Model model, AbstractContext* ctx,
    absl::Span<AbstractTensorHandle* const> inputs, int input_index,
    float* expected_grad, int num_grad, bool use_function,
    double abs_error = 1e-2) {
  Status s;
  AbstractTensorHandlePtr numerical_grad;
  {
    AbstractTensorHandle* numerical_grad_raw;
    s = CalcNumericalGrad(ctx, model, inputs, input_index, use_function,
                          &numerical_grad_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    numerical_grad.reset(numerical_grad_raw);
  }

  TF_Tensor* numerical_tensor;
  s = GetValue(numerical_grad.get(), &numerical_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto num_elem_numerical = TF_TensorElementCount(numerical_tensor);
  ASSERT_EQ(num_elem_numerical, num_grad);

  float* dnumerical = new float[num_elem_numerical]{0};
  memcpy(&dnumerical[0], TF_TensorData(numerical_tensor),
         TF_TensorByteSize(numerical_tensor));

  for (int j = 0; j < num_grad; j++) {
    ASSERT_NEAR(dnumerical[j], expected_grad[j], abs_error);
  }
  delete[] dnumerical;
  TF_DeleteTensor(numerical_tensor);
}

Status MatMulModel(AbstractContext* ctx,
                   absl::Span<AbstractTensorHandle* const> inputs,
                   absl::Span<AbstractTensorHandle*> outputs) {
  return ops::MatMul(ctx, inputs[0], inputs[1], &outputs[0],
                     /*transpose_a=*/false,
                     /*transpose_b=*/false, "MatMul");
}

Status MulModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
  return ops::Mul(ctx, inputs[0], inputs[1], &outputs[0], "Mul");
}

// TODO(vnvo2409): Add more tests from `python/ops/gradient_checker_v2_test.py`.
// These tests should not be confused with `[*]_grad_test` which compare the
// result of `gradient_checker` and `[*]_grad`. The tests here test the
// functionality of `gradient_checker` by comparing the result with expected
// manual user-provided gradients.
class GradientCheckerTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());

    {
      Status s = StatusFromTF_Status(status.get());
      CHECK_EQ(errors::OK, s.code()) << s.error_message();
    }

    {
      AbstractContext* ctx_raw = nullptr;
      Status s =
          BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
      ASSERT_EQ(errors::OK, s.code()) << s.error_message();
      ctx_.reset(ctx_raw);
    }

    // Computing numerical gradients with TensorFloat-32 is numerically
    // unstable. Some forward pass tests also fail with TensorFloat-32 due to
    // low tolerances
    enable_tensor_float_32_execution(false);
  }

  AbstractContextPtr ctx_;

 public:
  bool UseMlir() const { return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const { return std::get<2>(GetParam()); }
};

TEST_P(GradientCheckerTest, TestMatMul) {
  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t A_dims[] = {2, 2};
  AbstractTensorHandlePtr A;
  {
    AbstractTensorHandle* A_raw;
    Status s = TestTensorHandleWithDims<float, TF_FLOAT>(ctx_.get(), A_vals,
                                                         A_dims, 2, &A_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    A.reset(A_raw);
  }
  float B_vals[] = {.5f, -1.0f, 1.0f, 1.0f};
  int64_t B_dims[] = {2, 2};
  AbstractTensorHandlePtr B;
  {
    AbstractTensorHandle* B_raw;
    Status s = TestTensorHandleWithDims<float, TF_FLOAT>(ctx_.get(), B_vals,
                                                         B_dims, 2, &B_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    B.reset(B_raw);
  }

  float expected_dA[4] = {-.5f, 2.0f, -.5f, 2.0f};
  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndManualGradients(
      MatMulModel, ctx_.get(), {A.get(), B.get()}, 0, expected_dA, 4,
      UseFunction()));
}

TEST_P(GradientCheckerTest, TestMul) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s =
        TestScalarTensorHandle<float, TF_FLOAT>(ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    Status s =
        TestScalarTensorHandle<float, TF_FLOAT>(ctx_.get(), 7.0f, &y_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  float expected_dx[1] = {7.0f};
  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndManualGradients(
      MulModel, ctx_.get(), {x.get(), y.get()}, 0, expected_dx, 1,
      UseFunction()));
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, GradientCheckerTest,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, GradientCheckerTest,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
