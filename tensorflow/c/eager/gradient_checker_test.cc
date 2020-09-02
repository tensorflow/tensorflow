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
#include "tensorflow/c/eager/gradients_util.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/mnist_gradients_testutil.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

class GradientCheckerTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_SetTracingImplementation(std::get<0>(GetParam()));
  }
};

Status RegisterGradients(GradientRegistry* registry) {
  TF_RETURN_IF_ERROR(registry->Register("MatMul", MatMulRegisterer));
  TF_RETURN_IF_ERROR(
      registry->Register("SparseSoftmaxCrossEntropyWithLogits",
                         SparseSoftmaxCrossEntropyLossRegisterer));
  return Status::OK();
}

TEST_P(GradientCheckerTest, TestGradCheckMatMul) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t A_dims[] = {2, 2};
  float B_vals[] = {.5f, -1.0f, 1.0f, 1.0f};
  int64_t B_dims[] = {2, 2};
  int num_dims = 2;

  AbstractTensorHandlePtr A =
      GetTensorHandleUtilFloat(ctx.get(), A_vals, A_dims, num_dims);
  AbstractTensorHandlePtr B =
      GetTensorHandleUtilFloat(ctx.get(), B_vals, B_dims, num_dims);

  std::vector<AbstractTensorHandle*> inputs;
  inputs.push_back(A.get());
  inputs.push_back(B.get());

  float dapprox[4] = {0};
  Status s = CalcNumericalGrad(ctx.get(), MatMulModel, inputs, dapprox,
                               /*input_index=*/0,
                               /*use_function=*/!std::get<2>(GetParam()), 
                               /*is_scalar_out=*/false);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float expected_dA[4] = {-.5f, 2.0f, -.5f, 2.0f};
  float tolerance = 1e-2;
  for (int j = 0; j < 4; j++) {
    ASSERT_NEAR(dapprox[j], expected_dA[j], tolerance);
  }
}

TEST_P(GradientCheckerTest, TestGradCheckMul) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);

  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s = ScalarTensorHandle(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    Status s = ScalarTensorHandle(ctx.get(), 7.0f, &y_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  // Will perform z = x*y.
  // dz/dx = y

  std::vector<AbstractTensorHandle*> inputs;
  inputs.push_back(x.get());
  inputs.push_back(y.get());
  float dapprox[1] = {0};
  Status s =
      CalcNumericalGrad(ctx.get(), MulModel, inputs, dapprox, /*input_index=*/0,
                        /*use_function=*/!std::get<2>(GetParam()),
                        /*is_scalar_out=*/true);

  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  ASSERT_NEAR(dapprox[0], 7.0f, /*tolerance=*/1e-3);
}

TEST_P(GradientCheckerTest, TestGradCheckSoftmax) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);

  /** Test to show how to use this API with analytical gradients:
   *
   *  We have `SoftmaxLossGradModel`, which is a wrapper for the
   *  Softmax analytical gradient found in c/experimental/nn_grads.
   *
   *  We will use the GradientChecker by applying finite differences
   *  to the forward pass wrapped in `SoftmaxModel` and verify that
   *  both the analytical and numerical gradients are relatively
   *  close.
   *
   */

  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  // X = scores
  float X_vals[] = {1.0f, 2.0f, 3.0f, -5.0f, -4.0f, -3.0f, 2.0f, 0.0f, 1.0f};
  int64_t X_dims[] = {3, 3};
  int num_dims = 2;
  AbstractTensorHandlePtr X =
      GetTensorHandleUtilFloat(ctx.get(), X_vals, X_dims, num_dims);

  // y = labels
  int y_vals[] = {1, 0, 1};
  int64_t y_dims[] = {3};
  num_dims = sizeof(y_dims) / sizeof(y_dims[0]);
  AbstractTensorHandlePtr y =
      GetTensorHandleUtilInt(ctx.get(), y_vals, y_dims, num_dims);

  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  std::vector<AbstractTensorHandle*> inputs;
  inputs.push_back(X.get());
  inputs.push_back(y.get());

  // Run analytical gradient and get its data.
  std::vector<AbstractTensorHandle*> outputs(2);
  s = RunModel(SoftmaxLossGradModel, ctx.get(), absl::MakeSpan(inputs),
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* dX_tensor;
  s = GetValue(outputs[0], &dX_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float danalytical[9] = {0};  // Contains data from analytical gradient.
  memcpy(&danalytical[0], TF_TensorData(dX_tensor),
         TF_TensorByteSize(dX_tensor));

  // Run numerical gradient approximation using the GradientChecker API.
  float dapprox[9] = {0};  // Will contain numerical approximation data.
  s = CalcNumericalGrad(ctx.get(), SoftmaxModel, inputs, dapprox,
                        /*input_index=*/0,
                        /*use_function=*/!std::get<2>(GetParam()), 
                        /*is_scalar_out=*/false);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // Now compare the two implementations:
  for (int j = 0; j < 9; j++) {
    ASSERT_NEAR(dapprox[j], danalytical[j], /*tolerance=*/1e-3);
  }

  // Only Unref() first output as 2nd is nullptr grad for labels
  outputs[0]->Unref();
  TF_DeleteTensor(dX_tensor);
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, GradientCheckerTest,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, GradientCheckerTest,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
