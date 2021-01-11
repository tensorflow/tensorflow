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
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/gradients_util.h"
#include "tensorflow/c/eager/mnist_gradients_testutil.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {
using tensorflow::TF_StatusPtr;

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(errors::OK, s.code()) << s.error_message();

    // Computing numerical gradients with TensorFloat-32 is numerically
    // unstable. Some forward pass tests also fail with TensorFloat-32 due to
    // low tolerances
    enable_tensor_float_32_execution(false);
  }
};

Status RegisterGradients(GradientRegistry* registry) {
  TF_RETURN_IF_ERROR(registry->Register("Add", AddRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Exp", ExpRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("MatMul", MatMulRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Relu", ReluRegisterer));
  TF_RETURN_IF_ERROR(
      registry->Register("SparseSoftmaxCrossEntropyWithLogits",
                         SparseSoftmaxCrossEntropyWithLogitsRegisterer));
  return Status::OK();
}

TEST_P(CppGradients, TestMatMulGrad) {
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

  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  /* Pseudo-code:
   *
   * tape.watch(A)
   * tape.watch(B)
   * Y = AB
   * outputs = tape.gradient(Y, [A, B])
   */

  std::vector<AbstractTensorHandle*> outputs(2);
  s = RunModel(MatMulGradModel, ctx.get(), {A.get(), B.get()},
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* dA_tensor;
  s = GetValue(outputs[0], &dA_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float result_data[4] = {0};
  memcpy(&result_data[0], TF_TensorData(dA_tensor),
         TF_TensorByteSize(dA_tensor));

  float expected_dA[4] = {-.5f, 2.0f, -.5f, 2.0f};
  float tolerance = 1e-3;
  for (int j = 0; j < 4; j++) {
    ASSERT_NEAR(result_data[j], expected_dA[j], tolerance);
  }

  TF_Tensor* dB_tensor;
  s = GetValue(outputs[1], &dB_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  memcpy(&result_data[0], TF_TensorData(dB_tensor),
         TF_TensorByteSize(dB_tensor));

  float expected_dB[4] = {4.0f, 4.0f, 6.0f, 6.0f};
  for (int j = 0; j < 4; j++) {
    ASSERT_NEAR(result_data[j], expected_dB[j], tolerance);
  }

  outputs[0]->Unref();
  outputs[1]->Unref();
  TF_DeleteTensor(dA_tensor);
  TF_DeleteTensor(dB_tensor);
}

TEST_P(CppGradients, TestMNISTForward) {
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  // X = data
  float X_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t dims[] = {2, 2};
  int num_dims = 2;
  AbstractTensorHandlePtr X =
      GetTensorHandleUtilFloat(ctx.get(), X_vals, dims, num_dims);

  // W1 = first weights
  float W1_vals[] = {-1.0f, 10.0f, .5f, 1.0f};
  AbstractTensorHandlePtr W1 =
      GetTensorHandleUtilFloat(ctx.get(), W1_vals, dims, num_dims);

  // W2 = second weights
  float W2_vals[] = {.1f, .2f, .3f, -.5f};
  AbstractTensorHandlePtr W2 =
      GetTensorHandleUtilFloat(ctx.get(), W2_vals, dims, num_dims);

  // y = labels
  int y_vals[] = {1, 1};
  int64_t dims_y[] = {2};
  num_dims = sizeof(dims_y) / sizeof(dims_y[0]);
  AbstractTensorHandlePtr y =
      GetTensorHandleUtilInt(ctx.get(), y_vals, dims, num_dims);

  GradientRegistry registry;

  // Run the Forward Pass
  std::vector<AbstractTensorHandle*> outputs(2);
  Status s =
      RunModel(MNISTForwardModel, ctx.get(),
               {X.get(), W1.get(), W2.get(), y.get()}, absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // Verify the Results
  TF_Tensor* scores_tensor;
  s = GetValue(outputs[0], &scores_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float result_data[4] = {0};
  memcpy(&result_data[0], TF_TensorData(scores_tensor),
         TF_TensorByteSize(scores_tensor));

  float expected_scores[4] = {3.6f, -6.0f, 10.2f, -17.0f};
  float tolerance = 1e-3;
  for (int j = 0; j < 4; j++) {
    ASSERT_NEAR(result_data[j], expected_scores[j], tolerance);
  }

  TF_Tensor* loss_vals_tensor;
  s = GetValue(outputs[1], &loss_vals_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  memcpy(&result_data[0], TF_TensorData(loss_vals_tensor),
         TF_TensorByteSize(loss_vals_tensor));
  float expected_losses[2] = {9.6f, 27.2f};
  for (int j = 0; j < 2; j++) {
    ASSERT_NEAR(result_data[j], expected_losses[j], tolerance);
  }

  outputs[0]->Unref();
  outputs[1]->Unref();
  TF_DeleteTensor(scores_tensor);
  TF_DeleteTensor(loss_vals_tensor);
}

TEST_P(CppGradients, TestMNISTForward2) {
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  // X = data
  float X_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t X_dims[] = {3, 2};
  int num_dims = 2;
  AbstractTensorHandlePtr X =
      GetTensorHandleUtilFloat(ctx.get(), X_vals, X_dims, num_dims);

  // W1 = first weights
  float W1_vals[] = {-1.0f, 10.0f, .5f, 1.0f};
  int64_t dims[] = {2, 2};
  AbstractTensorHandlePtr W1 =
      GetTensorHandleUtilFloat(ctx.get(), W1_vals, dims, num_dims);

  // W2 = second weights
  float W2_vals[] = {.1f, .2f, .3f, -.5f};
  AbstractTensorHandlePtr W2 =
      GetTensorHandleUtilFloat(ctx.get(), W2_vals, dims, num_dims);

  // y = labels
  int y_vals[] = {1, 1, 1};
  int64_t y_dims[] = {3};
  num_dims = sizeof(y_dims) / sizeof(y_dims[0]);
  AbstractTensorHandlePtr y =
      GetTensorHandleUtilInt(ctx.get(), y_vals, y_dims, num_dims);

  GradientRegistry registry;

  // Run the Forward Pass
  std::vector<AbstractTensorHandle*> outputs(2);
  Status s =
      RunModel(MNISTForwardModel, ctx.get(),
               {X.get(), W1.get(), W2.get(), y.get()}, absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // Verify the Results
  TF_Tensor* scores_tensor;
  s = GetValue(outputs[0], &scores_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float result_data[6] = {0};
  memcpy(&result_data[0], TF_TensorData(scores_tensor),
         TF_TensorByteSize(scores_tensor));

  float expected_scores[6] = {3.6f, -6.0f, 10.2f, -17.0f, 16.8f, -28.0f};
  float tolerance = 1e-3;
  for (int j = 0; j < 6; j++) {
    ASSERT_NEAR(result_data[j], expected_scores[j], tolerance);
  }

  TF_Tensor* loss_vals_tensor;
  s = GetValue(outputs[1], &loss_vals_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  memcpy(&result_data[0], TF_TensorData(loss_vals_tensor),
         TF_TensorByteSize(loss_vals_tensor));
  float expected_losses[3] = {9.6f, 27.2f, 44.8f};
  for (int j = 0; j < 3; j++) {
    ASSERT_NEAR(result_data[j], expected_losses[j], tolerance);
  }

  outputs[0]->Unref();
  outputs[1]->Unref();
  TF_DeleteTensor(scores_tensor);
  TF_DeleteTensor(loss_vals_tensor);
}

TEST_P(CppGradients, TestMatMulTranspose) {
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

  // X = data
  float X_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t X_dims[] = {2, 3};
  int num_dims = 2;
  AbstractTensorHandlePtr X =
      GetTensorHandleUtilFloat(ctx.get(), X_vals, X_dims, num_dims);

  // W1 = first weights
  float W1_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t dims[] = {2, 2};
  AbstractTensorHandlePtr W1 =
      GetTensorHandleUtilFloat(ctx.get(), W1_vals, dims, num_dims);

  GradientRegistry registry;

  // Run the MatMul Op
  std::vector<AbstractTensorHandle*> outputs(1);

  Status s = RunModel(MatMulTransposeModel, ctx.get(), {X.get(), W1.get()},
                      absl::MakeSpan(outputs),
                      /*use_function=*/!std::get<2>(GetParam()), registry);

  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // Verify the Results
  TF_Tensor* scores_tensor;
  s = GetValue(outputs[0], &scores_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float result_data[6] = {0};
  memcpy(&result_data[0], TF_TensorData(scores_tensor),
         TF_TensorByteSize(scores_tensor));

  float expected_scores[6] = {13.0f, 18.0f, 17.0f, 24.0f, 21.0f, 30.0f};
  float tolerance = 1e-3;
  for (int j = 0; j < 6; j++) {
    ASSERT_NEAR(result_data[j], expected_scores[j], tolerance);
  }
}

TEST_P(CppGradients, TestMNISTGrad) {
  bool use_function = !std::get<2>(GetParam());
  if (use_function) {
    // TODO(b/168850692): Enable this.
    GTEST_SKIP() << "Can't take gradient of "
                    "SparseSoftmaxCrossEntropyWithLogits in tracing mode.";
  }
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

  // X = data
  float X_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t X_dims[] = {2, 2};
  int num_dims = 2;
  AbstractTensorHandlePtr X =
      GetTensorHandleUtilFloat(ctx.get(), X_vals, X_dims, num_dims);

  // W1 = first weights
  float W1_vals[] = {-1.0f, 10.0f, .5f, 1.0f};
  int64_t dims[] = {2, 2};
  AbstractTensorHandlePtr W1 =
      GetTensorHandleUtilFloat(ctx.get(), W1_vals, dims, num_dims);

  // W2 = second weights
  float W2_vals[] = {.1f, .2f, .3f, -.5f};
  AbstractTensorHandlePtr W2 =
      GetTensorHandleUtilFloat(ctx.get(), W2_vals, dims, num_dims);

  // y = labels
  int y_vals[] = {1, 1};
  int64_t y_dims[] = {2};
  num_dims = sizeof(y_dims) / sizeof(y_dims[0]);
  AbstractTensorHandlePtr y =
      GetTensorHandleUtilInt(ctx.get(), y_vals, y_dims, num_dims);

  // Register Grads
  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  /* Pseudo-code:
   *
   *
   * tape.watch(W1)
   * tape.watch(W2)
   * mm = X*W1
   * hidden = Relu(mm)
   * scores = W2*hidden
   * loss = SoftmaxLoss(scores, y)
   * outputs = tape.gradient(loss, [A, B])
   *
   */

  std::vector<AbstractTensorHandle*> outputs(3);
  s = RunModel(MNISTGradModel, ctx.get(),
               {X.get(), W1.get(), W2.get(), y.get()}, absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float tolerance = 1e-3;
  TF_Tensor* dW1_tensor;
  s = GetValue(outputs[0], &dW1_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float result_data[4] = {0};
  memcpy(&result_data[0], TF_TensorData(dW1_tensor),
         TF_TensorByteSize(dW1_tensor));

  float expected_dW1[4] = {0.0f, 3.2f, 0.0f, 4.8f};
  for (int j = 0; j < 4; j++) {
    ASSERT_NEAR(result_data[j], expected_dW1[j], tolerance);
  }

  TF_Tensor* dW2_tensor;
  s = GetValue(outputs[1], &dW2_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  memcpy(&result_data[0], TF_TensorData(dW2_tensor),
         TF_TensorByteSize(dW2_tensor));

  float expected_dW2[4] = {0.0f, 0.0f, 46.0f, -46.0f};  // dLoss
  for (int j = 0; j < 4; j++) {
    ASSERT_NEAR(result_data[j], expected_dW2[j], tolerance);
  }

  outputs[0]->Unref();
  outputs[1]->Unref();
  outputs[2]->Unref();
  TF_DeleteTensor(dW1_tensor);
  TF_DeleteTensor(dW2_tensor);
}

TEST_P(CppGradients, TestScalarMul) {
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

  AbstractTensorHandlePtr eta;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s = ScalarTensorHandle(ctx.get(), 1.5f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    eta.reset(x_raw);
  }

  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t A_dims[] = {2, 2};
  int num_dims = 2;

  AbstractTensorHandlePtr A =
      GetTensorHandleUtilFloat(ctx.get(), A_vals, A_dims, num_dims);

  GradientRegistry registry;
  std::vector<AbstractTensorHandle*> outputs(1);
  Status s = RunModel(ScalarMulModel, ctx.get(), {eta.get(), A.get()},
                      absl::MakeSpan(outputs),
                      /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* dA_tensor;
  s = GetValue(outputs[0], &dA_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float result_data[4] = {0};
  memcpy(&result_data[0], TF_TensorData(dA_tensor),
         TF_TensorByteSize(dA_tensor));

  float tolerance = 1e-3;
  float eta_val = 1.5f;
  for (int j = 0; j < 4; j++) {
    ASSERT_NEAR(result_data[j], eta_val * A_vals[j], tolerance);
  }

  outputs[0]->Unref();
  TF_DeleteTensor(dA_tensor);
}

TEST_P(CppGradients, TestMNIST_Training) {
  bool use_function = !std::get<2>(GetParam());
  if (use_function) {
    // TODO(b/168850692): Enable this.
    GTEST_SKIP() << "Can't take gradient of "
                    "SparseSoftmaxCrossEntropyWithLogits in tracing mode.";
  }
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

  // X = data
  float X_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t X_dims[] = {2, 2};
  int num_dims = 2;
  AbstractTensorHandlePtr X =
      GetTensorHandleUtilFloat(ctx.get(), X_vals, X_dims, num_dims);

  // TODO(amturati): use random initializer for weights instead of
  // constant values.

  // W1 = first weights
  float W1_vals[] = {-.01f, 0.4f, 0.5f, -.2f};
  int64_t dims[] = {2, 2};
  AbstractTensorHandlePtr W1 =
      GetTensorHandleUtilFloat(ctx.get(), W1_vals, dims, num_dims);

  // W2 = second weights
  float W2_vals[] = {.1f, .2f, .3f, -.5f};
  AbstractTensorHandlePtr W2 =
      GetTensorHandleUtilFloat(ctx.get(), W2_vals, dims, num_dims);

  // y = labels
  int y_vals[] = {1, 1};
  int64_t y_dims[] = {2};
  num_dims = sizeof(y_dims) / sizeof(y_dims[0]);
  AbstractTensorHandlePtr y =
      GetTensorHandleUtilInt(ctx.get(), y_vals, y_dims, num_dims);

  // Register Grads
  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // Prepare for training
  std::vector<AbstractTensorHandle*> weights;
  weights.push_back(W1.get());
  weights.push_back(W2.get());

  // Set learning rate to be 1e-1
  AbstractTensorHandle* learning_rate = nullptr;
  s = ScalarTensorHandle(ctx.get(), 1e-1, &learning_rate);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // Train
  int num_iters = 10;
  std::vector<AbstractTensorHandle*> mnist_outputs(3);
  std::vector<AbstractTensorHandle*> grads(2);
  for (int i = 0; i < num_iters; i++) {
    // Run Forward Pass
    s = RunModel(MNISTGradModel, ctx.get(),
                 {X.get(), weights[0], weights[1], y.get()},
                 absl::MakeSpan(mnist_outputs),
                 /*use_function=*/!std::get<2>(GetParam()), registry);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();

    // Fill grads
    grads[0] = mnist_outputs[0];
    grads[1] = mnist_outputs[1];

    // Gradient Update
    s = UpdateWeights(ctx.get(), grads, weights, learning_rate);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  }

  grads[0]->Unref();          // release W1_grad
  grads[1]->Unref();          // release W2_grad
  mnist_outputs[2]->Unref();  // release loss
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
