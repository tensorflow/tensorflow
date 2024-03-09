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
#include "tensorflow/c/experimental/gradients/nn_grad.h"

#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

Status ReluModel(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> inputs,
                 absl::Span<AbstractTensorHandle*> outputs) {
  return ops::Relu(ctx, inputs[0], &outputs[0], "Relu");
}

Status SparseSoftmaxCrossEntropyWithLogitsModel(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
    absl::Span<AbstractTensorHandle*> outputs) {
  AbstractTensorHandle* loss;
  AbstractTensorHandle* backprop;
  TF_RETURN_IF_ERROR(ops::SparseSoftmaxCrossEntropyWithLogits(
      ctx, inputs[0], inputs[1], &loss, &backprop,
      "SparseSoftmaxCrossEntropyWithLogits"));
  // `gradient_checker` only works with model that returns only 1 tensor.
  // Although, `ops::SparseSoftmaxCrossEntropyWithLogits` returns 2 tensors, the
  // second tensor isn't needed for computing gradient so we could safely drop
  // it.
  outputs[0] = loss;
  backprop->Unref();
  return absl::OkStatus();
}

Status BiasAddModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs) {
  return ops::BiasAdd(ctx, inputs[0], inputs[1], &outputs[0], "NHWC",
                      "BiasAdd");
}

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    status_ = StatusFromTF_Status(status.get());
    ASSERT_EQ(errors::OK, status_.code()) << status_.message();

    {
      AbstractContext* ctx_raw = nullptr;
      status_ =
          BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
      ASSERT_EQ(errors::OK, status_.code()) << status_.message();
      immediate_execution_ctx_.reset(ctx_raw);
    }

    // Computing numerical gradients with TensorFloat-32 is numerically
    // unstable. Some forward pass tests also fail with TensorFloat-32 due to
    // low tolerances
    enable_tensor_float_32_execution(false);
  }

  AbstractContextPtr immediate_execution_ctx_;
  GradientRegistry registry_;
  Status status_;

 public:
  bool UseMlir() const { return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const { return std::get<2>(GetParam()); }
};

TEST_P(CppGradients, TestReluGrad) {
  status_ = registry_.Register("Relu", ReluRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.message();

  auto ReluGradModel = BuildGradModel(ReluModel, registry_);

  float X_vals[] = {1.0f, 2.0f, 3.0f, -5.0f, -4.0f, -3.0f, 2.0f, 10.0f, -1.0f};
  int64_t X_dims[] = {3, 3};
  AbstractTensorHandlePtr X;
  {
    AbstractTensorHandle* X_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), X_vals, X_dims, 2, &X_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.message();
    X.reset(X_raw);
  }

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      ReluModel, ReluGradModel, immediate_execution_ctx_.get(), {X.get()},
      UseFunction()));

  // Mathematically, Relu isn't differentiable at `0`. So `gradient_checker`
  // does not work with it.
  AbstractTensorHandlePtr Y;
  {
    AbstractTensorHandle* Y_raw;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 0.0f, &Y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.message();
    Y.reset(Y_raw);
  }

  std::vector<AbstractTensorHandle*> outputs(1);
  status_ = RunModel(ReluGradModel, immediate_execution_ctx_.get(), {Y.get()},
                     absl::MakeSpan(outputs), UseFunction());
  ASSERT_EQ(errors::OK, status_.code()) << status_.message();
  ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[0], {0.0f}, /*dims*/ {},
                                           /*abs_error*/ 0));
  outputs[0]->Unref();
}

TEST_P(CppGradients, TestSparseSoftmaxCrossEntropyWithLogitsGrad) {
  if (UseFunction()) {
    // TODO(b/168850692): Enable this.
    GTEST_SKIP() << "Can't take gradient of "
                    "SparseSoftmaxCrossEntropyWithLogits in tracing mode.";
  }

  // Score
  float X_vals[] = {1.0f, 2.0f, 3.0f, -5.0f, -4.0f, -3.0f, 2.0f, 0.0f, -1.0f};
  int64_t X_dims[] = {3, 3};
  AbstractTensorHandlePtr X;
  {
    AbstractTensorHandle* X_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), X_vals, X_dims, 2, &X_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.message();
    X.reset(X_raw);
  }
  // Label
  int32_t Y_vals[] = {1, 0, 1};
  int64_t Y_dims[] = {3};
  AbstractTensorHandlePtr Y;
  {
    AbstractTensorHandle* Y_raw;
    status_ = TestTensorHandleWithDims<int32_t, TF_INT32>(
        immediate_execution_ctx_.get(), Y_vals, Y_dims, 1, &Y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.message();
    Y.reset(Y_raw);
  }

  status_ = registry_.Register("SparseSoftmaxCrossEntropyWithLogits",
                               SparseSoftmaxCrossEntropyWithLogitsRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      SparseSoftmaxCrossEntropyWithLogitsModel,
      BuildGradModel(SparseSoftmaxCrossEntropyWithLogitsModel, registry_),
      immediate_execution_ctx_.get(), {X.get(), Y.get()}, UseFunction()));
}

TEST_P(CppGradients, TestBiasAddGrad) {
  if (UseFunction() && UseMlir()) {
    GTEST_SKIP() << "SetAttrString has not been implemented yet.\n";
  }

  // A
  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t A_dims[] = {2, 2};
  AbstractTensorHandlePtr A;
  {
    AbstractTensorHandle* A_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), A_vals, A_dims, 2, &A_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.message();
    A.reset(A_raw);
  }
  // Bias
  float Bias_vals[] = {2.0f, 3.0f};
  int64_t Bias_dims[] = {2};
  AbstractTensorHandlePtr Bias;
  {
    AbstractTensorHandle* Bias_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), Bias_vals, Bias_dims, 1, &Bias_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.message();
    Bias.reset(Bias_raw);
  }

  status_ = registry_.Register("BiasAdd", BiasAddRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      BiasAddModel, BuildGradModel(BiasAddModel, registry_),
      immediate_execution_ctx_.get(), {A.get(), Bias.get()}, UseFunction()));
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
