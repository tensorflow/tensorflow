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
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

Status SparseSoftmaxCrossEntropyWithLogitsModel(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
    absl::Span<AbstractTensorHandle*> outputs) {
  return ops::SparseSoftmaxCrossEntropyWithLogits(
      ctx, inputs, outputs, "SparseSoftmaxCrossEntropyWithLogits");
}

Status SparseSoftmaxCrossEntropyWithLogitsGradModel(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
    absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(
      registry.Register("SparseSoftmaxCrossEntropyWithLogits",
                        SparseSoftmaxCrossEntropyWithLogitsRegisterer));

  Tape tape(/*persistent=*/false);
  tape.Watch(inputs[0]);  // Watch score.
  tape.Watch(inputs[1]);  // Watch label.
  std::vector<AbstractTensorHandle*> temp_outputs(2);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, &tape, registry));
  TF_RETURN_IF_ERROR(ops::SparseSoftmaxCrossEntropyWithLogits(
      tape_ctx.get(), inputs, absl::MakeSpan(temp_outputs),
      "SparseSoftmaxCrossEntropyWithLogitsGrad"));

  TF_RETURN_IF_ERROR(tape.ComputeGradient(ctx, /*targets=*/temp_outputs,
                                          /*sources=*/inputs,
                                          /*output_gradients=*/{}, outputs));
  for (auto temp_output : temp_outputs) {
    temp_output->Unref();
  }
  return Status::OK();
}

Status BiasAddModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs) {
  return ops::BiasAdd(ctx, inputs, outputs, "BiasAdd");
}

Status BiasAddGradModel(AbstractContext* ctx,
                        absl::Span<AbstractTensorHandle* const> inputs,
                        absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(registry.Register("BiasAdd", BiasAddRegisterer));

  Tape tape(/*persistent=*/false);
  tape.Watch(inputs[0]);  // Watch A.
  tape.Watch(inputs[1]);  // Watch Bias.
  std::vector<AbstractTensorHandle*> temp_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, &tape, registry));
  TF_RETURN_IF_ERROR(ops::BiasAdd(tape_ctx.get(), inputs,
                                  absl::MakeSpan(temp_outputs), "BiasAddGrad"));

  TF_RETURN_IF_ERROR(tape.ComputeGradient(ctx, /*targets=*/temp_outputs,
                                          /*sources=*/inputs,
                                          /*output_gradients=*/{}, outputs));
  for (auto temp_output : temp_outputs) {
    temp_output->Unref();
  }
  return Status::OK();
}

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = StatusFromTF_Status(status.get());
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();

    {
      AbstractContext* ctx_raw = nullptr;
      Status s =
          BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
      ASSERT_EQ(errors::OK, s.code()) << s.error_message();
      ctx_.reset(ctx_raw);
    }
  }

  AbstractContextPtr ctx_;

 public:
  bool UseMlir() const { return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const { return std::get<2>(GetParam()); }
};

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
    Status s =
        TestTensorHandleWithDimsFloat(ctx_.get(), X_vals, X_dims, 2, &X_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    X.reset(X_raw);
  }
  // Label
  int Y_vals[] = {1, 0, 1};
  int64_t Y_dims[] = {3};
  AbstractTensorHandlePtr Y;
  {
    AbstractTensorHandle* Y_raw;
    Status s =
        TestTensorHandleWithDimsInt(ctx_.get(), Y_vals, Y_dims, 1, &Y_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    Y.reset(Y_raw);
  }

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      SparseSoftmaxCrossEntropyWithLogitsModel,
      SparseSoftmaxCrossEntropyWithLogitsGradModel, ctx_.get(),
      {X.get(), Y.get()},
      /*use_function=*/UseFunction()));
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
    Status s =
        TestTensorHandleWithDimsFloat(ctx_.get(), A_vals, A_dims, 2, &A_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    A.reset(A_raw);
  }
  // Bias
  float Bias_vals[] = {2.0f, 3.0f};
  int64_t Bias_dims[] = {2};
  AbstractTensorHandlePtr Bias;
  {
    AbstractTensorHandle* Bias_raw;
    Status s = TestTensorHandleWithDimsFloat(ctx_.get(), Bias_vals, Bias_dims,
                                             1, &Bias_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    Bias.reset(Bias_raw);
  }

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      BiasAddModel, BiasAddGradModel, ctx_.get(), {A.get(), Bias.get()},
      /*use_function=*/UseFunction()));
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
