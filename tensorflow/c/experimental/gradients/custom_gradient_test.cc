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

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {
using std::vector;

class CustomGradientTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    absl::Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(errors::OK, s.code()) << s.message();
  }
};

class PassThroughGradientFunction : public GradientFunction {
 public:
  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    CHECK_EQ(grad_outputs.size(), 1);
    CHECK_EQ(grad_inputs.size(), 1);
    grad_inputs[0] = grad_outputs[0];
    if (grad_inputs[0]) {
      grad_inputs[0]->Ref();
    }
    return absl::OkStatus();
  }
};

// Computes:
//
// @tf.custom_gradient
// def f(input):
//   def grad(grads):
//     return grads[0]
//   return tf.exp(input), grad
// outputs = [f(inputs[0])]
absl::Status ExpWithPassThroughGrad(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
    absl::Span<AbstractTensorHandle*> outputs) {
  Tape tape(/*persistent=*/false);
  tape.Watch(inputs[0]);  // Watch x.
  AbstractTensorHandle* exp_output;
  TF_RETURN_IF_ERROR(ops::Exp(ctx, inputs[0], &exp_output, "Exp"));
  std::unique_ptr<GradientFunction> gradient_function(
      new PassThroughGradientFunction);
  tape.RecordOperation(inputs, {exp_output}, gradient_function.release());
  TF_RETURN_IF_ERROR(tape.ComputeGradient(ctx,
                                          /*targets*/ {exp_output},
                                          /*sources=*/inputs,
                                          /*output_gradients=*/{},
                                          /*result=*/outputs));
  exp_output->Unref();
  return absl::OkStatus();
}

TEST_P(CustomGradientTest, ExpWithPassThroughGrad) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    absl::Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    ctx.reset(ctx_raw);
  }

  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    absl::Status s =
        TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    x.reset(x_raw);
  }

  // Pseudo-code:
  //
  // tape.watch(x)
  // y = exp(x)
  // outputs = tape.gradient(y, x)
  std::vector<AbstractTensorHandle*> outputs(1);
  absl::Status s = RunModel(ExpWithPassThroughGrad, ctx.get(), {x.get()},
                            absl::MakeSpan(outputs),
                            /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.message();

  TF_Tensor* result_tensor;
  s = GetValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, 1.0);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;
}

INSTANTIATE_TEST_SUITE_P(
    CustomGradientTest, CustomGradientTest,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));

}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
