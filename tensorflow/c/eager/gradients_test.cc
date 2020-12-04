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
#include "tensorflow/c/eager/gradients.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/gradients/array_grad.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/not_differentiable.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {
using std::vector;
using tensorflow::TF_StatusPtr;
using tracing::TracingOperation;

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(errors::OK, s.code()) << s.error_message();
  }
};

Status RegisterGradients(GradientRegistry* registry) {
  // TODO(srbs): Rename ops::Add to ops::AddV2 and AddRegister to
  // AddV2Registerer.
  TF_RETURN_IF_ERROR(registry->Register("AddV2", AddRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Exp", ExpRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("IdentityN", IdentityNRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Sqrt", SqrtRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Neg", NegRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Sub", SubRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Mul", MulRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Log1p", Log1pRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("DivNoNan", DivNoNanRegisterer));
  TF_RETURN_IF_ERROR(RegisterNotDifferentiable(registry, "CheckNumerics"));
  return Status::OK();
}

// Computes
// y = inputs[0] + inputs[1]
// return grad(y, {inputs[0], inputs[1]})
Status AddGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(RegisterGradients(&registry));

  auto tape = std::make_unique<Tape>(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch x.
  tape->Watch(inputs[1]);  // Watch y.
  std::vector<AbstractTensorHandle*> add_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape.get(), registry));
  TF_RETURN_IF_ERROR(ops::Add(tape_ctx.get(), inputs,
                              absl::MakeSpan(add_outputs),
                              "Add"));  // Compute x+y.
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/add_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto add_output : add_outputs) {
    add_output->Unref();
  }
  return Status::OK();
}

// Computes
// y = exp(inputs[0])
// return grad(y, {inputs[0]})
Status ExpGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(RegisterGradients(&registry));

  auto tape = std::make_unique<Tape>(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch x.
  std::vector<AbstractTensorHandle*> exp_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape.get(), registry));
  TF_RETURN_IF_ERROR(
      ops::Exp(tape_ctx.get(), inputs, absl::MakeSpan(exp_outputs), "Exp"));
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/exp_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto exp_output : exp_outputs) {
    exp_output->Unref();
  }
  return Status::OK();
}

// Computes
// y = sqrt(inputs[0])
// return grad(y, {inputs[0]})
Status SqrtGradModel(AbstractContext* ctx,
                     absl::Span<AbstractTensorHandle* const> inputs,
                     absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(RegisterGradients(&registry));

  auto tape = std::make_unique<Tape>(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch x.
  std::vector<AbstractTensorHandle*> sqrt_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape.get(), registry));
  TF_RETURN_IF_ERROR(
      ops::Sqrt(tape_ctx.get(), inputs, absl::MakeSpan(sqrt_outputs), "Sqrt"));
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/sqrt_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto sqrt_output : sqrt_outputs) {
    sqrt_output->Unref();
  }
  return Status::OK();
}

// Computes
// ignored, y = IdentityN(inputs[0], inputs[1])
// return grad(y, {inputs[0], inputs[1]})
// This should return [nullptr, 1].
Status IdentityNGradModel(AbstractContext* ctx,
                          absl::Span<AbstractTensorHandle* const> inputs,
                          absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(RegisterGradients(&registry));

  auto tape = std::make_unique<Tape>(/*persistent=*/false);
  tape->Watch(inputs[0]);
  tape->Watch(inputs[1]);

  vector<AbstractTensorHandle*> identity_n_outputs(2);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape.get(), registry));
  TF_RETURN_IF_ERROR(ops::IdentityN(
      tape_ctx.get(), inputs, absl::MakeSpan(identity_n_outputs), "IdentityN"));
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx,
                                           /*targets=*/{identity_n_outputs[1]},
                                           /*sources=*/{inputs[0], inputs[1]},
                                           /*output_gradients=*/{}, outputs));
  for (auto identity_n_output : identity_n_outputs) {
    identity_n_output->Unref();
  }
  return Status::OK();
}

// Computes
// y = - inputs[0]
// return grad(y, {inputs[0]})
Status NegGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(RegisterGradients(&registry));

  auto tape = std::make_unique<Tape>(/*persistent=*/false);
  tape->Watch(inputs[0]);

  std::vector<AbstractTensorHandle*> neg_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape.get(), registry));
  TF_RETURN_IF_ERROR(
      ops::Neg(tape_ctx.get(), inputs, absl::MakeSpan(neg_outputs), "Neg"));
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/neg_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto neg_output : neg_outputs) {
    neg_output->Unref();
  }
  return Status::OK();
}

// Computes
// y = inputs[0] - inputs[1]
// return grad(y, {inputs[0], inputs[1]})
Status SubGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(RegisterGradients(&registry));

  auto tape = std::make_unique<Tape>(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch x.
  tape->Watch(inputs[1]);  // Watch y.
  std::vector<AbstractTensorHandle*> sub_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape.get(), registry));
  TF_RETURN_IF_ERROR(ops::Sub(tape_ctx.get(), inputs,
                              absl::MakeSpan(sub_outputs),
                              "Sub"));  // Compute x-y.
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/sub_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto sub_output : sub_outputs) {
    sub_output->Unref();
  }
  return Status::OK();
}

// Computes
// y = inputs[0] * inputs[1]
// return grad(y, {inputs[0], inputs[1]})
Status MulGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(RegisterGradients(&registry));

  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch x.
  tape->Watch(inputs[1]);  // Watch y.
  std::vector<AbstractTensorHandle*> mul_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::Mul(tape_ctx.get(), inputs,
                              absl::MakeSpan(mul_outputs),
                              "Mul"));  // Compute x*y.
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/mul_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto mul_output : mul_outputs) {
    mul_output->Unref();
  }
  delete tape;
  return Status::OK();
}

// Computes
// y = log(1 + inputs[0])
// return grad(y, {inputs[0]})
Status Log1pGradModel(AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(RegisterGradients(&registry));
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch x.
  std::vector<AbstractTensorHandle*> log1p_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::Log1p(tape_ctx.get(), inputs,
                                absl::MakeSpan(log1p_outputs),
                                "Log1p"));  // Compute log(1 + x).
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/log1p_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto log1p_output : log1p_outputs) {
    log1p_output->Unref();
  }
  delete tape;
  return Status::OK();
}

// Computes
// y = inputs[0] / inputs[1]
// return grad(y, {inputs[0], inputs[1]})
Status DivNoNanGradModel(AbstractContext* ctx,
                         absl::Span<AbstractTensorHandle* const> inputs,
                         absl::Span<AbstractTensorHandle*> outputs) {
  GradientRegistry registry;
  TF_RETURN_IF_ERROR(RegisterGradients(&registry));
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(inputs[0]);  // Watch x.
  tape->Watch(inputs[1]);  // Watch y.
  std::vector<AbstractTensorHandle*> div_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::DivNoNan(tape_ctx.get(), inputs,
                                   absl::MakeSpan(div_outputs),
                                   "DivNoNan"));  // Compute x / y.
  TF_RETURN_IF_ERROR(tape->ComputeGradient(ctx, /*targets=*/div_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{}, outputs));
  for (auto div_output : div_outputs) {
    div_output->Unref();
  }
  delete tape;
  return Status::OK();
}

Status getValue(AbstractTensorHandle* t, TF_Tensor** result_tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_TensorHandle* result_t =
      TF_AbstractTensorGetEagerTensor(wrap(t), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  *result_tensor = TFE_TensorHandleResolve(result_t, status.get());
  return Status::OK();
}

TEST_P(CppGradients, TestAddGrad) {
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
    Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // Pseudo-code:
  //
  // tape.watch(x)
  // tape.watch(y)
  // y = x + y
  // outputs = tape.gradient(y, [x, y])
  std::vector<AbstractTensorHandle*> outputs(2);
  s = RunModel(AddGradModel, ctx.get(), {x.get(), y.get()},
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* result_tensor;
  s = getValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, 1.0);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;

  s = getValue(outputs[1], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, 1.0);
  outputs[1]->Unref();
  TF_DeleteTensor(result_tensor);
}

TEST_P(CppGradients, TestExpGrad) {
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
    Status s = TestScalarTensorHandle(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  // Pseudo-code:
  //
  // tape.watch(x)
  // y = exp(x)
  // outputs = tape.gradient(y, x)
  std::vector<AbstractTensorHandle*> outputs(1);
  Status s =
      RunModel(ExpGradModel, ctx.get(), {x.get()}, absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* result_tensor;
  s = getValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_NEAR(*result_value, 2.718, 0.001);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;
}

TEST_P(CppGradients, TestSqrtGrad) {
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
    Status s = TestScalarTensorHandle(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  // Pseudo-code:
  //
  // tape.watch(x)
  // y = sqrt(x)
  // outputs = tape.gradient(y, x)
  std::vector<AbstractTensorHandle*> outputs(1);
  Status s =
      RunModel(SqrtGradModel, ctx.get(), {x.get()}, absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* result_tensor;
  s = getValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_NEAR(*result_value, 0.5, 0.001);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;
}

TEST_P(CppGradients, TestIdentityNGrad) {
  // Pseudo-code:
  //
  // tape.watch(x1)
  // tape.watch(x2)
  // unused, y = IdentityN([x1, x2])
  // outputs = tape.gradient(y, [x1, x2])
  // Expected: [nullptr, 1]
  //
  // This test is interesting because the current implementation of GradientTape
  // would return [0, 1] whereas we use build_default_zeros_grads=false here
  // so we get back [nullptr, 1].
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

  AbstractTensorHandlePtr x1;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s = TestScalarTensorHandle(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x1.reset(x_raw);
  }
  AbstractTensorHandlePtr x2;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s = TestScalarTensorHandle(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x2.reset(x_raw);
  }

  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  std::vector<AbstractTensorHandle*> outputs(2);
  s = RunModel(IdentityNGradModel, ctx.get(), {x1.get(), x2.get()},
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  EXPECT_EQ(outputs[0], nullptr);
  TF_Tensor* result_tensor;
  s = getValue(outputs[1], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, 1.0);
  outputs[1]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;
}

TEST_P(CppGradients, TestNegGrad) {
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
    Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // Pseudo-code:
  //
  // tape.watch(x)
  // y = - x
  // outputs = tape.gradient(y, x)
  std::vector<AbstractTensorHandle*> outputs(1);
  s = RunModel(NegGradModel, ctx.get(), {x.get()}, absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* result_tensor;
  s = getValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, -1.0);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;
}

TEST_P(CppGradients, TestSubGrad) {
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
    Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  // Pseudo-code:
  //
  // tape.watch(x)
  // tape.watch(y)
  // y = x - y
  // outputs = tape.gradient(y, [x, y])
  std::vector<AbstractTensorHandle*> outputs(2);
  Status s = RunModel(SubGradModel, ctx.get(), {x.get(), y.get()},
                      absl::MakeSpan(outputs),
                      /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* result_tensor;
  s = getValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, 1.0);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;

  s = getValue(outputs[1], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, -1.0);
  outputs[1]->Unref();
  TF_DeleteTensor(result_tensor);
}

TEST_P(CppGradients, TestMulGrad) {
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
    Status s = TestScalarTensorHandle(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  // Pseudo-code:
  //
  // tape.watch(x)
  // tape.watch(y)
  // y = x * y
  // outputs = tape.gradient(y, [x, y])
  std::vector<AbstractTensorHandle*> outputs(2);
  Status s = RunModel(MulGradModel, ctx.get(), {x.get(), y.get()},
                      absl::MakeSpan(outputs),
                      /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* result_tensor;
  s = getValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, 2.0);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;

  s = getValue(outputs[1], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, 1.0);
  outputs[1]->Unref();
  TF_DeleteTensor(result_tensor);
}

TEST_P(CppGradients, TestLog1pGrad) {
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
    Status s = TestScalarTensorHandle(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  // Pseudo-code:
  //
  // tape.watch(x)
  // y = log(1 + x)
  // outputs = tape.gradient(y, x)
  std::vector<AbstractTensorHandle*> outputs(1);
  Status s =
      RunModel(Log1pGradModel, ctx.get(), {x.get()}, absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* result_tensor;
  s = getValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_NEAR(*result_value, 0.5, 0.001);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;
}

TEST_P(CppGradients, TestDivNoNanGrad) {
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
    Status s = TestScalarTensorHandle(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  // Pseudo-code:
  //
  // tape.watch(x)
  // tape.watch(y)
  // y = x / y
  // outputs = tape.gradient(y, [x, y])
  std::vector<AbstractTensorHandle*> outputs(2);
  Status s = RunModel(DivNoNanGradModel, ctx.get(), {x.get(), y.get()},
                      absl::MakeSpan(outputs),
                      /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* result_tensor;
  s = getValue(outputs[0], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_NEAR(*result_value, 0.5, 0.001);
  outputs[0]->Unref();
  TF_DeleteTensor(result_tensor);
  result_tensor = nullptr;

  s = getValue(outputs[1], &result_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_NEAR(*result_value, -0.25, 0.001);
  outputs[1]->Unref();
  TF_DeleteTensor(result_tensor);
}

TEST_P(CppGradients, TestSetAttrString) {
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

  AbstractTensorHandlePtr t;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s = TestScalarTensorHandle(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    t.reset(x_raw);
  }

  AbstractOperationPtr check_numerics_op(ctx->CreateOperation());
  ForwardOperation forward_op;
  Status s = Reset(check_numerics_op.get(), "CheckNumerics",
                   /*raw_device_name=*/nullptr, &forward_op);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  if (isa<TracingOperation>(check_numerics_op.get())) {
    s = dyn_cast<TracingOperation>(check_numerics_op.get())
            ->SetOpName("check_numerics");
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  }
  s = AddInput(check_numerics_op.get(), t.get(), &forward_op);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  string message = "This is the way!";
  s = SetAttrString(check_numerics_op.get(), "message", message.data(),
                    message.length(), &forward_op);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  int num_retvals = 1;
  std::vector<AbstractTensorHandle*> outputs(1);
  GradientRegistry registry;
  s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  auto tape = std::make_unique<Tape>(/*persistent=*/false);
  s = Execute(check_numerics_op.get(), ctx.get(), absl::MakeSpan(outputs),
              &num_retvals, &forward_op, tape.get(), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  string read_message;
  s = forward_op.attrs.Get("message", &read_message);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  ASSERT_EQ(read_message, message);
}

Status RecordOperationWithNullGradientFunctionModel(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
    absl::Span<AbstractTensorHandle*> outputs) {
  Tape tape(/*persistent=*/false);
  tape.Watch(inputs[0]);
  std::vector<AbstractTensorHandle*> neg_outputs(1);
  TF_RETURN_IF_ERROR(ops::Neg(ctx, inputs, absl::MakeSpan(neg_outputs), "Neg"));
  tape.RecordOperation(inputs, neg_outputs, nullptr, "Neg");
  return tape.ComputeGradient(ctx, /*targets=*/neg_outputs,
                              /*sources=*/inputs,
                              /*output_gradients=*/{}, outputs);
}

TEST_P(CppGradients, TestRecordOperationWithNullGradientFunctionRaises) {
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
    Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  std::vector<AbstractTensorHandle*> outputs(1);
  Status s = RunModel(RecordOperationWithNullGradientFunctionModel, ctx.get(),
                      {x.get()}, absl::MakeSpan(outputs),
                      /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_EQ(
      "Provided null gradient_function for 'Neg'.\nIf the intent is to treat "
      "this op as non-differentiable consider using RegisterNotDifferentiable "
      "or NotDifferentiableGradientFunction.",
      s.error_message());
  ASSERT_EQ(nullptr, outputs[0]);
}

// TODO(b/164171226): Enable this test with tfrt after AddInputList is
// supported. It is needed for IdentityN.
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
