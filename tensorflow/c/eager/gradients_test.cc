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
  TF_RETURN_IF_ERROR(RegisterNotDifferentiable(registry, "CheckNumerics"));
  return OkStatus();
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
    Status s = TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 1.0f, &x_raw);
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
  AbstractTensorHandle* neg_output;
  TF_RETURN_IF_ERROR(ops::Neg(ctx, inputs[0], &neg_output, "Neg"));
  tape.RecordOperation(inputs, {neg_output}, nullptr, "Neg");
  return tape.ComputeGradient(ctx,
                              /*targets=*/{neg_output},
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
    Status s = TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 2.0f, &x_raw);
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
