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

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/tape.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/gradients/not_differentiable.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
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
    absl::Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(errors::OK, s.code()) << s.message();
  }
};

absl::Status RegisterGradients(GradientRegistry* registry) {
  TF_RETURN_IF_ERROR(RegisterNotDifferentiable(registry, "CheckNumerics"));
  return absl::OkStatus();
}

TEST_P(CppGradients, TestSetAttrString) {
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

  AbstractTensorHandlePtr t;
  {
    AbstractTensorHandle* x_raw = nullptr;
    absl::Status s =
        TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    t.reset(x_raw);
  }

  AbstractOperationPtr check_numerics_op(ctx->CreateOperation());
  ForwardOperation forward_op;
  absl::Status s = Reset(check_numerics_op.get(), "CheckNumerics",
                         /*raw_device_name=*/nullptr, &forward_op);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  if (isa<TracingOperation>(check_numerics_op.get())) {
    s = dyn_cast<TracingOperation>(check_numerics_op.get())
            ->SetOpName("check_numerics");
    ASSERT_EQ(errors::OK, s.code()) << s.message();
  }
  s = AddInput(check_numerics_op.get(), t.get(), &forward_op);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  std::string message = "This is the way!";
  s = SetAttrString(check_numerics_op.get(), "message", message.data(),
                    message.length(), &forward_op);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  int num_retvals = 1;
  std::vector<AbstractTensorHandle*> outputs(1);
  GradientRegistry registry;
  s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  auto tape = std::make_unique<Tape>(/*persistent=*/false);
  s = Execute(check_numerics_op.get(), ctx.get(), absl::MakeSpan(outputs),
              &num_retvals, &forward_op, tape.get(), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.message();

  std::string read_message;
  s = forward_op.attrs.Get("message", &read_message);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  ASSERT_EQ(read_message, message);
}

absl::Status RecordOperationWithNullGradientFunctionModel(
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
    absl::Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    ctx.reset(ctx_raw);
  }

  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    absl::Status s =
        TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    x.reset(x_raw);
  }

  std::vector<AbstractTensorHandle*> outputs(1);
  absl::Status s = RunModel(RecordOperationWithNullGradientFunctionModel,
                            ctx.get(), {x.get()}, absl::MakeSpan(outputs),
                            /*use_function=*/!std::get<2>(GetParam()));
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_EQ(
      "Provided null gradient_function for 'Neg'.\nIf the intent is to treat "
      "this op as non-differentiable consider using RegisterNotDifferentiable "
      "or NotDifferentiableGradientFunction.",
      s.message());
  ASSERT_EQ(nullptr, outputs[0]);
}

struct DummyTensor {
  int64_t id;
  int64_t GetID() const { return id; }
  tensorflow::DataType GetDType() const { return tensorflow::DT_FLOAT; }
  int* ZerosLike() const { return nullptr; }
};

struct DummyBackwardFunction {};

class MockVSpace
    : public eager::VSpace<int, DummyBackwardFunction, DummyTensor> {
 public:
  mutable int delete_gradient_called_ = 0;

  int64_t NumElements(int* tensor) const override { return 1; }
  int* AggregateGradients(
      gtl::ArraySlice<int*> gradient_tensors) const override {
    return gradient_tensors[0];
  }
  absl::Status CallBackwardFunction(
      const std::string& op_type, DummyBackwardFunction* backward_function,
      const std::vector<int64_t>& unneeded_gradients,
      gtl::ArraySlice<int*> output_gradients,
      absl::Span<int*> result) const override {
    for (int* g : output_gradients) {
      if (g) DeleteGradient(g);
    }
    return absl::InternalError("Intentional failure");
  }
  absl::Status BuildOnesLike(const DummyTensor& t,
                             int** result) const override {
    *result = new int(1);
    return absl::OkStatus();
  }
  int64_t TensorId(int* tensor) const override { return 0; }
  DummyTensor TapeTensorFromGradient(int* gradient) const override {
    return DummyTensor{0};
  }
  void MarkAsResult(int* gradient) const override {}
  void DeleteGradient(int* gradient) const override {
    delete_gradient_called_++;
    delete gradient;
  }
};

TEST(GradientTapeTest, MemoryLeakOnFailure) {
  eager::GradientTape<int, DummyBackwardFunction, DummyTensor> tape(
      /*persistent=*/false);
  tape.Watch(1);

  DummyBackwardFunction* bw = new DummyBackwardFunction();
  tape.RecordOperation(
      "TestOp", {DummyTensor{2}}, {1}, {tensorflow::DT_FLOAT},
      [bw]() { return bw; }, [](DummyBackwardFunction* bw) { delete bw; });

  MockVSpace vspace;
  std::vector<int*> results(1);
  absl::Status s =
      tape.ComputeGradient(vspace, {2}, {1}, {}, {}, absl::MakeSpan(results),
                           /*build_default_zeros_grads=*/false);

  ASSERT_EQ(error::INTERNAL, s.code());
  EXPECT_EQ(1, vspace.delete_gradient_called_)
      << "Expected gradient to be deleted (memory leak if 0)";
}

TEST_P(CppGradients, TestExecuteWithLargerOutputsVectorDoesNotCrash) {
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

  AbstractTensorHandlePtr t;
  {
    AbstractTensorHandle* x_raw = nullptr;
    absl::Status s =
        TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 1.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    t.reset(x_raw);
  }

  AbstractOperationPtr check_numerics_op(ctx->CreateOperation());
  ForwardOperation forward_op;
  absl::Status s = Reset(check_numerics_op.get(), "CheckNumerics",
                         /*raw_device_name=*/nullptr, &forward_op);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  if (isa<TracingOperation>(check_numerics_op.get())) {
    s = dyn_cast<TracingOperation>(check_numerics_op.get())
            ->SetOpName("check_numerics");
    ASSERT_EQ(errors::OK, s.code()) << s.message();
  }
  s = AddInput(check_numerics_op.get(), t.get(), &forward_op);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  std::string message = "This is the way!";
  s = SetAttrString(check_numerics_op.get(), "message", message.data(),
                    message.length(), &forward_op);
  ASSERT_EQ(errors::OK, s.code()) << s.message();

  int num_retvals = 1;
  // Allocate outputs with size 2, but we only expect 1 output.
  // The second element will be initialized to nullptr.
  std::vector<AbstractTensorHandle*> outputs(2, nullptr);

  GradientRegistry registry;
  s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  auto tape = std::make_unique<Tape>(/*persistent=*/false);

  // This call should NOT crash.
  s = Execute(check_numerics_op.get(), ctx.get(), absl::MakeSpan(outputs),
              &num_retvals, &forward_op, tape.get(), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  EXPECT_EQ(num_retvals, 1);
  EXPECT_NE(outputs[0], nullptr);
  EXPECT_EQ(outputs[1], nullptr);  // Should remain nullptr
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
