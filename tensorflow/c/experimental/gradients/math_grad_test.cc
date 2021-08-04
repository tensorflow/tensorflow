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
#include "tensorflow/c/experimental/gradients/math_grad.h"

#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

Status AddModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
  return ops::AddV2(ctx, inputs[0], inputs[1], &outputs[0], "Add");
}

Status ExpModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
  return ops::Exp(ctx, inputs[0], &outputs[0], "Exp");
}

Status SqrtModel(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> inputs,
                 absl::Span<AbstractTensorHandle*> outputs) {
  return ops::Sqrt(ctx, inputs[0], &outputs[0], "Sqrt");
}

Status NegModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
  return ops::Neg(ctx, inputs[0], &outputs[0], "Neg");
}

Status SubModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
  return ops::Sub(ctx, inputs[0], inputs[1], &outputs[0], "Sub");
}

Status MulModel(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs) {
  return ops::Mul(ctx, inputs[0], inputs[1], &outputs[0], "Mul");
}

Status Log1pModel(AbstractContext* ctx,
                  absl::Span<AbstractTensorHandle* const> inputs,
                  absl::Span<AbstractTensorHandle*> outputs) {
  return ops::Log1p(ctx, inputs[0], &outputs[0], "Log1p");
}

Status DivNoNanModel(AbstractContext* ctx,
                     absl::Span<AbstractTensorHandle* const> inputs,
                     absl::Span<AbstractTensorHandle*> outputs) {
  return ops::DivNoNan(ctx, inputs[0], inputs[1], &outputs[0], "DivNoNan");
}

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    status_ = StatusFromTF_Status(status.get());
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

    {
      AbstractContext* ctx_raw = nullptr;
      status_ =
          BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
      ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
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

TEST_P(CppGradients, TestAddGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    y.reset(y_raw);
  }

  // TODO(srbs): Rename ops::Add to ops::AddV2 and AddRegister to
  // AddV2Registerer.
  status_ = registry_.Register("AddV2", AddRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      AddModel, BuildGradModel(AddModel, registry_),
      immediate_execution_ctx_.get(), {x.get(), y.get()}, UseFunction()));
}

TEST_P(CppGradients, TestExpGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  status_ = registry_.Register("Exp", ExpRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      ExpModel, BuildGradModel(ExpModel, registry_),
      immediate_execution_ctx_.get(), {x.get()}, UseFunction()));
}

TEST_P(CppGradients, TestMatMulGrad) {
  // TODO(vnvo2409): Figure out why `gradient_checker` does not work very
  // well with `MatMul` and remove `TestMatMul*` in
  // `mnist_gradients_test` when done.
  GTEST_SKIP();

  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  int64_t A_dims[] = {3, 3};
  AbstractTensorHandlePtr A;
  {
    AbstractTensorHandle* A_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), A_vals, A_dims, 2, &A_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    A.reset(A_raw);
  }

  float B_vals[] = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  int64_t B_dims[] = {3, 3};
  AbstractTensorHandlePtr B;
  {
    AbstractTensorHandle* B_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), B_vals, B_dims, 2, &B_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    B.reset(B_raw);
  }

  status_ = registry_.Register("MatMul", MatMulRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  for (bool transpose_a : {false, true}) {
    for (bool transpose_b : {false, true}) {
      Model MatMulModel =
          [transpose_a, transpose_b](
              AbstractContext* ctx,
              absl::Span<AbstractTensorHandle* const> inputs,
              absl::Span<AbstractTensorHandle*> outputs) -> Status {
        return ops::MatMul(ctx, inputs[0], inputs[1], &outputs[0], transpose_a,
                           transpose_b, "MatMul");
      };
      ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
          MatMulModel, BuildGradModel(MatMulModel, registry_),
          immediate_execution_ctx_.get(), {A.get(), B.get()}, UseFunction()));
    }
  }
}

TEST_P(CppGradients, TestMatMulGradManual) {
  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  int64_t A_dims[] = {3, 3};
  AbstractTensorHandlePtr A;
  {
    AbstractTensorHandle* A_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), A_vals, A_dims, 2, &A_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    A.reset(A_raw);
  }

  float B_vals[] = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  int64_t B_dims[] = {3, 3};
  AbstractTensorHandlePtr B;
  {
    AbstractTensorHandle* B_raw;
    status_ = TestTensorHandleWithDims<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), B_vals, B_dims, 2, &B_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    B.reset(B_raw);
  }

  status_ = registry_.Register("MatMul", MatMulRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  bool transpose_a_vals[] = {false, false, true, true};
  bool transpose_b_vals[] = {false, true, false, true};
  float dA_vals[4][9] = {{24, 15, 6, 24, 15, 6, 24, 15, 6},
                         {18, 15, 12, 18, 15, 12, 18, 15, 12},
                         {24, 24, 24, 15, 15, 15, 6, 6, 6},
                         {18, 18, 18, 15, 15, 15, 12, 12, 12}};
  float dB_vals[4][9] = {{12, 12, 12, 15, 15, 15, 18, 18, 18},
                         {12, 15, 18, 12, 15, 18, 12, 15, 18},
                         {6, 6, 6, 15, 15, 15, 24, 24, 24},
                         {6, 15, 24, 6, 15, 24, 6, 15, 24}};

  for (int i{}; i < 4; ++i) {
    bool transpose_a = transpose_a_vals[i];
    bool transpose_b = transpose_b_vals[i];
    Model MatMulModel =
        [transpose_a, transpose_b](
            AbstractContext* ctx,
            absl::Span<AbstractTensorHandle* const> inputs,
            absl::Span<AbstractTensorHandle*> outputs) -> Status {
      return ops::MatMul(ctx, inputs[0], inputs[1], &outputs[0], transpose_a,
                         transpose_b, "MatMul");
    };
    Model MatMulGradModel = BuildGradModel(MatMulModel, registry_);
    std::vector<AbstractTensorHandle*> outputs(2);
    status_ =
        RunModel(MatMulGradModel, immediate_execution_ctx_.get(),
                 {A.get(), B.get()}, absl::MakeSpan(outputs), UseFunction());
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[0], dA_vals[i],
                                             /*dims*/ {3, 3},
                                             /*abs_error*/ 0));
    ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[1], dB_vals[i],
                                             /*dims*/ {3, 3},
                                             /*abs_error*/ 0));
    outputs[0]->Unref();
    outputs[1]->Unref();
  }
}

TEST_P(CppGradients, TestSqrtGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  status_ = registry_.Register("Sqrt", SqrtRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      SqrtModel, BuildGradModel(SqrtModel, registry_),
      immediate_execution_ctx_.get(), {x.get()}, UseFunction()));
}

TEST_P(CppGradients, TestNegGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  status_ = registry_.Register("Neg", NegRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      NegModel, BuildGradModel(NegModel, registry_),
      immediate_execution_ctx_.get(), {x.get()}, UseFunction()));
}

TEST_P(CppGradients, TestSubGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    y.reset(y_raw);
  }

  status_ = registry_.Register("Sub", SubRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      SubModel, BuildGradModel(SubModel, registry_),
      immediate_execution_ctx_.get(), {x.get(), y.get()}, UseFunction()));
}

TEST_P(CppGradients, TestMulGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    y.reset(y_raw);
  }

  status_ = registry_.Register("Mul", MulRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      MulModel, BuildGradModel(MulModel, registry_),
      immediate_execution_ctx_.get(), {x.get(), y.get()}, UseFunction()));
}

TEST_P(CppGradients, TestLog1pGrad) {
  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  status_ = registry_.Register("Log1p", Log1pRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      Log1pModel, BuildGradModel(Log1pModel, registry_),
      immediate_execution_ctx_.get(), {x.get()}, UseFunction()));
}

TEST_P(CppGradients, TestDivNoNanGrad) {
  status_ = registry_.Register("DivNoNan", DivNoNanRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  auto DivNoNanGradModel = BuildGradModel(DivNoNanModel, registry_);

  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 2.0f, &y_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    y.reset(y_raw);
  }

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      DivNoNanModel, DivNoNanGradModel, immediate_execution_ctx_.get(),
      {x.get(), y.get()}, UseFunction()));

  // `DivNoNanGradModel` should return {`0`, `0`} when the denominator is `0`.
  AbstractTensorHandlePtr z;
  {
    AbstractTensorHandle* z_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 0.0f, &z_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    z.reset(z_raw);
  }
  std::vector<AbstractTensorHandle*> outputs(2);
  status_ =
      RunModel(DivNoNanGradModel, immediate_execution_ctx_.get(),
               {x.get(), z.get()}, absl::MakeSpan(outputs), UseFunction());
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[0], {0.0f}, /*dims*/ {},
                                           /*abs_error*/ 0));
  ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[1], {0.0f}, /*dims*/ {},
                                           /*abs_error*/ 0));
  outputs[0]->Unref();
  outputs[1]->Unref();
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
