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
#include "tensorflow/c/experimental/gradients/array_grad.h"

#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

Status IdentityNModel(AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs) {
  std::vector<AbstractTensorHandle*> temp_outputs(2);
  TF_RETURN_IF_ERROR(
      ops::IdentityN(ctx, inputs, absl::MakeSpan(temp_outputs), "IdentityN"));
  // Although, `ops::IdentityN` returns 2 tensors, the first tensor isn't needed
  // for computing gradient so we could safely drop it.
  outputs[0] = temp_outputs[1];
  temp_outputs[0]->Unref();
  return Status::OK();
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

TEST_P(CppGradients, TestIdentityNGrad) {
  // This test is interesting because the current implementation of GradientTape
  // would return [0, 1] whereas we use build_default_zeros_grads=false here
  // so we get back [nullptr, 1].

  AbstractTensorHandlePtr x1;
  {
    AbstractTensorHandle* x1_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 1.0f, &x1_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x1.reset(x1_raw);
  }

  AbstractTensorHandlePtr x2;
  {
    AbstractTensorHandle* x2_raw = nullptr;
    status_ = TestScalarTensorHandle<float, TF_FLOAT>(
        immediate_execution_ctx_.get(), 1.0f, &x2_raw);
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
    x2.reset(x2_raw);
  }

  status_ = registry_.Register("IdentityN", IdentityNRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  auto IdentityNGradModel = BuildGradModel(IdentityNModel, registry_);

  std::vector<AbstractTensorHandle*> outputs(2);
  status_ =
      RunModel(IdentityNGradModel, immediate_execution_ctx_.get(),
               {x1.get(), x2.get()}, absl::MakeSpan(outputs), UseFunction());
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  EXPECT_EQ(outputs[0], nullptr);
  ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[1], {1.0f}, /*dims*/ {},
                                           /*abs_error*/ 0));
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
