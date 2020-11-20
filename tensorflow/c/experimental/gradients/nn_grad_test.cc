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

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/gradients_util.h"
#include "tensorflow/c/experimental/gradients/nn_grad_testutil.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
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
  TF_RETURN_IF_ERROR(registry->Register("BiasAdd", BiasAddRegisterer));
  return Status::OK();
}

TEST_P(CppGradients, TestBiasAddGrad) {
  if (std::get<0>(GetParam()) == "mlir" && std::get<2>(GetParam()) == 0) {
    GTEST_SKIP() << "SetAttrString has not been implemented yet.\n";
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

  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t A_dims[] = {2, 2};
  float Bias_vals[] = {2.0f, 3.0f};
  int64_t Bias_dims[] = {2};

  AbstractTensorHandlePtr A =
      GetTensorHandleUtilFloat(ctx.get(), A_vals, A_dims, 2);
  AbstractTensorHandlePtr B =
      GetTensorHandleUtilFloat(ctx.get(), Bias_vals, Bias_dims, 1);

  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  /* Pseudo-code:
   *
   * tape.watch(A)
   * tape.watch(B)
   * Y = BiasAdd(A, Bias)
   * outputs = tape.gradient(Y, [A, Bias])
   */
  std::vector<AbstractTensorHandle*> outputs(2);
  s = RunModel(BiasAddGradModel, ctx.get(), {A.get(), B.get()},
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* dA_tensor;
  s = GetValue(outputs[0], &dA_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float result_data_dA[4] = {0};
  memcpy(&result_data_dA[0], TF_TensorData(dA_tensor),
         TF_TensorByteSize(dA_tensor));

  float expected_dA[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float tolerance = 1e-3;
  for (int j = 0; j < 4; j++) {
    ASSERT_NEAR(result_data_dA[j], expected_dA[j], tolerance);
  }

  TF_Tensor* dBias_tensor;
  s = GetValue(outputs[1], &dBias_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float result_data_dBias[2] = {0};
  memcpy(&result_data_dBias[0], TF_TensorData(dBias_tensor),
         TF_TensorByteSize(dBias_tensor));

  float expected_dBias[2] = {2.0f, 2.0f};
  for (int j = 0; j < 2; j++) {
    ASSERT_NEAR(result_data_dBias[j], expected_dBias[j], tolerance);
  }

  outputs[0]->Unref();
  outputs[1]->Unref();
  TF_DeleteTensor(dA_tensor);
  TF_DeleteTensor(dBias_tensor);
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
