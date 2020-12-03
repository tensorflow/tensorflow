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
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;
using tracing::TracingOperation;

Status BiasAddModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
  return ops::BiasAdd(ctx, inputs, outputs, "BiasAdd");
}

Status BiasAddGradModel(AbstractContext* ctx,
                        absl::Span<AbstractTensorHandle* const> inputs,
                        absl::Span<AbstractTensorHandle*> outputs,
                        const GradientRegistry& registry) {
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch A.
  tape->Watch(ToId(inputs[1]));  // Watch Bias.
  std::vector<AbstractTensorHandle*> temp_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::BiasAdd(tape_ctx.get(), inputs,
                                  absl::MakeSpan(temp_outputs), "BiasAddGrad"));
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(temp_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  for (auto temp_output : temp_outputs) {
    temp_output->Unref();
  }
  outputs[0] = out_grads[0];
  outputs[1] = out_grads[1];
  delete tape;
  return Status::OK();
}

Status RegisterGradients(GradientRegistry* registry) {
  TF_RETURN_IF_ERROR(registry->Register("BiasAdd", BiasAddRegisterer));
  return Status::OK();
}

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(errors::OK, s.code()) << s.error_message();

    {
      AbstractContext* ctx_raw = nullptr;
      Status s =
          BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
      ASSERT_EQ(errors::OK, s.code()) << s.error_message();
      ctx_.reset(ctx_raw);
    }

    s = RegisterGradients(&registry_);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  }

  GradientRegistry registry_;
  AbstractContextPtr ctx_;

 public:
  bool UseMlir() const { return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const { return std::get<2>(GetParam()); }
};

TEST_P(CppGradients, TestBiasAddGrad) {
  if (!UseFunction() && UseMlir()) {
    GTEST_SKIP() << "SetAttrString has not been implemented yet.\n";
  }

  // A
  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t A_dims[] = {2, 2};
  AbstractTensorHandlePtr A =
      GetTensorHandleUtilFloat(ctx_.get(), A_vals, A_dims, 2);
  // Bias
  float Bias_vals[] = {2.0f, 3.0f};
  int64_t Bias_dims[] = {2};
  AbstractTensorHandlePtr Bias =
      GetTensorHandleUtilFloat(ctx_.get(), Bias_vals, Bias_dims, 1);

  std::vector<AbstractTensorHandle*> inputs{A.get(), Bias.get()};

  ASSERT_NO_FATAL_FAILURE(CompareNumericalAndAutodiffGradients(
      BiasAddModel, BiasAddGradModel, ctx_.get(), {A.get(), Bias.get()},
      /*use_function=*/!std::get<2>(GetParam()), registry_));
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
