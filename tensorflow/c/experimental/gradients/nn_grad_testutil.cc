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
#include "tensorflow/c/experimental/gradients/nn_grad_testutil.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace gradients {
namespace internal {

// Computes
// y = BiasAdd(inputs[0], inputs[1])
// return grad(y, {inputs[0], inputs[1]})
Status BiasAddGradModel(AbstractContext* ctx,
                        absl::Span<AbstractTensorHandle* const> inputs,
                        absl::Span<AbstractTensorHandle*> outputs,
                        const GradientRegistry& registry) {
  TapeVSpace vspace(ctx);
  auto tape = new Tape(/*persistent=*/false);
  tape->Watch(ToId(inputs[0]));  // Watch x.
  tape->Watch(ToId(inputs[1]));  // Watch y.
  std::vector<AbstractTensorHandle*> bias_add_outputs(1);
  AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));
  TF_RETURN_IF_ERROR(ops::BiasAdd(tape_ctx.get(), inputs,
                                  absl::MakeSpan(bias_add_outputs), "BiasAdd"));
  std::unordered_map<tensorflow::int64, TapeTensor>
      source_tensors_that_are_targets;

  std::vector<AbstractTensorHandle*> out_grads;
  TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(bias_add_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
  for (auto bias_add_output : bias_add_outputs) {
    bias_add_output->Unref();
  }
  outputs[0] = out_grads[0];
  outputs[1] = out_grads[1];
  delete tape;
  return Status::OK();
}

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
