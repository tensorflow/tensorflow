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
#ifndef TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_GRAD_TEST_HELPER_H_
#define TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_GRAD_TEST_HELPER_H_

#include "tensorflow/c/eager/gradients_util.h"

namespace tensorflow {
namespace gradients {
namespace internal {

// This macro will expand to a function that defines a `Model`. This `Model` is
// then used for testing by `nn_grad_test` and `math_grad_test`. `ops_call` is a
// statement that calls to a `ops::` and should be wrapped around by `{}`.
// `ops_call` has access to `inputs`. The output parameter of the ops should
// always be `absl::MakeSpan(temp_outputs)`. This macro supports most one-ops
// model.
// TODO(vnvo2409): Extend support for more complex model.
#define TF_MODEL_FACTORY(name, num_inputs, num_outputs, ops_call)      \
  Status name(AbstractContext* ctx,                                    \
              absl::Span<AbstractTensorHandle* const> inputs,          \
              absl::Span<AbstractTensorHandle*> outputs,               \
              const GradientRegistry& registry) {                      \
    auto tape = new Tape(/*persistent=*/false);                        \
    for (int i{}; i < num_inputs; ++i) {                               \
      tape->Watch(ToId(inputs[i]));                                    \
    }                                                                  \
                                                                       \
    AbstractTensorHandle* temp_outputs[num_outputs] = {};              \
    AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry)); \
    ops_call;                                                          \
                                                                       \
    for (int i{}; i < num_outputs; ++i) {                              \
      outputs[i] = temp_outputs[i];                                    \
    }                                                                  \
    delete tape;                                                       \
    return Status::OK();                                               \
  }

// This macro will expand to a function that defines a `GradModel`. This
// `GradModel` is then used for testing by `nn_grad_test` and `math_grad_test`.
// `ops_call` is a statement that calls to a `ops::` and should be wrapped
// around by `{}`. `ops_call` has access to `inputs`. The output parameter of
// the ops should always be `absl::MakeSpan(temp_outputs)`. This macro supports
// most one-ops model.
// TODO(vnvo2409): Extend support for more complex model.
#define TF_GRAD_MODEL_FACTORY(name, num_inputs, num_outputs, num_grad_outputs, \
                              ops_call)                                        \
  Status name(AbstractContext* ctx,                                            \
              absl::Span<AbstractTensorHandle* const> inputs,                  \
              absl::Span<AbstractTensorHandle*> outputs,                       \
              const GradientRegistry& registry) {                              \
    TapeVSpace vspace(ctx);                                                    \
    auto tape = new Tape(/*persistent=*/false);                                \
    for (int i{}; i < num_inputs; ++i) {                                       \
      tape->Watch(ToId(inputs[i]));                                            \
    }                                                                          \
                                                                               \
    AbstractTensorHandle* temp_outputs[num_outputs] = {};                      \
    AbstractContextPtr tape_ctx(new TapeContext(ctx, tape, registry));         \
    ops_call;                                                                  \
                                                                               \
    std::unordered_map<tensorflow::int64, TapeTensor>                          \
        source_tensors_that_are_targets;                                       \
    std::vector<AbstractTensorHandle*> out_grads(num_grad_outputs);            \
                                                                               \
    int64 target_tensor_ids[num_outputs] = {};                                 \
    for (int i{}; i < num_outputs; ++i) {                                      \
      target_tensor_ids[i] = ToId(temp_outputs[i]);                            \
    }                                                                          \
                                                                               \
    int64 source_tensor_ids[num_inputs] = {};                                  \
    for (int i{}; i < num_inputs; ++i) {                                       \
      source_tensor_ids[i] = ToId(inputs[i]);                                  \
    }                                                                          \
                                                                               \
    TF_RETURN_IF_ERROR(tape->ComputeGradient(                                  \
        vspace, target_tensor_ids, source_tensor_ids,                          \
        source_tensors_that_are_targets, /*output_gradients=*/{}, &out_grads,  \
        /*build_default_zeros_grads=*/false));                                 \
                                                                               \
    for (int i{}; i < num_outputs; ++i) {                                      \
      temp_outputs[i]->Unref();                                                \
    }                                                                          \
    for (int i{}; i < num_grad_outputs; ++i) {                                 \
      outputs[i] = out_grads[i];                                               \
    }                                                                          \
    delete tape;                                                               \
    return Status::OK();                                                       \
  }

void CompareWithGradientsCheckers(Model model, Model grad_model,
                                  AbstractContext* ctx,
                                  absl::Span<AbstractTensorHandle*> inputs,
                                  bool use_function,
                                  const GradientRegistry& registry);

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_GRAD_TEST_HELPER_H_
