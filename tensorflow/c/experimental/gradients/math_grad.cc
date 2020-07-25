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

#include "tensorflow/c/experimental/ops/array_ops.h"

using tensorflow::ops::Identity;

namespace tensorflow {
namespace gradients {
namespace {

class AddGradientFunction : public GradientFunction {
 public:
  Status Compute(Context* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_inputs,
                 std::vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(2);
    std::vector<AbstractTensorHandle*> identity_outputs(1);
    // TODO(b/145674566): Handle name unification in tracing code.
    // TODO(b/161805092): Support broadcasting.
    TF_RETURN_IF_ERROR(ops::Identity(ctx->ctx, {grad_inputs[0]},
                                     absl::MakeSpan(identity_outputs),
                                     "Identity0"));
    (*grad_outputs)[0] = identity_outputs[0];
    TF_RETURN_IF_ERROR(ops::Identity(ctx->ctx, {grad_inputs[0]},
                                     absl::MakeSpan(identity_outputs),
                                     "Identity1"));
    (*grad_outputs)[1] = identity_outputs[0];
    return Status::OK();
  }
  ~AddGradientFunction() override {}
};

}  // namespace

GradientFunction* AddRegisterer(const ForwardOperation& op) {
  return new AddGradientFunction;
}
}  // namespace gradients
}  // namespace tensorflow
