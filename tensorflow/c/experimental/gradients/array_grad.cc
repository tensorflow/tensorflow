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

#include "tensorflow/c/eager/abstract_context.h"

namespace tensorflow {
namespace gradients {
namespace {
class IdentityNGradientFunction : public GradientFunction {
 public:
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
    for (int i = 0; i < grad_outputs.size(); i++) {
      auto grad_input = grad_outputs[i];
      // TODO(srbs): Should we add a copy contructor to AbstractTensorHandle
      // that takes care of this similar to `Tensor`?
      if (grad_input) {
        grad_input->Ref();
      }
      grad_inputs[i] = grad_input;
    }
    return Status::OK();
  }
  ~IdentityNGradientFunction() override {}
};
}  // namespace

GradientFunction* IdentityNRegisterer(const ForwardOperation& op) {
  return new IdentityNGradientFunction;
}

}  // namespace gradients
}  // namespace tensorflow
