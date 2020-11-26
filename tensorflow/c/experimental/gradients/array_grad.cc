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

namespace tensorflow {
namespace gradients {
namespace {
using std::vector;
class IdentityNGradientFunction : public GradientFunction {
 public:
  Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                 vector<AbstractTensorHandle*>* grad_outputs) override {
    grad_outputs->resize(grad_inputs.size(), nullptr);
    for (int i = 0; i < grad_inputs.size(); i++) {
      auto grad_input = grad_inputs[i];
      // TODO(srbs): Should we add a copy contructor to AbstractTensorHandle
      // that takes care of this similar to `Tensor`?
      if (grad_input) {
        grad_input->Ref();
      }
      (*grad_outputs)[i] = grad_input;
    }
    return Status::OK();
  }
  ~IdentityNGradientFunction() override {}
};
}  // namespace

BackwardFunction* IdentityNRegisterer(const ForwardOperation& op) {
  auto gradient_function = new IdentityNGradientFunction;
  auto default_gradients = new PassThroughDefaultGradients(op);
  return new BackwardFunction(gradient_function, default_gradients);
}

}  // namespace gradients
}  // namespace tensorflow
