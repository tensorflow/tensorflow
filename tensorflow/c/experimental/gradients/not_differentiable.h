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
#ifndef TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_NOT_DIFFERENTIABLE_H_
#define TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_NOT_DIFFERENTIABLE_H_

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/gradients.h"

namespace tensorflow {
namespace gradients {
// Ignores `grad_outputs` and sets all entries in grad_inputs to nullptr.
class NotDifferentiableGradientFunction : public GradientFunction {
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override;
};
// Shorthand for registry->Register(op, new NotDifferentiableGradientFunction)
Status RegisterNotDifferentiable(GradientRegistry* registry, const string& op);
}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_NOT_DIFFERENTIABLE_H_
