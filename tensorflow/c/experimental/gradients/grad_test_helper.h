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

void CompareWithGradientsCheckers(Model model, Model grad_model,
                                  AbstractContext* ctx,
                                  std::vector<AbstractTensorHandle*> inputs,
                                  bool use_function,
                                  const GradientRegistry& registry);

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_GRAD_TEST_HELPER_H_
