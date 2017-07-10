/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_TRAINING_GRADIENT_DESCENT_OPTIMIZER_H_
#define TENSORFLOW_CC_TRAINING_GRADIENT_DESCENT_OPTIMIZER_H_

#include "tensorflow/cc/training/optimizer.h"

namespace tensorflow {

class GradientDescentOptimizer : public Optimizer {
 public:
  explicit GradientDescentOptimizer(float learning_rate)
      : Optimizer("GradientDescent"), learning_rate_(learning_rate) {}

  // Add ops to apply dense gradients to `var`.
  Output ApplyDense(const Scope& scope, const Output& grad,
                    const Output& var) const;

  void Prepare(const Scope& scope) {}  // useless for gradient descent

 private:
  float learning_rate_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_TRAINING_GRADIENT_DESCENT_OPTIMIZER_H_
