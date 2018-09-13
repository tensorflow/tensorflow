/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_LITE_TOOLS_ACCURACY_ACCURACY_EVAL_STAGE_H_
#define TENSORFLOW_CONTRIB_LITE_TOOLS_ACCURACY_ACCURACY_EVAL_STAGE_H_

#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace metrics {

// Base class for evaluation stage that evaluates the accuracy of the model.
// This stage calculates the accuracy metrics given the model outputs and
// expected ground truth.
class AccuracyEval {
 public:
  AccuracyEval() = default;
  AccuracyEval(const AccuracyEval&) = delete;
  AccuracyEval& operator=(const AccuracyEval&) = delete;

  AccuracyEval(const AccuracyEval&&) = delete;
  AccuracyEval& operator=(const AccuracyEval&&) = delete;

  virtual ~AccuracyEval() = default;

  // Evaluates the accuracy of the model for given `model_outputs` and the
  // `ground truth`.
  // Derived classes can do additional book keeping, calculate aggregrate
  // statistics etc for the given model.
  virtual Status ComputeEval(const std::vector<Tensor>& model_outputs,
                             const Tensor& ground_truth) = 0;
};
}  //  namespace metrics
}  //  namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_LITE_TOOLS_ACCURACY_ACCURACY_EVAL_STAGE_H_
