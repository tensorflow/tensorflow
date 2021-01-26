/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_ABSTRACT_MODEL_H_
#define TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_ABSTRACT_MODEL_H_

#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/unified_api_testutil.h"

namespace tensorflow {
namespace gradients {

class AbstractModel {
 protected:
  AbstractModel(AbstractContext* ctx, Model loss_fn, Model metric_fn,
                Model optimizer_fn, GradientRegistry registry,
                bool use_function);
  ~AbstractModel();

  std::vector<AbstractTensorHandle*> weights_;

  // Sometime, derived class `Compute` needs to access the current `tape`
  // (mostly for registering custom gradients). `tape_` is guaranted to be
  // `nullptr` when `Compute` is called outside of `Fit` (not training).
  Tape* tape_;

 public:
  // `Compute` will output the raw output of the model. It could be either
  // comparable with the `y_train` or not.
  virtual Status Compute(AbstractContext* ctx,
                         AbstractTensorHandle* const input,
                         AbstractTensorHandle** output) = 0;

  // `operator()` will output a comparable result with `y_train`.
  virtual Status operator()(AbstractContext* ctx,
                            AbstractTensorHandle* const input,
                            AbstractTensorHandle** output) = 0;

  // Train Model
  Status Fit(absl::Span<AbstractTensorHandle* const> x_train,
             absl::Span<AbstractTensorHandle* const> y_train, size_t epoch);

  // Same as `operator()` but works with list of `inputs`.
  Status Predict(absl::Span<AbstractTensorHandle* const> inputs,
                 absl::Span<AbstractTensorHandle*> outputs);

  // Return a scalar which is the accuracy of the model based on `metric_fn_`.
  Status Evaluate(absl::Span<AbstractTensorHandle* const> x_test,
                  absl::Span<AbstractTensorHandle* const> y_test,
                  absl::Span<AbstractTensorHandle*> outputs);

  std::vector<AbstractTensorHandle*>& GetWeights() { return weights_; }

 private:
  AbstractContext* ctx_;
  Model loss_fn_;
  Model metric_fn_;
  Model optimizer_fn_;
  GradientRegistry registry_;
  bool use_function_;
};

}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_ABSTRACT_MODEL_H_
