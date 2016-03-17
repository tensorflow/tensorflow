/* Copyright 2016 Google Inc. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LINEAR_OPTIMIZER_KERNELS_LOSS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LINEAR_OPTIMIZER_KERNELS_LOSS_H_

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class DualLossUpdater {
 public:
  virtual ~DualLossUpdater() {}

  // Compute update dual (alpha), based on a single example. Various strategies
  // can be employed here, like newton step and/or line search or approximate
  // step that decreases the dual sub-optimality.
  virtual double ComputeUpdatedDual(const double label,
                                    const double example_weight,
                                    const double current_dual, const double wx,
                                    const double weighted_example_norm,
                                    const double primal_loss,
                                    const double dual_loss) const = 0;

  // Compute dual loss based on the current dual (alpha), example label (y)
  // and example weight (cost).
  virtual double ComputeDualLoss(const double current_dual,
                                 const double example_label,
                                 const double example_weight) const = 0;

  // Compute the primal loss based on current estimate of log-odds(wx),
  // example label (y) and example weight (cost).
  virtual double ComputePrimalLoss(const double wx, const double example_label,
                                   const double example_weight) const = 0;

  // Converts binary example labels from 0.0 or 1.0 to appropriate range for
  // each loss function.
  virtual Status ConvertLabel(float* const example_label) const = 0;
};

}  // namespace tensorflow
#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LINEAR_OPTIMIZER_KERNELS_LOSS_H_
