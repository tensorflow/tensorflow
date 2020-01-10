/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SQUARED_LOSS_H_
#define TENSORFLOW_CORE_KERNELS_SQUARED_LOSS_H_

#include "tensorflow/core/kernels/loss.h"

namespace tensorflow {

class SquaredLossUpdater : public DualLossUpdater {
 public:
  // Closed form solution that decreases the dual squared loss.
  // See page 23 of http://arxiv.org/pdf/1309.2375v2.pdf for the derivation of
  // the update rule when the example weights are equal to 1.0.
  // Note: There is a typo in the formula in the paper: the denominator should
  // be 1 + ||x_i||^2/(\lambda n) (without the 2 multiplier).
  //
  // The CoCoA+ modification is detailed in readme.md.
  double ComputeUpdatedDual(const int num_loss_partitions, const double label,
                            const double example_weight,
                            const double current_dual, const double wx,
                            const double weighted_example_norm) const final {
    const double delta_numerator = label - current_dual - wx;
    const double delta_denominator =
        1 + num_loss_partitions * weighted_example_norm * example_weight;
    return current_dual + delta_numerator / delta_denominator;
  }

  // Dual of squared loss function.
  // https://en.wikipedia.org/wiki/Convex_conjugate
  double ComputeDualLoss(const double current_dual, const double example_label,
                         const double example_weight) const final {
    // Dual of the squared loss function = b * (y + b/2), where b is the
    // dual variable and y is the label.  This is Dual(-b).
    return current_dual * (0.5 * current_dual - example_label) * example_weight;
  }

  // Squared loss for linear regression.
  double ComputePrimalLoss(const double wx, const double example_label,
                           const double example_weight) const final {
    const double error = wx - example_label;
    return error * error * example_weight * 0.5;
  }

  inline double PrimalLossDerivative(const double wx, const double label,
                                     const double example_weight) const final {
    return (wx - label) * example_weight;
  }

  inline double SmoothnessConstant() const final { return 1.0; }

  // Labels don't require conversion for linear regression.
  Status ConvertLabel(float* const example_label) const final {
    return Status::OK();
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SQUARED_LOSS_H_
