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

#ifndef TENSORFLOW_CORE_KERNELS_SMOOTH_HINGE_LOSS_H_
#define TENSORFLOW_CORE_KERNELS_SMOOTH_HINGE_LOSS_H_

#include <limits>

#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class SmoothHingeLossUpdater : public DualLossUpdater {
 public:
  // Computes the updated dual variable (corresponding) to a single example. The
  // updated dual value maximizes the objective function of the dual
  // optimization problem associated with smooth hinge loss. The computations
  // are detailed in readme.md.
  double ComputeUpdatedDual(const int num_partitions, const double label,
                            const double example_weight,
                            const double current_dual, const double wx,
                            const double weighted_example_norm) const final {
    // Intuitively there are 3 cases:
    // a. new optimal value of the dual variable falls within the admissible
    // range [0, 1]. In this case we set new dual to this value.
    // b. new optimal value is < 0. Then, because of convexity, the optimal
    // valid value for new dual = 0
    // c. new optimal value > 1.0. Then new optimal value should be set to 1.0.
    const double candidate_optimal_dual =
        current_dual +
        (label - wx - gamma * current_dual) /
            (num_partitions * example_weight * weighted_example_norm + gamma);
    if (label * candidate_optimal_dual < 0) {
      return 0.0;
    }
    if (label * candidate_optimal_dual > 1.0) {
      return label;
    }
    return candidate_optimal_dual;
  }

  double ComputeDualLoss(const double current_dual, const double example_label,
                         const double example_weight) const final {
    // For binary classification, there are 2 conjugate functions, one per
    // label value (-1 and 1).
    const double y_alpha = current_dual * example_label;  // y \alpha
    if (y_alpha < 0 || y_alpha > 1.0) {
      return std::numeric_limits<double>::max();
    }
    return (-y_alpha + 0.5 * gamma * current_dual * current_dual) *
           example_weight;
  }

  double ComputePrimalLoss(const double wx, const double example_label,
                           const double example_weight) const final {
    const double y_wx = example_label * wx;
    if (y_wx >= 1) return 0;
    if (y_wx <= 1 - gamma) return (1 - y_wx - gamma / 2) * example_weight;
    return (1 - y_wx) * (1 - y_wx) * example_weight * 0.5 / gamma;
  }

  // Converts binary example labels from 0.0 or 1.0 to -1.0 or 1.0 respectively
  // as expected by smooth hinge loss.
  Status ConvertLabel(float* const example_label) const final {
    if (*example_label == 0.0) {
      *example_label = -1;
      return OkStatus();
    }
    if (*example_label == 1.0) {
      return OkStatus();
    }
    return errors::InvalidArgument(
        "Only labels of 0.0 or 1.0 are supported right now. "
        "Found example with label: ",
        *example_label);
  }

  double PrimalLossDerivative(const double wx, const double label,
                              const double example_weight) const final {
    if (label * wx >= 1) {
      return 0;
    }
    if (label * wx <= 1 - gamma) {
      return -label;
    }
    return (wx - label) / gamma;
  }

  double SmoothnessConstant() const final { return gamma; }

 private:
  // Smoothness constant of smooth hinge loss
  // TODO(rohananil): expose this parameter
  const double gamma = 1;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SMOOTH_HINGE_LOSS_H_
// TENSORFLOW_KERNELS_SMOOTH_HINGE_LOSS_H_
