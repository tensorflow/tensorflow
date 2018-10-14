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

#ifndef TENSORFLOW_CORE_KERNELS_HINGE_LOSS_H_
#define TENSORFLOW_CORE_KERNELS_HINGE_LOSS_H_

#include <algorithm>
#include <limits>

#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class HingeLossUpdater : public DualLossUpdater {
 public:
  // Computes the updated dual variable (corresponding) to a single example. The
  // updated dual value maximizes the objective function of the dual
  // optimization problem associated with hinge loss (conditioned on keeping the
  // rest of the dual variables intact). The method below finds an optimal delta
  // (difference between updated and previous dual value) using the update rule
  // within SDCA procedure (see http://arxiv.org/pdf/1209.1873v2.pdf, page 5)
  // and the particular form of conjugate function for hinge loss.
  //
  // The CoCoA+ modification is detailed in readme.md.
  //
  // TODO(sibyl-vie3Poto): Write up a doc with concrete derivation and point to it from
  // here.
  double ComputeUpdatedDual(const int num_loss_partitions, const double label,
                            const double example_weight,
                            const double current_dual, const double wx,
                            const double weighted_example_norm) const final {
    // Intutitvely there are 3 cases:
    // a. new optimal value of the dual variable falls within the admissible
    // range [0, 1]. In this case we set new dual to this value.
    // b. new optimal value is < 0. Then, because of convexity, the optimal
    // valid value for new dual = 0
    // c. new optimal value > 1.0. Then new optimal value should be set to 1.0.
    const double candidate_optimal_dual =
        current_dual + (label - wx) / (num_loss_partitions * example_weight *
                                       weighted_example_norm);
    if (label * candidate_optimal_dual < 0) {
      return 0.0;
    }
    if (label * candidate_optimal_dual > 1.0) {
      return label;
    }
    return candidate_optimal_dual;
  }

  // Conjugate of hinge loss. This is computed as:
  // \phi*(z) = z if z \in [-1, 0] and +infinity everywhere else. See for
  // instance http://www.eecs.berkeley.edu/~wainwrig/stat241b/lec10.pdf
  // Here we want the weighted version of the conjugate loss. It turns out, that
  // if w is the weight of an example, the conjugate of the weighted hinge loss
  // is given by:
  // \phi*(z) = z if z \in [-w, 0] and +infinity everywhere else. Here the
  // conjugate function depends not only on the weight of the example but also
  // on its label. In particular:
  // \phi_y*(z) = y*z if y*z \in [-w, 0] and +infinity everywhere else where
  // y \in {-1,1}. The following method implements \phi_y*(-\alpha/w).
  double ComputeDualLoss(const double current_dual, const double example_label,
                         const double example_weight) const final {
    // For binary classification, there are 2 conjugate functions, one per
    // label value (-1 and 1).
    const double y_alpha = current_dual * example_label;  // y \alpha
    if (y_alpha < 0 || y_alpha > 1.0) {
      return std::numeric_limits<double>::max();
    }
    return -y_alpha * example_weight;
  }

  // Hinge loss for binary classification for a single example. Hinge loss
  // equals max(0, 1 - y * wx) (see https://en.wikipedia.org/wiki/Hinge_loss).
  // For weighted instances loss should be multiplied by the instance weight.
  double ComputePrimalLoss(const double wx, const double example_label,
                           const double example_weight) const final {
    const double y_wx = example_label * wx;
    return std::max(0.0, 1 - y_wx) * example_weight;
  }

  double PrimalLossDerivative(const double wx, const double label,
                              const double example_weight) const final {
    if (label * wx < 1) {
      return -label * example_weight;
    }
    return 0;
  }

  // The smoothness constant is 0 since the derivative of the loss is not
  // Lipschitz
  double SmoothnessConstant() const final { return 0; }

  // Converts binary example labels from 0.0 or 1.0 to -1.0 or 1.0 respectively
  // as expected by hinge loss.
  Status ConvertLabel(float* const example_label) const final {
    if (*example_label == 0.0) {
      *example_label = -1;
      return Status::OK();
    }
    if (*example_label == 1.0) {
      return Status::OK();
    }
    return errors::InvalidArgument(
        "Only labels of 0.0 or 1.0 are supported right now. "
        "Found example with label: ",
        *example_label);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_HINGE_LOSS_H_
