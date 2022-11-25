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

#ifndef TENSORFLOW_CORE_KERNELS_POISSON_LOSS_H_
#define TENSORFLOW_CORE_KERNELS_POISSON_LOSS_H_

#include <cmath>

#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

class PoissonLossUpdater : public DualLossUpdater {
 public:
  // Update is found by a Newton algorithm (see readme.md).
  double ComputeUpdatedDual(const int num_loss_partitions, const double label,
                            const double example_weight,
                            const double current_dual, const double wx,
                            const double weighted_example_norm) const final {
    // Newton algorithm converges quadratically so 10 steps will be largely
    // enough to achieve a very good precision
    static const int newton_total_steps = 10;
    // Initialize the Newton optimization at x such that
    // exp(x) = label - current_dual
    const double y_minus_a = label - current_dual;
    double x = (y_minus_a > 0) ? log(y_minus_a) : 0;
    for (int i = 0; i < newton_total_steps; ++i) {
      x = NewtonStep(x, num_loss_partitions, label, wx, example_weight,
                     weighted_example_norm, current_dual);
    }
    return label - exp(x);
  }

  // Dual of poisson loss function.
  // https://en.wikipedia.org/wiki/Convex_conjugate
  double ComputeDualLoss(const double current_dual, const double example_label,
                         const double example_weight) const final {
    // Dual of the poisson loss function is
    // (y-a)*(log(y-a)-1), where a is the dual variable.
    // It is defined only for a<y.
    const double y_minus_a = example_label - current_dual;
    if (y_minus_a == 0.0) {
      // (y-a)*(log(y-a)-1) approaches 0 as y-a approaches 0.
      return 0.0;
    }
    if (y_minus_a < 0.0) {
      return std::numeric_limits<double>::max();
    }
    return y_minus_a * (log(y_minus_a) - 1) * example_weight;
  }

  double ComputePrimalLoss(const double wx, const double example_label,
                           const double example_weight) const final {
    return (exp(wx) - wx * example_label) * example_weight;
  }

  double PrimalLossDerivative(const double wx, const double label,
                              const double example_weight) const final {
    return (exp(wx) - label) * example_weight;
  }

  // TODO(chapelle): We need to introduce a maximum_prediction parameter,
  // expose that parameter to the user and have this method return
  // 1.0/maximum_prediction.
  // Setting this at 1 for now, it only impacts the adaptive sampling.
  double SmoothnessConstant() const final { return 1; }

  Status ConvertLabel(float* const example_label) const final {
    if (*example_label < 0.0) {
      return errors::InvalidArgument(
          "Only non-negative labels can be used with the Poisson log loss. "
          "Found example with label: ", *example_label);
    }
    return OkStatus();
  }

 private:
  // One Newton step (see readme.md).
  double NewtonStep(const double x, const int num_loss_partitions,
                    const double label, const double wx,
                    const double example_weight,
                    const double weighted_example_norm,
                    const double current_dual) const {
    const double expx = exp(x);
    const double numerator =
        x - wx - num_loss_partitions * weighted_example_norm *
        example_weight * (label - current_dual - expx);
    const double denominator =
       1 + num_loss_partitions * weighted_example_norm * example_weight * expx;
    return x - numerator / denominator;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOGISTIC_LOSS_H_
