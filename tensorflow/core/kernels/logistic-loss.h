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

#ifndef TENSORFLOW_CORE_KERNELS_LOGISTIC_LOSS_H_
#define TENSORFLOW_CORE_KERNELS_LOGISTIC_LOSS_H_

#include <cmath>

#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

class LogisticLossUpdater : public DualLossUpdater {
 public:
  // Adding vs. Averaging in Distributed Primal-Dual Optimization.
  // Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan, Peter
  // Richtarik, Martin Takac http://arxiv.org/abs/1502.03508
  double ComputeUpdatedDual(const int num_loss_partitions, const double label,
                            const double example_weight,
                            const double current_dual, const double wx,
                            const double weighted_example_norm) const final {
    // Newton algorithm converges quadratically so 10 steps will be largely
    // enough to achieve a very good precision
    static const int newton_total_steps = 10;
    double x = 0;
    for (int i = 0; i < newton_total_steps; ++i) {
      x = NewtonStep(x, num_loss_partitions, label, wx, example_weight,
                     weighted_example_norm, current_dual);
    }
    return 0.5 * (1 + tanh(x)) / label;
  }

  // Dual of logisitic loss function.
  // https://en.wikipedia.org/wiki/Convex_conjugate
  double ComputeDualLoss(const double current_dual, const double example_label,
                         const double example_weight) const final {
    // Dual of the logistic loss function is
    // ay * log(ay) + (1-ay) * log (1-ay), where a is the dual variable.
    const double ay = current_dual * example_label;
    const double log_ay = (ay > 0) ? log(ay) : 0;
    const double one_minus_ay = 1 - ay;
    const double log_one_minus_ay = (one_minus_ay > 0) ? log(one_minus_ay) : 0;
    return ((ay * log_ay) + (one_minus_ay * log_one_minus_ay)) * example_weight;
  }

  // Logistic loss for binary classification.
  // https://en.wikipedia.org/wiki/Loss_functions_for_classification
  double ComputePrimalLoss(const double wx, const double example_label,
                           const double example_weight) const final {
    // Logistic loss:
    //   log(1 + e^(-ywx))
    //   log(e^0 + e^(-ywx))
    //   a + log(e^(0-a) + e^(-ywx - a)),  where a is max(0, -ywx)
    // https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
    const double y_wx = example_label * wx;
    if (y_wx > 0) {
      // 0 + log(e^(0) + e^(-ywx - 0))
      // log(1 + e^(-ywx))
      return log(1 + exp(-y_wx)) * example_weight;
    }
    // -ywx + log(e^(ywx) + e^(-ywx + ywx))
    // log(e^(ywx) + e^(0)) - ywx
    // log(1 + e^(ywx)) - ywx
    return (log(1 + exp(y_wx)) - y_wx) * example_weight;
  }

  // Derivative of logistic loss
  double PrimalLossDerivative(const double wx, const double label,
                              const double example_weight) const final {
    double inverse_exp_term = 0;
    if (label * wx > 0) {
      inverse_exp_term = exp(-label * wx) / (1 + exp(-label * wx));
    } else {
      inverse_exp_term = 1 / (1 + exp(label * wx));
    }
    return -inverse_exp_term * label * example_weight;
  }

  // The smoothness constant is 4 since the derivative of logistic loss, which
  // is exp(-x) / (1 + exp(-x)) can be shown to 0.25-Lipschitz (its derivative
  // is bounded by 0.25)
  double SmoothnessConstant() const final { return 4; }

  // Converts binary example labels from 0.0 or 1.0 to -1.0 or 1.0 respectively
  // as expected by logistic regression.
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

 private:
  // We use Newton algorithm on a modified function (see readme.md).
  double NewtonStep(const double x, const int num_loss_partitions,
                    const double label, const double wx,
                    const double example_weight,
                    const double weighted_example_norm,
                    const double current_dual) const {
    const double tanhx = tanh(x);
    const double numerator = -2 * label * x - wx -
                             num_loss_partitions * weighted_example_norm *
                                 example_weight *
                                 (0.5 * (1 + tanhx) / label - current_dual);
    const double denominator =
        -2 * label - num_loss_partitions * weighted_example_norm *
                         example_weight * (1 - tanhx * tanhx) * 0.5 / label;
    return x - numerator / denominator;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOGISTIC_LOSS_H_
