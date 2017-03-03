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

#include "tensorflow/core/util/ctc/ctc_loss_calculator.h"

namespace tensorflow {
namespace ctc {

// Calculates the alpha(t, u) as described in (GravesTh) Section 7.3.
// Starting with t = 0 instead of t = 1 used in the text.
// Based on Kanishka's CTC.
void CTCLossCalculator::CalculateForwardVariables(
    const std::vector<int>& l_prime, const Matrix& y, bool ctc_merge_repeated,
    Matrix* log_alpha) const {
  // Number of cols is the number of time steps = number of cols in target
  // after the output delay.
  log_alpha->setConstant(kLogZero);

  int U = l_prime.size();
  int T = log_alpha->cols();

  CHECK_EQ(U, log_alpha->rows());

  // Initial alpha values in (GravesTh) Eq 7.5 and Eq 7.6.
  log_alpha->coeffRef(0, 0) = log(y(blank_index_, output_delay_));
  // Below, l_prime[1] == labels[0]
  auto label_0 = (l_prime.size() > 1) ? l_prime[1] : blank_index_;
  log_alpha->coeffRef(1, 0) = log(y(label_0, output_delay_));

  for (int t = 1; t < T; ++t) {
    // If there is not enough time to output the remaining labels or
    // some labels have been skipped, then let log_alpha(u, t) continue to
    // be kLogZero.
    for (int u = std::max(0, U - (2 * (T - t))); u < std::min(U, 2 * (t + 1));
         ++u) {
      // Begin (GravesTh) Eq 7.9
      // Add in the u, t - 1 term.
      float sum_log_alpha = kLogZero;
      if (ctc_merge_repeated || l_prime[u] == blank_index_) {
        sum_log_alpha = log_alpha->coeff(u, t - 1);
      }

      // Add in the u - 1, t - 1 term.
      if (u > 0) {
        sum_log_alpha =
            LogSumExp(sum_log_alpha, log_alpha->coeff(u - 1, t - 1));
      }

      // Add in the u - 2, t - 1 term if l_prime(u) != blank or l_prime(u-2).
      if (u > 1) {
        const bool matching_labels_merge =
            ctc_merge_repeated && (l_prime[u] == l_prime[u - 2]);
        if (l_prime[u] != blank_index_ && !matching_labels_merge) {
          sum_log_alpha =
              LogSumExp(sum_log_alpha, log_alpha->coeff(u - 2, t - 1));
        }
      }
      // Multiply the summed alphas with the activation log probability.
      log_alpha->coeffRef(u, t) =
          log(y(l_prime[u], output_delay_ + t)) + sum_log_alpha;
    }  // End (GravesTh) Eq 7.9.
  }
}

// Calculates the beta(t, u) as described in (GravesTh) Section 7.3.
void CTCLossCalculator::CalculateBackwardVariables(
    const std::vector<int>& l_prime, const Matrix& y, bool ctc_merge_repeated,
    Matrix* log_beta) const {
  // Number of cols is the number of time steps =  number of cols in target.
  // Matrix log_beta =
  //    Matrix::Constant(l_prime.size(), y.cols() - output_delay_,
  // kLogZero);
  log_beta->setConstant(kLogZero);
  int T = log_beta->cols();
  int U = l_prime.size();
  CHECK_EQ(U, log_beta->rows());

  // Initial beta values in (GravesTh) Eq 7.13: log of probability 1.
  for (int u = U - 2; u < U; ++u) log_beta->coeffRef(u, T - 1) = 0;

  for (int t = T - 1 - 1; t >= 0; --t) {
    // If there is not enough time to output the remaining labels or
    // some labels have been skipped, then let log_beta(u, t) continue to
    // be kLogZero.
    for (int u = std::max(0, U - (2 * (T - t))); u < std::min(U, 2 * (t + 1));
         ++u) {
      // Begin (GravesTh) Eq 7.15
      // Add in the u, t + 1 term.
      if (ctc_merge_repeated || l_prime[u] == blank_index_) {
        log_beta->coeffRef(u, t) =
            LogSumExp(log_beta->coeff(u, t),
                      log_beta->coeff(u, t + 1) +
                          log(y(l_prime[u], output_delay_ + t + 1)));
      }

      // Add in the u + 1, t + 1 term.
      if (u + 1 < U) {
        log_beta->coeffRef(u, t) =
            LogSumExp(log_beta->coeff(u, t),
                      log_beta->coeff(u + 1, t + 1) +
                          log(y(l_prime[u + 1], output_delay_ + t + 1)));
      }

      // Add in the u + 2, t + 1 term if l_prime(u) != blank or l_prime(u+2).
      if (u + 2 < U) {
        const bool matching_labels_merge =
            ctc_merge_repeated && (l_prime[u] == l_prime[u + 2]);
        if (l_prime[u] != blank_index_ && !matching_labels_merge) {
          // Add in u + 2 term.
          log_beta->coeffRef(u, t) =
              LogSumExp(log_beta->coeff(u, t),
                        log_beta->coeff(u + 2, t + 1) +
                            log(y(l_prime[u + 2], output_delay_ + t + 1)));
        }
      }  // End (GravesTh) Eq. 7.15
    }
  }
}

// Using (GravesTh) Eq 7.26 & 7.34.
void CTCLossCalculator::CalculateGradient(const std::vector<int>& l_prime,
                                          const Matrix& y,
                                          const Matrix& log_alpha,
                                          const Matrix& log_beta,
                                          float log_p_z_x, Matrix* dy) const {
  // Only working with the leftmost part of dy for this batch element.
  auto dy_b = dy->leftCols(y.cols());

  // It is possible that no valid path is found if the activations for the
  // targets are zero.
  if (log_p_z_x == kLogZero) {
    LOG(WARNING) << "No valid path found.";
    dy_b = y;
    return;
  }

  int L = y.rows();
  int T = y.cols();
  int U = l_prime.size();

  for (int t = 0; t < T - output_delay_; ++t) {
    Array prob_sum(L);
    prob_sum.setConstant(kLogZero);

    for (int u = 0; u < U; ++u) {
      int l = l_prime[u];
      prob_sum[l] = LogSumExp(prob_sum[l], log_alpha(u, t) + log_beta(u, t));
    }

    for (int l = 0; l < L; ++l) {
      // Negative term in (GravesTh) Eq 7.28.
      float negative_term = expf(prob_sum[l] - log_p_z_x);

      dy_b(l, output_delay_ + t) = y(l, output_delay_ + t) - negative_term;
    }
  }
}

void CTCLossCalculator::GetLPrimeIndices(const std::vector<int>& l,
                                         std::vector<int>* l_prime) const {
  // Assumption is that l_prime is empty.
  l_prime->reserve(2 * l.size() + 1);

  for (auto label : l) {
    l_prime->push_back(blank_index_);
    l_prime->push_back(label);
  }
  // Add final blank to l'.
  l_prime->push_back(blank_index_);
}

}  // namespace ctc
}  // namespace tensorflow
