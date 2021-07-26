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

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_LOSS_CALCULATOR_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_LOSS_CALCULATOR_H_

#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/ctc/ctc_loss_util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace ctc {

template <class T>
class CTCLossCalculator {
  // Connectionist Temporal Classification Loss
  //
  // Implementation by kanishkarao@, posenhuang@, and ebrevdo@.
  //
  // The CTC Loss layer learns a *transition* probability value for each
  // input time step.  The transitions are on the class alphabet
  //   {0, 1, ..., N-2}
  // where N is the depth of the input layer (the size of the alphabet is N-1).
  // Note: The token N-1 is reserved for the "no transition" output, so
  // make sure that your input layer has a depth that's one larger than
  // the set of classes you're training on.  Also make sure that your
  // training labels do not have a class value of N-1, as training will skip
  // these examples.
  //
  // Reference materials:
  //  GravesTh: Alex Graves, "Supervised Sequence Labeling with Recurrent
  //    Neural Networks" (PhD Thesis), Technische Universit¨at M¨unchen.
 public:
  typedef std::vector<std::vector<int>> LabelSequences;
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  // typedef Eigen::MatrixXd Matrix;
  using Array = Eigen::Array<T, Eigen::Dynamic, 1>;
  // typedef Eigen::ArrayXd Array;
  using InputMap = Eigen::Map<const Matrix>;
  // typedef Eigen::Map<const Eigen::MatrixXd> InputMap;
  using OutputMap = Eigen::Map<Matrix>;
  // typedef Eigen::Map<Eigen::MatrixXd> OutputMap;

  CTCLossCalculator(int blank_index, int output_delay)
      : blank_index_(blank_index), output_delay_(output_delay) {}

  template <typename VectorIn, typename VectorOut, typename MatrixIn,
            typename MatrixOut>
  Status CalculateLoss(const VectorIn& seq_len, const LabelSequences& labels,
                       const std::vector<MatrixIn>& inputs,
                       bool preprocess_collapse_repeated,
                       bool ctc_merge_repeated,
                       bool ignore_longer_outputs_than_inputs, VectorOut* loss,
                       std::vector<MatrixOut>* gradients,
                       DeviceBase::CpuWorkerThreads* workers = nullptr) const;

 private:
  void CalculateForwardVariables(const std::vector<int>& l_prime,
                                 const Matrix& y, bool ctc_merge_repeated,
                                 Matrix* log_alpha) const;

  void CalculateBackwardVariables(const std::vector<int>& l_prime,
                                  const Matrix& y, bool ctc_merge_repeated,
                                  Matrix* log_beta) const;

  void CalculateGradient(const std::vector<int>& l_prime, const Matrix& y,
                         const Matrix& log_alpha, const Matrix& log_beta,
                         T log_p_z_x, Matrix* dy) const;

  void GetLPrimeIndices(const std::vector<int>& l,
                        std::vector<int>* l_prime) const;

  // Helper function that calculates the l_prime indices for all
  // batches at the same time, and identifies errors for any given
  // batch.  Return value:
  //    max_{b in batch_size} l_primes[b].size()
  template <typename Vector>
  Status PopulateLPrimes(bool preprocess_collapse_repeated,
                         bool ignore_longer_outputs_than_inputs, int batch_size,
                         int num_classes, const Vector& seq_len,
                         const LabelSequences& labels, size_t* max_u_prime,
                         LabelSequences* l_primes) const;

  // Utility indices for the CTC algorithm.
  int blank_index_;

  // Delay for target labels in time steps.
  // The delay in time steps before the output sequence.
  const int output_delay_;
};

template <class T>
template <typename VectorIn, typename VectorOut, typename MatrixIn,
          typename MatrixOut>
Status CTCLossCalculator<T>::CalculateLoss(
    const VectorIn& seq_len, const LabelSequences& labels,
    const std::vector<MatrixIn>& inputs, bool preprocess_collapse_repeated,
    bool ctc_merge_repeated, bool ignore_longer_outputs_than_inputs,
    VectorOut* loss, std::vector<MatrixOut>* gradients,
    DeviceBase::CpuWorkerThreads* workers) const {
  using Eigen::numext::log;

  auto num_time_steps = inputs.size();

  if (loss == nullptr) {
    return errors::InvalidArgument("loss == nullptr");
  }

  bool requires_backprop = (gradients != nullptr);

  auto batch_size = inputs[0].rows();
  auto num_classes = inputs[0].cols();

  if (loss->size() != batch_size) {
    return errors::InvalidArgument("loss.size() != batch_size");
  }
  loss->setZero();

  for (int t = 1; t < num_time_steps; ++t) {
    if (inputs[t].rows() != batch_size) {
      return errors::InvalidArgument("Expected batch size at t: ", t,
                                     " to be: ", batch_size,
                                     " but got: ", inputs[t].rows());
    }
    if (inputs[t].cols() != num_classes) {
      return errors::InvalidArgument("Expected class count at t: ", t,
                                     " to be: ", num_classes,
                                     " but got: ", inputs[t].cols());
    }
  }

  // Check validity of sequence_length array values.
  auto max_seq_len = seq_len(0);
  for (int b = 0; b < batch_size; b++) {
    if (seq_len(b) < 0) {
      return errors::InvalidArgument("seq_len(", b, ") < 0");
    }
    if (seq_len(b) > num_time_steps) {
      return errors::InvalidArgument("seq_len(", b, ") > num_time_steps");
    }
    max_seq_len = std::max(seq_len(b), max_seq_len);
  }

  // Calculate the modified label sequence l' for each batch element,
  // and calculate the maximum necessary allocation size.
  LabelSequences l_primes(batch_size);
  size_t max_u_prime = 0;
  Status l_p_ret = PopulateLPrimes(
      preprocess_collapse_repeated, ignore_longer_outputs_than_inputs,
      batch_size, num_classes, seq_len, labels, &max_u_prime, &l_primes);
  if (!l_p_ret.ok()) {
    return l_p_ret;
  }

  // Process each item in a batch in parallel, using at most kMaxThreads.
  auto ComputeLossAndGradients = [this, num_classes, &labels, &l_primes,
                                  &seq_len, &inputs, requires_backprop,
                                  ctc_merge_repeated,
                                  ignore_longer_outputs_than_inputs, &loss,
                                  &gradients](int64_t start_row,
                                              int64_t limit_row) {
    for (int b = start_row; b < limit_row; b++) {
      // Return zero gradient for empty sequences or sequences with labels
      // longer than input, which is not supported by CTC.
      if (seq_len(b) == 0 ||
          (ignore_longer_outputs_than_inputs &&
           labels[b].size() > seq_len(b) - this->output_delay_)) {
        VLOG(1) << "The sequence length is either zero or shorter than the "
                   "target output (CTC works only with shorter target sequence "
                   "than input sequence). You can turn this into a warning by "
                   "using the flag ignore_longer_outputs_than_inputs - "
                << b << ": " << str_util::Join(labels[b], " ");
        continue;
      }

      // For each batch element, log(alpha) and log(beta).
      //   row size is: u_prime == l_prime.size()
      //   col size is: seq_len[b] - output_delay_
      const std::vector<int>& l_prime = l_primes[b];

      Matrix log_alpha_b(l_prime.size(), seq_len(b) - this->output_delay_);
      Matrix log_beta_b(l_prime.size(), seq_len(b) - this->output_delay_);

      // Work matrices, pre-allocated to the size required by this batch item.
      Matrix y(num_classes, seq_len(b));
      Matrix dy;
      if (requires_backprop) {
        dy = Matrix::Zero(y.rows(), y.cols());
      }

      // For this batch, we'll only work with this shortened sequence_length.
      Matrix y_b = y.leftCols(seq_len(b));

      // Convert label from DistBelief
      // y, prob are in num_classes x seq_len(b)
      // Output activations.
      Array y_b_col;
      for (int t = 0; t < seq_len(b); t++) {
        // Calculate the softmax of y_b.  Use original precision
        // arithmetic for the sum.
        T max_coeff = inputs[t].row(b).maxCoeff();
        y_b_col = (inputs[t].row(b).array() - max_coeff).exp();
        y_b.col(t) = y_b_col / y_b_col.sum();
      }

      // Compute forward, backward.
      // Forward variables.
      CalculateForwardVariables(l_prime, y_b, ctc_merge_repeated, &log_alpha_b);
      // Backward variables.
      CalculateBackwardVariables(l_prime, y_b, ctc_merge_repeated, &log_beta_b);

      // The loss is computed as the log(p(z|x)) between the target and
      // prediction. Do lazy evaluation of log_prob here.
      T log_p_z_x = kLogZero<T>();
      for (int u = 0; u < l_prime.size(); ++u) {
        // (GravesTh) Eq 7.26, sum over all paths for t = 0.
        log_p_z_x = LogSumExp(log_p_z_x, log_alpha_b(u, 0) + log_beta_b(u, 0));
      }

      (*loss)(b) = -log_p_z_x;  // Use negative log loss for display.

      // We compute the derivative if needed.
      if (requires_backprop) {
        // Gradients with respect to input activations.
        // Calculate gradient.
        dy.setZero();
        CalculateGradient(l_prime, y_b, log_alpha_b, log_beta_b, log_p_z_x,
                          &dy);

        // Convert gradient for current sample to DistBelief.
        for (int t = 0; t < seq_len(b); t++) {
          (*gradients)[t].row(b).array() = dy.col(t);
        }
      }
    }  // for (int b = ...
  };
  if (workers) {
    // *Rough* estimate of the cost for one item in the batch.
    // Forward, Backward: O(T * U (= 2L + 1)), Gradients: O(T * (U + L)).
    //
    // softmax: T * L * (Cost(Exp) + Cost(Div))softmax +
    // fwd,bwd: T * 2 * (2*L + 1) * (Cost(LogSumExp) + Cost(Log)) +
    // grad: T * ((2L + 1) * Cost(LogSumExp) + L * (Cost(Expf) + Cost(Add)).
    const int64_t cost_exp = Eigen::internal::functor_traits<
        Eigen::internal::scalar_exp_op<T>>::Cost;
    const int64_t cost_log = Eigen::internal::functor_traits<
        Eigen::internal::scalar_log_op<T>>::Cost;
    const int64_t cost_log_sum_exp =
        Eigen::TensorOpCost::AddCost<T>() + cost_exp + cost_log;
    const int64_t cost =
        max_seq_len * num_classes *
            (cost_exp + Eigen::TensorOpCost::DivCost<T>()) +
        max_seq_len * 2 * (2 * num_classes + 1) *
            (cost_log_sum_exp + cost_log) +
        max_seq_len *
            ((2 * num_classes + 1) * cost_log_sum_exp +
             num_classes * (cost_exp + Eigen::TensorOpCost::AddCost<T>()));
    Shard(workers->num_threads, workers->workers, batch_size, cost,
          ComputeLossAndGradients);
  } else {
    ComputeLossAndGradients(0, batch_size);
  }
  return Status::OK();
}

template <class T>
template <typename Vector>
Status CTCLossCalculator<T>::PopulateLPrimes(
    bool preprocess_collapse_repeated, bool ignore_longer_outputs_than_inputs,
    int batch_size, int num_classes, const Vector& seq_len,
    const LabelSequences& labels, size_t* max_u_prime,
    LabelSequences* l_primes) const {
  // labels is a Label array of size batch_size
  if (labels.size() != batch_size) {
    return errors::InvalidArgument(
        "labels.size() != batch_size: ", labels.size(), " vs. ", batch_size);
  }

  *max_u_prime = 0;  // keep track of longest l' modified label sequence.
  for (int b = 0; b < batch_size; b++) {
    // Assume label is in Label proto
    const std::vector<int>& label = labels[b];
    if (label.size() == 0) {
      return errors::InvalidArgument("Labels length is zero in batch ", b);
    }

    // If debugging: output the labels coming into training.
    //
    VLOG(2) << "label for batch: " << b << ": " << str_util::Join(label, " ");

    // Target indices, length = U.
    std::vector<int> l;

    // Convert label from DistBelief
    bool finished_sequence = false;
    for (int i = 0; i < label.size(); ++i) {
      if (i == 0 || !preprocess_collapse_repeated || label[i] != label[i - 1]) {
        if (label[i] >= num_classes - 1) {
          finished_sequence = true;
        } else {
          if (finished_sequence) {
            // Saw an invalid sequence with non-null following null
            // labels.
            return errors::InvalidArgument(
                "Saw a non-null label (index >= num_classes - 1) "
                "following a ",
                "null label, batch: ", b, " num_classes: ", num_classes,
                " labels: ", str_util::Join(label, ","),
                " labels seen so far: ", str_util::Join(l, ","));
          }
          l.push_back(label[i]);
        }
      }
    }

    for (int l_i : l) {
      if (l_i < 0) {
        return errors::InvalidArgument(
            "All labels must be nonnegative integers, batch: ", b,
            " labels: ", str_util::Join(l, ","));
      } else if (l_i >= num_classes) {
        return errors::InvalidArgument(
            "No label may be greater than num_classes. ",
            "num_classes: ", num_classes, ", batch: ", b,
            " labels: ", str_util::Join(l, ","));
      }
    }
    if (!ignore_longer_outputs_than_inputs) {
      // Make sure there is enough time to output the target indices.
      int time = seq_len(b) - output_delay_;
      int required_time = label.size();
      if (required_time > time) {
        return errors::InvalidArgument(
            "Not enough time for target transition sequence ("
            "required: ",
            required_time, ", available: ", time, ")", b,
            "You can turn this error into a warning by using the flag "
            "ignore_longer_outputs_than_inputs");
      }
    }
    // Target indices with blanks before each index and a blank at the end.
    // Length U' = 2U + 1.
    // Convert l to l_prime
    GetLPrimeIndices(l, &l_primes->at(b));
    *max_u_prime = std::max(*max_u_prime, l_primes->at(b).size());
  }
  return Status::OK();
}

// Calculates the alpha(t, u) as described in (GravesTh) Section 7.3.
// Starting with t = 0 instead of t = 1 used in the text.
// Based on Kanishka's CTC.
template <typename TT>
void CTCLossCalculator<TT>::CalculateForwardVariables(
    const std::vector<int>& l_prime, const Matrix& y, bool ctc_merge_repeated,
    Matrix* log_alpha) const {
  using Eigen::numext::log;

  // Number of cols is the number of time steps = number of cols in target
  // after the output delay.
  log_alpha->setConstant(kLogZero<TT>());

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
      auto sum_log_alpha = kLogZero<TT>();
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
template <class TT>
void CTCLossCalculator<TT>::CalculateBackwardVariables(
    const std::vector<int>& l_prime, const Matrix& y, bool ctc_merge_repeated,
    Matrix* log_beta) const {
  // Number of cols is the number of time steps =  number of cols in target.
  // Matrix log_beta =
  //    Matrix::Constant(l_prime.size(), y.cols() - output_delay_,
  // kLogZero);
  using Eigen::numext::log;

  log_beta->setConstant(kLogZero<TT>());
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
template <typename TT>
void CTCLossCalculator<TT>::CalculateGradient(const std::vector<int>& l_prime,
                                              const Matrix& y,
                                              const Matrix& log_alpha,
                                              const Matrix& log_beta,
                                              TT log_p_z_x, Matrix* dy) const {
  // Only working with the leftmost part of dy for this batch element.
  auto dy_b = dy->leftCols(y.cols());

  // It is possible that no valid path is found if the activations for the
  // targets are zero.
  if (log_p_z_x == kLogZero<TT>()) {
    LOG(WARNING) << "No valid path found.";
    dy_b = y;
    return;
  }

  int L = y.rows();
  int T = y.cols();
  int U = l_prime.size();

  for (int t = 0; t < T - output_delay_; ++t) {
    Array prob_sum(L);
    prob_sum.setConstant(kLogZero<TT>());

    for (int u = 0; u < U; ++u) {
      int l = l_prime[u];
      prob_sum[l] = LogSumExp(prob_sum[l], log_alpha(u, t) + log_beta(u, t));
    }

    for (int l = 0; l < L; ++l) {
      // Negative term in (GravesTh) Eq 7.28.
      auto negative_term = expf(prob_sum[l] - log_p_z_x);

      dy_b(l, output_delay_ + t) = y(l, output_delay_ + t) - negative_term;
    }
  }
}

template <class TT>
void CTCLossCalculator<TT>::GetLPrimeIndices(const std::vector<int>& l,
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

#endif  // TENSORFLOW_CORE_UTIL_CTC_CTC_LOSS_CALCULATOR_H_
