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

#ifndef TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_TREE_HELPER_H_
#define TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_TREE_HELPER_H_
#include <cmath>
#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/QR"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace boosted_trees {
// TODO(nponomareva, youngheek): consider using vector.
struct SplitCandidate {
  SplitCandidate() {}

  // Index in the tensor of node_ids for the feature with idx feature_idx.
  int64 candidate_idx = 0;

  int64 feature_id = 0;
  float gain = 0.0;
  int32 threshold = 0.0;
  int32 dimension_id = 0;
  std::vector<float> left_node_contribs;
  std::vector<float> right_node_contribs;
  // The split type, i.e., with missing value to left/right.
  string split_type;
};
}  // namespace boosted_trees

static bool GainsAreEqual(const float g1, const float g2) {
  const float kTolerance = 1e-15;
  return std::abs(g1 - g2) < kTolerance;
}

static bool GainIsLarger(const float g1, const float g2) {
  const float kTolerance = 1e-15;
  return g1 - g2 >= kTolerance;
}

static void MultiDimLogitSolveForWeightAndGain(
    const Eigen::MatrixXf& hessian_and_reg, const Eigen::VectorXf& g,
    Eigen::VectorXf* weight, float* gain) {
  *weight = -hessian_and_reg.colPivHouseholderQr().solve(g);
  *gain = -g.transpose() * (*weight);
}

// Used in stats_ops.cc to determine weights/gains for each feature split.
static void CalculateWeightsAndGains(const Eigen::VectorXf& g,
                                     const Eigen::VectorXf& h, const float l1,
                                     const float l2, Eigen::VectorXf* weight,
                                     float* gain) {
  const float kEps = 1e-15;
  const int32 logits_dim = g.size();
  if (logits_dim == 1) {
    // The formula for weight is -(g+l1*sgn(w))/(H+l2), for gain it is
    // (g+l1*sgn(w))^2/(h+l2).
    // This is because for each leaf we optimize
    // 1/2(h+l2)*w^2+g*w+l1*abs(w)
    float g_with_l1 = g[0];
    // Apply L1 regularization.
    // 1) Assume w>0 => w=-(g+l1)/(h+l2)=> g+l1 < 0 => g < -l1
    // 2) Assume w<0 => w=-(g-l1)/(h+l2)=> g-l1 > 0 => g > l1
    // For g from (-l1, l1), thus there is no solution => set to 0.
    if (l1 > 0) {
      if (g[0] > l1) {
        g_with_l1 -= l1;
      } else if (g[0] < -l1) {
        g_with_l1 += l1;
      } else {
        weight->coeffRef(0) = 0.0;
        *gain = 0.0;
        return;
      }
    }
    // Apply L2 regularization.
    if (h[0] + l2 <= kEps) {
      // Avoid division by 0 or infinitesimal.
      weight->coeffRef(0) = 0;
      *gain = 0;
    } else {
      weight->coeffRef(0) = -g_with_l1 / (h[0] + l2);
      *gain = -g_with_l1 * weight->coeffRef(0);
    }
  } else if (h.size() == logits_dim * logits_dim) { /* Full Hessian */
    Eigen::MatrixXf identity;
    identity.setIdentity(logits_dim, logits_dim);
    // TODO(crawles): figure out L1 regularization for matrix form.
    Eigen::MatrixXf hessian_and_reg =
        h.reshaped(logits_dim, logits_dim) + l2 * identity;
    MultiDimLogitSolveForWeightAndGain(hessian_and_reg, g, weight, gain);
  } else if (h.size() == logits_dim) { /* Diagonal Hessian approximation. */
    // TODO(crawles): figure out L1 regularization for matrix form.
    Eigen::ArrayXf hessian_and_reg = h.array() + l2;
    // Check if any of the elements are zeros.
    bool invertible = true;
    for (int i = 0; i < hessian_and_reg.size(); ++i) {
      if (hessian_and_reg[i] == 0.0) {
        invertible = false;
        break;
      }
    }
    if (invertible) {
      // Operations on arrays are element wise. The formulas are as for full
      // hessian, but for hessian of diagonal form they are simplified.
      Eigen::ArrayXf ones = Eigen::ArrayXf::Ones(logits_dim);
      Eigen::ArrayXf temp = ones / hessian_and_reg;
      *weight = -temp * g.array();
      *gain = (-g.array() * (*weight).array()).sum();
    } else {
      // Hessian is not invertible. We will go the same route as in full
      // hessian to get an approximate solution.
      MultiDimLogitSolveForWeightAndGain(hessian_and_reg.matrix().asDiagonal(),
                                         g, weight, gain);
    }
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_TREE_HELPER_H_
