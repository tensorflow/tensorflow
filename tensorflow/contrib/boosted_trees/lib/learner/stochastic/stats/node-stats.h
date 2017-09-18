// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_STATS_NODE_STATS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_STATS_NODE_STATS_H_

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Eigenvalues"
#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/stats/gradient-stats.h"
#include "tensorflow/contrib/boosted_trees/proto/learner.pb.h"
#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

using tensorflow::boosted_trees::learner::LearnerConfig;
using tensorflow::boosted_trees::learner::LearnerConfig_MultiClassStrategy;
using tensorflow::boosted_trees::learner::
    LearnerConfig_MultiClassStrategy_DIAGONAL_HESSIAN;
using tensorflow::boosted_trees::learner::
    LearnerConfig_MultiClassStrategy_FULL_HESSIAN;
using tensorflow::boosted_trees::learner::
    LearnerConfig_MultiClassStrategy_TREE_PER_CLASS;

// NodeStats holds aggregate gradient stats as well as metadata about the node.
struct NodeStats {
  // Initialize the NodeStats with 0 stats.  We need the output length
  // so that we can make weight_contribution the right length.
  explicit NodeStats(const int output_length)
      : weight_contribution(output_length, 0.0f), gain(0) {}

  NodeStats(const LearnerConfig& learner_config,
            const GradientStats& grad_stats)
      : NodeStats(learner_config.regularization().l1(),
                  learner_config.regularization().l2(),
                  learner_config.constraints().min_node_weight(),
                  learner_config.multi_class_strategy(), grad_stats) {}

  NodeStats(float l1_reg, float l2_reg, float min_node_weight,
            const LearnerConfig_MultiClassStrategy& strategy,
            const GradientStats& grad_stats)
      : gradient_stats(grad_stats), gain(0) {
    switch (strategy) {
      case LearnerConfig_MultiClassStrategy_TREE_PER_CLASS: {
        float g;
        float h;
        // Initialize now in case of early return.
        weight_contribution.push_back(0.0f);

        if (grad_stats.first.t.NumElements() == 0 ||
            grad_stats.second.t.NumElements() == 0) {
          return;
        }

        g = grad_stats.first.t.unaligned_flat<float>()(0);
        h = grad_stats.second.t.unaligned_flat<float>()(0);

        if (grad_stats.IsAlmostZero() || h <= min_node_weight) {
          return;
        }

        // Apply L1 regularization.
        if (l1_reg > 0) {
          if (g > l1_reg) {
            g -= l1_reg;
          } else if (g < -l1_reg) {
            g += l1_reg;
          } else {
            return;
          }
        }

        // The node gain is given by: (l'^2) / (l'' + l2_reg) and the node
        // weight
        // contribution is given by: (-l') / (l'' + l2_reg).
        // Note that l'' can't be zero here because of the min node weight check
        // since min node weight must be >= 0.
        weight_contribution[0] = -g / (h + l2_reg);
        gain = (weight_contribution[0] * -g);
        break;
      }
      case LearnerConfig_MultiClassStrategy_FULL_HESSIAN: {
        weight_contribution.clear();

        if (grad_stats.first.t.NumElements() == 0 ||
            grad_stats.second.t.NumElements() == 0) {
          return;
        }
        const int64 grad_dim = grad_stats.first.t.dim_size(1);

        QCHECK(grad_stats.first.t.dims() == 2)
            << strings::Printf("Gradient should be of rank 2, got rank %d",
                               grad_stats.first.t.dims());
        QCHECK(grad_stats.first.t.dim_size(0) == 1) << strings::Printf(
            "Gradient must be of shape 1 x %lld, got %lld x %lld", grad_dim,
            grad_stats.first.t.dim_size(0), grad_dim);
        QCHECK(grad_stats.second.t.dims() == 3)
            << strings::Printf("Hessian should be of rank 3, got rank %d",
                               grad_stats.second.t.dims());
        QCHECK(grad_stats.second.t.shape() ==
               TensorShape({1, grad_dim, grad_dim}))
            << strings::Printf(
                   "Hessian must be of shape 1 x %lld x %lld, got %lld x % lld "
                   " x % lld ",
                   grad_dim, grad_dim, grad_stats.second.t.shape().dim_size(0),
                   grad_stats.second.t.shape().dim_size(1),
                   grad_stats.second.t.shape().dim_size(2));

        // Check if we're violating min weight constraint.

        if (grad_stats.IsAlmostZero() ||
            grad_stats.second.Magnitude() <= min_node_weight) {
          return;
        }
        // TODO(nponomareva): figure out l1 in matrix form.
        // g is a vector of gradients, H is a hessian matrix.
        Eigen::VectorXf g = TensorToEigenVector(grad_stats.first.t, grad_dim);

        Eigen::MatrixXf hessian =
            TensorToEigenMatrix(grad_stats.second.t, grad_dim, grad_dim);
        // I is an identity matrix.
        // The gain in general form is -g^T (H+l2 I)^-1 g.
        // The node weights are -(H+l2 I)^-1 g.
        Eigen::MatrixXf identity;
        identity.setIdentity(grad_dim, grad_dim);

        Eigen::MatrixXf hessian_and_reg = hessian + l2_reg * identity;

        CalculateWeightAndGain(hessian_and_reg, g);
        break;
      }
      case LearnerConfig_MultiClassStrategy_DIAGONAL_HESSIAN: {
        weight_contribution.clear();
        if (grad_stats.first.t.NumElements() == 0 ||
            grad_stats.second.t.NumElements() == 0) {
          return;
        }
        const int64 grad_dim = grad_stats.first.t.dim_size(1);

        QCHECK(grad_stats.first.t.dims() == 2)
            << strings::Printf("Gradient should be of rank 2, got rank %d",
                               grad_stats.first.t.dims());
        QCHECK(grad_stats.first.t.dim_size(0) == 1) << strings::Printf(
            "Gradient must be of shape 1 x %lld, got %lld x %lld", grad_dim,
            grad_stats.first.t.dim_size(0), grad_dim);
        QCHECK(grad_stats.second.t.dims() == 2)
            << strings::Printf("Hessian should be of rank 2, got rank %d",
                               grad_stats.second.t.dims());
        QCHECK(grad_stats.second.t.shape() == TensorShape({1, grad_dim}))
            << strings::Printf(
                   "Hessian must be of shape 1 x %lld, got %lld x %lld",
                   grad_dim, grad_stats.second.t.shape().dim_size(0),
                   grad_stats.second.t.shape().dim_size(1));

        // Check if we're violating min weight constraint.
        if (grad_stats.IsAlmostZero() ||
            grad_stats.second.Magnitude() <= min_node_weight) {
          return;
        }
        // TODO(nponomareva): figure out l1 in matrix form.
        // Diagonal of the hessian.
        Eigen::ArrayXf hessian =
            TensorToEigenArray(grad_stats.second.t, grad_dim);
        Eigen::ArrayXf hessian_and_reg = hessian + l2_reg;

        // Check if any of the elements are zeros.
        bool invertible = true;
        for (int i = 0; i < hessian_and_reg.size(); ++i) {
          if (hessian_and_reg[i] == 0.0) {
            invertible = false;
            break;
          }
        }
        if (invertible) {
          Eigen::ArrayXf g = TensorToEigenArray(grad_stats.first.t, grad_dim);
          // Operations on arrays are element wise. The formulas are as for full
          // hessian, but for hessian of diagonal form they are simplified.
          Eigen::ArrayXf ones = Eigen::ArrayXf::Ones(grad_dim);
          Eigen::ArrayXf temp = ones / hessian_and_reg;
          Eigen::ArrayXf weight = -temp * g;

          // Copy over weights to weight_contribution.
          weight_contribution =
              std::vector<float>(weight.data(), weight.data() + weight.rows());
          gain = (-g * weight).sum();
        } else {
          Eigen::VectorXf g = TensorToEigenVector(grad_stats.first.t, grad_dim);
          // Hessian is not invertible. We will go the same route as in full
          // hessian to get an approximate solution.
          CalculateWeightAndGain(hessian_and_reg.matrix().asDiagonal(), g);
        }
        break;
      }
      default:
        LOG(FATAL) << "Unknown multi-class strategy " << strategy;
        break;
    }
  }

  string DebugString() const {
    return strings::StrCat(
        gradient_stats.DebugString(), "\n",
        "Weight_contrib = ", str_util::Join(weight_contribution, ","),
        "Gain = ", gain);
  }

  // Use these node stats to populate a Leaf's model.
  void FillLeaf(const int class_id, boosted_trees::trees::Leaf* leaf) const {
    if (class_id == -1) {
      for (int i = 0; i < weight_contribution.size(); i++) {
        leaf->mutable_vector()->add_value(weight_contribution[i]);
      }
    } else {
      CHECK(weight_contribution.size() == 1)
          << "Weight contribution size = " << weight_contribution.size();
      leaf->mutable_sparse_vector()->add_index(class_id);
      leaf->mutable_sparse_vector()->add_value(weight_contribution[0]);
    }
  }

  // Sets the weight_contribution and gain member variables based on the
  // given regularized Hessian and gradient vector g.
  void CalculateWeightAndGain(const Eigen::MatrixXf& hessian_and_reg,
                              const Eigen::VectorXf& g) {
    // The gain in general form is -g^T (Hessian_and_regularization)^-1 g.
    // The node weights are -(Hessian_and_regularization)^-1 g.
    Eigen::VectorXf weight;
    // If we want to calculate x = K^-1 v, instead of explicitly calculating
    // K^-1 and multiplying by v, we can solve this matrix equation using
    // solve method.
    weight = -hessian_and_reg.colPivHouseholderQr().solve(g);
    // Copy over weights to weight_contribution.
    weight_contribution =
        std::vector<float>(weight.data(), weight.data() + weight.rows());

    gain = -g.transpose() * weight;
  }

  static Eigen::MatrixXf TensorToEigenMatrix(const Tensor& tensor,
                                             const int num_rows,
                                             const int num_cols) {
    return Eigen::Map<const Eigen::MatrixXf>(tensor.flat<float>().data(),
                                             num_rows, num_cols);
  }

  static Eigen::VectorXf TensorToEigenVector(const Tensor& tensor,
                                             const int num_elements) {
    return Eigen::Map<const Eigen::VectorXf>(tensor.flat<float>().data(),
                                             num_elements);
  }

  static Eigen::ArrayXf TensorToEigenArray(const Tensor& tensor,
                                           const int num_elements) {
    return Eigen::Map<const Eigen::ArrayXf>(tensor.flat<float>().data(),
                                            num_elements);
  }

  GradientStats gradient_stats;
  std::vector<float> weight_contribution;
  float gain;
};

// Helper macro to check std::vector<float> approximate equality.
#define EXPECT_VECTOR_FLOAT_EQ(x, y)       \
  {                                        \
    EXPECT_EQ((x).size(), (y).size());     \
    for (int i = 0; i < (x).size(); ++i) { \
      EXPECT_FLOAT_EQ((x)[i], (y)[i]);     \
    }                                      \
  }

// Helper macro to check node stats approximate equality.
#define EXPECT_NODE_STATS_EQ(val1, val2)                                      \
  EXPECT_GRADIENT_STATS_EQ(val1.gradient_stats, val2.gradient_stats);         \
  EXPECT_VECTOR_FLOAT_EQ(val1.weight_contribution, val2.weight_contribution); \
  EXPECT_FLOAT_EQ(val1.gain, val2.gain);

}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_STATS_NODE_STATS_H_
