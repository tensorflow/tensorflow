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
#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/stats/node-stats.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::test::AsTensor;
using std::vector;

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {
namespace {

const double kDelta = 1e-5;

TEST(NodeStatsTest, AlmostZero) {
  LearnerConfig learner_config;
  learner_config.set_multi_class_strategy(LearnerConfig::TREE_PER_CLASS);
  NodeStats node_stats(learner_config, GradientStats(1e-8f, 1e-8f));
  EXPECT_EQ(0, node_stats.weight_contribution[0]);
  EXPECT_EQ(0, node_stats.gain);
}

TEST(NodeStatsTest, LessThanMinWeightConstraint) {
  LearnerConfig learner_config;
  learner_config.set_multi_class_strategy(LearnerConfig::TREE_PER_CLASS);
  learner_config.mutable_constraints()->set_min_node_weight(3.2f);
  NodeStats node_stats(learner_config, GradientStats(7.32f, 1.63f));
  EXPECT_EQ(0, node_stats.weight_contribution[0]);
  EXPECT_EQ(0, node_stats.gain);
}

TEST(NodeStatsTest, L1RegSquashed) {
  LearnerConfig learner_config;
  learner_config.set_multi_class_strategy(LearnerConfig::TREE_PER_CLASS);
  learner_config.mutable_regularization()->set_l1(10.0f);
  NodeStats node_stats(learner_config, GradientStats(7.32f, 1.63f));
  EXPECT_EQ(0, node_stats.weight_contribution[0]);
  EXPECT_EQ(0, node_stats.gain);
}

TEST(NodeStatsTest, L1RegPos) {
  LearnerConfig learner_config;
  learner_config.set_multi_class_strategy(LearnerConfig::TREE_PER_CLASS);
  learner_config.mutable_regularization()->set_l1(5.0f);
  NodeStats node_stats(learner_config, GradientStats(7.32f, 1.63f));
  const float expected_clipped_grad = 7.32f - 5.0f;
  const float expected_weight_contribution = -expected_clipped_grad / 1.63f;
  const float expected_gain =
      expected_clipped_grad * expected_clipped_grad / 1.63f;
  EXPECT_FLOAT_EQ(expected_weight_contribution,
                  node_stats.weight_contribution[0]);
  EXPECT_FLOAT_EQ(expected_gain, node_stats.gain);
}

TEST(NodeStatsTest, L1RegNeg) {
  LearnerConfig learner_config;
  learner_config.set_multi_class_strategy(LearnerConfig::TREE_PER_CLASS);
  learner_config.mutable_regularization()->set_l1(5.0f);
  NodeStats node_stats(learner_config, GradientStats(-7.32f, 1.63f));
  const float expected_clipped_grad = -7.32f + 5.0f;
  const float expected_weight_contribution = -expected_clipped_grad / 1.63f;
  const float expected_gain =
      expected_clipped_grad * expected_clipped_grad / 1.63f;
  EXPECT_FLOAT_EQ(expected_weight_contribution,
                  node_stats.weight_contribution[0]);
  EXPECT_FLOAT_EQ(expected_gain, node_stats.gain);
}

TEST(NodeStatsTest, L2Reg) {
  LearnerConfig learner_config;
  learner_config.set_multi_class_strategy(LearnerConfig::TREE_PER_CLASS);
  learner_config.mutable_regularization()->set_l2(8.0f);
  NodeStats node_stats(learner_config, GradientStats(7.32f, 1.63f));
  const float expected_denom = 1.63f + 8.0f;
  const float expected_weight_contribution = -7.32f / expected_denom;
  const float expected_gain = 7.32f * 7.32f / expected_denom;
  EXPECT_FLOAT_EQ(expected_weight_contribution,
                  node_stats.weight_contribution[0]);
  EXPECT_FLOAT_EQ(expected_gain, node_stats.gain);
}

TEST(NodeStatsTest, L1L2Reg) {
  LearnerConfig learner_config;
  learner_config.set_multi_class_strategy(LearnerConfig::TREE_PER_CLASS);
  learner_config.mutable_regularization()->set_l1(5.0f);
  learner_config.mutable_regularization()->set_l2(8.0f);
  NodeStats node_stats(learner_config, GradientStats(7.32f, 1.63f));
  const float expected_clipped_grad = 7.32f - 5.0f;
  const float expected_denom = 1.63f + 8.0f;
  const float expected_weight_contribution =
      -expected_clipped_grad / expected_denom;
  const float expected_gain =
      expected_clipped_grad * expected_clipped_grad / expected_denom;
  EXPECT_FLOAT_EQ(expected_weight_contribution,
                  node_stats.weight_contribution[0]);
  EXPECT_FLOAT_EQ(expected_gain, node_stats.gain);
}

TEST(NodeStatsTest, MulticlassFullHessianTest) {
  LearnerConfig learner_config;
  learner_config.set_multi_class_strategy(LearnerConfig::FULL_HESSIAN);
  learner_config.mutable_regularization()->set_l2(0.3f);

  const int kNumClasses = 4;
  const auto& g_shape = TensorShape({1, kNumClasses});
  Tensor g = AsTensor<float>({0.5, 0.33, -9, 1}, g_shape);
  const auto& hessian_shape = TensorShape({1, kNumClasses, kNumClasses});
  Tensor h = AsTensor<float>({3, 5, 7, 8, 5, 4, 1, 5, 7, 1, 8, 4, 8, 5, 4, 9},
                             hessian_shape);

  NodeStats node_stats(learner_config, GradientStats(g, h));

  // Index 1 has 0 value because of l1 regularization,
  std::vector<float> expected_weight = {0.9607576, 0.4162569, 0.9863192,
                                        -1.5820024};

  EXPECT_EQ(kNumClasses, node_stats.weight_contribution.size());
  for (int i = 0; i < kNumClasses; ++i) {
    EXPECT_NEAR(expected_weight[i], node_stats.weight_contribution[i], kDelta);
  }
  EXPECT_NEAR(9.841132, node_stats.gain, kDelta);
}

TEST(NodeStatsTest, MulticlassDiagonalHessianTest) {
  // Normal case.
  {
    LearnerConfig learner_config;
    learner_config.set_multi_class_strategy(LearnerConfig::FULL_HESSIAN);
    learner_config.mutable_regularization()->set_l2(0.3f);

    const int kNumClasses = 4;
    const auto& g_shape = TensorShape({1, kNumClasses});
    Tensor g = AsTensor<float>({0.5, 0.33, -9, 1}, g_shape);
    Tensor h;
    // Full hessian.
    {
      const auto& hessian_shape = TensorShape({1, kNumClasses, kNumClasses});
      // Construct full hessian.
      h = AsTensor<float>({3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 9},
                          hessian_shape);
    }
    NodeStats full_node_stats(learner_config, GradientStats(g, h));

    // Diagonal only.
    {
      const auto& hessian_shape = TensorShape({1, kNumClasses});
      // Construct diagonal of hessian.
      h = AsTensor<float>({3, 4, 8, 9}, hessian_shape);
    }
    learner_config.set_multi_class_strategy(LearnerConfig::DIAGONAL_HESSIAN);
    NodeStats diag_node_stats(learner_config, GradientStats(g, h));

    // Full and diagonal hessian should return the same results.
    EXPECT_EQ(full_node_stats.weight_contribution.size(),
              diag_node_stats.weight_contribution.size());
    for (int i = 0; i < full_node_stats.weight_contribution.size(); ++i) {
      EXPECT_FLOAT_EQ(full_node_stats.weight_contribution[i],
                      diag_node_stats.weight_contribution[i]);
    }
    EXPECT_EQ(full_node_stats.gain, diag_node_stats.gain);
  }
  // Zero entries in diagonal, no regularization
  {
    LearnerConfig learner_config;
    learner_config.set_multi_class_strategy(LearnerConfig::FULL_HESSIAN);

    const int kNumClasses = 4;
    const auto& g_shape = TensorShape({1, kNumClasses});
    Tensor g = AsTensor<float>({0.5, 0.33, -9, 1}, g_shape);
    Tensor h;
    // Full hessian.
    {
      const auto& hessian_shape = TensorShape({1, kNumClasses, kNumClasses});
      // Construct full hessian.
      h = AsTensor<float>({3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0},
                          hessian_shape);
    }
    NodeStats full_node_stats(learner_config, GradientStats(g, h));

    // Diagonal only.
    {
      const auto& hessian_shape = TensorShape({1, kNumClasses});
      // Diagonal of hessian, two entries are 0
      h = AsTensor<float>({3, 0, 8, 0}, hessian_shape);
    }
    learner_config.set_multi_class_strategy(LearnerConfig::DIAGONAL_HESSIAN);
    NodeStats diag_node_stats(learner_config, GradientStats(g, h));

    // Full and diagonal hessian should return the same results.
    EXPECT_EQ(full_node_stats.weight_contribution.size(),
              diag_node_stats.weight_contribution.size());
    for (int i = 0; i < full_node_stats.weight_contribution.size(); ++i) {
      EXPECT_FLOAT_EQ(full_node_stats.weight_contribution[i],
                      diag_node_stats.weight_contribution[i]);
    }
    EXPECT_EQ(full_node_stats.gain, diag_node_stats.gain);
  }
}

}  // namespace
}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow
