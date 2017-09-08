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
#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/handlers/categorical-feature-column-handler.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {
namespace {

using boosted_trees::learner::LearnerConfig;

const auto kClassId = 7;
const auto kSlotId = 0;
const auto kBatchSize = 4;
const auto kFeatureColumn = 3;

using FeatureStatsAccumulator =
    FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>;

class CategoricalFeatureColumnHandlerTest : public ::testing::Test {
 protected:
  // The data looks like the following:
  // Example |  Gradients    | Partition | Feature Id |
  // i0      |  (0.2, 0.12)  |     0     |    1,2     |
  // i1      |  (-0.5, 0.07) |     0     |            |
  // i2      |  (1.2, 0.2)   |     0     |     2      |
  // i3      |  (4.0, 0.13)  |     1     |     0      |
  CategoricalFeatureColumnHandlerTest()
      : example_first_order_gradients_(
            test::AsTensor<float>({0.2f, -0.5f, 1.2f, 4.0f}, {4})),
        example_second_order_gradients_(
            test::AsTensor<float>({0.12f, 0.07f, 0.2f, 0.13f}, {4})),
        example_partitions_({0, 0, 0, 1}),
        indices_(test::AsTensor<int64>({0, 0, 0, 1, 2, 0, 3, 0}, {4, 2})),
        values_(test::AsTensor<int64>({1, 2, 2, 0}, {4})) {
    // Set L2 regularization.
    learner_config_.mutable_regularization()->set_l2(2.0f);
    learner_config_.set_multi_class_strategy(LearnerConfig::TREE_PER_CLASS);
    // Create handler.
    handler_.reset(new CategoricalFeatureColumnHandler(
        kClassId, kSlotId, kBatchSize, kFeatureColumn, indices_.matrix<int64>(),
        values_.vec<int64>()));
  }

  LearnerConfig learner_config_;
  const Tensor example_first_order_gradients_;
  const Tensor example_second_order_gradients_;
  const std::vector<int32> example_partitions_;
  const Tensor indices_;
  const Tensor values_;
  std::unique_ptr<FeatureColumnHandler> handler_;
};

TEST_F(CategoricalFeatureColumnHandlerTest, AggregateGradientStats) {
  // Create handler.
  FeatureStatsAccumulator accumulator(1);
  handler_->AggregateGradientStats(
      example_partitions_, example_first_order_gradients_,
      example_second_order_gradients_, &accumulator);

  // Check stats for each partition and feature.
  // Partition 0, Feature 0.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(0.0f, 0.0f),
                           accumulator.GetStats(kSlotId, kClassId, 0, 0));
  // Partition 0, Feature 1.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(0.2f, 0.12f),
                           accumulator.GetStats(kSlotId, kClassId, 0, 1));
  // Partition 0, Feature 2.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(0.2f + 1.2f, 0.12f + 0.2f),
                           accumulator.GetStats(kSlotId, kClassId, 0, 2));

  // Partition 1, Feature 0.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(4.0f, 0.13f),
                           accumulator.GetStats(kSlotId, kClassId, 1, 0));
  // Partition 1, Feature 1.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(0.0f, 0.0f),
                           accumulator.GetStats(kSlotId, kClassId, 1, 1));
  // Partition 1, Feature 2.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(0.0f, 0.0f),
                           accumulator.GetStats(kSlotId, kClassId, 1, 2));
}

TEST_F(CategoricalFeatureColumnHandlerTest, GenerateFeatureSplitCandidates) {
  // Create handler.
  FeatureStatsAccumulator accumulator(1);
  handler_->AggregateGradientStats(
      example_partitions_, example_first_order_gradients_,
      example_second_order_gradients_, &accumulator);

  // Get feature split candidates for two roots 0 and 1.
  // The root stats are derived from the per-partition total gradient stats.
  const std::vector<int32> roots = {0, 1, 5};
  const std::vector<NodeStats>& root_stats = {
      NodeStats(learner_config_, GradientStats(0.9f, 0.39f)),
      NodeStats(learner_config_, GradientStats(4.0f, 0.13f)), NodeStats(1)};
  std::vector<FeatureSplitCandidate> split_candidates;
  handler_->GenerateFeatureSplitCandidates(learner_config_, roots, root_stats,
                                           accumulator, &split_candidates);
  // Expect three candidate splits (one per root).
  EXPECT_EQ(3, split_candidates.size());

  // Verify candidate for root 0, the best split occurs when we route
  // example i0, i2 left and i1 right.
  const NodeStats expected_left_node0(learner_config_,
                                      GradientStats(0.2f + 1.2f, 0.12f + 0.2f));
  const NodeStats expected_right_node0(
      learner_config_,
      root_stats[0].gradient_stats - expected_left_node0.gradient_stats);
  const SplitStats expected_split_stats0(learner_config_, root_stats[0],
                                         expected_left_node0,
                                         expected_right_node0);
  EXPECT_SPLIT_STATS_EQ(expected_split_stats0, split_candidates[0].split_stats);

  const auto& tree_node0 = split_candidates[0].tree_node;
  EXPECT_EQ(
      boosted_trees::trees::TreeNode::kCategoricalIdBinarySplitFieldNumber,
      tree_node0.node_case());
  const auto& split0 = tree_node0.categorical_id_binary_split();
  EXPECT_EQ(2, split0.feature_id());
  EXPECT_EQ(kFeatureColumn, split0.feature_column());

  // Verify candidate for root 1, there's only one active feature here
  // so zero gain is expected.
  const NodeStats expected_left_node1(learner_config_,
                                      root_stats[1].gradient_stats);
  const NodeStats expected_right_node1(learner_config_, GradientStats(0, 0));
  const SplitStats expected_split_stats1(learner_config_, root_stats[1],
                                         expected_left_node1,
                                         expected_right_node1);
  EXPECT_SPLIT_STATS_EQ(expected_split_stats1, split_candidates[1].split_stats);
  const auto& tree_node1 = split_candidates[1].tree_node;
  EXPECT_EQ(
      boosted_trees::trees::TreeNode::kCategoricalIdBinarySplitFieldNumber,
      tree_node1.node_case());
  const auto& split1 = tree_node1.categorical_id_binary_split();
  EXPECT_EQ(0, split1.feature_id());
  EXPECT_EQ(kFeatureColumn, split1.feature_column());

  // Verify there are no candidate splits for root 5.
  const auto& tree_node2 = split_candidates[2].tree_node;
  EXPECT_EQ(boosted_trees::trees::TreeNode::NODE_NOT_SET,
            tree_node2.node_case());
}

}  // namespace
}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow
