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
#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/handlers/bias-feature-column-handler.h"

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

using FeatureStatsAccumulator =
    FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>;

class BiasFeatureColumnHandlerTest : public ::testing::Test {
 protected:
  BiasFeatureColumnHandlerTest()
      : example_first_order_gradients_(
            test::AsTensor<float>({0.2f, -0.5f, 1.2f, 4.0f}, {4})),
        example_second_order_gradients_(
            test::AsTensor<float>({0.12f, 0.07f, 0.2f, 0.13f}, {4})),
        example_partitions_({0, 0, 1, 3}) {
    // Set L2 regularization.
    learner_config_.mutable_regularization()->set_l2(2.0f);

    // Create handler.
    handler_.reset(new BiasFeatureColumnHandler(kClassId, kSlotId, kBatchSize));
  }

  LearnerConfig learner_config_;
  const Tensor example_first_order_gradients_;
  const Tensor example_second_order_gradients_;
  const std::vector<int32> example_partitions_;
  std::unique_ptr<BiasFeatureColumnHandler> handler_;
};

TEST_F(BiasFeatureColumnHandlerTest, AggregateGradientStats) {
  // Create handler.
  FeatureStatsAccumulator accumulator(1);
  handler_->AggregateGradientStats(
      example_partitions_, example_first_order_gradients_,
      example_second_order_gradients_, &accumulator);

  // Check stats for each partition.
  // Partition 0.
  EXPECT_GRADIENT_STATS_EQ(
      GradientStats(-0.3f, 0.19f),
      accumulator.GetStats(kSlotId, kClassId, 0,
                           BiasFeatureColumnHandler::kBiasFeatureId));
  // Partition 1.
  EXPECT_GRADIENT_STATS_EQ(
      GradientStats(1.2f, 0.2f),
      accumulator.GetStats(kSlotId, kClassId, 1,
                           BiasFeatureColumnHandler::kBiasFeatureId));
  // Partition 2.
  EXPECT_GRADIENT_STATS_EQ(
      GradientStats(0.0f, 0.0f),
      accumulator.GetStats(kSlotId, kClassId, 2,
                           BiasFeatureColumnHandler::kBiasFeatureId));
  // Partition 3.
  EXPECT_GRADIENT_STATS_EQ(
      GradientStats(4.0f, 0.13f),
      accumulator.GetStats(kSlotId, kClassId, 3,
                           BiasFeatureColumnHandler::kBiasFeatureId));
}

TEST_F(BiasFeatureColumnHandlerTest, GenerateFeatureSplitCandidates) {
  // Create handler.
  FeatureStatsAccumulator accumulator(1);
  handler_->AggregateGradientStats(
      example_partitions_, example_first_order_gradients_,
      example_second_order_gradients_, &accumulator);

  // Get feature split candidates for two roots 0 and 3.
  // Root 0 has zero gain and root 3 has the same gain as the leaf.
  const std::vector<int32> roots = {0, 3};
  const std::vector<NodeStats>& root_stats = {
      NodeStats(1), NodeStats(learner_config_, GradientStats(4.0f, 0.13f))};
  std::vector<FeatureSplitCandidate> split_candidates;
  handler_->GenerateFeatureSplitCandidates(learner_config_, roots, root_stats,
                                           accumulator, &split_candidates);
  // Expect two candidate splits (one per root).
  EXPECT_EQ(2, split_candidates.size());

  // Verify first candidate for root 0, gain is expected to be the same as
  // the left child since the root node gain is zero.
  const SplitStats expected_split_stats0(learner_config_, root_stats[0]);
  EXPECT_SPLIT_STATS_EQ(expected_split_stats0, split_candidates[0].split_stats);
  const auto& tree_node0 = split_candidates[0].tree_node;
  EXPECT_EQ(boosted_trees::trees::TreeNode::kLeaf, tree_node0.node_case());
  EXPECT_EQ(1, tree_node0.leaf().sparse_vector().index_size());
  EXPECT_EQ(kClassId, tree_node0.leaf().sparse_vector().index(0));
  EXPECT_EQ(1, tree_node0.leaf().sparse_vector().value_size());
  EXPECT_EQ(root_stats[0].weight_contribution[0],
            tree_node0.leaf().sparse_vector().value(0));

  // Verify second candidate for root 3, gain is expected to be zero as
  // the left child gain is equal to the parent gain.
  const SplitStats expected_split_stats1(learner_config_, root_stats[1]);
  EXPECT_SPLIT_STATS_EQ(expected_split_stats1, split_candidates[1].split_stats);
  const auto& tree_node1 = split_candidates[1].tree_node;
  EXPECT_EQ(boosted_trees::trees::TreeNode::kLeaf, tree_node1.node_case());
  EXPECT_EQ(1, tree_node1.leaf().sparse_vector().index_size());
  EXPECT_EQ(kClassId, tree_node1.leaf().sparse_vector().index(0));
  EXPECT_EQ(1, tree_node1.leaf().sparse_vector().value_size());
  EXPECT_EQ(root_stats[1].weight_contribution[0],
            tree_node1.leaf().sparse_vector().value(0));
}

}  // namespace
}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow
