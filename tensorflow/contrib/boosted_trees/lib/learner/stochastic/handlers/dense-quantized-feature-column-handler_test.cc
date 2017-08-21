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
#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/handlers/dense-quantized-feature-column-handler.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {
namespace {

using boosted_trees::learner::LearnerConfig;

const auto kClassId = 1;
const auto kSlotId = 0;
const auto kBatchSize = 4;
const auto kFeatureColumn = 2;

using FeatureStatsAccumulator =
    FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>;

class DenseQuantizedFeatureColumnHandlerTest : public ::testing::Test {
 protected:
  // The data looks like the following:
  // Example |  Gradients    | Partition | Dense Quantile |
  // i0      |  (0.2, 0.12)  | 0         | 1              |
  // i1      |  (-0.5, 0.07) | 0         | 1              |
  // i2      |  (1.2, 0.2)   | 0         | 0              |
  // i3      |  (4.0, 0.13)  | 1         | 1              |
  DenseQuantizedFeatureColumnHandlerTest()
      : example_first_order_gradients_(
            test::AsTensor<float>({0.2f, -0.5f, 1.2f, 4.0f}, {4})),
        example_second_order_gradients_(
            test::AsTensor<float>({0.12f, 0.07f, 0.2f, 0.13f}, {4})),
        example_partitions_({0, 0, 0, 1}),
        dense_quantiles_(test::AsTensor<float>({0.3f, 0.52f}, {2})),
        dense_quantized_values_(test::AsTensor<int32>({1, 1, 0, 1}, {4})) {
    // Set L2 regularization.
    learner_config_.mutable_regularization()->set_l2(2.0f);

    // Create handler.
    handler_.reset(new DenseQuantizedFeatureColumnHandler(
        kClassId, kSlotId, kBatchSize, kFeatureColumn,
        dense_quantiles_.vec<float>(), dense_quantized_values_.vec<int32>()));
  }

  LearnerConfig learner_config_;
  const Tensor example_first_order_gradients_;
  const Tensor example_second_order_gradients_;
  const std::vector<int32> example_partitions_;
  const Tensor dense_quantiles_;
  const Tensor dense_quantized_values_;
  std::unique_ptr<FeatureColumnHandler> handler_;
};

TEST_F(DenseQuantizedFeatureColumnHandlerTest, AggregateGradientStats) {
  // Create handler.
  FeatureStatsAccumulator accumulator(1);
  handler_->AggregateGradientStats(
      example_partitions_, example_first_order_gradients_,
      example_second_order_gradients_, &accumulator);

  // Check stats for each partition and feature.
  // Partition 0, Feature 0.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(1.2f, 0.2f),
                           accumulator.GetStats(kSlotId, kClassId, 0, 0));
  // Partition 0, Feature 1.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(-0.3f, 0.19f),
                           accumulator.GetStats(kSlotId, kClassId, 0, 1));
  // Partition 1, Feature 0.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(0.0f, 0.0f),
                           accumulator.GetStats(kSlotId, kClassId, 1, 0));
  // Partition 1, Feature 1.
  EXPECT_GRADIENT_STATS_EQ(GradientStats(4.0f, 0.13f),
                           accumulator.GetStats(kSlotId, kClassId, 1, 1));
}

TEST_F(DenseQuantizedFeatureColumnHandlerTest, GenerateFeatureSplitCandidates) {
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
  // example i2 left and i0, i1 right.
  const NodeStats expected_left_node0(learner_config_,
                                      GradientStats(1.2f, 0.2f));
  const NodeStats expected_right_node0(
      learner_config_,
      root_stats[0].gradient_stats - expected_left_node0.gradient_stats);
  const SplitStats expected_split_stats0(learner_config_, root_stats[0],
                                         expected_left_node0,
                                         expected_right_node0);
  EXPECT_SPLIT_STATS_EQ(expected_split_stats0, split_candidates[0].split_stats);
  const auto& tree_node0 = split_candidates[0].tree_node;
  EXPECT_EQ(boosted_trees::trees::TreeNode::kDenseFloatBinarySplit,
            tree_node0.node_case());
  const auto& split0 = tree_node0.dense_float_binary_split();
  EXPECT_FLOAT_EQ(dense_quantiles_.vec<float>()(0), split0.threshold());
  EXPECT_EQ(kFeatureColumn, split0.feature_column());

  // Verify candidate for root 1, there's only one active bucket here
  // so zero gain is expected.
  const NodeStats expected_left_node1(learner_config_,
                                      root_stats[1].gradient_stats);
  const NodeStats expected_right_node1(learner_config_, GradientStats(0, 0));
  const SplitStats expected_split_stats1(learner_config_, root_stats[1],
                                         expected_left_node1,
                                         expected_right_node1);
  EXPECT_SPLIT_STATS_EQ(expected_split_stats1, split_candidates[1].split_stats);
  const auto& tree_node1 = split_candidates[1].tree_node;
  EXPECT_EQ(boosted_trees::trees::TreeNode::kDenseFloatBinarySplit,
            tree_node1.node_case());
  const auto& split1 = tree_node1.dense_float_binary_split();
  EXPECT_FLOAT_EQ(dense_quantiles_.vec<float>()(1), split1.threshold());
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
