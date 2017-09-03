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

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

namespace {

// Creates a dense split node without assigning children.
boosted_trees::trees::TreeNode CreateDenseSplitNode(const int32 feature_column,
                                                    const float threshold) {
  boosted_trees::trees::TreeNode split_node;
  auto* split = split_node.mutable_dense_float_binary_split();
  split->set_feature_column(feature_column);
  split->set_threshold(threshold);
  return split_node;
}

}  // namespace

void DenseQuantizedFeatureColumnHandler::AggregateGradientStats(
    const std::vector<int32>& example_partition_ids,
    const Tensor& example_first_order_gradients,
    const Tensor& example_second_order_gradients,
    FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>*
        gradient_stats_accumulator) const {
  // Pass over all examples and aggregate gradient stats for each partition
  // and quantized feature bucket.
  for (int64 example_idx = 0; example_idx < batch_size_; ++example_idx) {
    auto partition_id = example_partition_ids[example_idx];
    auto feature_id = dense_quantized_values_(example_idx);
    gradient_stats_accumulator->AddStats(
        slot_id_, class_id_, partition_id, feature_id,
        GradientStats(example_first_order_gradients,
                      example_second_order_gradients, example_idx));
  }
}

void DenseQuantizedFeatureColumnHandler::GenerateFeatureSplitCandidates(
    const LearnerConfig& learner_config, const std::vector<int32>& roots,
    const std::vector<NodeStats>& root_stats,
    const FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>&
        gradient_stats_accumulator,
    std::vector<FeatureSplitCandidate>* split_candidates) const {
  // Evaluate split candidates for every root as each is a separate
  // logical partition over the examples.
  // Then for each root, we do a forward-only pass over the quantized
  // feature buckets accumulating gradients from left to right.
  // Split gains are evaluated at every threshold and the best split is picked.
  split_candidates->clear();
  split_candidates->reserve(roots.size());
  for (size_t root_idx = 0; root_idx < roots.size(); ++root_idx) {
    // Get partition Id and root node stats.
    const int32 partition_id = roots[root_idx];
    const NodeStats& root_node_stats = root_stats[root_idx];

    // Forward left to right pass over quantiles.
    GradientStats left_gradient_stats;
    GradientStats right_gradient_stats(root_node_stats.gradient_stats);
    FeatureSplitCandidate best_split_candidate(
        root_node_stats.weight_contribution.size());
    best_split_candidate.split_stats.gain =
        std::numeric_limits<float>::lowest();
    for (int bucket_id = 0; bucket_id < dense_quantiles_.size(); ++bucket_id) {
      // Get gradient stats.
      auto gradient_stats = gradient_stats_accumulator.GetStats(
          slot_id_, class_id_, partition_id, bucket_id);
      if (gradient_stats.IsZero()) {
        continue;
      }

      // Update gradient stats.
      left_gradient_stats += gradient_stats;
      right_gradient_stats =
          root_node_stats.gradient_stats - left_gradient_stats;

      // Get node stats
      NodeStats left_node_stats(learner_config, left_gradient_stats);
      NodeStats right_node_stats(learner_config, right_gradient_stats);

      // Generate split candidate.
      const float threshold = dense_quantiles_(bucket_id);
      FeatureSplitCandidate split_candidate(
          slot_id_, CreateDenseSplitNode(dense_feature_column_, threshold),
          SplitStats(learner_config, root_node_stats, left_node_stats,
                     right_node_stats));
      if (split_candidate.split_stats.gain >
          best_split_candidate.split_stats.gain) {
        best_split_candidate = std::move(split_candidate);
      }
    }

    // Add best candidate for partition.
    split_candidates->push_back(std::move(best_split_candidate));
  }
}

}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow
