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
#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/handlers/sparse-quantized-feature-column-handler.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

namespace {

// Creates a sparse default right split node without assigning children.
boosted_trees::trees::TreeNode CreateSparseSplitNodeDefaultRight(
    int32 feature_column, float threshold) {
  boosted_trees::trees::TreeNode split_node;
  auto* split = split_node.mutable_sparse_float_binary_split_default_right()
                    ->mutable_split();
  split->set_feature_column(feature_column);
  split->set_threshold(threshold);
  return split_node;
}

// Creates a sparse default left split node without assigning children.
boosted_trees::trees::TreeNode CreateSparseSplitNodeDefaultLeft(
    int32 feature_column, float threshold) {
  boosted_trees::trees::TreeNode split_node;
  auto* split = split_node.mutable_sparse_float_binary_split_default_left()
                    ->mutable_split();
  split->set_feature_column(feature_column);
  split->set_threshold(threshold);
  return split_node;
}

}  // namespace

void SparseQuantizedFeatureColumnHandler::AggregateGradientStats(
    const std::vector<int32>& example_partition_ids,
    const Tensor& example_first_order_gradients,
    const Tensor& example_second_order_gradients,
    FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>*
        gradient_stats_accumulator) const {
  // Pass over all rows and aggregate gradient stats for each partition
  // and quantized feature bucket.
  const int64 num_rows = sparse_indices_.dimension(0);
  for (int64 row_idx = 0; row_idx < num_rows; ++row_idx) {
    auto example_idx = sparse_indices_(row_idx, 0);
    auto partition_id = example_partition_ids[example_idx];
    auto feature_id = sparse_quantized_values_(row_idx);
    gradient_stats_accumulator->AddStats(
        slot_id_, class_id_, partition_id, feature_id,
        GradientStats(example_first_order_gradients,
                      example_second_order_gradients, example_idx));
  }
}

void SparseQuantizedFeatureColumnHandler::GenerateFeatureSplitCandidates(
    const LearnerConfig& learner_config, const std::vector<int32>& roots,
    const std::vector<NodeStats>& root_stats,
    const FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>&
        gradient_stats_accumulator,
    std::vector<FeatureSplitCandidate>* split_candidates) const {
  // Evaluate split candidates for every root as each is a separate
  // logical partition over the examples.
  // Then for each root, we do both a forward left to right pass and a backward
  // right to left pass over the quantized feature buckets accumulating
  // gradients on one side and using the root aggregate gradients to get the
  // gradients for the other side. Split gains are evaluated for each pass at
  // every threshold and the best split is picked.
  split_candidates->clear();
  split_candidates->reserve(roots.size());
  for (size_t root_idx = 0; root_idx < roots.size(); ++root_idx) {
    // Get partition Id and root node stats.
    const int32 partition_id = roots[root_idx];
    const NodeStats& root_node_stats = root_stats[root_idx];

    // Forward pass with right default direction.
    GradientStats left_gradient_stats;
    GradientStats right_gradient_stats(root_node_stats.gradient_stats);
    FeatureSplitCandidate best_split_candidate(
        root_node_stats.weight_contribution.size());
    best_split_candidate.split_stats.gain =
        std::numeric_limits<float>::lowest();
    for (int bucket_id = 0; bucket_id < sparse_quantiles_.size(); ++bucket_id) {
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
      const float threshold = sparse_quantiles_(bucket_id);
      FeatureSplitCandidate split_candidate(
          slot_id_,
          CreateSparseSplitNodeDefaultRight(sparse_feature_column_, threshold),
          SplitStats(learner_config, root_node_stats, left_node_stats,
                     right_node_stats));
      if (split_candidate.split_stats.gain >
          best_split_candidate.split_stats.gain) {
        best_split_candidate = std::move(split_candidate);
      }
    }

    // Determine if we need a backward pass by checking if the residual gradient
    // after forward aggregation is almost the same as the aggregated gradient.
    // for the current root. This helps avoid unnecessary computation as well
    // as consistency due to floating point precision.
    if (!right_gradient_stats.IsAlmostZero()) {
      // Backward pass with left default direction.
      right_gradient_stats = GradientStats();
      left_gradient_stats = root_node_stats.gradient_stats;
      for (int bucket_id = sparse_quantiles_.size() - 1; bucket_id > 0;
           --bucket_id) {
        // Get gradient stats.
        auto gradient_stats = gradient_stats_accumulator.GetStats(
            slot_id_, class_id_, partition_id, bucket_id);
        if (gradient_stats.IsZero()) {
          continue;
        }

        // Update gradient stats.
        right_gradient_stats += gradient_stats;
        left_gradient_stats = root_node_stats.gradient_stats - gradient_stats;

        // Get node stats
        NodeStats left_node_stats(learner_config, left_gradient_stats);
        NodeStats right_node_stats(learner_config, right_gradient_stats);

        // Generate split candidate.
        const float threshold = sparse_quantiles_(bucket_id - 1);
        FeatureSplitCandidate split_candidate(
            slot_id_,
            CreateSparseSplitNodeDefaultLeft(sparse_feature_column_, threshold),
            SplitStats(learner_config, root_node_stats, left_node_stats,
                       right_node_stats));
        if (split_candidate.split_stats.gain >
            best_split_candidate.split_stats.gain) {
          best_split_candidate = std::move(split_candidate);
        }
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
