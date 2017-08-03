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

#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

namespace {

// Creates a categorical Id split node without assigning children.
boosted_trees::trees::TreeNode CreateCategoricalIdNode(
    const int32 feature_column, const int32 id) {
  boosted_trees::trees::TreeNode split_node;
  auto* split = split_node.mutable_categorical_id_binary_split();
  split->set_feature_column(feature_column);
  split->set_feature_id(id);
  return split_node;
}

}  // namespace

void CategoricalFeatureColumnHandler::AggregateGradientStats(
    const std::vector<int32>& example_partition_ids,
    const Tensor& example_first_order_gradients,
    const Tensor& example_second_order_gradients,
    FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>*
        gradient_stats_accumulator) const {
  // Pass over all rows and aggregate gradient stats for each feature id.
  const int64 num_rows = indices_.dimension(0);
  for (int64 row_idx = 0; row_idx < num_rows; ++row_idx) {
    auto example_idx = indices_(row_idx, 0);
    auto feature_id = values_(row_idx);
    const GradientStats norm_gradient_stats(example_first_order_gradients,
                                            example_second_order_gradients,
                                            example_idx);
    auto partition_id = example_partition_ids[example_idx];
    gradient_stats_accumulator->AddStats(slot_id_, class_id_, partition_id,
                                         feature_id, norm_gradient_stats);
  }
}

void CategoricalFeatureColumnHandler::GenerateFeatureSplitCandidates(
    const LearnerConfig& learner_config, const std::vector<int32>& roots,
    const std::vector<NodeStats>& root_stats,
    const FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>&
        gradient_stats_accumulator,
    std::vector<FeatureSplitCandidate>* split_candidates) const {
  // Build a reverse lookup of partition id to root idx.
  std::unordered_map<int32, size_t> partition_id_to_root_idx;
  partition_id_to_root_idx.reserve(roots.size());
  for (size_t root_idx = 0; root_idx < roots.size(); ++root_idx) {
    partition_id_to_root_idx[roots[root_idx]] = root_idx;
  }

  // Initialize split candidates.
  split_candidates->clear();
  if (!roots.empty()) {
    FeatureSplitCandidate empty_candidate(
        root_stats[0].weight_contribution.size());
    split_candidates->resize(roots.size(), empty_candidate);
  }
  for (auto& split_candidate : *split_candidates) {
    split_candidate.split_stats.gain = std::numeric_limits<float>::lowest();
  }

  // Evaluate split candidates for every root as each is a separate
  // logical partition over the examples.
  // Then for each root, we evaluate every feature id as an equality split
  // and pick the highest split gain.
  for (const auto& entry :
       gradient_stats_accumulator.GetFeatureStats(slot_id_)) {
    DCHECK_EQ(entry.first.class_id, class_id_);

    // Get partition id and root node stats.
    const int32 partition_id = entry.first.partition_id;
    auto root_idx_it = partition_id_to_root_idx.find(partition_id);
    if (root_idx_it == partition_id_to_root_idx.end()) {
      // Inactive partition.
      continue;
    }
    size_t root_idx = root_idx_it->second;
    const NodeStats& root_node_stats = root_stats[root_idx];

    // Get gradient stats.
    const auto& left_gradient_stats = entry.second;
    auto right_gradient_stats =
        root_node_stats.gradient_stats - left_gradient_stats;

    // Get node stats.
    NodeStats left_node_stats(learner_config, left_gradient_stats);
    NodeStats right_node_stats(learner_config, right_gradient_stats);

    // Generate split candidate and update best split candidate for the
    // current root if needed.
    FeatureSplitCandidate split_candidate(
        slot_id_,
        CreateCategoricalIdNode(feature_column_, entry.first.feature_id),
        SplitStats(learner_config, root_node_stats, left_node_stats,
                   right_node_stats));
    FeatureSplitCandidate& best_split_candidate = (*split_candidates)[root_idx];
    if (TF_PREDICT_FALSE(best_split_candidate.tree_node.node_case() ==
                         boosted_trees::trees::TreeNode::NODE_NOT_SET)) {
      // Always replace candidates with no node set.
      best_split_candidate = std::move(split_candidate);
    } else if (TF_PREDICT_FALSE(split_candidate.split_stats.gain ==
                                best_split_candidate.split_stats.gain)) {
      // Tie break on feature id.
      auto best_split_feature_id =
          best_split_candidate.tree_node.categorical_id_binary_split()
              .feature_id();
      if (entry.first.feature_id < best_split_feature_id) {
        best_split_candidate = std::move(split_candidate);
      }
    } else if (split_candidate.split_stats.gain >
               best_split_candidate.split_stats.gain) {
      best_split_candidate = std::move(split_candidate);
    }
  }
}

}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow
