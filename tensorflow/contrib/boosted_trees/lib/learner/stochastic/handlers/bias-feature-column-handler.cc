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

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

void BiasFeatureColumnHandler::AggregateGradientStats(
    const std::vector<int32>& example_partition_ids,
    const Tensor& example_first_order_gradients,
    const Tensor& example_second_order_gradients,
    FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>*
        gradient_stats_accumulator) const {
  // Pass over all examples and aggregate gradient stats for each sub-root.
  for (int64 example_idx = 0; example_idx < batch_size_; ++example_idx) {
    auto partition_id = example_partition_ids[example_idx];
    gradient_stats_accumulator->AddStats(
        slot_id_, class_id_, partition_id, kBiasFeatureId,
        GradientStats(example_first_order_gradients,
                      example_second_order_gradients, example_idx));
  }
}

void BiasFeatureColumnHandler::GenerateFeatureSplitCandidates(
    const LearnerConfig& learner_config, const std::vector<int32>& roots,
    const std::vector<NodeStats>& root_stats,
    const FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>&
        gradient_stats_accumulator,
    std::vector<FeatureSplitCandidate>* split_candidates) const {
  split_candidates->clear();
  split_candidates->reserve(roots.size());
  boosted_trees::trees::TreeNode tree_node;
  for (size_t root_idx = 0; root_idx < roots.size(); ++root_idx) {
    const NodeStats& root_node_stats = root_stats[root_idx];
    tree_node.Clear();
    root_node_stats.FillLeaf(class_id_, tree_node.mutable_leaf());
    split_candidates->emplace_back(slot_id_, tree_node,
                                   SplitStats(learner_config, root_node_stats));
  }
}

}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow
