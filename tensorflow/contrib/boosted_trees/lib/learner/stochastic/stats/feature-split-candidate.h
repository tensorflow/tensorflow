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
//
// =============================================================================
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_STATS_FEATURE_SPLIT_CANDIDATE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_STATS_FEATURE_SPLIT_CANDIDATE_H_

#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/stats/split-stats.h"
#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

// FeatureSplitCandidate holds the split candidate node along with the stats.
struct FeatureSplitCandidate {
  // Empty split candidate.
  explicit FeatureSplitCandidate(const int output_length)
      : feature_column_slot_id(kInvalidFeatureColumnSlot),
        split_stats(output_length) {}

  // Feature binary split candidate.
  FeatureSplitCandidate(const int64 fc_slot_id,
                        const boosted_trees::trees::TreeNode& node,
                        const SplitStats& stats)
      : feature_column_slot_id(fc_slot_id),
        tree_node(node),
        split_stats(stats) {}

  // Globally unique slot Id identifying the feature column
  // used in this split candidates.
  int64 feature_column_slot_id;

  // Tree node for the candidate split.
  boosted_trees::trees::TreeNode tree_node;

  // Split stats.
  SplitStats split_stats;

  // Invalid feature column slot reserved value.
  static constexpr int64 kInvalidFeatureColumnSlot = -1;
};

}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_STATS_FEATURE_SPLIT_CANDIDATE_H_
