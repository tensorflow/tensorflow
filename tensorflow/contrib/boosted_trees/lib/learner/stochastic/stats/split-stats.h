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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_STATS_SPLIT_STATS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_STATS_SPLIT_STATS_H_

#include <string>

#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/stats/node-stats.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

// FeatureSplitCandidate holds the split candidate node along with the stats.
struct SplitStats {
  // Initialize with 0 stats.
  explicit SplitStats(const int output_length)
      : root_node_stats(output_length),
        left_node_stats(output_length),
        right_node_stats(output_length),
        gain(0) {}

  // Feature unary split candidate, we don't apply tree complexity
  // regularization as no new nodes are being added with this candidate.
  SplitStats(const LearnerConfig& learner_config, const NodeStats& root_stats)
      : root_node_stats(root_stats),
        left_node_stats(root_stats.weight_contribution.size()),
        right_node_stats(root_stats.weight_contribution.size()),
        gain(0) {}

  // Feature binary split candidate, we apply tree complexity regularization
  // over the split gain to trade-off adding new nodes with loss reduction.
  SplitStats(const LearnerConfig& learner_config, const NodeStats& root_stats,
             const NodeStats& left_stats, const NodeStats& right_stats)
      : root_node_stats(root_stats),
        left_node_stats(left_stats),
        right_node_stats(right_stats),
        gain(left_stats.gain + right_stats.gain - root_stats.gain -
             learner_config.regularization().tree_complexity()) {}

  // Root Stats.
  NodeStats root_node_stats;

  // Children stats.
  NodeStats left_node_stats;
  NodeStats right_node_stats;

  // Split gain.
  float gain;

  string DebugString() const {
    return "Root = " + root_node_stats.DebugString() +
           "\nLeft = " + left_node_stats.DebugString() +
           "\nRight = " + right_node_stats.DebugString() +
           "\nGain = " + std::to_string(gain);
  }
};

// Helper macro to check split stats approximate equality.
#define EXPECT_SPLIT_STATS_EQ(val1, val2)                             \
  EXPECT_NODE_STATS_EQ(val1.root_node_stats, val2.root_node_stats);   \
  EXPECT_NODE_STATS_EQ(val1.left_node_stats, val2.left_node_stats);   \
  EXPECT_NODE_STATS_EQ(val1.right_node_stats, val2.right_node_stats); \
  EXPECT_FLOAT_EQ(val1.gain, val2.gain);

}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_STATS_SPLIT_STATS_H_
