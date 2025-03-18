/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_HLO_GUMGRAPH_MATCHER_H_
#define XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_HLO_GUMGRAPH_MATCHER_H_

#include "absl/log/die_if_null.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla {
namespace hlo_diff {

// Options allowing configuration of the instruction matching algorithm.
struct MatchOptions {
  bool use_top_down_matcher = true;
};

// Base class for all node matchers. Each matcher implements a unique algorithm
// to match nodes between two HLO graphs. The base class standardizes input and
// output types, ensuring seamless integration and compatibility within any
// matcher sequence.
class HloGumgraphMatcher {
 public:
  virtual ~HloGumgraphMatcher() = default;
  virtual void Match(HloGumgraphMappings& mappings) const = 0;

 protected:
  explicit HloGumgraphMatcher(MatcherType type) : type_(type) {}
  const MatcherType type_;
};

// Matcher that matches identical subgraphs starting with the tallest.
class GreedySubGraphExactMatcher : public HloGumgraphMatcher {
 public:
  GreedySubGraphExactMatcher(const HloGumgraph* left, const HloGumgraph* right)
      : HloGumgraphMatcher(MatcherType::kGreedySubGraphExactMatcher),
        left_(*ABSL_DIE_IF_NULL(left)),
        right_(*ABSL_DIE_IF_NULL(right)) {}
  void Match(HloGumgraphMappings& mappings) const override;

 private:
  const HloGumgraph& left_;
  const HloGumgraph& right_;
};

// Matcher that matches nodes bottom up by dice similarity. For each left node,
// mappings of the already matched descendants are considered as seeds, from
// which we traverse back the graph to find nodes with same opcode as
// candidates. The candidate with the highest similarity is chosen as the match.
// Nodes mapped by this matcher in earlier iterations are also considered as
// seeds for later iterations.
//
// Seeds: Number of nodes to traverse to find seeds are limited.
// Candidates: Number of nodes to traverse to find candidates are limited.
// Dice similarity: Number of nodes to traverse in subgraph are limited.
class GreedyLimitedCandidatesBottomUpMatcher : public HloGumgraphMatcher {
 public:
  GreedyLimitedCandidatesBottomUpMatcher(const HloGumgraph* left,
                                         const HloGumgraph* right,
                                         double min_similarity = 1.2,
                                         int max_dice_subgraph_size = 200,
                                         int max_ancestors_to_consider = 100,
                                         int right_seeds_traversal_limit = 40,
                                         int candidate_traversal_limit = 200)
      : HloGumgraphMatcher(
            MatcherType::kGreedyLimitedCandidatesBottomUpMatcher),
        left_(*ABSL_DIE_IF_NULL(left)),
        right_(*ABSL_DIE_IF_NULL(right)),
        min_similarity_(min_similarity),
        max_dice_subgraph_size_(max_dice_subgraph_size),
        max_ancestors_to_consider_(max_ancestors_to_consider),
        right_seeds_traversal_limit_(right_seeds_traversal_limit),
        candidate_traversal_limit_(candidate_traversal_limit) {}
  void Match(HloGumgraphMappings& mappings) const override;

 private:
  const HloGumgraph& left_;
  const HloGumgraph& right_;

  // Minimum similarity to consider a match.
  const double min_similarity_;

  // Maximum size of the subgraph to consider when calculating dice similarity.
  const int max_dice_subgraph_size_;

  // Maximum number of ancestors to consider when calculating ancestor
  // similarity.
  const int max_ancestors_to_consider_;

  // Maximum number of nodes to traverse to find right seeds.
  const int right_seeds_traversal_limit_;

  // Maximum number of nodes to traverse from seeds. Nodes with the same
  // opcode are considered as candidates.
  const int candidate_traversal_limit_;
};

// Matcher that matches nodes top down by same type sequence along the path.
class GreedyTopDownMatcher : public HloGumgraphMatcher {
 public:
  GreedyTopDownMatcher(const HloGumgraph* left, const HloGumgraph* right)
      : HloGumgraphMatcher(MatcherType::kGreedyTopDownMatcher),
        left_(*ABSL_DIE_IF_NULL(left)),
        right_(*ABSL_DIE_IF_NULL(right)) {}
  void Match(HloGumgraphMappings& mappings) const override;

 private:
  const HloGumgraph& left_;
  const HloGumgraph& right_;
};

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_HLO_GUMGRAPH_MATCHER_H_
