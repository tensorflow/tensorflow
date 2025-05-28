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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_EXACT_SUBGRAPH_MATCHER_H_
#define XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_EXACT_SUBGRAPH_MATCHER_H_

#include "absl/log/die_if_null.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/matchers/gumgraph_matcher.h"

namespace xla {
namespace hlo_diff {

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

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_EXACT_SUBGRAPH_MATCHER_H_
