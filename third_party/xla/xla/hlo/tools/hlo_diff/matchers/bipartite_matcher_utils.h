/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_BIPARTITE_MATCHER_UTILS_H_
#define XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_BIPARTITE_MATCHER_UTILS_H_

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla {
namespace hlo_diff {

// Find optimal matches between the left and right instruction set.
// The goal is to establish a mapping between corresponding instructions from
// the 'left_instructions' and 'right_instructions' sets, all of the same type.
void MatchSameTypeInstructions(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const absl::flat_hash_set<const HloInstructionNode*>& left_instructions,
    const absl::flat_hash_set<const HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_BIPARTITE_MATCHER_UTILS_H_
