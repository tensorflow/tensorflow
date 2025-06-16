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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_BIPARTITE_MATCHING_H_
#define XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_BIPARTITE_MATCHING_H_

#include <vector>

#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla {
namespace hlo_diff {

// Find optimal matches between the left and right instruction set.
// The goal is to establish a mapping between corresponding instructions from
// the 'left_instructions' and 'right_instructions' sets, all of the same type.
// The instructions are first matched by node properties like shape, metadata,
// etc. If 'map_by_position' is set to true, the left unmatched instructions
// will try to be matched by position one by one if they share the same size.
void MatchSameTypeInstructions(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const std::vector<const HloInstructionNode*>& left_instructions,
    const std::vector<const HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type,
    bool map_by_position = false);

// Find optimal matches between the left and right instruction set.
// Sort the instructions by opcode and call MatchSameTypeInstructions for each
// opcode.
void MatchLeafInstructions(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const std::vector<HloInstructionNode*>& left_instructions,
    const std::vector<HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type,
    bool map_by_position = false);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_BIPARTITE_MATCHING_H_
