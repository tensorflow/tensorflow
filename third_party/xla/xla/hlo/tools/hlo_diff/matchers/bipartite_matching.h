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

// Enums to define the mapping by position behavior
enum class MapByPositionMode {
  kNever,           // Never map by position
  kAlways,          // Always attempt to map by position
  kOnlyIfSameSize,  // Map by position only if sets are of the same size
};

// Find optimal matches between the left and right instruction lists.
// The goal is to establish a mapping between corresponding instructions from
// the 'left_instructions' and 'right_instructions' lists, all of the same type.
// The instructions are first matched by node properties like shape, metadata,
// etc. If 'map_by_position' is set to true, the left unmatched instructions
// will try to be matched by position one by one.
void MatchSameOpcodeInstructions(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const std::vector<const HloInstructionNode*>& left_instructions,
    const std::vector<const HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type,
    MapByPositionMode map_by_position = MapByPositionMode::kNever);

// Find optimal matches between the left and right instruction lists.
// Sort the instructions by opcode and call MatchSameOpcodeInstructions for each
// opcode.
void MatchInstructions(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const std::vector<HloInstructionNode*>& left_instructions,
    const std::vector<HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type,
    MapByPositionMode map_by_position = MapByPositionMode::kNever);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_BIPARTITE_MATCHING_H_
