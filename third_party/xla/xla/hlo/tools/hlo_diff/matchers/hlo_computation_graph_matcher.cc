// Copyright 2025 The OpenXLA Authors
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

#include "xla/hlo/tools/hlo_diff/matchers/hlo_computation_graph_matcher.h"

#include <vector>

#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/matchers/bipartite_matching.h"
#include "xla/service/call_graph.h"

namespace xla {
namespace hlo_diff {
namespace {

// Match the call site instruction and it's operands for a matched left and
// right computation.
void MatchCallSites(const HloGumgraph& left, const HloGumgraph& right,
                    const CallGraphNode& left_computation,
                    const CallGraphNode& right_computation,
                    HloGumgraphMappings& mappings) {
  // Only match call sites if both computations are called from exactly one call
  // site. In case a computation is called from multiple call sites, we cannot
  // disambiguate between the call sites. The subsequent matchers should be able
  // to find the matches between the call sites in such cases.
  if (left_computation.caller_callsites().size() != 1 ||
      right_computation.caller_callsites().size() != 1) {
    return;
  }

  const CallSite& left_call_site = *left_computation.caller_callsites().begin();
  const CallSite& right_call_site =
      *right_computation.caller_callsites().begin();

  // Match the call site instruction.
  mappings.MapInstructionsIfAbsent(
      left.GetNode(left_call_site.instruction()),
      right.GetNode(right_call_site.instruction()),
      MatcherType::kComputationGraphExactSignatureMatcher);
}

}  // namespace

void MatchComputationGraphs(const HloGumgraph& left, const HloGumgraph& right,
                            const CallGraphNode& left_computation,
                            const CallGraphNode& right_computation,
                            HloGumgraphMappings& mappings) {
  if (!mappings.left_to_right_computation_map.ContainsLeft(&left_computation)) {
    return;
  }

  MatchCallSites(left, right, left_computation, right_computation, mappings);

  // If the two computations are exact matches, we can match all
  // instructions in the two computations.
  if (mappings.left_to_right_computation_map.GetPropsByLeft(&left_computation)
          ->computation_match_type == ComputationMatchType::kExact) {
    auto left_instructions =
        left_computation.computation()->MakeInstructionPostOrder();
    auto right_instructions =
        right_computation.computation()->MakeInstructionPostOrder();
    if (left_instructions.size() != right_instructions.size()) {
      LOG(WARNING) << "Computation size mismatch: Left computation: "
                   << left_computation.computation()->name() << " has "
                   << left_instructions.size()
                   << " instructions and right computation: "
                   << right_computation.computation()->name() << " has "
                   << right_instructions.size() << " instructions";
      return;
    }

    for (int i = 0; i < left_instructions.size(); ++i) {
      mappings.MapInstructionsIfAbsent(
          left.GetNode(left_instructions[i]),
          right.GetNode(right_instructions[i]),
          MatcherType::kComputationGraphExactFingerprintMatcher);
    }
  } else {
    // If the two computations are signature matches, we can match the
    // leaves - (parameters, constants, etc) and root instruction of the two
    // computation graph.
    std::vector<HloInstructionNode*> left_leafs, right_leafs;
    for (const HloInstruction* instruction :
         left_computation.computation()->MakeInstructionPostOrder()) {
      HloInstructionNode* left_node = left.GetNode(instruction);
      if (left_node->instruction->opcode() == HloOpcode::kParameter ||
          left_node->children.empty()) {
        left_leafs.push_back(left_node);
      }
    }
    for (const HloInstruction* instruction :
         right_computation.computation()->MakeInstructionPostOrder()) {
      HloInstructionNode* right_node = right.GetNode(instruction);
      if (right_node->instruction->opcode() == HloOpcode::kParameter ||
          right_node->children.empty()) {
        right_leafs.push_back(right_node);
      }
    }

    MapByPositionMode map_by_position_mode =
        left_computation.computation()->IsEntryComputation()
            ? MapByPositionMode::kNever
            : MapByPositionMode::kOnlyIfSameSize;
    MatchInstructions(left, right, left_leafs, right_leafs, mappings,
                      MatcherType::kComputationGraphExactSignatureMatcher,
                      map_by_position_mode);

    if (left_computation.computation()->root_instruction()->opcode() ==
        right_computation.computation()->root_instruction()->opcode()) {
      mappings.MapInstructionsIfAbsent(
          left.GetNode(left_computation.computation()->root_instruction()),
          right.GetNode(right_computation.computation()->root_instruction()),
          MatcherType::kComputationGraphExactSignatureMatcher);
    }
  }
}
}  // namespace hlo_diff
}  // namespace xla
