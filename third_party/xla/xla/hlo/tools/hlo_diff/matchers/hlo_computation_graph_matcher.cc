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
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/matchers/bipartite_matching.h"
#include "xla/service/call_graph.h"

namespace xla {
namespace hlo_diff {
namespace {

// Match parameter instructions between the left and right computations.
void MatchComputationParams(const HloGumgraph& left_graph,
                            const HloGumgraph& right_graph,
                            const CallGraphNode& left_computation,
                            const CallGraphNode& right_computation,
                            HloGumgraphMappings& mappings,
                            const MatcherType& matcher_type) {
  std::vector<const HloInstructionNode*> left_params, right_params;
  for (const HloInstruction* param :
       left_computation.computation()->parameter_instructions()) {
    left_params.push_back(left_graph.GetNode(param));
  }
  for (const HloInstruction* param :
       right_computation.computation()->parameter_instructions()) {
    right_params.push_back(right_graph.GetNode(param));
  }
  bool map_by_position = !left_computation.computation()->IsEntryComputation();
  MatchSameTypeInstructions(left_graph, right_graph, left_params, right_params,
                            mappings, matcher_type, map_by_position);
}

// Match constant instructions between the left and right computations.
void MatchComputationConstants(const HloGumgraph& left_graph,
                               const HloGumgraph& right_graph,
                               const CallGraphNode& left_computation,
                               const CallGraphNode& right_computation,
                               HloGumgraphMappings& mappings,
                               const MatcherType& matcher_type) {
  std::vector<const HloInstructionNode*> left_constants, right_constants;
  for (const HloInstruction* instruction :
       left_computation.computation()->instructions()) {
    if (instruction->IsConstant()) {
      left_constants.push_back(left_graph.GetNode(instruction));
    }
  }
  for (const HloInstruction* instruction :
       right_computation.computation()->instructions()) {
    if (instruction->IsConstant()) {
      right_constants.push_back(right_graph.GetNode(instruction));
    }
  }
  bool map_by_position = !left_computation.computation()->IsEntryComputation();
  MatchSameTypeInstructions(left_graph, right_graph, left_constants,
                            right_constants, mappings, matcher_type,
                            map_by_position);
}

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
  auto it = mappings.left_to_right_computation_map.left.find(&left_computation);
  if (it == mappings.left_to_right_computation_map.left.end()) {
    return;
  }

  MatchCallSites(left, right, left_computation, right_computation, mappings);

  // If the two computations are exact matches, we can match all
  // instructions in the two computations.
  if (it->info.computation_match_type == ComputationMatchType::kExact) {
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
    // inputs (parameters, constants) and root instruction of the two
    // computation graph.
    MatchComputationParams(left, right, left_computation, right_computation,
                           mappings,
                           MatcherType::kComputationGraphExactSignatureMatcher);
    MatchComputationConstants(
        left, right, left_computation, right_computation, mappings,
        MatcherType::kComputationGraphExactSignatureMatcher);

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
