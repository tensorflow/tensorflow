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

#include "xla/hlo/tools/hlo_diff/matchers/bipartite_matching.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/matchers/similarity.h"

namespace xla {
namespace hlo_diff {
namespace {

// Match instructions with multiple match candidates using similarity measures.
void MatchInstructionsWithMultipleCandidates(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const std::vector<const HloInstructionNode*>& left_instructions,
    const std::vector<const HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type) {
  for (const HloInstructionNode* left : left_instructions) {
    double max_match_score = 0.0;
    std::vector<const HloInstructionNode*> right_candidates;
    for (const HloInstructionNode* right : right_instructions) {
      double similarity = PropertySimilarityFnForOpcode(
                              left->instruction->opcode())(left, right) +
                          AncestorSubGraphLcsSimilarity(
                              left, right, 20, 1, left_graph.GetNodeCount(),
                              right_graph.GetNodeCount());
      if (similarity > max_match_score) {
        max_match_score = similarity;
        right_candidates.clear();
        right_candidates.push_back(right);
      } else if (similarity == max_match_score) {
        right_candidates.push_back(right);
      }
    }

    // Avoid matching instructions with multiple candidates.
    if (right_candidates.size() == 1) {
      mappings.MapInstructionsIfAbsent(left, right_candidates[0], matcher_type);
    }
  }
}

void MatchInstructionsByPosition(
    const std::vector<const HloInstructionNode*>& left_instructions,
    const std::vector<const HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type,
    bool only_if_same_size) {
  std::vector<const HloInstructionNode*> unmatched_left_instructions,
      unmatched_right_instructions;
  for (const HloInstructionNode* left_instruction : left_instructions) {
    if (!mappings.InstructionMapContainsLeft(left_instruction)) {
      unmatched_left_instructions.push_back(left_instruction);
    }
  }
  for (const HloInstructionNode* right_instruction : right_instructions) {
    if (!mappings.InstructionMapContainsRight(right_instruction)) {
      unmatched_right_instructions.push_back(right_instruction);
    }
  }
  // Map by position regardless of size if only_if_same_size is false,
  // or if sizes are the same when only_if_same_size is true.
  if (only_if_same_size && unmatched_left_instructions.size() !=
                               unmatched_right_instructions.size()) {
    return;
  }
  for (int i = 0; i < std::min(unmatched_left_instructions.size(),
                               unmatched_right_instructions.size());
       ++i) {
    mappings.MapInstructionsIfAbsent(unmatched_left_instructions[i],
                                     unmatched_right_instructions[i],
                                     matcher_type);
  }
}

}  // namespace

void MatchSameOpcodeInstructions(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const std::vector<const HloInstructionNode*>& left_instructions,
    const std::vector<const HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type,
    MapByPositionMode map_by_position_mode) {
  if (left_instructions.empty() || right_instructions.empty()) {
    return;
  }
  // Check that all instructions have the same opcode.
  const HloOpcode left_opcode =
      (*left_instructions.begin())->instruction->opcode();
  for (const HloInstructionNode* instruction : left_instructions) {
    CHECK(instruction->instruction->opcode() == left_opcode)
        << "All instructions must have the same opcode.";
  }
  for (const HloInstructionNode* instruction : right_instructions) {
    CHECK(instruction->instruction->opcode() == left_opcode)
        << "All instructions must have the same opcode.";
  }

  // Phase 0: Direct mapping if only one instruction in each set.
  if (left_instructions.size() == 1 && right_instructions.size() == 1) {
    mappings.MapInstructionsIfAbsent(*left_instructions.begin(),
                                     *right_instructions.begin(), matcher_type);
    return;  // Early return after direct mapping.
  }

  MatchInstructionsWithMultipleCandidates(left_graph, right_graph,
                                          left_instructions, right_instructions,
                                          mappings, matcher_type);

  if (map_by_position_mode != MapByPositionMode::kNever) {
    MatchInstructionsByPosition(
        left_instructions, right_instructions, mappings, matcher_type,
        map_by_position_mode == MapByPositionMode::kOnlyIfSameSize);
  }
}

// Find optimal matches between the left and right instruction lists.
// The goal is to establish a mapping between corresponding instructions from
// the 'left_instructions' and 'right_instructions' lists. These lists are
// derived from the two computations being mapped, or two parents being mapped.
void MatchInstructions(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const std::vector<HloInstructionNode*>& left_instructions,
    const std::vector<HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type,
    MapByPositionMode map_by_position_mode) {
  absl::flat_hash_map<const HloOpcode,
                      std::pair<std::vector<const HloInstructionNode*>,
                                std::vector<const HloInstructionNode*>>>
      instructions_by_opcode;
  for (const HloInstructionNode* l : left_instructions) {
    instructions_by_opcode[l->instruction->opcode()].first.push_back(l);
  }
  for (const HloInstructionNode* r : right_instructions) {
    instructions_by_opcode[r->instruction->opcode()].second.push_back(r);
  }
  for (const auto& [opcode, instructions] : instructions_by_opcode) {
    MatchSameOpcodeInstructions(left_graph, right_graph, instructions.first,
                                instructions.second, mappings, matcher_type,
                                map_by_position_mode);
  }
}

}  // namespace hlo_diff
}  // namespace xla
