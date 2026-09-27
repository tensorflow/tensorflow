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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/matchers/similarity.h"
#include "xla/shape_util.h"

namespace xla {
namespace hlo_diff {
namespace {

template <typename Map>
std::vector<typename Map::key_type> GetSortedKeys(const Map& map) {
  std::vector<typename Map::key_type> keys;
  keys.reserve(map.size());
  for (const auto& [key, _] : map) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

bool AreConstantsValueEqual(const HloInstruction* left,
                            const HloInstruction* right) {
  if (left->opcode() != HloOpcode::kConstant ||
      right->opcode() != HloOpcode::kConstant) {
    return false;
  }
  auto* left_constant = Cast<HloConstantInstruction>(left);
  auto* right_constant = Cast<HloConstantInstruction>(right);
  if (!left_constant->HasLiteral() || !right_constant->HasLiteral()) {
    return false;
  }
  if (ShapeUtil::IsScalar(left->shape()) &&
      ShapeUtil::IsScalar(right->shape())) {
    auto left_val = left->literal().GetAsDouble({});
    auto right_val = right->literal().GetAsDouble({});
    if (left_val.has_value() && right_val.has_value()) {
      return *left_val == *right_val;
    }
  }
  // For non-scalars or non-numeric scalars, fallback to strict literal
  // equality.
  return left->literal() == right->literal();
}

// Match instructions with multiple match candidates using similarity measures.
void MatchInstructionsWithMultipleCandidates(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const absl::flat_hash_set<const HloInstructionNode*>& left_instructions,
    const absl::flat_hash_set<const HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type) {
  // Sort left_instructions to ensure deterministic processing order.
  std::vector<const HloInstructionNode*> sorted_left_instructions(
      left_instructions.begin(), left_instructions.end());
  std::sort(sorted_left_instructions.begin(), sorted_left_instructions.end(),
            [](const HloInstructionNode* a, const HloInstructionNode* b) {
              return a->unique_node_index < b->unique_node_index;
            });

  for (const HloInstructionNode* left : sorted_left_instructions) {
    double max_match_score = 0.0;
    std::vector<const HloInstructionNode*> right_candidates;
    for (const HloInstructionNode* right : right_instructions) {
      // For Constants, bypass if values differ.
      if (left->instruction->opcode() == HloOpcode::kConstant) {
        if (!AreConstantsValueEqual(left->instruction, right->instruction)) {
          continue;
        }
      }

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

// Map instructions with the same shape and metadata op name if its specified.
// This name is often unique within a computation and specified by the
// frameworks. Note that for XLA generated computations, the metadata is not
// consistently specified.
void MatchInstructionsWithSameMetadataOpName(
    const std::vector<const HloInstructionNode*>& left_instructions,
    const std::vector<const HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type) {
  for (const HloInstructionNode* left_instruction : left_instructions) {
    if (left_instruction->instruction->metadata().op_name().empty()) {
      continue;
    }
    int candidates_found = 0;
    const HloInstructionNode* candidate = nullptr;

    for (const HloInstructionNode* right_instruction : right_instructions) {
      bool same_shape = left_instruction->instruction->shape().ToString(
                            /*print_layout=*/false) ==
                        right_instruction->instruction->shape().ToString(
                            /*print_layout=*/false);
      bool same_op_name = left_instruction->instruction->metadata().op_name() ==
                          right_instruction->instruction->metadata().op_name();
      if (same_shape && same_op_name) {
        ++candidates_found;
        candidate = right_instruction;
      }
    }

    // Avoid matching instructions with multiple candidates.
    if (candidates_found == 1) {
      mappings.MapInstructionsIfAbsent(left_instruction, candidate,
                                       matcher_type);
    }
  }
}

// Match instructions by grouping them by shape.
// - Match unique instructions with the same shape.
// - Match instructions with multiple candidates using similarity measures.
void MatchInstructionsByShape(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const std::vector<const HloInstructionNode*>& left_instructions,
    const std::vector<const HloInstructionNode*>& right_instructions,
    HloGumgraphMappings& mappings, const MatcherType& matcher_type) {
  absl::flat_hash_map<std::string,
                      absl::flat_hash_set<const HloInstructionNode*>>
      left_instructions_by_shape;
  for (const HloInstructionNode* instruction : left_instructions) {
    if (!mappings.InstructionMapContainsLeft(instruction)) {
      left_instructions_by_shape[instruction->instruction->shape().ToString(
                                     /*print_layout=*/false)]
          .insert(instruction);
    }
  }

  absl::flat_hash_map<std::string,
                      absl::flat_hash_set<const HloInstructionNode*>>
      right_instructions_by_shape;
  for (const HloInstructionNode* instruction : right_instructions) {
    if (!mappings.InstructionMapContainsRight(instruction)) {
      right_instructions_by_shape[instruction->instruction->shape().ToString(
                                      /*print_layout=*/false)]
          .insert(instruction);
    }
  }

  std::vector<std::string> sorted_shapes =
      GetSortedKeys(left_instructions_by_shape);

  for (const auto& shape : sorted_shapes) {
    const auto& shape_left_instructions = left_instructions_by_shape.at(shape);
    if (auto it = right_instructions_by_shape.find(shape);
        it != right_instructions_by_shape.end()) {
      const absl::flat_hash_set<const HloInstructionNode*>&
          shape_right_instructions = it->second;
      // Match unique instructions with the same shape.
      if (shape_left_instructions.size() == 1 &&
          shape_right_instructions.size() == 1) {
        mappings.MapInstructionsIfAbsent(*shape_left_instructions.begin(),
                                         *shape_right_instructions.begin(),
                                         matcher_type);
      } else {
        // Match instructions with multiple candidates using
        // similarity measures.
        MatchInstructionsWithMultipleCandidates(
            left_graph, right_graph, shape_left_instructions,
            shape_right_instructions, mappings, matcher_type);
      }
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
    MapByPositionMode map_by_position_mode, double phase_zero_threshold) {
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

  // Phase 0: Direct mapping if only one instruction in each set, provided they
  // are similar enough.
  if (left_instructions.size() == 1 && right_instructions.size() == 1) {
    const HloInstructionNode* left = *left_instructions.begin();
    const HloInstructionNode* right = *right_instructions.begin();

    // Option B: For Constants, bypass if values differ and we are using
    // similarity (phase_zero_threshold > 0.0).
    bool bypass_phase_zero = false;
    if (left->instruction->opcode() == HloOpcode::kConstant &&
        phase_zero_threshold > 0.0) {
      if (!AreConstantsValueEqual(left->instruction, right->instruction)) {
        bypass_phase_zero = true;
      }
    }

    if (!bypass_phase_zero) {
      double similarity = PropertySimilarityFnForOpcode(
                              left->instruction->opcode())(left, right) +
                          AncestorSubGraphLcsSimilarity(
                              left, right, 20, 1, left_graph.GetNodeCount(),
                              right_graph.GetNodeCount());
      if (similarity >= phase_zero_threshold) {
        mappings.MapInstructionsIfAbsent(left, right, matcher_type);
        return;  // Early return after direct mapping.
      }
    }
  }

  // Phase 1: Match instructions with the same metadata op name.
  MatchInstructionsWithSameMetadataOpName(left_instructions, right_instructions,
                                          mappings, matcher_type);

  // Phase 2: Match instructions by shape.
  MatchInstructionsByShape(left_graph, right_graph, left_instructions,
                           right_instructions, mappings, matcher_type);

  // Phase 3: Map still unmatched instructions by position.
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
    MapByPositionMode map_by_position_mode, double phase_zero_threshold) {
  absl::flat_hash_map<HloOpcode,
                      std::pair<std::vector<const HloInstructionNode*>,
                                std::vector<const HloInstructionNode*>>>
      instructions_by_opcode;
  for (const HloInstructionNode* l : left_instructions) {
    instructions_by_opcode[l->instruction->opcode()].first.push_back(l);
  }
  for (const HloInstructionNode* r : right_instructions) {
    instructions_by_opcode[r->instruction->opcode()].second.push_back(r);
  }
  std::vector<HloOpcode> sorted_opcodes = GetSortedKeys(instructions_by_opcode);

  for (const auto& opcode : sorted_opcodes) {
    const auto& instructions = instructions_by_opcode.at(opcode);
    MatchSameOpcodeInstructions(left_graph, right_graph, instructions.first,
                                instructions.second, mappings, matcher_type,
                                map_by_position_mode, phase_zero_threshold);
  }
}

}  // namespace hlo_diff
}  // namespace xla
