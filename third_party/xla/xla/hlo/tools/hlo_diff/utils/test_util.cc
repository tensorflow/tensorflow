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

#include "xla/hlo/tools/hlo_diff/utils/test_util.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla {
namespace hlo_diff {

const HloInstructionNode* GetNodeByName(const HloGumgraph& graph,
                                        absl::string_view name) {
  for (const auto* node : graph.AllNodes()) {
    if (!node->is_root && node->instruction->name() == name) {
      return node;
    }
  }
  return nullptr;
}

void OverwriteMapInstructions(const HloInstructionNode* left,
                              const HloInstructionNode* right,
                              HloGumgraphMappings& mappings,
                              bool position_unchanged) {
  ASSERT_NE(left, nullptr);
  ASSERT_NE(right, nullptr);
  if (auto it = mappings.left_to_right_instruction_map.left.find(left);
      it != mappings.left_to_right_instruction_map.left.end()) {
    mappings.left_to_right_instruction_map.left.erase(it);
  }

  if (auto it = mappings.left_to_right_instruction_map.right.find(right);
      it != mappings.left_to_right_instruction_map.right.end()) {
    mappings.left_to_right_instruction_map.right.erase(it);
  }

  mappings.left_to_right_instruction_map.insert(
      InstructionPair(left, right, {.matcher_type = MatcherType::kManual}));
  if (position_unchanged) {
    mappings.left_to_right_instruction_map.left.find(left)->info.unchanged =
        true;
  }
}

void MatchAllNodesByName(const HloGumgraph& left, const HloGumgraph& right,
                         HloGumgraphMappings& mappings) {
  for (const auto* left_node : left.AllNodes()) {
    if (left_node->is_root) {
      continue;
    }
    const HloInstructionNode* right_node = nullptr;
    for (const auto* node : right.AllNodes()) {
      if (!node->is_root &&
          node->instruction->name() == left_node->instruction->name()) {
        right_node = node;
        break;
      }
    }
    if (right_node != nullptr) {
      mappings.MapInstructionsIfAbsent(left_node, right_node,
                                       MatcherType::kManual);
    }
  }
}

absl::flat_hash_map<std::string, std::string> ExtractMappedInstructionNames(
    const HloGumgraphMappings& mappings) {
  absl::flat_hash_map<std::string, std::string> mapped_nodes;
  for (auto it = mappings.left_to_right_instruction_map.begin();
       it != mappings.left_to_right_instruction_map.end(); ++it) {
    absl::string_view left_name =
        it->left->is_root ? "root_L" : it->left->instruction->name();
    absl::string_view right_name =
        it->right->is_root ? "root_R" : it->right->instruction->name();
    mapped_nodes[left_name] = right_name;
  }

  return mapped_nodes;
}

absl::flat_hash_map<std::string, std::string> ExtractMappedComputationNames(
    const HloGumgraphMappings& mappings) {
  absl::flat_hash_map<std::string, std::string> mapped_computations;
  for (auto it = mappings.left_to_right_computation_map.left.begin();
       it != mappings.left_to_right_computation_map.left.end(); ++it) {
    mapped_computations[it->first->computation()->name()] =
        it->second->computation()->name();
  }
  return mapped_computations;
}

absl::flat_hash_map<std::string, ComputationMatchType>
ExtractComputationMatchType(const HloGumgraphMappings& mappings) {
  absl::flat_hash_map<std::string, ComputationMatchType> computation_match_type;
  for (auto it = mappings.left_to_right_computation_map.left.begin();
       it != mappings.left_to_right_computation_map.left.end(); ++it) {
    computation_match_type[it->first->computation()->name()] =
        it->info.computation_match_type;
  }
  return computation_match_type;
}

absl::StatusOr<HloInstruction*> GetInstructionByName(HloModule& module,
                                                     absl::string_view name) {
  for (HloComputation* computation : module.computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->name() == name) {
        return instruction;
      }
    }
  }
  return absl::InvalidArgumentError("instruction not found");
}

}  // namespace hlo_diff
}  // namespace xla
