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

#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_bfs.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/proto/diff_result.pb.h"

namespace xla {
namespace hlo_diff {
namespace {

bool IsChangedInstruction(const HloInstructionNode* left_node,
                          const HloInstructionNode* right_node) {
  return left_node->props.canonical_fingerprint !=
         right_node->props.canonical_fingerprint;
}

}  // namespace

std::unique_ptr<const DiffResult> ConstructDiffResult(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const HloGumgraphMappings& mappings) {
  LOG(INFO) << "Constructing diff result";
  const std::vector<const HloInstructionNode*> left_all_nodes =
      GetAllNodesInBfsOrder(left_graph.GetRoot(),
                            BfsTraversalDirection::kForward,
                            left_graph.GetNodeCount());
  const std::vector<const HloInstructionNode*> right_all_nodes =
      GetAllNodesInBfsOrder(right_graph.GetRoot(),
                            BfsTraversalDirection::kForward,
                            right_graph.GetNodeCount());
  auto diff_result = std::make_unique<DiffResult>();
  for (const HloInstructionNode* left_node : left_all_nodes) {
    if (left_node->is_root) {
      continue;
    }

    diff_result->node_props_left.insert(
        {left_node->instruction, left_node->props});

    // The node is unmatched
    if (!mappings.InstructionMapContainsLeft(left_node)) {
      diff_result->left_module_unmatched_instructions.insert(
          left_node->instruction);
      continue;
    }

    // The node is matched
    const HloInstructionNode* right_node =
        mappings.left_to_right_instruction_map.left.find(left_node)->second;
    const HloInstructionNodeMappingProps& mapping_props =
        mappings.left_to_right_instruction_map.left.find(left_node)->info;

    // Fill in matcher debug info.
    diff_result->map_by[std::make_pair(left_node->instruction,
                                       right_node->instruction)] =
        mapping_props.matcher_type;
    diff_result->matcher_debug_info[std::make_pair(left_node->instruction,
                                                   right_node->instruction)] =
        mapping_props.matcher_debug_info;

    if (IsChangedInstruction(left_node, right_node)) {
      diff_result->changed_instructions[left_node->instruction] =
          right_node->instruction;
      continue;
    }
    // If node position is unchanged, add to unchanged instructions.
    if (mapping_props.unchanged) {
      diff_result->unchanged_instructions[left_node->instruction] =
          right_node->instruction;
      continue;
    }
    // TODO(b/369851244): Add moved instructions to diff result.
    diff_result->unchanged_instructions[left_node->instruction] =
        right_node->instruction;
  }

  for (const HloInstructionNode* right_node : right_all_nodes) {
    if (right_node->is_root) {
      continue;
    }
    diff_result->node_props_right.insert(
        {right_node->instruction, right_node->props});
    if (!mappings.InstructionMapContainsRight(right_node)) {
      diff_result->right_module_unmatched_instructions.insert(
          right_node->instruction);
    }
  }

  return diff_result;
}

DiffResultProto DiffResult::ToProto() const {
  DiffResultProto proto;
  for (const auto& [left_instruction, right_instruction] :
       unchanged_instructions) {
    MatchedInstructionPairProto* pair = proto.add_unchanged_instructions();
    pair->set_left(std::string(left_instruction->name()));
    pair->set_right(std::string(right_instruction->name()));
  }
  for (const auto& [left_instruction, right_instruction] :
       changed_instructions) {
    MatchedInstructionPairProto* pair = proto.add_changed_instructions();
    pair->set_left(std::string(left_instruction->name()));
    pair->set_right(std::string(right_instruction->name()));
  }
  for (const HloInstruction* instruction : left_module_unmatched_instructions) {
    proto.add_left_unmatched_instructions(std::string(instruction->name()));
  }
  for (const HloInstruction* instruction :
       right_module_unmatched_instructions) {
    proto.add_right_unmatched_instructions(std::string(instruction->name()));
  }
  return proto;
}

DiffResult DiffResult::FromProto(const DiffResultProto& proto,
                                 const HloModule& left_module,
                                 const HloModule& right_module) {
  // Get instructions from modules.
  absl::flat_hash_map<std::string, const HloInstruction*>
      left_instructions_by_name;
  for (const HloComputation* computation : left_module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      left_instructions_by_name[instruction->name()] = instruction;
    }
  }
  absl::flat_hash_map<std::string, const HloInstruction*>
      right_instructions_by_name;
  for (const HloComputation* computation : right_module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      right_instructions_by_name[instruction->name()] = instruction;
    }
  }

  DiffResult diff_result;
  for (const MatchedInstructionPairProto& pair :
       proto.unchanged_instructions()) {
    diff_result.unchanged_instructions[left_instructions_by_name[pair.left()]] =
        right_instructions_by_name[pair.right()];
  }
  for (const MatchedInstructionPairProto& pair : proto.changed_instructions()) {
    diff_result.changed_instructions[left_instructions_by_name[pair.left()]] =
        right_instructions_by_name[pair.right()];
  }
  for (const std::string& name : proto.left_unmatched_instructions()) {
    diff_result.left_module_unmatched_instructions.insert(
        left_instructions_by_name[name]);
  }
  for (const std::string& name : proto.right_unmatched_instructions()) {
    diff_result.right_module_unmatched_instructions.insert(
        right_instructions_by_name[name]);
  }

  return diff_result;
}

void LogDiffResult(const DiffResult& diff_result) {
  LOG(INFO) << "Unmatched instructions in the left module: "
            << diff_result.left_module_unmatched_instructions.size();
  LOG(INFO) << "Unmatched instructions in the right module: "
            << diff_result.right_module_unmatched_instructions.size();
  LOG(INFO) << "Changed instructions: "
            << diff_result.changed_instructions.size();
  LOG(INFO) << "Moved instructions: " << diff_result.moved_instructions.size();
  LOG(INFO) << "Unchanged instructions: "
            << diff_result.unchanged_instructions.size();
}

}  // namespace hlo_diff
}  // namespace xla
