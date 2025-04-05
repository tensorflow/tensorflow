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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_bfs.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/utils/hlo_diff_util.h"

namespace xla {
namespace hlo_diff {
namespace {

bool IsChangedInstruction(const HloInstructionNode* left_node,
                          const HloInstructionNode* right_node) {
  uint64_t left_fingerprint = GetHloInstructionFingerprint(
      left_node->instruction, HloPrintOptions::Fingerprint());
  uint64_t right_fingerprint = GetHloInstructionFingerprint(
      right_node->instruction, HloPrintOptions::Fingerprint());
  return left_fingerprint != right_fingerprint;
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
    diff_result->node_props.insert({left_node->instruction, left_node->props});
    if (!mappings.InstructionMapContainsLeft(left_node)) {
      diff_result->left_module_unmatched_instructions.push_back(
          left_node->instruction);
      continue;
    }
    const HloInstructionNode* right_node =
        mappings.left_to_right_instruction_map.left.find(left_node)->second;
    const HloInstructionNodeMappingProps& mapping_props =
        mappings.left_to_right_instruction_map.left.find(left_node)->info;

    if (IsChangedInstruction(left_node, right_node)) {
      diff_result->changed_instructions[left_node->instruction] =
          right_node->instruction;
      diff_result->map_by[std::make_pair(left_node->instruction,
                                         right_node->instruction)] =
          mapping_props.matcher_type;
      continue;
    }
    // If node position is unchanged, add to unchanged instructions.
    if (mapping_props.unchanged) {
      diff_result->unchanged_instructions[left_node->instruction] =
          right_node->instruction;
      diff_result->map_by[std::make_pair(left_node->instruction,
                                         right_node->instruction)] =
          mapping_props.matcher_type;
      continue;
    }
    // TODO(b/369851244): Add moved instructions to diff result.
    diff_result->unchanged_instructions[left_node->instruction] =
        right_node->instruction;
    diff_result->map_by[std::make_pair(left_node->instruction,
                                       right_node->instruction)] =
        mapping_props.matcher_type;
  }

  for (const HloInstructionNode* right_node : right_all_nodes) {
    if (right_node->is_root) {
      continue;
    }
    diff_result->node_props.insert(
        {right_node->instruction, right_node->props});
    if (!mappings.InstructionMapContainsRight(right_node)) {
      diff_result->right_module_unmatched_instructions.push_back(
          right_node->instruction);
    }
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
