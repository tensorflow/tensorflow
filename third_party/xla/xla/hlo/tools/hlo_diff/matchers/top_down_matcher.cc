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

#include "xla/hlo/tools/hlo_diff/matchers/top_down_matcher.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_dfs.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/matchers/bipartite_matching.h"

namespace xla::hlo_diff {
namespace {

// Returns true if the left and right nodes have different number of children or
// different child opcodes. This is called for Strict TopDownMatcher.
bool ShouldSkipMatching(const HloInstructionNode& left_node,
                        const HloInstructionNode& right_node) {
  if (left_node.children.size() != right_node.children.size()) {
    return true;
  }
  for (auto i = 0; i < left_node.children.size(); ++i) {
    if (left_node.children[i]->instruction->opcode() !=
        right_node.children[i]->instruction->opcode()) {
      return true;
    }
  }
  return false;
}

// Returns the unmatched children of a given node. It takes a predicate to
// determine if a child is already mapped.
template <typename IsMappedPredicate>
std::vector<HloInstructionNode*> GetUnmatchedChildren(
    const std::vector<HloInstructionNode*>& children,
    IsMappedPredicate is_mapped) {
  std::vector<HloInstructionNode*> unmatched_children;
  absl::flat_hash_set<const HloInstructionNode*> visited;
  for (HloInstructionNode* child : children) {
    if (!is_mapped(child) && !visited.contains(child)) {
      unmatched_children.push_back(child);
      visited.insert(child);
    }
  }
  return unmatched_children;
}

}  // namespace

void GreedyTopDownMatcher::Match(HloGumgraphMappings& mappings) const {
  LOG(INFO) << "Running GreedyTopDownMatcher: matching umatched nodes";
  int current_mapping_count = mappings.left_to_right_instruction_map.size();
  HloGumgraphDfs(
      left_.GetRoot(),
      [&](const HloInstructionNode& left_node) {
        auto right_node =
            mappings.left_to_right_instruction_map.GetRight(&left_node);
        if (!right_node) {
          return;
        }

        if (require_same_children_ &&
            ShouldSkipMatching(left_node, **right_node)) {
          return;
        }

        std::vector<HloInstructionNode*> left_children = GetUnmatchedChildren(
            left_node.children, [&mappings](const HloInstructionNode* node) {
              return mappings.InstructionMapContainsRight(node);
            });

        std::vector<HloInstructionNode*> right_children = GetUnmatchedChildren(
            (*right_node)->children,
            [&mappings](const HloInstructionNode* node) {
              return mappings.InstructionMapContainsLeft(node);
            });

        MatchInstructions(left_, right_, left_children, right_children,
                          mappings, type_, MapByPositionMode::kAlways);
      },
      DfsTraversalOrder::kPreOrder, left_.GetNodeCount());
  LOG(INFO) << "Finished GreedyTopDownMatcher. Total left to right mappings: "
            << mappings.left_to_right_instruction_map.size() -
                   current_mapping_count;
}

}  // namespace xla::hlo_diff
