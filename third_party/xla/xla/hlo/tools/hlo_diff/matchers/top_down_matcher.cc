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

#include "absl/log/log.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_dfs.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla::hlo_diff {

namespace {

// Recursively matches the two nodes top down when the opcodes and the
// position of the nodes in their parents' children list match.
void RecursiveTopDownMatcher(const HloInstructionNode* left,
                             const HloInstructionNode* right,
                             const MatcherType matcher_type,
                             HloGumgraphMappings& mappings,
                             bool require_same_children) {
  if (require_same_children) {
    if (left->children.size() != right->children.size()) {
      return;
    }
    for (auto i = 0; i < left->children.size(); ++i) {
      if (left->children[i]->instruction->opcode() !=
          right->children[i]->instruction->opcode()) {
        return;
      }
    }
  }
  for (auto i = 0; i < left->children.size() && i < right->children.size();
       ++i) {
    const HloInstructionNode* left_child = left->children[i];
    const HloInstructionNode* right_child = right->children[i];
    // TODO(b/360878130) - Use fingerprint to compare nodes.
    if (left_child->instruction->opcode() !=
            right_child->instruction->opcode() ||
        !(mappings.MapInstructionsIfAbsent(left_child, right_child,
                                           matcher_type))) {
      // Stop recursive matching if the nodes are not matched, or
      // non-overwriting mapping failed.
      continue;
    }
    RecursiveTopDownMatcher(left_child, right_child, matcher_type, mappings,
                            require_same_children);
  }
}

}  // namespace

void GreedyTopDownMatcher::Match(HloGumgraphMappings& mappings) const {
  LOG(INFO) << "Running GreedyTopDownMatcher: matching umatched nodes";
  int current_mapping_count = mappings.left_to_right_instruction_map.size();
  HloGumgraphDfs(
      left_.GetRoot(),
      [&](const HloInstructionNode& left_node) {
        auto it = mappings.left_to_right_instruction_map.left.find(&left_node);
        if (it == mappings.left_to_right_instruction_map.left.end()) {
          return;
        }

        RecursiveTopDownMatcher(&left_node, it->second, type_, mappings,
                                require_same_children_);
      },
      DfsTraversalOrder::kPostOrder, left_.GetNodeCount());
  LOG(INFO) << "Finished GreedyTopDownMatcher. Total left to right mappings: "
            << mappings.left_to_right_instruction_map.size() -
                   current_mapping_count;
}

}  // namespace xla::hlo_diff
