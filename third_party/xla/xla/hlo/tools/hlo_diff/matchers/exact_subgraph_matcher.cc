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

#include "xla/hlo/tools/hlo_diff/matchers/exact_subgraph_matcher.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_bfs.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla {
namespace hlo_diff {
namespace {

// Maps the two subgraphs starting from the given nodes.
void MapSubgraph(const HloInstructionNode* absl_nonnull left,
                 const HloGumgraph& left_graph,
                 const HloInstructionNode* absl_nonnull right,
                 const HloGumgraph& right_graph, const MatcherType matcher_type,
                 HloGumgraphMappings& mappings) {
  std::vector<const HloInstructionNode*> left_subgraph;
  HloGumgraphBfs(
      *left,
      [&](const HloInstructionNode& node) {
        left_subgraph.push_back(&node);
        return true;
      },
      BfsTraversalDirection::kForward, left_graph.GetNodeCount(),
      [&](const HloInstructionNode& node) {
        // Do not traverse into an already matched subgraph.
        return !mappings.InstructionMapContainsLeft(&node) || &node == left;
      });

  std::vector<const HloInstructionNode*> right_subgraph;
  HloGumgraphBfs(
      *right,
      [&](const HloInstructionNode& node) {
        right_subgraph.push_back(&node);
        return true;
      },
      BfsTraversalDirection::kForward, right_graph.GetNodeCount(),
      [&](const HloInstructionNode& node) {
        // Do not traverse into an already matched subgraph.
        return !mappings.InstructionMapContainsRight(&node) || &node == right;
      });

  if (left_subgraph.size() != right_subgraph.size()) {
    LOG(WARNING) << "Subgraph (" << left->instruction->name() << " vs "
                 << right->instruction->name() << ") with same fingerprint "
                 << left->props.subgraph_fingerprint
                 << " but different size: " << left_subgraph.size() << " vs "
                 << right_subgraph.size();
    return;
  }

  for (size_t i = 0; i < left_subgraph.size(); ++i) {
    if (left_subgraph[i]->instruction->opcode() !=
        right_subgraph[i]->instruction->opcode()) {
      LOG(WARNING) << "Subgraph (" << left->instruction->name() << " vs "
                   << right->instruction->name() << ") with same fingerprint "
                   << left->props.subgraph_fingerprint << " and size "
                   << left_subgraph.size() << " but has diff type at node " << i
                   << ":" << left_subgraph[i]->instruction->name() << " vs "
                   << right_subgraph[i]->instruction->name();
      return;
    }
  }

  for (size_t i = 0; i < left_subgraph.size(); ++i) {
    if (mappings.MapInstructionsIfAbsent(left_subgraph[i], right_subgraph[i],
                                         matcher_type)) {
      // Mark all nodes except the root as unchanged.
      if (i != 0) {
        mappings.left_to_right_instruction_map.left.find(left_subgraph[i])
            ->second.props->unchanged = true;
      }
    }
  }
}

}  // namespace

void GreedySubGraphExactMatcher::Match(HloGumgraphMappings& mappings) const {
  LOG(INFO) << "Running GreedySubgraphExactMatcher: matching subgraphs that "
               "match exactly";
  int current_mapping_count = mappings.left_to_right_instruction_map.size();

  // Cache all nodes at each height.
  absl::flat_hash_map<int, std::vector<const HloInstructionNode*>>
      source_nodes_by_height;
  for (const auto* node : left_.AllNodes()) {
    source_nodes_by_height[node->props.height].push_back(node);
  }
  absl::flat_hash_map<int, std::vector<const HloInstructionNode*>>
      target_nodes_by_height;
  for (const auto* node : right_.AllNodes()) {
    target_nodes_by_height[node->props.height].push_back(node);
  }

  int max_height =
      std::max(left_.GetRoot().props.height, right_.GetRoot().props.height);

  // Greedily find exact match subgraphs from tallest to shortest.
  for (int height = max_height; height > 0; --height) {
    if (!source_nodes_by_height.contains(height) ||
        !target_nodes_by_height.contains(height)) {
      continue;
    }

    absl::flat_hash_map<uint64_t, std::vector<const HloInstructionNode*>>
        source_by_fingerprint;
    for (const HloInstructionNode* source_node :
         source_nodes_by_height[height]) {
      if (!mappings.InstructionMapContainsLeft(source_node)) {
        source_by_fingerprint[source_node->props.subgraph_fingerprint]
            .push_back(source_node);
      }
    }

    absl::flat_hash_map<uint64_t, std::vector<const HloInstructionNode*>>
        target_by_fingerprint;
    for (const HloInstructionNode* target_node :
         target_nodes_by_height[height]) {
      if (!mappings.InstructionMapContainsRight(target_node)) {
        target_by_fingerprint[target_node->props.subgraph_fingerprint]
            .push_back(target_node);
      }
    }

    for (auto& [fingerprint, source_nodes] : source_by_fingerprint) {
      auto it = target_by_fingerprint.find(fingerprint);
      if (it == target_by_fingerprint.end()) {
        continue;
      }

      auto& target_nodes = it->second;
      // For now, only map 1:1 candidates to avoid ambiguity.
      if (source_nodes.size() == 1 && target_nodes.size() == 1) {
        MapSubgraph(source_nodes[0], left_, target_nodes[0], right_, type_,
                    mappings);
      }
    }
  }

  LOG(INFO)
      << "Finished GreedySubGraphExactMatcher. Found left to right mappings: "
      << mappings.left_to_right_instruction_map.size() - current_mapping_count;
}

}  // namespace hlo_diff
}  // namespace xla
