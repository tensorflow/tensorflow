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
#include <utility>
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

// Gets all mapped descendants of a node, i.e. all descendants that have a
// mapping to a node in the other graph.
template <typename IsMappedFn>
absl::flat_hash_set<const HloInstructionNode*> GetMappedDescendants(
    const HloInstructionNode* node, const HloGumgraph& graph,
    IsMappedFn&& is_mapped_fn) {
  absl::flat_hash_set<const HloInstructionNode*> descendants;
  HloGumgraphBfs(
      *node,
      [&](const HloInstructionNode& descendant) {
        if (is_mapped_fn(&descendant)) {
          descendants.insert(&descendant);
        }
        return true;
      },
      BfsTraversalDirection::kForward, graph.GetNodeCount(),
      [](const HloInstructionNode&) { return true; });
  return descendants;
}

// Takes the roots of two subgraphs that are found to be matching, performs a
// BFS to collect all nodes in each subgraph, verifies they are structurally
// identical, and then adds the one-to-one mappings for all nodes.
void MapSubgraph(const HloInstructionNode* absl_nonnull left,
                 const HloGumgraph& left_graph,
                 const HloInstructionNode* absl_nonnull right,
                 const HloGumgraph& right_graph, const MatcherType matcher_type,
                 HloGumgraphMappings& mappings) {
  std::vector<const HloInstructionNode*> left_subgraph;
  HloGumgraphBfs(
      *left,
      [&left_subgraph](const HloInstructionNode& node) {
        left_subgraph.push_back(&node);
        return true;
      },
      BfsTraversalDirection::kForward, left_graph.GetNodeCount(),
      [&mappings](const HloInstructionNode& node) {
        auto props =
            mappings.left_to_right_instruction_map.GetPropsByLeft(&node);
        // Do not traverse into a subgraph already matched by this matcher.
        return !props.has_value() ||
               props->matcher_type != MatcherType::kGreedySubGraphExactMatcher;
      });

  std::vector<const HloInstructionNode*> right_subgraph;
  HloGumgraphBfs(
      *right,
      [&right_subgraph](const HloInstructionNode& node) {
        right_subgraph.push_back(&node);
        return true;
      },
      BfsTraversalDirection::kForward, right_graph.GetNodeCount(),
      [&mappings](const HloInstructionNode& node) {
        auto props =
            mappings.left_to_right_instruction_map.GetPropsByRight(&node);
        // Do not traverse into an subgraph already matched by this matcher.
        return !props.has_value() ||
               props->matcher_type != MatcherType::kGreedySubGraphExactMatcher;
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

  // Add mappings for each pair of nodes in the subgraphs
  for (size_t i = 0; i < left_subgraph.size(); ++i) {
    if (mappings.MapInstructionsIfAbsent(left_subgraph[i], right_subgraph[i],
                                         matcher_type)) {
      // Mark all nodes except the root as unchanged.
      if (i != 0) {
        auto props = mappings.left_to_right_instruction_map.GetPropsByLeft(
            left_subgraph[i]);
        props->unchanged = true;
        mappings.left_to_right_instruction_map.SetPropsByLeft(left_subgraph[i],
                                                              *props);
      }
    }
  }
}

// Handles cases where multiple subgraphs share the same fingerprint, making the
// match ambiguous. It attempts to resolve ambiguity by comparing the sets of
// already-mapped descendants for each potential pair.
void MatchAmbiguousSubgraphs(
    const std::vector<const HloInstructionNode*>& left_nodes,
    const std::vector<const HloInstructionNode*>& right_nodes,
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    HloGumgraphMappings& mappings, const MatcherType matcher_type) {
  // Precompute the set of mapped descendants for each ambiguous node.
  absl::flat_hash_map<const HloInstructionNode*,
                      absl::flat_hash_set<const HloInstructionNode*>>
      left_nodes_descendants, right_nodes_descendants;
  for (const HloInstructionNode* node : left_nodes) {
    left_nodes_descendants[node] = GetMappedDescendants(
        node, left_graph, [&mappings](const HloInstructionNode* node) {
          return mappings.left_to_right_instruction_map.ContainsLeft(node);
        });
  }
  for (const HloInstructionNode* node : right_nodes) {
    right_nodes_descendants[node] = GetMappedDescendants(
        node, right_graph, [&mappings](const HloInstructionNode* node) {
          return mappings.left_to_right_instruction_map.ContainsRight(node);
        });
  }

  for (const HloInstructionNode* left_node : left_nodes) {
    const auto& left_node_descendants = left_nodes_descendants[left_node];
    if (left_node_descendants.empty()) {
      continue;
    }

    // Get the corresponding peers in the right graph for the left descendants.
    absl::flat_hash_set<const HloInstructionNode*> matched_peers;
    for (const HloInstructionNode* node : left_node_descendants) {
      matched_peers.insert(
          *mappings.left_to_right_instruction_map.GetRight(node));
    }

    // Find a right node whose descendants perfectly match the left node's
    // descendant peers.
    for (const HloInstructionNode* right_node : right_nodes) {
      const auto& right_node_descendants = right_nodes_descendants[right_node];
      if (matched_peers == right_node_descendants) {
        MapSubgraph(left_node, left_graph, right_node, right_graph,
                    matcher_type, mappings);
        left_nodes_descendants.erase(left_node);
        right_nodes_descendants.erase(right_node);
        break;
      }
    }
  }

  // If only one ambiguous pair remains, they must be a match.
  if (left_nodes_descendants.size() == 1 &&
      right_nodes_descendants.size() == 1) {
    MapSubgraph(left_nodes_descendants.begin()->first, left_graph,
                right_nodes_descendants.begin()->first, right_graph,
                matcher_type, mappings);
  }
}

}  // namespace

void GreedySubGraphExactMatcher::Match(HloGumgraphMappings& mappings) const {
  LOG(INFO) << "Running GreedySubgraphExactMatcher: matching subgraphs that "
               "match exactly";
  int current_mapping_count = mappings.left_to_right_instruction_map.size();

  absl::flat_hash_map<int, std::vector<const HloInstructionNode*>>
      left_nodes_by_height, right_nodes_by_height;
  for (const auto* node : left_.AllNodes()) {
    left_nodes_by_height[node->props.height].push_back(node);
  }
  for (const auto* node : right_.AllNodes()) {
    right_nodes_by_height[node->props.height].push_back(node);
  }

  int max_height =
      std::max(left_.GetRoot().props.height, right_.GetRoot().props.height);
  for (int height = max_height; height > 0; --height) {
    if (!left_nodes_by_height.contains(height) ||
        !right_nodes_by_height.contains(height)) {
      continue;
    }

    // Within each height, group nodes by their subgraph fingerprint.
    absl::flat_hash_map<uint64_t, std::vector<const HloInstructionNode*>>
        left_nodes_by_fingerprint, right_nodes_by_fingerprint;
    for (const HloInstructionNode* node : left_nodes_by_height[height]) {
      if (!mappings.InstructionMapContainsLeft(node)) {
        left_nodes_by_fingerprint[node->props.subgraph_fingerprint].push_back(
            node);
      }
    }
    for (const HloInstructionNode* node : right_nodes_by_height[height]) {
      if (!mappings.InstructionMapContainsRight(node)) {
        right_nodes_by_fingerprint[node->props.subgraph_fingerprint].push_back(
            node);
      }
    }

    for (auto& [fingerprint, left_nodes] : left_nodes_by_fingerprint) {
      if (auto it = right_nodes_by_fingerprint.find(fingerprint);
          it != right_nodes_by_fingerprint.end()) {
        auto& right_nodes = it->second;
        if (left_nodes.size() == 1 && right_nodes.size() == 1) {
          MapSubgraph(left_nodes[0], left_, right_nodes[0], right_, type_,
                      mappings);
        } else {
          MatchAmbiguousSubgraphs(left_nodes, right_nodes, left_, right_,
                                  mappings, type_);
        }
      }
    }
  }

  LOG(INFO)
      << "Finished GreedySubGraphExactMatcher. Found left to right mappings: "
      << mappings.left_to_right_instruction_map.size() - current_mapping_count;
}

}  // namespace hlo_diff
}  // namespace xla
