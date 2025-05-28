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

struct NodePairSimilarity {
  const HloInstructionNode* left;
  const HloInstructionNode* right;
  double similarity;
};

// Maps the two subgraphs starting from the given nodes.
void MapSubgraph(const HloInstructionNode* absl_nonnull left,
                 int left_graph_size,
                 const HloInstructionNode* absl_nonnull right,
                 int right_graph_size, const MatcherType matcher_type,
                 HloGumgraphMappings& mappings,
                 absl::flat_hash_set<const HloInstructionNode*>&
                     exact_mapped_subgraph_roots) {
  std::vector<const HloInstructionNode*> left_subgraph;
  HloGumgraphBfs(
      *left,
      [&left_subgraph](const HloInstructionNode& node) {
        left_subgraph.push_back(&node);
        return true;
      },
      BfsTraversalDirection::kForward, left_graph_size,
      [&exact_mapped_subgraph_roots](const HloInstructionNode& node) {
        return !exact_mapped_subgraph_roots.contains(&node);
      });
  std::vector<const HloInstructionNode*> right_subgraph;
  HloGumgraphBfs(
      *right,
      [&right_subgraph](const HloInstructionNode& node) {
        right_subgraph.push_back(&node);
        return true;
      },
      BfsTraversalDirection::kForward, right_graph_size,
      [&exact_mapped_subgraph_roots](const HloInstructionNode& node) {
        return !exact_mapped_subgraph_roots.contains(&node);
      });
  if (left_subgraph.size() != right_subgraph.size()) {
    LOG(WARNING) << "Subgraph (" << left->instruction->name() << " vs "
                 << right->instruction->name() << ") with same fingerprint "
                 << left->props.subgraph_fingerprint
                 << " but different size: " << left_subgraph.size() << " vs "
                 << right_subgraph.size();
    return;
  }
  for (int i = 0; i < left_subgraph.size(); ++i) {
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
  for (int i = 0; i < left_subgraph.size(); ++i) {
    mappings.MapInstructionsIfAbsent(left_subgraph[i], right_subgraph[i],
                                     matcher_type);
    exact_mapped_subgraph_roots.insert(left_subgraph[i]);
    exact_mapped_subgraph_roots.insert(right_subgraph[i]);
    // Mark all nodes except the root as unchanged.
    if (i != 0) {
      mappings.left_to_right_instruction_map.left.find(left_subgraph[i])
          ->info.unchanged = true;
    }
  }
}

}  // namespace

void GreedySubGraphExactMatcher::Match(HloGumgraphMappings& mappings) const {
  // Find candidate subgraphs that match exactly.
  LOG(INFO) << "Running GreedySubgraphExactMatcher: matching subgraphs that "
               "match exactly";
  int current_mapping_count = mappings.left_to_right_instruction_map.size();
  absl::flat_hash_map<const HloInstructionNode*,
                      std::vector<const HloInstructionNode*>>
      candidates, candidates_reverse;
  int max_height =
      std::max(left_.GetRoot().props.height, right_.GetRoot().props.height);
  // Cache all subgraphs at each height.
  absl::flat_hash_map<int, std::vector<const HloInstructionNode*>>
      source_subgraphs;
  HloGumgraphBfs(
      left_.GetRoot(),
      [&source_subgraphs](const HloInstructionNode& node) {
        if (!node.is_root) {
          source_subgraphs[node.props.height].push_back(&node);
        }
        return true;
      },
      BfsTraversalDirection::kForward, left_.GetNodeCount());
  absl::flat_hash_map<int, std::vector<const HloInstructionNode*>>
      target_subgraphs;
  HloGumgraphBfs(
      right_.GetRoot(),
      [&target_subgraphs](const HloInstructionNode& node) {
        if (!node.is_root) {
          target_subgraphs[node.props.height].push_back(&node);
        }
        return true;
      },
      BfsTraversalDirection::kForward, right_.GetNodeCount());

  absl::flat_hash_set<const HloInstructionNode*> ignored;
  absl::flat_hash_set<const HloInstructionNode*> exact_mapped_subgraph_roots;
  // Find exact match left-right subgraphs candidates greedly from high to low
  // height.
  for (int height = max_height; height >= 0; --height) {
    if (!source_subgraphs.contains(height) ||
        !target_subgraphs.contains(height)) {
      continue;
    }
    absl::flat_hash_set<const HloInstructionNode*> found;
    // Find exact match left-right subgraph candidates at the current height.
    absl::flat_hash_map<uint64_t,
                        absl::flat_hash_set<const HloInstructionNode*>>
        source_by_fingerprint;
    absl::flat_hash_map<uint64_t,
                        absl::flat_hash_set<const HloInstructionNode*>>
        target_by_fingerprint;
    for (const HloInstructionNode* source_node : source_subgraphs[height]) {
      if (ignored.contains(source_node) ||
          mappings.InstructionMapContainsLeft(source_node)) {
        continue;
      }
      source_by_fingerprint[source_node->props.subgraph_fingerprint].insert(
          source_node);
    }
    for (const HloInstructionNode* target_node : target_subgraphs[height]) {
      if (ignored.contains(target_node) ||
          mappings.InstructionMapContainsRight(target_node)) {
        continue;
      }
      target_by_fingerprint[target_node->props.subgraph_fingerprint].insert(
          target_node);
    }
    for (const auto& [fingerprint, source_nodes] : source_by_fingerprint) {
      if (auto it = target_by_fingerprint.find(fingerprint);
          it != target_by_fingerprint.end()) {
        // Map 1:1 candidates. Check if the source and target subgraphs are
        // exactly the same, if so, map them.
        if (source_nodes.size() == 1 && it->second.size() == 1) {
          MapSubgraph(*source_nodes.begin(), left_.GetNodeCount(),
                      *it->second.begin(), right_.GetNodeCount(), type_,
                      mappings, exact_mapped_subgraph_roots);
        }
        found.insert(source_nodes.begin(), source_nodes.end());
        found.insert(it->second.begin(), it->second.end());
      }
    }
    // Ignore all nodes in the subgraphs that matched in later traversals.
    for (const HloInstructionNode* found_node : found) {
      HloGumgraphBfs(
          *found_node, [](const HloInstructionNode& node) { return true; },
          BfsTraversalDirection::kForward,
          std::max(left_.GetNodeCount(), right_.GetNodeCount()),
          [&ignored](const HloInstructionNode& node) {
            if (ignored.contains(&node)) {
              return false;
            }
            ignored.insert(&node);
            return true;
          });
    }
  }

  LOG(INFO)
      << "Finished GreedySubGraphExactMatcher. Found left to right mappings: "
      << mappings.left_to_right_instruction_map.size() - current_mapping_count;
}

}  // namespace hlo_diff
}  // namespace xla
