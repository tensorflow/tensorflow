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

#include "xla/hlo/tools/hlo_diff/matchers/bottom_up_matcher.h"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_bfs.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_dfs.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/matchers/similarity.h"

namespace xla::hlo_diff {

namespace {

constexpr double kOperandsMatchScore = 0.75;
constexpr double kOperandsFingerprintsMatchScore = 0.5;

constexpr int kProgressBarWidth = 60;
constexpr char kProgressBarBlock = '|';
constexpr char kProgressBarEmpty = ' ';

void PrintProgress(int percentage) {
  int lpad = static_cast<int>(percentage / 100.0 * kProgressBarWidth);
  int rpad = kProgressBarWidth - lpad;
  std::cout << "\r" << std::setw(3) << percentage << "% ["
            << std::string(lpad, kProgressBarBlock)
            << std::string(rpad, kProgressBarEmpty) << "]" << std::flush;
}

absl::flat_hash_set<const HloInstructionNode*> GetSubgraphForDiceSim(
    const HloInstructionNode* start_node, int graph_size, int max_subgraph_size,
    int min_bfs_distance) {
  absl::flat_hash_set<const HloInstructionNode*> nodes;
  nodes.reserve(max_subgraph_size);
  HloGumgraphBfs(
      *start_node,
      [&](const HloInstructionNode& node, int distance) {
        nodes.insert(&node);
        return distance <= min_bfs_distance || nodes.size() < max_subgraph_size;
      },
      BfsTraversalDirection::kForward, graph_size);
  return nodes;
}

// DiceSim similarity score between two subgraphs. Subgraphs are limited to
// first max_subgraph_size nodes of BFS starting from the given nodes.
double DiceSimLimitedSubgraph(
    const HloInstructionNode* absl_nonnull left,
    const HloInstructionNode* absl_nonnull right, HloGumgraphMappings& mappings,
    int max_subgraph_size, int min_bfs_distance, int right_graph_size,
    absl::flat_hash_set<const HloInstructionNode*>& left_nodes,
    absl::flat_hash_map<const HloInstructionNode*,
                        absl::flat_hash_set<const HloInstructionNode*>>&
        right_bfs_set_cache) {
  auto get_right_subgraph_set = [&](const HloInstructionNode* start_node,
                                    int graph_size)
      -> const absl::flat_hash_set<const HloInstructionNode*>& {
    if (right_bfs_set_cache.contains(start_node)) {
      return right_bfs_set_cache.at(start_node);
    }

    absl::flat_hash_set<const HloInstructionNode*> nodes =
        GetSubgraphForDiceSim(start_node, graph_size, max_subgraph_size,
                              min_bfs_distance);
    auto [it, inserted] =
        right_bfs_set_cache.try_emplace(start_node, std::move(nodes));
    return it->second;
  };

  const absl::flat_hash_set<const HloInstructionNode*>& right_nodes_set =
      get_right_subgraph_set(right, right_graph_size);

  int common = 0;
  for (const HloInstructionNode* left_node : left_nodes) {
    if (auto right_node =
            mappings.left_to_right_instruction_map.GetRight(left_node);
        right_node.has_value() && right_nodes_set.contains(*right_node)) {
      ++common;
    }
  }

  double denominator =
      static_cast<double>(left_nodes.size() + right_nodes_set.size());
  if (denominator == 0) {
    return 0.0;
  }

  return 2.0 * static_cast<double>(common) / denominator;
}

// Returns true if all HloValues used by the left and right nodes have their
// defining instructions matched.
double AllOperandHloValuesMatchedScore(
    const HloInstructionNode* left_node, const HloInstructionNode* right_node,
    const HloGumgraph& left, const HloGumgraph& right,
    HloGumgraphMappings& mappings) {
  const auto& left_hlo_values = left_node->used_values;
  const auto& right_hlo_values = right_node->used_values;
  if (left_hlo_values.empty() || right_hlo_values.empty() ||
      (left_hlo_values.size() != right_hlo_values.size())) {
    return 0.0;
  }

  bool fingerprints_matched = true;
  bool mappings_matched = true;
  for (int i = 0; i < left_hlo_values.size(); ++i) {
    if (!fingerprints_matched && !mappings_matched) {
      // stop if both fingerprints and mappings are not matched.
      break;
    }

    HloInstructionNode* left_hlo_value_node =
        left.GetNode(left_hlo_values[i]->defining_instruction());
    HloInstructionNode* right_hlo_value_node =
        right.GetNode(right_hlo_values[i]->defining_instruction());
    if (auto right_node = mappings.left_to_right_instruction_map.GetRight(
            left_hlo_value_node);
        right_node != right_hlo_value_node) {
      mappings_matched = false;
    }
    if (left_hlo_value_node->props.fingerprint !=
        right_hlo_value_node->props.fingerprint) {
      fingerprints_matched = false;
    }
  }

  if (mappings_matched) {
    return kOperandsMatchScore;
  }
  if (fingerprints_matched) {
    return kOperandsFingerprintsMatchScore;
  }
  return 0.0;
}

}  // namespace

void GreedyLimitedCandidatesBottomUpMatcher::Match(
    HloGumgraphMappings& mappings) const {
  LOG(INFO) << "Running GreedyLimitedCandidatesBottomUpMatcher: matching "
               "subgraphs that match based on Dice similarity";
  absl::flat_hash_map<const HloInstructionNode*,
                      absl::flat_hash_set<const HloInstructionNode*>>
      right_bfs_set_cache;

  int current_mapping_count = mappings.left_to_right_instruction_map.size();
  std::vector<const HloInstructionNode*> left_postorder = GetAllNodesInDfsOrder(
      left_.GetRoot(), DfsTraversalOrder::kPostOrder, left_.GetNodeCount());
  int progress = 0;
  int total_steps = left_postorder.size();
  for (size_t i = 0; i < total_steps; ++i) {
    const auto* left_node = left_postorder[i];
    int current_progress = static_cast<int>((i * 100.0) / total_steps);
    if (current_progress > progress) {
      PrintProgress(current_progress);
      progress = current_progress;
    }
    // Skip matched nodes or ones without children.
    if (mappings.InstructionMapContainsLeft(left_node) ||
        left_node->children.empty()) {
      continue;
    }

    // Pre-compute the left_node's subgraph once for this iteration.
    absl::flat_hash_set<const HloInstructionNode*> left_nodes_for_dice_sim =
        GetSubgraphForDiceSim(left_node, left_.GetNodeCount(),
                              max_dice_subgraph_size_, min_bfs_distance_);

    std::vector<const HloInstructionNode*> right_seeds;
    int count = 0;
    HloGumgraphBfs(
        *left_node,
        [&](const HloInstructionNode& node, int distance) {
          if (auto right_node =
                  mappings.left_to_right_instruction_map.GetRight(&node);
              right_node.has_value()) {
            right_seeds.push_back(*right_node);
          }
          // Don't pursue subgraphs with too many childrens. Allows us to visit
          // deeper subgraphs without getting stuck on a single node with a
          // large number of children.
          if (node.children.size() > right_seeds_traversal_limit_ / 2) {
            return false;
          }
          return distance <= min_bfs_distance_ ||
                 ++count < right_seeds_traversal_limit_;
        },
        BfsTraversalDirection::kForward, left_.GetNodeCount());

    // Find right candidates and maxSimilarity on the fly.
    double max_similarity = 0;
    const HloInstructionNode* right_candidate = nullptr;
    count = 0;
    std::string debug_string;
    if (debug_mode_) {
      for (const HloInstructionNode* right_seed : right_seeds) {
        absl::StrAppend(&debug_string,
                        "seed: ", right_seed->instruction->name(), "\n");
      }
    }
    HloGumgraphBfs(
        right_seeds,
        [&](const HloInstructionNode& node, int distance) {
          if (!mappings.InstructionMapContainsRight(&node) &&
              node.instruction->opcode() == left_node->instruction->opcode()) {
            // Found candidate. Calculate similarity.
            double operands_match_similarity = AllOperandHloValuesMatchedScore(
                left_node, &node, left_, right_, mappings);
            double dice_sim = DiceSimLimitedSubgraph(
                left_node, &node, mappings, max_dice_subgraph_size_,
                min_bfs_distance_, right_.GetNodeCount(),
                left_nodes_for_dice_sim, right_bfs_set_cache);
            double node_property_similarity =
                NodePropertySimilarity(left_node, &node);
            double ancestor_similarity = AncestorSubGraphLcsSimilarity(
                left_node, &node, max_ancestors_to_consider_, min_bfs_distance_,
                left_.GetNodeCount(), right_.GetNodeCount());
            // We give ancestor similarity a lower weight as its lower signal
            // in comparison to dice similarity and node attributes similarity.
            double similarity = operands_match_similarity +
                                node_property_similarity + dice_sim +
                                ancestor_similarity;
            if (similarity > max_similarity) {
              max_similarity = similarity;
              right_candidate = &node;
            }
            if (debug_mode_) {
              absl::StrAppend(
                  &debug_string, "Similarity(", left_node->instruction->name(),
                  ", ", node.instruction->name(), "): ", similarity,
                  " (operands_match_similarity: ", operands_match_similarity,
                  ", node_property_similarity: ", node_property_similarity,
                  ", dice_sim: ", dice_sim,
                  ", ancestor_similarity: ", ancestor_similarity, ")\n");
            }
          }
          return distance <= min_bfs_distance_ ||
                 ++count < right_seeds_traversal_limit_;
        },
        BfsTraversalDirection::kReverse, right_.GetNodeCount());
    if (max_similarity > min_similarity_) {
      mappings.MapInstructionsIfAbsent(left_node, right_candidate, type_,
                                       debug_string);
    }
  }
  PrintProgress(100);
  LOG(INFO) << "Finished GreedyLimitedCandidatesBottomUpMatcher. Total left to "
               "right mappings: "
            << mappings.left_to_right_instruction_map.size() -
                   current_mapping_count;
}

}  // namespace xla::hlo_diff
