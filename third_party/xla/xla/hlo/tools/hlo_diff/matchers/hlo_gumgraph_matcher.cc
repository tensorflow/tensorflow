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

#include "xla/hlo/tools/hlo_diff/matchers/hlo_gumgraph_matcher.h"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_bfs.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_dfs.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/matchers/similarity.h"
#include "xla/service/hlo_value.h"

namespace xla {
namespace hlo_diff {
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

// DiceSim similarity score between two subgraphs. Subgraphs are limited to
// first max_subgraph_size nodes of BFS starting from the given nodes.
double DiceSimLimitedSubgraph(const HloInstructionNode* absl_nonnull left,
                              const HloInstructionNode* absl_nonnull right,
                              HloGumgraphMappings& mappings,
                              int max_subgraph_size, int min_bfs_distance,
                              int left_graph_size, int right_graph_size) {
  absl::flat_hash_set<const HloInstructionNode*> left_nodes;
  absl::flat_hash_set<const HloInstructionNode*> right_nodes;
  HloGumgraphBfs(
      *left,
      [&](const HloInstructionNode& node, int distance) {
        left_nodes.insert(&node);
        return distance <= min_bfs_distance ||
               left_nodes.size() < max_subgraph_size;
      },
      BfsTraversalDirection::kForward, left_graph_size);
  HloGumgraphBfs(
      *right,
      [&](const HloInstructionNode& node, int distance) {
        right_nodes.insert(&node);
        return distance <= min_bfs_distance ||
               right_nodes.size() < max_subgraph_size;
      },
      BfsTraversalDirection::kForward, right_graph_size);
  int common = 0;
  for (const HloInstructionNode* left_node : left_nodes) {
    if (auto it = mappings.left_to_right_instruction_map.left.find(left_node);
        it != mappings.left_to_right_instruction_map.left.end() &&
        right_nodes.contains(it->second)) {
      ++common;
    }
  }

  return 2 * static_cast<double>(common) /
         static_cast<double>((left_nodes.size() + right_nodes.size()));
}

// Returns all HloValues used by the given instruction.
std::vector<const HloValue*> GetAllValuesUsedByInstruction(
    const HloInstruction* instruction, const HloGumgraph& gumgraph) {
  if (instruction->opcode() == HloOpcode::kParameter) {
    if (instruction->parent()->IsEntryComputation() ||
        gumgraph.GetHloValueTracing().ValueIsDefinedAt(instruction)) {
      return std::vector<const HloValue*>();
    }

    return gumgraph.GetHloValueTracing()
        .GetFlattenedValueSet(instruction)
        .values();
  }

  std::vector<const HloValue*> values_used_by_instruction;
  for (const HloInstruction* operand : instruction->operands()) {
    const HloValueSet operand_value_set =
        gumgraph.GetHloValueTracing().GetFlattenedValueSet(operand);
    for (const HloValue* value : operand_value_set.values()) {
      absl::Span<const HloUse> uses = value->GetUses();
      for (const HloUse& use : uses) {
        if (use.instruction == instruction) {
          values_used_by_instruction.push_back(value);
          break;
        }
      }
    }
  }

  return values_used_by_instruction;
}

// Returns true if all HloValues used by the left and right nodes have their
// defining instructions matched.
double AllOperandHloValuesMatchedScore(
    const HloInstructionNode* left_node, const HloInstructionNode* right_node,
    const HloGumgraph& left, const HloGumgraph& right,
    absl::flat_hash_map<const HloInstruction*,
                        const std::vector<const HloValue*>>&
        instruction_used_values_cache,
    HloGumgraphMappings& mappings) {
  if (!instruction_used_values_cache.contains(left_node->instruction)) {
    instruction_used_values_cache.emplace(
        left_node->instruction,
        GetAllValuesUsedByInstruction(left_node->instruction, left));
  }
  if (!instruction_used_values_cache.contains(right_node->instruction)) {
    instruction_used_values_cache.emplace(
        right_node->instruction,
        GetAllValuesUsedByInstruction(right_node->instruction, right));
  }
  auto& left_hlo_values = instruction_used_values_cache[left_node->instruction];
  auto& right_hlo_values =
      instruction_used_values_cache[right_node->instruction];

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
    if (auto it = mappings.left_to_right_instruction_map.left.find(
            left_hlo_value_node);
        it == mappings.left_to_right_instruction_map.left.end() ||
        it->second != right_hlo_value_node) {
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

void GreedyLimitedCandidatesBottomUpMatcher::Match(
    HloGumgraphMappings& mappings) const {
  LOG(INFO) << "Running GreedyLimitedCandidatesBottomUpMatcher: matching "
               "subgraphs that match based on Dice similarity";
  absl::flat_hash_map<const HloInstruction*, const std::vector<const HloValue*>>
      instruction_used_values_cache;
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

    std::vector<const HloInstructionNode*> right_seeds;
    int count = 0;
    HloGumgraphBfs(
        *left_node,
        [&](const HloInstructionNode& node, int distance) {
          if (auto it = mappings.left_to_right_instruction_map.left.find(&node);
              it != mappings.left_to_right_instruction_map.left.end()) {
            right_seeds.push_back(it->second);
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
    HloGumgraphBfs(
        right_seeds,
        [&](const HloInstructionNode& node, int distance) {
          if (!mappings.InstructionMapContainsRight(&node) &&
              node.instruction->opcode() == left_node->instruction->opcode()) {
            // Found candidate. Calculate similarity.
            double operands_match_similarity = AllOperandHloValuesMatchedScore(
                left_node, &node, left_, right_, instruction_used_values_cache,
                mappings);
            double dice_sim = DiceSimLimitedSubgraph(
                left_node, &node, mappings, max_dice_subgraph_size_,
                min_bfs_distance_, left_.GetNodeCount(), right_.GetNodeCount());
            double node_attributes_similarity =
                NodeAttributesSimilarity(left_node, &node);
            double ancestor_similarity = AncestorSubGraphSimilarity(
                left_node, &node, max_ancestors_to_consider_, min_bfs_distance_,
                left_.GetNodeCount(), right_.GetNodeCount());
            // We give ancestor similarity a lower weight as its lower signal
            // in comparison to dice similarity and node attributes similarity.
            double similarity = operands_match_similarity +
                                node_attributes_similarity + dice_sim +
                                ancestor_similarity / 2;
            if (similarity > max_similarity) {
              max_similarity = similarity;
              right_candidate = &node;
            }
          }
          return distance <= min_bfs_distance_ ||
                 ++count < right_seeds_traversal_limit_;
        },
        BfsTraversalDirection::kReverse, right_.GetNodeCount());
    if (max_similarity > min_similarity_) {
      mappings.MapInstructionsIfAbsent(left_node, right_candidate, type_);
    }
  }
  LOG(INFO) << "Finished GreedyLimitedCandidatesBottomUpMatcher. Total left to "
               "right mappings: "
            << mappings.left_to_right_instruction_map.size() -
                   current_mapping_count;
}

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

}  // namespace hlo_diff
}  // namespace xla
