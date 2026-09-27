/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/tools/comparison/original_tensor_summary_key_matcher.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/hlo_diff/utils/bidirectional_map.h"
#include "xla/shape_util.h"

namespace xla::numerics::comparison {

void OriginalTensorSummaryKeyMatcher::InsertKey(
    TrieNode* root, const AbsoluteScopedTensorKey& key,
    absl::Span<const int64_t> dimensions) {
  TrieNode* curr = root;
  for (const auto& scope : key.scope_instructions) {
    auto& child = curr->children[scope.instruction_name];
    if (!child) {
      child = std::make_unique<TrieNode>(curr, scope.instruction_name);
    }
    curr = child.get();
    curr->iterations[scope.iteration_index]++;
  }
  auto& child = curr->children[key.tensor_key.instruction_name];
  if (!child) {
    child = std::make_unique<TrieNode>(curr, key.tensor_key.instruction_name);
  }
  curr = child.get();
  curr->tensors.try_emplace(
      key.tensor_key.shape_index,
      TensorInfo{/*shape_index = */ key.tensor_key.shape_index,
                 /*dimensions = */ {dimensions.begin(), dimensions.end()}});
}

absl::Status OriginalTensorSummaryKeyMatcher::ReadKeys(
    absl::string_view filename, TrieNode* root) {
  riegeli::RecordReader reader(riegeli::Maker<riegeli::FdReader>(filename));
  if (!reader.ok()) {
    return reader.status();
  }
  RecoveredTensorSummaryProto summary_proto;
  while (reader.ReadRecord(summary_proto)) {
    InsertKey(root,
              AbsoluteScopedTensorKey::FromProto(summary_proto.tensor_key()),
              summary_proto.original_tensor_summary().dimensions());
  }
  if (!reader.Close()) {
    return reader.status();
  }
  return absl::OkStatus();
}

bool OriginalTensorSummaryKeyMatcher::AreTensorsSimilar(
    const std::map<ShapeIndex, TensorInfo>& tensors1,
    const std::map<ShapeIndex, TensorInfo>& tensors2) {
  if (tensors1.size() != tensors2.size()) {
    return false;
  }
  for (const auto& [shape_index, tensor_info1] : tensors1) {
    auto it = tensors2.find(shape_index);
    if (it == tensors2.end() ||
        it->second.dimensions != tensor_info1.dimensions) {
      return false;
    }
  }
  return true;
}

int OriginalTensorSummaryKeyMatcher::CalculateSimilarity(
    const TrieNode* node1, const TrieNode* node2) {
  if (node1->tensors.empty() != node2->tensors.empty()) {
    return 0;
  }
  int score = 1;
  if (!node1->tensors.empty()) {
    if (AreTensorsSimilar(node1->tensors, node2->tensors)) {
      score++;
    }
  } else {
    if (node1->iterations == node2->iterations) {
      score += node1->iterations.size();
    }
  }
  if (node1->children.size() == node2->children.size()) {
    score++;
  }
  return score;
}

void OriginalTensorSummaryKeyMatcher::MatchAndQueue(
    TrieNode* baseline_node, TrieNode* target_node,
    std::vector<std::pair<TrieNode*, TrieNode*>>& to_process) {
  baseline_node->match = target_node;
  target_node->match = baseline_node;
  for (auto& [shape_index, baseline_tensor_info] : baseline_node->tensors) {
    auto tensor_it = target_node->tensors.find(shape_index);
    if (tensor_it != target_node->tensors.end()) {
      TensorInfo& target_tensor_info = tensor_it->second;
      // Only consider matching if the dimensions are the same.
      if (baseline_tensor_info.dimensions == target_tensor_info.dimensions) {
        baseline_tensor_info.match = &target_tensor_info;
        target_tensor_info.match = &baseline_tensor_info;
      }
    }
  }
  to_process.push_back({baseline_node, target_node});
}

void OriginalTensorSummaryKeyMatcher::MatchTrees(
    TrieNode* baseline_root, TrieNode* target_root,
    const BidirectionalMap<std::string, std::string, std::monostate>&
        hlo_diff_bimap) {
  std::vector<std::pair<TrieNode*, TrieNode*>> to_process;
  // The roots of both tries are considered matched, representing the entry
  // computation.
  to_process.push_back({baseline_root, target_root});
  baseline_root->match = target_root;
  target_root->match = baseline_root;

  // We explore matches depth-first starting from already-matched node pairs.
  while (!to_process.empty()) {
    auto [baseline_node, target_node] = to_process.back();
    to_process.pop_back();

    // NOLINTNEXTLINE
    for (auto& [name, baseline_child] : baseline_node->children) {
      // Skip nodes that have already been matched.
      if (baseline_child->match != nullptr) {
        continue;
      }

      // Attempt to find a corresponding instruction in the target trie using
      // the bimap.
      auto target_name = hlo_diff_bimap.GetRight(name);
      if (target_name) {
        auto it = target_node->children.find(*target_name);
        if (it != target_node->children.end() && it->second->match == nullptr) {
          MatchAndQueue(baseline_child.get(), it->second.get(), to_process);
        }
      }
    }

    // Heuristic matching for unmatched children.
    std::vector<TrieNode*> unmatched_baseline_children;
    // NOLINTNEXTLINE
    for (auto& [name, baseline_child] : baseline_node->children) {
      if (baseline_child->match == nullptr) {
        unmatched_baseline_children.push_back(baseline_child.get());
      }
    }
    std::vector<TrieNode*> unmatched_target_children;
    // NOLINTNEXTLINE
    for (auto& [name, target_child] : target_node->children) {
      if (target_child->match == nullptr) {
        unmatched_target_children.push_back(target_child.get());
      }
    }

    for (TrieNode* baseline_child : unmatched_baseline_children) {
      if (baseline_child->match != nullptr) {
        continue;
      }

      TrieNode* best_match = nullptr;
      int max_similarity = 0;
      bool ambiguous = false;

      for (TrieNode* target_child : unmatched_target_children) {
        if (target_child->match != nullptr) {
          continue;
        }

        int similarity = CalculateSimilarity(baseline_child, target_child);
        if (similarity > max_similarity) {
          max_similarity = similarity;
          best_match = target_child;
          ambiguous = false;
        } else if (similarity == max_similarity && max_similarity > 0) {
          ambiguous = true;
        }
      }

      if (best_match != nullptr && !ambiguous && max_similarity > 1) {
        MatchAndQueue(baseline_child, best_match, to_process);
      }
    }
  }
}

OriginalTensorSummaryKeyMatcher::OriginalTensorSummaryKeyMatcher()
    : baseline_root_(/*parent=*/nullptr, "<root>"),
      target_root_(/*parent=*/nullptr, "<root>") {}

std::optional<AbsoluteScopedTensorKey>
OriginalTensorSummaryKeyMatcher::FindMatchingKey(
    AbsoluteScopedTensorKey input_key, ComparisonVariant input_key_variant) {
  TrieNode* root = input_key_variant == ComparisonVariant::kBaseline
                       ? &baseline_root_
                       : &target_root_;
  // `in_curr` tracks our position in the input trie.
  TrieNode* in_curr = root;
  // `out_curr` tracks the position in the output trie corresponding to
  // `in_curr->parent->match`.
  TrieNode* out_curr = root->match;

  if (!out_curr) {
    return std::nullopt;
  }

  AbsoluteScopedTensorKey matched_key;
  // Combine scope instructions and tensor key instruction into a single list of
  // path segments for easier iteration. Each pair is {instruction_name,
  // iteration_index}.
  absl::Span<const ScopeInstruction> key_scopes = input_key.scope_instructions;
  std::vector<std::pair<std::string, int64_t>> key_parts;
  key_parts.reserve(key_scopes.size());
  for (const auto& s : key_scopes) {
    key_parts.push_back({s.instruction_name, s.iteration_index});
  }
  key_parts.push_back({input_key.tensor_key.instruction_name, 0});

  for (size_t i = 0; i < key_parts.size(); ++i) {
    const auto& key_part = key_parts[i];
    // Find the child in the input trie corresponding to the current instruction
    // name.
    auto it = in_curr->children.find(key_part.first);
    if (it == in_curr->children.end() || it->second->match == nullptr) {
      // If the instruction or its match doesn't exist, we can't find a
      // corresponding key.
      return std::nullopt;
    }
    // Move to the child in the input trie and find its match in the output
    // trie.
    in_curr = it->second.get();
    TrieNode* next_out = in_curr->match;

    // The path between out_curr and next_out in the output trie might contain
    // multiple nodes if the structure of the two tries differs (e.g., due to
    // inlining one or more calls on one side but not the other). We need to add
    // all nodes on this path to the matched key. We collect the path by
    // traversing upwards from next_out until we hit out_curr.
    std::vector<TrieNode*> path;
    TrieNode* p = next_out;
    while (p != out_curr && p != nullptr) {
      path.push_back(p);
      p = p->parent;
    }
    if (p == nullptr) {
      // This indicates an inconsistency in matching or trie structure, as
      // next_out should be a descendant of out_curr.
      return std::nullopt;
    }

    // Add the nodes from the path to matched_key in reverse order (root to
    // leaf).
    for (auto rit = path.rbegin(); rit != path.rend(); ++rit) {
      TrieNode* node = *rit;
      // Check if this node is the end of the path segment (the node that
      // directly corresponds to `in_curr`).
      bool is_leaf_node_in_path = (node == next_out);
      // Check if we are processing the last segment of the key (the tensor
      // instruction).
      bool processing_leaf_part = (i == key_parts.size() - 1);

      if (processing_leaf_part && is_leaf_node_in_path) {
        // If this is the last segment and the end of the path, this node
        // corresponds to the tensor-producing instruction.
        matched_key.tensor_key.instruction_name = node->instruction_name;
        auto tensor_it =
            in_curr->tensors.find(input_key.tensor_key.shape_index);
        if (tensor_it == in_curr->tensors.end() ||
            tensor_it->second.match == nullptr) {
          return std::nullopt;
        }
        matched_key.tensor_key.shape_index =
            tensor_it->second.match->shape_index;
      } else {
        // Otherwise, this node is part of the scope.
        // The iteration index from the input key only applies to the leaf node
        // of this path segment. Intermediate nodes in the path (if any)
        // don't have a corresponding iteration index from the input key part.
        int64_t iter_idx = 0;
        if (is_leaf_node_in_path) {
          iter_idx = key_part.second;
        }
        matched_key.scope_instructions.push_back(
            ScopeInstruction::Create(node->instruction_name, iter_idx));
      }
    }
    // The new "parent" in the output trie for the next iteration is the node
    // we just matched.
    out_curr = next_out;
  }

  return matched_key;
}

absl::StatusOr<std::shared_ptr<OriginalTensorSummaryKeyMatcher>>
OriginalTensorSummaryKeyMatcher::Create(
    std::shared_ptr<
        const BidirectionalMap<std::string, std::string, std::monostate>>
        hlo_diff_bimap,
    absl::Span<const AbsoluteScopedTensorKey> baseline_keys,
    absl::Span<const AbsoluteScopedTensorKey> target_keys) {
  auto matcher = std::shared_ptr<OriginalTensorSummaryKeyMatcher>(
      new OriginalTensorSummaryKeyMatcher());
  for (const auto& key : baseline_keys) {
    InsertKey(&matcher->baseline_root_, key, /*dimensions=*/{});
  }
  for (const auto& key : target_keys) {
    InsertKey(&matcher->target_root_, key, /*dimensions=*/{});
  }
  MatchTrees(&matcher->baseline_root_, &matcher->target_root_, *hlo_diff_bimap);
  return matcher;
}

absl::StatusOr<std::shared_ptr<OriginalTensorSummaryKeyMatcher>>
OriginalTensorSummaryKeyMatcher::Create(
    std::shared_ptr<
        const BidirectionalMap<std::string, std::string, std::monostate>>
        hlo_diff_bimap,
    absl::string_view baseline_recovered_tensor_summaries_file,
    absl::string_view target_recovered_tensor_summaries_file) {
  auto matcher = std::shared_ptr<OriginalTensorSummaryKeyMatcher>(
      new OriginalTensorSummaryKeyMatcher());
  ABSL_RETURN_IF_ERROR(ReadKeys(baseline_recovered_tensor_summaries_file,
                           &matcher->baseline_root_));
  ABSL_RETURN_IF_ERROR(
      ReadKeys(target_recovered_tensor_summaries_file, &matcher->target_root_));
  MatchTrees(&matcher->baseline_root_, &matcher->target_root_, *hlo_diff_bimap);
  return matcher;
}

}  // namespace xla::numerics::comparison
