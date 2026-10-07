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

#ifndef XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_KEY_MATCHER_H_
#define XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_KEY_MATCHER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"
#include "xla/hlo/tools/hlo_diff/utils/bidirectional_map.h"
#include "xla/shape_util.h"

namespace xla::numerics::comparison {
// Class for matching absolute tensor keys from two HLO module runs.
//
// This class is used to match absolute tensor keys from two HLO module runs.
// It uses a trie data structure to hierarchically store the keys and match
// them between the two runs. The matching process proceeds in two stages:
// 1. The HLO diff bimap is used to match nodes in the tries to start the
//    matching process.
// 2. The iteration indices and tensor dimensions are matched between
//    corresponding nodes in the tries.
//
// The matching process is greedy and terminates when no more matches can be
// found. A matching score is assigned to each node based on how many
// attributes (e.g., iterations, tensors, and children) match between the two
// nodes.
class OriginalTensorSummaryKeyMatcher {
 public:
  struct CreationMetrics {
    // The total number of tensors in the baseline module. Note that this is
    // not the number of instructions in the module. Instead, it counts arrays
    // in tuple-shaped instructions individually.
    int64_t baseline_tensor_count = 0;
    // The total number of tensors in the target module.
    int64_t target_tensor_count = 0;
    // The number of tensor pairs that are unchanged between baseline and
    // target.
    int64_t unchanged_tensor_pair_count = 0;
    // The number of tensor pairs that are changed between baseline and
    // target.
    int64_t changed_tensor_pair_count = 0;
  };
  static absl::StatusOr<std::shared_ptr<OriginalTensorSummaryKeyMatcher>>
  Create(std::shared_ptr<
             const BidirectionalMap<std::string, std::string, std::monostate>>
             hlo_diff_bimap,
         absl::Span<const AbsoluteScopedTensorKey> baseline_keys,
         absl::Span<const AbsoluteScopedTensorKey> target_keys);

  // Creates a key matcher from the HLO diff bimap and the recovered tensor
  // summaries files.
  //
  // Args:
  // hlo_diff_bimap: The bimap containing the mapping between HLO instructions
  //   in the baseline and target runs.
  // baseline_recovered_tensor_summaries_file: The Riegeli file containing
  //   RecoveredTensorSummaryProto messages for the baseline run.
  // target_recovered_tensor_summaries_file: The Riegeli file containing
  //   RecoveredTensorSummaryProto messages for the target run.
  static absl::StatusOr<std::shared_ptr<OriginalTensorSummaryKeyMatcher>>
  Create(std::shared_ptr<
             const BidirectionalMap<std::string, std::string, std::monostate>>
             hlo_diff_bimap,
         absl::string_view baseline_recovered_tensor_summaries_file,
         absl::string_view target_recovered_tensor_summaries_file);

  // Finds a matching key in the other variant of the input key variant.
  // Returns std::nullopt if no matching key is found.
  std::optional<AbsoluteScopedTensorKey> FindMatchingKey(
      AbsoluteScopedTensorKey input_key, ComparisonVariant input_key_variant);

 private:
  // Represents a specific tensor output of an HLO instruction, uniquely
  // identified by its shape index within the instruction's output tuple.
  struct TensorInfo {
    // The index identifying this tensor within the HLO instruction's output.
    // If the instruction's output is not a tuple, this will be empty.
    ShapeIndex shape_index;
    // The dimensions of the tensor.
    std::vector<int64_t> dimensions;
    // Pointer to the corresponding `TensorInfo` in the other tree (e.g.,
    // target if this is baseline), if a match has been found. Otherwise
    // `nullptr`.
    TensorInfo* match = nullptr;
  };

  // Represents a node in a trie data structure used to hierarchically store
  // AbsoluteScopedTensorKeys. Each path from the root to a node with entries
  // in its `tensors` map corresponds to one or more
  // `AbsoluteScopedTensorKey`s. This structure facilitates matching keys
  // between baseline and target runs.
  struct TrieNode {
    // Pointer to the parent node in the trie, representing the calling HLO
    // instruction or outer loop scope. This is `nullptr` for root nodes.
    TrieNode* parent = nullptr;
    // The name of the HLO instruction that this node represents (e.g.,
    // 'while.1', 'fusion.3').
    std::string instruction_name;
    // If this node represents a loop or is part of a loop scope, this map
    // tracks the iteration indices observed and their frequencies.
    // Key: iteration index ('-1' indicates a wildcard matching any iteration),
    // Value: count/frequency of observation.
    absl::flat_hash_map<int64_t, int> iterations;
    // If this node represents a tensor-producing HLO instruction (i.e., it is
    // the final instruction in a key), this map stores information about its
    // tensor outputs, keyed by ShapeIndex. For non-tensor-producing nodes
    // (e.g., call or while instructions in the scope), this map is empty.
    std::map<ShapeIndex, TensorInfo> tensors;
    // Child nodes representing HLO instructions nested within this node's
    // scope (e.g., instructions in a called computation or loop body).
    // Keyed by instruction name.
    absl::flat_hash_map<std::string, std::unique_ptr<TrieNode>> children;
    // Pointer to the corresponding `TrieNode` in the other tree (e.g., target
    // if this is baseline), if a match has been found. Otherwise `nullptr`.
    TrieNode* match = nullptr;

    explicit TrieNode(TrieNode* parent, std::string name)
        : parent(parent), instruction_name(std::move(name)) {}
  };

  OriginalTensorSummaryKeyMatcher();

  // Inserts a single key into the trie.
  static void InsertKey(TrieNode* root, const AbsoluteScopedTensorKey& key,
                        absl::Span<const int64_t> dimensions);

  // Reads all tensor keys from a Riegeli file containing
  // RecoveredTensorSummaryProto messages and inserts them into the given
  // trie. Returns an error status if file reading fails.
  static absl::Status ReadKeys(absl::string_view filename, TrieNode* root);

  // Establishes matches between nodes in the baseline and target tries.
  // This function populates the `match` field in TrieNodes by traversing
  // both tries simultaneously. The provided `hlo_diff_bimap` is used as
  // the source of truth for instruction correspondence between the baseline
  // and target.
  static void MatchTrees(
      TrieNode* baseline_root, TrieNode* target_root,
      const BidirectionalMap<std::string, std::string, std::monostate>&
          hlo_diff_bimap);

  // Checks if two tensor maps are similar based on shape indices and
  // dimensions.
  static bool AreTensorsSimilar(
      const std::map<ShapeIndex, TensorInfo>& tensors1,
      const std::map<ShapeIndex, TensorInfo>& tensors2);

  // Calculates a similarity score between two trie nodes based on their type
  // (tensor-producing or scope), iterations, tensors, and number of children.
  static int CalculateSimilarity(const TrieNode* node1, const TrieNode* node2);

  // Marks two nodes as matched, matches their tensors, and adds them to the
  // processing queue.
  static void MatchAndQueue(
      TrieNode* baseline_node, TrieNode* target_node,
      std::vector<std::pair<TrieNode*, TrieNode*>>& to_process);

  TrieNode baseline_root_;
  TrieNode target_root_;
};
}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_KEY_MATCHER_H_
