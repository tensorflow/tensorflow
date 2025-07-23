/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/tools/hlo_diff/matchers/similarity.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_bfs.h"

namespace xla {
namespace hlo_diff {
namespace {

// Returns true if all the users of the left instruction are matched to the
// right instruction users by fingerprint.
bool AllInstructionUsersAreMatched(const HloInstructionNode* left,
                                   const HloInstructionNode* right) {
  if (left->parents.size() != right->parents.size()) {
    return false;
  }
  absl::flat_hash_set<uint64_t> left_user_fingerprints;
  for (const HloInstructionNode* user : left->parents) {
    left_user_fingerprints.insert(user->props.fingerprint);
  }

  absl::flat_hash_set<uint64_t> right_user_fingerprints;
  for (const HloInstructionNode* user : right->parents) {
    right_user_fingerprints.insert(user->props.fingerprint);
  }

  return left_user_fingerprints == right_user_fingerprints;
}

bool InSameChildPositionOfEachParent(const HloInstructionNode* left,
                                     const HloInstructionNode* right) {
  if (left->i_th_children.size() != right->i_th_children.size()) {
    return false;
  }
  for (int i = 0; i < left->i_th_children.size(); ++i) {
    if (left->i_th_children[i] != right->i_th_children[i]) {
      return false;
    }
  }
  return true;
}

bool InSameParentPositionOfEachChild(const HloInstructionNode* left,
                                     const HloInstructionNode* right) {
  if (left->i_th_parents.size() != right->i_th_parents.size()) {
    return false;
  }
  for (int i = 0; i < left->i_th_parents.size(); ++i) {
    if (left->i_th_parents[i] != right->i_th_parents[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

constexpr double kFingerprintMatchScore = 0.5;
constexpr double kUnitMatchScore = 0.1;
constexpr double kPositionMatchBonus = 0.01;

double AncestorSubGraphLcsSimilarity(const HloInstructionNode* left,
                                     const HloInstructionNode* right,
                                     int candidate_traversal_limit,
                                     int min_bfs_distance, int left_graph_size,
                                     int right_graph_size) {
  std::vector<uint64_t> left_fingerprints, right_fingerprints;

  left_fingerprints.reserve(candidate_traversal_limit);
  right_fingerprints.reserve(candidate_traversal_limit);

  int left_traversal_count = 0;
  HloGumgraphBfs(
      *left,
      [&](const HloInstructionNode& node, int distance) {
        if (node.instruction == nullptr) {
          return true;
        }
        left_fingerprints.push_back(node.props.fingerprint);
        ++left_traversal_count;
        return distance <= min_bfs_distance ||
               left_traversal_count < candidate_traversal_limit;
      },
      BfsTraversalDirection::kReverse, left_graph_size);
  int right_traversal_count = 0;
  HloGumgraphBfs(
      *right,
      [&](const HloInstructionNode& node, int distance) {
        if (node.instruction == nullptr) {
          return true;
        }
        right_fingerprints.push_back(node.props.fingerprint);
        ++right_traversal_count;
        return distance <= min_bfs_distance ||
               right_traversal_count < candidate_traversal_limit;
      },
      BfsTraversalDirection::kReverse, right_graph_size);

  const std::vector<uint64_t>* s1 = &left_fingerprints;
  const std::vector<uint64_t>* s2 = &right_fingerprints;

  // Ensure s2 is the smaller sequence to optimize space for the DP table.
  if (s1->size() < s2->size()) {
    std::swap(s1, s2);
  }

  int m = s1->size();
  int n = s2->size();

  if (n == 0) {
    return 0.0;
  }

  std::vector<int> prev(n + 1, 0);
  std::vector<int> curr(n + 1, 0);

  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      if ((*s1)[i - 1] == (*s2)[j - 1]) {
        curr[j] = prev[j - 1] + 1;
      } else {
        curr[j] = std::max(prev[j], curr[j - 1]);
      }
    }
    prev = curr;
  }

  int lcs_length = prev[n];
  double denominator =
      static_cast<double>(left_traversal_count + right_traversal_count);
  if (denominator == 0) {
    return 0.0;
  }

  return (2.0 * lcs_length) / denominator;
}

double NodePropertySimilarity(const HloInstructionNode* left,
                              const HloInstructionNode* right) {
  double sim_score = 0.0;
  if (left->instruction->shape().has_layout() &&
      right->instruction->shape().has_layout() &&
      (left->instruction->shape().layout() ==
       right->instruction->shape().layout())) {
    sim_score += kUnitMatchScore;
  }

  if (left->instruction->has_sharding() && right->instruction->has_sharding() &&
      (left->instruction->sharding() == right->instruction->sharding())) {
    sim_score += kUnitMatchScore;
  }

  if (right->props.fingerprint == left->props.fingerprint) {
    sim_score += kFingerprintMatchScore;
  }

  if (!left->instruction->metadata().op_name().empty() &&
      left->instruction->metadata().op_name() ==
          right->instruction->metadata().op_name()) {
    sim_score += kUnitMatchScore;
  }
  if (!left->instruction->metadata().source_file().empty() &&
      left->instruction->metadata().source_file() ==
          right->instruction->metadata().source_file()) {
    sim_score += kUnitMatchScore;
  }
  if (left->instruction->metadata().source_line() != 0 &&
      left->instruction->metadata().source_line() ==
          right->instruction->metadata().source_line()) {
    sim_score += kUnitMatchScore;
  }

  if (AllInstructionUsersAreMatched(left, right)) {
    sim_score += kUnitMatchScore;
  }

  if (InSameChildPositionOfEachParent(left, right)) {
    sim_score += kPositionMatchBonus;
  }

  if (InSameParentPositionOfEachChild(left, right)) {
    sim_score += kPositionMatchBonus;
  }

  return sim_score;
}

double ParamPropertySimilarity(const HloInstructionNode* left,
                               const HloInstructionNode* right) {
  double sim_score = NodePropertySimilarity(left, right);
  // A Parameter's name and parameter number are typically consistently
  // generated by the frameworks. But in some cases, the name and parameter
  // numbers might be differently generated for the same instruction, but if
  // they are both same, we can be pretty confident that they are the same
  // instruction.
  if ((left->instruction->name() == right->instruction->name()) &&
      (left->instruction->parameter_number() ==
       right->instruction->parameter_number())) {
    sim_score += kUnitMatchScore;
  }

  return sim_score;
}

double ConstantPropertySimilarity(const HloInstructionNode* left,
                                  const HloInstructionNode* right) {
  double sim_score = NodePropertySimilarity(left, right);
  // Use the canonical options as fingerprint ignore float values.
  if (left->instruction->ToString(HloPrintOptions::Canonical()) ==
      right->instruction->ToString(HloPrintOptions::Canonical())) {
    sim_score += kUnitMatchScore;
  }

  if (left->parents.size() == right->parents.size()) {
    sim_score += kUnitMatchScore;
  }
  return sim_score;
}

}  // namespace hlo_diff
}  // namespace xla
