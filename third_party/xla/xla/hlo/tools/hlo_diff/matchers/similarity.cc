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

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
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
  absl::flat_hash_set<const HloInstructionNode*> left_users, right_users;
  for (const HloInstructionNode* user : left->parents) {
    left_users.insert(user);
  }
  for (const HloInstructionNode* user : right->parents) {
    right_users.insert(user);
  }

  for (const HloInstructionNode* user : left_users) {
    if (!right_users.contains(user)) {
      return false;
    }
  }
  return true;
}

}  // namespace

constexpr double kFingerprintMatchScore = 0.5;
constexpr double kUnitMatchScore = 0.1;

double NodeAttributesSimilarity(const HloInstructionNode* absl_nonnull left,
                                const HloInstructionNode* absl_nonnull right) {
  double sim_score = 0.0;

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

  return sim_score;
}

double AncestorSubGraphSimilarity(const HloInstructionNode* left,
                                  const HloInstructionNode* right,
                                  int candidate_traversal_limit,
                                  int min_bfs_distance, int left_graph_size,
                                  int right_graph_size) {
  absl::flat_hash_map<uint64_t, int> left_ancestor_fingerprints,
      right_ancestor_fingerprints;
  int left_traversal_count = 0;
  HloGumgraphBfs(
      *left,
      [&](const HloInstructionNode& node, int distance) {
        ++left_ancestor_fingerprints[node.props.fingerprint];
        ++left_traversal_count;
        return distance <= min_bfs_distance ||
               left_traversal_count < candidate_traversal_limit;
      },
      BfsTraversalDirection::kReverse, left_graph_size);
  int right_traversal_count = 0;
  HloGumgraphBfs(
      *right,
      [&](const HloInstructionNode& node, int distance) {
        ++right_ancestor_fingerprints[node.props.fingerprint];
        ++right_traversal_count;
        return distance <= min_bfs_distance ||
               right_traversal_count < candidate_traversal_limit;
      },
      BfsTraversalDirection::kReverse, right_graph_size);

  int matching_ancestors = 0;
  for (const auto& [fingerprint, count] : left_ancestor_fingerprints) {
    if (right_ancestor_fingerprints.contains(fingerprint)) {
      matching_ancestors +=
          std::min(count, right_ancestor_fingerprints[fingerprint]);
    }
  }

  return 2.0 * static_cast<double>(matching_ancestors) /
         static_cast<double>(left_traversal_count + right_traversal_count);
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

  sim_score += NodeAttributesSimilarity(left, right);

  if (AllInstructionUsersAreMatched(left, right)) {
    sim_score += kUnitMatchScore;
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
