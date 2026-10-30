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

#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"

#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_bfs.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/proto/diff_result.pb.h"

namespace xla {
namespace hlo_diff {
namespace {

bool IsChangedInstruction(const HloInstructionNode* left_node,
                          const HloInstructionNode* right_node) {
  return left_node->props.canonical_fingerprint !=
         right_node->props.canonical_fingerprint;
}

// Returns true if the order of the operands of `node` doesn't matter, e.g. for
// add(a, b) vs. add(b, a).
bool IsOperandOrderInsensitive(const HloInstructionNode* node) {
  if (node->is_root || node->instruction == nullptr) {
    return false;
  }
  return node->children.size() == 2 &&
         HloOpcodeIsBinaryCommutative(node->instruction->opcode());
}

// Returns true if the mapping was made by a similarity-based matcher, which can
// pick the wrong one of several look-alike candidates.
bool IsHeuristicMapping(
    const std::optional<HloInstructionNodeMappingProps>& props) {
  if (!props.has_value()) {
    return false;
  }
  switch (props->matcher_type) {
    case MatcherType::kGreedyLimitedCandidatesBottomUpMatcher:
    case MatcherType::kStrictGreedyTopDownMatcher:
    case MatcherType::kGreedyTopDownMatcher:
      return true;
    default:
      return false;
  }
}

// An operand slot of a node: a user and the operand index of the node in it,
// or -1 if the user's operand order doesn't matter.
using OperandSlot = std::pair<const HloInstructionNode*, int>;

// Returns the operand slots of `node`. Users are expressed as nodes of the
// right graph: left users are translated through the mapping. Unmatched users
// are skipped, as they are already reported as unmatched.
std::vector<OperandSlot> GetOperandSlots(const HloInstructionNode* node,
                                         bool is_left,
                                         const HloGumgraphMappings& mappings) {
  std::vector<OperandSlot> slots;
  for (const HloInstructionNode* user : node->parents) {
    const HloInstructionNode* right_user = user;
    if (is_left) {
      std::optional<const HloInstructionNode*> mapped =
          mappings.left_to_right_instruction_map.GetRight(user);
      if (!mapped.has_value()) {
        continue;
      }
      right_user = *mapped;
    } else if (!mappings.InstructionMapContainsRight(user)) {
      continue;
    }
    const bool order_insensitive = IsOperandOrderInsensitive(user);
    for (int i = 0; i < user->children.size(); ++i) {
      if (user->children[i] == node) {
        slots.push_back({right_user, order_insensitive ? -1 : i});
      }
    }
  }
  absl::c_sort(slots);
  return slots;
}

// Returns true if `other`, the node at an operand slot that `node` occupies on
// one side only, explains the difference: it is unmatched (already reported),
// it computes the same value as `node`, or it looks like `node` and one of the
// mappings is heuristic (the matcher likely picked the wrong one of two
// look-alikes). `node` and `other` are on different sides.
bool IsSlotDifferenceExplained(const HloInstructionNode* node,
                               const HloInstructionNode* other,
                               bool node_is_left, bool is_permutation,
                               const HloGumgraphMappings& mappings) {
  // `other` is unmatched and already reported as such, e.g. for sine and
  // cosine in sine(x) - cosine(x) vs. sine(x) - tanh(x), cosine isn't moved
  // since tanh is reported as unmatched.
  const bool other_matched = node_is_left
                                 ? mappings.InstructionMapContainsRight(other)
                                 : mappings.InstructionMapContainsLeft(other);
  if (!other_matched) {
    return true;
  }
  // `node` and `other` compute the same value, i.e. identical subgraphs were
  // cross-matched, e.g. two identical sine(x) whose mappings are swapped.
  if (node->props.canonical_subgraph_fingerprint ==
      other->props.canonical_subgraph_fingerprint) {
    return true;
  }
  // `node` and `other` look alike but compute different values. A permutation
  // of the same user's operands is a real swap, e.g. sine(x) - sine(y) vs.
  // sine(y) - sine(x).
  if (is_permutation ||
      node->props.canonical_fingerprint != other->props.canonical_fingerprint) {
    return false;
  }
  // A similarity-based matcher may have picked the wrong one of two
  // look-alikes, e.g. sine(x) and sine(y) used by different users.
  const auto& map = mappings.left_to_right_instruction_map;
  return node_is_left ? IsHeuristicMapping(map.GetPropsByLeft(node)) ||
                            IsHeuristicMapping(map.GetPropsByRight(other))
                      : IsHeuristicMapping(map.GetPropsByRight(node)) ||
                            IsHeuristicMapping(map.GetPropsByLeft(other));
}

// Returns true if `user` (a node of the other side than `node`) has `node`'s
// counterpart among its operands, i.e. the slot difference is a permutation of
// the user's operands.
bool HasOperand(const HloInstructionNode* user,
                const HloInstructionNode* operand) {
  return absl::c_linear_search(user->children, operand);
}

// Returns true if the matched nodes `left_node` and `right_node` are used
// differently: they are operands of users that don't correspond, or at
// different operand positions of corresponding users, e.g. sine and cosine in
// sine(x) - cosine(x) vs. cosine(x) - sine(x).
bool IsMovedInstruction(const HloInstructionNode* left_node,
                        const HloInstructionNode* right_node,
                        const HloGumgraphMappings& mappings) {
  std::vector<OperandSlot> left_slots =
      GetOperandSlots(left_node, /*is_left=*/true, mappings);
  std::vector<OperandSlot> right_slots =
      GetOperandSlots(right_node, /*is_left=*/false, mappings);
  if (left_slots == right_slots) {
    return false;
  }
  std::vector<OperandSlot> left_only, right_only;
  absl::c_set_difference(left_slots, right_slots,
                         std::back_inserter(left_only));
  absl::c_set_difference(right_slots, left_slots,
                         std::back_inserter(right_only));
  // Left node is at a slot of a right user where the right node isn't.
  for (const auto& [right_user, index] : left_only) {
    // Commutative users (index -1) are conservatively treated as moved when
    // the multisets of operand slots differ: we don't try to explain the
    // difference by examining all their operands. An index out of range means
    // that the user's operand count changed.
    if (index < 0 || index >= right_user->children.size()) {
      return true;
    }
    if (!IsSlotDifferenceExplained(
            left_node, right_user->children[index], /*node_is_left=*/true,
            /*is_permutation=*/HasOperand(right_user, right_node), mappings)) {
      return true;
    }
  }
  // Right node is at a slot of a right user where the left node isn't.
  for (const auto& [right_user, index] : right_only) {
    std::optional<const HloInstructionNode*> left_user =
        mappings.left_to_right_instruction_map.GetLeft(right_user);
    // As above, commutative users and changed operand counts are treated as
    // moved.
    if (index < 0 || !left_user.has_value() ||
        index >= (*left_user)->children.size()) {
      return true;
    }
    if (!IsSlotDifferenceExplained(
            right_node, (*left_user)->children[index], /*node_is_left=*/false,
            /*is_permutation=*/HasOperand(*left_user, left_node), mappings)) {
      return true;
    }
  }
  return false;
}

}  // namespace

void DiffResult::AddUnchangedInstruction(const HloInstruction* left,
                                         const HloInstruction* right) {
  unchanged_instructions[left] = right;
  left_diff_codes[left] = DiffType::kUnchanged;
  right_diff_codes[right] = DiffType::kUnchanged;
}

void DiffResult::AddChangedInstruction(const HloInstruction* left,
                                       const HloInstruction* right) {
  changed_instructions[left] = right;
  left_diff_codes[left] = DiffType::kChanged;
  right_diff_codes[right] = DiffType::kChanged;
}

void DiffResult::AddMovedInstruction(const HloInstruction* left,
                                     const HloInstruction* right) {
  moved_instructions[left] = right;
  left_diff_codes[left] = DiffType::kMoved;
  right_diff_codes[right] = DiffType::kMoved;
}

void DiffResult::AddUnmatchedInstruction(const HloInstruction* left,
                                         const HloInstruction* right) {
  if (left != nullptr) {
    left_module_unmatched_instructions.insert(left);
    left_diff_codes[left] = DiffType::kUnmatched;
  }
  if (right != nullptr) {
    right_module_unmatched_instructions.insert(right);
    right_diff_codes[right] = DiffType::kUnmatched;
  }
}

std::unique_ptr<const DiffResult> ConstructDiffResult(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const HloGumgraphMappings& mappings) {
  const std::vector<const HloInstructionNode*> left_all_nodes =
      GetAllNodesInBfsOrder(left_graph.GetRoot(),
                            BfsTraversalDirection::kForward,
                            left_graph.GetNodeCount());
  const std::vector<const HloInstructionNode*> right_all_nodes =
      GetAllNodesInBfsOrder(right_graph.GetRoot(),
                            BfsTraversalDirection::kForward,
                            right_graph.GetNodeCount());
  auto diff_result = std::make_unique<DiffResult>();
  for (const HloInstructionNode* left_node : left_all_nodes) {
    if (left_node->is_root) {
      continue;
    }

    diff_result->node_props_left.insert(
        {left_node->instruction, left_node->props});

    // The node is unmatched
    if (!mappings.InstructionMapContainsLeft(left_node)) {
      diff_result->AddUnmatchedInstruction(left_node->instruction, nullptr);
      continue;
    }

    // The node is matched
    const HloInstructionNode* right_node =
        *mappings.left_to_right_instruction_map.GetRight(left_node);
    const std::optional<HloInstructionNodeMappingProps> mapping_props =
        mappings.left_to_right_instruction_map.GetPropsByLeft(left_node);

    // Fill in matcher debug info.
    diff_result->map_by[std::make_pair(left_node->instruction,
                                       right_node->instruction)] =
        mapping_props->matcher_type;
    diff_result->matcher_debug_info[std::make_pair(left_node->instruction,
                                                   right_node->instruction)] =
        mapping_props->matcher_debug_info;

    if (IsChangedInstruction(left_node, right_node)) {
      diff_result->AddChangedInstruction(left_node->instruction,
                                         right_node->instruction);
      continue;
    }
    if (IsMovedInstruction(left_node, right_node, mappings)) {
      diff_result->AddMovedInstruction(left_node->instruction,
                                       right_node->instruction);
      continue;
    }
    diff_result->AddUnchangedInstruction(left_node->instruction,
                                         right_node->instruction);
  }

  for (const HloInstructionNode* right_node : right_all_nodes) {
    if (right_node->is_root) {
      continue;
    }
    diff_result->node_props_right.insert(
        {right_node->instruction, right_node->props});
    if (!mappings.InstructionMapContainsRight(right_node)) {
      diff_result->AddUnmatchedInstruction(nullptr, right_node->instruction);
    }
  }

  return diff_result;
}

DiffResultProto DiffResult::ToProto() const {
  DiffResultProto proto;
  // The order of the instructions in the proto is unspecified.
  // NOLINTBEGIN(*-custom-deterministic-iteration-order)
  for (const auto& [left_instruction, right_instruction] :
       unchanged_instructions) {
    MatchedInstructionPairProto* pair = proto.add_unchanged_instructions();
    pair->set_left(left_instruction->name());
    pair->set_right(right_instruction->name());
  }
  for (const auto& [left_instruction, right_instruction] :
       changed_instructions) {
    MatchedInstructionPairProto* pair = proto.add_changed_instructions();
    pair->set_left(left_instruction->name());
    pair->set_right(right_instruction->name());
  }
  for (const auto& [left_instruction, right_instruction] : moved_instructions) {
    MatchedInstructionPairProto* pair = proto.add_moved_instructions();
    pair->set_left(left_instruction->name());
    pair->set_right(right_instruction->name());
  }
  for (const HloInstruction* instruction : left_module_unmatched_instructions) {
    proto.add_left_unmatched_instructions(std::string(instruction->name()));
  }
  for (const HloInstruction* instruction :
       right_module_unmatched_instructions) {
    proto.add_right_unmatched_instructions(std::string(instruction->name()));
  }
  // NOLINTEND(*-custom-deterministic-iteration-order)
  return proto;
}

DiffResult DiffResult::FromProto(const DiffResultProto& proto,
                                 const HloModule& left_module,
                                 const HloModule& right_module) {
  // Get instructions from modules.
  absl::flat_hash_map<std::string, const HloInstruction*>
      left_instructions_by_name;
  for (const HloComputation* computation : left_module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      left_instructions_by_name[instruction->name()] = instruction;
    }
  }
  absl::flat_hash_map<std::string, const HloInstruction*>
      right_instructions_by_name;
  for (const HloComputation* computation : right_module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      right_instructions_by_name[instruction->name()] = instruction;
    }
  }

  DiffResult diff_result;
  for (const MatchedInstructionPairProto& pair :
       proto.unchanged_instructions()) {
    diff_result.AddUnchangedInstruction(
        left_instructions_by_name[pair.left()],
        right_instructions_by_name[pair.right()]);
  }
  for (const MatchedInstructionPairProto& pair : proto.changed_instructions()) {
    diff_result.AddChangedInstruction(left_instructions_by_name[pair.left()],
                                      right_instructions_by_name[pair.right()]);
  }
  for (const MatchedInstructionPairProto& pair : proto.moved_instructions()) {
    diff_result.AddMovedInstruction(left_instructions_by_name[pair.left()],
                                    right_instructions_by_name[pair.right()]);
  }
  for (const std::string& name : proto.left_unmatched_instructions()) {
    diff_result.AddUnmatchedInstruction(left_instructions_by_name[name],
                                        nullptr);
  }
  for (const std::string& name : proto.right_unmatched_instructions()) {
    diff_result.AddUnmatchedInstruction(nullptr,
                                        right_instructions_by_name[name]);
  }

  return diff_result;
}

void LogDiffResult(const DiffResult& diff_result) {
  LOG(INFO) << "Unmatched instructions in the left module: "
            << diff_result.left_module_unmatched_instructions.size();
  LOG(INFO) << "Unmatched instructions in the right module: "
            << diff_result.right_module_unmatched_instructions.size();
  LOG(INFO) << "Changed instructions: "
            << diff_result.changed_instructions.size();
  LOG(INFO) << "Moved instructions: " << diff_result.moved_instructions.size();
  LOG(INFO) << "Unchanged instructions: "
            << diff_result.unchanged_instructions.size();
}

}  // namespace hlo_diff
}  // namespace xla
