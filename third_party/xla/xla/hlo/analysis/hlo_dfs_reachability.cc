/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/analysis/hlo_dfs_reachability.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

bool HloDfsReachability::IsPresent(const HloInstruction* instruction) const {
  return instruction_to_idx_.contains(instruction);
}

bool HloDfsReachability::IsReachable(const HloInstruction* from,
                                     const HloInstruction* to) const {
  if (from == to) {
    return true;
  }
  if (to->operand_count() == 0 && to->control_predecessors().empty()) {
    return false;
  }

  size_t target_node_idx = instruction_to_idx_.at(from);
  size_t dfs_root_idx = instruction_to_idx_.at(to);

  // Note that the DFS goes from the "uses" root towards the "defs", i.e. from
  // `to` node to `from` node, so the node indices are decreasing.
  if (dfs_root_idx < target_node_idx) {
    return false;
  }

  // We use LLVM support library here because it has stack-allocated bit vector
  // which significantly improves performance by avoiding heap allocations when
  // instructions are reachable via a short chain.
  llvm::SmallVector<const HloInstruction*> stack{to};

  // We will visit instructions in the [target_node_idx, dfs_root_idx] range, so
  // we can construct a smaller bit vector.
  llvm::BitVector visited_idxs(1 + (dfs_root_idx - target_node_idx));
  visited_idxs.set(dfs_root_idx - target_node_idx);

  auto check_and_enqueue = [&](const HloInstruction* instr) {
    if (instr == from) {
      return true;
    }
    size_t instr_idx = instruction_to_idx_.at(instr);
    if (instr_idx < target_node_idx) {
      return false;
    }
    size_t visited_idx = instr_idx - target_node_idx;
    if (visited_idxs.test(visited_idx)) {
      return false;
    }
    visited_idxs.set(visited_idx);
    stack.push_back(instr);
    return false;
  };

  while (!stack.empty()) {
    const HloInstruction* instr = stack.pop_back_val();

    if (absl::c_any_of(instr->operands(), check_and_enqueue) ||
        absl::c_any_of(instr->control_predecessors(), check_and_enqueue)) {
      return true;
    }
  }
  return false;
}

bool HloDfsReachability::IsConnected(const HloInstruction* a,
                                     const HloInstruction* b) const {
  return IsReachable(a, b) || IsReachable(b, a);
}

namespace {

using InstructionIndex = llvm::DenseMap<const HloInstruction*, size_t>;

// Returns true if no direct input of `instr` is ordered at or after `idx`. Such
// an instruction cannot be reached from the instruction at `idx`.
bool AllInputsBefore(const InstructionIndex& index, const HloInstruction* instr,
                     size_t idx) {
  auto is_before = [&](const HloInstruction* input) {
    return index.at(input) < idx;
  };
  return absl::c_all_of(instr->operands(), is_before) &&
         absl::c_all_of(instr->control_predecessors(), is_before);
}

enum class WalkDirection { kUp, kDown };

// Walks from `start` through its inputs (kUp) or its users (kDown) and marks
// what it reaches in `visited`. Bit 0 of `visited` and `is_target` stands for
// the instruction at `min_idx`, and the walk stays inside the range the bit
// vectors cover. Stops early once it reached `num_targets` instructions that
// are marked in `is_target`.
void Walk(const InstructionIndex& index, const HloInstruction* start,
          WalkDirection direction, size_t min_idx,
          const llvm::BitVector& is_target, size_t num_targets,
          llvm::BitVector& visited) {
  llvm::SmallVector<const HloInstruction*> stack{start};
  auto enqueue = [&](const HloInstruction* instr) {
    auto it = index.find(instr);
    CHECK(it != index.end())
        << instr->name() << " is missing from the reachability map";
    const size_t idx = it->second;
    if (idx < min_idx || idx - min_idx >= visited.size() ||
        visited.test(idx - min_idx)) {
      return;
    }
    visited.set(idx - min_idx);
    if (is_target.test(idx - min_idx)) {
      --num_targets;
    }
    stack.push_back(instr);
  };
  while (!stack.empty() && num_targets > 0) {
    const HloInstruction* instr = stack.pop_back_val();
    if (direction == WalkDirection::kUp) {
      absl::c_for_each(instr->operands(), enqueue);
      absl::c_for_each(instr->control_predecessors(), enqueue);
    } else {
      absl::c_for_each(instr->users(), enqueue);
      absl::c_for_each(instr->control_successors(), enqueue);
    }
  }
}

}  // namespace

absl::flat_hash_set<const HloInstruction*> HloDfsReachability::FindConnected(
    const HloInstruction* a,
    absl::Span<const HloInstruction* const> candidates) const {
  absl::flat_hash_set<const HloInstruction*> connected;
  if (candidates.size() == 1) {
    // A single pairwise query stops at the first hit, so it is cheaper than the
    // two walks below.
    if (IsConnected(a, candidates[0])) {
      connected.insert(candidates[0]);
    }
    return connected;
  }
  const size_t a_idx = instruction_to_idx_.at(a);
  // Candidates that need a walk. Only instructions ordered between `a` and
  // these candidates can lie on a path between them.
  llvm::SmallVector<std::pair<const HloInstruction*, size_t>> pending;
  size_t min_idx = a_idx;
  size_t max_idx = a_idx;
  for (const HloInstruction* candidate : candidates) {
    const size_t idx = instruction_to_idx_.at(candidate);
    if (idx == a_idx) {
      connected.insert(candidate);
      continue;
    }
    // `a` can only reach a later candidate through one of that candidate's
    // inputs, and such an input is `a` itself or ordered after it.
    if (idx > a_idx && AllInputsBefore(instruction_to_idx_, candidate, a_idx)) {
      continue;
    }
    pending.emplace_back(candidate, idx);
    min_idx = std::min(min_idx, idx);
    max_idx = std::max(max_idx, idx);
  }
  if (pending.empty()) {
    return connected;
  }
  const size_t window = 1 + (max_idx - min_idx);
  // Distinct pending candidates before and after `a`. Each walk stops once it
  // reached all of its targets.
  size_t targets_before = 0;
  size_t targets_after = 0;
  llvm::BitVector is_target(window);
  for (const auto& [candidate, idx] : pending) {
    if (!is_target.test(idx - min_idx)) {
      is_target.set(idx - min_idx);
      ++(idx < a_idx ? targets_before : targets_after);
    }
  }
  // Ancestors of `a` are ordered before it and descendants after it, so both
  // walks can share one bit vector.
  llvm::BitVector visited(window);
  visited.set(a_idx - min_idx);
  Walk(instruction_to_idx_, a, WalkDirection::kUp, min_idx, is_target,
       targets_before, visited);
  Walk(instruction_to_idx_, a, WalkDirection::kDown, min_idx, is_target,
       targets_after, visited);
  for (const auto& [candidate, idx] : pending) {
    if (visited.test(idx - min_idx)) {
      connected.insert(candidate);
    }
  }
  return connected;
}

std::unique_ptr<HloDfsReachability> HloDfsReachability::Build(
    const HloComputation* computation) {
  auto res = std::make_unique<HloDfsReachability>();
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();

  res->instruction_to_idx_.reserve(instructions.size());
  for (size_t i = 0; i < instructions.size(); ++i) {
    res->instruction_to_idx_[instructions[i]] = i;
  }

  return res;
}

void HloDfsReachability::OnInstructionReplaced(const HloInstruction* previous,
                                               const HloInstruction* now) {
  auto it = instruction_to_idx_.find(previous);
  CHECK(it != instruction_to_idx_.end());
  auto idx = it->second;
  instruction_to_idx_.erase(it);
  auto inserted = instruction_to_idx_.insert({now, idx}).second;
  CHECK(inserted);
}

}  // namespace xla
