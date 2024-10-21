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

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "llvm/ADT/BitVector.h"
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
  if (to->operand_count() == 0 && from->control_predecessors().empty()) {
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

std::unique_ptr<HloDfsReachability> HloDfsReachability::Build(
    const HloComputation* computation) {
  auto res = std::make_unique<HloDfsReachability>();

  // For instruction reachability we do not care about correct order of
  // collective operations as we only care about use-def chains.
  HloComputation::ChannelDependencies empty_channel_dependencies;
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder(empty_channel_dependencies);

  res->instruction_to_idx_.reserve(instructions.size());
  for (size_t i = 0; i < instructions.size(); ++i) {
    res->instruction_to_idx_[instructions[i]] = i;
  }

  return res;
}

}  // namespace xla
