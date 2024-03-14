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

#include "xla/hlo/ir/hlo_dfs_reachability.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
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

  const size_t target_node_idx = instruction_to_idx_.at(from);
  const size_t dfs_root_idx = instruction_to_idx_.at(to);
  // Note that the DFS goes from the "uses" root towards the "defs", i.e. from
  // `to` node to `from` node, so the node indices are decreasing.
  if (target_node_idx > dfs_root_idx) {
    return false;
  }
  absl::flat_hash_set<const HloInstruction*> visited{to};
  std::vector<const HloInstruction*> stack{to};

  auto check_and_enqueue = [&](const HloInstruction* instruction) {
    if (instruction == from) {
      return true;
    }
    if (visited.contains(instruction)) {
      return false;
    }
    if (instruction_to_idx_.at(instruction) < target_node_idx) {
      return false;
    }
    visited.insert(instruction);
    stack.push_back(instruction);
    return false;
  };

  while (!stack.empty()) {
    const HloInstruction* instr = stack.back();
    stack.pop_back();

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

  HloComputation::ChannelDependencies channel_dependencies =
      computation->ComputeChannelDependencies();
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder(channel_dependencies);

  for (size_t i = 0; i < instructions.size(); ++i) {
    res->instruction_to_idx_[instructions[i]] = i;
  }

  return res;
}

}  // namespace xla
