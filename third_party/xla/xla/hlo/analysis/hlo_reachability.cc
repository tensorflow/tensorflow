/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/analysis/hlo_reachability.h"

#include <cstddef>
#include <memory>
#include <queue>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

HloReachabilityMap::HloReachabilityMap(
    absl::Span<const HloInstruction* const> instructions)
    : bit_sets_(instructions.size(), BitSet(instructions.size())) {
  indices_.reserve(instructions.size());
  for (size_t i = 0; i < instructions.size(); ++i) {
    bit_sets_[i].Set(i);  // Instructions are reachable from themselves.
    indices_[GetKey(instructions[i])] = i;
  }
}

bool HloReachabilityMap::SetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  Index index = GetIndex(instruction);
  BitSet& bit_set = bit_sets_[index];
  tmp_bit_set_ = bit_set;
  SetReachabilityToUnionHelper(inputs, index);
  return bit_set != tmp_bit_set_;
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  SetReachabilityToUnionHelper(inputs, GetIndex(instruction));
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const Index> input_indices, Index index) {
  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const HloInstruction* const> inputs, Index index) {
  absl::InlinedVector<Index, 16> input_indices;
  input_indices.reserve(inputs.size());
  for (const HloInstruction* input : inputs) {
    input_indices.push_back(GetIndex(input));
  }
  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const Index> input_indices, Index index) {
  BitSet& bit_set = bit_sets_[index];
  // If instruction is part of inputs, don't reset the bit-set.
  if (!absl::c_linear_search(input_indices, index)) {
    bit_set.SetToZero();
  }
  bit_set.Set(index);
  for (Index input_index : input_indices) {
    if (input_index != index) {
      bit_set |= bit_sets_[input_index];
    }
  }
}

void HloReachabilityMap::Replace(const HloInstruction* original,
                                 const HloInstruction* replacement) {
  if (GetKey(original) != GetKey(replacement)) {
    indices_[GetKey(replacement)] = GetIndex(original);
    indices_.erase(GetKey(original));
  }
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::BuildWithRestrictions(
    const HloComputation* computation,
    absl::FunctionRef<void(const HloInstruction*,
                           std::vector<HloInstruction*>*)>
        add_dependencies) {
  const auto& all = computation->MakeInstructionPostOrder();
  auto result = std::make_unique<HloReachabilityMap>(all);

  std::vector<HloInstruction*> inputs;
  for (const HloInstruction* hlo : all) {
    inputs.clear();
    add_dependencies(hlo, &inputs);
    result->FastSetReachabilityToUnion(inputs, hlo);
  }
  return result;
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::Build(
    const HloComputation* computation) {
  HloComputation::ChannelDependencies channel_dependencies =
      computation->ComputeChannelDependencies();
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder(channel_dependencies);
  auto result = std::make_unique<HloReachabilityMap>(instructions);

  auto get_bit_set = [&](const HloInstruction* instruction) -> BitSet& {
    return result->bit_sets_[result->GetIndex(instruction)];
  };

  for (const HloInstruction* instruction : instructions) {
    BitSet& bit_set = get_bit_set(instruction);

    auto add_dependencies = [&](const HloInstruction* instruction) {
      for (const HloInstruction* operand : instruction->operands()) {
        bit_set |= get_bit_set(operand);
      }
      for (const HloInstruction* predecessor :
           instruction->control_predecessors()) {
        bit_set |= get_bit_set(predecessor);
      }
    };

    add_dependencies(instruction);

    // If an instruction has channel depencencies, they are also reachable.
    auto it = channel_dependencies.find(instruction);
    if (it != channel_dependencies.end()) {
      absl::c_for_each(it->second, add_dependencies);
    }
  }
  return result;
}

void HloReachabilityMap::UpdateReachabilityThroughInstruction(
    const HloInstruction* instruction) {
  std::queue<const HloInstruction*> worklist;
  worklist.push(instruction);

  std::vector<HloInstruction*> inputs;

  while (!worklist.empty()) {
    const HloInstruction* item = worklist.front();
    worklist.pop();

    inputs.assign(item->operands().begin(), item->operands().end());
    inputs.insert(inputs.end(), item->control_predecessors().begin(),
                  item->control_predecessors().end());

    if (SetReachabilityToUnion(inputs, item)) {
      // Add immediate successors to worklist.
      for (const HloInstruction* user : item->users()) {
        worklist.push(user);
      }
      for (const HloInstruction* succ : item->control_successors()) {
        worklist.push(succ);
      }
    }
  }
}

}  // namespace xla
