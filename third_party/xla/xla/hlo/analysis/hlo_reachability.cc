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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

HloReachabilityMap::HloReachabilityMap(
    absl::Span<const HloInstruction* const> instructions)
    : bits_per_bitset_(instructions.size()),
      words_per_bitset_((bits_per_bitset_ + BitSet::kBits - 1) / BitSet::kBits),
      total_words_((instructions.size() + 1 /*for tmp_bit_set_*/) *
                   words_per_bitset_) {
  int row = 0;
  int total_rows = instructions.size() + 1;  // for tmp_bit_set_
  while (row < total_rows) {
    const int rows_to_allocate = std::min(kRowsPerAllocation, total_rows - row);
    size_t words_to_allocate = rows_to_allocate * words_per_bitset_;
    // make_unique initializes the array of words to 0
    bit_storage_.push_back(std::make_unique<BitSet::Word[]>(words_to_allocate));
    row += rows_to_allocate;
  }

  tmp_bit_set_ = BitSetFromIndex(instructions.size());
  indices_.reserve(instructions.size());
  for (size_t i = 0; i < instructions.size(); ++i) {
    BitSetFromIndex(i).Set(i);  // Instructions are reachable from themselves.
    indices_[GetKey(instructions[i])] = i;
  }
}

bool HloReachabilityMap::SetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  Index index = GetIndex(instruction);
  BitSet bit_set = BitSetFromIndex(index);
  tmp_bit_set_.CopyBitSet(bit_set);
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
  BitSet bit_set = BitSetFromIndex(index);
  // If instruction is part of inputs, don't reset the bit-set.
  if (!absl::c_linear_search(input_indices, index)) {
    bit_set.SetToZero();
  }
  bit_set.Set(index);
  for (Index input_index : input_indices) {
    if (input_index != index) {
      bit_set |= BitSetFromIndex(input_index);
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
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();
  auto result = std::make_unique<HloReachabilityMap>(instructions);

  auto get_bit_set = [&](const HloInstruction* instruction) -> BitSet {
    return result->BitSetFromIndex(result->GetIndex(instruction));
  };

  for (const HloInstruction* instruction : instructions) {
    BitSet bit_set = get_bit_set(instruction);

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
  }
  return result;
}

void HloReachabilityMap::UpdateReachabilityThroughInstruction(
    const HloInstruction* instruction) {
  std::queue<const HloInstruction*> worklist;
  worklist.push(instruction);

  std::vector<HloInstruction*> inputs;

  // Keep track of the number of times an instruction is in the worklist and
  // only process it only if it is the last occurrence. Note that this might
  // still mean that an instruction is processed multiple times.
  absl::flat_hash_map<const HloInstruction*, int64_t> in_worklist;

  while (!worklist.empty()) {
    const HloInstruction* item = worklist.front();
    worklist.pop();
    --in_worklist[item];
    if (in_worklist[item] > 0) {
      continue;
    }

    inputs.assign(item->operands().begin(), item->operands().end());
    inputs.insert(inputs.end(), item->control_predecessors().begin(),
                  item->control_predecessors().end());

    if (SetReachabilityToUnion(inputs, item)) {
      // Add immediate successors to worklist.
      for (const HloInstruction* user : item->users()) {
        worklist.push(user);
        ++in_worklist[user];
      }
      for (const HloInstruction* succ : item->control_successors()) {
        worklist.push(succ);
        ++in_worklist[succ];
      }
    }
  }
}

}  // namespace xla
