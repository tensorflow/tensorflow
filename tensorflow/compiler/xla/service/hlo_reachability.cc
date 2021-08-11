/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_reachability.h"

#include <queue>

#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {

HloReachabilityMap::HloReachabilityMap(
    absl::Span<const HloInstruction* const> instructions)
    : size_(instructions.size()) {
  bit_vectors_.reserve(size_);
  for (const HloInstruction* hlo : instructions) {
    indices_[GetKey(hlo)] = bit_vectors_.size();
    bit_vectors_.emplace_back(size_);
  }
  CHECK_EQ(size_, indices_.size());  // instructions should be unique
}

bool HloReachabilityMap::SetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  Index index = GetIndex(instruction);
  BitVector& bit_vector = GetBitVector(index);
  tmp_bit_vector_ = bit_vector;
  SetReachabilityToUnionHelper(inputs, index);
  return bit_vector != tmp_bit_vector_;
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  Index index = GetIndex(instruction);
  SetReachabilityToUnionHelper(inputs, index);
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
  BitVector& bit_vector = GetBitVector(index);
  // If instruction is part of inputs, don't reset the bit_vector.
  if (!absl::c_linear_search(input_indices, index)) {
    bit_vector.SetToZero();
  }
  bit_vector.Set(index.v);
  for (Index input_index : input_indices) {
    if (input_index != index) {
      bit_vector.OrWith(GetBitVector(input_index));
    }
  }
}

void HloReachabilityMap::Replace(const HloInstruction* original,
                                 const HloInstruction* replacement) {
  if (GetKey(original) == GetKey(replacement)) {
    return;
  }
  indices_[GetKey(replacement)] = GetIndex(original).v;
  indices_.erase(GetKey(original));
}

void HloReachabilityMap::SetReachable(Index a, Index b) {
  GetBitVector(b).Set(a.v);
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::BuildWithRestrictions(
    const HloComputation* computation,
    absl::FunctionRef<void(const HloInstruction*,
                           std::vector<HloInstruction*>*)>
        add_dependencies) {
  const auto& all = computation->MakeInstructionPostOrder();
  auto result = absl::make_unique<HloReachabilityMap>(all);

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
  const auto& all = computation->MakeInstructionPostOrder();
  auto result = absl::make_unique<HloReachabilityMap>(all);
  auto channel_group = computation->ComputeChannelDependencies();

  std::vector<HloInstruction*> inputs;

  const auto add_input = [&channel_group, &inputs](HloInstruction* input) {
    inputs.push_back(input);
    if ((input->opcode() == HloOpcode::kAllReduce ||
         input->opcode() == HloOpcode::kReduceScatter) &&
        input->channel_id()) {
      auto it = channel_group.find(*input->channel_id());
      if (it != channel_group.end()) {
        inputs.insert(inputs.end(), it->second.begin(), it->second.end());
      }
    }
  };

  const auto add_dependencies = [&add_input](const HloInstruction* hlo) {
    for (HloInstruction* operand : hlo->operands()) {
      add_input(operand);
    }
    for (HloInstruction* predecessor : hlo->control_predecessors()) {
      add_input(predecessor);
    }
  };

  for (const HloInstruction* hlo : all) {
    inputs.clear();
    add_dependencies(hlo);

    switch (hlo->opcode()) {
      case HloOpcode::kRecvDone: {
        auto it = channel_group.find(*hlo->channel_id());
        if (it != channel_group.end()) {
          for (HloInstruction* channel : it->second) {
            if (channel->opcode() == HloOpcode::kSend) {
              add_input(channel);
            }
          }
        }
        break;
      }
      case HloOpcode::kAllReduce:
      case HloOpcode::kReduceScatter: {
        auto channel_id = hlo->channel_id();
        if (channel_id) {
          auto it = channel_group.find(channel_id.value());
          if (it != channel_group.end()) {
            for (HloInstruction* all_reduce : it->second) {
              add_dependencies(all_reduce);
            }
          }
        }
        break;
      }
      default:
        break;
    }

    result->FastSetReachabilityToUnion(inputs, hlo);
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
