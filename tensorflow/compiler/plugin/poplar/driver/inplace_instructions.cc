/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/inplace_instructions.h"

#include <array>
#include <map>
#include <set>
#include <vector>

namespace xla {
namespace poplarplugin {
namespace {}

InplaceInstructions::InplaceInstructions(){};

void InplaceInstructions::AddTo(const InplaceInstructions::Priority priority,
                                const HloInstruction* inst) {
  priority_instructions[priority].insert(inst);
}

void InplaceInstructions::RemoveFrom(
    const InplaceInstructions::Priority priority, const HloInstruction* inst) {
  priority_instructions[priority].erase(inst);
}

bool InplaceInstructions::IsIn(const InplaceInstructions::Priority priority,
                               const HloInstruction* inst) const {
  auto it = priority_instructions.find(priority);
  if (it != priority_instructions.end()) {
    return it->second.find(inst) != it->second.end();
  }
  return false;
}

bool InplaceInstructions::IsInPlace(const HloInstruction* inst) const {
  return IsIn(GetPriorityOrder()[0], inst);
}

void InplaceInstructions::MovePriority(const InplaceInstructions::Priority from,
                                       const InplaceInstructions::Priority to,
                                       const HloInstruction* inst) {
  if (from != to) {
    AddTo(to, inst);
    RemoveFrom(from, inst);
  }
}

std::array<const InplaceInstructions::Priority, 3>
InplaceInstructions::GetPriorityOrder() const {
  return priority_order;
}

const std::set<const HloInstruction*>& InplaceInstructions::GetPrioritySet(
    const InplaceInstructions::Priority priority) {
  return priority_instructions[priority];
}

}  // namespace poplarplugin
}  // namespace xla
