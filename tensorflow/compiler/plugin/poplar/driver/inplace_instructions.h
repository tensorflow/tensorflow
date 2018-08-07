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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_INPLACE_INSTRUCTIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_INPLACE_INSTRUCTIONS_H_

#include <array>
#include <map>
#include <set>
#include <vector>

namespace xla {
class HloInstruction;

namespace poplarplugin {

class InplaceInstructions {
 public:
  enum class Priority { HIGH = 0, MEDIUM, LOW };

  InplaceInstructions();

  // Adds to a particular set
  void AddTo(const InplaceInstructions::Priority priority,
             const HloInstruction* inst);

  // Removes from a particular set
  void RemoveFrom(const InplaceInstructions::Priority priority,
                  const HloInstruction* inst);

  // Checks if instruction is in a particular priority set
  bool IsIn(const InplaceInstructions::Priority priority,
            const HloInstruction* inst) const;

  // Checks if instruction is in the top priority set which has all the
  // instructions which are to be inplace
  bool IsInPlace(const HloInstruction* inst) const;

  // Move inst from one priority set to another
  void MovePriority(const InplaceInstructions::Priority from,
                    const InplaceInstructions::Priority to,
                    const HloInstruction* inst);

  // Gets the priorities in decreasing order
  std::array<const InplaceInstructions::Priority, 3> GetPriorityOrder() const;
  const std::set<const HloInstruction*>& GetPrioritySet(
      const InplaceInstructions::Priority priority);

 private:
  std::map<const InplaceInstructions::Priority, std::set<const HloInstruction*>>
      priority_instructions;
  const std::array<const InplaceInstructions::Priority, 3> priority_order{
      {InplaceInstructions::Priority::HIGH,
       InplaceInstructions::Priority::MEDIUM,
       InplaceInstructions::Priority::LOW}};
};

}  // namespace poplarplugin
}  // namespace xla

#endif