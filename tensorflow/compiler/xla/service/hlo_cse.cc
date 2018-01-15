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

#include "tensorflow/compiler/xla/service/hlo_cse.h"

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace xla {

namespace {

// Find and combine identical constants. Constants are identical if they have
// the same type and value.
bool CombineConstants(HloComputation* computation, bool is_layout_sensitive) {
  bool changed = false;

  // Map from ShortDebugString of the layoutless shape of the constant to the
  // set of constant instructions with that shape. Layoutless shape is used to
  // bin possible common constants together to reduce number of constant
  // comparisons. If we end up having too many constant comparisons, a more
  // precise binning might have to be used.
  std::multimap<string, HloInstruction*> constants;

  auto inst_it = computation->instructions().begin();
  while (inst_it != computation->instructions().end()) {
    HloInstruction* instruction = *inst_it;

    // Advance list iterator before loop body because iterator may be
    // invalidated due to deletion.
    ++inst_it;

    if (instruction->opcode() == HloOpcode::kConstant) {
      Shape shape = instruction->shape();
      if (!is_layout_sensitive) {
        LayoutUtil::ClearLayout(&shape);
      }
      string shape_string = shape.ShortDebugString();

      // Compare against all constants with the same shape
      auto range = constants.equal_range(shape_string);
      HloInstruction* match = nullptr;
      for (auto it = range.first; it != range.second; ++it) {
        if (instruction->literal() == it->second->literal()) {
          match = it->second;
          break;
        }
      }
      if (match == nullptr) {
        constants.emplace(shape_string, instruction);
      } else {
        // Match found, replace this instruction with the one in the multimap.
        TF_CHECK_OK(instruction->ReplaceAllUsesWith(match));
        TF_CHECK_OK(computation->RemoveInstruction(instruction));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace

StatusOr<bool> HloCSE::Run(HloModule* module) {
  bool changed = false;
  const std::function<bool(const HloInstruction*, const HloInstruction*)>
      eq_instructions = std::equal_to<const HloInstruction*>();
  const std::function<bool(const HloComputation*, const HloComputation*)>
      eq_computations = std::equal_to<const HloComputation*>();
  for (auto* computation : module->computations()) {
    changed |= CombineConstants(computation, is_layout_sensitive_);

    std::list<HloInstruction*> post_order =
        computation->MakeInstructionPostOrder();
    std::set<HloInstruction*> removed_instructions;
    for (auto instruction : post_order) {
      // If the instruction has already been removed by CSE skip over it.
      if (removed_instructions.count(instruction) > 0 ||
          instruction->operand_count() == 0) {
        continue;
      }

      // An instruction is considered to be equivalent to another only if they
      // share the exact same set of operands. So to find equivalent
      // instructions, we just search among instructions which share operand(0)
      // of this instruction.
      const HloInstruction* operand = instruction->operand(0);

      tensorflow::gtl::InlinedVector<HloInstruction*, 8>
          equivalent_instructions;
      for (HloInstruction* user : operand->users()) {
        if (user != instruction &&
            user->Identical(*instruction, eq_instructions, eq_computations) &&
            (!is_layout_sensitive_ ||
             ShapeUtil::Equal(user->shape(), instruction->shape()))) {
          equivalent_instructions.push_back(user);
        }
      }

      // Replace all equivalent instructions with this instruction.
      for (HloInstruction* equivalent_instruction : equivalent_instructions) {
        TF_RETURN_IF_ERROR(
            equivalent_instruction->ReplaceAllUsesWith(instruction));
        TF_RETURN_IF_ERROR(
            computation->RemoveInstruction(equivalent_instruction));
        removed_instructions.insert(equivalent_instruction);
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
