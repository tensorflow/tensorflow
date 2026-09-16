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

#include "xla/hlo/transforms/collectives/all_gather_cse.h"

#include <cstdint>

#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_cse.h"

namespace xla {
namespace {

bool ShouldEliminateInstruction(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kAllGather) {
    return false;
  }
  VLOG(2) << "Finding raw parameter for instruction: "
          << instruction->ToString();
  const HloInstruction* current = instruction->operand(0);
  int64_t tuple_index = -1;
  while (current != nullptr) {
    if (current->opcode() == HloOpcode::kParameter) {
      VLOG(2) << "Found parameter: " << current->ToString();
      return true;
    }
    if (current->opcode() == HloOpcode::kGetTupleElement) {
      tuple_index = current->tuple_index();
      VLOG(2) << "Found get-tuple-element at index: " << tuple_index;
      current = current->operand(0);
    } else if (current->opcode() == HloOpcode::kTuple) {
      if (tuple_index >= 0 && tuple_index < current->operand_count()) {
        VLOG(2) << "Found tuple, moving to element at index: " << tuple_index;
        current = current->operand(tuple_index);
        tuple_index = -1;  // Reset tuple index
      } else {
        VLOG(2) << "Invalid tuple index: " << tuple_index;
        return false;
      }
    } else if (current->opcode() == HloOpcode::kOptimizationBarrier) {
      VLOG(2) << "Found optimization barrier, moving to its input";
      current = current->operand(0);
    } else if (current->opcode() == HloOpcode::kConvert) {
      VLOG(2) << "Found convert operation, moving to its input";
      current = current->operand(0);
    } else if (current->opcode() == HloOpcode::kAllGather) {
      // When you code motion AllGathers out of nested while loops you may end
      // up with two all gathers trying to all gather each other as they are the
      // same parameter. We check the shape of whats being all gathered is the
      // same as the all gather shape. Then it is safe to traverse.
      if (current->shape() == current->operand(0)->shape()) {
        VLOG(2) << "Found all-gather operation, moving to its input";
        current = current->operand(0);
      } else {
        VLOG(2) << "All gather of an all gather but we did not match shape. "
                << current->ToString();
        return false;
      }
    } else {
      VLOG(2) << "Unsupported instruction: " << current->ToString();
      return false;
    }
  }
  VLOG(2) << "Raw parameter not found";
  return false;
}

}  // namespace

AllGatherCSE::AllGatherCSE()
    : HloCSE(
          /*is_layout_sensitive=*/true,
          /*ignore_control_dependencies=*/false,
          /*should_eliminate_computation=*/nullptr,
          /*should_eliminate_instruction=*/&ShouldEliminateInstruction) {}

}  // namespace xla
