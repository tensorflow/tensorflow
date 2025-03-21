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
#include <tuple>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

absl::StatusOr<bool> AllGatherCSE::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Running AllGatherCSE pass";
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    VLOG(2) << "Processing computation: " << computation->name();
    absl::flat_hash_map<std::tuple<HloInstruction*, int64_t, PrimitiveType>,
                        HloInstruction*>
        all_gather_map;  // Every region has its own all gather map
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAllGather) {
        VLOG(2) << "Found all-gather instruction: " << instruction->ToString();
        auto raw_parameter_tuple =
            FindRawParameter(instruction->mutable_operand(0));
        HloInstruction* raw_parameter = std::get<0>(raw_parameter_tuple);
        if (raw_parameter != nullptr &&
            raw_parameter->opcode() == HloOpcode::kParameter) {
          auto it = all_gather_map.find(raw_parameter_tuple);
          if (it != all_gather_map.end()) {
            VLOG(2) << "Replacing all-gather with previous result: "
                    << it->second->ToString();
            TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(it->second));
            changed = true;
          } else {
            VLOG(2) << "Storing all-gather result for future use";
            all_gather_map[raw_parameter_tuple] = instruction;
          }
        }
      }
    }
  }

  VLOG(2) << "AllGatherCSE pass complete";
  return changed;
}

std::tuple<HloInstruction*, int64_t, PrimitiveType>
AllGatherCSE::FindRawParameter(HloInstruction* instruction) {
  VLOG(2) << "Finding raw parameter for instruction: "
          << instruction->ToString();
  HloInstruction* current = instruction;
  int64_t tuple_index = -1;
  PrimitiveType dtype = instruction->shape().element_type();
  while (current != nullptr) {
    if (current->opcode() == HloOpcode::kParameter) {
      VLOG(2) << "Found parameter: " << current->ToString();
      return std::make_tuple(current, tuple_index, dtype);
    } else if (current->opcode() == HloOpcode::kGetTupleElement) {
      tuple_index = current->tuple_index();
      VLOG(2) << "Found get-tuple-element at index: " << tuple_index;
      current = current->mutable_operand(0);
    } else if (current->opcode() == HloOpcode::kTuple) {
      if (tuple_index >= 0 && tuple_index < current->operand_count()) {
        VLOG(2) << "Found tuple, moving to element at index: " << tuple_index;
        current = current->mutable_operand(tuple_index);
        tuple_index = -1;  // Reset tuple index
      } else {
        VLOG(2) << "Invalid tuple index: " << tuple_index;
        return std::make_tuple(nullptr, -1, PRIMITIVE_TYPE_INVALID);
      }
    } else if (current->opcode() == HloOpcode::kOptimizationBarrier) {
      VLOG(2) << "Found optimization barrier, moving to its input";
      current = current->mutable_operand(0);
    } else if (current->opcode() == HloOpcode::kConvert) {
      VLOG(2) << "Found convert operation, moving to its input";
      current = current->mutable_operand(0);
    } else if (current->opcode() == HloOpcode::kAllGather) {
      // When you code motion AllGathers out of nested while loops you may end
      // up with two all gathers trying to all gather each other as they are the
      // same parameter. We check the shape of whats being all gathered is the
      // same as the all gather shape. Then it is safe to traverse.
      if (current->shape() == current->mutable_operand(0)->shape()) {
        VLOG(2) << "Found all-gather operation, moving to its input";
        current = current->mutable_operand(0);
      }
      VLOG(2) << "All gather of an all gather but we did not match shape. "
              << current->ToString();
      return std::make_tuple(nullptr, -1, PRIMITIVE_TYPE_INVALID);
    } else {
      VLOG(2) << "Unsupported instruction: " << current->ToString();
      return std::make_tuple(nullptr, -1, PRIMITIVE_TYPE_INVALID);
    }
  }
  VLOG(2) << "Raw parameter not found";
  return std::make_tuple(nullptr, -1, PRIMITIVE_TYPE_INVALID);
}

}  // namespace xla
