/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/dynamic_index_splitter.h"

#include <map>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"

namespace xla {

absl::StatusOr<bool> DynamicIndexSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  std::vector<HloComputation*> computations =
      module->MakeNonfusionComputations(execution_threads);
  for (HloComputation* computation : computations) {
    for (HloInstruction* dynamic_op : computation->MakeInstructionPostOrder()) {
      switch (dynamic_op->opcode()) {
        case HloOpcode::kDynamicSlice:
        case HloOpcode::kDynamicUpdateSlice:
          break;
        default:
          continue;
      }
      auto parent = dynamic_op->parent();
      bool is_update = dynamic_op->opcode() == HloOpcode::kDynamicUpdateSlice;
      int64_t num_indices = dynamic_op->operand(0)->shape().rank();

      if (num_indices == 0) {
        // If the operand rank is 0, directly replace R0 DS/DUS with the
        // operand (for DS) or update (for DUS).
        if (is_update) {
          TF_CHECK_OK(parent->ReplaceInstruction(
              dynamic_op, dynamic_op->mutable_operand(1)));
        } else {
          TF_CHECK_OK(parent->ReplaceInstruction(
              dynamic_op, dynamic_op->mutable_operand(0)));
        }
        changed = true;
        continue;
      }

      int64_t index_operand_number =
          Cast<HloDynamicIndexInstruction>(dynamic_op)
              ->first_index_operand_number();
      auto index_operand = dynamic_op->mutable_operand(index_operand_number);
      if (ShapeUtil::IsScalar(index_operand->shape())) {
        // This DS/DUS already uses scalar indices.
        continue;
      }
      TF_RET_CHECK(index_operand->shape().rank() == 1);
      auto index_element_type = index_operand->shape().element_type();
      std::vector<HloInstruction*> index_array;
      index_array.reserve(num_indices);
      for (int64_t dim = 0; dim < num_indices; ++dim) {
        auto slice = parent->AddInstruction(HloInstruction::CreateSlice(
            ShapeUtil::MakeShape(index_element_type, {1}), index_operand, {dim},
            {dim + 1}, {1}));
        auto bitcast = parent->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(index_element_type, {}), slice));
        index_array.push_back(bitcast);
      }
      auto new_dynamic_op =
          is_update
              ? HloInstruction::CreateDynamicUpdateSlice(
                    dynamic_op->shape(), dynamic_op->mutable_operand(0),
                    dynamic_op->mutable_operand(1), absl::MakeSpan(index_array))
              : HloInstruction::CreateDynamicSlice(
                    dynamic_op->shape(), dynamic_op->mutable_operand(0),
                    absl::MakeSpan(index_array),
                    dynamic_op->dynamic_slice_sizes());
      TF_CHECK_OK(parent->ReplaceWithNewInstruction(dynamic_op,
                                                    std::move(new_dynamic_op)));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
