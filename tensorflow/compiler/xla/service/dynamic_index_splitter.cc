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

#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"

#include <map>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

StatusOr<bool> DynamicIndexSplitter::Run(HloModule* module) {
  bool changed = false;

  std::vector<HloComputation*> computations =
      module->MakeNonfusionComputations();
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
      int64 index_operand_number = Cast<HloDynamicIndexInstruction>(dynamic_op)
                                       ->first_index_operand_number();
      auto index_operand = dynamic_op->mutable_operand(index_operand_number);
      if (ShapeUtil::IsScalar(index_operand->shape())) {
        // This DS/DUS already uses scalar indices.
        continue;
      }
      TF_RET_CHECK(index_operand->shape().rank() == 1);
      int64 num_indices = index_operand->shape().dimensions(0);
      if (num_indices == 0) {
        // If the operand dimension is 0, directly replace R0 DS/DUS with the
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
      auto index_element_type = index_operand->shape().element_type();
      std::vector<HloInstruction*> index_array;
      for (int64 dim = 0; dim < num_indices; ++dim) {
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
