/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/async_collective_creator.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {

StatusOr<bool> AsyncCollectiveCreator::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    // Find all all-reduce ops first as we can't modify the instructions while
    // iterating through them.
    std::vector<HloInstruction*> supported_collectives;
    for (HloInstruction* instruction : computation->instructions()) {
      if ((convert_all_reduce_ &&
           instruction->opcode() == HloOpcode::kAllReduce) ||
          (convert_all_gather_ &&
           instruction->opcode() == HloOpcode::kAllGather)) {
        supported_collectives.push_back(instruction);
      }
    }

    for (HloInstruction* instruction : supported_collectives) {
      if (HloAllReduceInstruction* ar =
              DynCast<HloAllReduceInstruction>(instruction)) {
        Shape shape = ShapeUtil::MakeTupleShape({ar->shape(), ar->shape()});
        HloInstruction* start =
            computation->AddInstruction(HloInstruction::CreateAllReduceStart(
                shape, ar->operands(), ar->to_apply(), ar->replica_groups(),
                ar->constrain_layout(), ar->channel_id(),
                ar->use_global_device_ids()));
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            computation->ReplaceWithNewInstruction(
                ar, HloInstruction::CreateUnary(
                        ar->shape(), HloOpcode::kAllReduceDone, start)),
            "replacing ", ar->ToShortString());
        changed = true;
        continue;
      }
      if (HloAllGatherInstruction* ag =
              DynCast<HloAllGatherInstruction>(instruction)) {
        std::vector<Shape> operand_shapes;
        operand_shapes.reserve(ag->operand_count());
        for (const HloInstruction* op : ag->operands()) {
          operand_shapes.push_back(op->shape());
        }
        Shape shape = ShapeUtil::MakeTupleShape(
            {ag->operand_count() > 1 ? ShapeUtil::MakeTupleShape(operand_shapes)
                                     : operand_shapes[0],
             ag->shape()});
        HloInstruction* start =
            computation->AddInstruction(HloInstruction::CreateAllGatherStart(
                shape, ag->operands(), ag->all_gather_dimension(),
                ag->replica_groups(), ag->constrain_layout(), ag->channel_id(),
                ag->use_global_device_ids()));
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            computation->ReplaceWithNewInstruction(
                ag, HloInstruction::CreateUnary(
                        ag->shape(), HloOpcode::kAllGatherDone, start)),
            "replacing ", ag->ToShortString());
        changed = true;
        continue;
      }
    }
  }
  return changed;
}

}  // namespace xla
