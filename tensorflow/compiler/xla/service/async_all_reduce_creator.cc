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

#include "tensorflow/compiler/xla/service/async_all_reduce_creator.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {

StatusOr<bool> AsyncAllReduceCreator::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    // Find all all-reduce ops first as we can't modify the instructions while
    // iterating through them.
    std::vector<HloInstruction*> all_reduces;
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAllReduce) {
        all_reduces.push_back(instruction);
      }
    }

    for (HloInstruction* instruction : all_reduces) {
      HloAllReduceInstruction* ar = Cast<HloAllReduceInstruction>(instruction);
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
    }
  }
  return changed;
}

}  // namespace xla
