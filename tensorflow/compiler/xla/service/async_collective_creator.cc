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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {

StatusOr<bool> AsyncCollectiveCreator::Run(HloModule* module) {
  bool changed = false;
  struct ReplacedAsync {
    HloInstruction* start;
    HloInstruction* done;
  };
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    // Find all all-reduce ops first as we can't modify the instructions while
    // iterating through them.
    std::vector<HloInstruction*> supported_collectives;
    for (HloInstruction* instruction : computation->instructions()) {
      if ((instruction->opcode() == HloOpcode::kAllReduce &&
           convert_all_reduce_(instruction)) ||
          (instruction->opcode() == HloOpcode::kAllGather &&
           convert_all_gather_(instruction)) ||
          (instruction->opcode() == HloOpcode::kCollectivePermute &&
           convert_collective_permute_(instruction))) {
        supported_collectives.push_back(instruction);
      }
    }
    if (supported_collectives.empty()) {
      continue;
    }

    absl::flat_hash_map<HloInstruction*, ReplacedAsync> replaced_pairs;
    bool should_update_schedule =
        module->has_schedule() &&
        module->schedule().is_computation_scheduled(computation);
    for (HloInstruction* instruction : supported_collectives) {
      if (HloAllReduceInstruction* ar =
              DynCast<HloAllReduceInstruction>(instruction)) {
        HloInstruction* start =
            computation->AddInstruction(HloInstruction::CreateAllReduceStart(
                ar->shape(), ar->operands(), ar->to_apply(),
                ar->replica_groups(), ar->constrain_layout(), ar->channel_id(),
                ar->use_global_device_ids()));
        std::unique_ptr<HloInstruction> done = HloInstruction::CreateUnary(
            ar->shape(), HloOpcode::kAllReduceDone, start);
        start->set_metadata(ar->metadata());
        start->CopyBackendConfigFrom(ar);
        if (should_update_schedule) {
          replaced_pairs[ar] = ReplacedAsync{start, done.get()};
        }
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            computation->ReplaceWithNewInstruction(ar, std::move(done)),
            "replacing ", ar->ToShortString());
        changed = true;
        continue;
      }
      if (HloAllGatherInstruction* ag =
              DynCast<HloAllGatherInstruction>(instruction)) {
        std::vector<const Shape*> operand_shapes;
        operand_shapes.reserve(ag->operand_count());
        for (const HloInstruction* op : ag->operands()) {
          operand_shapes.push_back(&op->shape());
        }
        Shape shape = ShapeUtil::MakeTupleShape(
            {ag->operand_count() > 1
                 ? ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes)
                 : *operand_shapes[0],
             ag->shape()});
        HloInstruction* start =
            computation->AddInstruction(HloInstruction::CreateAllGatherStart(
                shape, ag->operands(), ag->all_gather_dimension(),
                ag->replica_groups(), ag->constrain_layout(), ag->channel_id(),
                ag->use_global_device_ids()));
        std::unique_ptr<HloInstruction> done = HloInstruction::CreateUnary(
            ag->shape(), HloOpcode::kAllGatherDone, start);
        start->set_metadata(ag->metadata());
        start->CopyBackendConfigFrom(ag);
        if (should_update_schedule) {
          replaced_pairs[ag] = ReplacedAsync{start, done.get()};
        }
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            computation->ReplaceWithNewInstruction(ag, std::move(done)),
            "replacing ", ag->ToShortString());
        changed = true;
        continue;
      }
      if (HloCollectivePermuteInstruction* cp =
              DynCast<HloCollectivePermuteInstruction>(instruction)) {
        HloInstruction* collective_permute_start;
        HloInstruction* operand = cp->mutable_operand(0);
        if (cp->operand_count() == 1) {
          collective_permute_start = computation->AddInstruction(
              HloInstruction::CreateCollectivePermuteStart(
                  ShapeUtil::MakeTupleShape(
                      {operand->shape(), cp->shape(),
                       ShapeUtil::MakeShape(U32, {}, {}),
                       ShapeUtil::MakeShape(U32, {}, {})}),
                  operand, cp->source_target_pairs(), cp->channel_id()));
        } else {
          CHECK_EQ(cp->operand_count(), 4);
          std::vector<const Shape*> operand_shapes;
          absl::c_transform(cp->operands(), std::back_inserter(operand_shapes),
                            [](const HloInstruction* operand) {
                              return &(operand->shape());
                            });
          collective_permute_start = computation->AddInstruction(
              HloInstruction::CreateCollectivePermuteStart(
                  ShapeInference::InferCollectivePermuteStartShape(
                      operand_shapes)
                      .ValueOrDie(),
                  operand, cp->mutable_operand(1), cp->mutable_operand(2),
                  cp->mutable_operand(3), cp->source_target_pairs(),
                  cp->dynamic_slice_sizes_list(), cp->channel_id()));
        }
        collective_permute_start->set_metadata(cp->metadata());
        collective_permute_start->CopyBackendConfigFrom(cp);
        HloInstruction* collective_permute_done =
            computation->AddInstruction(HloInstruction::CreateUnary(
                cp->shape(), HloOpcode::kCollectivePermuteDone,
                collective_permute_start));
        if (should_update_schedule) {
          replaced_pairs[cp] =
              ReplacedAsync{collective_permute_start, collective_permute_done};
        }
        TF_RETURN_IF_ERROR(
            computation->ReplaceInstruction(cp, collective_permute_done));
        changed = true;
        continue;
      }
    }
    if (should_update_schedule) {
      std::vector<HloInstruction*> new_sequence;
      const HloInstructionSequence& sequence =
          module->schedule().sequence(computation);
      new_sequence.reserve(sequence.size() + replaced_pairs.size());
      for (HloInstruction* instr : sequence.instructions()) {
        auto it = replaced_pairs.find(instr);
        if (it != replaced_pairs.end()) {
          new_sequence.push_back(it->second.start);
          new_sequence.push_back(it->second.done);
          continue;
        }
        new_sequence.push_back(instr);
      }
      module->schedule().set_sequence(computation, new_sequence);
    }
  }
  return changed;
}

}  // namespace xla
