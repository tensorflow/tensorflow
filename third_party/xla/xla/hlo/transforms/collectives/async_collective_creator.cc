/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/async_collective_creator.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/frontend_attributes.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

struct ReplacedAsync {
  HloInstruction* start;
  HloInstruction* done;
};

absl::StatusOr<ReplacedAsync> CreateAsyncAllReduce(
    HloInstruction* instruction) {
  auto* ar = Cast<HloAllReduceInstruction>(instruction);
  HloInstruction* start =
      instruction->AddInstruction(HloInstruction::CreateAllReduceStart(
          ar->shape(), ar->operands(), ar->to_apply(), ar->device_list(),
          ar->constrain_layout(), ar->channel_id(),
          ar->use_global_device_ids()));
  HloInstruction* done =
      instruction->AddInstruction(HloInstruction::CreateUnary(
          ar->shape(), HloOpcode::kAllReduceDone, start));
  return ReplacedAsync{start, done};
}

absl::StatusOr<ReplacedAsync> CreateAsyncAllGather(
    HloInstruction* instruction) {
  auto* ag = Cast<HloAllGatherInstruction>(instruction);
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
      instruction->AddInstruction(HloInstruction::CreateAllGatherStart(
          shape, ag->operands(), ag->all_gather_dimension(), ag->device_list(),
          ag->constrain_layout(), ag->channel_id(),
          ag->use_global_device_ids()));
  HloInstruction* done =
      instruction->AddInstruction(HloInstruction::CreateUnary(
          ag->shape(), HloOpcode::kAllGatherDone, start));
  return ReplacedAsync{start, done};
}

absl::StatusOr<ReplacedAsync> CreateAsyncCollectivePermute(
    HloInstruction* instruction, absl::Span<const Shape> context_shapes) {
  auto* cp = Cast<HloCollectivePermuteInstruction>(instruction);
  HloInstruction* start;
  HloInstruction* operand = cp->mutable_operand(0);
  if (cp->operand_count() == 1) {
    start = instruction->AddInstruction(
        HloInstruction::CreateCollectivePermuteStart(
            ShapeInference::InferCollectivePermuteStartShape(
                {&operand->shape()}, context_shapes)
                .value(),
            operand, cp->source_target_pairs(), cp->channel_id()));
  } else {
    CHECK_EQ(cp->operand_count(), 4);
    std::vector<const Shape*> operand_shapes;
    absl::c_transform(
        cp->operands(), std::back_inserter(operand_shapes),
        [](const HloInstruction* operand) { return &(operand->shape()); });
    start = instruction->AddInstruction(
        HloInstruction::CreateCollectivePermuteStart(
            ShapeInference::InferCollectivePermuteStartShape(operand_shapes,
                                                             context_shapes)
                .value(),
            operand, cp->mutable_operand(1), cp->mutable_operand(2),
            cp->mutable_operand(3), cp->source_target_pairs(),
            cp->dynamic_slice_sizes_list(), cp->channel_id()));
    if (HasDisjointReadWriteRegionsAttr(cp)) {
      SetDisjointReadWriteRegionsAttr(start);
    }
  }
  HloInstruction* done =
      instruction->AddInstruction(HloInstruction::CreateUnary(
          cp->shape(), HloOpcode::kCollectivePermuteDone, start));
  return ReplacedAsync{start, done};
}

absl::StatusOr<ReplacedAsync> CreateAsyncStartDone(
    HloInstruction* instruction, absl::Span<const Shape> context_shapes) {
  HloComputation* computation = instruction->parent();
  TF_ASSIGN_OR_RETURN(
      HloInstruction * done,
      computation->CreateAsyncInstructions(instruction, context_shapes,
                                           HloInstruction::kMainExecutionThread,
                                           /*replace=*/false));
  HloInstruction* start = done->mutable_operand(0);
  FrontendAttributes fas = instruction->frontend_attributes();
  start->set_frontend_attributes(fas);
  done->set_frontend_attributes(fas);
  return ReplacedAsync{start, done};
}

int64_t GetShapeSize(const Shape& shape) {
  int64_t size_in_bytes = 0;
  if (shape.IsTuple()) {
    for (int64_t i = 0; i < shape.tuple_shapes_size(); ++i) {
      size_in_bytes += GetShapeSize(shape.tuple_shapes(i));
    }
    return size_in_bytes;
  }
  return ShapeUtil::ByteSizeOfElements(shape);
}

}  // namespace

// Find all supported collective ops first as we can't modify the instructions
// while iterating through them.
std::vector<HloInstruction*> AsyncCollectiveCreator::MatchCollectives(
    HloComputation* computation) {
  std::vector<HloInstruction*> supported_collectives;
  for (HloInstruction* instruction : computation->instructions()) {
    const HloOpcode op = instruction->opcode();
    if ((op == HloOpcode::kAllReduce &&
         config_.convert_all_reduce(instruction) &&
         GetShapeSize(instruction->shape()) >=
             config_.all_reduce_min_threshold_in_bytes) ||
        (op == HloOpcode::kAllGather &&
         config_.convert_all_gather(instruction) &&
         GetShapeSize(instruction->shape()) >=
             config_.all_gather_min_threshold_in_bytes) ||
        (op == HloOpcode::kCollectiveBroadcast &&
         config_.convert_collective_broadcast(instruction)) ||
        (op == HloOpcode::kCollectivePermute &&
         config_.convert_collective_permute(instruction)) ||
        (op == HloOpcode::kAllToAll &&
         config_.convert_all_to_all(instruction)) ||
        (op == HloOpcode::kReduceScatter &&
         config_.convert_reduce_scatter(instruction)) ||
        (op == HloOpcode::kRaggedAllToAll &&
         config_.convert_ragged_all_to_all(instruction))) {
      supported_collectives.push_back(instruction);
    }
  }
  return supported_collectives;
}

absl::StatusOr<bool> AsyncCollectiveCreator::ReplaceCollectives(
    HloComputation* computation,
    std::vector<HloInstruction*>& supported_collectives) {
  bool changed = false;
  HloModule* module = computation->parent();
  absl::flat_hash_map<HloInstruction*, ReplacedAsync> replaced_pairs;
  const bool should_update_schedule =
      module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation);
  for (HloInstruction* instruction : supported_collectives) {
    absl::StatusOr<ReplacedAsync> async_pair;
    switch (instruction->opcode()) {
      case HloOpcode::kAllReduce:
        async_pair = CreateAsyncAllReduce(instruction);
        break;
      case HloOpcode::kAllGather:
        async_pair = CreateAsyncAllGather(instruction);
        break;
      case HloOpcode::kCollectivePermute:
        async_pair = CreateAsyncCollectivePermute(
            instruction, config_.get_context_shapes(instruction));
        break;
      case HloOpcode::kCollectiveBroadcast:
      case HloOpcode::kAllToAll:
      case HloOpcode::kReduceScatter:
      case HloOpcode::kRaggedAllToAll:
        async_pair = CreateAsyncStartDone(
            instruction, config_.get_context_shapes(instruction));
        break;
      default:
        return Internal("Unexpected opcode %s",
                        HloOpcodeString(instruction->opcode()));
    }
    TF_RETURN_IF_ERROR(async_pair.status());
    async_pair->start->set_metadata(instruction->metadata());
    async_pair->start->CopyBackendConfigFrom(instruction);
    if (should_update_schedule) {
      replaced_pairs[instruction] = *async_pair;
    }

    // Update control dependencies if present.
    TF_RETURN_IF_ERROR(
        instruction->CopyAllControlDepsTo(async_pair->start, async_pair->done));
    TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());

    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        computation->ReplaceInstruction(instruction, async_pair->done),
        "replacing ", instruction->ToShortString());
    changed = true;
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
  return changed;
}

absl::StatusOr<bool> AsyncCollectiveCreator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int64_t collectives_replaced = 0;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> supported_collectives =
        MatchCollectives(computation);
    if (supported_collectives.empty()) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool comp_changed,
                        ReplaceCollectives(computation, supported_collectives));
    collectives_replaced += supported_collectives.size();
    changed |= comp_changed;
  }
  VLOG(1) << "Replaced " << collectives_replaced
          << " sync collectives with async versions.";
  return changed;
}

}  // namespace xla
