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

#include "xla/service/gpu/transforms/collective_send_recv_combiner.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
namespace xla {

namespace {

// BuildWrappedComputationForAsyncStart is a side-effecting function that
// returns a clone of the given instruction and populates the async_start_inputs
// and async_start_input_shapes vectors with the operands and operand shapes of
// the cloned instruction.
HloInstruction* BuildWrappedComputationForAsyncStart(
    HloComputation::Builder& builder, HloInstruction* instruction,
    std::vector<HloInstruction*>& async_start_inputs,
    std::vector<Shape>& async_start_input_shapes) {
  int operand_counter = 0;
  std::vector<HloInstruction*> operands;
  for (auto src_operand : instruction->operands()) {
    operands.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        operand_counter, src_operand->shape(),
        absl::StrCat("param", operand_counter))));
    async_start_inputs.push_back(src_operand);
    async_start_input_shapes.push_back(src_operand->shape());
    ++operand_counter;
  }
  return builder.AddInstruction(
      instruction->CloneWithNewOperands(instruction->shape(), operands));
}

absl::Status UpdateControlDependencies(HloInstruction* old_instruction,
                                       HloInstruction* new_instruction) {
  if (!old_instruction->HasControlDependencies()) {
    return absl::OkStatus();
  }
  for (auto predecessor : old_instruction->control_predecessors()) {
    TF_RETURN_IF_ERROR(predecessor->RemoveControlDependencyTo(old_instruction));
    TF_RETURN_IF_ERROR(new_instruction->AddControlDependencyTo(predecessor));
  }
  return absl::OkStatus();
}

absl::Status CreateAsyncStartAndAsyncDone(
    HloInstruction* root, HloComputation::Builder& builder,
    HloInstruction* instruction, HloComputation* computation, HloModule* module,
    std::vector<HloInstruction*>& async_start_inputs,
    std::vector<Shape>& async_start_input_shapes, bool& changed) {
  for (auto instruction_user : instruction->users()) {
    if (instruction_user->opcode() != HloOpcode::kSendDone &&
        instruction_user->opcode() != HloOpcode::kRecvDone) {
      // Ignore instruction users that are not send-done or recv-done.
      continue;
    }
    Shape async_start_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeTupleShape(async_start_input_shapes), root->shape(),
         ShapeUtil::MakeScalarShape(S32)});
    auto async_start =
        computation->AddInstruction(HloInstruction::CreateAsyncStart(
            async_start_shape, async_start_inputs,
            module->AddEmbeddedComputation(builder.Build(root))));
    auto async_done = computation->AddInstruction(
        HloInstruction::CreateAsyncDone(root->shape(), async_start));
    TF_RETURN_IF_ERROR(UpdateControlDependencies(instruction, async_start));
    TF_RETURN_IF_ERROR(UpdateControlDependencies(instruction_user, async_done));
    TF_RETURN_IF_ERROR(
        instruction_user->ReplaceAllUsesWithDifferentShape(async_done));
    TF_RETURN_IF_ERROR(
        instruction_user->parent()->RemoveInstruction(instruction_user));
    TF_RETURN_IF_ERROR(
        instruction->ReplaceAllUsesWithDifferentShape(async_start));
    TF_RETURN_IF_ERROR(instruction->parent()->RemoveInstruction(instruction));
    changed = true;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> CollectiveSendRecvCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int wrapped_computation_index = 0;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kSend &&
          instruction->opcode() != HloOpcode::kRecv) {
        continue;
      }

      // Create a new computation that wraps the send/recv instruction.
      ++wrapped_computation_index;
      auto builder = HloComputation::Builder(absl::StrCat(
          "wrapped_", instruction->name(), wrapped_computation_index));
      std::vector<HloInstruction*> async_start_inputs;
      std::vector<Shape> async_start_input_shapes;
      auto root = BuildWrappedComputationForAsyncStart(
          builder, instruction, async_start_inputs, async_start_input_shapes);

      TF_RETURN_IF_ERROR(CreateAsyncStartAndAsyncDone(
          root, builder, instruction, computation, module, async_start_inputs,
          async_start_input_shapes, changed));
    }
  }
  return changed;
}

}  // namespace xla
