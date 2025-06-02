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

#include "xla/service/gpu/transforms/collectives/collective_send_recv_combiner.h"

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


// WrapMultipleSendRecvInstructions is a side-effecting function that
// creates a single computation that wraps all the send/recv instructions.
// As a side effect, the function populates the async_start_inputs
// and async_start_input_shapes vectors with the operands and operand shapes of
// the cloned instruction.
static HloComputation* WrapMultipleSendRecvInstructions(
    std::vector<HloInstruction*>& send_recv_instructions,
    std::vector<HloInstruction*>& async_start_inputs,
    std::vector<Shape>& async_start_input_shapes,
    HloComputation::Builder& builder, HloModule* module) {
  int operand_counter = 0;
  std::vector<HloInstruction*> new_send_recv_instructions;
  for (HloInstruction* instruction : send_recv_instructions) {
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : instruction->operands()) {
      new_operands.push_back(
          builder.AddInstruction(HloInstruction::CreateParameter(
              operand_counter, operand->shape(),
              absl::StrCat("param", operand_counter))));
      async_start_inputs.push_back(operand);
      async_start_input_shapes.push_back(operand->shape());
      operand_counter++;
    }
    new_send_recv_instructions.push_back(builder.AddInstruction(
        instruction->CloneWithNewOperands(instruction->shape(), new_operands)));
  }
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateTuple(new_send_recv_instructions));
  return module->AddEmbeddedComputation(builder.Build(root));
}

static absl::Status UpdateControlDependencies(HloInstruction* old_instruction,
                                              HloInstruction* new_instruction) {
  for (HloInstruction* predecessor : old_instruction->control_predecessors()) {
    TF_RETURN_IF_ERROR(predecessor->RemoveControlDependencyTo(old_instruction));
    TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(new_instruction));
  }
  for (HloInstruction* successor : old_instruction->control_successors()) {
    TF_RETURN_IF_ERROR(old_instruction->RemoveControlDependencyTo(successor));
    TF_RETURN_IF_ERROR(new_instruction->AddControlDependencyTo(successor));
  }
  return absl::OkStatus();
}

static absl::Status CreateAsyncStartAndAsyncDone(
    std::vector<HloInstruction*>& send_recv_instructions,
    HloComputation* async_computation, HloComputation* computation,
    HloModule* module, std::vector<HloInstruction*>& async_start_inputs,
    std::vector<Shape>& async_start_input_shapes, bool& changed) {
  // Async-start shape consists of (tuple_of_operand_shapes,
  // func_output_shape, s32[]), where s32[] is the context state that is
  // used to keep track of the asynchronous operation. For more details,
  // see https://openxla.org/xla/async_ops.
  Shape async_start_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape(async_start_input_shapes),
       async_computation->root_instruction()->shape(),
       ShapeUtil::MakeScalarShape(S32)});
  HloInstruction* async_start =
      computation->AddInstruction(HloInstruction::CreateAsyncStart(
          async_start_shape, async_start_inputs, async_computation));
  HloInstruction* async_done =
      computation->AddInstruction(HloInstruction::CreateAsyncDone(
          async_computation->root_instruction()->shape(), async_start));
  HloInstruction* replacement_async_done = nullptr;
  int async_done_gte_index = 0;
  for (HloInstruction* instruction : send_recv_instructions) {
    // Create the gte(async-done) instructions to replace send-done/recv-done
    HloInstruction* unwrapped_async_done =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            instruction->shape(), async_done, async_done_gte_index));
    ++async_done_gte_index;
    if (HloPredicateIsOp<HloOpcode::kSend>(instruction)) {
      // send-done only returns the control-flow token, which is the last
      // element in the unwrapped async-done tuple
      replacement_async_done =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              unwrapped_async_done->shape().tuple_shapes(2),
              unwrapped_async_done, 2));
    } else if (HloPredicateIsOp<HloOpcode::kRecv>(instruction)) {
      // recv-done returns the received data and the control-flow token
      HloInstruction* first_element_in_recv_done =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              unwrapped_async_done->shape().tuple_shapes(0),
              unwrapped_async_done, 0));
      HloInstruction* second_element_in_recv_done =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              unwrapped_async_done->shape().tuple_shapes(2),
              unwrapped_async_done, 2));
      HloInstruction* recv_done_tuple =
          computation->AddInstruction(HloInstruction::CreateTuple(
              {first_element_in_recv_done, second_element_in_recv_done}));
      replacement_async_done = recv_done_tuple;
    }

    for (HloInstruction* instruction_user : instruction->users()) {
      if (HloPredicateIsOp<HloOpcode::kSendDone, HloOpcode::kRecvDone>(
              instruction_user)) {
        TF_RETURN_IF_ERROR(UpdateControlDependencies(instruction, async_start));
        TF_RETURN_IF_ERROR(UpdateControlDependencies(instruction_user,
                                                     replacement_async_done));
        TF_RETURN_IF_ERROR(
            instruction_user->ReplaceAllUsesWith(replacement_async_done));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction_user));
        changed = true;
      }
    }
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> CollectiveSendRecvCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int wrapped_computation_index = 0;
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    std::vector<HloInstruction*> send_recv_instructions;
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      // For now we don't transform partially pipelined send/recv instructions;
      // in practice this means that the instruction does not feed into a
      // send-done or recv-done instruction.
      if (HloPredicateIsNotOp<HloOpcode::kSend, HloOpcode::kRecv>(
              instruction)) {
        continue;
      }
      if (instruction->users().size() != 1 ||
          HloPredicateIsNotOp<HloOpcode::kSendDone, HloOpcode::kRecvDone>(
              instruction->users()[0])) {
        continue;
      }
      send_recv_instructions.push_back(instruction);
    }
    if (send_recv_instructions.empty()) {
      continue;
    }
    // Create a new computation that wraps the send/recv instructions.
    ++wrapped_computation_index;
    HloComputation::Builder builder = HloComputation::Builder(
        absl::StrCat("wrapped_send_recv_", wrapped_computation_index));
    std::vector<HloInstruction*> async_start_inputs;
    std::vector<Shape> async_start_input_shapes;
    HloComputation* async_computation = WrapMultipleSendRecvInstructions(
        send_recv_instructions, async_start_inputs, async_start_input_shapes,
        builder, module);
    TF_RETURN_IF_ERROR(CreateAsyncStartAndAsyncDone(
        send_recv_instructions, async_computation, computation, module,
        async_start_inputs, async_start_input_shapes, changed));
  }
  return changed;
}

}  // namespace xla
