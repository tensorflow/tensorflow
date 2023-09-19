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

#include "xla/service/collective_into_scan_loop_code_motion.h"

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = match;

// Returns the output idx from the corresponding loop, if this collective
// instruction is movable.
std::optional<int64_t> GetMovableCollectiveOutputIdx(
    const HloInstruction* instruction) {
  // We only support AllReduce and CollectivePermute.
  CHECK(instruction->opcode() == HloOpcode::kAllReduce ||
        instruction->opcode() == HloOpcode::kCollectivePermute);
  // Check that the collective input is coming from a while loop.
  const HloInstruction* gte = instruction->operand(0);
  const HloInstruction* while_;
  if (!Match(gte, m::GetTupleElement(
                      m::Op(&while_).WithOpcode(HloOpcode::kWhile)))) {
    return std::nullopt;
  }

  const Shape& output_shape = gte->shape();
  if (!output_shape.IsArray() || (output_shape.rank() < 1) ||
      (output_shape.dimensions(0) < 2)) {
    return std::nullopt;
  }

  int64_t num_iters = output_shape.dimensions(0);

  // Is the while loop performing a `scan` function?
  // Check that the loop range is `[0, N)` where `N` is the leading dimension of
  // our output (which we expect to be a `DynamicUpdateSlice`).
  //
  // NOTE: It is not sufficient to check the trip count for the loop. The loop
  // must write to every index in the output. Strictly speaking, the fact that
  // the loop writes to every output is the only condition for correctness, but
  // scan loops are the most common where that is easily provable.

  // TODO(cjfj): Check reversed condition / `<=`.
  const HloInstruction* loop_iter;
  const HloInstruction* loop_end;
  if (!Match(while_->while_condition()->root_instruction(),
             m::Lt(m::GetTupleElement(&loop_iter, m::Parameter(0)),
                   m::ConstantScalar(&loop_end, num_iters)))) {
    return std::nullopt;
  }

  int64_t loop_idx =
      static_cast<const HloGetTupleElementInstruction*>(loop_iter)
          ->tuple_index();

  if (!Match(while_->while_init()->operand(loop_idx),
             m::ConstantScalar(0).WithElementType(S32))) {
    return std::nullopt;
  }

  int64_t output_idx =
      static_cast<const HloGetTupleElementInstruction*>(gte)->tuple_index();

  // Check that the collective input is produced by a dynamic update slice,
  // using the loop iteration count to slice in the leading dimension.
  //
  // JAX "scan" loops emit code that supports negative update indices, despite
  // us knowing that it will never be negative.
  Shape expected_update_shape = output_shape;
  expected_update_shape.set_dimensions(0, 1);

  const HloInstruction* dyn_update_slice =
      while_->while_body()->root_instruction()->operand(output_idx);

  if ((dyn_update_slice->opcode() != HloOpcode::kDynamicUpdateSlice) ||
      !Match(dyn_update_slice->operand(0),
             m::GetTupleElement(m::Parameter(0), output_idx)) ||
      !(dyn_update_slice->operand(1)->shape() == expected_update_shape) ||
      !Match(dyn_update_slice->operand(2),
             m::AnyOf<const HloInstruction>(
                 m::GetTupleElement(m::Parameter(0), loop_idx),
                 // Reversed index (`num_iters - 1 - i`).
                 m::Subtract(m::ConstantScalar(num_iters - 1),
                             m::GetTupleElement(m::Parameter(0), loop_idx)),
                 // Negative index select (`(i >= 0) ? i : (i + num_iters)`).
                 m::Select(m::Lt(m::GetTupleElement(m::Parameter(0), loop_idx),
                                 m::ConstantScalar(0)),
                           m::AddAnyOrder(
                               m::GetTupleElement(m::Parameter(0), loop_idx),
                               m::ConstantScalar(num_iters)),
                           m::GetTupleElement(m::Parameter(0), loop_idx)),
                 // Reversed with negative index select.
                 // (`i = num_iters - 1 - i`; (i >= 0) ? i : (i + num_iters)`).
                 m::Select(
                     m::Lt(m::Subtract(
                               m::ConstantScalar(num_iters - 1),
                               m::GetTupleElement(m::Parameter(0), loop_idx)),
                           m::ConstantScalar(0)),
                     m::Subtract(m::ConstantScalar(2 * num_iters - 1),
                                 m::GetTupleElement(m::Parameter(0), loop_idx)),
                     m::Subtract(
                         m::ConstantScalar(num_iters - 1),
                         m::GetTupleElement(m::Parameter(0), loop_idx)))))) {
    return std::nullopt;
  }
  return output_idx;
}

absl::Status MoveCollective(HloInstruction* collective, int64_t output_idx) {
  HloComputation* computation = collective->parent();
  HloInstruction* loop = collective->mutable_operand(0)->mutable_operand(0);
  HloComputation* loop_body = loop->while_body();
  HloInstruction* update_slice =
      loop_body->root_instruction()->mutable_operand(output_idx);
  HloInstruction* new_collective_input = update_slice->mutable_operand(1);
  HloInstruction* new_collective =
      loop_body->AddInstruction(collective->CloneWithNewOperands(
          new_collective_input->shape(), {new_collective_input}));

  if (collective->operand(0)->user_count() == 1) {
    // If the output upon which the collective acts has no other users, we can
    // reuse the existing loop variable.
    TF_RETURN_IF_ERROR(update_slice->ReplaceOperandWith(1, new_collective));
    return collective->ReplaceAllUsesWith(collective->mutable_operand(0));
  } else {
    // We require the pre-collective output. We add the collective output as an
    // extra loop output.
    int64_t new_output_idx = loop->shape().tuple_shapes_size();
    // Append new output shape to the loop shape.
    Shape new_output_shape = loop->shape().tuple_shapes(output_idx);
    *loop->mutable_shape()->add_tuple_shapes() = new_output_shape;

    // Update computation parameter shapes.
    HloComputation* loop_cond = loop->while_condition();
    loop_cond->ReplaceParameter(
        0,
        loop_cond->parameter_instruction(0)->CloneWithNewShape(loop->shape()));
    HloInstruction* new_parameter = loop_body->ReplaceParameter(
        0,
        loop_body->parameter_instruction(0)->CloneWithNewShape(loop->shape()));

    // Add the new loop body output.
    using InstVec = HloInstruction::InstructionVector;
    InstVec new_update_slice_operands = update_slice->mutable_operands();
    new_update_slice_operands[0] = loop_body->AddInstruction(
        HloInstruction::CreateGetTupleElement(new_parameter, new_output_idx));
    new_update_slice_operands[1] = new_collective;

    HloInstruction* new_update_slice =
        loop_body->AddInstruction(update_slice->CloneWithNewOperands(
            update_slice->shape(), new_update_slice_operands));

    InstVec new_outputs = loop_body->root_instruction()->mutable_operands();
    new_outputs.push_back(new_update_slice);
    HloInstruction* new_root_instruction = loop_body->AddInstruction(
        loop_body->root_instruction()->CloneWithNewOperands(loop->shape(),
                                                            new_outputs));
    loop_body->set_root_instruction(new_root_instruction,
                                    /*accept_different_shape=*/true);

    // Add the new loop variable to the loop init.
    HloInstruction* loop_init = loop->mutable_operand(0);
    InstVec new_loop_operands = loop_init->mutable_operands();
    new_loop_operands.push_back(loop_init->mutable_operand(output_idx));
    HloInstruction* new_loop_init = computation->AddInstruction(
        loop_init->CloneWithNewOperands(loop->shape(), new_loop_operands));
    TF_RETURN_IF_ERROR(
        loop->ReplaceOperandWithDifferentShape(0, new_loop_init));

    // Replace collective uses with the new loop output.
    HloInstruction* new_output = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(loop, new_output_idx));
    return collective->ReplaceAllUsesWith(new_output);
  }
}

absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                      HloOpcode opcode) {
  bool changed = false;
  for (HloInstruction* instr : computation->instructions()) {
    if (instr->opcode() != opcode) continue;

    std::optional<int64_t> output_idx = GetMovableCollectiveOutputIdx(instr);
    if (output_idx.has_value()) {
      TF_RETURN_IF_ERROR(MoveCollective(instr, *output_idx));
      changed = true;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> CollectiveIntoScanLoopCodeMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool applied, RunOnComputation(computation, opcode_));
    changed |= applied;
  }
  return changed;
}

}  // namespace xla
