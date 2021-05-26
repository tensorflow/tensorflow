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

#include "tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

// Returns true if all instructions are supported operations.
bool AreInstructionSupported(HloComputation* comp) {
  for (HloInstruction* instr: comp->instructions()) {
    bool supported =
      (HloInstruction::IsOpElementwise(instr->opcode()) ||
       instr->opcode() == HloOpcode::kConstant ||
       instr->opcode() == HloOpcode::kReduce ||
       instr->opcode() == HloOpcode::kTuple ||
       instr->opcode() == HloOpcode::kParameter ||
       (instr->opcode() == HloOpcode::kBitcast &&
        instr->shape().rank() < instr->operand(0)->shape().rank()) ||
       (instr->opcode() == HloOpcode::kBroadcast &&
        (instr->dimensions().size() == 0 ||   // scalar broadcasting
         (instr->dimensions().size() == 1 &&  // row broadcasting
          instr->dimensions()[0] == (instr->shape().rank() - 1)))));
    if (!supported) {
      VLOG(2) << "NOT SUPPORTED " << instr->ToString();
      return false;
    }
  }
  return true;
}

StatusOr<bool> FusionBitcastLift::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "FusionBitcastLift::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    // Copy the instruction list as we modify the HloComputation.
    std::vector<HloInstruction*> comp_instruction(comp->instructions().begin(),
                                                  comp->instructions().end());
    for (auto it = comp_instruction.begin(); it != comp_instruction.end();
         it++) {
      // 1) Is this a fusion that we want to modify.
      HloInstruction* instr = *it;
      if (HloFusionInstruction* fusion = DynCast<HloFusionInstruction>(instr)) {
        // 1.1) We only support kInput fusion and some operations.
        if (fusion->fusion_kind() != HloInstruction::FusionKind::kInput ||
	    !AreInstructionSupported(
                fusion->fused_instructions_computation())) {
          continue;
        }
        // 1.2) Check if there is a bitcast that we lift. Currently
        //      we do not lift(merge) bitcast above(with) broadcast.
        if (!std::any_of(fusion->fused_instructions().begin(),
                         fusion->fused_instructions().end(),
                         [](HloInstruction* inner) {
              return inner->opcode() == HloOpcode::kBitcast &&
                  inner->operand(0)->opcode() != HloOpcode::kBroadcast;
                         })) {
          continue;
        }
        // 1.3) Check that all the bitcast have the same shape pattern.
        //      Multiple bitcast pattern isn't supported/tested.
	std::vector<HloInstruction*> bitcasts;
	for (HloInstruction* fused_instr : fusion->fused_instructions()) {
	  if (fused_instr->opcode() == HloOpcode::kBitcast &&
	      fused_instr->shape().rank() <
	      fused_instr->operand(0)->shape().rank()) {
	    if (bitcasts.size() > 0 && (
                    !ShapeUtil::Equal(fused_instr->shape(),
				      bitcasts[0]->shape()) ||
                    !ShapeUtil::Equal(bitcasts[0]->operand(0)->shape(),
				      fused_instr->operand(0)->shape()))) {
	      continue;
	    }
	    bitcasts.push_back(fused_instr);
	  }
	}

	// 2) Now that we have found a fusion that we want to modify,
	//    create the new fusion. We do so by:
	//    a) Cloning the old fusion.
	//    b) Recursively walk the graph from the root and lift
	//       the bitcast one instruction at a time.
	std::unique_ptr<HloInstruction> cloned_fusion =
	  fusion->Clone("bitcast");
	std::vector<HloInstruction*> stack(
					   {cloned_fusion->fused_expression_root()});
	bool clone_changed = false;
	while (stack.size() > 0) {
	  HloInstruction* i = stack.back();
	  stack.pop_back();
	  if (i->opcode() == HloOpcode::kTuple) {
	    stack.insert(stack.end(), i->operands().begin(),
			 i->operands().end());
	    continue;
	  } else if (i->opcode() == HloOpcode::kParameter &&
		     absl::c_all_of(i->users(), [](HloInstruction* u) {
                         return u->opcode() == HloOpcode::kBitcast;
                       })) {
	    // Replace the parameter inside the fusion.
	    Shape new_shape = i->users()[0]->shape();
	    int64 parameter_number = i->parameter_number();
	    string name = i->name();
	    auto n = HloInstruction::CreateParameter(parameter_number,
						     new_shape, name);
	    HloInstruction* new_parameter =
	      i->parent()->ReplaceParameter(parameter_number,
					    std::move(n));
	    // Remove the old inner bitcast.
	    auto old_users = new_parameter->users();
	    for (HloInstruction* param_user : old_users) {
	      DCHECK(param_user->opcode() == HloOpcode::kBitcast)
		<< "Expected a bitcast";
	      param_user->parent()->ReplaceInstructionWithDifferentShape(
									 param_user, new_parameter);
	    }
	    // Replace the corresponding fusion operands with a new bitcast.
	    HloInstruction* old_outer_parameter =
	      cloned_fusion->mutable_operand(parameter_number);
	    HloInstruction* new_op =
	      old_outer_parameter->parent()->AddInstruction(
                    HloInstruction::CreateBitcast(new_shape,
						  old_outer_parameter));
	    cloned_fusion->ReplaceOperandWithDifferentShape(
                parameter_number, new_op);
	    clone_changed = true;
	    changed = true;
	  } else if (i->opcode() == HloOpcode::kBroadcast) {
	    // For now, do nothing. Later we can merge the broadcast
	    // and the bitcast, but this doesn't bring benefit in my
	    // current case.
	    stack.push_back(i->mutable_operand(0));
	  } else if (i->users().size() > 0 &&
		     absl::c_all_of(i->users(), [](HloInstruction* u) {
                         return u->opcode() == HloOpcode::kBitcast;
                       })) {
	    // All users are bitcast, so lift the bitcast.
	    Shape new_shape = i->users()[0]->shape();
	    std::vector<HloInstruction*> new_operands;
	    for (HloInstruction* opnd : i->operands()) {
	      Shape dtyped_new_shape = ShapeUtil::ChangeElementType(
                  new_shape, opnd->shape().element_type());
	      HloInstruction* new_opnd = opnd->parent()->AddInstruction(
                  HloInstruction::CreateBitcast(dtyped_new_shape, opnd));
	      new_operands.push_back(new_opnd);
	      // Handle the operand right before the inserted bitcast now.
	      if (std::find(stack.begin(), stack.end(), opnd) ==
		  stack.end()) {
		stack.push_back(opnd);
	      }
	    }
	    Shape dtyped_new_shape = ShapeUtil::ChangeElementType(
                new_shape, i->shape().element_type());
	    HloInstruction* cloned_i = i->parent()->AddInstruction(
                i->CloneWithNewOperands(dtyped_new_shape, new_operands));
	    // Replace the old bitcasts with the new instruction to
	    // remove it.
	    for (HloInstruction* user: i->users()) {
	      i->parent()->ReplaceInstructionWithDifferentShape(
                  user, cloned_i);
	    }
	    clone_changed = true;
	    changed = true;
	  } else {
	    stack.insert(stack.end(), i->operands().begin(),
			 i->operands().end());
	  }
	}  // while
	DCHECK(clone_changed) << "We should have changed the fusion!";
	if (clone_changed) {
	  // 3) Replace the old fusion with the new fusion.
	  fusion->parent()->ReplaceWithNewInstruction(
              fusion, std::move(cloned_fusion));
	}
      } // if fusion
    }
  }
  XLA_VLOG_LINES(2, "FusionBitcastLift::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
