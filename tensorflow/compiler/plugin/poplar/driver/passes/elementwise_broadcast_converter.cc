/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_broadcast_converter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include "absl/container/flat_hash_set.h"

namespace xla {
namespace poplarplugin {
namespace {
bool InputIsBroadcast(const HloInstruction* inst, int64 operand_idx) {
  return inst->operand(operand_idx)->opcode() == HloOpcode::kBroadcast;
}

StatusOr<bool> ConvertBroadcastsToImplicit(HloInstruction* inst) {
  HloComputation* comp = inst->parent();
  // Find all the operands which are broadcasting.
  absl::flat_hash_set<int64> constant_broadcast_operands;
  absl::flat_hash_set<int64> broadcast_operands;
  absl::flat_hash_set<int64> non_broadcast_operands;
  for (int64 i = 0; i < inst->operand_count(); i++) {
    if (InputIsBroadcast(inst, i)) {
      auto broadcast_input = inst->operand(i)->operand(0);
      if (broadcast_input->opcode() == HloOpcode::kConstant &&
          ShapeUtil::ElementsIn(broadcast_input->shape()) == 1) {
        // If it's a broadcast of a scalar constant, we can pull it in.
        // This way we can use the popops expression Const.
        constant_broadcast_operands.insert(i);
      } else {
        broadcast_operands.insert(i);
      }
    } else {
      non_broadcast_operands.insert(i);
    }
  }

  if (non_broadcast_operands.size() == inst->operand_count()) {
    // Non of the operands are broadcasting.
    return false;
  }

  VLOG(1) << "Instruction " << inst->ToString()
          << " has broadcasting operands on indices "
          << absl::StrJoin(non_broadcast_operands, ", ");

  // Create the fusion name, depending whether this is binary or ternary.
  std::string op_name = "_pop_op_implicit_";
  if (inst->operand_count() == 2) {
    op_name += "binary";
  } else {
    op_name += "ternary";
  }

  // Check whether this is inplace.
  auto inplace_description = HloInstructionDescription(inst);
  if (inplace_description.GetType() == HloInstructionType::kInplaceReadWrite) {
    // To be inplace, all the inplace operands have to be non_broadcasting.
    bool is_inplace = true;
    for (auto idx : inplace_description.GetInplaceOperandIndexes()) {
      if (!non_broadcast_operands.contains(idx)) {
        is_inplace = false;
        break;
      }
    }
    if (is_inplace) {
      op_name += "_inplace";
    }
  }

  // Create a fused computation, and pull in the broadcasts (and constants),
  // into the fusion.
  const int64 num_inputs =
      broadcast_operands.size() + non_broadcast_operands.size();
  // Inputs to the fusion.
  std::vector<HloInstruction*> fusion_inputs(num_inputs);
  // Inputs to the op inside the fusion.
  std::vector<HloInstruction*> op_inputs(inst->operand_count());

  auto builder = HloComputation::Builder(op_name);
  // Clone all the operands.
  int64 next_param_idx = 0;
  for (int64 i = 0; i < inst->operand_count(); i++) {
    HloInstruction* old_operand = inst->mutable_operand(i);
    // New operand which will be used inside the fusion.
    HloInstruction* new_operand;
    if (constant_broadcast_operands.contains(i)) {
      // Input is a broadcast of a constant.
      // First clone the constant.
      HloInstruction* const_inst =
          builder.AddInstruction(old_operand->operand(0)->Clone());
      // Clone the broadcast.
      new_operand = builder.AddInstruction(old_operand->CloneWithNewOperands(
          old_operand->shape(), {const_inst}));
    } else if (broadcast_operands.contains(i)) {
      // Input is a broadcast of a non-constant.
      // Set up the parameter.
      HloInstruction* param_inst =
          builder.AddInstruction(HloInstruction::CreateParameter(
              next_param_idx, old_operand->operand(0)->shape(),
              absl::StrCat("arg_", next_param_idx)));
      fusion_inputs[next_param_idx++] = old_operand->mutable_operand(0);
      // Clone the broadcast.
      new_operand = builder.AddInstruction(old_operand->CloneWithNewOperands(
          old_operand->shape(), {param_inst}));
    } else {
      // Input is not a broadcast.
      CHECK(non_broadcast_operands.contains(i));
      // Set up the parameter.
      new_operand = builder.AddInstruction(HloInstruction::CreateParameter(
          next_param_idx, old_operand->shape(),
          absl::StrCat("arg_", next_param_idx)));
      fusion_inputs[next_param_idx++] = old_operand;
    }
    op_inputs[i] = new_operand;
  }
  // Clone the actual operation.
  HloInstruction* root = builder.AddInstruction(
      inst->CloneWithNewOperands(inst->shape(), op_inputs));

  // Create a fusion instruction.
  HloComputation* fusion_computation =
      inst->GetModule()->AddEmbeddedComputation(builder.Build(root));

  HloInstruction* fusion = comp->AddInstruction(HloInstruction::CreateFusion(
      root->shape(), HloInstruction::FusionKind::kCustom, fusion_inputs,
      fusion_computation));

  fusion_computation->SetFusionInstruction(fusion);
  VLOG(1) << "Replacing " << inst->ToString() << " with " << fusion->ToString()
          << " and fusion " << fusion_computation->ToString();
  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, fusion));
  return true;
}
}  // namespace

StatusOr<bool> ElementwiseBroadcastConverter::Run(HloModule* module) {
  absl::flat_hash_set<HloInstruction*> binary_and_ternary_ops;
  // Find all ternary and binary ops.
  for (auto comp : module->MakeNonfusionComputations()) {
    for (auto inst : comp->instructions()) {
      if (inst->IsElementwise() &&
          inst->opcode() != HloOpcode::kDynamicUpdateSlice) {
        switch (inst->operand_count()) {
          case 2:
          case 3:
            binary_and_ternary_ops.insert(inst);
            break;
          default:
            break;
        }
      }
    }
  }

  bool changed = false;
  for (auto op : binary_and_ternary_ops) {
    TF_ASSIGN_OR_RETURN(auto changed_op, ConvertBroadcastsToImplicit(op));
    changed |= changed_op;
  }
  return changed;
}  // namespace poplarplugin

}  // namespace poplarplugin
}  // namespace xla
