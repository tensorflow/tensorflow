/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/convert_mover.h"

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"

namespace xla {
namespace {

// Checks that the literal roundtrips to dst_ty and back to its original type
// without modification.
static bool IsLosslesslyConvertibleTo(const Literal& literal,
                                      PrimitiveType dst_ty) {
  PrimitiveType orig_ty = literal.shape().element_type();

  // The only reason Convert() should fail is if we don't support converting
  // from x to y, which indeed means it's not losslessly-convertible.
  StatusOr<Literal> converted1 = literal.Convert(dst_ty);
  if (!converted1.ok()) {
    return false;
  }
  StatusOr<Literal> converted2 = converted1->Convert(orig_ty);
  if (!converted2.ok()) {
    return false;
  }

  return literal == *converted2;
}

// Opcodes for which convert(op(x)) == op(convert(x)).
//
// TODO(jlebar): This is not a complete list.  For example, we're missing:
//  - dynamic-slice/dynamic-update-slice/gather (we'd need to handle the fact
//    that only *some* of the operands to these ops are to be converted)
//  - bitcast (intentionally excluded because this pass doesn't attempt to be
//    correct WRT layouts; this should be run before layout assignment).
//  - scatter/reduce where the reduction function commutes with convert (e.g.
//    reduce-min works, but reduce-add doesn't).
bool OpCommutesWithConvert(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kConcatenate:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}

StatusOr<bool> MoveConvertPrecisionOps(HloComputation* comp) {
  bool changed = false;

  // Move increase_precision "down" the graph:
  // instr(increase_precision(x)) -> increase_precision(instr(x)).
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    if (!OpCommutesWithConvert(instr->opcode()) ||
        instr->operand_count() == 0 ||
        !absl::c_all_of(instr->operands(), [](const HloInstruction* operand) {
          // TODO(jlebar): Is the user_count == 1 constraint too restrictive?
          return (operand->opcode() == HloOpcode::kConvert &&
                  operand->user_count() == 1) ||
                 operand->opcode() == HloOpcode::kConstant;
        })) {
      continue;
    }
    // At least one of the operands must be a kConvert op, and all of the
    // kConverts must have the same src data type.
    auto convert_op_it = absl::c_find_if(instr->operands(),
                                         HloPredicateIsOp<HloOpcode::kConvert>);
    if (convert_op_it == instr->operands().end()) {
      continue;
    }
    const HloInstruction* convert_op = *convert_op_it;
    if (!absl::c_all_of(instr->operands(), [&](const HloInstruction* operand) {
          return operand->opcode() != HloOpcode::kConvert ||
                 operand->operand(0)->shape().element_type() ==
                     convert_op->operand(0)->shape().element_type();
        })) {
      continue;
    }

    PrimitiveType src_ty = convert_op->operand(0)->shape().element_type();
    PrimitiveType dst_ty = convert_op->shape().element_type();
    if (primitive_util::BitWidth(src_ty) >= primitive_util::BitWidth(dst_ty)) {
      continue;
    }

    // If the input is e.g. pad(convert_to_fp32(x_f16), const_f32), we can
    // transform this to convert_to_fp32(pad(x_f16, convert_to_f16(const_f32)))
    // iff const_f32 == convert_to_f32(convert_to_f16(const_f32)) -- that is, if
    // the constant doesn't lose any information by being converted to a lower
    // precision.
    if (absl::c_any_of(instr->operands(), [&](const HloInstruction* operand) {
          return operand->opcode() == HloOpcode::kConstant &&
                 !IsLosslesslyConvertibleTo(operand->literal(), src_ty);
        })) {
      continue;
    }

    VLOG(2) << "Moving increase-precision convert op " << convert_op->ToString()
            << " down the graph: " << instr->ToString();

    absl::InlinedVector<HloInstruction*, 8> new_operands;
    new_operands.reserve(instr->operand_count());
    for (HloInstruction* operand : instr->operands()) {
      // All operands are either kConvert or kConstant. Unwrap kConvert ops, and
      // wrap constants in a kConvert to dst_ty. (Constant-folding will then
      // fold this into a new constant.)
      switch (operand->opcode()) {
        case HloOpcode::kConvert:
          new_operands.push_back(operand->mutable_operand(0));
          break;
        case HloOpcode::kConstant:
          new_operands.push_back(MakeConvertToHlo(operand, src_ty));
          break;
        default:
          LOG(FATAL) << "Unexpected opcode in " << operand->ToString();
      }
    }
    Shape new_shape = instr->shape();
    new_shape.set_element_type(src_ty);
    HloInstruction* new_instr = comp->AddInstruction(
        instr->CloneWithNewOperands(new_shape, new_operands));
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        instr, HloInstruction::CreateConvert(instr->shape(), new_instr)));
    changed = true;
  }

  // Move decrease_precision "up" the graph:
  // decrease_precision(instr(x)) -> instr(decrease_precision(x)).
  //
  // Walk the graph from the bottom this time since our changes go up the graph.
  std::deque<HloInstruction*> work_queue;
  std::vector<HloInstruction*> instrs = comp->MakeInstructionPostOrder();
  work_queue.insert(work_queue.end(), instrs.rbegin(), instrs.rend());
  while (!work_queue.empty()) {
    HloInstruction* instr = work_queue.front();
    work_queue.pop_front();
    if (instr->opcode() != HloOpcode::kConvert ||
        instr->operand(0)->user_count() != 1 ||
        !OpCommutesWithConvert(instr->operand(0)->opcode())) {
      continue;
    }
    PrimitiveType src_ty = instr->operand(0)->shape().element_type();
    PrimitiveType dst_ty = instr->shape().element_type();
    if (primitive_util::BitWidth(src_ty) <= primitive_util::BitWidth(dst_ty)) {
      continue;
    }

    VLOG(2) << "Moving decrease-precision convert up the graph: "
            << instr->ToString();

    HloInstruction* to_convert = instr->mutable_operand(0);

    absl::InlinedVector<HloInstruction*, 8> new_operands;
    new_operands.reserve(to_convert->operand_count());
    for (HloInstruction* operand : to_convert->operands()) {
      work_queue.push_front(MakeConvertToHlo(operand, dst_ty));
      new_operands.push_back(work_queue.front());
    }
    Shape new_shape = to_convert->shape();
    new_shape.set_element_type(dst_ty);
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        instr, to_convert->CloneWithNewOperands(new_shape, new_operands)));
    changed = true;
  }

  return changed;
}

}  // anonymous namespace

StatusOr<bool> ConvertMover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool changed_computation,
                        MoveConvertPrecisionOps(comp));
    changed |= changed_computation;
  }
  return changed;
}

}  // namespace xla
