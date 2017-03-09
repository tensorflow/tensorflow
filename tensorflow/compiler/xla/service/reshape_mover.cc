/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/reshape_mover.h"

#include <algorithm>
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace {

// Finds the first operand of an instruction that is a reshape or transpose and
// returns the operand if it is found or nullptr if not found.
HloInstruction* FirstReshapeOrTransposeOperand(const HloInstruction* hlo) {
  for (HloInstruction* op : hlo->operands()) {
    if (op->opcode() == HloOpcode::kReshape ||
        op->opcode() == HloOpcode::kTranspose) {
      return op;
    }
  }
  return nullptr;
}

// Check if an operand of an instruction can change its shape simply by
// adjusting metadata. This is the case if an operand does not have any
// producers like Constants or Rng instruction, or is a scalar.
bool OperandCanTrivallyChangeShape(const HloInstruction* instruction,
                                   const HloInstruction* operand) {
  // Scalars can operate with any shape.
  if (ShapeUtil::IsScalar(operand->shape())) {
    return true;
  }

  // A constant can trivially reshape the literal it holds.
  if (operand->opcode() == HloOpcode::kConstant &&
      ShapeUtil::SameDimensions(operand->shape(), instruction->shape())) {
    return true;
  }

  // An Rng instruction can be any shape as long as it has one user. Two copies
  // of the same Rng would be problematic if an Rng of a different shape would
  // produce random numbers in a different order.
  if (operand->opcode() == HloOpcode::kRng &&
      ShapeUtil::SameDimensions(operand->shape(), instruction->shape()) &&
      operand->user_count() == 1) {
    return true;
  }
  return false;
}

// Returns whether `a` and `b` are equivalent for the purposes of this pass.
bool AreEquivalentReshapes(const HloInstruction* a, const HloInstruction* b) {
  if (a->opcode() != b->opcode() ||
      !ShapeUtil::SameDimensions(a->shape(), b->shape())) {
    return false;
  }
  switch (a->opcode()) {
    case HloOpcode::kTranspose:
      return a->dimensions() == b->dimensions();
    case HloOpcode::kReshape:
      return ShapeUtil::SameDimensions(a->operand(0)->shape(),
                                       b->operand(0)->shape());
    default:
      return false;
  }
}

// Returns true if an elementwise operation has all operands that can easily
// change shape. Operands can easily change shape if they are all
// reshapes/transposes to and from the same shape. Additionally, operands like
// constant, rng, and any scalar change shape with only an adjustment of
// metadata.
bool IsElementwiseOfEquivalentReshapesOrTransposes(
    const HloInstruction* instruction) {
  const std::vector<HloInstruction*>& operands = instruction->operands();
  HloInstruction* first_reshape_or_transpose =
      FirstReshapeOrTransposeOperand(instruction);
  // If there are no reshapes or transposes, then there is nothing to sink below
  // the elemntwise operation.
  if (!first_reshape_or_transpose) {
    return false;
  }
  return (instruction->user_count() > 0 ||
          instruction == instruction->parent()->root_instruction()) &&
         instruction->IsElementwise() && !operands.empty() &&
         // Check whether all operands:
         //    1. are a reshapes or transposes that has the same input and
         //    output shapes as all other reshaped or transposed operands.
         //      or
         //    2. can be any shape like kConstant, kRng, and scalars.
         std::all_of(
             operands.begin(), operands.end(),
             [instruction,
              first_reshape_or_transpose](const HloInstruction* operand) {
               return AreEquivalentReshapes(first_reshape_or_transpose,
                                            operand) ||
                      OperandCanTrivallyChangeShape(instruction, operand);
             });
}

// Try to sink any reshape or transpose operands of `instruction` across it. We
// do so if `instruction` is elementwise and all operands are equivalent
// reshapes or transposes.
bool TrySinkReshapeOrTranspose(HloComputation* computation,
                               HloInstruction* instruction) {
  if (IsElementwiseOfEquivalentReshapesOrTransposes(instruction)) {
    std::vector<HloInstruction*> operands = instruction->operands();
    HloInstruction* old_reshape = FirstReshapeOrTransposeOperand(instruction);
    CHECK(old_reshape != nullptr);
    Shape new_elementwise_shape = old_reshape->operand(0)->shape();
    for (size_t i = 0; i < operands.size(); ++i) {
      if (ShapeUtil::IsScalar(operands[i]->shape()) &&
          operands[i]->opcode() != HloOpcode::kReshape &&
          operands[i]->opcode() != HloOpcode::kTranspose) {
        continue;
      }
      auto element_type = operands[i]->shape().element_type();
      switch (operands[i]->opcode()) {
        case HloOpcode::kConstant: {
          if (old_reshape->opcode() == HloOpcode::kReshape) {
            operands[i] = instruction->parent()->AddInstruction(
                HloInstruction::CreateReshape(
                    ShapeUtil::ChangeElementType(new_elementwise_shape,
                                                 element_type),
                    operands[i]));
          } else {
            CHECK_EQ(old_reshape->opcode(), HloOpcode::kTranspose);
            std::vector<int64> inverse_permutation =
                InversePermutation(old_reshape->dimensions());
            operands[i] = instruction->parent()->AddInstruction(
                HloInstruction::CreateTranspose(
                    ShapeUtil::ChangeElementType(new_elementwise_shape,
                                                 element_type),
                    operands[i], inverse_permutation));
          }
          break;
        }
        case HloOpcode::kRng: {
          CHECK_EQ(operands[i]->user_count(), 1);
          operands[i] = instruction->parent()->AddInstruction(
              operands[i]->CloneWithNewOperands(
                  ShapeUtil::ChangeElementType(new_elementwise_shape,
                                               element_type),
                  operands[i]->operands()));
          break;
        }
        case HloOpcode::kReshape:
        case HloOpcode::kTranspose:
          operands[i] = operands[i]->mutable_operand(0);
          break;
        default:
          LOG(FATAL) << "Unexpected opcode while trying to sink reshapes or "
                        "transposes.";
      }
    }
    if (HloOpcode::kFusion == instruction->opcode()) {
      // Here we already know `instruction` is elementwise, and no operand is
      // implicit broadcast as if it were the operands would not be equivalent
      // reshapes, so all the fused instructions have the same dimensions.
      for (const auto& fused_instruction : instruction->fused_instructions()) {
        Shape* shape = fused_instruction->mutable_shape();
        *shape->mutable_dimensions() = new_elementwise_shape.dimensions();
        *shape->mutable_layout() = new_elementwise_shape.layout();
      }
    }
    auto new_elementwise =
        computation->AddInstruction(instruction->CloneWithNewOperands(
            // `instruction` may change the element type, e.g., from
            //   operands[0] -> reshape -> convert (`instruction`)
            // to
            //   operands[0] -> convert' -> reshape'
            //
            // In this case, convert' should have the same element type as
            // `convert` and the same dimensions as operands[0].
            ShapeUtil::ChangeElementType(new_elementwise_shape,
                                         instruction->shape().element_type()),
            operands));
    std::unique_ptr<HloInstruction> new_reshape;
    switch (old_reshape->opcode()) {
      case HloOpcode::kReshape:
        new_reshape = HloInstruction::CreateReshape(instruction->shape(),
                                                    new_elementwise);
        break;
      case HloOpcode::kTranspose:
        new_reshape = HloInstruction::CreateTranspose(
            instruction->shape(), new_elementwise, old_reshape->dimensions());
        break;
      default:
        LOG(FATAL) << "Bad opcode";
    }
    TF_CHECK_OK(computation->ReplaceWithNewInstruction(instruction,
                                                       std::move(new_reshape)));
    return true;
  }
  return false;
}

}  // namespace

StatusOr<bool> ReshapeMover::Run(HloModule* module) {
  return std::any_of(
      module->computations().begin(), module->computations().end(),
      [](const std::unique_ptr<HloComputation>& computation) {
        std::list<HloInstruction*> postorder =
            computation->MakeInstructionPostOrder();
        return std::any_of(postorder.begin(), postorder.end(),
                           [&computation](HloInstruction* instruction) {
                             return TrySinkReshapeOrTranspose(computation.get(),
                                                              instruction);
                           });
      });
}

}  // namespace xla
