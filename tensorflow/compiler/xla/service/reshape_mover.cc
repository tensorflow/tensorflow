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

// Implementation note:
//
// The general idea behind this pass is that we're converting from this:
//   %param.A = OldShape
//   %param.B = OldShape
//   %reshape.A = NewShape reshape(%param.A)
//   %reshape.B = NewShape reshape(%param.B)
//   %instruction = NewShape instruction(%reshape.A, %reshape.B)
// To this:
//   %param.A = OldShape
//   %param.B = OldShape
//   %instruction = OldShape instruction(%param.A, %param.B)
//   %reshape = NewShape reshape(%instruction)
//
// Where the instruction must be elementwise, and both reshapes and transposes
// are moved.
//
// Most elementwise instructions support implicit broadcast of scalar operands,
// but select is a special-case.  The signature is Select(Pred, A, B), and the
// only implicit scalar broadcast is on Pred, not on A or B. Since reshapes or
// transposes to a scalar should be cheap, we simply never move them.

#include "tensorflow/compiler/xla/service/reshape_mover.h"

#include <algorithm>
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

// Checks if an instruction can change its shape simply by adjusting metadata.
// This is the case if it is:
//
// - an instruction does not have any producers like Constants
// or Rng instruction, or is a scalar.
//
// Or
//
// - an reshape/transpose instruction with an operand that can trivially change
// its shape.
bool InstructionCanTriviallyChangeShape(const HloInstruction* instruction) {
  // Reshape/Transposes are only trivial if their operand is trivial.
  if (instruction->opcode() == HloOpcode::kReshape ||
      instruction->opcode() == HloOpcode::kTranspose) {
    CHECK_EQ(instruction->operand_count(), 1);
    return InstructionCanTriviallyChangeShape(instruction->operand(0));
  }

  // Scalars can operate with any shape.
  if (ShapeUtil::IsScalar(instruction->shape())) {
    return true;
  }

  // A constant can trivially reshape the literal it holds.
  if (instruction->opcode() == HloOpcode::kConstant) {
    return true;
  }

  // An Rng instruction can be any shape as long as it has one user. Two copies
  // of the same Rng would be problematic if an Rng of a different shape would
  // produce random numbers in a different order.
  if (instruction->opcode() == HloOpcode::kRng &&
      instruction->user_count() == 1) {
    return true;
  }
  return false;
}

// Finds the first non-scalar operand of an instruction that is a non-trivial
// reshape or transpose. Returns the operand if it is found or nullptr if not
// found.
HloInstruction* FirstNonScalarAndNonTrivialReshapeOperand(
    const HloInstruction* hlo) {
  for (HloInstruction* operand : hlo->operands()) {
    if (!ShapeUtil::IsScalar(operand->shape()) &&
        ((operand->opcode() == HloOpcode::kReshape ||
          operand->opcode() == HloOpcode::kTranspose) &&
         !InstructionCanTriviallyChangeShape(operand->operand(0)))) {
      VLOG(5) << "Found first non-scalar and non-trivial reshape operand of "
              << hlo->ToStringNoMetadata() << ":\n\t"
              << operand->ToStringNoMetadata();
      return operand;
    }
  }
  return nullptr;
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
  const auto& operands = instruction->operands();
  HloInstruction* first_reshape_operand =
      FirstNonScalarAndNonTrivialReshapeOperand(instruction);
  // If there are no non-trivial reshapes or transposes, then there is nothing
  // to sink below the elementwise operation.
  if (!first_reshape_operand) {
    return false;
  }
  VLOG(3) << "** Checking whether instruction is an elementwise operation of "
             "equivalent reshapes/transposes: "
          << instruction->ToStringNoMetadata();
  bool result = (instruction->user_count() > 0 ||
                 instruction == instruction->parent()->root_instruction()) &&
                instruction->IsElementwise() && !operands.empty();

  // Check whether all operands:
  //    0. Have the same dimensions as the output -- if not, it may be
  //       implicitly broadcast, which can confound the movement's
  //       correctness.
  //
  // And one of the following:
  //    1. Are reshapes or transposes that have the same input and
  //       output shapes as all other reshaped or transposed operands.
  //     or
  //    2. Are one of kConstant, kRng, and scalars that can change shape
  //    trivially,
  if (result) {
    for (auto& operand : operands) {
      if (!ShapeUtil::SameDimensions(operand->shape(), instruction->shape())) {
        VLOG(5) << "Operand shape differs from output shape; may be "
                   "implicitly broadcast, so preventing "
                   "movement\n\toperand: "
                << operand->ToStringNoMetadata()
                << "\n\tinstruction: " << instruction->ToStringNoMetadata();
        result = false;
        break;
      }

      if (AreEquivalentReshapes(first_reshape_operand, operand)) {
        VLOG(5) << "Are equivalent reshapes:\n\tfirst_reshape_operand: "
                << first_reshape_operand->ToStringNoMetadata()
                << "\n\toperand: " << operand->ToStringNoMetadata();
        continue;
      }

      if (InstructionCanTriviallyChangeShape(operand)) {
        VLOG(5) << "Operand can trivially change shape: "
                << operand->ToStringNoMetadata();
        continue;
      }

      // TODO(someone): Look into supporting general ops for the operands as
      // well.
      VLOG(5) << "Operand is neither equalivant to the first Reshape operand"
                 "nor can trivially change shape: "
              << operand->ToStringNoMetadata();
      result = false;
      break;
    }
  }

  VLOG(3) << "ElementwiseOfEquivalentReshapesOrTransposes result for "
          << instruction->ToStringNoMetadata() << ": " << result;
  return result;
}

// Try to sink any reshape or transpose operands of `instruction` across it. We
// do so if `instruction` is elementwise and all operands are either equivalent
// reshapes/transposes or are trivially reshapable. Note that no move is
// performend if there is no nontrivial reshapes/transposes.
StatusOr<bool> TrySinkReshapeOrTranspose(HloComputation* computation,
                                         HloInstruction* instruction) {
  if (!IsElementwiseOfEquivalentReshapesOrTransposes(instruction)) {
    return false;
  }

  HloInstruction* old_reshape =
      FirstNonScalarAndNonTrivialReshapeOperand(instruction);
  TF_RET_CHECK(old_reshape != nullptr);
  Shape new_elementwise_shape = old_reshape->operand(0)->shape();

  VLOG(3) << "** Trying to sink reshape or transpose: "
          << instruction->ToStringNoMetadata()
          << "\n\told reshape: " << old_reshape->ToStringNoMetadata()
          << "\n\tnew elementwise shape: "
          << ShapeUtil::HumanString(new_elementwise_shape);

  auto operands = instruction->operands();
  for (size_t i = 0; i < operands.size(); ++i) {
    // All scalar operands remain as-is, even if they're reshape or transpose,
    // to simplify handling wrt special scalar broadcast rules for ops like
    // Select. Scalar reshapes should be cheap anyways.
    if (ShapeUtil::IsScalar(operands[i]->shape())) {
      continue;
    }
    PrimitiveType element_type = operands[i]->shape().element_type();
    switch (operands[i]->opcode()) {
      case HloOpcode::kConstant: {
        if (old_reshape->opcode() == HloOpcode::kReshape) {
          VLOG(3) << "Creating reshape for kConstant operand " << i << ": "
                  << operands[i]->ToStringNoMetadata();
          operands[i] = instruction->parent()->AddInstruction(
              HloInstruction::CreateReshape(
                  ShapeUtil::ChangeElementType(new_elementwise_shape,
                                               element_type),
                  operands[i]));
        } else {
          TF_RET_CHECK(old_reshape->opcode() == HloOpcode::kTranspose);
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
  HloInstruction* new_elementwise =
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
      VLOG(3) << "Creating new reshape for new elementwise op: "
              << new_elementwise->ToStringNoMetadata();
      new_reshape =
          HloInstruction::CreateReshape(instruction->shape(), new_elementwise);
      break;
    case HloOpcode::kTranspose:
      new_reshape = HloInstruction::CreateTranspose(
          instruction->shape(), new_elementwise, old_reshape->dimensions());
      break;
    default:
      LOG(FATAL) << "Bad opcode";
  }
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      instruction, std::move(new_reshape)));
  return true;
}

}  // namespace

StatusOr<bool> ReshapeMover::Run(HloModule* module) {
  bool changed = false;
  std::vector<HloComputation*> computations;
  for (auto& computation : module->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    computations.push_back(computation.get());
  }
  for (const auto& comp : computations) {
    for (HloInstruction* instruction : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool did_change,
                          TrySinkReshapeOrTranspose(comp, instruction));
      changed |= did_change;
    }
  }
  return changed;
}

}  // namespace xla
