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

bool IsReshapeOrTranspose(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kReshape ||
         instruction->opcode() == HloOpcode::kTranspose;
}

// Returns true iff `instruction` can change its shape simply by adjusting
// metadata.
bool CanTriviallyChangeShape(const HloInstruction* instruction) {
  // NOTE: Technically a sequence of reshape(reshape(constant)) is also
  // trivially reshapable, so we might be tempted to simply recurse if
  // IsReshapeOrTranspose(instruction)==true.
  //
  // But it's not that simple. E.g. reshape(reshape(rng)) is only trivially
  // reshapable if *all* instructions in the chain have user_count == 1. And
  // reshape(scalar) isn't trivial at all if the reshape itself isn't scalar; we
  // rely on implicit scalar broadcast for scalars to be trivial. In addition,
  // these cases make it harder to maintain correctness of the UpdateOperand
  // logic below.
  //
  // So don't handle these chains, unless you update the tests and code to deal
  // with these properly. One idea is to add a pass immediately beforehand that
  // collapses trivial runs of reshapes / transposes.

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
        IsReshapeOrTranspose(operand) &&
        !CanTriviallyChangeShape(operand->operand(0))) {
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

// Returns true if all operands of `instruction` can easily change shape.
// Operands can easily change shape if they are all reshapes/transposes to and
// from the same shape. Additionally, operands like constant, rng, and any
// scalar change shape with only an adjustment of metadata.
bool AllOperandsHaveEasyShapeChanges(
    const HloInstruction* instruction,
    const HloInstruction* first_reshape_operand) {
  VLOG(3) << "** Checking whether all operands have easy shape changes: "
          << instruction->ToStringNoMetadata();
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
  for (const HloInstruction* operand : instruction->operands()) {
    if (!ShapeUtil::SameDimensions(operand->shape(), instruction->shape())) {
      VLOG(5) << "Operand shape differs from output shape; may be "
                 "implicitly broadcast, so preventing "
                 "movement\n\toperand: "
              << operand->ToStringNoMetadata()
              << "\n\tinstruction: " << instruction->ToStringNoMetadata();
      return false;
    }

    if (AreEquivalentReshapes(first_reshape_operand, operand)) {
      VLOG(5) << "Are equivalent reshapes:\n\tfirst_reshape_operand: "
              << first_reshape_operand->ToStringNoMetadata()
              << "\n\toperand: " << operand->ToStringNoMetadata();
      continue;
    }

    if (CanTriviallyChangeShape(operand)) {
      VLOG(5) << "Operand can trivially change shape: "
              << operand->ToStringNoMetadata();
      continue;
    }

    // TODO(someone): Look into supporting general ops for the operands as
    // well.
    VLOG(5) << "Operand is neither equalivant to the first Reshape operand"
               "nor can trivially change shape: "
            << operand->ToStringNoMetadata();
    return false;
  }

  VLOG(3) << "All operands have easy shape changes: "
          << instruction->ToStringNoMetadata();
  return true;
}

// This function is called once we've decided to sink reshape/transpose operands
// across an instruction. It returns an updated `operand` with a shape that
// plays nicely with `new_operand_shape`; either it has the same shape (of the
// correct type), or it is a scalar that may be implicitly broadcast.
HloInstruction* UpdateOperand(HloComputation* computation,
                              const HloInstruction* first_reshape_operand,
                              const Shape& new_operand_shape,
                              HloInstruction* operand) {
  const PrimitiveType element_type = operand->shape().element_type();
  const Shape new_shape =
      ShapeUtil::ChangeElementType(new_operand_shape, element_type);

  switch (operand->opcode()) {
    case HloOpcode::kConstant: {
      if (first_reshape_operand->opcode() == HloOpcode::kReshape) {
        VLOG(5) << "Adding reshape to kConstant operand";
        return computation->AddInstruction(
            HloInstruction::CreateReshape(new_shape, operand));
      } else {
        CHECK(first_reshape_operand->opcode() == HloOpcode::kTranspose);
        VLOG(5) << "Adding transpose to kConstant operand";
        std::vector<int64> inverse_permutation =
            InversePermutation(first_reshape_operand->dimensions());
        return computation->AddInstruction(HloInstruction::CreateTranspose(
            new_shape, operand, inverse_permutation));
      }
    }
    case HloOpcode::kRng: {
      CHECK_EQ(operand->user_count(), 1);
      VLOG(5) << "Cloning kRng operand with new shape";
      return computation->AddInstruction(
          operand->CloneWithNewOperands(new_shape, operand->operands()));
    }
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose: {
      VLOG(5) << "Using existing operand of kReshape or kTranspose";
      return operand->mutable_operand(0);
    }
    default:
      LOG(FATAL) << "Unexpected operand opcode during update: " << operand;
  }
}

// Try to sink any reshape or transpose operands of `instruction` across it. We
// do so if `instruction` is elementwise and all operands are either equivalent
// reshapes/transposes or are trivially reshapable.
StatusOr<bool> TrySinkReshapeOrTranspose(HloComputation* computation,
                                         HloInstruction* instruction) {
  // Only perform sinks for live elementwise instructions with operands.
  const bool is_dead = instruction->user_count() == 0 &&
                       instruction != computation->root_instruction();
  if (!instruction->IsElementwise() || instruction->operands().empty() ||
      is_dead) {
    return false;
  }

  // Only perform sinks if there are any nontrivial reshape/transpose operands.
  const HloInstruction* first_reshape_operand =
      FirstNonScalarAndNonTrivialReshapeOperand(instruction);
  if (!first_reshape_operand) {
    return false;
  }

  // Only perform sinks if all operands can easily change shape.
  if (!AllOperandsHaveEasyShapeChanges(instruction, first_reshape_operand)) {
    return false;
  }

  // At this point we've decided to sink reshape/transpose operands.
  const Shape& new_operand_shape = first_reshape_operand->operand(0)->shape();
  VLOG(3) << "** Sinking reshape or transpose: "
          << instruction->ToStringNoMetadata() << "\n\tfirst reshape operand: "
          << first_reshape_operand->ToStringNoMetadata()
          << "\n\tnew operand shape: "
          << ShapeUtil::HumanString(new_operand_shape);

  auto operands = instruction->operands();
  for (size_t i = 0; i < operands.size(); ++i) {
    // All scalar operands remain as-is, even if they're reshape or transpose,
    // to simplify handling wrt special scalar broadcast rules for ops like
    // Select. Scalar reshapes should be cheap anyways.
    if (ShapeUtil::IsScalar(operands[i]->shape())) {
      continue;
    }
    VLOG(3) << "Updating operand #" << i << ": "
            << operands[i]->ToStringNoMetadata();
    operands[i] = UpdateOperand(computation, first_reshape_operand,
                                new_operand_shape, operands[i]);
  }
  if (HloOpcode::kFusion == instruction->opcode()) {
    // Here we already know `instruction` is elementwise, and no operand is
    // implicit broadcast as if it were the operands would not have easy shape
    // changes, so all the fused instructions have the same dimensions.
    for (const auto& fused_instruction : instruction->fused_instructions()) {
      Shape* shape = fused_instruction->mutable_shape();
      *shape->mutable_dimensions() = new_operand_shape.dimensions();
      *shape->mutable_layout() = new_operand_shape.layout();
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
          ShapeUtil::ChangeElementType(new_operand_shape,
                                       instruction->shape().element_type()),
          operands));

  std::unique_ptr<HloInstruction> new_reshape;
  switch (first_reshape_operand->opcode()) {
    case HloOpcode::kReshape:
      VLOG(3) << "Creating new reshape for new elementwise op: "
              << new_elementwise->ToStringNoMetadata();
      new_reshape =
          HloInstruction::CreateReshape(instruction->shape(), new_elementwise);
      break;
    case HloOpcode::kTranspose:
      new_reshape =
          HloInstruction::CreateTranspose(instruction->shape(), new_elementwise,
                                          first_reshape_operand->dimensions());
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
  VLOG(2) << "Pre ReshapeMover HLO:";
  XLA_VLOG_LINES(2, module->ToString());
  for (auto* comp : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool did_change,
                          TrySinkReshapeOrTranspose(comp, instruction));
      changed |= did_change;
    }
  }
  VLOG(2) << "Post ReshapeMover HLO:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace xla
