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

// Returns true if `instruction` can change its shape simply by adjusting
// metadata or if `instruction` is a broadcast of a scalar value.
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

  // A broadcase of scalar can trivially change its shape.
  if (instruction->opcode() == HloOpcode::kBroadcast &&
      ShapeUtil::IsScalar(instruction->operand(0)->shape())) {
    return true;
  }

  return false;
}

// Returns true iff `instruction` is a reshape/transpose instruction for which
// a shape change is nontrivial.
bool IsNontrivialReshape(const HloInstruction* instruction) {
  return !ShapeUtil::IsScalar(instruction->shape()) &&
         IsReshapeOrTranspose(instruction) &&
         !CanTriviallyChangeShape(instruction->operand(0));
}

// Finds the first operand of an instruction that is a non-trivial reshape or
// transpose. Returns such an operand or nullptr if not found.
HloInstruction* FirstNonScalarAndNonTrivialReshapeOperand(
    const HloInstruction* hlo) {
  for (HloInstruction* operand : hlo->operands()) {
    if (IsNontrivialReshape(operand)) {
      VLOG(5) << "Found first non-trivial reshape operand of "
              << hlo->ToString(HloPrintOptions().set_print_metadata(false))
              << ":\n\t"
              << operand->ToString(HloPrintOptions().set_print_metadata(false));
      return operand;
    }
  }
  return nullptr;
}

// Returns whether `a` and `b` are equivalent reshapes/transposes.
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

// This function is called once we've decided to sink reshape/transpose operands
// across an instruction. It returns an updated `operand` with a shape that
// plays nicely with `new_operand_shape`; either it has the same shape (of the
// correct type), or it is a scalar that may be implicitly broadcast.
HloInstruction* UpdateOperand(const HloInstruction* first_reshape_operand,
                              const Shape& new_operand_shape,
                              HloInstruction* operand) {
  HloComputation* computation = operand->parent();
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
    case HloOpcode::kBroadcast: {
      CHECK(ShapeUtil::IsScalar(operand->operand(0)->shape()));
      HloInstruction* inst = computation->AddInstruction(
          operand->CloneWithNewOperands(new_shape, operand->operands()));
      VLOG(5) << "Changing broadcast from " << operand->ToString() << " to "
              << inst->ToString();
      return inst;
    }

    default:
      LOG(FATAL) << "Unexpected operand opcode during update: " << operand;
  }
}

// Actually performs the reshape-move transformation -- that is, sinks the
// reshape or transpose operands of `instruction` across it.
StatusOr<bool> PerformSinkReshapeOrTranspose(
    HloInstruction* instruction, const HloInstruction* first_reshape_operand) {
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);
  // At this point we've decided to sink reshape/transpose operands.
  const Shape& new_operand_shape = first_reshape_operand->operand(0)->shape();
  VLOG(3) << "** Sinking reshape or transpose: "
          << instruction->ToString(print_no_metadata)
          << "\n\tfirst reshape operand: "
          << first_reshape_operand->ToString(print_no_metadata)
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
            << operands[i]->ToString(print_no_metadata);
    operands[i] =
        UpdateOperand(first_reshape_operand, new_operand_shape, operands[i]);
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
  HloComputation* computation = instruction->parent();
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
              << new_elementwise->ToString(print_no_metadata);
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

// Returns true if the instruction is a reshape-move candidate.
//
// An instruction is a reshape-move candidate if the instruction is elementwise,
// has at least one nontrivial reshape/transpose operand, and its operands are
// either trivially reshapable or are equivalent nontrivial reshapes/transposes.
bool IsReshapeMoveCandidate(HloInstruction* instruction) {
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);
  VLOG(5) << "** Checking instruction: "
          << instruction->ToString(print_no_metadata);

  // Only perform reshape-move for live elementwise instructions with operands.
  const bool is_dead = instruction->user_count() == 0 &&
                       instruction != instruction->parent()->root_instruction();
  if (!instruction->IsElementwise() || instruction->operands().empty() ||
      is_dead) {
    return false;
  }

  // Check whether all operands:
  //    0. Have the same dimensions as the output -- if not, they may be
  //       implicitly broadcast, which can confound the movement's
  //       correctness.
  //
  // And one of the following:
  //    1. Are reshapes or transposes that have the same input and
  //       output shapes as all other reshaped or transposed operands.
  //     or
  //    2. Are one of kConstant, kRng, broadcast of a scalar value, and scalars
  //     that can change shape trivially.
  const HloInstruction* first_reshape_operand = nullptr;
  for (const HloInstruction* operand : instruction->operands()) {
    if (!ShapeUtil::SameDimensions(operand->shape(), instruction->shape())) {
      VLOG(5) << "Operand shape differs from output shape; may be "
                 "implicitly broadcast, so preventing "
                 "movement\n\toperand: "
              << operand->ToString(print_no_metadata) << "\n\tinstruction: "
              << instruction->ToString(print_no_metadata);
      return false;
    }

    if (CanTriviallyChangeShape(operand)) {
      VLOG(5) << "Operand can trivially change shape: "
              << operand->ToString(print_no_metadata);
      continue;
    }

    if (!IsNontrivialReshape(operand)) {
      VLOG(5) << "Operand can't trivially change shape: "
              << operand->ToString(print_no_metadata);
      return false;
    }

    if (first_reshape_operand == nullptr) {
      first_reshape_operand = operand;
      VLOG(5) << "First reshape operand "
              << operand->ToString(print_no_metadata);
    } else if (AreEquivalentReshapes(first_reshape_operand, operand)) {
      VLOG(5)
          << "Operand is an equivalent reshape of the first reshape operand "
          << operand->ToString(print_no_metadata);
    } else {
      // TODO(someone): Look into supporting general ops for the operands as
      // well.
      VLOG(5) << "Operand is a reshape but is not equivalent to the first "
                 "Reshape operand"
              << operand->ToString(print_no_metadata);
      return false;
    }
  }

  if (first_reshape_operand) {
    VLOG(5) << "All operands have easy shape changes: "
            << instruction->ToString(print_no_metadata);
  }

  return first_reshape_operand != nullptr;
}

// Reshape-moves all qualifying instructions in reshape_candidates.  Returns
// true if it makes changes.
//
// `reshape_candidates` is a set of HloInstructions with nontrivial reshape
// operands, and a instruction in the set can be reshape-moved iff all the users
// of its nontrivial reshape operands can also be reshaped-moved.
//
// The algorithm here iteratively finds the nontrivial operands with users that
// are outside the set of `reshape_candidates`, and removes their users from
// `reshape_candidates`, until either `reshape_candidates` becomes empty or none
// of the remaining nontrivial operands have users outside `reshape_candidates`.
// In the later case, all the remaining instructions in `reshape_candidates`
// are reshape-moved and the routine returns true.
StatusOr<bool> TryReshapeMoveOnCandidates(
    HloInstructionSet* reshape_candidates) {
  bool removed = true;
  while (!reshape_candidates->empty() && removed) {
    if (VLOG_IS_ON(5)) {
      for (const HloInstruction* instruction : *reshape_candidates) {
        VLOG(5) << "candidate " << instruction->ToString();
      }
    }
    ConstHloInstructionSet nontrivial_operands;
    for (const HloInstruction* instruction : *reshape_candidates) {
      for (const auto* operand : instruction->operands()) {
        if (IsNontrivialReshape(operand)) {
          nontrivial_operands.insert(operand);
        }
      }
    }

    removed = false;
    for (auto operand : nontrivial_operands) {
      if (c_any_of(operand->users(), [&](HloInstruction* user) {
            return !reshape_candidates->count(user);
          })) {
        for (auto* user : operand->users()) {
          removed |= reshape_candidates->erase(user) > 0;
        }
      }
    }
  }

  if (reshape_candidates->empty()) {
    return false;
  }
  for (HloInstruction* instruction : *reshape_candidates) {
    const HloInstruction* first_reshape_operand =
        FirstNonScalarAndNonTrivialReshapeOperand(instruction);
    TF_ASSIGN_OR_RETURN(
        bool did_change,
        PerformSinkReshapeOrTranspose(instruction, first_reshape_operand));
    CHECK(did_change);
  }
  return true;
}

}  // namespace

StatusOr<bool> ReshapeMover::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Pre ReshapeMover HLO:";
  XLA_VLOG_LINES(2, module->ToString());
  for (auto* comp : module->MakeNonfusionComputations()) {
    HloInstructionSet reshape_candidates;
    for (HloInstruction* instruction : comp->instructions()) {
      if (IsReshapeMoveCandidate(instruction)) {
        reshape_candidates.insert(instruction);
      }
    }
    TF_ASSIGN_OR_RETURN(bool did_change,
                        TryReshapeMoveOnCandidates(&reshape_candidates));
    changed |= did_change;
  }
  VLOG(2) << "Post ReshapeMover HLO:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace xla
