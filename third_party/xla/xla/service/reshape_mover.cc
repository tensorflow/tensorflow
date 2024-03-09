/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/reshape_mover.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "xla/permutation_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

namespace {

// In this file, let a "rearrange" op be a reshape or a transpose.
bool IsRearrange(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kReshape ||
         instruction->opcode() == HloOpcode::kTranspose;
}

// Returns whether `a` and `b` are equivalent reshapes/transposes.
bool AreEquivalentRearranges(const HloInstruction* a, const HloInstruction* b) {
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

// Computes where broadcast dims end up after a transpose.
//
// Consider a simple case:
//
//  bcast = f32[1,2,3,4] broadcast(f32[2,4] x), dimensions={1,3}
//  trans = f32[2,3,1,4] transpose(f32[1,2,3,4] bcast), dimensions={1,2,0,3}.
//
// We want to transform this into
//
//  bcast' = f32[2,3,1,4] broadcast(f32[2,4] x), dimensions={0,3}.
//
// The algorithm is:
//
//  * Invert the permutation {1,2,0,3} to give us p' = {2,0,1,3}.
//
//  * Compute where each broadcast dim ends up after the transpose.  p'[1] = 0,
//    meaning that broadcast dim 1 (size 2) ends up at index 0 after the
//    transpose.  Similarly, p'[3] = 3.
//
// Thus the new broadcast's dims are [p'[dim] for dim in bcast.dimensions()].
absl::InlinedVector<int64_t, 4> TransposedBcastDims(
    absl::Span<const int64_t> bcast_dims,
    absl::Span<const int64_t> transpose_dims) {
  auto inv_perm = InversePermutation(transpose_dims);
  absl::InlinedVector<int64_t, 4> new_bcast_dims;
  for (int64_t dim : bcast_dims) {
    new_bcast_dims.push_back(inv_perm[dim]);
  }
  return new_bcast_dims;
}

}  // namespace

// Returns true if `instr` can easily change its shape according to the inverse
// of `rearrange`, which must be a kReshape or kTranspose op.
bool ReshapeMover::CanTriviallyRearrange(const HloInstruction* instr,
                                         const HloInstruction* rearrange) {
  CHECK(IsRearrange(rearrange)) << rearrange->ToString();

  // Check for nop reshapes / transposes.  These are, by definition, trivial.
  // These "shouldn't happen", because algsimp should run before this pass.  But
  // sometimes they appear anyway, e.g. because algsimp does not run to a fixed
  // point before this pass runs.
  if (rearrange->opcode() == HloOpcode::kReshape &&
      ShapeUtil::Equal(rearrange->shape(), rearrange->operand(0)->shape())) {
    return true;
  }
  if (rearrange->opcode() == HloOpcode::kTranspose &&
      IsIdentityPermutation(rearrange->dimensions())) {
    return true;
  }

  // NOTE: Technically a sequence of rearrange(rearrange(constant)) is also
  // trivially rearrangeable, so we might be tempted to simply recurse if
  // instruction is kReshape or kTranspose.
  //
  // But it's not that simple. E.g. reshape(reshape(rng)) is only trivially
  // reshapable if *all* instructions in the chain have user_count == 1. And
  // reshape(scalar) isn't trivial at all if the reshape itself isn't scalar.
  //
  // So don't handle these chains, unless you update the tests and code to deal
  // with these properly. One idea is to add a pass immediately beforehand that
  // collapses trivial runs of reshapes / transposes.

  // A constant can trivially rearrange the literal it holds.
  if (instr->opcode() == HloOpcode::kConstant) {
    return true;
  }

  // An Rng instruction can be any shape as long as it has one user. Two copies
  // of the same Rng would be problematic if an Rng of a different shape would
  // produce random numbers in a different order.
  if (instr->opcode() == HloOpcode::kRng && instr->user_count() == 1) {
    return true;
  }

  if (instr->opcode() == HloOpcode::kBroadcast) {
    // Cowardly refuse to handle broadcasts where the broadcast dims are not
    // sorted.  Such broadcasts are basically transposes, which is confusing.
    if (!absl::c_is_sorted(instr->dimensions())) {
      return false;
    }

    // reshape(broadcast(x)) is trivially representable as broadcast'(x) if
    //  * the reshape does not modify any of the broadcasted dims, or
    //  * x is scalar or effective rank 1 (in which case, the reshape is trivial
    //    even if it *does* modify broadcasted dims).
    //
    // (It doesn't really matter, but note that we go *from* rearrange->shape()
    // *to* its operand shape -- not the other way around -- because we're
    // interested in applying the *inverse* of the rearrange.
    //
    // TODO(jlebar): Get rid of the reshape_of_1d_broadcast_is_cheap check on
    // the ReshapeLeavesDimensionsUnmodified branch.  I think this is needed
    // only because algsimp doesn't currently do this simplification itself.
    if (rearrange->opcode() == HloOpcode::kReshape) {
      return ShapeUtil::IsScalar(instr->operand(0)->shape()) ||
             (options_.reshape_of_1d_broadcast_is_cheap &&
              ShapeUtil::TrueRank(instr->operand(0)->shape()) <= 1) ||
             (options_.reshape_of_1d_broadcast_is_cheap &&
              ShapeUtil::ReshapeLeavesDimensionsUnmodified(
                  /*from_shape=*/rearrange->shape(),
                  /*to_shape=*/rearrange->operand(0)->shape(),
                  instr->dimensions())
                  .has_value());
    }

    // Similarly, transpose(broadcast(x)) is trivially representable as
    // broadcast'(x) if the transpose does not change the relative order of any
    // of the broadcasted dims.
    //
    // (The permutation we're interested in is the inverse of `transpose`
    // because we're considering applying transpose' to the broadcast operand.
    // Although like in the case of kReshape, this doesn't really matter,
    // because the inverse permutation leaves the relative order of the dims
    // unchanged iff the non-inverse permutation leaves them unchanged.)
    if (rearrange->opcode() == HloOpcode::kTranspose) {
      return absl::c_is_sorted(TransposedBcastDims(
          instr->dimensions(), InversePermutation(rearrange->dimensions())));
    }
  }

  return false;
}

const HloInstruction* ReshapeMover::FirstNontrivialRearrange(
    absl::Span<const HloInstruction* const> instrs) {
  auto rearrange_it = absl::c_find_if(instrs, [&](const HloInstruction* instr) {
    return IsRearrange(instr) &&
           !CanTriviallyRearrange(instr->operand(0), instr);
  });
  if (rearrange_it == instrs.end()) {
    return nullptr;
  }
  return *rearrange_it;
}

// Returns true if the instruction is a reshape-move candidate:
//
//   * at least one operand is a rearrange, and
//   * all rearrange operands are equivalent (if there's more than one), and
//   * we can trivially apply the inverse rearrange to all other operands.
bool ReshapeMover::IsReshapeMoveCandidate(HloInstruction* instruction) {
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);
  VLOG(5) << "** Checking instruction: "
          << instruction->ToString(print_no_metadata);

  // Only perform reshape-move for elementwise instructions.
  if (!instruction->IsElementwise()) {
    return false;
  }

  const HloInstruction* rearrange =
      FirstNontrivialRearrange(instruction->operands());
  if (rearrange == nullptr) {
    return false;
  }
  return absl::c_all_of(
      instruction->operands(), [&](const HloInstruction* operand) {
        return (IsRearrange(operand) &&
                AreEquivalentRearranges(operand, rearrange)) ||
               (!IsRearrange(operand) &&
                CanTriviallyRearrange(operand, rearrange));
      });
}

// Returns a reshape/transpose of `operand` according to the inverse of
// `rearrange`.
//
// This will often create redundant operations that we expect to be eliminated
// by algsimp.  For example, if we have an operand rearrange(x), this will
// produce rearrange'(rearrange(x)), which can be simplified to x.
absl::StatusOr<HloInstruction*> ReshapeMover::ApplyInverseRearrange(
    const HloInstruction* rearrange, HloInstruction* operand) {
  switch (rearrange->opcode()) {
    case HloOpcode::kReshape: {
      // To make algsimp's life a little easier, don't insert a nop reshape.
      Shape new_shape = ShapeUtil::ChangeElementType(
          rearrange->operand(0)->shape(), operand->shape().element_type());
      if (operand->shape() != new_shape) {
        return MakeReshapeHlo(new_shape, operand);
      } else {
        return operand;
      }
    }
    case HloOpcode::kTranspose: {
      // To make algsimp's life a little easier, don't insert a nop transpose.
      if (!IsIdentityPermutation(rearrange->dimensions())) {
        return MakeTransposeHlo(operand,
                                InversePermutation(rearrange->dimensions()));
      } else {
        return operand;
      }
    }
    default:
      LOG(FATAL) << "Invalid rearrange op: " << rearrange->ToString();
  }
}

// Actually performs the reshape-move transformation -- that is, sinks the
// reshape or transpose operands of `instruction` across it.
absl::StatusOr<bool> ReshapeMover::SinkRearrangeOperands(
    HloInstruction* instruction) {
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);

  HloComputation* computation = instruction->parent();

  const HloInstruction* rearrange =
      FirstNontrivialRearrange(instruction->operands());
  CHECK(rearrange != nullptr);

  const Shape& new_operand_shape = rearrange->operand(0)->shape();
  VLOG(3) << "** Sinking reshape or transpose: "
          << instruction->ToString(print_no_metadata)
          << "\n\tfirst rearrange operand: "
          << rearrange->ToString(print_no_metadata)  //
          << "\n\tnew operand shape: "
          << ShapeUtil::HumanString(new_operand_shape);

  auto operands = instruction->operands();
  for (size_t i = 0; i < operands.size(); ++i) {
    VLOG(3) << "Updating operand #" << i << ": "
            << operands[i]->ToString(print_no_metadata);
    TF_ASSIGN_OR_RETURN(operands[i],
                        ApplyInverseRearrange(rearrange, operands[i]));
    VLOG(3) << "Updated operand #" << i
            << " to: " << operands[i]->ToString(print_no_metadata);
  }

  HloInstruction* new_elementwise =
      computation->AddInstruction(instruction->CloneWithNewOperands(
          ShapeUtil::ChangeElementType(new_operand_shape,
                                       instruction->shape().element_type()),
          operands));

  std::unique_ptr<HloInstruction> new_rearrange;
  switch (rearrange->opcode()) {
    case HloOpcode::kReshape:
      VLOG(3) << "Creating new reshape for new elementwise op: "
              << new_elementwise->ToString(print_no_metadata);
      new_rearrange =
          HloInstruction::CreateReshape(instruction->shape(), new_elementwise);
      break;
    case HloOpcode::kTranspose:
      new_rearrange = HloInstruction::CreateTranspose(
          instruction->shape(), new_elementwise, rearrange->dimensions());
      break;
    default:
      LOG(FATAL) << "Bad opcode";
  }

  // Sinking the rearrange ops can change the shape of the elementwise op. This
  // may make any sharding annotations (which, as of now, can only be present if
  // auto-sharding is turned on) on the instruction out of sync. We therefore
  // drop any shardings here.
  if (instruction->has_sharding()) {
    new_elementwise->clear_sharding();
  }

  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      instruction, std::move(new_rearrange)));
  return true;
}

// Reshape-moves all qualifying instructions in candidates.  Returns true if it
// makes changes.
//
// `candidates` is a set of HloInstructions with rearrange operands, and a
// instruction in the set can be reshape-moved iff all the users of its
// rearrange operands can also be reshaped-moved.
//
// The algorithm here iteratively finds the rearrange operands with users that
// are outside the set of `candidates`, and removes their users from
// `candidates`, until either `candidates` becomes empty or none of the
// remaining rearrange operands have users outside `candidates`.  In the later
// case, all the remaining instructions in `candidates` are reshape-moved and
// the routine returns true.
absl::StatusOr<bool> ReshapeMover::TryReshapeMoveOnCandidates(
    HloInstructionSet* candidates) {
  bool removed = true;
  while (!candidates->empty() && removed) {
    if (VLOG_IS_ON(5)) {
      for (const HloInstruction* instruction : *candidates) {
        VLOG(5) << "candidate " << instruction->ToString();
      }
    }
    ConstHloInstructionSet rearrange_operands;
    for (const HloInstruction* instruction : *candidates) {
      for (const auto* operand : instruction->operands()) {
        if (IsRearrange(operand)) {
          rearrange_operands.insert(operand);
        }
      }
    }

    removed = false;
    for (auto operand : rearrange_operands) {
      if (absl::c_any_of(operand->users(), [&](HloInstruction* user) {
            return !candidates->count(user);
          })) {
        for (auto* user : operand->users()) {
          removed |= candidates->erase(user) > 0;
        }
      }
    }
  }

  if (candidates->empty()) {
    return false;
  }
  for (HloInstruction* instruction : *candidates) {
    if (!ConsumeFuel("reshape-mover", [&] {
          return absl::StrCat("instruction: ", instruction->ToString(),
                              "\nFull module:\n",
                              instruction->GetModule()->ToString());
        })) {
      break;
    }
    TF_ASSIGN_OR_RETURN(bool did_change, SinkRearrangeOperands(instruction));
    CHECK(did_change);
  }
  return true;
}

absl::StatusOr<bool> ReshapeMover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    HloInstructionSet candidates;
    for (HloInstruction* instruction : comp->instructions()) {
      if (IsReshapeMoveCandidate(instruction)) {
        candidates.insert(instruction);
      }
    }
    TF_ASSIGN_OR_RETURN(bool did_change,
                        TryReshapeMoveOnCandidates(&candidates));
    changed |= did_change;
  }
  return changed;
}

}  // namespace xla
