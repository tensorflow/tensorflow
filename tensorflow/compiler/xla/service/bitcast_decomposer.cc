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

#include "tensorflow/compiler/xla/service/bitcast_decomposer.h"

#include <vector>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

namespace {
absl::InlinedVector<int64_t, 8> ReverseIota(int64_t n) {
  absl::InlinedVector<int64_t, 8> ret;
  for (int64_t i = n - 1; i >= 0; --i) {
    ret.push_back(i);
  }
  return ret;
}

int64_t Rank(const Shape& s) {
  return static_cast<int64_t>(s.dimensions().size());
}
int64_t Rank(const HloInstruction* instr) { return Rank(instr->shape()); }

absl::Span<const int64_t> LayoutPerm(const Shape& s) {
  return s.layout().minor_to_major();
}
absl::Span<const int64_t> LayoutPerm(const HloInstruction* instr) {
  return LayoutPerm(instr->shape());
}

}  // anonymous namespace

// Terminology:
//  rib: a bitcast with reshape-is-bitcast(src, dst) == true.
//  tib: a transpose with transpose-is-bitcast(src, dst) == true.
//
// Significant fact: Given a shape with a descending layout (e.g. {3,2,1,0}), a
// rib can arbitrarily rearrange its logical dims. This is because a reshape can
// be thought of as the following operation.
//   - Transpose from input layout to descending layout, then
//   - reinterpret_cast to new logical shape, and finally
//   - transpose from descending layout to output layout.
// If the input/output layouts are both descending, then the first and last
// parts of the operation are nops, and the whole operation becomes a plain
// reinterpret_cast, also known as a bitcast.
//
// This pass decomposes an arbitrary bitcast into tib(rib(tib(input))), although
// sometimes we can leave out one or more of these ops.
//
// Algorithm for decomposing a bitcast into a sequence of tibs and ribs:
//
//  - First, use a tib to get a descending layout.
//  - Per the significant fact above, we can now use a rib to arbitrarily
//    rearrange its logical dims.
//  - We can also use a tib to arbitrarily permute its layout (along with
//    its logical dims) -- that's what a tib does.
//  - So working backwards, first find a tib that gives us the desired layout.
//    Then choose a set of logical dims for the rib so that when they are
//    permuted by the tib we chose, we get the desired output logical dims.
StatusOr<bool> BitcastDecomposer::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->computations()) {
    if (!comp->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kBitcast || !instr->shape().IsArray() ||
          instr->shape().is_dynamic()) {
        continue;
      }
      const Shape& dst_shape = instr->shape();
      PrimitiveType elem_ty = instr->shape().element_type();
      if (ShapeUtil::ReshapeIsBitcast(instr->operand(0)->shape(),
                                      instr->shape())) {
        VLOG(3) << "Not decomposing instruction; it is already a "
                   "reshape-is-bitcast: "
                << instr->ToString();
        continue;
      }

      if (!ConsumeFuel("bitcast_decomposer", [&] {
            return absl::StrCat("Not decomposing ", instr->ToString());
          })) {
        continue;
      }

      VLOG(1) << "Decomposing instruction: " << instr->ToString();

      // Important fact: A tib is a transpose which satisfies
      //   in_layout = perm . out_layout =>
      //   perm = in_layout . out_layout'
      auto create_tib = [&](const Shape& shape, HloInstruction* input,
                            absl::Span<const int64_t> perm) {
        CHECK(ShapeUtil::TransposeIsBitcast(input->shape(), shape, perm))
            << "from=" << input->shape() << " to=" << shape
            << " perm=" << absl::StrJoin(perm, ",");
        HloInstruction* ret = comp->AddInstruction(
            HloInstruction::CreateTranspose(shape, input, perm));
        VLOG(3) << "Transposed " << input->ToString()
                << " to: " << ret->ToString();
        return ret;
      };

      auto create_rib = [&](const Shape& shape, HloInstruction* input) {
        CHECK(ShapeUtil::ReshapeIsBitcast(input->shape(), shape))
            << "from=" << ShapeUtil::HumanStringWithLayout(input->shape())
            << " to=" << ShapeUtil::HumanStringWithLayout(shape);
        HloInstruction* ret =
            comp->AddInstruction(HloInstruction::CreateBitcast(shape, input));
        VLOG(3) << "Reshaped " << input->ToString()
                << " to: " << ret->ToString();
        return ret;
      };

      // Check if we can transform the bitcast instr into just a single tib,
      // without a rib at all. (Normally we'd rely on algsimp to fix this
      // instead, but this pass runs very late and makes the graph "more
      // complex", so we don't want to run algsimp after it.)
      //
      // A tib is a transpose which satisfies
      //   perm = in_layout . out_layout',
      // so we know what perm we'd have to use if this bitcast is a single tib.
      //
      // Check whether permuting the input logical dims according to perm gives
      // us the desired output dims -- if so, we can just use the single tib.
      HloInstruction* operand = instr->mutable_operand(0);
      if (Rank(operand) == Rank(dst_shape)) {
        std::vector<int64_t> transpose_perm =
            ComposePermutations(LayoutPerm(instr->operand(0)),
                                InversePermutation(LayoutPerm(instr)));
        std::vector<int64_t> new_dims =
            ComposePermutations(operand->shape().dimensions(), transpose_perm);
        if (absl::c_equal(instr->shape().dimensions(), new_dims)) {
          TF_RETURN_IF_ERROR(instr->parent()->ReplaceInstruction(
              instr, create_tib(instr->shape(), operand, transpose_perm)));
          changed = true;
          continue;
        }
      }

      // Use a tib to convert to descending layout if operand doesn't already
      // have a descending layout.
      if (!absl::c_equal(LayoutPerm(operand), ReverseIota(Rank(operand)))) {
        Shape new_shape =
            ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
                operand->shape());
        // A tib has perm = in_layout . out_layout'. In this case out_layout is
        // the descending layout, which is its own inverse.
        operand = create_tib(new_shape, operand,
                             ComposePermutations(LayoutPerm(operand),
                                                 ReverseIota(Rank(operand))));
      }

      // At this point, operand has a descending layout and we can use a rib
      // to arbitrarily rearrange its dims.
      //
      // We want to emit tib(rib(operand)) such that the result has the correct
      // logical dims and layout. Notice that the rib does not change the
      // layout, only the logical dims. Therefore to figure out the tib's
      // permutation PT, we need only look at the layout of operand.  Then we
      // can work backwards to choose a rib shape which, when permuted by PT,
      // gives the desired final logical dims.
      //
      // A tib satisfies
      //   perm = in_layout . out_layout',
      // which in this case is
      //  TP = operand.layout() . dst_shape.layout()'.
      //
      // operand has descending layout, so we can simplify this further to:
      //  TP = descending_layout . dst_shape.layout().
      std::vector<int64_t> transpose_perm =
          ComposePermutations(ReverseIota(Rank(dst_shape)),
                              InversePermutation(LayoutPerm(dst_shape)));

      // Now we solve for the reshape output shape R.
      //
      //   R . PT = dst_shape.dimensions() =>
      //   R = dst_shape.dimensions() . PT'
      Shape reshape_shape = ShapeUtil::MakeShapeWithDescendingLayout(
          elem_ty, ComposePermutations(dst_shape.dimensions(),
                                       InversePermutation(transpose_perm)));

      if (!ShapeUtil::Equal(operand->shape(), reshape_shape)) {
        operand = create_rib(reshape_shape, operand);
      }

      if (!ShapeUtil::Equal(operand->shape(), dst_shape)) {
        operand = create_tib(dst_shape, operand, transpose_perm);
      }

      if (instr != operand) {
        TF_RETURN_IF_ERROR(instr->parent()->ReplaceInstruction(instr, operand));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
