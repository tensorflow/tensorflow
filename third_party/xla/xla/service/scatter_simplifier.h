/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SCATTER_SIMPLIFIER_H_
#define XLA_SERVICE_SCATTER_SIMPLIFIER_H_

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// This pass rewrites scatter operations into a combination of transposes,
// reshapes and a simpler scatter.
//
// It implements the first two steps of the algorithm described in
// ScatterExpander::ExpandInstruction (scatter_expander.cc). Additionally, it
// transposes updates and operands to transform scatter_dims_to_operand_dims
// into the identity mapping. This is different from the algorithm in
// ScatterExpander, which instead applies the mapping in scatter_indices.
//
// The semantics of the output "simple" scatter are indeed simpler than that
// of a general scatter. If a scatter is simple (see IsSimplifiedScatter() for
// the necessary restrictions), only the following arguments/flags are relevant:
//
// Arguments:
//   - operand: tensor<D_1 x ... x D_N x elem_type>. Tensor to scatter into.
//   - updates: tensor<M x d_1 x ... x d_N x elem_type>. M N-dimensional slices,
//     to be combined with `operand`.
//   - indices: tensor<M x n x index>, where n <= N. M lists of indices.
//     If n < N, the remaining indices are set to 0.
//   - update_computation: computation to combine `updates` and `operand`.
//
// Flags: (only used for optimizations)
//   - indices_are_sorted (bool): whether the indices are sorted.
//   - unique_indices (bool): whether the indices are unique.
//
// A reference implementation would apply the M update slices in random order.
// In pseudo-code:
//   if n < N:
//     "Fill `indices` with trailing zeroes so that it is of shape M x N"
//   for update, index = shuffle(zip(updates, indices)) {  # M iterations.
//     assert(update.dimensions_size() == len(index) == N)
//     if "update fits in operand at index":
//       operand[index] = update_computation(operand[index], update)
//
// Examples of simple scatter can be found in scatter_simplifier_test.cc.
class ScatterSimplifier : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "scatter_simplifier"; }

  static bool IsSimplifiedScatter(const HloScatterInstruction* scatter);

 protected:
  bool InstructionMatchesPattern(HloInstruction* inst) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* inst) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_SCATTER_SIMPLIFIER_H_
