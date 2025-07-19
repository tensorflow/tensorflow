/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_REDUCE_WINDOW_REWRITER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_REDUCE_WINDOW_REWRITER_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/shape.h"

namespace xla {

// Rewrite ReduceWindow to be more performant in cases it is written in a
// quadratic way:
//
// 1) Work around unimplemented cases in the implementation of ReduceWindow.
//
// This rewrites all R1 ReduceWindow nodes. We reshape the operand to an
// R2, perform the operation, and reshape back to R1. The reshapes correspond to
// a bitcast if the tensor length is less than or equal to a passed parameter.
// The motivation for this is to avoid use of overly large reductions and the
// complexities and restrictions therein.
//
// 2) Rewrite ReduceWindow ops that represent a CumSum/CumProd into a
// tree-reduction (see details in the implementation).
// Note that this may itself generate R1 ReduceWindow ops, which means this pass
// needs to be run to a fixed point.
class ReduceWindowRewriter : public HloModulePass {
 public:
  // `base_length` is a size of a reduce-window we are comfortable with
  // executing.
  explicit ReduceWindowRewriter(int64_t base_length)
      : base_length_(base_length) {}

  absl::string_view name() const override { return "reduce-window-rewriter"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Helper methods to optimize ReduceWindow ops.

  // Transposes the inputs if the scan dimension is not the last dimension.
  // Returns the permutation of the dimensions.
  std::vector<int64_t> GetTransposedInputs(HloComputation* hlo_computation,
                                           std::vector<HloInstruction*>& inputs,
                                           int64_t rank, int64_t scan_dim,
                                           int64_t last_dim);

  // Adds padding (if necessary) to enable further rewrites working properly.
  int64_t PreparePaddingForRewrite(HloReduceWindowInstruction* reduce_window,
                                   std::vector<HloInstruction*>& inputs,
                                   int64_t scan_length, int64_t last_dim);

  // [x, y] -> [x, y/base, base]
  int64_t ExpandToNewMajorDimension(HloComputation* hlo_computation,
                                    std::vector<HloInstruction*>& inputs,
                                    std::vector<HloInstruction*>& tiled_inputs,
                                    std::vector<Shape>& tiled_shapes,
                                    int64_t padded_length, int64_t last_dim);

  // reduce_window ( [x, y/base, base] window [1, 1, base] )
  HloInstruction* GenerateNewReduceWindowWithTiledInputs(
      HloReduceWindowInstruction* reduce_window,
      std::vector<HloInstruction*>& tiled_inputs,
      std::vector<Shape>& tiled_shapes, bool forward_scan);

  // Slice out the last (first if reverse scan) column.
  // slices [x, y/base, base] -> [x, y/base, 1] slice {x, y/base}
  // reshape [x, y/base, 1] -> [x, y/base]
  void SliceOutLastColumn(HloComputation* hlo_computation,
                          const Shape& subshape, HloInstruction* outer_shape,
                          int64_t rank, int64_t last_dim, bool forward_scan,
                          int64_t num_columns,
                          std::vector<Shape>& column_shapes,
                          std::vector<HloInstruction*>& last_cols);

  absl::Status ReplaceReduceWindowWithReshape(
      HloReduceWindowInstruction* reduce_window);

  absl::StatusOr<bool> TryOptimizeCumSumOrProd(
      HloReduceWindowInstruction* reduce_window);

  int64_t base_length_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_REDUCE_WINDOW_REWRITER_H_
