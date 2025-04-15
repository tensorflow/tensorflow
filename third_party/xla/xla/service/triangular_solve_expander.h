/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_TRIANGULAR_SOLVE_EXPANDER_H_
#define XLA_SERVICE_TRIANGULAR_SOLVE_EXPANDER_H_

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

class TriangularSolveExpander : public OpExpanderPass {
 public:
  explicit TriangularSolveExpander(int64_t block_size = 128);

  absl::string_view name() const override {
    return "triangular_solve_expander";
  }

 protected:
  // Should we use direct solves for batched inputs?
  virtual bool UseDirectSolves() const { return true; }

  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

  // Performs a triangular solve using an algorithm from MAGMA, which inverts
  // diagonal blocks and multiplies them using matrix multiplications.
  XlaOp SolveByInvertingDiagonalBlocks(XlaOp a, XlaOp b, bool left_side,
                                       bool lower, bool transpose_a,
                                       bool conjugate_a, bool unit_diagonal,
                                       PrecisionConfig::Precision precision);

  // Helper function used by SolveByInvertingDiagonalBlocks
  virtual XlaOp InvertDiagonalBlocks(XlaOp diag_blocks, bool lower_triangular,
                                     PrecisionConfig::Precision precision);

  // Performs a direct triangular solve, suitable for case with small matrices
  // or with large batch.
  XlaOp SolveDirectly(XlaOp a, XlaOp b, bool left_side, bool lower,
                      bool transpose_a, bool conjugate_a, bool unit_diagonal,
                      PrecisionConfig::Precision precision);

  XlaOp BuildTriangularSolve(XlaOp a, XlaOp b, bool left_side, bool lower,
                             bool transpose_a, bool conjugate_a,
                             bool unit_diagonal, int64_t block_size,
                             PrecisionConfig::Precision precision);

 private:
  // Block size for BuildTriangularSolve
  const int64_t block_size_;
  // Mapping from op signatures to existing computations.
  absl::flat_hash_map<std::string, HloComputation*> computation_cache_;
};

}  // namespace xla

#endif  // XLA_SERVICE_TRIANGULAR_SOLVE_EXPANDER_H_
