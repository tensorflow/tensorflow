/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_LOWERING_UTILS_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_LOWERING_UTILS_H_

#include <cstdint>

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::triton::xla {

using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;

enum class DotOperandSide { kLhs, kRhs };

// Lowers stablehlo.reshape to ttir.reshape, ttir.splat, or ttir.unsplat.
class LowerReshape : public mlir::OpRewritePattern<stablehlo::ReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      stablehlo::ReshapeOp op, mlir::PatternRewriter& rewriter) const override;
};

// Returns the AddOp and its non-dot operand (accumulator) if the 'op' is
// consumed by an AddOp. This is used to identify GEMM fusion units.
mlir::LogicalResult GetFusedAddUnit(mlir::Operation* op,
                                    mlir::PatternRewriter& rewriter,
                                    mlir::Operation*& add_op,
                                    mlir::Value& accumulator);

// Canonicalizes a dot operand to match Triton hardware rank (2D).
mlir::LogicalResult CanonicalizeOperand(mlir::ImplicitLocOpBuilder& b,
                                        mlir::Value& operand,
                                        int64_t contracting_dim_idx,
                                        DotOperandSide side);

// Aligns the MJIR rank of a fused AddOp after hardware math lowering by
// inserting rank-restoration reshapes.
mlir::LogicalResult CanonicalizeFusedAddUnit(mlir::Operation* add_op,
                                             mlir::Value math_result,
                                             mlir::Value accumulator,
                                             mlir::PatternRewriter& rewriter);

// Returns true if the dot dimension numbers specify exactly one contracting
// dimension for each operand.
inline bool IsDotHasOneContractingDimension(
    mlir::stablehlo::DotDimensionNumbersAttr dims) {
  return dims.getLhsContractingDimensions().size() == 1 &&
         dims.getRhsContractingDimensions().size() == 1;
}

// Returns true if the dot dimension numbers are in the canonical 2D hardware-
// compatible format: no batch dimensions, contracting dims are [1] x [0].
inline bool IsDotDimensionNumbersCanonical(
    mlir::stablehlo::DotDimensionNumbersAttr dims) {
  return IsDotHasOneContractingDimension(dims) &&
         dims.getLhsBatchingDimensions().empty() &&
         dims.getRhsBatchingDimensions().empty() &&
         dims.getLhsContractingDimensions()[0] == 1 &&
         dims.getRhsContractingDimensions()[0] == 0;
}

}  // namespace mlir::triton::xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TRANSFORMS_LOWERING_UTILS_H_
