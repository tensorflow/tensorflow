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

#include "xla/backends/gpu/codegen/triton/transforms/lowering_utils.h"

#include <cstdint>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

namespace {

// Returns `shape` without all its unit dimensions, as well as the index of the
// remaining dimensions in the original `shape`.
std::pair<llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>>
CollapseUnitDims(llvm::ArrayRef<int64_t> shape,
                 llvm::ArrayRef<int64_t> counterpart_shape) {
  CHECK_EQ(shape.size(), counterpart_shape.size())
      << "CollapseUnitDims requires aligned ranks for operand and counterpart.";
  llvm::SmallVector<int64_t> shape_without_unit_dims;
  llvm::SmallVector<int64_t> non_unit_dims_indices;
  for (auto [i, size] : llvm::enumerate(shape)) {
    if (size != 1 || size != counterpart_shape[i]) {
      shape_without_unit_dims.push_back(size);
      non_unit_dims_indices.push_back(i);
    }
  }
  return {std::move(shape_without_unit_dims), std::move(non_unit_dims_indices)};
}

// Canonicalizes the given operand of a dot operation, i.e. make it a 2D tensor,
// and make sure that the contracting dimension is where we expect it to be for
// the given side (the second dimension for LHS, the first dimension for the
// RHS).
//
// If it is a scaled-dot scale operand then we drop the extra dims only
// when they equal to 1  and are matching with the corresponding operand.
//
// Returns an error if canonicalization is not possible.
absl::StatusOr<TensorValue> CanonicalizeDotOperand(
    mlir::ImplicitLocOpBuilder& b, TensorValue operand,
    int64_t contracting_dim_idx, DotOperandSide side,
    TensorValue counterpart_operand = nullptr) {
  llvm::ArrayRef<int64_t> shape = operand.getType().getShape();
  llvm::ArrayRef<int64_t> counterpart_shape =
      counterpart_operand == nullptr ? shape
                                     : counterpart_operand.getType().getShape();

  auto [shape_without_unit_dims, non_unit_dims_indices] =
      CollapseUnitDims(shape, counterpart_shape);

  if (shape_without_unit_dims.size() != 2) {
    return absl::FailedPreconditionError(
        "Expected dot operand tile to have exactly two non-unit tile sizes");
  }
  if (shape.size() != shape_without_unit_dims.size()) {
    ASSIGN_OR_RETURN(operand, ::xla::xtile::EmitTiledReshape(
                                  b, shape_without_unit_dims, operand));
  }
  int expected_contracting_dim_position = side == DotOperandSide::kLhs ? 1 : 0;
  bool is_transposed =
      non_unit_dims_indices[expected_contracting_dim_position] !=
      contracting_dim_idx;

  if (is_transposed) {
    llvm::SmallVector<int64_t, 2> transposed_shape{shape_without_unit_dims[1],
                                                   shape_without_unit_dims[0]};
    operand = ::xla::xtile::EmitTiledTranspose(b, transposed_shape,
                                               /*dimensions=*/{1, 0}, operand);
  }
  return operand;
}

}  // namespace

mlir::LogicalResult GetFusedAddUnit(mlir::Operation* op,
                                    mlir::PatternRewriter& rewriter,
                                    mlir::Operation*& add_op,
                                    mlir::Value& accumulator) {
  const Location op_loc = op->getLoc();
  if (!op->hasOneUse()) {
    return rewriter.notifyMatchFailure(
        op_loc,
        "Dot op must have exactly one user in order to be lowered to triton.");
  }

  add_op = *op->getUsers().begin();
  if (!mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(add_op)) {
    return rewriter.notifyMatchFailure(
        op_loc,
        "Dot op must be consumed by an AddOp to be convertible to triton dot.");
  }

  accumulator = add_op->getOperand(0) == op->getResult(0)
                    ? add_op->getOperand(1)
                    : add_op->getOperand(0);
  return mlir::success();
}

mlir::LogicalResult CanonicalizeOperand(mlir::ImplicitLocOpBuilder& b,
                                        mlir::Value& operand,
                                        int64_t contracting_dim_idx,
                                        DotOperandSide side) {
  auto operand_tensor = mlir::cast<TensorValue>(operand);
  absl::StatusOr<TensorValue> canonical =
      CanonicalizeDotOperand(b, operand_tensor, contracting_dim_idx, side);
  if (!canonical.ok()) {
    return mlir::failure();
  }
  operand = *canonical;
  if (mlir::cast<mlir::ShapedType>(operand.getType()).getRank() != 2) {
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult CanonicalizeFusedAddUnit(mlir::Operation* add_op,
                                             mlir::Value math_result,
                                             mlir::Value accumulator,
                                             mlir::PatternRewriter& rewriter) {
  mlir::RankedTensorType new_result_type =
      mlir::cast<mlir::RankedTensorType>(math_result.getType());
  const mlir::Location op_loc = add_op->getLoc();
  mlir::ImplicitLocOpBuilder builder(op_loc, rewriter);

  absl::StatusOr<::xla::xtile::TensorValue> acc_canonical =
      ::xla::xtile::EmitTiledReshape(builder, new_result_type.getShape(),
                                     mlir::cast<TensorValue>(accumulator));
  if (!acc_canonical.ok()) {
    return rewriter.notifyMatchFailure(op_loc,
                                       "Failed to canonicalize accumulator.");
  }

  mlir::Value new_add;
  if (mlir::isa<mlir::IntegerType>(new_result_type.getElementType())) {
    new_add = mlir::arith::AddIOp::create(builder, math_result, *acc_canonical);
  } else {
    new_add = mlir::arith::AddFOp::create(builder, math_result, *acc_canonical);
  }

  llvm::ArrayRef<int64_t> result_shape =
      mlir::cast<mlir::ShapedType>(add_op->getResult(0).getType()).getShape();
  absl::StatusOr<::xla::xtile::TensorValue> reshaped_result =
      ::xla::xtile::EmitTiledReshape(builder, result_shape,
                                     mlir::cast<TensorValue>(new_add));
  if (!reshaped_result.ok()) {
    return rewriter.notifyMatchFailure(op_loc, "Failed to reshape result.");
  }

  rewriter.replaceOp(add_op, *reshaped_result);
  return mlir::success();
}

mlir::LogicalResult LowerReshape::matchAndRewrite(
    stablehlo::ReshapeOp op, mlir::PatternRewriter& rewriter) const {
  bool input_is_0d = op.getOperand().getType().getRank() == 0;
  bool output_is_0d = op.getType().getRank() == 0;

  if (input_is_0d && output_is_0d) {
    rewriter.replaceAllUsesWith(op, op.getOperand());
    return mlir::success();
  }

  if (input_is_0d) {
    auto to_scalar = mlir::tensor::ExtractOp::create(rewriter, op->getLoc(),
                                                     op.getOperand());
    rewriter.replaceOpWithNewOp<ttir::SplatOp>(op, op.getType(), to_scalar);
    return mlir::success();
  }

  if (output_is_0d) {
    auto input_tensor_type = op.getOperand().getType();
    auto element_type = input_tensor_type.getElementType();
    auto unsplat = ttir::UnsplatOp::create(rewriter, op.getLoc(), element_type,
                                           op.getOperand());
    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(
        op, op.getType(), unsplat.getResult());
    return mlir::success();
  }

  // Conservatively prevent Triton from reordering elements within the tile.
  // TODO(b/353637689): see if this restriction can be lifted.
  bool allow_reorder = false;
  rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(op, op.getResult().getType(),
                                               op.getOperand(), allow_reorder);
  return mlir::success();
}

}  // namespace mlir::triton::xla
