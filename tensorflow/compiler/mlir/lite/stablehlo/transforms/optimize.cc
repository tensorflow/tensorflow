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

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

// Convert mhlo.dot to mhlo.dot_general.
LogicalResult ConvertDotToDotGeneral(mhlo::DotOp op,
                                     PatternRewriter &rewriter) {
  auto lhs_type = op.getLhs().getType().cast<ShapedType>();
  auto rhs_type = op.getRhs().getType().cast<ShapedType>();
  if (!lhs_type.hasRank() || !rhs_type.hasRank()) {
    return rewriter.notifyMatchFailure(op, "unsupported unranked input type");
  }
  if (lhs_type.getRank() < 1 || 2 < lhs_type.getRank() ||
      rhs_type.getRank() < 1 || 2 < rhs_type.getRank()) {
    return rewriter.notifyMatchFailure(
        op,
        "unsupported dot operation type; operands must be vectors or "
        "matrices");
  }
  rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
      op, op.getType(), op.getLhs(), op.getRhs(),
      mhlo::DotDimensionNumbersAttr::get(
          op.getContext(),
          /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimensions=*/{lhs_type.getRank() - 1},
          /*rhsContractingDimensions=*/{0}),
      op.getPrecisionConfigAttr());
  return success();
}

// Convert reshape(dot_general(reshape(%y), %z)) to
// dot_general(%y, %z) if possible.
LogicalResult RemoveReshapeAroundDotGeneral(mhlo::ReshapeOp reshape_after,
                                            PatternRewriter &rewriter) {
  auto dot = dyn_cast_or_null<mhlo::DotGeneralOp>(
      reshape_after.getOperand().getDefiningOp());
  if (!dot) return failure();

  auto reshape_before =
      dyn_cast_or_null<mhlo::ReshapeOp>(dot.getLhs().getDefiningOp());
  if (!reshape_before) return failure();

  if (!dot.getLhs().getType().hasStaticShape() ||
      !dot.getRhs().getType().hasStaticShape() ||
      !reshape_before.getOperand().getType().hasStaticShape() ||
      !dot.getType().hasStaticShape() ||
      !reshape_after.getType().hasStaticShape()) {
    return rewriter.notifyMatchFailure(reshape_after,
                                       "dynamic shapes not supported");
  }

  const auto range = [](int64_t begin, int n) {
    SmallVector<int64_t> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
      result.push_back(i + begin);
    }
    return result;
  };

  // We only support the mhlo.dot style input layouts, i.e.:
  //   lhs: [batch, other dims, contract dims]
  //   rhs: [batch, contract dims, other dims]
  auto dim_nums = dot.getDotDimensionNumbers();
  int batch_dims_count = dim_nums.getLhsBatchingDimensions().size();
  int contracting_dims_count = dim_nums.getLhsContractingDimensions().size();
  if (dim_nums.getLhsBatchingDimensions() !=
          ArrayRef<int64_t>(range(0, batch_dims_count)) ||
      dim_nums.getRhsBatchingDimensions() !=
          ArrayRef<int64_t>(range(0, batch_dims_count)) ||
      dim_nums.getLhsContractingDimensions() !=
          ArrayRef<int64_t>(
              range(dot.getLhs().getType().getRank() - contracting_dims_count,
                    contracting_dims_count)) ||
      dim_nums.getRhsContractingDimensions() !=
          ArrayRef<int64_t>(range(batch_dims_count, contracting_dims_count))) {
    return rewriter.notifyMatchFailure(reshape_after,
                                       "unsupported dot_general layout");
  }

  // (B = batch dims, C = contracting dims, Y/Z = other dims)
  //
  // This pattern converts:
  //   %before = "mhlo.reshape"(%lhs : BxY1xC) : BxY2xC
  //   %dot = "mhlo.dot_general"(%before, %rhs : BxCxZ) : BxY2xZ
  //   %after = "mhlo.reshape"(%dot) : BxY1xZ
  // to:
  //   %dot : "mhlo.dot_general"(%lhs : BxY1xC, %rhs : BxCxZ) : BxY1xZ

  // Extract B, Y1, C from %lhs.
  ArrayRef<int64_t> shape_lhs =
      reshape_before.getOperand().getType().getShape();
  ArrayRef<int64_t> shape_b = shape_lhs.take_front(batch_dims_count);
  ArrayRef<int64_t> shape_c = shape_lhs.take_back(contracting_dims_count);
  ArrayRef<int64_t> shape_y1 =
      shape_lhs.drop_front(shape_b.size()).drop_back(shape_c.size());

  // Check %before shape, and extract Y2 from it.
  ArrayRef<int64_t> shape_before = reshape_before.getType().getShape();
  if (shape_before.take_front(shape_b.size()) != shape_b ||
      shape_before.take_back(shape_c.size()) != shape_c) {
    return failure();
  }
  ArrayRef<int64_t> shape_y2 =
      shape_before.drop_front(shape_b.size()).drop_back(shape_c.size());

  // No need to check %dot; dot_general verifier ensures correct shapes.
  // Extract Z from %dot.
  ArrayRef<int64_t> shape_z =
      dot.getType().getShape().drop_front(shape_b.size() + shape_y2.size());

  // Check %after shape.
  if (reshape_after.getType().getShape() !=
      ArrayRef<int64_t>(llvm::to_vector(
          llvm::concat<const int64_t>(shape_b, shape_y1, shape_z)))) {
    return failure();
  }

  rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
      reshape_after, reshape_after.getType(), reshape_before.getOperand(),
      dot.getRhs(),
      mhlo::DotDimensionNumbersAttr::get(
          reshape_after.getContext(),
          /*lhsBatchingDimensions=*/range(0, batch_dims_count),
          /*rhsBatchingDimensions=*/range(0, batch_dims_count),
          /*lhsContractingDimensions=*/
          range(batch_dims_count + shape_y1.size(), contracting_dims_count),
          /*rhsContractingDimensions=*/
          range(batch_dims_count, contracting_dims_count)),
      dot.getPrecisionConfigAttr());
  return success();
}

class OptimizePass
    : public PassWrapper<OptimizePass, OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const final { return "mhlo-optimize"; }
  StringRef getDescription() const final {
    return "Applies various optimizations on MHLO IR";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(ConvertDotToDotGeneral);
    patterns.add(RemoveReshapeAroundDotGeneral);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createOptimizePass() {
  return std::make_unique<OptimizePass>();
}

static PassRegistration<OptimizePass> pass;

}  // namespace odml
}  // namespace mlir
