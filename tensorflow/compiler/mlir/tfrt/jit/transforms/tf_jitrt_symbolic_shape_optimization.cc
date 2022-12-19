/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <sys/types.h>

#include <memory>
#include <string>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Analysis/shape_component_analysis.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace tensorflow {
namespace {

using llvm::ArrayRef;
using llvm::SmallVector;

using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::failure;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::OperationPass;
using mlir::RankedTensorType;
using mlir::ShapeComponentAnalysis;
using mlir::success;
using mlir::TypeRange;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::ConstantIndexOp;
using mlir::arith::ConstantOp;
using mlir::arith::IndexCastOp;
using mlir::func::FuncOp;

namespace linalg = mlir::linalg;
namespace mhlo = mlir::mhlo;
namespace shape = mlir::shape;
namespace tensor = mlir::tensor;

#define GEN_PASS_DEF_SYMBOLICSHAPEOPTIMIZATION
#define GEN_PASS_DECL_SYMBOLICSHAPEOPTIMIZATION
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

// -------------------------------------------------------------------------- //

// Replace shape.broadcast with a shape if it's statically known.
class BroadcastOpLowering final
    : public mlir::OpRewritePattern<shape::BroadcastOp> {
 public:
  explicit BroadcastOpLowering(MLIRContext* ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(shape::BroadcastOp op,
                                mlir::PatternRewriter& rewriter) const override;
};

// Returns a shape tensor if the shapes can be broadcasted to a known shape.
// Will either return one of the shapes or a generated mix of the shapes.
llvm::Optional<Value> simplifyBroadcast(ShapeComponentAnalysis& analysis,
                                        ValueRange shapes, Location loc,
                                        OpBuilder* builder) {
  // First find the input shape with the largest rank.
  SmallVector<ArrayRef<ShapeComponentAnalysis::SymbolicExpr>> shapes_found;
  size_t maxRank = 0;
  for (const auto& shape : llvm::enumerate(shapes)) {
    auto found_shape = analysis.GetValueInfo(shape.value());
    if (!found_shape) return {};
    shapes_found.push_back(*found_shape);
    maxRank = std::max(maxRank, found_shape->size());
  }
  if (maxRank == 0) {
    return Value(builder->create<tensor::FromElementsOp>(
        loc, shapes[0].getType(), SmallVector<Value>()));
  }

  SmallVector<const ShapeComponentAnalysis::SymbolicExpr*> joined_dimensions(
      maxRank);
  SmallVector<std::pair<Value, int64_t>> shape_and_rank_for_dim(maxRank);
  for (const auto& shape : llvm::enumerate(shapes_found)) {
    for (const auto& dim : llvm::enumerate(llvm::reverse(shape.value()))) {
      // 1 dimensions don't contribute to the final result.
      if (dim.value().isConstant(1)) continue;
      // If it's not a 1 dimension it will be present in the result. Remember
      // where it came from.
      auto index = maxRank - dim.index() - 1;
      if (!joined_dimensions[index]) {
        joined_dimensions[index] = &dim.value();
        shape_and_rank_for_dim[index] =
            std::make_pair(shapes[shape.index()], shape.value().size());
        continue;
      }
      // Bail if the dimensions are neither equal nor 1.
      if (*joined_dimensions[index] != dim.value()) return {};
    }
  }
  // If the output is the same as one of the inputs just return that.
  if (llvm::all_equal(shape_and_rank_for_dim) &&
      shape_and_rank_for_dim[0].first) {
    return shape_and_rank_for_dim[0].first;
  }
  // Otherwise rematerialize the shape from the pieces we have.
  SmallVector<Value> elements;
  for (int i = 0; i != maxRank; ++i) {
    // 1 dimensions are filtered above, recreate the constant.
    if (!shape_and_rank_for_dim[i].first) {
      auto one = builder->getIntegerAttr(
          shapes[0].getType().cast<RankedTensorType>().getElementType(), 1);
      elements.push_back(builder->create<ConstantOp>(loc, one));
      continue;
    }
    // Extract from one of the shapes, accounting for the reverse indexing
    // performed by broadcast.
    Value index = builder->create<ConstantIndexOp>(
        loc, i - maxRank + shape_and_rank_for_dim[i].second);
    elements.push_back(builder->create<tensor::ExtractOp>(
        loc, shape_and_rank_for_dim[i].first, index));
  }
  return Value(builder->create<tensor::FromElementsOp>(loc, elements));
}

LogicalResult BroadcastOpLowering::matchAndRewrite(
    shape::BroadcastOp op, mlir::PatternRewriter& rewriter) const {
  ShapeComponentAnalysis shape_component_analysis;
  auto new_broadcast = simplifyBroadcast(
      shape_component_analysis, op.getShapes(), op.getLoc(), &rewriter);
  if (!new_broadcast) return failure();
  rewriter.replaceOp(op, {*new_broadcast});
  return success();
}

// -------------------------------------------------------------------------- //
// Optimize function based on the symbolic shape attributes.
// -------------------------------------------------------------------------- //

struct SymbolicShapeOptimizationPass
    : public impl::SymbolicShapeOptimizationBase<
          SymbolicShapeOptimizationPass> {
  SymbolicShapeOptimizationPass() = default;

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // Rewrite shape.broadcast based on the symbolic shapes.
    patterns.add<BroadcastOpLowering>(ctx);

    // Add shape dialect canonicalization patterns to fold shape operations
    // after constraints are replaced with constant witness.
    for (auto op : ctx->getRegisteredOperations()) {
      if (llvm::isa<shape::ShapeDialect>(op.getDialect()))
        op.getCanonicalizationPatterns(patterns, ctx);
    }

    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateSymbolicShapeOptimizationPass() {
  return std::make_unique<SymbolicShapeOptimizationPass>();
}

}  // namespace tensorflow
