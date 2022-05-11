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

#include <string>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

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

#define GEN_PASS_CLASSES
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
  for (const auto &shape : llvm::enumerate(shapes)) {
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
  for (const auto &shape : llvm::enumerate(shapes_found)) {
    for (const auto &dim : llvm::enumerate(llvm::reverse(shape.value()))) {
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
  if (llvm::is_splat(shape_and_rank_for_dim) &&
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

// Rewrite mhlo.dynamic_broadcast_in_dim operation into linalg.generic operation
// if can infer the indexing maps for the operand from the symbolic shapes.
class DynamicBroadcastInDimOpLowering
    : public mlir::OpRewritePattern<mhlo::DynamicBroadcastInDimOp> {
 public:
  using Base = OpRewritePattern<mhlo::DynamicBroadcastInDimOp>;

  explicit DynamicBroadcastInDimOpLowering(MLIRContext* ctx);

  LogicalResult matchAndRewrite(mhlo::DynamicBroadcastInDimOp op,
                                mlir::PatternRewriter& rewriter) const override;
};

DynamicBroadcastInDimOpLowering::DynamicBroadcastInDimOpLowering(
    MLIRContext* ctx)
    : Base(ctx) {}

// Check if broadcasting `from` to `to_shape` is statically known to only have
// dimensions that never expand or always expand.
llvm::Optional<AffineMap> isNonExpandingBroadcast(
    ShapeComponentAnalysis& analysis, Value from, Value to_shape) {
  auto in_shape = analysis.GetShapeInfo(from);
  auto out_shape = analysis.GetValueInfo(to_shape);
  if (!in_shape || !out_shape) return {};

  SmallVector<AffineExpr> input_map_exprs;
  size_t rank = out_shape->size();
  MLIRContext* ctx = (*out_shape)[0].expr.getContext();
  size_t d = 0;
  auto affine_zero = getAffineConstantExpr(0, ctx);
  for (auto zip :
       llvm::zip(llvm::reverse(*in_shape), llvm::reverse(*out_shape))) {
    const auto& in = std::get<0>(zip);
    const auto& out = std::get<1>(zip);
    bool extend = in.isConstant(1) && !out.isConstant(1);
    input_map_exprs.push_back(extend ? affine_zero
                                     : getAffineDimExpr(rank - d - 1, ctx));
    ++d;

    // Bail if this is neither a known expansion nor a known non-expansion.
    if (!extend && in != out) return {};
  }
  // Any leading dimensions will be expanded.
  input_map_exprs.resize(in_shape->size(), affine_zero);
  std::reverse(input_map_exprs.begin(), input_map_exprs.end());
  return AffineMap::get(/*dimCount=*/rank,
                        /*symbolCount=*/0, input_map_exprs, ctx);
}

LogicalResult DynamicBroadcastInDimOpLowering::matchAndRewrite(
    mhlo::DynamicBroadcastInDimOp op, mlir::PatternRewriter& rewriter) const {
  MLIRContext* ctx = getContext();

  auto in_type = op.operand().getType().dyn_cast<RankedTensorType>();
  auto out_type = op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!in_type || !out_type) return failure();

  // Check that broadcast is right-aligned (numpy style), so that operand
  // dimensions broadcasted to match inner-most dimensions of the output.
  auto bcast_dims = op.broadcast_dimensions().getValues<int64_t>();
  auto expected_bcast_dims = llvm::seq<int64_t>(
      out_type.getRank() - in_type.getRank(), out_type.getRank());
  if (!llvm::equal(bcast_dims, expected_bcast_dims)) return failure();

  ShapeComponentAnalysis shape_component_analysis;
  auto input_map = isNonExpandingBroadcast(
      shape_component_analysis, op.operand(), op.output_dimensions());
  if (!input_map) return failure();

  // Resolve dynamic output dimensions for the `linalg.init_tensor` operation.
  SmallVector<Value> output_dyn_dimensions;
  Location loc = op.getLoc();
  int64_t rank = out_type.getRank();
  for (size_t d = 0; d < rank; ++d) {
    int64_t output_dim = out_type.getShape()[d];

    // Skip static output dimensions, they will be resolved from the shape.
    if (output_dim >= 0) continue;

    // Resolve the dynamic size of the output dimension.
    Value output_dyn_dim = rewriter.create<tensor::ExtractOp>(
        loc, op.output_dimensions(),
        ValueRange{rewriter.create<ConstantIndexOp>(loc, d)});

    // Symbolic shape analysis might have given us an i32 or i64. Cast to index.
    if (!output_dyn_dim.getType().isIndex())
      output_dyn_dim = rewriter.create<IndexCastOp>(
          loc, rewriter.getIndexType(), output_dyn_dim);

    output_dyn_dimensions.push_back(output_dyn_dim);
  }

  // Create a linalg.tensor_init operation to initialize output.
  Value init = rewriter.create<linalg::InitTensorOp>(loc, output_dyn_dimensions,
                                                     out_type.getShape(),
                                                     out_type.getElementType());

  // Output indexing map is an identity with `rank` number of loops.
  AffineMap output_map = AffineMap::getMultiDimIdentityMap(rank, ctx);

  // All iterators are parallel.
  SmallVector<llvm::StringRef> iterator_types(rank, "parallel");

  rewriter.replaceOpWithNewOp<linalg::GenericOp>(
      op, /*resultTensorTypes=*/TypeRange{init.getType()},
      /*inputs=*/ValueRange{op.operand()},
      /*outputs=*/ValueRange{init},
      /*indexingMaps=*/llvm::makeArrayRef({*input_map, output_map}),
      /*iteratorTypes=*/iterator_types,
      [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
        nested_builder.create<linalg::YieldOp>(nested_loc, args[0]);
      });

  return success();
}

// -------------------------------------------------------------------------- //
// Optimize function based on the symbolic shape attributes.
// -------------------------------------------------------------------------- //

struct SymbolicShapeOptimizationPass
    : public SymbolicShapeOptimizationBase<SymbolicShapeOptimizationPass> {
  SymbolicShapeOptimizationPass() = default;

  explicit SymbolicShapeOptimizationPass(bool constraints_only) {
    this->optimize_only_constraints = constraints_only;
  }

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // Rewrite shape.broadcast based on the symbolic shapes.
    patterns.add<BroadcastOpLowering>(ctx);

    // Rewrite broadcasts based on the symbolic shapes if enabled.
    if (!optimize_only_constraints)
      patterns.add<DynamicBroadcastInDimOpLowering>(ctx);

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

std::unique_ptr<OperationPass<FuncOp>> CreateSymbolicShapeOptimizationPass(
    bool constraints_only) {
  return std::make_unique<SymbolicShapeOptimizationPass>(constraints_only);
}

}  // namespace tensorflow
