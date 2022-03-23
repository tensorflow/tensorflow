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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir-hlo/Analysis/shape_component_analysis.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

using ShapeOrValueInfo = ShapeComponentAnalysis::ShapeOrValueInfo;
using Symbol = ShapeComponentAnalysis::Symbol;
using SymbolicExpr = ShapeComponentAnalysis::SymbolicExpr;

namespace {

LogicalResult AnalyzeDynamicBroadcastInDimExpandingBehavior(
    ShapeComponentAnalysis &analysis, Value value, Value shape,
    llvm::SmallSetVector<int64_t, 4> *known_expanding_dims,
    llvm::SmallSetVector<int64_t, 4> *known_nonexpanding_dims) {
  // Require successful analysis of shapes.
  auto shape_in = analysis.GetShapeInfo(value);
  auto shape_out = analysis.GetValueInfo(shape);
  if (!shape_in || !shape_out) return failure();

  // Analyze per argument dimension.
  size_t rank_in = shape_in->size();
  size_t rank_out = shape_out->size();
  assert(rank_in <= rank_out);
  size_t dim_out_offset = rank_out - rank_in;

  for (size_t i = 0; i < rank_in; ++i) {
    SymbolicExpr dim_in = (*shape_in)[i];
    SymbolicExpr dim_out = (*shape_out)[dim_out_offset + i];
    if (dim_in.isConstant(1) && dim_out.isKnownNotOne())
      known_expanding_dims->insert(i);
    if (dim_in == dim_out || dim_out.isConstant(1))
      known_nonexpanding_dims->insert(i);
  }
  return success();
}

// Analyze `mhlo.dynamic_broadcast_in_dim` op and populate attributes for
// statically known expanding and non-expanding dimensions.
struct AnnotateExpandingDimensionsInDynamicBroadcastInDim
    : public mlir::OpRewritePattern<mhlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      mhlo::DynamicBroadcastInDimOp op,
      mlir::PatternRewriter &rewriter) const override {
    // Analyze shapes and identify expanding and non-expanding dims.
    ShapeComponentAnalysis analysis;
    llvm::SmallSetVector<int64_t, 4> known_expanding_dims,
        known_nonexpanding_dims;
    if (failed(AnalyzeDynamicBroadcastInDimExpandingBehavior(
            analysis, op.operand(), op.output_dimensions(),
            &known_expanding_dims, &known_nonexpanding_dims))) {
      return failure();
    }

    // Collect possibly already annotated info.
    auto insert_all = [](llvm::SmallSetVector<int64_t, 4> &dst,
                         Optional<DenseIntElementsAttr> src) {
      if (!src) return;
      for (auto it : *src) dst.insert(it.getLimitedValue());
    };
    insert_all(known_expanding_dims, op.known_expanding_dimensions());
    insert_all(known_nonexpanding_dims, op.known_nonexpanding_dimensions());

    // Fail pattern application if there is nothing new to annotate.
    auto is_equal = [](llvm::SmallSetVector<int64_t, 4> &set,
                       DenseIntElementsAttr attr) {
      return set.size() == attr.size() && llvm::all_of(attr, [&](auto it) {
               return set.count(it.getLimitedValue());
             });
    };
    if (op.known_expanding_dimensions() && op.known_nonexpanding_dimensions() &&
        is_equal(known_expanding_dims, *op.known_expanding_dimensions()) &&
        is_equal(known_nonexpanding_dims,
                 *op.known_nonexpanding_dimensions())) {
      return failure();
    }

    // Annotate op in place.
    rewriter.startRootUpdate(op);
    op.known_expanding_dimensionsAttr(
        rewriter.getI64TensorAttr(known_expanding_dims.takeVector()));
    op.known_nonexpanding_dimensionsAttr(
        rewriter.getI64TensorAttr(known_nonexpanding_dims.takeVector()));
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

// Returns true if `reshape` only adds `1` dimensions.
bool IsExpandShape(ShapeComponentAnalysis &shapeComponentAnalysis,
                   mhlo::DynamicReshapeOp reshape) {
  auto output_shape =
      shapeComponentAnalysis.GetValueInfo(reshape.output_shape());
  auto operand_shape = shapeComponentAnalysis.GetShapeInfo(reshape.operand());
  if (!output_shape || !operand_shape ||
      output_shape->size() <= operand_shape->size())
    return false;
  int inputDim = 0;
  // Check if the reshape only inserts 1 dimensions and everything else is the
  // same.
  for (const auto &dim : *output_shape) {
    if (dim.isConstant(1)) continue;
    if (inputDim >= operand_shape->size() || dim != (*operand_shape)[inputDim])
      return false;
    ++inputDim;
  }
  return inputDim == operand_shape->size();
}

// Rewrite dynamic reshapes that only insert one dimensions into
// tensor.expand_shape.
struct ReshapeToExpandShape final
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shapeComponentAnalysis;
    if (!IsExpandShape(shapeComponentAnalysis, op)) return failure();
    auto output_shape = shapeComponentAnalysis.GetValueInfo(op.output_shape());
    SmallVector<ReassociationExprs> reassociations(output_shape->size());
    auto old_result_type = op.getResult().getType().cast<ShapedType>();
    auto output_dimensions = llvm::to_vector<>(old_result_type.getShape());
    auto *it = reassociations.begin();
    int64_t runningIndex = 0;
    for (const auto &dim : *output_shape) {
      it->push_back(rewriter.getAffineDimExpr(runningIndex++));
      if (dim.isConstant(1)) {
        output_dimensions[runningIndex - 1] = 1;
      } else {
        ++it;
      }
    }
    // If the last dimension was a 1 expand it from the penultimate dim.
    if (it != reassociations.begin() && output_shape->back().isConstant(1))
      std::prev(it)->append(*it);
    reassociations.erase(it, reassociations.end());

    // mhlo.dynamic_reshape is more lenient about the type. Add the static
    // knowledge we have about 1 dims.
    auto new_result_type = RankedTensorType::get(
        output_dimensions, old_result_type.getElementType());
    Location loc = op.getLoc();
    Value expanded_shape = rewriter.create<tensor::ExpandShapeOp>(
        loc, new_result_type, op.operand(), reassociations);
    if (old_result_type != new_result_type) {
      expanded_shape =
          rewriter.create<tensor::CastOp>(loc, old_result_type, expanded_shape);
    }
    rewriter.replaceOp(op, expanded_shape);
    return success();
  }
};

// Remove compute_reshape_shape if we can prove that the dynamic shape does not
// contain a `-1` dimension.
struct RemoveComputeReshapeShape final
    : public OpRewritePattern<mhlo::ComputeReshapeShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ComputeReshapeShapeOp op,
                                PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shapeComponentAnalysis;
    auto dynamic_shape =
        shapeComponentAnalysis.GetValueInfo(op.dynamic_shape());
    if (!dynamic_shape) return failure();

    if (llvm::any_of(*dynamic_shape, [](const auto &dim) {
          return !dim.isKnownNotNegativeOne();
        })) {
      return failure();
    }
    rewriter.replaceOp(op, op.dynamic_shape());
    return success();
  }
};

bool IsProduct(AffineExpr expr,
               llvm::function_ref<void(AffineConstantExpr)> cbkConstantFactor,
               llvm::function_ref<void(AffineSymbolExpr)> cbkSymbolicFactor) {
  auto binExpr = expr.dyn_cast<AffineBinaryOpExpr>();
  if (binExpr && binExpr.getKind() == AffineExprKind::Mul) {
    return IsProduct(binExpr.getLHS(), cbkConstantFactor, cbkSymbolicFactor) &&
           IsProduct(binExpr.getRHS(), cbkConstantFactor, cbkSymbolicFactor);
  }
  if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
    cbkSymbolicFactor(symExpr);
    return true;
  }
  if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    cbkConstantFactor(constExpr);
    return true;
  }
  return false;
}

bool IsSymbolicProduct(const SymbolicExpr &symbolicExpr,
                       llvm::function_ref<void(int64_t)> cbkConstantFactor,
                       llvm::function_ref<void(Symbol)> cbkSymbolicFactor) {
  return IsProduct(
      symbolicExpr.expr,
      [&](AffineConstantExpr cexpr) { cbkConstantFactor(cexpr.getValue()); },
      [&](AffineSymbolExpr sexpr) {
        cbkSymbolicFactor(symbolicExpr.symbols[sexpr.getPosition()]);
      });
}

// Represents a product of symbolic and concrete factors. This will allow us to
// prove product equalities symbolically.
struct SymbolicProduct {
  // Product of all concrete factors.
  int64_t concrete = 1;
  // List all symbolic factors as they can not be aggregated.
  llvm::SmallVector<Symbol> symbolic;
  bool empty() { return concrete == 1 && symbolic.empty(); }
};

bool IsSymbolicProduct(const SymbolicExpr &symbolicExpr,
                       SymbolicProduct *product) {
  return IsSymbolicProduct(
      symbolicExpr, [&](int64_t c) { product->concrete *= c; },
      [&](Symbol s) { product->symbolic.push_back(s); });
}

struct RemoveRedundantCstrReshapable final
    : public OpRewritePattern<mhlo::CstrReshapableOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::CstrReshapableOp op,
                                PatternRewriter &rewriter) const override {
    // Get shape analysis info for the number of elements.
    ShapeComponentAnalysis shapeComponentAnalysis;
    auto numElementsInfo =
        shapeComponentAnalysis.GetValueInfo(op.num_elements());
    if (!numElementsInfo) return failure();
    assert(numElementsInfo->size() == 1 && "expect one value for a scalar");
    auto numElements = numElementsInfo->front();

    // Get shape analysis info for the dynamic shape.
    auto dynShapeDims = shapeComponentAnalysis.GetValueInfo(op.dynamic_shape());
    if (!dynShapeDims) return failure();

    // We can handle two cases:
    //   - there is exactly one -1 in the dynamic shape, i.e. a unique wildcard
    //     dimension, or
    //   - there is no -1 in the dynamic shape, i.e. no wildcard dimension.
    bool unique_wildcard_dimension = false;
    for (const auto &d : *dynShapeDims) {
      if (d.isConstant(-1)) {
        if (unique_wildcard_dimension) return failure();
        unique_wildcard_dimension = true;
      } else if (!d.isKnownNotNegativeOne()) {
        return failure();
      }
    }

    // We can only handle simple products with constants and symbols. Find all
    // the factors based on the number of elements.
    SymbolicProduct numElementsRemainingFactors;
    if (!IsSymbolicProduct(numElements, &numElementsRemainingFactors)) {
      return failure();
    }
    assert(numElementsRemainingFactors.concrete >= 1 &&
           "number of elements cannot entail negative or zero factors");

    // Find all factors based on the dynamic shape.
    //   - Accumulate the conrete product to later compare it against its
    //     equivalent based on the number of elements.
    //   - Remove symbolic factors from the list and fail if we find an unknown
    //     factor, i.e. if the symbolic factors based on the dynamic shape are
    //     not a subset of the factors based on the number of elements.
    int64_t concreteProductDynShape = 1;
    for (const auto &dim : *dynShapeDims) {
      SmallVector<Symbol> partialSymbolicFactorsDynShape;
      if (!IsSymbolicProduct(
              dim,
              [&](int64_t c) {
                if (c != -1) concreteProductDynShape *= c;
              },
              [&](Symbol s) { partialSymbolicFactorsDynShape.push_back(s); })) {
        return failure();
      }
      for (const Symbol &symDynShape : partialSymbolicFactorsDynShape) {
        auto *it =
            llvm::find(numElementsRemainingFactors.symbolic, symDynShape);
        if (it == numElementsRemainingFactors.symbolic.end()) return failure();
        numElementsRemainingFactors.symbolic.erase(it);
      }
    }
    assert(concreteProductDynShape >= 1 &&
           "concrete product must not aggregate negative or zero factors");

    // A wildcard dimension can subsume the remaining symbolic factors and
    // potentially also a concrete factor.
    if (unique_wildcard_dimension) {
      if (numElementsRemainingFactors.concrete % concreteProductDynShape != 0)
        return failure();
      rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
      return success();
    }

    // W/o a wildcard, the symbolic and concrete products must be equal.
    bool isReshapable =
        numElementsRemainingFactors.symbolic.empty() &&
        numElementsRemainingFactors.concrete == concreteProductDynShape;
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, isReshapable);
    return success();
  }
};

struct TurnDynamicReshapeIntoCollapseShape final
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto input_type = op.operand().getType().dyn_cast<RankedTensorType>();
    auto output_type = op.getType().dyn_cast<RankedTensorType>();
    if (!input_type || !output_type ||
        input_type.getRank() <= output_type.getRank())
      return failure();

    // Require sucessful shape analysis for operand and shape.
    ShapeComponentAnalysis shapeComponentAnalysis;
    auto argShapeInfo = shapeComponentAnalysis.GetShapeInfo(op.operand());
    if (!argShapeInfo) return failure();
    auto shapeInfo = shapeComponentAnalysis.GetValueInfo(op.output_shape());
    if (!shapeInfo) return failure();

    // The next dimension of the operand shape to look at.
    int i = 0;

    // For each dimension of the target shape, consume the matching dimensions
    // of the operand shape and build the reassociation map on the fly.
    SmallVector<ReassociationIndices> reassociation_map;
    for (const auto &shapeDim : *shapeInfo) {
      reassociation_map.push_back({});

      // Find the concrete/symbolic factors for the current dimension of the
      // target shape.
      SymbolicProduct remainingFactorsShapeDim;
      if (!IsSymbolicProduct(shapeDim, &remainingFactorsShapeDim)) {
        return failure();
      }

      // Consume (and collapse) as many of the operand dimensions as needed to
      // match the target dimension. This is monotonic.
      while (!remainingFactorsShapeDim.empty()) {
        // Fail if there are no more operand dimensions to consume.
        if (i >= argShapeInfo->size()) return failure();

        // Find the concrete/symbolic factors for the next dimension of the
        // operand shape.
        SymbolicProduct remainingFactorsArgShapeDim;
        if (!IsSymbolicProduct((*argShapeInfo)[i],
                               &remainingFactorsArgShapeDim)) {
          return failure();
        }

        // Eliminate the common concrete factors. Fail if we cannot consume a
        // concrete factor of the operand shape.
        if (remainingFactorsShapeDim.concrete %
                remainingFactorsArgShapeDim.concrete !=
            0)
          return failure();
        remainingFactorsShapeDim.concrete /=
            remainingFactorsArgShapeDim.concrete;

        // Eliminate the common symbolic factors. Fail if we cannot consume a
        // symbolic factor of the operand shape.
        for (const Symbol &symArgShapeDim :
             remainingFactorsArgShapeDim.symbolic) {
          auto *it =
              llvm::find(remainingFactorsShapeDim.symbolic, symArgShapeDim);
          if (it == remainingFactorsShapeDim.symbolic.end()) return failure();
          remainingFactorsShapeDim.symbolic.erase(it);
        }

        // If all the concrete/symbolic factors were consumable, collapse this
        // dimension (and continue if needed).
        reassociation_map.back().push_back(i++);
      }

      // Consume trailing 1 dimensions.
      while (i < argShapeInfo->size() && (*argShapeInfo)[i].isConstant(1))
        reassociation_map.back().push_back(i++);

      // This is effectively a shape expansion that we cannot handle yet.
      // TODO(b/217611473): Implement shape expansion cases.
      if (reassociation_map.back().empty()) return failure();
    }

    // Fail if not all of the operand shape could be consumed.
    if (i < argShapeInfo->size()) return failure();

    // Replace reshape op with its equivalent collapse shape op.
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, op.operand(),
                                                         reassociation_map);
    return success();
  }
};

// Returns true if all of bcasted_shapes can be broadcasted with output_shape.
bool IsKnownBroadcastable(ShapeComponentAnalysis &analysis,
                          ValueRange bcasted_shapes, Value output_shape) {
  auto output_shape_dims = analysis.GetValueInfo(output_shape);
  if (!output_shape_dims) return false;
  for (Value shape : bcasted_shapes) {
    auto shape_dims = analysis.GetValueInfo(shape);
    if (!shape_dims) return false;
    // Iterate backwards over the smallest input shape.
    for (auto zip : llvm::zip(llvm::reverse(*output_shape_dims),
                              llvm::reverse(*shape_dims))) {
      const auto &first = std::get<0>(zip);
      const auto &second = std::get<1>(zip);
      // TODO(ezhulenev): What to do with dimensions statically known to be
      // zero?
      // Numpy can only broadcast [0] with [1], however Tensorflow can broadcast
      // [0] with any dimension size, and produces dimension of size [0].
      // Currently we'll conservatively return failure and will not proceed with
      // a rewrite.
      if (first.isConstant(0) || second.isConstant(0)) return false;
      // If either shape has a static one dimension the broadcast will always
      // succeed.
      if (first.isConstant(1) || second.isConstant(1)) continue;
      // Otherwise dims have to be equal.
      if (first != second) return false;
    }
  }
  return true;
}

// Rewrite `shape.cstr_broadcastable` with constant witness if can prove that
// shapes are broadcastable from a symbolic analysis.
struct CstrBroadcastableOpLowering
    : public OpRewritePattern<shape::CstrBroadcastableOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shape_component_analysis;
    if (!IsKnownBroadcastable(shape_component_analysis, op.getShapes(),
                              op.getShapes().front())) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};

class SymbolicShapeOptimizationPass final
    : public SymbolicShapeOptimizationBase<SymbolicShapeOptimizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // clang-format off
    patterns.insert<
        AnnotateExpandingDimensionsInDynamicBroadcastInDim,
        CstrBroadcastableOpLowering,
        RemoveComputeReshapeShape,
        RemoveRedundantCstrReshapable,
        ReshapeToExpandShape,
        TurnDynamicReshapeIntoCollapseShape>(ctx);
    // clang-format on
    shape::AssumingOp::getCanonicalizationPatterns(patterns, ctx);

    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // end namespace

std::unique_ptr<OperationPass<FuncOp>> createSymbolicShapeOptimizationPass() {
  return std::make_unique<SymbolicShapeOptimizationPass>();
}

}  // end namespace mlir
