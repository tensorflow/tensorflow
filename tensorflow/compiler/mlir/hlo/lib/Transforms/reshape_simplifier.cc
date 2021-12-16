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

// Returns true if `reshape` only adds `1` dimensions.
bool isExpandShape(ShapeComponentAnalysis &shapeComponentAnalysis,
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
  ReshapeToExpandShape(MLIRContext *ctx) : OpRewritePattern(ctx) {}
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shapeComponentAnalysis;
    if (!isExpandShape(shapeComponentAnalysis, op)) return failure();
    auto output_shape = shapeComponentAnalysis.GetValueInfo(op.output_shape());
    SmallVector<ReassociationExprs> reassociations(output_shape->size());
    auto it = reassociations.begin();
    int64_t runningIndex = 0;
    for (const auto &dim : *output_shape) {
      it->push_back(rewriter.getAffineDimExpr(runningIndex++));
      if (!dim.isConstant(1)) ++it;
    }
    // If the last dimension was a 1 expand it from the penultimate dim.
    if (output_shape->back().isConstant(1)) std::prev(it)->append(*it);
    reassociations.erase(it, reassociations.end());

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResult().getType(), op.operand(), reassociations);
    return success();
  }
};

// Remove compute_reshape_shape if we can prove that the dynamic shape does not
// contain a `-1` dimension.
struct RemoveComputeReshapeShape final
    : public OpRewritePattern<mhlo::ComputeReshapeShapeOp> {
  RemoveComputeReshapeShape(MLIRContext *ctx) : OpRewritePattern(ctx) {}
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

bool IsSimpleProduct(
    AffineExpr expr,
    llvm::function_ref<void(AffineConstantExpr)> cbkConstantFactor,
    llvm::function_ref<void(AffineSymbolExpr)> cbkSymbolicFactor) {
  auto binExpr = expr.dyn_cast<AffineBinaryOpExpr>();
  if (binExpr && binExpr.getKind() == AffineExprKind::Mul) {
    return IsSimpleProduct(binExpr.getLHS(), cbkConstantFactor,
                           cbkSymbolicFactor) &&
           IsSimpleProduct(binExpr.getRHS(), cbkConstantFactor,
                           cbkSymbolicFactor);
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

bool IsSimpleProduct(const SymbolicExpr &symbolicExpr,
                     llvm::function_ref<void(int64_t)> cbkConstantFactor,
                     llvm::function_ref<void(Symbol)> cbkSymbolicFactor) {
  return IsSimpleProduct(
      symbolicExpr.expr,
      [&](AffineConstantExpr cexpr) { cbkConstantFactor(cexpr.getValue()); },
      [&](AffineSymbolExpr sexpr) {
        cbkSymbolicFactor(symbolicExpr.symbols[sexpr.getPosition()]);
      });
}

bool IsSimpleProduct(const SymbolicExpr &symbolicExpr, int64_t *concreteProduct,
                     SmallVectorImpl<Symbol> *symbolicFactors) {
  return IsSimpleProduct(
      symbolicExpr, [&](int64_t c) { *concreteProduct *= c; },
      [&](Symbol s) { symbolicFactors->push_back(s); });
}

struct RemoveRedundantCstrReshapable final
    : public OpRewritePattern<mhlo::CstrReshapableOp> {
  RemoveRedundantCstrReshapable(MLIRContext *ctx) : OpRewritePattern(ctx) {}
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
    int64_t concreteProductNumElems = 1;
    SmallVector<Symbol> remainingSymbolicFactorsNumElems;
    if (!IsSimpleProduct(numElements, &concreteProductNumElems,
                         &remainingSymbolicFactorsNumElems)) {
      return failure();
    }
    assert(concreteProductNumElems >= 1 &&
           "number of elements cannot entail negative or zero factors");

    // Find all factors based on the dynamic shape.
    //   - Accumulate the conrete product to later compare it against its
    //     equivalent based on the number of elements.
    //   - Remove symbolic factors from the list and fail if we find an unknown
    //     factor, i.e. if the symbolic factors based on the dynamic shape are
    //     not a subset of the factors based on the number of elements.
    int64_t concreteProductDynShape = 1;
    for (auto dim : *dynShapeDims) {
      SmallVector<Symbol> partialSymbolicFactorsDynShape;
      if (!IsSimpleProduct(
              dim,
              [&](int64_t c) {
                if (c != -1) concreteProductDynShape *= c;
              },
              [&](Symbol s) { partialSymbolicFactorsDynShape.push_back(s); })) {
        return failure();
      }
      for (const Symbol &symDynShape : partialSymbolicFactorsDynShape) {
        auto it = llvm::find(remainingSymbolicFactorsNumElems, symDynShape);
        if (it == remainingSymbolicFactorsNumElems.end()) return failure();
        remainingSymbolicFactorsNumElems.erase(it);
      }
    }
    assert(concreteProductDynShape >= 1 &&
           "concrete product must not aggregate negative or zero factors");

    // A wildcard dimension can subsume the remaining symbolic factors and
    // potentially also a concrete factor.
    if (unique_wildcard_dimension) {
      if (concreteProductNumElems % concreteProductDynShape != 0)
        return failure();
      rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
      return success();
    }

    // W/o a wildcard, the symbolic and concrete products must be equal.
    bool isReshapable = remainingSymbolicFactorsNumElems.empty() &&
                        concreteProductNumElems == concreteProductDynShape;
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, isReshapable);
    return success();
  }
};

struct TurnDynamicReshapeIntoCollapseShape final
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  TurnDynamicReshapeIntoCollapseShape(MLIRContext *ctx)
      : OpRewritePattern(ctx) {}
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
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
      int64_t remainingConcreteProductShapeDim = 1;
      SmallVector<Symbol> remainingSymbolicFactorsShapeDim;
      if (!IsSimpleProduct(shapeDim, &remainingConcreteProductShapeDim,
                           &remainingSymbolicFactorsShapeDim)) {
        return failure();
      }

      // Consume (and collapse) as many of the operand dimensions as needed to
      // match the target dimension. This is monotonic.
      while (remainingConcreteProductShapeDim != 1 ||
             !remainingSymbolicFactorsShapeDim.empty()) {
        // Fail if there are no more operand dimensions to consume.
        if (i >= argShapeInfo->size()) return failure();

        // Find the concrete/symbolic factors for the next dimension of the
        // operand shape.
        int64_t concreteProductArgShapeDim = 1;
        SmallVector<Symbol> symbolicFactorsArgShapeDim;
        if (!IsSimpleProduct((*argShapeInfo)[i], &concreteProductArgShapeDim,
                             &symbolicFactorsArgShapeDim)) {
          return failure();
        }

        // Eliminate the common concrete factors. Fail if we cannot consume a
        // concrete factor of the operand shape.
        if (remainingConcreteProductShapeDim % concreteProductArgShapeDim != 0)
          return failure();
        remainingConcreteProductShapeDim /= concreteProductArgShapeDim;

        // Eliminate the common symbolic factors. Fail if we cannot consume a
        // symbolic factor of the operand shape.
        for (const Symbol &symArgShapeDim : symbolicFactorsArgShapeDim) {
          auto it =
              llvm::find(remainingSymbolicFactorsShapeDim, symArgShapeDim);
          if (it == remainingSymbolicFactorsShapeDim.end()) return failure();
          remainingSymbolicFactorsShapeDim.erase(it);
        }

        // If all the concrete/symbolic factors were consumable, collapse this
        // dimension (and continue if needed).
        reassociation_map.back().push_back(i++);
      }

      // Consume trailing 1 dimensions.
      while (i < argShapeInfo->size() && (*argShapeInfo)[i].isConstant(1))
        reassociation_map.back().push_back(i++);
    }

    // Fail if not all of the operand shape could be consumed.
    if (i < argShapeInfo->size()) return failure();

    // Replace reshape op with its equivalent collapse shape op.
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, op.operand(),
                                                         reassociation_map);
    return success();
  }
};

class ReshapeSimplifierPass final
    : public ReshapeSimplifierBase<ReshapeSimplifierPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnFunction() override;

 private:
};
}  // end namespace

void ReshapeSimplifierPass::runOnFunction() {
  MLIRContext *ctx = &getContext();
  mlir::RewritePatternSet patterns(ctx);

  // clang-format off
  patterns.insert<
      ReshapeToExpandShape,
      RemoveComputeReshapeShape,
      RemoveRedundantCstrReshapable,
      TurnDynamicReshapeIntoCollapseShape>(ctx);
  // clang-format on
  shape::AssumingOp::getCanonicalizationPatterns(patterns, ctx);

  if (failed(mlir::applyPatternsAndFoldGreedily(getFunction(),
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<FunctionPass> createReshapeSimplifierPass() {
  return std::make_unique<ReshapeSimplifierPass>();
}

}  // end namespace mlir
