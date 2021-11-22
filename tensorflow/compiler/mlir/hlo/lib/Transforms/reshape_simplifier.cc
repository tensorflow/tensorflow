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
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

namespace {

// Returns true if `reshape` only adds `1` dimensions.
bool isExpandShape(ShapeComponentAnalysis &shapeComponentAnalysis,
                   mhlo::DynamicReshapeOp reshape) {
  auto output_shape =
      shapeComponentAnalysis.dimensionsForShapeTensor(reshape.output_shape());
  auto operand_shape =
      shapeComponentAnalysis.dimensionsForShape(reshape.operand());
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
// linalg.tensor_expand_shape.
struct ReshapeToExpandShape final
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  ReshapeToExpandShape(MLIRContext *ctx,
                       ShapeComponentAnalysis &shapeComponentAnalysis)
      : OpRewritePattern(ctx), shapeComponentAnalysis(shapeComponentAnalysis) {}
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (!isExpandShape(shapeComponentAnalysis, op)) return failure();
    auto output_shape =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.output_shape());
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

    rewriter.replaceOpWithNewOp<linalg::TensorExpandShapeOp>(
        op, op.getResult().getType(), op.operand(), reassociations);
    shapeComponentAnalysis.reset();
    return success();
  }
  ShapeComponentAnalysis &shapeComponentAnalysis;
};

// Remove compute_reshape_shape if we can prove that the dynamic shape does not
// contain a `-1` dimension.
struct RemoveComputeReshapeShape final
    : public OpRewritePattern<mhlo::ComputeReshapeShapeOp> {
  RemoveComputeReshapeShape(MLIRContext *ctx,
                            ShapeComponentAnalysis &shapeComponentAnalysis)
      : OpRewritePattern(ctx), shapeComponentAnalysis(shapeComponentAnalysis) {}
  LogicalResult matchAndRewrite(mhlo::ComputeReshapeShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto dynamic_shape =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.dynamic_shape());
    if (!dynamic_shape) return failure();

    if (llvm::any_of(*dynamic_shape, [](const auto &dim) {
          return !dim.isKnownNotNegativeOne();
        })) {
      return failure();
    }
    rewriter.replaceOp(op, op.dynamic_shape());
    shapeComponentAnalysis.reset();
    return success();
  }
  ShapeComponentAnalysis &shapeComponentAnalysis;
};

struct RemoveRedundantCstrReshapable final
    : public OpRewritePattern<mhlo::CstrReshapableOp> {
  RemoveRedundantCstrReshapable(MLIRContext *ctx,
                                ShapeComponentAnalysis &shapeComponentAnalysis)
      : OpRewritePattern(ctx), shapeComponentAnalysis(shapeComponentAnalysis) {}
  LogicalResult matchAndRewrite(mhlo::CstrReshapableOp op,
                                PatternRewriter &rewriter) const override {
    // Get shape analysis info for the number of elements.
    auto numElementsDims =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.num_elements());
    if (!numElementsDims) return failure();
    assert(numElementsDims->size() == 1 && "expect one value for a scalar");
    auto numElementsDim = numElementsDims->front();

    // Get shape analysis info for the dynamic shape.
    auto dynShapeDims =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.dynamic_shape());
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
    SmallVector<AffineSymbolExpr> remainingSymbolicFactorsNumElems;
    int64_t concreteProductNumElems = 1;
    if (!IsSimpleProduct(numElementsDim.expr, &concreteProductNumElems,
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
    for (auto d : *dynShapeDims) {
      if (auto constExpr = d.expr.dyn_cast<AffineConstantExpr>()) {
        if (constExpr.getValue() != -1)
          concreteProductDynShape *= constExpr.getValue();
        continue;
      }
      if (auto symExpr = d.expr.dyn_cast<AffineSymbolExpr>()) {
        auto symDynShape = d.symbols[symExpr.getPosition()];
        bool isFactorInBothProducts = false;
        for (int i = 0; i < remainingSymbolicFactorsNumElems.size(); ++i) {
          auto symNumElements =
              numElementsDim
                  .symbols[remainingSymbolicFactorsNumElems[i].getPosition()];
          if (symDynShape == symNumElements) {
            remainingSymbolicFactorsNumElems[i] =
                remainingSymbolicFactorsNumElems.back();
            remainingSymbolicFactorsNumElems.pop_back();
            isFactorInBothProducts = true;
            break;
          }
        }
        if (!isFactorInBothProducts) return failure();
        continue;
      }
      return failure();
    }
    assert(concreteProductDynShape >= 1 &&
           "concrete product must not aggregate negative or zero factors");

    if (unique_wildcard_dimension) {
      // The wildcard dimension subsumes the remaining symbolic factors and
      // potentially also a concrete factor.
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
  bool IsSimpleProduct(
      AffineExpr expr, int64_t *concreteProduct,
      SmallVectorImpl<AffineSymbolExpr> *symbolicFactors) const {
    auto binExpr = expr.dyn_cast<AffineBinaryOpExpr>();
    if (binExpr && binExpr.getKind() == AffineExprKind::Mul) {
      return IsSimpleProduct(binExpr.getLHS(), concreteProduct,
                             symbolicFactors) &&
             IsSimpleProduct(binExpr.getRHS(), concreteProduct,
                             symbolicFactors);
    }
    if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
      symbolicFactors->push_back(symExpr);
      return true;
    }
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
      *concreteProduct *= constExpr.getValue();
      return true;
    }
    return false;
  }
  ShapeComponentAnalysis &shapeComponentAnalysis;
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
  ShapeComponentAnalysis shapeComponentAnalysis;

  // clang-format off
  patterns.insert<ReshapeToExpandShape,
                  RemoveComputeReshapeShape,
                  RemoveRedundantCstrReshapable>(ctx, shapeComponentAnalysis);
  // clang-format on

  if (failed(mlir::applyPatternsAndFoldGreedily(getFunction(),
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<FunctionPass> createReshapeSimplifierPass() {
  return std::make_unique<ReshapeSimplifierPass>();
}

}  // end namespace mlir
