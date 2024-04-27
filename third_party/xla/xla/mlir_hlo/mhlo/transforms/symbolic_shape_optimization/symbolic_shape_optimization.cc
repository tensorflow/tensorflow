/* Copyright 2021 The OpenXLA Authors.

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
#include <numeric>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/analysis/shape_component_analysis.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_SYMBOLICSHAPEOPTIMIZATION
#include "mhlo/transforms/mhlo_passes.h.inc"

using ShapeOrValueInfo = ShapeComponentAnalysis::ShapeOrValueInfo;
using Symbol = ShapeComponentAnalysis::Symbol;
using SymbolicExpr = ShapeComponentAnalysis::SymbolicExpr;

namespace {

// Temporary data structure to hold a single dimension of the symbolic result of
// `shape.broadcast`.
struct SymbolicBroadcastDimension {
  size_t operandIndex;
  size_t operandDim;
  SymbolicExpr expr;
};

// Replace shape.broadcast with a shape if it's statically known.
struct SimplifyBroadcasts : public mlir::OpRewritePattern<shape::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      shape::BroadcastOp op, mlir::PatternRewriter &rewriter) const override {
    // Require successful shape analysis.
    ShapeComponentAnalysis shapeAnalysis;
    llvm::SmallVector<ArrayRef<SymbolicExpr>> shapesInfo;
    auto shapes = op.getShapes();
    shapesInfo.reserve(shapes.size());
    for (Value s : shapes) {
      auto sInfo = shapeAnalysis.GetValueInfo(s);
      if (!sInfo) return failure();
      shapesInfo.push_back(*sInfo);
    }

    // Find the result rank.
    size_t rank = 0;
    for (const auto &sInfo : shapesInfo) rank = std::max(rank, sInfo.size());

    // Compute broadcast symbolically.
    SmallVector<std::optional<SymbolicBroadcastDimension>> symResult(
        rank, std::nullopt);
    for (const auto &sInfo : llvm::enumerate(shapesInfo)) {
      size_t dimOffset = rank - sInfo.value().size();
      for (const auto &symExpr : llvm::enumerate(sInfo.value())) {
        // Unit dimensions are neutral to the final result.
        if (symExpr.value().isConstant(1)) continue;

        // Use unique expression.
        size_t i = dimOffset + symExpr.index();
        if (!symResult[i]) {
          symResult[i] = {sInfo.index(), symExpr.index(), symExpr.value()};
          continue;
        }

        // Bail if the dimensions are neither equal nor 1.
        if (symResult[i]->expr != symExpr.value()) return failure();
      }
    }

    // Materialize broadcast result.
    auto loc = op.getLoc();
    DenseMap<int64_t, Value> constants;
    auto findOrCreateConstant = [&](int64_t c) {
      auto it = constants.find(c);
      if (it != constants.end()) return it->second;
      Value newlyCreated = rewriter.create<arith::ConstantIndexOp>(loc, c);
      constants[c] = newlyCreated;
      return newlyCreated;
    };
    auto elements = llvm::to_vector<8>(
        llvm::map_range(symResult, [&](const auto &symResultDim) {
          // If we know the dimension statically, use a constant.
          if (!symResultDim) return findOrCreateConstant(1);
          if (auto cexpr =
                  dyn_cast<AffineConstantExpr>(symResultDim->expr.expr)) {
            return findOrCreateConstant(cexpr.getValue());
          }

          // Othwerise, extract the dimension from the unique operand.
          Value operand = shapes[symResultDim->operandIndex];
          Value operandDim = findOrCreateConstant(symResultDim->operandDim);
          return rewriter.create<tensor::ExtractOp>(loc, operand, operandDim)
              .getResult();
        }));
    Type indexTy = rewriter.getIndexType();
    Type concreteResultTy =
        RankedTensorType::get({static_cast<int64_t>(elements.size())}, indexTy);
    Value result = rewriter.create<tensor::FromElementsOp>(
        loc, concreteResultTy, elements);

    // Insert cast, if needed.
    Type expectedTy = op.getResult().getType();
    if (result.getType() != expectedTy) {
      result = rewriter.create<tensor::CastOp>(loc, expectedTy, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

LogicalResult analyzeDynamicBroadcastInDimExpandingBehavior(
    ShapeComponentAnalysis &analysis, Value value, Value shape,
    llvm::SmallSetVector<int64_t, 4> *knownExpandingDims,
    llvm::SmallSetVector<int64_t, 4> *knownNonexpandingDims) {
  // Require successful analysis of shapes.
  auto shapeIn = analysis.GetShapeInfo(value);
  auto shapeOut = analysis.GetValueInfo(shape);
  if (!shapeIn || !shapeOut) return failure();

  // Analyze per argument dimension.
  size_t rankIn = shapeIn->size();
  size_t rankOut = shapeOut->size();
  assert(rankIn <= rankOut);
  size_t dimOutOffset = rankOut - rankIn;
  for (size_t i = 0; i < rankIn; ++i) {
    SymbolicExpr dimIn = (*shapeIn)[i];
    SymbolicExpr dimOut = (*shapeOut)[dimOutOffset + i];
    if (dimIn.isConstant(1) && dimOut.isKnownNotOne())
      knownExpandingDims->insert(i);
    if (dimIn == dimOut || dimOut.isConstant(1))
      knownNonexpandingDims->insert(i);
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
    llvm::SmallSetVector<int64_t, 4> knownExpandingDims, knownNonexpandingDims;
    if (failed(analyzeDynamicBroadcastInDimExpandingBehavior(
            analysis, op.getOperand(), op.getOutputDimensions(),
            &knownExpandingDims, &knownNonexpandingDims))) {
      return failure();
    }

    // Collect possibly already annotated info.
    auto insertAll = [](llvm::SmallSetVector<int64_t, 4> &dst,
                        std::optional<DenseIntElementsAttr> src) {
      if (!src) return;
      for (auto it : *src) dst.insert(it.getLimitedValue());
    };
    insertAll(knownExpandingDims, op.getKnownExpandingDimensions());
    insertAll(knownNonexpandingDims, op.getKnownNonexpandingDimensions());

    // Fail pattern application if there is nothing new to annotate.
    auto isEqual = [](llvm::SmallSetVector<int64_t, 4> &set,
                      DenseIntElementsAttr attr) {
      return static_cast<int64_t>(set.size()) == attr.size() &&
             llvm::all_of(attr, [&](auto it) {
               return set.count(it.getLimitedValue());
             });
    };
    if (op.getKnownExpandingDimensions() &&
        op.getKnownNonexpandingDimensions() &&
        isEqual(knownExpandingDims, *op.getKnownExpandingDimensions()) &&
        isEqual(knownNonexpandingDims, *op.getKnownNonexpandingDimensions())) {
      return failure();
    }

    // Annotate op in place.
    rewriter.startOpModification(op);
    op.setKnownExpandingDimensionsAttr(
        rewriter.getI64TensorAttr(knownExpandingDims.takeVector()));
    op.setKnownNonexpandingDimensionsAttr(
        rewriter.getI64TensorAttr(knownNonexpandingDims.takeVector()));
    rewriter.finalizeOpModification(op);
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
    auto dynamicShape =
        shapeComponentAnalysis.GetValueInfo(op.getDynamicShape());
    if (!dynamicShape) return failure();

    if (llvm::any_of(*dynamicShape, [](const auto &dim) {
          return !dim.isKnownNotNegativeOne();
        })) {
      return failure();
    }
    rewriter.replaceOp(op, op.getDynamicShape());
    return success();
  }
};

bool isProduct(AffineExpr expr,
               llvm::function_ref<void(AffineConstantExpr)> cbkConstantFactor,
               llvm::function_ref<void(AffineSymbolExpr)> cbkSymbolicFactor) {
  auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (binExpr && binExpr.getKind() == AffineExprKind::Mul) {
    return isProduct(binExpr.getLHS(), cbkConstantFactor, cbkSymbolicFactor) &&
           isProduct(binExpr.getRHS(), cbkConstantFactor, cbkSymbolicFactor);
  }
  if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    cbkSymbolicFactor(symExpr);
    return true;
  }
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
    cbkConstantFactor(constExpr);
    return true;
  }
  return false;
}

bool isSymbolicProduct(const SymbolicExpr &symbolicExpr,
                       llvm::function_ref<void(int64_t)> cbkConstantFactor,
                       llvm::function_ref<void(Symbol)> cbkSymbolicFactor) {
  return isProduct(
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

bool isSymbolicProduct(const SymbolicExpr &symbolicExpr,
                       SymbolicProduct *product) {
  return isSymbolicProduct(
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
        shapeComponentAnalysis.GetValueInfo(op.getNumElements());
    if (!numElementsInfo) return failure();
    assert(numElementsInfo->size() == 1 && "expect one value for a scalar");
    auto numElements = numElementsInfo->front();

    // Get shape analysis info for the dynamic shape.
    auto dynShapeDims =
        shapeComponentAnalysis.GetValueInfo(op.getDynamicShape());
    if (!dynShapeDims) return failure();

    // We can handle two cases:
    //   - there is exactly one -1 in the dynamic shape, i.e. a unique wildcard
    //     dimension, or
    //   - there is no -1 in the dynamic shape, i.e. no wildcard dimension.
    bool uniqueWildcardDimension = false;
    for (const auto &d : *dynShapeDims) {
      if (d.isConstant(-1)) {
        if (uniqueWildcardDimension) return failure();
        uniqueWildcardDimension = true;
      } else if (!d.isKnownNotNegativeOne()) {
        return failure();
      }
    }

    // We can only handle simple products with constants and symbols. Find all
    // the factors based on the number of elements.
    SymbolicProduct numElementsRemainingFactors;
    if (!isSymbolicProduct(numElements, &numElementsRemainingFactors)) {
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
      if (!isSymbolicProduct(
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
    if (uniqueWildcardDimension) {
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

LogicalResult materializeReshapeAsScalarExpand(RankedTensorType operandTy,
                                               RankedTensorType resultTy,
                                               mhlo::DynamicReshapeOp op,
                                               PatternRewriter &rewriter) {
  assert(operandTy.getRank() == 0 && "expect scalar operand");
  auto loc = op.getLoc();
  SmallVector<int64_t> unitDims(resultTy.getRank(), 1);
  auto expandedTy = RankedTensorType::get(unitDims, resultTy.getElementType());
  Value expandedScalar = rewriter.create<tensor::ExpandShapeOp>(
      loc, expandedTy, op.getOperand(), ArrayRef<ReassociationIndices>{});
  if (expandedScalar.getType() != resultTy) {
    expandedScalar =
        rewriter.create<tensor::CastOp>(loc, resultTy, expandedScalar);
  }
  rewriter.replaceOp(op, expandedScalar);
  return success();
}

LogicalResult materializeReshapeAsScalarCollapse(RankedTensorType operandTy,
                                                 RankedTensorType resultTy,
                                                 mhlo::DynamicReshapeOp op,
                                                 PatternRewriter &rewriter) {
  assert(resultTy.getRank() == 0 && "expect scalar result");
  auto loc = op.getLoc();
  Value operand = op.getOperand();
  SmallVector<int64_t> unitDims(operandTy.getRank(), 1);
  auto castedOperandTy =
      RankedTensorType::get(unitDims, operandTy.getElementType());
  if (operand.getType() != castedOperandTy) {
    operand = rewriter.create<tensor::CastOp>(loc, castedOperandTy, operand);
  }
  Value collapsedScalar = rewriter.create<tensor::CollapseShapeOp>(
      loc, operand, ArrayRef<ReassociationIndices>{});
  rewriter.replaceOp(op, collapsedScalar);
  return success();
}

enum class DimensionGroupKind {
  kNone,
  kExpanding,
  kCollapsing,
};

struct DimensionGroup {
  int64_t size = 0;
  DimensionGroupKind kind = DimensionGroupKind::kNone;
};

SymbolicProduct eliminateCommonFactors(SymbolicProduct &a, SymbolicProduct &b) {
  SymbolicProduct gcd;

  // Eliminate common concrete factors.
  gcd.concrete = std::gcd(a.concrete, b.concrete);
  a.concrete /= gcd.concrete;
  b.concrete /= gcd.concrete;

  // Eliminate common symbolic factors.
  int64_t i = 0;
  while (i < static_cast<int64_t>(a.symbolic.size())) {
    auto *it = llvm::find(b.symbolic, a.symbolic[i]);
    if (it != b.symbolic.end()) {
      gcd.symbolic.push_back(*it);
      std::swap(a.symbolic[i], a.symbolic.back());
      a.symbolic.pop_back();
      b.symbolic.erase(it);
    } else {
      i++;
    }
  }

  return gcd;
}

bool isUnpairedUnitDimension(
    ArrayRef<ShapeComponentAnalysis::SymbolicExpr>::iterator it,
    ArrayRef<ShapeComponentAnalysis::SymbolicExpr>::iterator end,
    ArrayRef<ShapeComponentAnalysis::SymbolicExpr>::iterator otherIt,
    ArrayRef<ShapeComponentAnalysis::SymbolicExpr>::iterator otherEnd) {
  return it != end && it->isConstant(1) &&
         (otherIt == otherEnd || !otherIt->isConstant(1));
}

int64_t getShapedTypyDimSize(const SymbolicProduct &symProduct) {
  return symProduct.symbolic.empty() ? symProduct.concrete
                                     : ShapedType::kDynamic;
}

// Iterate over the operand's and the result's shape dimensions and find
// dimension groups that are collapsing, expanding, or untouched:
//   - Collapsing: Multiple dimensions of the operand shape can be collapsed
//     into a single dimension of the result shape. We must prove that the
//     product of the operand shape's dimensions is equal to the corresponding
//     result dimension.
//   - Expanding: A single dimension of the operand shape can be expanded into
//     multiple dimensions of the result shape. We must prove that the product
//     of the result shape's dimensions is equal to the corresponding operand
//     dimension. This case is limited to at most one dynamic dimension per
//     expansion group as otherwise not supported by the `expand_shape` op.
//   - Untouched: There is a 1:1 correspondance between an operand and a result
//     shape dimension.
//
// We can determine the optimal dimension groups greedily by consuming operand
// and result dimensions from left to right. If the leading operand dimension is
// a strict divisor of the leading result dimension, collapsing is required. In
// this case, we keep consuming the operand dimensions until the products are
// equal. If the leading result dimension is a strict divisor of the leading
// operand dimension, expanding is required. In this case, we keep consuming the
// result dimensions until the products are equal. Trailing unit dimensions may
// be inlcuded in the dimension group. This is useful iff they are "unpaired",
// in which case they would only limit us in the subsequent iteration.
//
LogicalResult findExpandingAndCollapsingDimensionGroups(
    ArrayRef<SymbolicExpr> operandShapeInfo,
    ArrayRef<SymbolicExpr> resultShapeInfo,
    SmallVector<DimensionGroup> *dimensionGroups,
    SmallVector<int64_t> *expandedIntermShape) {
  const auto *operandShapeIt = operandShapeInfo.begin();
  const auto *operandShapeEnd = operandShapeInfo.end();
  const auto *resultShapeIt = resultShapeInfo.begin();
  const auto *resultShapeEnd = resultShapeInfo.end();

  // Crucial iteration state.
  SymbolicProduct remainingOperandShapeFactors;
  SymbolicProduct remainingResultShapeFactors;
  auto anyRemainingFactors = [&]() {
    return !remainingOperandShapeFactors.empty() ||
           !remainingResultShapeFactors.empty();
  };

  while (operandShapeIt != operandShapeEnd && resultShapeIt != resultShapeEnd) {
    assert(!anyRemainingFactors() &&
           "expect no remaining factors from previous iteration");
    DimensionGroup &dimGroup = dimensionGroups->emplace_back();

    // Consume at least one operand and result dimension.
    {
      if (!isSymbolicProduct(*operandShapeIt++,
                             &remainingOperandShapeFactors) ||
          !isSymbolicProduct(*resultShapeIt++, &remainingResultShapeFactors)) {
        return failure();
      }
      dimGroup.size++;
      SymbolicProduct gcd = eliminateCommonFactors(remainingOperandShapeFactors,
                                                   remainingResultShapeFactors);
      expandedIntermShape->push_back(getShapedTypyDimSize(gcd));
    }

    // Fail if there are unresolvable, contradicting factors remaining.
    if (!remainingOperandShapeFactors.empty() &&
        !remainingResultShapeFactors.empty()) {
      return failure();
    }

    // Collapsing: Create a collapsing dimension group.
    bool requiresCollapsing =
        remainingOperandShapeFactors.empty() &&
        (!remainingResultShapeFactors.empty() ||
         isUnpairedUnitDimension(operandShapeIt, operandShapeEnd, resultShapeIt,
                                 resultShapeEnd));
    if (requiresCollapsing) {
      dimGroup.kind = DimensionGroupKind::kCollapsing;

      // Consume operand shape dimensions until their product matches the
      // corresponding result dimension (or fail if unresolvable/contradicting
      // factors are found).
      while (operandShapeIt != operandShapeEnd &&
             remainingOperandShapeFactors.empty() &&
             !remainingResultShapeFactors.empty()) {
        if (!isSymbolicProduct(*operandShapeIt++,
                               &remainingOperandShapeFactors)) {
          return failure();
        }
        dimGroup.size++;
        SymbolicProduct gcd = eliminateCommonFactors(
            remainingOperandShapeFactors, remainingResultShapeFactors);
        expandedIntermShape->push_back(getShapedTypyDimSize(gcd));
      }
      if (anyRemainingFactors()) return failure();

      // Consume trailing, unpaired unit dimensions.
      while (isUnpairedUnitDimension(operandShapeIt, operandShapeEnd,
                                     resultShapeIt, resultShapeEnd)) {
        operandShapeIt++;
        dimGroup.size++;
        expandedIntermShape->push_back(1);
      }

      continue;
    }

    // Expanding: Create an expanding dimension group.
    bool requiresExpanding =
        remainingResultShapeFactors.empty() &&
        (!remainingOperandShapeFactors.empty() ||
         isUnpairedUnitDimension(resultShapeIt, resultShapeEnd, operandShapeIt,
                                 operandShapeEnd));
    if (requiresExpanding) {
      dimGroup.kind = DimensionGroupKind::kExpanding;
      int64_t numDynamicDims = 0;

      // Consume result shape dimensions until their product matches the
      // corresponding operand dimension (or fail if unresolvable/contradicting
      // factors are found).
      while (resultShapeIt != resultShapeEnd &&
             remainingResultShapeFactors.empty() &&
             !remainingOperandShapeFactors.empty()) {
        if (!isSymbolicProduct(*resultShapeIt++,
                               &remainingResultShapeFactors)) {
          return failure();
        }
        dimGroup.size++;
        SymbolicProduct gcd = eliminateCommonFactors(
            remainingOperandShapeFactors, remainingResultShapeFactors);
        int64_t tyDimSize = getShapedTypyDimSize(gcd);

        // Allow no more than one dynamic dimension per expansion group.
        if (tyDimSize == ShapedType::kDynamic) {
          numDynamicDims++;
          if (numDynamicDims > 1) return failure();
        }
        expandedIntermShape->push_back(tyDimSize);
      }
      if (anyRemainingFactors()) return failure();

      // Consume trailing, unpaired unit dimensions.
      while (isUnpairedUnitDimension(resultShapeIt, resultShapeEnd,
                                     operandShapeIt, operandShapeEnd)) {
        resultShapeIt++;
        dimGroup.size++;
        expandedIntermShape->push_back(1);
      }

      continue;
    }

    // Untouched: 1:1 mapping between operand and result shape dimension. This
    // is neither expanding nor collapsing.
    assert(!requiresCollapsing && !requiresExpanding && "expect id case");
    assert(dimGroup.size == 1 && dimGroup.kind == DimensionGroupKind::kNone &&
           "expect simple dimension group");
  }

  // Fail if there are remaining dimensions that could not be consumed.
  assert(!anyRemainingFactors() && "expect no remaining factors");
  if (operandShapeIt != operandShapeEnd || resultShapeIt != resultShapeEnd) {
    return failure();
  }

  return success();
}

SmallVector<int64_t> concretizeOperandShape(
    ArrayRef<int64_t> operandShape, ArrayRef<SymbolicExpr> operandShapeInfo) {
  SmallVector<int64_t> result;
  for (auto it : llvm::zip(operandShape, operandShapeInfo)) {
    auto dimSize = std::get<0>(it);
    auto sExpr = std::get<1>(it);
    if (auto cexpr = dyn_cast<AffineConstantExpr>(sExpr.expr)) {
      int64_t alsoDimSize = cexpr.getValue();
      assert((ShapedType::isDynamic(dimSize) || dimSize == alsoDimSize) &&
             "expect shape analysis result to be compatible with type");
      result.push_back(alsoDimSize);
      continue;
    }
    result.push_back(dimSize);
  }
  return result;
}

std::optional<SmallVector<ReassociationIndices>> requiresReassociationOfKind(
    DimensionGroupKind kind, const SmallVector<DimensionGroup> &dimGroups) {
  SmallVector<ReassociationIndices> reassociation;
  reassociation.reserve(dimGroups.size());
  bool isStrictlyReassociating = false;
  int64_t i = 0;
  for (const DimensionGroup &g : dimGroups) {
    if (g.kind == kind) {
      isStrictlyReassociating = true;
      reassociation.push_back(
          llvm::to_vector(llvm::seq<int64_t>(i, i + g.size)));
      i += g.size;
      continue;
    }
    for (int64_t j = 0; j < g.size; j++) reassociation.push_back({i++});
  }

  // Return the reassociation if expansion is required.
  if (isStrictlyReassociating) return reassociation;
  return std::nullopt;
}

LogicalResult materializeReshapeAsExpandAndCollapse(
    ShapeComponentAnalysis &shapeAnalysis, RankedTensorType operandTy,
    RankedTensorType resultTy, mhlo::DynamicReshapeOp op,
    PatternRewriter &rewriter) {
  // Require sucessful shape analysis for operand and result shape.
  auto operandShapeInfo = shapeAnalysis.GetShapeInfo(op.getOperand());
  if (!operandShapeInfo) return failure();
  auto resultShapeInfo = shapeAnalysis.GetValueInfo(op.getOutputShape());
  if (!resultShapeInfo) return failure();

  // Identify dimension groups and the intermediate expanded type.
  SmallVector<DimensionGroup> dimensionGroups;
  SmallVector<int64_t> expandedIntermShape;
  if (failed(findExpandingAndCollapsingDimensionGroups(
          *operandShapeInfo, *resultShapeInfo, &dimensionGroups,
          &expandedIntermShape))) {
    return failure();
  }

  // Materialize cast, expand, collapse, and cast, as needed.
  auto loc = op.getLoc();
  Value interm = op.getOperand();
  auto castedOperandTy = RankedTensorType::get(
      concretizeOperandShape(operandTy.getShape(), *operandShapeInfo),
      operandTy.getElementType());
  if (operandTy != castedOperandTy) {
    interm = rewriter.create<tensor::CastOp>(loc, castedOperandTy, interm);
  }
  if (auto reassociation = requiresReassociationOfKind(
          DimensionGroupKind::kExpanding, dimensionGroups)) {
    interm = rewriter.create<tensor::ExpandShapeOp>(
        loc,
        RankedTensorType::get(expandedIntermShape, operandTy.getElementType()),
        interm, *reassociation);
  }
  if (auto reassociation = requiresReassociationOfKind(
          DimensionGroupKind::kCollapsing, dimensionGroups)) {
    interm =
        rewriter.create<tensor::CollapseShapeOp>(loc, interm, *reassociation);
  }
  if (interm.getType() != resultTy) {
    interm = rewriter.create<tensor::CastOp>(loc, resultTy, interm);
  }
  rewriter.replaceOp(op, interm);
  return success();
}

// Tries to express `dynamic_reshape` ops through `expand_shape` and
// `collapse_shape` ops.
struct DynamicReshapeToExpandAndCollapseShape final
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto operandTy =
        mlir::dyn_cast<RankedTensorType>(op.getOperand().getType());
    if (!operandTy) return failure();
    auto resultTy = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy) return failure();

    // Handle degenerate scalar expand case.
    if (operandTy.getRank() == 0) {
      return materializeReshapeAsScalarExpand(operandTy, resultTy, op,
                                              rewriter);
    }

    // Handle degenerate scalar collapse case.
    if (resultTy.getRank() == 0) {
      return materializeReshapeAsScalarCollapse(operandTy, resultTy, op,
                                                rewriter);
    }

    ShapeComponentAnalysis shapeAnalysis;
    return materializeReshapeAsExpandAndCollapse(shapeAnalysis, operandTy,
                                                 resultTy, op, rewriter);
  }
};

// Returns true if all of bcasted_shapes can be broadcasted with output_shape.
bool isKnownBroadcastable(ShapeComponentAnalysis &analysis,
                          ValueRange bcastedShapes, Value outputShape) {
  auto outputShapeDims = analysis.GetValueInfo(outputShape);
  if (!outputShapeDims) return false;
  for (Value shape : bcastedShapes) {
    auto shapeDims = analysis.GetValueInfo(shape);
    if (!shapeDims) return false;
    // Iterate backwards over the smallest input shape.
    for (auto zip : llvm::zip(llvm::reverse(*outputShapeDims),
                              llvm::reverse(*shapeDims))) {
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
    ShapeComponentAnalysis shapeComponentAnalysis;
    if (!isKnownBroadcastable(shapeComponentAnalysis, op.getShapes(),
                              op.getShapes().front())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};

// Returns a shape tensor if the shapes can be broadcasted to a known shape.
// Will either return one of the shapes or a generated mix of the shapes.
std::optional<Value> simplifyBroadcast(ShapeComponentAnalysis &analysis,
                                       ValueRange shapes, Location loc,
                                       OpBuilder *builder) {
  // First find the input shape with the largest rank.
  SmallVector<ArrayRef<ShapeComponentAnalysis::SymbolicExpr>> shapesFound;
  size_t maxRank = 0;
  for (const auto &shape : llvm::enumerate(shapes)) {
    auto foundShape = analysis.GetValueInfo(shape.value());
    if (!foundShape) return {};
    shapesFound.push_back(*foundShape);
    maxRank = std::max(maxRank, foundShape->size());
  }
  if (maxRank == 0) {
    return Value(builder->create<tensor::FromElementsOp>(
        loc, shapes[0].getType(), SmallVector<Value>()));
  }

  SmallVector<const ShapeComponentAnalysis::SymbolicExpr *> joinedDimensions(
      maxRank);
  SmallVector<std::pair<Value, int64_t>> shapeAndRankForDim(maxRank);
  for (const auto &shape : llvm::enumerate(shapesFound)) {
    for (const auto &dim : llvm::enumerate(llvm::reverse(shape.value()))) {
      // 1 dimensions don't contribute to the final result.
      if (dim.value().isConstant(1)) continue;
      // If it's not a 1 dimension it will be present in the result. Remember
      // where it came from.
      auto index = maxRank - dim.index() - 1;
      if (!joinedDimensions[index]) {
        joinedDimensions[index] = &dim.value();
        shapeAndRankForDim[index] =
            std::make_pair(shapes[shape.index()], shape.value().size());
        continue;
      }
      // Bail if the dimensions are neither equal nor 1.
      if (*joinedDimensions[index] != dim.value()) return {};
    }
  }
  // If the output is the same as one of the inputs just return that.
  if (llvm::all_equal(shapeAndRankForDim) && shapeAndRankForDim[0].first) {
    return shapeAndRankForDim[0].first;
  }
  // Otherwise rematerialize the shape from the pieces we have.
  SmallVector<Value> elements;
  for (size_t i = 0; i != maxRank; ++i) {
    // 1 dimensions are filtered above, recreate the constant.
    if (!shapeAndRankForDim[i].first) {
      auto one = builder->getIntegerAttr(
          mlir::cast<RankedTensorType>(shapes[0].getType()).getElementType(),
          1);
      elements.push_back(builder->create<arith::ConstantOp>(loc, one));
      continue;
    }
    // Extract from one of the shapes, accounting for the reverse indexing
    // performed by broadcast.
    Value index = builder->create<arith::ConstantIndexOp>(
        loc, i - maxRank + shapeAndRankForDim[i].second);
    elements.push_back(builder->create<tensor::ExtractOp>(
        loc, shapeAndRankForDim[i].first, index));
  }
  return Value(builder->create<tensor::FromElementsOp>(loc, elements));
}

// Replace shape.broadcast with a shape if it's statically known.
struct BroadcastOpLowering final
    : public mlir::OpRewritePattern<shape::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      shape::BroadcastOp op, mlir::PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shapeComponentAnalysis;
    auto newBroadcast = simplifyBroadcast(
        shapeComponentAnalysis, op.getShapes(), op.getLoc(), &rewriter);
    if (!newBroadcast) return failure();

    // Insert cast, if needed.
    Type expectedTy = op.getType();
    if (newBroadcast->getType() != expectedTy) {
      newBroadcast = rewriter.create<tensor::CastOp>(op.getLoc(), expectedTy,
                                                     *newBroadcast);
    }

    rewriter.replaceOp(op, {*newBroadcast});
    return success();
  }
};

class SymbolicShapeOptimizationPass final
    : public impl::SymbolicShapeOptimizationBase<
          SymbolicShapeOptimizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // clang-format off
    patterns.insert<
        AnnotateExpandingDimensionsInDynamicBroadcastInDim,
        BroadcastOpLowering,
        CstrBroadcastableOpLowering,
        DynamicReshapeToExpandAndCollapseShape,
        RemoveComputeReshapeShape,
        RemoveRedundantCstrReshapable,
        SimplifyBroadcasts>(ctx);
    // clang-format on

    // Collect some relevant canonicalization patterns.
    shape::AssumingOp::getCanonicalizationPatterns(patterns, ctx);
    shape::ShapeOfOp::getCanonicalizationPatterns(patterns, ctx);

    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSymbolicShapeOptimizationPass() {
  return std::make_unique<SymbolicShapeOptimizationPass>();
}

}  // end namespace mhlo
}  // end namespace mlir
