//===- VectorToLoops.cpp - Conversion within the Vector dialect -----------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements target-independent rewrites as 1->N patterns.
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Analysis/VectorAnalysis.h"
#include "mlir/Conversion/VectorConversions/VectorConversions.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "vector-to-vector"

using namespace mlir;
using llvm::dbgs;
using mlir::functional::zipMap;

/// Given a shape with sizes greater than 0 along all dimensions,
/// returns the distance, in number of elements, between a slice in a dimension
/// and the next slice in the same dimension.
///   e.g. shape[3, 4, 5] -> linearization_basis[20, 5, 1]
static SmallVector<int64_t, 8> computeStrides(ArrayRef<int64_t> shape) {
  if (shape.empty())
    return {};
  SmallVector<int64_t, 8> tmp;
  tmp.reserve(shape.size());
  int64_t running = 1;
  for (auto size : llvm::reverse(shape)) {
    assert(size > 0 && "size must be nonnegative");
    tmp.push_back(running);
    running *= size;
  }
  return SmallVector<int64_t, 8>(tmp.rbegin(), tmp.rend());
}

static int64_t computeMaxLinearIndex(ArrayRef<int64_t> basis) {
  if (basis.empty())
    return 0;
  int64_t res = 1;
  for (auto b : basis)
    res *= b;
  return res;
}

/// Given a shape with sizes greater than 0 along all dimensions, returns the
/// delinearized components of linearIndex along shape.
static SmallVector<int64_t, 8> delinearize(int64_t linearIndex,
                                           ArrayRef<int64_t> basis) {
  SmallVector<int64_t, 8> res;
  res.reserve(basis.size());
  for (unsigned idx = 0, e = basis.size(); idx < e; ++idx) {
    assert(basis[idx] > 0);
    res.push_back(linearIndex / basis[idx]);
    linearIndex %= basis[idx];
  }
  // Sanity check.
  assert(linearIndex == 0 && "linear index remainder must be 0");
  return res;
}

static constexpr auto kFakeForkOp = "__fake_fork__";
static constexpr auto kFakeJoinOp = "__fake_join__";
static constexpr auto kUnrollAttrName = "__unroll__";
static constexpr auto kBaseCoordAttrName = "__base_coord__";

// Reads the IntegerArray attribute named `kUnrollAttrName` from `op` and
// returns its representation as a vector of integers.
static SmallVector<int64_t, 8> extractUnrollFactors(Operation *op) {
  SmallVector<int64_t, 8> res;
  auto unrollAttr = op->getAttr(kUnrollAttrName);
  if (!unrollAttr)
    return res;
  auto unrollArrayAttr = unrollAttr.cast<ArrayAttr>();
  res.reserve(unrollArrayAttr.size());
  for (auto attr : unrollArrayAttr) {
    auto unroll = attr.cast<IntegerAttr>().getValue().getSExtValue();
    assert(unroll > 0);
    res.push_back(unroll);
  }
  return res;
}

// Creates a custom `kFakeForkOp` used in progressive lowering to other vector
// operations.
static Operation *createFakeForkOp(PatternRewriter &builder, Location loc,
                                   Value *operand, ArrayRef<Type> resultTypes,
                                   ArrayRef<int64_t> unrollFactors = {}) {
  OperationState *forkOp =
      new OperationState(loc, kFakeForkOp, operand, resultTypes, {});
  if (!unrollFactors.empty())
    forkOp->addAttribute(kUnrollAttrName,
                         builder.getI64ArrayAttr(unrollFactors));
  return builder.createOperation(*forkOp);
}

// Creates a custom `kFakeJoinOp` used in progressive lowering to other vector
// operations.
static Operation *createFakeJoinOp(PatternRewriter &builder, Location loc,
                                   ArrayRef<Value *> operands, Type resultType,
                                   ArrayRef<int64_t> unrollFactors = {},
                                   ArrayRef<int64_t> baseCoords = {}) {
  OperationState *joinOp =
      new OperationState(loc, kFakeJoinOp, operands, resultType, {});
  if (!unrollFactors.empty())
    joinOp->addAttribute(kUnrollAttrName,
                         builder.getI64ArrayAttr(unrollFactors));
  if (!baseCoords.empty())
    joinOp->addAttribute(kBaseCoordAttrName,
                         builder.getI64ArrayAttr(baseCoords));
  return builder.createOperation(*joinOp);
}

// Clones `op` into a new operations that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(PatternRewriter &builder,
                                              Location loc, Operation *op,
                                              ArrayRef<Value *> operands,
                                              ArrayRef<Type> resultTypes) {
  OperationState *res = new OperationState(loc, op->getName().getStringRef(),
                                           operands, resultTypes, {});
  return builder.createOperation(*res);
}

// Helper function for Tablegen.
static bool hasShape(Value *v, ArrayRef<int64_t> shape) {
  auto t = v->getType().dyn_cast<ShapedType>();
  if (!t)
    return false;
  return std::equal(t.getShape().begin(), t.getShape().end(), shape.begin());
}

// Entry point for unrolling declarative pattern rewrites.
// `op` is unrolled to the `targetShape` as follows, for each of its operands:
//   1. the unrolled type `unrolledVectorType` and number of unrolled instances
//   `numUnrolledInstances` are computed from the `targetShape`. For now it is
//   assumed the unrolling factors divide the vector sizes.
//   2. a fakeFork cast op is inserted that takes the operand and returns
//   `numUnrolledInstances` results of type `unrolledVectorType`.
//   3. the original op is cloned `numUnrolledInstances` times, once for each
//   result of the fakeFork cast op.
//   4. a fakeJoin cast op takes all these results and merges them into a single
//   aggregate vector result whose size matches the original non-unrolled op
//   operand types.
//
// Example:
//
//    opA(operand0, operand1)  // numUnrolledInstances = 3
//
//            operand0                   operand1
//               |                          |
//             fork                       fork
//        <----------gather all fork ops --------->
//              /|\                        /|\
//          f00 f01 f02                f10 f11 f12
//        <---------- clone op 3 times --------->
//          opA0(f00, f10), opA1(f01, f11), opA2(f02, f12)
//                 \            |            /
//      <-------------------- join ------------------------->
//
// Other local patterns then kick in iteratively (including DCE) and compose
// until all the fakeFork and fakeJoin ops are removed.
//
// This will be extended in the future to support more advanced use cases than
// simple pointwise ops.
static Value *unrollSingleResultOpMatchingType(PatternRewriter &builder,
                                               Operation *op,
                                               ArrayRef<int64_t> targetShape) {
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE
                       "]: unrollSingleResultOpMatchingType on func:\n");
  LLVM_DEBUG(op->getParentOfType<FuncOp>().print(dbgs()));
  if (!op->getNumResults())
    assert(false && "Use precondition till RewriterGen can act on nullptr");

  auto shapedType = op->getResult(0)->getType().dyn_cast_or_null<ShapedType>();
  if (!shapedType || !shapedType.hasStaticShape())
    assert(false && "Use precondition till RewriterGen can act on nullptr");

  auto shape = shapedType.getShape();
  auto maybeUnrollFactors = shapeRatio(shape, targetShape);
  if (!maybeUnrollFactors.hasValue())
    assert(false && "Use precondition till RewriterGen can act on nullptr");
  auto unrollFactors = *maybeUnrollFactors;

  auto loc = op->getLoc();
  auto numUnrolledInstances = computeMaxLinearIndex(unrollFactors);
  auto unrolledVectorType =
      VectorType::get(targetShape, shapedType.getElementType());
  SmallVector<Type, 4> forkedType(numUnrolledInstances, unrolledVectorType);
  SmallVector<Operation *, 4> forkeds;
  forkeds.reserve(numUnrolledInstances);
  // Create a new forkOp for each operand.
  for (auto *operand : op->getOperands())
    forkeds.push_back(
        createFakeForkOp(builder, loc, operand, forkedType, unrollFactors));

  SmallVector<Operation *, 4> newOps;
  newOps.reserve(numUnrolledInstances);
  for (int64_t idx = 0; idx < numUnrolledInstances; ++idx) {
    SmallVector<Value *, 4> operands;
    operands.reserve(forkeds.size());
    for (auto *fork : forkeds) {
      operands.push_back(fork->getResult(idx));
    }
    newOps.push_back(cloneOpWithOperandsAndTypes(builder, loc, op, operands,
                                                 unrolledVectorType));
  }

  SmallVector<Value *, 4> newOpResults;
  newOpResults.reserve(newOps.size());
  for (auto *newOp : newOps)
    newOpResults.push_back(newOp->getResult(0));

  return createFakeJoinOp(builder, loc, newOpResults, shapedType, unrollFactors,
                          {0})
      ->getResult(0);
}

// Patterns with this benefit just forwards arguments to clean up fake fork and
// fake joins. It is a nicer and more direct cleanup when we can use it so it
// kicks in with higher precedence.
static constexpr int64_t kMatchingFakeForkFakeJoinBenefit = 1;

namespace mlir {
namespace vector {
namespace {
#include "mlir/Dialect/VectorOps/VectorTransformPatterns.h.inc"
} // end namespace
} // end namespace vector
} // end namespace mlir

// Match a fakeFork fed by a fakeJoin and just forward its operands.
// This is akin to calling `replaceAllUsesOf` but made to play nice with all the
// other RewritePattern.
struct ConvertMatchingFakeForkFakeJoinOp : public RewritePattern {
  ConvertMatchingFakeForkFakeJoinOp(MLIRContext *context)
      // low-benefit to kick-in late
      : RewritePattern(kFakeForkOp, kMatchingFakeForkFakeJoinBenefit, context) {
  }

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1)
      return matchFailure();

    auto *definingOp = op->getOperand(0)->getDefiningOp();
    if (!definingOp || definingOp->getName().getStringRef() != kFakeJoinOp)
      return matchFailure();

    if (definingOp->getNumOperands() != op->getNumResults())
      return matchFailure();

    for (auto it : llvm::zip(definingOp->getOperands(), op->getResults())) {
      if (std::get<0>(it)->getType() != std::get<1>(it)->getType())
        return matchFailure();
    }

    LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE
                         "]: ConvertMatchingFakeForkFakeJoinOp on op: "
                      << *op << " in func:\n");
    LLVM_DEBUG(op->getParentOfType<FuncOp>().print(dbgs()));
    SmallVector<Value *, 4> forwardedOperands;
    forwardedOperands.append(definingOp->getOperands().begin(),
                             definingOp->getOperands().end());
    rewriter.replaceOp(op, forwardedOperands);
    return matchSuccess();
  }
};

// Rewrites a fakeFork, whose (unique) operand is a blockArgument, into multiple
// vector.strided_slice ops.
struct ConvertFakeForkFromBlockArgsOp : public RewritePattern {
  ConvertFakeForkFromBlockArgsOp(MLIRContext *context)
      // low-benefit to kick-in late
      : RewritePattern(kFakeForkOp, 0, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1)
      return matchFailure();

    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return matchSuccess();
    }

    auto *blockArg = op->getOperand(0);
    if (!isa<BlockArgument>(blockArg))
      return matchFailure();

    LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE
                         "]: ConvertFakeForkFromBlockArgsOp on op: "
                      << *op << " in func:\n");
    LLVM_DEBUG(op->getParentOfType<FuncOp>().print(dbgs()));

    // Look at the unroll factors remaining on this op and act on the first one.
    auto unrollFactorsStorage = extractUnrollFactors(op);
    ArrayRef<int64_t> unrollFactors{unrollFactorsStorage};
    if (unrollFactors.empty()) {
      // No more unrollFactors, just sanity check + forward the unique operand.
      assert(op->getNumResults() == 1);
      assert(op->getOperand(0)->getType() == op->getResult(0)->getType());
      rewriter.replaceOp(op, op->getOperand(0));
      return matchSuccess();
    }

    // Strides are always 1 for now.
    // TODO(b/144845578) support non-1 strides.
    auto forkedVectorType = op->getOperand(0)->getType().cast<VectorType>();
    SmallVector<int64_t, 4> strides(unrollFactors.size(), 1);
    auto nUnrolled = computeMaxLinearIndex(unrollFactors);
    SmallVector<Value *, 4> extractedVectors;
    extractedVectors.reserve(op->getNumResults());
    auto linearizationBasis = computeStrides(unrollFactors);
    for (unsigned idx = 0; idx < nUnrolled; ++idx) {
      auto offsets = delinearize(idx, linearizationBasis);
      offsets = zipMap([](int64_t v1, int64_t v2) { return v1 * v2; }, offsets,
                       unrollFactors);
      auto leadingSize =
          forkedVectorType.getShape().take_front(unrollFactors.size());
      auto sizes = zipMap([](int64_t v1, int64_t v2) { return v1 / v2; },
                          leadingSize, unrollFactors);
      extractedVectors.push_back(
          rewriter
              .create<vector::StridedSliceOp>(op->getLoc(), blockArg, offsets,
                                              sizes, strides)
              .getResult());
    }
    rewriter.replaceOp(op, extractedVectors);
    return matchSuccess();
  }
};

static Value *makeSplatZero(Location loc, PatternRewriter &rewriter,
                            VectorType vt) {
  auto t = vt.getElementType();
  Value *f = nullptr;
  if (t.isBF16() || t.isF16())
    f = rewriter.create<ConstantOp>(loc, t, rewriter.getF16FloatAttr(0.0f))
            .getResult();
  else if (t.isF32())
    f = rewriter.create<ConstantOp>(loc, t, rewriter.getF32FloatAttr(0.0f))
            .getResult();
  else if (t.isF64())
    f = rewriter.create<ConstantOp>(loc, t, rewriter.getF64FloatAttr(0.0f))
            .getResult();
  if (f)
    return rewriter.create<SplatOp>(loc, vt, f).getResult();
  llvm_unreachable("Unsupported type in `makeSplatZero`");
}

// Rewrites a fakeJoin, whose (unique) operand is a blockArgument, into multiple
// vector.strided_slice ops.
struct ConvertFakeJoinOp : public RewritePattern {
  ConvertFakeJoinOp(MLIRContext *context)
      // low-benefit to kick-in late
      : RewritePattern(kFakeJoinOp, 0, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return matchFailure();

    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return matchSuccess();
    }

    auto resultVectorType = op->getResult(0)->getType().cast<VectorType>();
    auto loc = op->getLoc();
    auto *res = makeSplatZero(loc, rewriter, resultVectorType);

    auto unrollFactorsStorage = extractUnrollFactors(op);
    ArrayRef<int64_t> unrollFactors{unrollFactorsStorage};
    auto linearizationBasis = computeStrides(unrollFactors);
    auto nUnrolled = computeMaxLinearIndex(unrollFactors);
    SmallVector<int64_t, 4> strides(unrollFactors.size(), 1);
    for (unsigned idx = 0; idx < nUnrolled; ++idx) {
      auto offsets = delinearize(idx, linearizationBasis);
      offsets = zipMap([](int64_t v1, int64_t v2) { return v1 * v2; }, offsets,
                       unrollFactors);
      res = rewriter.create<vector::InsertStridedSliceOp>(
          loc, op->getOperand(idx), res, offsets, strides);
    }

    rewriter.replaceOp(op, res);
    return matchSuccess();
  }
};

// Simple DCE for fakeForkOps/fakeJoinOps, we do not want them to escape a
// transformation (otherwise the transformation is considered incorrect).
struct FakeForkTrait {
  static constexpr char const *name = kFakeForkOp;
};
struct FakeJoinTrait {
  static constexpr char const *name = kFakeJoinOp;
};

template <typename OpNameTrait> struct DCEPattern : public RewritePattern {
  DCEPattern(MLIRContext *context)
      // low-benefit to kick-in late
      : RewritePattern(OpNameTrait::name, 0, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    assert(op->getName().getStringRef() == kFakeForkOp ||
           op->getName().getStringRef() == kFakeJoinOp);
    if (!op->use_empty())
      return matchFailure();
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

void mlir::populateVectorToVectorConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    ArrayRef<int64_t> coarseVectorShape, ArrayRef<int64_t> fineVectorShape) {
  vector::populateWithGenerated(context, &patterns);
  patterns.insert<ConvertMatchingFakeForkFakeJoinOp,
                  ConvertFakeForkFromBlockArgsOp, ConvertFakeJoinOp,
                  DCEPattern<FakeForkTrait>, DCEPattern<FakeJoinTrait>>(
      context);
}
