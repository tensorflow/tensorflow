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

#include "mlir/Dialect/VectorOps/Utils.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/Dialect/VectorOps/VectorTransforms.h"
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

/// Computes and returns the linearized index of 'offsets' w.r.t. 'basis'.
static int64_t linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis) {
  assert(offsets.size() == basis.size());
  int64_t linearIndex = 0;
  for (unsigned idx = 0, e = basis.size(); idx < e; ++idx)
    linearIndex += offsets[idx] * basis[idx];
  return linearIndex;
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
  OperationState res(loc, op->getName().getStringRef(), operands, resultTypes,
                     op->getAttrs());
  return builder.createOperation(res);
}

// Helper function for Tablegen.
static bool hasShape(Value *v, ArrayRef<int64_t> shape) {
  auto t = v->getType().dyn_cast<ShapedType>();
  if (!t)
    return false;
  return std::equal(t.getShape().begin(), t.getShape().end(), shape.begin());
}

static Value *makeSplatZero(Location loc, PatternRewriter &rewriter,
                            VectorType vt) {
  auto t = vt.getElementType();
  Value *f = nullptr;
  if (t.isBF16() || t.isF16())
    f = rewriter.create<ConstantOp>(loc, t, rewriter.getF64FloatAttr(0.0f));
  else if (t.isF32())
    f = rewriter.create<ConstantOp>(loc, t, rewriter.getF32FloatAttr(0.0f));
  else if (t.isF64())
    f = rewriter.create<ConstantOp>(loc, t, rewriter.getF64FloatAttr(0.0f));
  if (f)
    return rewriter.create<SplatOp>(loc, vt, f);
  llvm_unreachable("Unsupported type in `makeSplatZero`");
}

// Populates 'resultElements[indexMap[i]]' with elements from 'inputElements[i]'
// for each index 'i' in inputElements with a valid mapping in 'indexMap'.
static void getMappedElements(const DenseMap<int64_t, int64_t> &indexMap,
                              ArrayRef<int64_t> inputElements,
                              SmallVectorImpl<int64_t> &resultElements) {
  assert(indexMap.size() == resultElements.size());
  assert(inputElements.size() >= resultElements.size());
  for (unsigned i = 0, e = inputElements.size(); i < e; ++i) {
    auto it = indexMap.find(i);
    if (it != indexMap.end())
      resultElements[it->second] = inputElements[i];
  }
}

// UnrolledOperandState aggregates per-operand state required for op unrolling.
struct UnrolledOperandState {
  Value *operand;
  SmallVector<int64_t, 4> unrolledShape;
  SmallVector<int64_t, 4> unrollFactors;
  SmallVector<int64_t, 8> basis;
  int64_t numInstances;
};

// Populates 'state' with unrolled shape, unroll factors, basis and
// num unrolled instances for 'operand'.
static void getUnrolledOperandState(Value *operand,
                                    const DenseMap<int64_t, int64_t> &indexMap,
                                    ArrayRef<int64_t> targetShape,
                                    UnrolledOperandState &state) {
  auto vectorType = operand->getType().cast<VectorType>();
  state.operand = operand;
  // Compute unrolled shape of 'operand'.
  state.unrolledShape.resize(vectorType.getRank());
  getMappedElements(indexMap, targetShape, state.unrolledShape);
  // Compute unroll factors for unrolled shape.
  auto maybeUnrollFactors =
      shapeRatio(vectorType.getShape(), state.unrolledShape);
  assert(maybeUnrollFactors.hasValue());
  state.unrollFactors = *maybeUnrollFactors;
  // Compute 'basis' and 'numInstances' based on 'state.unrollFactors'.
  state.basis = computeStrides(state.unrollFactors);
  state.numInstances = computeMaxLinearIndex(state.unrollFactors);
}

// Computes and returns the linear index of the unrolled vector at
// 'vectorOffsets' within the vector operand represented by 'state'.
static int64_t
getUnrolledOperandLinearIndex(UnrolledOperandState &state,
                              ArrayRef<int64_t> vectorOffsets,
                              DenseMap<int64_t, int64_t> &indexMap) {
  // Compute operand offsets.
  SmallVector<int64_t, 4> sliceOffsets(state.unrolledShape.size());
  getMappedElements(indexMap, vectorOffsets, sliceOffsets);
  // Compute and return linear index of 'sliceOffsets' w.r.t 'state.basis'.
  return linearize(sliceOffsets, state.basis);
}

// Returns an unrolled vector at 'vectorOffsets' within the vector operand
// represented by 'state'. The value is created if not present in 'cache'.
static Value *getOrCreateUnrolledOperandSlice(
    Location loc, UnrolledOperandState &state, ArrayRef<int64_t> vectorOffsets,
    ArrayRef<int64_t> offsets, DenseMap<int64_t, int64_t> &indexMap,
    SmallVectorImpl<Value *> &cache, PatternRewriter &builder) {
  // Compute operand offsets.
  SmallVector<int64_t, 4> sliceOffsets(state.unrolledShape.size());
  getMappedElements(indexMap, offsets, sliceOffsets);
  // TODO(b/144845578) Support non-1 strides.
  SmallVector<int64_t, 4> sliceStrides(state.unrolledShape.size(), 1);
  // Compute linear index of 'sliceOffsets' w.r.t 'state.basis'.
  int64_t sliceLinearIndex =
      getUnrolledOperandLinearIndex(state, vectorOffsets, indexMap);
  assert(sliceLinearIndex < static_cast<int64_t>(cache.size()));
  auto *operandSlice = cache[sliceLinearIndex];
  if (operandSlice == nullptr) {
    // Initialize 'cache' with slice from 'state.operand'.
    operandSlice = builder.create<vector::StridedSliceOp>(
        loc, state.operand, sliceOffsets, state.unrolledShape, sliceStrides);
    // Store value back to 'cache'.
    cache[sliceLinearIndex] = operandSlice;
  }
  return operandSlice;
}

//
// unrollSingleResultStructuredOp
//
// Returns a value representing the result of structured operation 'op'
// with iteration bounds 'iterationBounds' unrolled to 'targetShape'.
// An iteration space index map argument 'iterationIndexMapList' must be
// specified, with a map for each structured op input and a single map for the
// single result. The map at index 'indexMapListResultIndex' in the list must
// be the single result map.
//
// Example:
//
//  // Before unrolling
//
//   operand0                operand1                operand2
//       \                      |                      /
//        -------------------- opA --------------------
//
//  // After unrolling by 2
//
//   operand0                operand1                operand2
//   /      \                /      \                /      \
// slice00  slice01       slice10  slice11        slice20  slice21
//   \         |            |          |            /          |
//    -------------------- opA0 --------------------           |
//             |            |          |                       |
//              \           |          |                      /
//               -------------------- opA1 -------------------
//                          |          |
//                           \        /
//                           insertslice
//                                |

// TODO(andydavis) Generalize this to support structured ops beyond
// vector ContractionOp, and merge it with 'unrollSingleResultOpMatchingType'
static Value *unrollSingleResultStructuredOp(
    Operation *op, ArrayRef<int64_t> iterationBounds,
    std::vector<DenseMap<int64_t, int64_t>> &iterationIndexMapList,
    unsigned indexMapListResultIndex, ArrayRef<int64_t> targetShape,
    PatternRewriter &builder) {
  auto shapedType = op->getResult(0)->getType().dyn_cast_or_null<ShapedType>();
  if (!shapedType || !shapedType.hasStaticShape())
    assert(false && "Expected a statically shaped result type");

  // Compute unroll factors for 'iterationBounds' based on 'targetShape'
  auto maybeUnrollFactors = shapeRatio(iterationBounds, targetShape);
  if (!maybeUnrollFactors.hasValue())
    assert(false && "Failed to compute unroll factors for target shape");
  auto unrollFactors = *maybeUnrollFactors;

  // Compute unrolled operation state for each mapped operand.
  unsigned numMaps = iterationIndexMapList.size();
  SmallVector<UnrolledOperandState, 3> unrolledOperandState(numMaps);
  assert(op->getNumOperands() >= numMaps);
  for (unsigned i = 0; i < numMaps; ++i) {
    getUnrolledOperandState(op->getOperand(i), iterationIndexMapList[i],
                            targetShape, unrolledOperandState[i]);
  }
  // Compute number of total unrolled instances.
  auto numUnrolledInstances = computeMaxLinearIndex(unrollFactors);
  auto basis = computeStrides(unrollFactors);

  auto &resultOperandState = unrolledOperandState[indexMapListResultIndex];
  auto unrolledResultType = VectorType::get(resultOperandState.unrolledShape,
                                            shapedType.getElementType());

  // Initialize caches for intermediate vector results.
  std::vector<SmallVector<Value *, 4>> caches(numMaps);
  for (unsigned i = 0; i < numMaps; ++i) {
    caches[i].resize(unrolledOperandState[i].numInstances);
  }

  // Unroll 'numUnrolledInstances' of 'op', storing results in 'caches'.
  for (unsigned i = 0; i < numUnrolledInstances; ++i) {
    // De-linearize w.r.t. 'basis'.
    auto vectorOffsets = delinearize(i, basis);
    // Convert from unrolled vector-space offsets to element-space offsets.
    auto offsets = zipMap([](int64_t v1, int64_t v2) { return v1 * v2; },
                          vectorOffsets, targetShape);
    // Get cached slice (or create slice) for each operand at 'offsets'.
    SmallVector<Value *, 3> operands;
    operands.reserve(numMaps);
    for (unsigned i = 0; i < numMaps; ++i) {
      operands.push_back(getOrCreateUnrolledOperandSlice(
          op->getLoc(), unrolledOperandState[i], vectorOffsets, offsets,
          iterationIndexMapList[i], caches[i], builder));
    }
    // Create op on sliced vector arguments.
    auto resultVector =
        cloneOpWithOperandsAndTypes(builder, op->getLoc(), op, operands,
                                    unrolledResultType)
            ->getResult(0);

    // Compute linear result index.
    int64_t resultIndex = getUnrolledOperandLinearIndex(
        resultOperandState, vectorOffsets,
        iterationIndexMapList[indexMapListResultIndex]);
    // Update result cache at 'resultIndex'.
    caches[indexMapListResultIndex][resultIndex] = resultVector;
  }

  // Make zero splat into which we will insert results from
  // 'cache[indexMapListResultIndex]'
  auto resultVectorType = op->getResult(0)->getType().cast<VectorType>();
  auto *res = makeSplatZero(op->getLoc(), builder, resultVectorType);
  SmallVector<int64_t, 4> strides(resultOperandState.unrollFactors.size(), 1);
  // Insert vector accumulators into output.
  for (unsigned i = 0; i < resultOperandState.numInstances; ++i) {
    auto vectorOffsets = delinearize(i, resultOperandState.basis);
    // Convert from unrolled vector-space offsets to element-space offsets.
    auto offsets = zipMap([](int64_t v1, int64_t v2) { return v1 * v2; },
                          vectorOffsets, resultOperandState.unrolledShape);
    res = builder.create<vector::InsertStridedSliceOp>(
        op->getLoc(), caches[indexMapListResultIndex][i], res, offsets,
        strides);
  }

  return res;
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
Value * mlir::vector::unrollSingleResultOpMatchingType(PatternRewriter &builder,
                                               Operation *op,
                                               ArrayRef<int64_t> targetShape) {
  if (auto contractionOp = dyn_cast<vector::ContractionOp>(op)) {
    // Get contraction op iteration bounds.
    SmallVector<int64_t, 6> iterationBounds;
    contractionOp.getIterationBounds(iterationBounds);
    assert(iterationBounds.size() == targetShape.size());
    // Get map from iteration space index to lhs/rhs/result shape index.
    std::vector<DenseMap<int64_t, int64_t>> iterationIndexMapList;
    contractionOp.getIterationIndexMap(iterationIndexMapList);
    if (llvm::size(contractionOp.masks()) == 2) {
      // Add maps for lhs/rhs vector mask arguments (same lhs/rhs vector shape)
      iterationIndexMapList.push_back(iterationIndexMapList[0]);
      iterationIndexMapList.push_back(iterationIndexMapList[1]);
    }
    // Unroll 'op' 'iterationBounds' to 'targetShape'.
    // TODO(andydavis) Use linalg style 'args_in'/'args_out' to partition
    // 'iterationIndexMapList' instead of 'indexMapListResultIndex'.
    return unrollSingleResultStructuredOp(
        op, iterationBounds, iterationIndexMapList,
        /*indexMapListResultIndex=*/2, targetShape, builder);
  }
  // TODO(andydavis) Create trivial iteration bounds and index map for
  // elementwise operations and call 'unrollSingleResultStructuredOp'. Remove
  // fakefork/join if possible.

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
struct ConvertFakeForkFromBlockArgsOrTransferReadOp : public RewritePattern {
  ConvertFakeForkFromBlockArgsOrTransferReadOp(MLIRContext *context)
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

    auto *arg = op->getOperand(0);
    if (!isa<BlockArgument>(arg) &&
        !isa<vector::TransferReadOp>(arg->getDefiningOp()))
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
      assert(arg->getType() == op->getResult(0)->getType());
      rewriter.replaceOp(op, arg);
      return matchSuccess();
    }

    // Strides are always 1 for now.
    // TODO(b/144845578) support non-1 strides.
    auto forkedVectorType = arg->getType().cast<VectorType>();
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
              .create<vector::StridedSliceOp>(op->getLoc(), arg, offsets, sizes,
                                              strides)
              .getResult());
    }
    rewriter.replaceOp(op, extractedVectors);
    return matchSuccess();
  }
};

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
  vector::populateVectorToVectorCanonicalizationPatterns(patterns, context);
  patterns
      .insert<ConvertMatchingFakeForkFakeJoinOp,
              ConvertFakeForkFromBlockArgsOrTransferReadOp, ConvertFakeJoinOp,
              DCEPattern<FakeForkTrait>, DCEPattern<FakeJoinTrait>>(context);
}
