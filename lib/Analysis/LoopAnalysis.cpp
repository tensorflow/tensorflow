//===- LoopAnalysis.cpp - Misc loop analysis routines //-------------------===//
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
// This file implements miscellaneous loop analysis routines.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopAnalysis.h"

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Analysis/VectorAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Instructions.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/SuperVectorOps/SuperVectorOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/MathExtras.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include <type_traits>

using namespace mlir;

/// Returns the trip count of the loop as an affine expression if the latter is
/// expressible as an affine expression, and nullptr otherwise. The trip count
/// expression is simplified before returning.
AffineExpr mlir::getTripCountExpr(const ForInst &forInst) {
  // upper_bound - lower_bound
  int64_t loopSpan;

  int64_t step = forInst.getStep();
  auto *context = forInst.getContext();

  if (forInst.hasConstantBounds()) {
    int64_t lb = forInst.getConstantLowerBound();
    int64_t ub = forInst.getConstantUpperBound();
    loopSpan = ub - lb;
  } else {
    auto lbMap = forInst.getLowerBoundMap();
    auto ubMap = forInst.getUpperBoundMap();
    // TODO(bondhugula): handle max/min of multiple expressions.
    if (lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1)
      return nullptr;

    // TODO(bondhugula): handle bounds with different operands.
    // Bounds have different operands, unhandled for now.
    if (!forInst.matchingBoundOperandList())
      return nullptr;

    // ub_expr - lb_expr
    AffineExpr lbExpr(lbMap.getResult(0));
    AffineExpr ubExpr(ubMap.getResult(0));
    auto loopSpanExpr = simplifyAffineExpr(
        ubExpr - lbExpr, std::max(lbMap.getNumDims(), ubMap.getNumDims()),
        std::max(lbMap.getNumSymbols(), ubMap.getNumSymbols()));
    auto cExpr = loopSpanExpr.dyn_cast<AffineConstantExpr>();
    if (!cExpr)
      return loopSpanExpr.ceilDiv(step);
    loopSpan = cExpr.getValue();
  }

  // 0 iteration loops.
  if (loopSpan < 0)
    return 0;

  return getAffineConstantExpr(static_cast<uint64_t>(ceilDiv(loopSpan, step)),
                               context);
}

/// Returns the trip count of the loop if it's a constant, None otherwise. This
/// method uses affine expression analysis (in turn using getTripCount) and is
/// able to determine constant trip count in non-trivial cases.
llvm::Optional<uint64_t> mlir::getConstantTripCount(const ForInst &forInst) {
  auto tripCountExpr = getTripCountExpr(forInst);

  if (!tripCountExpr)
    return None;

  if (auto constExpr = tripCountExpr.dyn_cast<AffineConstantExpr>())
    return constExpr.getValue();

  return None;
}

/// Returns the greatest known integral divisor of the trip count. Affine
/// expression analysis is used (indirectly through getTripCount), and
/// this method is thus able to determine non-trivial divisors.
uint64_t mlir::getLargestDivisorOfTripCount(const ForInst &forInst) {
  auto tripCountExpr = getTripCountExpr(forInst);

  if (!tripCountExpr)
    return 1;

  if (auto constExpr = tripCountExpr.dyn_cast<AffineConstantExpr>()) {
    uint64_t tripCount = constExpr.getValue();

    // 0 iteration loops (greatest divisor is 2^64 - 1).
    if (tripCount == 0)
      return ULONG_MAX;

    // The greatest divisor is the trip count.
    return tripCount;
  }

  // Trip count is not a known constant; return its largest known divisor.
  return tripCountExpr.getLargestKnownDivisor();
}

bool mlir::isAccessInvariant(const Value &iv, const Value &index) {
  assert(isForInductionVar(&iv) && "iv must be a ForInst");
  assert(index.getType().isa<IndexType>() && "index must be of IndexType");
  SmallVector<OperationInst *, 4> affineApplyOps;
  getReachableAffineApplyOps({const_cast<Value *>(&index)}, affineApplyOps);

  if (affineApplyOps.empty()) {
    // Pointer equality test because of Value pointer semantics.
    return &index != &iv;
  }

  if (affineApplyOps.size() > 1) {
    affineApplyOps[0]->emitError(
        "CompositionAffineMapsPass must have been run: there should be at most "
        "one AffineApplyOp");
    return false;
  }

  auto composeOp = affineApplyOps[0]->cast<AffineApplyOp>();
  // We need yet another level of indirection because the `dim` index of the
  // access may not correspond to the `dim` index of composeOp.
  return !AffineValueMap(*composeOp).isFunctionOf(0, const_cast<Value *>(&iv));
}

llvm::DenseSet<const Value *>
mlir::getInvariantAccesses(const Value &iv,
                           llvm::ArrayRef<const Value *> indices) {
  llvm::DenseSet<const Value *> res;
  for (unsigned idx = 0, n = indices.size(); idx < n; ++idx) {
    auto *val = indices[idx];
    if (isAccessInvariant(iv, *val)) {
      res.insert(val);
    }
  }
  return res;
}

/// Given:
///   1. an induction variable `iv` of type ForInst;
///   2. a `memoryOp` of type const LoadOp& or const StoreOp&;
///   3. the index of the `fastestVaryingDim` along which to check;
/// determines whether `memoryOp`[`fastestVaryingDim`] is a contiguous access
/// along `iv`.
/// Contiguous is defined as either invariant or varying only along
/// `fastestVaryingDim`.
///
/// Prerequisites:
///   1. `iv` of the proper type;
///   2. the MemRef accessed by `memoryOp` has no layout map or at most an
///      identity layout map.
///
/// Currently only supports no layoutMap or identity layoutMap in the MemRef.
/// Returns false if the MemRef has a non-identity layoutMap or more than
/// 1 layoutMap. This is conservative.
///
// TODO(ntv): check strides.
template <typename LoadOrStoreOp>
static bool isContiguousAccess(const Value &iv, const LoadOrStoreOp &memoryOp,
                               unsigned fastestVaryingDim) {
  static_assert(std::is_same<LoadOrStoreOp, LoadOp>::value ||
                    std::is_same<LoadOrStoreOp, StoreOp>::value,
                "Must be called on either const LoadOp & or const StoreOp &");
  auto memRefType = memoryOp.getMemRefType();
  if (fastestVaryingDim >= memRefType.getRank()) {
    memoryOp.emitError("fastest varying dim out of bounds");
    return false;
  }

  auto layoutMap = memRefType.getAffineMaps();
  // TODO(ntv): remove dependence on Builder once we support non-identity
  // layout map.
  Builder b(memoryOp.getInstruction()->getContext());
  if (layoutMap.size() >= 2 ||
      (layoutMap.size() == 1 &&
       !(layoutMap[0] ==
         b.getMultiDimIdentityMap(layoutMap[0].getNumDims())))) {
    return memoryOp.emitError("NYI: non-trivial layoutMap"), false;
  }

  auto indices = memoryOp.getIndices();
  auto numIndices = llvm::size(indices);
  unsigned d = 0;
  for (auto index : indices) {
    if (fastestVaryingDim == (numIndices - 1) - d++) {
      continue;
    }
    if (!isAccessInvariant(iv, *index)) {
      return false;
    }
  }
  return true;
}

template <typename LoadOrStoreOpPointer>
static bool isVectorElement(LoadOrStoreOpPointer memoryOp) {
  auto memRefType = memoryOp->getMemRefType();
  return memRefType.getElementType().template isa<VectorType>();
}

static bool isVectorTransferReadOrWrite(const Instruction &inst) {
  const auto *opInst = cast<OperationInst>(&inst);
  return opInst->isa<VectorTransferReadOp>() ||
         opInst->isa<VectorTransferWriteOp>();
}

using VectorizableInstFun =
    std::function<bool(const ForInst &, const OperationInst &)>;

static bool isVectorizableLoopWithCond(const ForInst &loop,
                                       VectorizableInstFun isVectorizableInst) {
  if (!matcher::isParallelLoop(loop) && !matcher::isReductionLoop(loop)) {
    return false;
  }

  // No vectorization across conditionals for now.
  auto conditionals = matcher::If();
  auto *forInst = const_cast<ForInst *>(&loop);
  SmallVector<NestedMatch, 8> conditionalsMatched;
  conditionals.match(forInst, &conditionalsMatched);
  if (!conditionalsMatched.empty()) {
    return false;
  }

  // No vectorization across unknown regions.
  auto regions = matcher::Op([](const Instruction &inst) -> bool {
    auto &opInst = cast<OperationInst>(inst);
    return opInst.getNumBlockLists() != 0 && !opInst.isa<AffineIfOp>();
  });
  SmallVector<NestedMatch, 8> regionsMatched;
  regions.match(forInst, &regionsMatched);
  if (!regionsMatched.empty()) {
    return false;
  }

  auto vectorTransfers = matcher::Op(isVectorTransferReadOrWrite);
  SmallVector<NestedMatch, 8> vectorTransfersMatched;
  vectorTransfers.match(forInst, &vectorTransfersMatched);
  if (!vectorTransfersMatched.empty()) {
    return false;
  }

  auto loadAndStores = matcher::Op(matcher::isLoadOrStore);
  SmallVector<NestedMatch, 8> loadAndStoresMatched;
  loadAndStores.match(forInst, &loadAndStoresMatched);
  for (auto ls : loadAndStoresMatched) {
    auto *op = cast<OperationInst>(ls.getMatchedInstruction());
    auto load = op->dyn_cast<LoadOp>();
    auto store = op->dyn_cast<StoreOp>();
    // Only scalar types are considered vectorizable, all load/store must be
    // vectorizable for a loop to qualify as vectorizable.
    // TODO(ntv): ponder whether we want to be more general here.
    bool vector = load ? isVectorElement(load) : isVectorElement(store);
    if (vector) {
      return false;
    }
    if (!isVectorizableInst(loop, *op)) {
      return false;
    }
  }
  return true;
}

bool mlir::isVectorizableLoopAlongFastestVaryingMemRefDim(
    const ForInst &loop, unsigned fastestVaryingDim) {
  VectorizableInstFun fun(
      [fastestVaryingDim](const ForInst &loop, const OperationInst &op) {
        auto load = op.dyn_cast<LoadOp>();
        auto store = op.dyn_cast<StoreOp>();
        return load ? isContiguousAccess(*loop.getInductionVar(), *load,
                                         fastestVaryingDim)
                    : isContiguousAccess(*loop.getInductionVar(), *store,
                                         fastestVaryingDim);
      });
  return isVectorizableLoopWithCond(loop, fun);
}

bool mlir::isVectorizableLoop(const ForInst &loop) {
  VectorizableInstFun fun(
      // TODO: implement me
      [](const ForInst &loop, const OperationInst &op) { return true; });
  return isVectorizableLoopWithCond(loop, fun);
}

/// Checks whether SSA dominance would be violated if a for inst's body
/// instructions are shifted by the specified shifts. This method checks if a
/// 'def' and all its uses have the same shift factor.
// TODO(mlir-team): extend this to check for memory-based dependence
// violation when we have the support.
bool mlir::isInstwiseShiftValid(const ForInst &forInst,
                                ArrayRef<uint64_t> shifts) {
  auto *forBody = forInst.getBody();
  assert(shifts.size() == forBody->getInstructions().size());
  unsigned s = 0;
  for (const auto &inst : *forBody) {
    // A for or if inst does not produce any def/results (that are used
    // outside).
    if (const auto *opInst = dyn_cast<OperationInst>(&inst)) {
      for (unsigned i = 0, e = opInst->getNumResults(); i < e; ++i) {
        const Value *result = opInst->getResult(i);
        for (const InstOperand &use : result->getUses()) {
          // If an ancestor instruction doesn't lie in the block of forInst,
          // there is no shift to check. This is a naive way. If performance
          // becomes an issue, a map can be used to store 'shifts' - to look up
          // the shift for a instruction in constant time.
          if (auto *ancInst = forBody->findAncestorInstInBlock(*use.getOwner()))
            if (shifts[s] != shifts[forBody->findInstPositionInBlock(*ancInst)])
              return false;
        }
      }
    }
    s++;
  }
  return true;
}
