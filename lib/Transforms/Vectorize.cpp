//===- Vectorize.cpp - Vectorize Pass Impl ----------------------*- C++ -*-===//
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
// This file implements vectorization of loops, operations and data types to
// a target-independent, n-D virtual vector abstraction.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/MLFunctionMatcher.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLValue.h"
#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

/// This pass implements a high-level vectorization strategy at the MLFunction
/// level. This is implemented by:
///   1. matching arbitrarily nested loop patterns that are vectorizable;
///   2. analyzing those patterns for profitability;
///   3. applying those patterns iteratively by coarsening the loops, inserting
///      a single explicit vector element AllocOp and an unaligned load/store
///      operation. The full semantics of this unaligned load/store is still
///      TBD.
///
/// Loop transformation:
//  ====================
/// The choice of loop transformation to apply for coarsening vectorized loops
/// is still subject to exploratory tradeoffs. In particular, say we want to
/// vectorize by a factor 128, we want to transform the following input:
///     for %i = %M to %N {
///       %a = load(f(i))
///
///   Traditionally, one would vectorize late (after scheduling, tiling,
///   memory promotion etc) say after stripmining (and potentially unrolling in
///   the case of LLVM's SLP vectorizer):
///     for %i = floor(%M, 128) to ceil(%N, 128) {
///       for %ii = max(%M, 128 * %i) to min(%N, 128*%i + 127) {
///         load/store(f(ii))
///
///   We seek to vectorize early and freeze vector types before scheduling, so
///   we want to generate a pattern that resembles:
///     for %i = ? to ? step ? {
///       unaligned_load/unaligned_store(g(i))
///
///   i. simply dividing the lower / upper bounds by 128 creates issues
///   with representing expressions such as ii + 1 because now we only
///   have access to original values that have been divided. Additional
///   information is needed to specify accesses at below 128 granularity;
///   ii. another alternative is to coarsen the loop step but this may have
///   consequences on dependence analysis and fusability of loops: fusable
///   loops probably need to have the same step (because we don't want to
///   stripmine/unroll to enable fusion).
/// As a consequence, we choose to represent the coarsening using the loop
/// step for now and reevaluate in the future. Note that we can renormalize
/// loop steps later if/when we have evidence that they are problematic.
///
/// For the simple strawman example above, vectorizing for a 1-D vector
/// abstraction of size 128 returns code similar to:
///   %c0 = constant 0 : index
///   for %i = %M to %N + 127 step 128 {
///     %a = alloc() : memref<1xvector<128xf32>>
///     %r = "n_d_unaligned_load"(%tensor, %i, %a, %c0)
///     %16 = load %a[%c0] : memref<1xvector<128xf32>>
///
/// Note this is still work in progress and not yet functional.

#define DEBUG_TYPE "early-vect"

static cl::list<int> clVirtualVectorSize(
    "virtual-vector-size",
    cl::desc("Specify n-D virtual vector size for vectorization"),
    cl::ZeroOrMore);

static cl::list<int> clFastestVaryingPattern(
    "test-fastest-varying",
    cl::desc("Specify a 1-D, 2-D or 3-D pattern of fastest varying memory "
             "dimensions to match. See defaultPatterns in Vectorize.cpp for a "
             "description and examples. This is used for testing purposes"),
    cl::ZeroOrMore);

/// Forward declaration.
static FilterFunctionType
isVectorizableLoopPtrFactory(unsigned fastestVaryingMemRefDimension);

// Build a bunch of predetermined patterns that will be traversed in order.
// Due to the recursive nature of MLFunctionMatchers, this captures
// arbitrarily nested pairs of loops at any position in the tree.
/// Note that this currently only matches 2 nested loops and will be extended.
// TODO(ntv): support 3-D loop patterns with a common reduction loop that can
// be matched to GEMMs.
static std::vector<MLFunctionMatcher> defaultPatterns() {
  using matcher::For;
  return std::vector<MLFunctionMatcher>{
      // 3-D patterns
      For(isVectorizableLoopPtrFactory(2),
          For(isVectorizableLoopPtrFactory(1),
              For(isVectorizableLoopPtrFactory(0)))),
      // for i { for j { A[??f(not i, not j), f(i, not j), f(not i, j)];}}
      // test independently with:
      //   --test-fastest-varying=1 --test-fastest-varying=0
      For(isVectorizableLoopPtrFactory(1),
          For(isVectorizableLoopPtrFactory(0))),
      // for i { for j { A[??f(not i, not j), f(i, not j), ?, f(not i, j)];}}
      // test independently with:
      //   --test-fastest-varying=2 --test-fastest-varying=0
      For(isVectorizableLoopPtrFactory(2),
          For(isVectorizableLoopPtrFactory(0))),
      // for i { for j { A[??f(not i, not j), f(i, not j), ?, ?, f(not i, j)];}}
      // test independently with:
      //   --test-fastest-varying=3 --test-fastest-varying=0
      For(isVectorizableLoopPtrFactory(3),
          For(isVectorizableLoopPtrFactory(0))),
      // for i { for j { A[??f(not i, not j), f(not i, j), f(i, not j)];}}
      // test independently with:
      //   --test-fastest-varying=0 --test-fastest-varying=1
      For(isVectorizableLoopPtrFactory(0),
          For(isVectorizableLoopPtrFactory(1))),
      // for i { for j { A[??f(not i, not j), f(not i, j), ?, f(i, not j)];}}
      // test independently with:
      //   --test-fastest-varying=0 --test-fastest-varying=2
      For(isVectorizableLoopPtrFactory(0),
          For(isVectorizableLoopPtrFactory(2))),
      // for i { for j { A[??f(not i, not j), f(not i, j), ?, ?, f(i, not j)];}}
      // test independently with:
      //   --test-fastest-varying=0 --test-fastest-varying=3
      For(isVectorizableLoopPtrFactory(0),
          For(isVectorizableLoopPtrFactory(3))),
      // for i { A[??f(not i) , f(i)];}
      // test independently with:  --test-fastest-varying=0
      For(isVectorizableLoopPtrFactory(0)),
      // for i { A[??f(not i) , f(i), ?];}
      // test independently with:  --test-fastest-varying=1
      For(isVectorizableLoopPtrFactory(1)),
      // for i { A[??f(not i) , f(i), ?, ?];}
      // test independently with:  --test-fastest-varying=2
      For(isVectorizableLoopPtrFactory(2)),
      // for i { A[??f(not i) , f(i), ?, ?, ?];}
      // test independently with:  --test-fastest-varying=3
      For(isVectorizableLoopPtrFactory(3))};
}

static std::vector<MLFunctionMatcher> makePatterns() {
  using matcher::For;
  if (clFastestVaryingPattern.empty()) {
    return defaultPatterns();
  }
  switch (clFastestVaryingPattern.size()) {
  case 1:
    return {For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[0]))};
  case 2:
    return {For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[0]),
                For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[1])))};
  case 3:
    return {For(
        isVectorizableLoopPtrFactory(clFastestVaryingPattern[0]),
        For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[1]),
            For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[2]))))};
  default:
    assert(false && "Only up to 3-D fastest varying pattern supported atm");
  }
  return std::vector<MLFunctionMatcher>();
}

namespace {

struct Vectorize : public FunctionPass {
  PassResult runOnMLFunction(MLFunction *f) override;

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext MLContext;
};

} // end anonymous namespace

/////// TODO(ntv): Hoist to a VectorizationStrategy.cpp when appropriate. //////
namespace {

struct Strategy {
  DenseMap<ForStmt *, unsigned> loopToVectorDim;
};

} // end anonymous namespace

/// Implements a simple strawman strategy for vectorization.
/// Given a matched pattern `matches` of depth `patternDepth`, this strategy
/// greedily assigns the fastest varying dimension ** of the vector ** to the
/// innermost loop in the pattern.
/// When coupled with a pattern that looks for the fastest varying dimension in
/// load/store MemRefs, this creates a generic vectorization strategy that works
/// for any loop in a hierarchy (outermost, innermost or intermediate).
///
/// TODO(ntv): In the future we should additionally increase the power of the
/// profitability analysis along 3 directions:
///   1. account for loop extents (both static and parametric + annotations);
///   2. account for data layout permutations;
///   3. account for impact of vectorization on maximal loop fusion.
/// Then we can quantify the above to build a cost model and search over
/// strategies.
static bool analyzeProfitability(MLFunctionMatches matches,
                                 unsigned depthInPattern, unsigned patternDepth,
                                 Strategy *strategy) {
  for (auto m : matches) {
    auto *loop = cast<ForStmt>(m.first);
    LLVM_DEBUG(dbgs() << "[early-vect][profitability] patternDepth: "
                      << patternDepth << " depthInPattern: " << depthInPattern
                      << " loop ");
    LLVM_DEBUG(loop->print(dbgs()));
    LLVM_DEBUG(dbgs() << "\n");
    bool fail = analyzeProfitability(m.second, depthInPattern + 1, patternDepth,
                                     strategy);
    if (fail) {
      return fail;
    }
    assert(patternDepth > depthInPattern);
    if (patternDepth - depthInPattern <= clVirtualVectorSize.size()) {
      strategy->loopToVectorDim[loop] =
          clVirtualVectorSize.size() - (patternDepth - depthInPattern);
      LLVM_DEBUG(dbgs() << "[early-vect][profitability] vectorize @ "
                        << strategy->loopToVectorDim[loop] << " loop ");
      LLVM_DEBUG(loop->print(dbgs()));
      LLVM_DEBUG(dbgs() << "\n");
    } else {
      // Don't vectorize
      strategy->loopToVectorDim[loop] = -1;
    }
  }
  return false;
}
///// end TODO(ntv): Hoist to a VectorizationStrategy.cpp when appropriate /////

////// TODO(ntv): Hoist to a VectorizationMaterialize.cpp when appropriate. ////
/// Gets a MemRefType of 1 vector with the same elemental type as `tmpl` and
/// sizes specified by vectorSize. The MemRef lives in the same memory space as
/// tmpl. The MemRef should be promoted to a closer memory address space in a
/// later pass.
static MemRefType getVectorizedMemRefType(MemRefType tmpl,
                                          ArrayRef<int> vectorSizes) {
  auto elementType = tmpl.getElementType();
  assert(!elementType.dyn_cast<VectorType>() &&
         "Can't vectorize an already vector type");
  assert(tmpl.getAffineMaps().empty() &&
         "Unsupported non-implicit identity map");
  return MemRefType::get({1}, VectorType::get(vectorSizes, elementType), {},
                         tmpl.getMemorySpace());
}

/// Creates an unaligned load with the following semantics:
///   1. TODO(ntv): apply a `srcMap` to a `srcIndex` to represent a `srcMemRef`
///   slice + permutations for loading from non-fastest varying dimensions.
///   Note that generally, the fastest varying dimension should be part of the
///   map otherwise global layout changes are likely needed to obtain an
///   efficient load. This is an orthogonal cost model consideration;
///   2. load from the `srcMemRef` resulting from 1.;
///   3. store into a `dstMemRef` starting at offset `dstIndex`;
///   4. copy sizeof(dstMemRef) bytes with adjustements for boundaries;
///   5. TODO(ntv): broadcast along `broadcastMap` inside the `dstMemRef` to
///   support patterns like scalar-to-vector and N-k dim MemRef slice
/// The copy may overflow on the src side but not on the dst side. If the copy
/// overflows on the src side, the `dstMemRef` will be padded with enough values
/// to fill it completely.
///
/// Usage:
///   This n_d_unaligned_load op will be implemented as a PseudoOp for different
///   backends. In its current form it is only used to load into a <1xvector>;
///   where the vector may have any shape that is some multiple of the
///   hardware-specific vector size used to implement the PseudoOp efficiently.
///   This is used to implement "non-effecting padding" for early vectorization
///   and allows higher-level passes in the codegen to not worry about
///   hardware-specific implementation details.
///
/// TODO(ntv):
///   1. implement this end-to-end for some backend;
///   2. support operation-specific padding values to properly implement
///      "non-effecting padding";
///   3. support input map for on-the-fly transpositions (point 1 above);
///   4. support broadcast map (point 5 above).
///
/// TODO(andydavis,bondhugula,ntv):
///   1. generalize to support padding semantics and offsets within vector type.
static void createUnalignedLoad(MLFuncBuilder *b, Location *loc,
                                SSAValue *srcMemRef,
                                ArrayRef<SSAValue *> srcIndices,
                                SSAValue *dstMemRef,
                                ArrayRef<SSAValue *> dstIndices) {
  SmallVector<SSAValue *, 8> operands;
  operands.reserve(1 + srcIndices.size() + 1 + dstIndices.size());
  operands.insert(operands.end(), srcMemRef);
  operands.insert(operands.end(), srcIndices.begin(), srcIndices.end());
  operands.insert(operands.end(), dstMemRef);
  operands.insert(operands.end(), dstIndices.begin(), dstIndices.end());
  using functional::map;
  std::function<Type(SSAValue *)> getType = [](SSAValue *v) -> Type {
    return v->getType();
  };
  auto types = map(getType, operands);
  OperationState opState(b->getContext(), loc, "n_d_unaligned_load", operands,
                         types);
  b->createOperation(opState);
}

/// Creates an unaligned store with the following semantics:
///   1. TODO(ntv): apply a `srcMap` to a `srcIndex` to represent a `srcMemRef`
///   slice to support patterns like vector-to-scalar and N-k dim MemRef slice.
///   This is used as the counterpart to the broadcast map in the UnalignedLoad;
///   2. load from the `srcMemRef` resulting from 1.;
///   3. store into a `dstMemRef` starting at offset `dstIndex`;
///   4. TODO(ntv): apply a `dstMap` to a `dstIndex` to represent a `dstMemRef`
///   slice + permutations for storing into non-fastest varying dimensions.
///   Note that generally, the fastest varying dimension should be part of the
///   map otherwise global layout changes are likely needed to obtain an
///   efficient store. This is an orthogonal cost model consideration;
///   5. copy sizeof(srcMemRef) bytes with adjustements for boundaries;
/// The copy may overflow on the dst side but not on the dst side. If the copy
/// overflows on the dst side, the underlying implementation needs to resolve
/// potential races.
///
/// Usage:
///   This n_d_unaligned_store op will be implemented as a PseudoOp for
///   different backends. In its current form it is only used to store from a
///   <1xvector>; where the vector may have any shape that is some multiple of
///   the hardware-specific vector size used to implement the PseudoOp
///   efficiently. This is used to implement "non-effecting padding" for early
///   vectorization and allows higher-level passes in the codegen to not worry
///   about hardware-specific implementation details.
///
/// TODO(ntv):
///   1. implement this end-to-end for some backend;
///   2. support write-back in the presence of races and ;
///   3. support input map for counterpart of broadcast (point 1 above);
///   4. support dstMap for writing back in non-contiguous memory regions
///   (point 4 above).
static void createUnalignedStore(MLFuncBuilder *b, Location *loc,
                                 SSAValue *srcMemRef,
                                 ArrayRef<SSAValue *> srcIndices,
                                 SSAValue *dstMemRef,
                                 ArrayRef<SSAValue *> dstIndices) {
  SmallVector<SSAValue *, 8> operands;
  operands.reserve(1 + srcIndices.size() + 1 + dstIndices.size());
  operands.insert(operands.end(), srcMemRef);
  operands.insert(operands.end(), srcIndices.begin(), srcIndices.end());
  operands.insert(operands.end(), dstMemRef);
  operands.insert(operands.end(), dstIndices.begin(), dstIndices.end());
  using functional::map;
  std::function<Type(SSAValue *)> getType = [](SSAValue *v) -> Type {
    return v->getType();
  };
  auto types = map(getType, operands);
  OperationState opState(b->getContext(), loc, "n_d_unaligned_store", operands,
                         types);
  b->createOperation(opState);
}

/// The current implementation of vectorization materializes an AllocOp of
/// MemRef<1 x vector_type> + a custom unaligned load/store pseudoop.
/// The vector load/store accessing this MemRef always accesses element 0, so we
/// just memoize a single 0 SSAValue, once upon function entry to avoid clutter.
/// We create one such SSAValue per function.
static SSAValue *insertZeroIndex(MLFunction *f) {
  static thread_local DenseMap<MLFunction *, SSAValue *> zeros;
  if (zeros.find(f) == zeros.end()) {
    MLFuncBuilder b(f);
    auto zero = b.create<ConstantIndexOp>(b.getUnknownLoc(), 0);
    zeros.insert(std::make_pair(f, zero));
  }
  return zeros.lookup(f);
}

/// Unwraps a pointer type to another type (possibly the same).
/// Used in particular to allow easier compositions of
///   llvm::iterator_range<ForStmt::operand_iterator> types.
template <typename T, typename ToType = T>
static std::function<ToType *(T *)> unwrapPtr() {
  return [](T *val) { return dyn_cast<ToType>(val); };
}

/// Materializes the n-D vector into an unpromoted temporary storage and
/// explicitly copy into it. Materialization occurs in a MemRef containing 1
/// vector that lives in the same memory space as the base MemRef. Later passes
/// should make the decision to promote this materialization to a faster address
/// space.
template <typename LoadOrStoreOpPointer>
static MLValue *materializeVector(MLValue *iv, LoadOrStoreOpPointer memoryOp,
                                  ArrayRef<int> vectorSize) {
  auto memRefType =
      memoryOp->getMemRef()->getType().template cast<MemRefType>();
  auto vectorMemRefType = getVectorizedMemRefType(memRefType, vectorSize);

  // Materialize a MemRef with 1 vector.
  auto *opStmt = cast<OperationStmt>(memoryOp->getOperation());
  MLFuncBuilder b(opStmt);
  // Create an AllocOp to apply the new shape.
  auto allocOp = b.create<AllocOp>(opStmt->getLoc(), vectorMemRefType,
                                   ArrayRef<SSAValue *>{});
  auto *allocMemRef = memoryOp->getMemRef();
  using namespace functional;
  if (opStmt->template isa<LoadOp>()) {
    createUnalignedLoad(&b, opStmt->getLoc(), allocMemRef,
                        map(unwrapPtr<SSAValue>(), memoryOp->getIndices()),
                        allocOp->getResult(),
                        {insertZeroIndex(iv->getFunction())});
  } else {
    createUnalignedStore(&b, opStmt->getLoc(), allocOp->getResult(),
                         {insertZeroIndex(iv->getFunction())}, allocMemRef,
                         map(unwrapPtr<SSAValue>(), memoryOp->getIndices()));
  }

  return cast<MLValue>(allocOp->getResult());
}
/// end TODO(ntv): Hoist to a VectorizationMaterialize.cpp when appropriate. ///

namespace {

struct VectorizationState {
  // `vectorizedByThisPattern` keeps track of statements that have already been
  // vectorized by this pattern. This allows distinguishing between
  DenseSet<OperationStmt *> vectorizedByThisPattern;
  DenseSet<ForStmt *> vectorized;
  const Strategy *strategy;
};
} // end anonymous namespace

/// Terminal template function for creating a LoadOp.
static OpPointer<LoadOp> createLoad(MLFuncBuilder *b, Location *loc,
                                    MLValue *memRef) {
  return b->create<LoadOp>(
      loc, memRef,
      ArrayRef<SSAValue *>{insertZeroIndex(memRef->getFunction())});
}

/// Terminal template function for creating a StoreOp.
static OpPointer<StoreOp> createStore(MLFuncBuilder *b, Location *loc,
                                      MLValue *memRef,
                                      OpPointer<StoreOp> store) {
  return b->create<StoreOp>(
      loc, store->getValueToStore(), memRef,
      ArrayRef<SSAValue *>{insertZeroIndex(memRef->getFunction())});
}

/// Vectorizes the `memoryOp` of type LoadOp or StoreOp along loop `iv` by
/// factor `vectorSize`.
/// In a first implementation, this triggers materialization of a vector Alloc.
// TODO(ntv): this could be a view that changes the underlying element type.
// Materialization of this view may or may not happen before legalization.
template <typename LoadOrStoreOpPointer>
static bool vectorize(MLValue *iv, LoadOrStoreOpPointer memoryOp,
                      ArrayRef<int> vectorSize, VectorizationState *state) {
  auto *materializedMemRef = materializeVector(iv, memoryOp, vectorSize);
  auto *opStmt = cast<OperationStmt>(memoryOp->getOperation());
  MLFuncBuilder b(opStmt);
  Operation *resultOperation;
  if (auto load = opStmt->template dyn_cast<LoadOp>()) {
    auto res = createLoad(&b, opStmt->getLoc(), materializedMemRef);
    resultOperation = res->getOperation();
  } else {
    auto store = opStmt->template dyn_cast<StoreOp>();
    auto res = createStore(&b, opStmt->getLoc(), materializedMemRef, store);
    resultOperation = res->getOperation();
  }
  state->vectorizedByThisPattern.insert(cast<OperationStmt>(resultOperation));
  return false;
}

// result == true => failure, TO
// (ntv): Status enum
static bool vectorizeForStmt(ForStmt *loop, AffineMap upperBound,
                             ArrayRef<int> vectorSize, int64_t step,
                             VectorizationState *state) {
  LLVM_DEBUG(dbgs() << "[early-vect] vectorize loop ");
  LLVM_DEBUG(loop->print(dbgs()));
  LLVM_DEBUG(dbgs() << "\n");

  using namespace functional;
  loop->setUpperBound(map(unwrapPtr<MLValue>(), loop->getUpperBoundOperands()),
                      upperBound);
  loop->setStep(step);

  FilterFunctionType notVectorizedThisRound = [state](const Statement &stmt) {
    if (!matcher::isLoadOrStore(stmt)) {
      return false;
    }
    return state->vectorizedByThisPattern.count(cast<OperationStmt>(&stmt)) ==
           0;
  };
  auto loadAndStores = matcher::Op(notVectorizedThisRound);
  auto matches = loadAndStores.match(loop);
  for (auto ls : matches) {
    auto *opStmt = cast<OperationStmt>(ls.first);
    auto load = opStmt->dyn_cast<LoadOp>();
    auto store = opStmt->dyn_cast<StoreOp>();
    LLVM_DEBUG(dbgs() << "[early-vect] vectorize op: ");
    LLVM_DEBUG(opStmt->print(dbgs()));
    LLVM_DEBUG(dbgs() << "\n");
    bool vectorizationFails = load ? vectorize(loop, load, vectorSize, state)
                                   : vectorize(loop, store, vectorSize, state);
    LLVM_DEBUG(dbgs() << "fail: " << vectorizationFails << "\n");
    if (vectorizationFails) {
      // Early exit and trigger RAII cleanups at the root.
      return true;
    }
    // Erase the original op.
    opStmt->erase();
  }
  return false;
}

/// Returns a FilterFunctionType that can be used in MLFunctionMatcher to
/// match a loop whose underlying load/store accesses are all varying along the
/// `fastestVaryingMemRefDimension`.
/// TODO(ntv): In the future, allow more interesting mixed layout permutation
/// once we understand better the performance implications and we are confident
/// we can build a cost model and a search procedure.
static FilterFunctionType
isVectorizableLoopPtrFactory(unsigned fastestVaryingMemRefDimension) {
  return [fastestVaryingMemRefDimension](const Statement &forStmt) {
    const auto &loop = cast<ForStmt>(forStmt);
    return isVectorizableLoopAlongFastestVaryingMemRefDim(
        loop, fastestVaryingMemRefDimension);
  };
}

/// Forward-declaration.
static bool vectorizeNonRoot(MLFunctionMatches matches,
                             VectorizationState *state);

/// Apply vectorization of `loop` according to `state`. This is only triggered
/// if all vectorizations in `childrenMatches` have already succeeded
/// recursively in DFS post-order.
static bool doVectorize(ForStmt *loop, MLFunctionMatches childrenMatches,
                        VectorizationState *state) {
  // 1. DFS postorder recursion, if any of my children fails, I fail too.
  auto fail = vectorizeNonRoot(childrenMatches, state);
  if (fail) {
    // Early exit and trigger RAII cleanups at the root.
    return true;
  }

  // 2. This loop may have been omitted from vectorization for various reasons
  // (e.g. due to the performance model or pattern depth > vector size).
  assert(state->strategy->loopToVectorDim.count(loop));
  assert(state->strategy->loopToVectorDim.find(loop) !=
             state->strategy->loopToVectorDim.end() &&
         "Key not found");
  int vectorDim = state->strategy->loopToVectorDim.lookup(loop);
  if (vectorDim < 0) {
    return false;
  }

  // 3. Actual post-order transformation.
  assert(vectorDim < clVirtualVectorSize.size() && "vector dim overflow");
  //   a. get actual vector size
  auto vectorSize = clVirtualVectorSize[vectorDim];
  //   b. loop transformation for early vectorization is still subject to
  //     exploratory tradeoffs (see top of the file).
  auto ubMap = loop->getUpperBoundMap();
  assert(ubMap.getRangeSizes().empty());
  //   c. apply coarsening, i.e.:
  //        | ub -> ub = vectorSize - 1
  //        | step -> step * vectorSize
  std::function<AffineExpr(AffineExpr)> coarsenUb =
      [vectorSize](AffineExpr expr) { return expr + vectorSize - 1; };
  auto newUbs = functional::map(coarsenUb, ubMap.getResults());
  //   d. recurse
  return vectorizeForStmt(
      loop,
      AffineMap::get(ubMap.getNumDims(), ubMap.getNumSymbols(), newUbs, {}),
      clVirtualVectorSize, loop->getStep() * vectorSize, state);
}

/// Non-root pattern iterates over the matches at this level, calls doVectorize
/// and exits early if anything below fails.
static bool vectorizeNonRoot(MLFunctionMatches matches,
                             VectorizationState *state) {
  for (auto m : matches) {
    auto fail = doVectorize(cast<ForStmt>(m.first), m.second, state);
    if (fail) {
      // Early exit and trigger RAII cleanups at the root.
      return true;
    }
  }
  return false;
}

/// Sets up error handling for this root loop.
/// Vectorization is a recursive procedure where anything below can fail.
/// The root match thus needs to maintain a clone for handling failure.
/// Each root may succeed independently but will otherwise clean after itself if
/// anything below it fails.
static bool vectorizeRoot(MLFunctionMatches matches,
                          VectorizationState *state) {
  for (auto m : matches) {
    auto *loop = cast<ForStmt>(m.first);
    // Since patterns are recursive, they can very well intersect.
    // Since we do not want a fully greedy strategy in general, we decouple
    // pattern matching, from profitability analysis, from application.
    // As a consequence we must check that each root pattern is still
    // vectorizable. If a pattern is not vectorizable anymore, we just skip it.
    // TODO(ntv): implement a non-greedy profitability analysis that keeps only
    // non-intersecting patterns.
    if (!isVectorizableLoop(*loop)) {
      continue;
    }
    DenseMap<const MLValue *, MLValue *> nomap;
    MLFuncBuilder builder(loop->getFunction());
    ForStmt *clonedLoop = cast<ForStmt>(builder.clone(*loop, nomap));
    auto fail = doVectorize(loop, m.second, state);
    if (!fail) {
      LLVM_DEBUG(dbgs() << "[early-vect] success vectorizing loop: ");
      LLVM_DEBUG(loop->print(dbgs()));
      LLVM_DEBUG(dbgs() << "\n");
    }
    fail ? loop->erase() : clonedLoop->erase();
  }
  return false;
}

/// Applies vectorization to the current MLFunction by searching over a bunch of
/// predetermined patterns.
PassResult Vectorize::runOnMLFunction(MLFunction *f) {
  /// Build a zero at the entry of the function to avoid clutter in every single
  /// vectorized loop.
  insertZeroIndex(f);

  for (auto pat : makePatterns()) {
    LLVM_DEBUG(dbgs() << "\n[early-vect] Input function is now:\n");
    LLVM_DEBUG(f->print(dbgs()));
    LLVM_DEBUG(dbgs() << "\n[early-vect] match:\n");
    auto matches = pat.match(f);
    Strategy strategy;
    auto fail = analyzeProfitability(matches, 0, pat.getDepth(), &strategy);
    assert(!fail);
    VectorizationState state;
    state.strategy = &strategy;
    fail = vectorizeRoot(matches, &state);
    assert(!fail);
  }

  return PassResult::Success;
}

FunctionPass *mlir::createVectorizePass() { return new Vectorize(); }
