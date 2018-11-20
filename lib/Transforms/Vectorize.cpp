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
#include "mlir/Analysis/VectorAnalysis.h"
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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
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
///   3. applying those patterns iteratively by coarsening the loops and turning
///      load/store operations into opaque vector_transfer_read/write ops that
///      will be lowered in a subsequent pass (either into finer-grained MLIR
///      ops or in the lower-level emitters);
///   4. traversing the use-def chains to propagate the vector types to ops.
///
/// Vector granularity:
/// ===================
/// This pass is designed to perform vectorization at the granularity of
/// super-vectors. For a particular target, a notion of minimal n-d vector size
/// will be specified and early vectorization targets a multiple of those.
/// Some particular sizes of interest include:
///   - CPU: (HW vector size), (core count x HW vector size),
///          (socket count x core count x HW vector size);
///   - GPU: warp size, (warp x float2, float4, 4x4x4 tensor core) sizes.
/// Loops, load/stores and operations are emitted that operate on super-vector
/// shapes. A later lowering pass will materialize to actual HW vector sizes.
/// This lowering may be occur at different times:
///   1. at the MLIR level into DmaStartOp + DmaWaitOp + vectorized operations
///      for data transformations and shuffle; thus opening opportunities for
///      unrolling and pipelining; or
///   2. later in the a target-specific lowering pass, achieving full separation
///      of concerns; or
///   3. a partial mix of both.
///
/// Loop transformation:
//  ====================
/// The choice of loop transformation to apply for coarsening vectorized loops
/// is still subject to exploratory tradeoffs. In particular, say we want to
/// vectorize by a factor 128, we want to transform the following input:
///     for %i = %M to %N {
///       %a = load A[%i] : memref<?xf32>
///     }
///
///   Traditionally, one would vectorize late (after scheduling, tiling,
///   memory promotion etc) say after stripmining (and potentially unrolling in
///   the case of LLVM's SLP vectorizer):
///     for %i = floor(%M, 128) to ceil(%N, 128) {
///       for %ii = max(%M, 128 * %i) to min(%N, 128*%i + 127) {
///         %a = load A[%ii] : memref<?xf32>
///
///   Instead, we seek to vectorize early and freeze vector types before
///   scheduling, so we want to generate a pattern that resembles:
///     for %i = ? to ? step ? {
///       %v_a = "vector_transfer_read" (A, %i) : (memref<?xf32>, index) ->
///       vector<128xf32>
///
///   i. simply dividing the lower / upper bounds by 128 creates issues
///      when representing expressions such as ii + 1 because now we only
///      have access to original values that have been divided. Additional
///      information is needed to specify accesses at below-128 granularity;
///   ii. another alternative is to coarsen the loop step but this may have
///      consequences on dependence analysis and fusability of loops: fusable
///      loops probably need to have the same step (because we don't want to
///      stripmine/unroll to enable fusion).
/// As a consequence, we choose to represent the coarsening using the loop
/// step for now and reevaluate in the future. Note that we can renormalize
/// loop steps later if/when we have evidence that they are problematic.
///
/// For the simple strawman example above, vectorizing for a 1-D vector
/// abstraction of size 128 returns code similar to:
///   for %i = %M to %N step 128 {
///     %v_a = "vector_transfer_read" (A, %i) : (memref<?xf32>, index) ->
///     vector<128xf32>
///
/// Note this is still work in progress and not yet functional.
/// It is the reponsibility of the implementation of the vector_transfer_read
/// implementation's responsibility to turn scalar memrefs into vector
/// registers. This is target dependent. In the future, these operations will
/// expose a contract to constrain early vectorization.

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
  Vectorize() : FunctionPass(&Vectorize::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext MLContext;

  static char passID;
};

} // end anonymous namespace

char Vectorize::passID = 0;

/////// TODO(ntv): Hoist to a VectorizationStrategy.cpp when appropriate. //////
namespace {

struct Strategy {
  ArrayRef<int> vectorSizes;
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
    bool fail = analyzeProfitability(m.second, depthInPattern + 1, patternDepth,
                                     strategy);
    if (fail) {
      return fail;
    }
    assert(patternDepth > depthInPattern);
    if (patternDepth - depthInPattern <= strategy->vectorSizes.size()) {
      strategy->loopToVectorDim[loop] =
          strategy->vectorSizes.size() - (patternDepth - depthInPattern);
    } else {
      // Don't vectorize
      strategy->loopToVectorDim[loop] = -1;
    }
  }
  return false;
}
///// end TODO(ntv): Hoist to a VectorizationStrategy.cpp when appropriate /////

namespace {

struct VectorizationState {
  /// Adds an entry of pre/post vectorization statements in the state.
  void registerReplacement(OperationStmt *key, OperationStmt *value);
  /// When the current vectorization pattern is successful, this erases the
  /// instructions that were marked for erasure in the proper order and resets
  /// the internal state for the next pattern.
  void finishVectorizationPattern();

  // In-order tracking of original OperationStmt that have been vectorized.
  // Erase in reverse order.
  SmallVector<OperationStmt *, 16> toErase;
  // Set of OperationStmt that have been vectorized (the values in the
  // vectorizationMap for hashed access)
  DenseSet<OperationStmt *> vectorizedSet;
  // Map of old unvectorized OperationStmt to new vectorized OperationStmt.
  DenseMap<OperationStmt *, OperationStmt *> vectorizationMap;
  // Map of old unvectorized MLValue to new vectorized MLValue.
  DenseMap<const MLValue *, MLValue *> replacementMap;
  // Enclosing loops are pushed, popped as the vectorization algorithm recurses.
  SmallVector<ForStmt *, 8> enclosingLoops;
  // The strategy drives which loop to vectorize by which amount.
  const Strategy *strategy;

  void enterLoop(ForStmt *loop) { enclosingLoops.push_back(loop); }
  void exitLoop(ForStmt *loop) {
    auto *poppedLoop = enclosingLoops.pop_back_val();
    (void)poppedLoop;
    assert(poppedLoop == loop && "Not the same loop");
  }

private:
  void registerReplacement(const SSAValue *key, SSAValue *value);
};

} // end namespace

void VectorizationState::registerReplacement(OperationStmt *key,
                                             OperationStmt *value) {
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ vectorize op: ");
  LLVM_DEBUG(key->print(dbgs()));
  LLVM_DEBUG(dbgs() << "  into  ");
  LLVM_DEBUG(value->print(dbgs()));
  assert(key->getNumResults() == 1);
  assert(value->getNumResults() == 1);
  assert(vectorizedSet.count(value) == 0);
  assert(vectorizationMap.count(key) == 0);
  toErase.push_back(key);
  vectorizedSet.insert(value);
  vectorizationMap.insert(std::make_pair(key, value));
  registerReplacement(key->getResult(0), value->getResult(0));
}

void VectorizationState::finishVectorizationPattern() {
  while (!toErase.empty()) {
    auto *stmt = toErase.pop_back_val();
    stmt->erase();
  }
  vectorizationMap.clear();
  replacementMap.clear();
}

void VectorizationState::registerReplacement(const SSAValue *key,
                                             SSAValue *value) {
  assert(replacementMap.count(cast<MLValue>(key)) == 0);
  replacementMap.insert(
      std::make_pair(cast<MLValue>(key), cast<MLValue>(value)));
}

////// TODO(ntv): Hoist to a VectorizationMaterialize.cpp when appropriate. ////

/// Creates a vector_transfer_read that loads a scalar MemRef into a
/// super-vector register.
///
/// Usage:
///   This vector_transfer_read op will be implemented as a PseudoOp for
///   different backends. In its current form it is only used to load into a
///   vector; where the vector may have any shape that is some multiple of the
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
static OperationStmt *
createVectorTransferRead(MLFuncBuilder *b, Location loc, VectorType vectorType,
                         SSAValue *srcMemRef, ArrayRef<SSAValue *> srcIndices) {
  SmallVector<SSAValue *, 8> operands;
  operands.reserve(1 + srcIndices.size());
  operands.insert(operands.end(), srcMemRef);
  operands.insert(operands.end(), srcIndices.begin(), srcIndices.end());
  OperationState opState(b->getContext(), loc, kVectorTransferReadOpName,
                         operands, vectorType);
  return b->createOperation(opState);
}

/// Creates a vector_transfer_write writes a super-vector register into a scalar
/// MemRef.
///
/// Usage:
///   This vector_transfer_read op will be implemented as a PseudoOp for
///   different backends. In its current form it is only used to load into a
///   vector; where the vector may have any shape that is some multiple of the
///   hardware-specific vector size used to implement the PseudoOp efficiently.
///   This is used to implement "non-effecting padding" for early vectorization
///   and allows higher-level passes in the codegen to not worry about
///   hardware-specific implementation details.
///
/// Usage:
///   This vector_transfer_write op will be implemented as a PseudoOp for
///   different backends. In its current form it is only used to store from a
///   vector; where the vector may have any shape that is some multiple of
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
static OperationStmt *
createVectorTransferWrite(MLFuncBuilder *b, Location loc, VectorType vectorType,
                          OperationStmt *storeOp,
                          ArrayRef<SSAValue *> dstIndices) {
  auto store = storeOp->cast<StoreOp>();
  SmallVector<SSAValue *, 8> operands;
  operands.reserve(1 + 1 + dstIndices.size());
  // If the value to store is:
  // 1. a vector == vectorType, we just insert the value;
  // 2. a scalar constant, we splat it into the vectorType;
  // 3. a scalar of non-index type, we insert the value, it will be turned into
  //    a vector when traversing the use-def chains;
  // 4. a non-constant scalar of index type: unsupported, it may be loop
  //    dependent and broadcasting into a vector requires additional machinery.
  //    TODO(ntv): support non-constant loop-variant scalars.
  // 5. a vector != vectorType, this is unsupported atm.
  //    TODO(ntv): support broadcasting if the types are comformable.
  auto *value = store->getValueToStore();
  if (value->getType() == vectorType) {
    operands.insert(operands.end(), value);
  } else if (VectorType::isValidElementType(value->getType())) {
    if (auto constant = value->getDefiningStmt()->dyn_cast<ConstantOp>()) {
      assert(constant && "NYI: non-constant scalar broadcast");
      auto attr = SplatElementsAttr::get(vectorType, constant->getValue());
      auto *constantOpStmt = cast<OperationStmt>(constant->getOperation());
      SmallString<16> name(constantOpStmt->getName().getStringRef());
      OperationState opState(b->getContext(), loc, name, {}, {vectorType});
      auto *splat = cast<OperationStmt>(b->createOperation(opState));
      splat->setAttr(Identifier::get("value", b->getContext()), attr);
      operands.insert(operands.end(), cast<OperationStmt>(splat)->getResult(0));
    } else if (!value->getType().isa<IndexType>()) {
      operands.insert(operands.end(), value);
    } else {
      assert(false && "NYI: cannot vectorize index, it may be loop dependent");
    }
  } else {
    assert(false && "NYI: cannot vectorize an invalid element type");
  }
  operands.insert(operands.end(), store->getMemRef());
  operands.insert(operands.end(), dstIndices.begin(), dstIndices.end());
  OperationState opState(b->getContext(), loc, kVectorTransferWriteOpName,
                         operands, {});
  return b->createOperation(opState);
}

/// Unwraps a pointer type to another type (possibly the same).
/// Used in particular to allow easier compositions of
///   llvm::iterator_range<ForStmt::operand_iterator> types.
template <typename T, typename ToType = T>
static std::function<ToType *(T *)> unwrapPtr() {
  return [](T *val) { return dyn_cast<ToType>(val); };
}

/// Materializes the n-D vector into an explicit vector type.
template <typename LoadOrStoreOpPointer>
static OperationStmt *materializeVector(MLValue *iv,
                                        LoadOrStoreOpPointer memoryOp,
                                        VectorizationState *state) {
  auto memRefType =
      memoryOp->getMemRef()->getType().template cast<MemRefType>();

  auto elementType = memRefType.getElementType();
  assert(VectorType::isValidElementType(elementType) &&
         "Can't vectorize an already vector type");
  auto vectorType = VectorType::get(state->strategy->vectorSizes, elementType);

  // Materialize a MemRef with 1 vector.
  auto *opStmt = cast<OperationStmt>(memoryOp->getOperation());
  MLFuncBuilder b(opStmt);
  OperationStmt *res;
  // For now, vector_transfers must be aligned, operate only on indices with an
  // identity subset of AffineMap and do not change layout.
  // TODO(ntv): increase the expressiveness power of vector_transfer operations
  // as needed by various targets.
  if (opStmt->template isa<LoadOp>()) {
    res = createVectorTransferRead(
        &b, opStmt->getLoc(), vectorType, memoryOp->getMemRef(),
        functional::map(unwrapPtr<SSAValue>(), memoryOp->getIndices()));
  } else {
    res = createVectorTransferWrite(
        &b, opStmt->getLoc(), vectorType, opStmt,
        functional::map(unwrapPtr<SSAValue>(), memoryOp->getIndices()));
  }

  return res;
}
/// end TODO(ntv): Hoist to a VectorizationMaterialize.cpp when appropriate. ///

/// Vectorizes the `store` along loop `iv` according to `state`.
static bool vectorizeStore(MLValue *iv, OpPointer<StoreOp> store,
                           VectorizationState *state) {
  materializeVector(iv, store, state);
  // Stores define no result and do not need to be registered for replacements,
  // we can immediately delete them.
  store->erase();
  return false;
}

/// Vectorizes the `load` along loop `iv` accordingto `state`.
static bool vectorizeLoad(MLValue *iv, OpPointer<LoadOp> load,
                          VectorizationState *state) {
  auto *vectorizedLoad = materializeVector(iv, load, state);
  MLFuncBuilder b(cast<OperationStmt>(load->getOperation()));
  state->registerReplacement(cast<OperationStmt>(load->getOperation()),
                             vectorizedLoad);
  return false;
}

// Coarsens the loops bounds and transforms all remaining load and store
// operations into the appropriate vector_transfer.
static bool vectorizeForStmt(ForStmt *loop, int64_t step,
                             VectorizationState *state) {
  using namespace functional;
  loop->setStep(step);

  FilterFunctionType notVectorizedThisPattern = [state](const Statement &stmt) {
    if (!matcher::isLoadOrStore(stmt)) {
      return false;
    }
    auto *opStmt = cast<OperationStmt>(&stmt);
    return state->vectorizationMap.count(opStmt) == 0 &&
           state->vectorizedSet.count(opStmt) == 0;
  };
  auto loadAndStores = matcher::Op(notVectorizedThisPattern);
  auto matches = loadAndStores.match(loop);
  for (auto ls : matches) {
    auto *opStmt = cast<OperationStmt>(ls.first);
    auto load = opStmt->dyn_cast<LoadOp>();
    auto store = opStmt->dyn_cast<StoreOp>();
    LLVM_DEBUG(opStmt->print(dbgs()));
    auto fail = load ? vectorizeLoad(loop, load, state)
                     : vectorizeStore(loop, store, state);
    if (fail) {
      return fail;
    }
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
static bool doVectorize(MLFunctionMatches::EntryType oneMatch,
                        VectorizationState *state) {
  ForStmt *loop = cast<ForStmt>(oneMatch.first);
  state->enterLoop(loop);
  functional::ScopeGuard sg([state, loop]() { state->exitLoop(loop); });
  MLFunctionMatches childrenMatches = oneMatch.second;

  // 1. DFS postorder recursion, if any of my children fails, I fail too.
  auto fail = vectorizeNonRoot(childrenMatches, state);
  if (fail) {
    // Early exit and trigger RAII cleanups at the root.
    return fail;
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
  assert(vectorDim < state->strategy->vectorSizes.size() &&
         "vector dim overflow");
  //   a. get actual vector size
  auto vectorSize = state->strategy->vectorSizes[vectorDim];
  //   b. loop transformation for early vectorization is still subject to
  //     exploratory tradeoffs (see top of the file). Apply coarsening, i.e.:
  //        | ub -> ub
  //        | step -> step * vectorSize
  LLVM_DEBUG(dbgs() << "\n[early-vect] vectorizeForStmt by " << vectorSize
                    << " : ");
  LLVM_DEBUG(loop->print(dbgs()));
  return vectorizeForStmt(loop, loop->getStep() * vectorSize, state);
}

/// Non-root pattern iterates over the matches at this level, calls doVectorize
/// and exits early if anything below fails.
static bool vectorizeNonRoot(MLFunctionMatches matches,
                             VectorizationState *state) {
  for (auto m : matches) {
    auto fail = doVectorize(m, state);
    if (fail) {
      // Early exit and trigger RAII cleanups at the root.
      return fail;
    }
  }
  return false;
}

/// Iterates over the OperationStmt in the loop and rewrites them using their
/// vectorized counterpart by:
///   1. iteratively building a worklist of uses of the OperationStmt vectorized
///   so far by this pattern;
///   2. for each OperationStmt in the worklist, create the vector form of this
///   operation and replace all its uses by the vectorized form. For this step,
///   the worklist must be traversed in order;
///   3. verify that all operands of the newly vectorized operation have been
///   vectorized by this pattern.
/// TODO(ntv): step 3. can be relaxed with simple broadcast.
static bool vectorizeOperations(ForStmt *loop, VectorizationState *state) {
  LLVM_DEBUG(dbgs() << "\n[early-vect] vectorizeOperations in: ");
  LLVM_DEBUG(loop->print(dbgs()));

  // 1. create initial worklist.
  SetVector<OperationStmt *> worklist;
  auto insertUsesOf = [&worklist, state](Operation *vectorized) {
    for (auto *r : cast<OperationStmt>(vectorized)->getResults())
      for (auto &u : r->getUses()) {
        auto *stmt = cast<OperationStmt>(u.getOwner());
        // Ignore vector_transfer_write from worklist, they do not create uses.
        if (stmt->getName().getStringRef() == kVectorTransferWriteOpName ||
            state->vectorizedSet.count(stmt) > 0 ||
            state->vectorizationMap.count(stmt) > 0) {
          continue;
        }
        worklist.insert(stmt);
      }
  };
  auto getDefiningOperation = [](const MLValue *val) {
    return const_cast<MLValue *>(val)->getDefiningOperation();
  };
  using IterTy = decltype(*(state->replacementMap.begin()));
  auto getKey = [](IterTy it) { return it.first; };
  // 1.b. do it.
  using namespace functional;
  apply(insertUsesOf,
        map(getDefiningOperation, map(getKey, state->replacementMap)));

  // Note: Worklist size increases iteratively. At each round we evaluate the
  // size again. By construction, the order of elements in the worklist is
  // consistent across iterations.
  for (unsigned i = 0; i < worklist.size(); ++i) {
    auto *stmt = worklist[i];
    bool alreadyFixed = state->vectorizationMap.count(stmt) > 0;
    if (!alreadyFixed) {
      // 2. Create vectorized form of the statement.
      // Insert it just before stmt, on success register stmt as replaced.
      MLFuncBuilder b(stmt);
      std::function<Type(SSAValue *)> getVectorType =
          [state](SSAValue *v) -> VectorType {
        return VectorType::get(state->strategy->vectorSizes, v->getType());
      };
      auto types = map(getVectorType, stmt->getResults());
      std::function<SSAValue *(SSAValue *)> vectorizeOperands =
          [state](SSAValue *v) -> SSAValue * {
        return state->replacementMap.lookup(cast<MLValue>(v));
      };
      auto operands = map(vectorizeOperands, stmt->getOperands());
      // TODO(ntv): The following assumes there is always an op with a fixed
      // name works both in scalar mode and vector mode.
      // TODO(ntv): Is it worth considering an OperationStmt.clone operation
      // which changes the type so we can promote an OperationStmt with less
      // boilerplate?
      SmallString<16> name(stmt->getName().getStringRef());
      OperationState opState(b.getContext(), stmt->getLoc(), name, operands,
                             types);
      auto *vectorizedStmt = cast<OperationStmt>(b.createOperation(opState));
      assert(stmt->getNumResults() == 1);
      assert(vectorizedStmt->getNumResults() == 1);

      // 3. Replace all uses of the old statement by the new statement.
      // TODO(ntv): use implicit conversion of result to SSAValue once we have
      // an actual Op for vector_transfer.
      state->registerReplacement(cast<OperationStmt>(stmt), vectorizedStmt);
      stmt->getResult(0)->replaceAllUsesWith(vectorizedStmt->getResult(0));

      // 4. Augment the worklist with uses of the statement we just vectorized.
      // This preserves the proper order in the worklist.
      functional::apply(insertUsesOf, ArrayRef<Operation *>{vectorizedStmt});

      // 5. Check if all operands have been vectorized, if any remains it means
      // we need extra processing that we do not support atm.
      // TODO(ntv): such a non-vectorized operand should come from outside the
      // current vectorization pattern and a broadcast will be necessary.
      // Intuitively it seems it seems such a case is always a simple
      // broadcast. This is further complicated by loop-invariant scalars vs
      // scalars involving loops. This is left for future work for now.
      for (auto *operand : vectorizedStmt->getOperands()) {
        auto *def = cast<OperationStmt>(operand->getDefiningOperation());
        if (state->vectorizedSet.count(def) == 0 &&
            state->vectorizationMap.count(def) == 0) {
          LLVM_DEBUG(
              dbgs()
              << "\n[early-vect] Def needs transitive vectorization -> fail");
          LLVM_DEBUG(def->print(dbgs()));
          return true;
        }
      }
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
    MLFuncBuilder builder(loop); // builder to insert in place of loop
    DenseMap<const MLValue *, MLValue *> nomap;
    ForStmt *clonedLoop = cast<ForStmt>(builder.clone(*loop, nomap));
    auto fail = doVectorize(m, state);
    functional::ScopeGuard sg2([&fail, loop, clonedLoop]() {
      fail ? loop->erase() : clonedLoop->erase();
    });
    if (fail) {
      LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ failed root doVectorize");
      continue;
    }

    fail |= vectorizeOperations(loop, state);
    if (fail) {
      LLVM_DEBUG(
          dbgs() << "\n[early-vect]+++++ failed root vectorizeOperations");
      continue;
    }

    state->finishVectorizationPattern();
  }
  return false;
}

/// Applies vectorization to the current MLFunction by searching over a bunch of
/// predetermined patterns.
PassResult Vectorize::runOnMLFunction(MLFunction *f) {
  for (auto pat : makePatterns()) {
    LLVM_DEBUG(dbgs() << "\n******************************************");
    LLVM_DEBUG(dbgs() << "\n******************************************");
    LLVM_DEBUG(dbgs() << "\n[early-vect] new pattern on MLFunction\n");
    LLVM_DEBUG(f->print(dbgs()));
    auto matches = pat.match(f);
    Strategy strategy;
    // TODO(ntv): depending on profitability, elect to reduce the vector size.
    strategy.vectorSizes = clVirtualVectorSize;
    auto fail = analyzeProfitability(matches, 0, pat.getDepth(), &strategy);
    if (fail) {
      continue;
    }
    VectorizationState state;
    state.strategy = &strategy;
    // TODO(ntv): if pattern does not apply, report it; alter the cost/benefit.
    vectorizeRoot(matches, &state);
  }
  LLVM_DEBUG(dbgs() << "\n");
  return PassResult::Success;
}

FunctionPass *mlir::createVectorizePass() { return new Vectorize(); }

static PassRegistration<Vectorize>
    pass("vectorize",
         "Vectorize to a target independent n-D vector abstraction");
