//===- MaterializeVectors.cpp - MaterializeVectors Pass Impl ----*- C++ -*-===//
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
// This file implements target-dependent materialization of super-vectors to
// vectors of the proper size for the hardware.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/MLFunctionMatcher.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Analysis/VectorAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLValue.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

///
/// Implements target-dependent materialization of virtual super-vectors to
/// vectors of the proper size for the hardware.
///
/// While the physical vector size is target-dependent, the pass is written in
/// a target-independent way: the target vector size is specified as a parameter
/// to the pass. This pass is thus a partial lowering that opens the "greybox"
/// that is the super-vector abstraction. In particular, this pass can turn the
/// vector_transfer_read and vector_transfer_write ops in either:
///   1. a loop nest with either scalar and vector load/store instructions; or
///   2. a loop-nest with DmaStartOp / DmaWaitOp; or
///   3. a pre-existing blackbox library call that can be written manually or
///      synthesized using search and superoptimization.
/// An important feature that either of these 3 target lowering abstractions
/// must handle is the handling of "non-effecting" padding with the proper
/// neutral element in order to guarantee that all "partial tiles" are actually
/// "full tiles" in practice.
///
/// In particular this pass is a MLIR-MLIR rewriting and does not concern itself
/// with target-specific instruction-selection and register allocation. These
/// will happen downstream in LLVM.
///
/// In this sense, despite performing lowering to a target-dependent size, this
/// pass is still target-agnostic.
///
/// Implementation details
/// ======================
/// The current decisions made by the super-vectorization pass guarantee that
/// use-def chains do not escape an enclosing vectorized ForStmt. In other
/// words, this pass operates on a scoped program slice. Furthermore, since we
/// do not vectorize in the presence of conditionals for now, sliced chains are
/// guaranteed not to escape the innermost scope, which has to be either the top
/// MLFunction scope of the innermost loop scope, by construction. As a
/// consequence, the implementation just starts from vector_transfer_write
/// operations and builds the slice scoped the innermost loop enclosing the
/// current vector_transfer_write. These assumptions and the implementation
/// details are subject to revision in the future.

using llvm::dbgs;
using llvm::DenseSet;
using llvm::SetVector;

using namespace mlir;

using functional::makePtrDynCaster;
using functional::map;

static llvm::cl::list<int>
    clVectorSize("vector-size",
                 llvm::cl::desc("Specify the HW vector size for vectorization"),
                 llvm::cl::ZeroOrMore);

#define DEBUG_TYPE "materialize-vect"

namespace {
struct MaterializationState {
  /// In practice, the determination of the HW-specific vector type to use when
  /// lowering a super-vector type must be based on the elemental type. The
  /// elemental type must be retrieved from the super-vector type. In the future
  /// information about hardware vector type for a particular elemental type
  /// will be part of the contract between MLIR and the backend.
  ///
  /// For example, 8xf32 has the same size as 16xf16 but the targeted HW itself
  /// may exhibit the following property:
  /// 1. have a special unit for a 128xf16 datapath;
  /// 2. no F16 FPU support on the regular 8xf32/16xf16 vector datapath.
  ///
  /// For now, we just assume hwVectorSize has the proper information regardless
  /// of the type and we assert everything is f32.
  /// TODO(ntv): relax the assumptions on admissible element type once a
  /// contract exists.
  MaterializationState() : hwVectorSize(clVectorSize.size(), 0) {
    std::copy(clVectorSize.begin(), clVectorSize.end(), hwVectorSize.begin());
  }
  SmallVector<int, 8> hwVectorSize;
  VectorType superVectorType;
  VectorType hwVectorType;
  SmallVector<unsigned, 8> hwVectorInstance;
  DenseMap<const MLValue *, MLValue *> *substitutionsMap;
};

struct MaterializeVectors : public FunctionPass {
  MaterializeVectors() : FunctionPass(&MaterializeVectors::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext mlContext;

  static char passID;
};

} // end anonymous namespace

char MaterializeVectors::passID = 0;

// Returns the distance, in number of elements, between a slice in a dimension
// and the next slice in the same dimension.
//   e.g. shape[3, 4, 5] -> strides[20, 5, 1]
static SmallVector<unsigned, 8> makeStrides(ArrayRef<unsigned> shape) {
  SmallVector<unsigned, 8> tmp;
  tmp.reserve(shape.size());
  unsigned running = 1;
  for (auto rit = shape.rbegin(), reit = shape.rend(); rit != reit; ++rit) {
    assert(*rit > 0 && "NYI: symbolic or null shape dimension");
    tmp.push_back(running);
    running *= *rit;
  }
  return SmallVector<unsigned, 8>(tmp.rbegin(), tmp.rend());
}

// Returns the linearized expression.
static SmallVector<unsigned, 8> delinearize(unsigned linearIndex,
                                            ArrayRef<unsigned> shape) {
  SmallVector<unsigned, 8> res;
  res.reserve(shape.size());
  auto strides = makeStrides(shape);
  for (unsigned idx = 0; idx < strides.size(); ++idx) {
    assert(strides[idx] > 0);
    auto val = linearIndex / strides[idx];
    res.push_back(val);
    assert((val >= 0 && val < shape[idx]) &&
           "delinearization is out of bounds");
    linearIndex %= strides[idx];
  }
  // Sanity check.
  assert(linearIndex == 0 && "linear index constructed from shape must "
                             "have 0 remainder after delinearization");
  return res;
}

// Since this is used during a traversal of a topologically sorted set, we
// can just return the original SSAValue if we do not have a substitution.
// The topological order guarantees there will never be one.
static MLValue *
substitute(SSAValue *v,
           const DenseMap<const MLValue *, MLValue *> &substitutionsMap) {
  auto it = substitutionsMap.find(cast<MLValue>(v));
  if (it == substitutionsMap.end()) {
    return cast<MLValue>(v);
  }
  return it->second;
};

/// Returns an AffineMap that reindexed the memRefIndices by the
/// multi-dimensional hwVectorInstance.
/// This is used by the function that materialized a vector_transfer operation
/// to use hardware vector types instead of super-vector types.
///
/// The general problem this pass solves is as follows:
/// Assume a vector_transfer operation at the super-vector granularity that has
/// `l` enclosing loops (ForStmt). Assume the vector transfer operation operates
/// on a MemRef of rank `r`, a super-vector of rank `s` and a hardware vector of
/// rank `h`.
/// For the purpose of illustration assume l==4, r==3, s==2, h==1 and that the
/// super-vector is vector<3x32xf32> and the hardware vector is vector<8xf32>.
/// Assume the following MLIR snippet after super-vectorizationhas been applied:
/// ```mlir
/// for %i0 = 0 to %M {
///   for %i1 = 0 to %N step 3 {
///     for %i2 = 0 to %O {
///       for %i3 = 0 to %P step 32 {
///         %r = vector_transfer_read(%A, map(%i..)#0, map(%i..)#1, map(%i..)#2)
///                                   -> vector<3x32xf32>
///         ...
/// }}}}
/// ```
/// where map denotes an AffineMap operating on enclosing loops with properties
/// compatible for vectorization (i.e. some contiguity left unspecified here).
/// Note that the vectorized loops are %i1 and %i3.
/// This function translates the vector_transfer_read operation to multiple
/// instances of vector_transfer_read that operate on vector<8x32>.
///
/// Without loss of generality, we assume hwVectorInstance is: {2, 1}.
/// The only constraints on hwVectorInstance is they belong to:
///   [0, 2] x [0, 3], which is the span of ratio of super-vector shape to
/// hardware vector shape in our example.
///
/// This function instantiates the iteration <2, 1> of vector_transfer_read
/// into the set of operations in pseudo-MLIR:
/// ```mlir
///   map2 = (d0, d1, d2, d3) -> (d0, d1 + 2, d2, d3 + 1 * 8)
///   map3 = map o map2 // where o denotes composition
///   %r = vector_transfer_read(%A, map3(%i..)#0, map3(%i..)#1, map3(%i..)#2)
///                             -> vector<3x32xf32>
/// ```
///
/// Practical considerations
/// ========================
/// For now, `map` is assumed to be the identity map and the indices are
/// specified just as vector_transfer_read(%A, %i0, %i1, %i2, %i3). This will be
/// extended in the future once we have a proper Op for vector transfers.
/// Additionally, the example above is specified in pseudo-MLIR form; once we
/// have proper support for generic maps we can generate the code and show
/// actual MLIR.
///
/// TODO(ntv): support a concrete AffineMap and compose with it.
/// TODO(ntv): these implementation details should be captured in a
/// vectorization trait at the op level directly.
static SmallVector<SSAValue *, 8>
reindexAffineIndices(MLFuncBuilder *b, VectorType hwVectorType,
                     ArrayRef<unsigned> hwVectorInstance,
                     ArrayRef<SSAValue *> memrefIndices) {
  auto vectorShape = hwVectorType.getShape();
  assert(hwVectorInstance.size() >= vectorShape.size());

  unsigned numIndices = memrefIndices.size();
  auto numMemRefIndices = numIndices - hwVectorInstance.size();
  auto numSuperVectorIndices = hwVectorInstance.size() - vectorShape.size();

  SmallVector<AffineExpr, 8> affineExprs;
  // TODO(ntv): support a concrete map and composition.
  unsigned i = 0;
  // The first numMemRefIndices correspond to ForStmt that have not been
  // vectorized, the transformation is the identity on those.
  for (i = 0; i < numMemRefIndices; ++i) {
    auto d_i = b->getAffineDimExpr(i);
    affineExprs.push_back(d_i);
  }
  // The next numSuperVectorIndices correspond to super-vector dimensions that
  // do not have a hardware vector dimension counterpart. For those we only
  // need to increment the index by the corresponding hwVectorInstance.
  for (i = numMemRefIndices; i < numMemRefIndices + numSuperVectorIndices;
       ++i) {
    auto d_i = b->getAffineDimExpr(i);
    auto offset = hwVectorInstance[i - numMemRefIndices];
    affineExprs.push_back(d_i + offset);
  }
  // The remaining indices correspond to super-vector dimensions that
  // have a hardware vector dimension counterpart. For those we to increment the
  // index by "hwVectorInstance" multiples of the corresponding hardware
  // vector size.
  for (; i < numIndices; ++i) {
    auto d_i = b->getAffineDimExpr(i);
    auto offset = hwVectorInstance[i - numMemRefIndices];
    auto stride = vectorShape[i - numMemRefIndices - numSuperVectorIndices];
    affineExprs.push_back(d_i + offset * stride);
  }
  auto affineMap = AffineMap::get(numIndices, 0, affineExprs, {});

  // TODO(ntv): support a concrete map and composition.
  auto app = b->create<AffineApplyOp>(b->getInsertionPoint()->getLoc(),
                                      affineMap, memrefIndices);
  return SmallVector<SSAValue *, 8>{app->getResults()};
}

/// Returns attributes with the following substitutions applied:
///   - splat of `superVectorType` is replaced by splat of `hwVectorType`.
/// TODO(ntv): add more substitutions on a per-need basis.
static SmallVector<NamedAttribute, 1>
materializeAttributes(OperationStmt *opStmt, VectorType superVectorType,
                      VectorType hwVectorType) {
  SmallVector<NamedAttribute, 1> res;
  for (auto a : opStmt->getAttrs()) {
    auto splat = a.second.dyn_cast<SplatElementsAttr>();
    bool splatOfSuperVectorType = splat && (splat.getType() == superVectorType);
    if (splatOfSuperVectorType) {
      auto attr = SplatElementsAttr::get(hwVectorType, splat.getValue());
      res.push_back(NamedAttribute(a.first, attr));
    } else {
      res.push_back(a);
    }
  }
  return res;
}

/// Creates an instantiated version of `opStmt`.
/// Ops other than VectorTransferReadOp/VectorTransferWriteOp require no
/// affine reindexing. Just substitute their SSAValue* operands and be done. For
/// this case the actual instance is irrelevant. Just use the SSA values in
/// substitutionsMap.
static OperationStmt *
instantiate(MLFuncBuilder *b, OperationStmt *opStmt, VectorType superVectorType,
            VectorType hwVectorType,
            DenseMap<const MLValue *, MLValue *> *substitutionsMap) {
  assert(!opStmt->isa<VectorTransferReadOp>() &&
         "Should call the function specialized for VectorTransferReadOp");
  assert(!opStmt->isa<VectorTransferWriteOp>() &&
         "Should call the function specialized for VectorTransferWriteOp");
  auto operands =
      map([substitutionsMap](
              SSAValue *v) { return substitute(v, *substitutionsMap); },
          opStmt->getOperands());
  return b->createOperation(
      opStmt->getLoc(), opStmt->getName(), operands, {hwVectorType},
      materializeAttributes(opStmt, superVectorType, hwVectorType));
}

/// Creates an instantiated version of `read` for the instance of
/// `hwVectorInstance` when lowering from a super-vector type to
/// `hwVectorType`. `hwVectorInstance` represents one particular instance of
/// `hwVectorType` int the covering of the super-vector type. For a more
/// detailed description of the problem, see the description of
/// reindexAffineIndices.
static OperationStmt *
instantiate(MLFuncBuilder *b, VectorTransferReadOp *read,
            VectorType hwVectorType, ArrayRef<unsigned> hwVectorInstance,
            DenseMap<const MLValue *, MLValue *> *substitutionsMap) {
  SmallVector<SSAValue *, 8> indices =
      map(makePtrDynCaster<SSAValue>(), read->getIndices());
  auto affineIndices =
      reindexAffineIndices(b, hwVectorType, hwVectorInstance, indices);
  auto cloned = b->create<VectorTransferReadOp>(
      read->getLoc(), hwVectorType, read->getMemRef(), affineIndices,
      makePermutationMap(read->getMemRefType(), hwVectorType),
      read->getPaddingValue());
  return cast<OperationStmt>(cloned->getOperation());
}

/// Creates an instantiated version of `write` for the instance of
/// `hwVectorInstance` when lowering from a super-vector type to
/// `hwVectorType`. `hwVectorInstance` represents one particular instance of
/// `hwVectorType` int the covering of th3e super-vector type. For a more
/// detailed description of the problem, see the description of
/// reindexAffineIndices.
static OperationStmt *
instantiate(MLFuncBuilder *b, VectorTransferWriteOp *write,
            VectorType hwVectorType, ArrayRef<unsigned> hwVectorInstance,
            DenseMap<const MLValue *, MLValue *> *substitutionsMap) {
  SmallVector<SSAValue *, 8> indices =
      map(makePtrDynCaster<SSAValue>(), write->getIndices());
  auto affineIndices =
      reindexAffineIndices(b, hwVectorType, hwVectorInstance, indices);
  auto cloned = b->create<VectorTransferWriteOp>(
      write->getLoc(), substitute(write->getVector(), *substitutionsMap),
      write->getMemRef(), affineIndices,
      makePermutationMap(write->getMemRefType(), hwVectorType));
  return cast<OperationStmt>(cloned->getOperation());
}

/// Returns `true` if stmt instance is properly cloned and inserted, false
/// otherwise.
/// The multi-dimensional `hwVectorInstance` belongs to the shapeRatio of
/// super-vector type to hw vector type.
/// A cloned instance of `stmt` is formed as follows:
///   1. vector_transfer_read: the return `superVectorType` is replaced by
///      `hwVectorType`. Additionally, affine indices are reindexed with
///      `reindexAffineIndices` using `hwVectorInstance` and vector type
///      information;
///   2. vector_transfer_write: the `valueToStore` type is simply substituted.
///      Since we operate on a topologically sorted slice, a substitution must
///      have been registered for non-constant ops. Additionally, affine indices
///      are reindexed in the same way as for vector_transfer_read;
///   3. constant ops are splats of the super-vector type by construction.
///      They are cloned to a splat on the hw vector type with the same value;
///   4. remaining ops are cloned to version of the op that returns a hw vector
///      type, all operands are substituted according to `substitutions`. Thanks
///      to the topological order of a slice, the substitution is always
///      possible.
static bool instantiateMaterialization(Statement *stmt,
                                       MaterializationState *state) {
  LLVM_DEBUG(dbgs() << "\ninstantiate: " << *stmt);

  // Fail hard and wake up when needed.
  if (isa<ForStmt>(stmt)) {
    stmt->emitError("NYI path ForStmt");
    return true;
  }

  // Fail hard and wake up when needed.
  if (isa<IfStmt>(stmt)) {
    stmt->emitError("NYI path IfStmt");
    return true;
  }

  // Create a builder here for unroll-and-jam effects.
  MLFuncBuilder b(stmt);
  auto *opStmt = cast<OperationStmt>(stmt);
  if (auto write = opStmt->dyn_cast<VectorTransferWriteOp>()) {
    instantiate(&b, write, state->hwVectorType, state->hwVectorInstance,
                state->substitutionsMap);
    return false;
  } else if (auto read = opStmt->dyn_cast<VectorTransferReadOp>()) {
    auto *clone = instantiate(&b, read, state->hwVectorType,
                              state->hwVectorInstance, state->substitutionsMap);
    state->substitutionsMap->insert(std::make_pair(
        cast<MLValue>(read->getResult()), cast<MLValue>(clone->getResult(0))));
    return false;
  }
  // The only op with 0 results reaching this point must, by construction, be
  // VectorTransferWriteOps and have been caught above. Ops with >= 2 results
  // are not yet supported. So just support 1 result.
  if (opStmt->getNumResults() != 1) {
    stmt->emitError("NYI: ops with != 1 results");
    return true;
  }
  if (opStmt->getResult(0)->getType() != state->superVectorType) {
    stmt->emitError("Op does not return a supervector.");
    return true;
  }
  auto *clone = instantiate(&b, opStmt, state->superVectorType,
                            state->hwVectorType, state->substitutionsMap);
  state->substitutionsMap->insert(std::make_pair(
      cast<MLValue>(opStmt->getResult(0)), cast<MLValue>(clone->getResult(0))));
  return false;
}

/// Takes a slice and rewrites the operations in it so that occurrences
/// of `superVectorType` are replaced by `hwVectorType`.
///
/// Implementation
/// ==============
///   1. computes the shape ratio of super-vector to HW vector shapes. This
///      gives for each op in the slice, how many instantiations are required
///      in each dimension;
///   2. performs the concrete materialization. Note that in a first
///      implementation we use full unrolling because it pragmatically removes
///      the need to explicitly materialize an AllocOp. Thanks to the properties
///      of super-vectors, this unrolling is always possible and simple:
///      vectorizing to a super-vector abstraction already achieved the
///      equivalent of loop strip-mining + loop sinking and encoded this in the
///      vector type.
///
/// TODO(ntv): materialized allocs.
/// TODO(ntv): full loops + materialized allocs.
/// TODO(ntv): partial unrolling + materialized allocs.
static void emitSlice(MaterializationState *state,
                      SetVector<Statement *> *slice) {
  auto ratio = shapeRatio(state->superVectorType, state->hwVectorType);
  assert(ratio.hasValue() &&
         "ratio of super-vector to HW-vector shape is not integral");
  // The number of integer points in a hyperrectangular region is:
  // shape[0] * strides[0].
  auto numValueToUnroll = (*ratio)[0] * makeStrides(*ratio)[0];
  // Full unrolling to hardware vectors in a first approximation.
  for (unsigned idx = 0; idx < numValueToUnroll; ++idx) {
    // Fresh RAII instanceIndices and substitutionsMap.
    MaterializationState scopedState = *state;
    scopedState.hwVectorInstance = delinearize(idx, *ratio);
    DenseMap<const MLValue *, MLValue *> substitutionMap;
    scopedState.substitutionsMap = &substitutionMap;
    // slice are topologically sorted, we can just clone them in order.
    for (auto *stmt : *slice) {
      auto fail = instantiateMaterialization(stmt, &scopedState);
      (void)fail;
      assert(!fail && "Unhandled super-vector materialization failure");
    }
  }

  LLVM_DEBUG(dbgs() << "\nMLFunction is now\n");
  LLVM_DEBUG(
      cast<OperationStmt>((*slice)[0])->getOperationFunction()->print(dbgs()));

  // slice are topologically sorted, we can just erase them in reverse
  // order. Reverse iterator does not just work simply with an operator*
  // dereference.
  for (int idx = slice->size() - 1; idx >= 0; --idx) {
    LLVM_DEBUG(dbgs() << "\nErase: ");
    LLVM_DEBUG((*slice)[idx]->print(dbgs()));
    (*slice)[idx]->erase();
  }
}

/// Materializes super-vector types into concrete hw vector types as follows:
///   1. start from super-vector terminators (current vector_transfer_write
///      ops);
///   2. collect all the statements that can be reached by transitive use-defs
///      chains;
///   3. get the superVectorType for this particular terminator and the
///      corresponding hardware vector type (for now limited to F32)
///      TODO(ntv): be more general than F32.
///   4. emit the transitive useDef set to operate on the finer grain vector
///      types.
///
/// Notes
/// =====
/// The `slice` is sorted in topological order by construction.
/// Additionally, this set is limited to statements in the same lexical scope
/// because we currently disallow vectorization of defs that come from another
/// scope.
static void materialize(MLFunction *f,
                        const SetVector<OperationStmt *> &terminators,
                        MaterializationState *state) {
  DenseSet<Statement *> seen;
  for (auto *term : terminators) {
    // Short-circuit test, a given terminator may have been reached by some
    // other previous transitive use-def chains.
    if (seen.count(term) > 0) {
      continue;
    }

    auto terminator = term->cast<VectorTransferWriteOp>();
    LLVM_DEBUG(dbgs() << "\nFrom terminator:" << *term);

    // Get the transitive use-defs starting from terminator, limited to the
    // current enclosing scope of the terminator. See the top of the function
    // Note for the justification of this restriction.
    // TODO(ntv): relax scoping constraints.
    auto *enclosingScope = term->getParentStmt();
    auto keepIfInSameScope = [enclosingScope](Statement *stmt) {
      assert(stmt && "NULL stmt");
      if (!enclosingScope) {
        // by construction, everyone is always under the top scope (null scope).
        return true;
      }
      return properlyDominates(*enclosingScope, *stmt);
    };
    SetVector<Statement *> slice =
        getSlice(term, keepIfInSameScope, keepIfInSameScope);
    assert(!slice.empty());

    // Sanity checks: transitive slice must be completely disjoint from
    // what we have seen so far.
    LLVM_DEBUG(dbgs() << "\nTransitive use-defs:");
    for (auto *ud : slice) {
      LLVM_DEBUG(dbgs() << "\nud:" << *ud);
      assert(seen.count(ud) == 0 &&
             "Transitive use-defs not disjoint from already seen");
      seen.insert(ud);
    }

    // Emit the current slice.
    // Set scoped super-vector and corresponding hw vector types.
    state->superVectorType = terminator->getVectorType();
    assert((state->superVectorType.getElementType() ==
            Type::getF32(term->getContext())) &&
           "Only f32 supported for now");
    state->hwVectorType = VectorType::get(
        state->hwVectorSize, state->superVectorType.getElementType());
    emitSlice(state, &slice);
    LLVM_DEBUG(dbgs() << "\nMLFunction is now\n");
    LLVM_DEBUG(f->print(dbgs()));
  }
}

PassResult MaterializeVectors::runOnMLFunction(MLFunction *f) {
  using matcher::Op;
  LLVM_DEBUG(dbgs() << "\nMaterializeVectors on MLFunction\n");
  LLVM_DEBUG(f->print(dbgs()));

  MaterializationState state;
  // Get the hardware vector type.
  // TODO(ntv): get elemental type from super-vector type rather than force f32.
  auto subVectorType =
      VectorType::get(state.hwVectorSize, Type::getF32(f->getContext()));

  // Capture terminators; i.e. vector_transfer_write ops involving a strict
  // super-vector of subVectorType.
  auto filter = [subVectorType](const Statement &stmt) {
    const auto &opStmt = cast<OperationStmt>(stmt);
    if (!opStmt.isa<VectorTransferWriteOp>()) {
      return false;
    }
    return matcher::operatesOnStrictSuperVectors(opStmt, subVectorType);
  };
  auto pat = Op(filter);
  auto matches = pat.match(f);
  SetVector<OperationStmt *> terminators;
  for (auto m : matches) {
    terminators.insert(cast<OperationStmt>(m.first));
  }

  // Call materialization.
  materialize(f, terminators, &state);
  return PassResult::Success;
}

FunctionPass *mlir::createMaterializeVectors() {
  return new MaterializeVectors();
}

static PassRegistration<MaterializeVectors>
    pass("materialize-vectors", "Materializes super-vectors to vectors of the "
                                "proper size for the hardware");

#undef DEBUG_TYPE
