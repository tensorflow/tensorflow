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

#include "mlir/Analysis/AffineAnalysis.h"
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
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/SuperVectorOps/SuperVectorOps.h"
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
/// use-def chains do not escape an enclosing vectorized ForInst. In other
/// words, this pass operates on a scoped program slice. Furthermore, since we
/// do not vectorize in the presence of conditionals for now, sliced chains are
/// guaranteed not to escape the innermost scope, which has to be either the top
/// Function scope of the innermost loop scope, by construction. As a
/// consequence, the implementation just starts from vector_transfer_write
/// operations and builds the slice scoped the innermost loop enclosing the
/// current vector_transfer_write. These assumptions and the implementation
/// details are subject to revision in the future.
///
/// Example
/// ========
/// In the following, the single vector_transfer_write op operates on a
/// vector<4x4x4xf32>. Let's assume the HW supports vector<4x4xf32>.
/// Materialization is achieved by instantiating each occurrence of the leading
/// dimension of vector<4x4x4xf32> into a vector<4x4xf32>.
/// The program transformation that implements this instantiation is a
/// multi-loop unroll-and-jam (it can be partial or full depending on the ratio
/// of super-vector shape to HW-vector shape).
///
/// As a simple case, the following:
/// ```mlir
///    mlfunc @materialize(%M : index, %N : index, %O : index, %P : index) {
///      %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
///      %f1 = constant splat<vector<4x4x4xf32>, 1.000000e+00> :
///      vector<4x4x4xf32> for %i0 = 0 to %M step 4 {
///        for %i1 = 0 to %N step 4 {
///          for %i2 = 0 to %O {
///            for %i3 = 0 to %P step 4 {
///              vector_transfer_write %f1, %A, %i0, %i1, %i2, %i3
///                {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d0)} :
///                 vector<4x4x4xf32>, memref<?x?x?x?xf32, 0>,
///                 index, index, index, index
///      }}}}
///      return
///    }
/// ```
///
/// is instantiated by unroll-and-jam (just unroll in this case) into:
///
/// ```mlir
///    mlfunc @materialize(%M : index, %N : index, %O : index, %P : index) {
///      %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
///      %f1 = constant splat<vector<4x4xf32>, 1.000000e+00> : vector<4x4x4xf32>
///       for %i0 = 0 to %arg0 step 4 {
///         for %i1 = 0 to %arg1 step 4 {
///           for %i2 = 0 to %arg2 {
///             for %i3 = 0 to %arg3 step 4 {
///               %1 = affine_apply (d0, d1, d2, d3) -> (d0, d1, d2, d3)
///                    (%i0, %i1, %i2, %i3)
///               vector_transfer_write f1, %0, %1#0, %1#1, %1#2, %1#3
///                 {permutation_map: (d0, d1, d2, d3) -> (d1, d0)} :
///                 vector<4x4xf32>, memref<?x?x?x?xf32>,
///                 index, index, index, index
///               %2 = affine_apply (d0, d1, d2, d3) -> (d0, d1, d2, d3 + 1)
///                    (%i0, %i1, %i2, %i3)
///               vector_transfer_write {{.*}}, %0, %2#0, %2#1, %2#2, %2#3
///                 {permutation_map: (d0, d1, d2, d3) -> (d1, d0)} :
///                 vector<4x4xf32>, memref<?x?x?x?xf32>,
///                 index, index, index, index
///               %3 = affine_apply (d0, d1, d2, d3) -> (d0, d1, d2, d3 + 2)
///                    (%i0, %i1, %i2, %i3)
///               vector_transfer_write {{.*}}, %0, %3#0, %3#1, %3#2, %3#3
///                 {permutation_map: (d0, d1, d2, d3) -> (d1, d0)} :
///                 vector<4x4xf32>, memref<?x?x?x?xf32>,
///                 index, index, index, index
///               %4 = affine_apply (d0, d1, d2, d3) -> (d0, d1, d2, d3 + 3)
///                    (%i0, %i1, %i2, %i3)
///               vector_transfer_write {{.*}}, %0, %4#0, %4#1, %4#2, %4#3
///                 {permutation_map: (d0, d1, d2, d3) -> (d1, d0)} :
///                 vector<4x4xf32>, memref<?x?x?x?xf32>,
///                 index, index, index, index
///      }}}}
///      return
///    }
/// ```

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
  DenseMap<const Value *, Value *> *substitutionsMap;
};

struct MaterializeVectorsPass : public FunctionPass {
  MaterializeVectorsPass() : FunctionPass(&MaterializeVectorsPass::passID) {}

  PassResult runOnMLFunction(Function *f) override;

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext mlContext;

  static char passID;
};

} // end anonymous namespace

char MaterializeVectorsPass::passID = 0;

/// Given a shape with sizes greater than 0 along all dimensions,
/// returns the distance, in number of elements, between a slice in a dimension
/// and the next slice in the same dimension.
///   e.g. shape[3, 4, 5] -> strides[20, 5, 1]
static SmallVector<unsigned, 8> makeStrides(ArrayRef<unsigned> shape) {
  SmallVector<unsigned, 8> tmp;
  tmp.reserve(shape.size());
  unsigned running = 1;
  for (auto rit = shape.rbegin(), reit = shape.rend(); rit != reit; ++rit) {
    assert(*rit > 0 && "size must be greater than 0 along all dimensions of "
                       "shape");
    tmp.push_back(running);
    running *= *rit;
  }
  return SmallVector<unsigned, 8>(tmp.rbegin(), tmp.rend());
}

/// Given a shape with sizes greater than 0 along all dimensions, returns the
/// delinearized components of linearIndex along shape.
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

static OperationInst *
instantiate(FuncBuilder *b, OperationInst *opInst, VectorType hwVectorType,
            DenseMap<const Value *, Value *> *substitutionsMap);

/// Not all Values belong to a program slice scoped within the immediately
/// enclosing loop.
/// One simple example is constants defined outside the innermost loop scope.
/// For such cases the substitutionsMap has no entry and we allow an additional
/// insertion.
/// For now, this is limited to ConstantOp because we do not vectorize loop
/// indices and will need to be extended in the future.
///
/// If substitution fails, returns nullptr.
static Value *substitute(Value *v, VectorType hwVectorType,
                         DenseMap<const Value *, Value *> *substitutionsMap) {
  auto it = substitutionsMap->find(v);
  if (it == substitutionsMap->end()) {
    auto *opInst = v->getDefiningInst();
    if (opInst->isa<ConstantOp>()) {
      FuncBuilder b(opInst);
      auto *inst = instantiate(&b, opInst, hwVectorType, substitutionsMap);
      auto res =
          substitutionsMap->insert(std::make_pair(v, inst->getResult(0)));
      assert(res.second && "Insertion failed");
      return res.first->second;
    }
    v->getDefiningInst()->emitError("Missing substitution");
    return nullptr;
  }
  return it->second;
}

/// Returns an AffineMap that reindexed the memRefIndices by the
/// multi-dimensional hwVectorInstance.
/// This is used by the function that materialized a vector_transfer operation
/// to use hardware vector types instead of super-vector types.
///
/// The general problem this pass solves is as follows:
/// Assume a vector_transfer operation at the super-vector granularity that has
/// `l` enclosing loops (ForInst). Assume the vector transfer operation operates
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
static SmallVector<mlir::Value *, 8>
reindexAffineIndices(FuncBuilder *b, VectorType hwVectorType,
                     ArrayRef<unsigned> hwVectorInstance,
                     ArrayRef<Value *> memrefIndices) {
  auto vectorShape = hwVectorType.getShape();
  assert(hwVectorInstance.size() >= vectorShape.size());

  unsigned numIndices = memrefIndices.size();
  auto numMemRefIndices = numIndices - hwVectorInstance.size();
  auto numSuperVectorIndices = hwVectorInstance.size() - vectorShape.size();

  SmallVector<AffineExpr, 8> affineExprs;
  // TODO(ntv): support a concrete map and composition.
  unsigned i = 0;
  // The first numMemRefIndices correspond to ForInst that have not been
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
  return SmallVector<mlir::Value *, 8>{app->getResults()};
}

/// Returns attributes with the following substitutions applied:
///   - constant splat is replaced by constant splat of `hwVectorType`.
/// TODO(ntv): add more substitutions on a per-need basis.
static SmallVector<NamedAttribute, 1>
materializeAttributes(OperationInst *opInst, VectorType hwVectorType) {
  SmallVector<NamedAttribute, 1> res;
  for (auto a : opInst->getAttrs()) {
    if (auto splat = a.second.dyn_cast<SplatElementsAttr>()) {
      auto attr = SplatElementsAttr::get(hwVectorType, splat.getValue());
      res.push_back(NamedAttribute(a.first, attr));
    } else {
      res.push_back(a);
    }
  }
  return res;
}

/// Creates an instantiated version of `opInst`.
/// Ops other than VectorTransferReadOp/VectorTransferWriteOp require no
/// affine reindexing. Just substitute their Value operands and be done. For
/// this case the actual instance is irrelevant. Just use the values in
/// substitutionsMap.
///
/// If the underlying substitution fails, this fails too and returns nullptr.
static OperationInst *
instantiate(FuncBuilder *b, OperationInst *opInst, VectorType hwVectorType,
            DenseMap<const Value *, Value *> *substitutionsMap) {
  assert(!opInst->isa<VectorTransferReadOp>() &&
         "Should call the function specialized for VectorTransferReadOp");
  assert(!opInst->isa<VectorTransferWriteOp>() &&
         "Should call the function specialized for VectorTransferWriteOp");
  bool fail = false;
  auto operands = map(
      [hwVectorType, substitutionsMap, &fail](Value *v) -> Value * {
        auto *res =
            fail ? nullptr : substitute(v, hwVectorType, substitutionsMap);
        fail |= !res;
        return res;
      },
      opInst->getOperands());
  if (fail)
    return nullptr;

  auto attrs = materializeAttributes(opInst, hwVectorType);

  OperationState state(b->getContext(), opInst->getLoc(),
                       opInst->getName().getStringRef(), operands,
                       {hwVectorType}, attrs);
  return b->createOperation(state);
}

/// Computes the permutationMap required for a VectorTransferOp from the memref
/// to the `hwVectorType`.
/// This is achieved by returning the projection of the permutationMap along the
/// dimensions of the super-vector type that remain in the hwVectorType.
/// In particular, if a dimension is fully instantiated (i.e. unrolled) then it
/// is projected out in the final result.
template <typename VectorTransferOpTy>
static AffineMap projectedPermutationMap(VectorTransferOpTy *transfer,
                                         VectorType hwVectorType) {
  static_assert(
      std::is_same<VectorTransferOpTy, VectorTransferReadOp>::value ||
          std::is_same<VectorTransferOpTy, VectorTransferWriteOp>::value,
      "Must be called on a VectorTransferOp");
  auto superVectorType = transfer->getVectorType();
  auto optionalRatio = shapeRatio(superVectorType, hwVectorType);
  assert(optionalRatio &&
         (optionalRatio->size() == superVectorType.getShape().size()) &&
         "Shape and ratio not of the same size");
  unsigned dim = 0;
  SmallVector<AffineExpr, 4> keep;
  MLIRContext *context = transfer->getInstruction()->getContext();
  functional::zipApply(
      [&dim, &keep, context](int shape, int ratio) {
        assert(shape >= ratio && "shape dim must be greater than ratio dim");
        if (shape != ratio) {
          // HW vector is not full instantiated along this dim, keep it.
          keep.push_back(getAffineDimExpr(dim, context));
        }
        ++dim;
      },
      superVectorType.getShape(), *optionalRatio);
  auto projectionMap = AffineMap::get(optionalRatio->size(), 0, keep, {});
  auto permutationMap = transfer->getPermutationMap();
  LLVM_DEBUG(projectionMap.print(dbgs() << "\nprojectionMap: "));
  LLVM_DEBUG(permutationMap.print(dbgs() << "\npermutationMap: "));
  return composeUnboundedMaps(projectionMap, permutationMap);
}

/// Creates an instantiated version of `read` for the instance of
/// `hwVectorInstance` when lowering from a super-vector type to
/// `hwVectorType`. `hwVectorInstance` represents one particular instance of
/// `hwVectorType` int the covering of the super-vector type. For a more
/// detailed description of the problem, see the description of
/// reindexAffineIndices.
static OperationInst *
instantiate(FuncBuilder *b, VectorTransferReadOp *read, VectorType hwVectorType,
            ArrayRef<unsigned> hwVectorInstance,
            DenseMap<const Value *, Value *> *substitutionsMap) {
  SmallVector<Value *, 8> indices =
      map(makePtrDynCaster<Value>(), read->getIndices());
  auto affineIndices =
      reindexAffineIndices(b, hwVectorType, hwVectorInstance, indices);
  auto cloned = b->create<VectorTransferReadOp>(
      read->getLoc(), hwVectorType, read->getMemRef(), affineIndices,
      projectedPermutationMap(read, hwVectorType), read->getPaddingValue());
  return cloned->getInstruction();
}

/// Creates an instantiated version of `write` for the instance of
/// `hwVectorInstance` when lowering from a super-vector type to
/// `hwVectorType`. `hwVectorInstance` represents one particular instance of
/// `hwVectorType` int the covering of th3e super-vector type. For a more
/// detailed description of the problem, see the description of
/// reindexAffineIndices.
static OperationInst *
instantiate(FuncBuilder *b, VectorTransferWriteOp *write,
            VectorType hwVectorType, ArrayRef<unsigned> hwVectorInstance,
            DenseMap<const Value *, Value *> *substitutionsMap) {
  SmallVector<Value *, 8> indices =
      map(makePtrDynCaster<Value>(), write->getIndices());
  auto affineIndices =
      reindexAffineIndices(b, hwVectorType, hwVectorInstance, indices);
  auto cloned = b->create<VectorTransferWriteOp>(
      write->getLoc(),
      substitute(write->getVector(), hwVectorType, substitutionsMap),
      write->getMemRef(), affineIndices,
      projectedPermutationMap(write, hwVectorType));
  return cloned->getInstruction();
}

/// Returns `true` if inst instance is properly cloned and inserted, false
/// otherwise.
/// The multi-dimensional `hwVectorInstance` belongs to the shapeRatio of
/// super-vector type to hw vector type.
/// A cloned instance of `inst` is formed as follows:
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
///
/// Returns true on failure.
static bool instantiateMaterialization(Instruction *inst,
                                       MaterializationState *state) {
  LLVM_DEBUG(dbgs() << "\ninstantiate: " << *inst);

  if (isa<ForInst>(inst))
    return inst->emitError("NYI path ForInst");

  if (isa<IfInst>(inst))
    return inst->emitError("NYI path IfInst");

  // Create a builder here for unroll-and-jam effects.
  FuncBuilder b(inst);
  auto *opInst = cast<OperationInst>(inst);
  if (auto write = opInst->dyn_cast<VectorTransferWriteOp>()) {
    instantiate(&b, write, state->hwVectorType, state->hwVectorInstance,
                state->substitutionsMap);
    return false;
  } else if (auto read = opInst->dyn_cast<VectorTransferReadOp>()) {
    auto *clone = instantiate(&b, read, state->hwVectorType,
                              state->hwVectorInstance, state->substitutionsMap);
    state->substitutionsMap->insert(
        std::make_pair(read->getResult(), clone->getResult(0)));
    return false;
  }
  // The only op with 0 results reaching this point must, by construction, be
  // VectorTransferWriteOps and have been caught above. Ops with >= 2 results
  // are not yet supported. So just support 1 result.
  if (opInst->getNumResults() != 1)
    return inst->emitError("NYI: ops with != 1 results");
  if (opInst->getResult(0)->getType() != state->superVectorType)
    return inst->emitError("Op does not return a supervector.");
  auto *clone =
      instantiate(&b, opInst, state->hwVectorType, state->substitutionsMap);
  if (!clone) {
    return true;
  }
  state->substitutionsMap->insert(
      std::make_pair(opInst->getResult(0), clone->getResult(0)));
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
/// Returns true on failure.
///
/// TODO(ntv): materialized allocs.
/// TODO(ntv): full loops + materialized allocs.
/// TODO(ntv): partial unrolling + materialized allocs.
static bool emitSlice(MaterializationState *state,
                      SetVector<Instruction *> *slice) {
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
    DenseMap<const Value *, Value *> substitutionMap;
    scopedState.substitutionsMap = &substitutionMap;
    // slice are topologically sorted, we can just clone them in order.
    for (auto *inst : *slice) {
      auto fail = instantiateMaterialization(inst, &scopedState);
      if (fail) {
        inst->emitError("Unhandled super-vector materialization failure");
        return true;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "\nMLFunction is now\n");
  LLVM_DEBUG(cast<OperationInst>((*slice)[0])->getFunction()->print(dbgs()));

  // slice are topologically sorted, we can just erase them in reverse
  // order. Reverse iterator does not just work simply with an operator*
  // dereference.
  for (int idx = slice->size() - 1; idx >= 0; --idx) {
    LLVM_DEBUG(dbgs() << "\nErase: ");
    LLVM_DEBUG((*slice)[idx]->print(dbgs()));
    (*slice)[idx]->erase();
  }
  return false;
}

/// Materializes super-vector types into concrete hw vector types as follows:
///   1. start from super-vector terminators (current vector_transfer_write
///      ops);
///   2. collect all the instructions that can be reached by transitive use-defs
///      chains;
///   3. get the superVectorType for this particular terminator and the
///      corresponding hardware vector type (for now limited to F32)
///      TODO(ntv): be more general than F32.
///   4. emit the transitive useDef set to operate on the finer-grain vector
///      types.
///
/// Notes
/// =====
/// The `slice` is sorted in topological order by construction.
/// Additionally, this set is limited to instructions in the same lexical scope
/// because we currently disallow vectorization of defs that come from another
/// scope.
static bool materialize(Function *f,
                        const SetVector<OperationInst *> &terminators,
                        MaterializationState *state) {
  DenseSet<Instruction *> seen;
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
    auto *enclosingScope = term->getParentInst();
    auto keepIfInSameScope = [enclosingScope](Instruction *inst) {
      assert(inst && "NULL inst");
      if (!enclosingScope) {
        // by construction, everyone is always under the top scope (null scope).
        return true;
      }
      return properlyDominates(*enclosingScope, *inst);
    };
    SetVector<Instruction *> slice =
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
    auto fail = emitSlice(state, &slice);
    if (fail) {
      return true;
    }
    LLVM_DEBUG(dbgs() << "\nMLFunction is now\n");
    LLVM_DEBUG(f->print(dbgs()));
  }
  return false;
}

PassResult MaterializeVectorsPass::runOnMLFunction(Function *f) {
  using matcher::Op;
  LLVM_DEBUG(dbgs() << "\nMaterializeVectors on Function\n");
  LLVM_DEBUG(f->print(dbgs()));

  MaterializationState state;
  // Get the hardware vector type.
  // TODO(ntv): get elemental type from super-vector type rather than force f32.
  auto subVectorType =
      VectorType::get(state.hwVectorSize, Type::getF32(f->getContext()));

  // Capture terminators; i.e. vector_transfer_write ops involving a strict
  // super-vector of subVectorType.
  auto filter = [subVectorType](const Instruction &inst) {
    const auto &opInst = cast<OperationInst>(inst);
    if (!opInst.isa<VectorTransferWriteOp>()) {
      return false;
    }
    return matcher::operatesOnStrictSuperVectors(opInst, subVectorType);
  };
  auto pat = Op(filter);
  auto matches = pat.match(f);
  SetVector<OperationInst *> terminators;
  for (auto m : matches) {
    terminators.insert(cast<OperationInst>(m.first));
  }

  auto fail = materialize(f, terminators, &state);
  return fail ? PassResult::Failure : PassResult::Success;
}

FunctionPass *mlir::createMaterializeVectorsPass() {
  return new MaterializeVectorsPass();
}

static PassRegistration<MaterializeVectorsPass>
    pass("materialize-vectors", "Materializes super-vectors to vectors of the "
                                "proper size for the hardware");

#undef DEBUG_TYPE
