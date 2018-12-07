//===- LowerVectorTransfers.cpp - LowerVectorTransfers Pass Impl *- C++ -*-===//
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
// This file implements target-dependent lowering of vector transfer operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/MLFunctionMatcher.h"
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

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

///
/// Implements lowering of VectorTransferReadOp and VectorTransferWriteOp to a
/// proper abstraction for the hardware.
///
/// For now only a simple loop nest is emitted.
///

using llvm::dbgs;
using llvm::SetVector;

using namespace mlir;

#define DEBUG_TYPE "lower-vector-transfers"

/// Creates and returns a memoized `constant 0 : index` at the top level of the
/// each function `f` on which it is called.
static SSAValue *getZero(MLFunction *f) {
  static thread_local llvm::DenseMap<MLFunction *, SSAValue *> zeros;
  auto it = zeros.find(f);
  if (it == zeros.end()) {
    MLFuncBuilder b(f);
    b.setInsertionPointToStart(f);
    zeros.insert(
        std::make_pair(f, b.create<ConstantIndexOp>(b.getUnknownLoc(), 0)));
    it = zeros.find(f);
  }
  return it->second;
}

namespace {

struct LowerVectorTransfersPass : public FunctionPass {
  LowerVectorTransfersPass()
      : FunctionPass(&LowerVectorTransfersPass::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext mlContext;

  static char passID;
};

} // end anonymous namespace

char LowerVectorTransfersPass::passID = 0;

/// Creates the SSAValue for the sum of `a` and `b` without building a
/// full-fledged AffineMap for all indices.
///
/// Prerequisites:
///   `a` and `b` must be of IndexType.
static SSAValue *add(MLFuncBuilder *b, Location loc, SSAValue *v, SSAValue *w) {
  assert(v->getType().isa<IndexType>() && "v must be of IndexType");
  assert(w->getType().isa<IndexType>() && "w must be of IndexType");
  auto *context = b->getContext();
  auto d0 = getAffineDimExpr(0, context);
  auto d1 = getAffineDimExpr(1, context);
  auto map = AffineMap::get(2, 0, {d0 + d1}, {});
  return b->create<AffineApplyOp>(loc, map, ArrayRef<SSAValue *>{v, w})
      ->getResult(0);
}

/// Performs simple lowering into a combination of:
///   1. local memory allocation,
///   2. vector_load/vector_store from/to local buffer
///   3. perfect loop nest over scalar loads/stores from/to remote memory.
///
/// This is a simple sketch for now but does the job.
// TODO(ntv): This function has a lot of code conditioned on the template
// argument being one of the two types. Extract the common behavior into helper
// functions and detemplatizing it.
template <typename VectorTransferOpTy>
static void lowerAsLoops(VectorTransferOpTy *transfer) {
  static_assert(
      std::is_same<VectorTransferOpTy, VectorTransferReadOp>::value ||
          std::is_same<VectorTransferOpTy, VectorTransferWriteOp>::value,
      "Must be called on either VectorTransferReadOp or VectorTransferWriteOp");
  auto vectorType = transfer->getVectorType();
  auto vectorShape = vectorType.getShape();
  // tmpMemRefType is used for staging the transfer in a local scalar buffer.
  auto tmpMemRefType =
      MemRefType::get(vectorShape, vectorType.getElementType(), {}, 0);
  // vectorMemRefType is a view of tmpMemRefType as one vector.
  auto vectorMemRefType = MemRefType::get({1}, vectorType, {}, 0);

  MLFuncBuilder b(cast<OperationStmt>(transfer->getOperation()));
  auto *zero = getZero(b.getFunction());

  // 1. First allocate the local buffer in fast memory.
  // TODO(ntv): CL memory space.
  // TODO(ntv): Allocation padding for potential bank conflicts (e.g. GPUs).
  auto tmpScalarAlloc = b.create<AllocOp>(transfer->getLoc(), tmpMemRefType);
  // TODO(ntv): Proper OperationStmt.
  OperationState state(b.getContext(), transfer->getLoc(), "vector_type_cast",
                       ArrayRef<SSAValue *>{tmpScalarAlloc->getResult()},
                       ArrayRef<Type>{vectorMemRefType});
  auto vecView = b.createOperation(state);

  // 2. Store the vector to local storage in case of a vector_transfer_write.
  // TODO(ntv): This vector_store operation should be further lowered in the
  // case of GPUs.
  if (std::is_same<VectorTransferOpTy, VectorTransferWriteOp>::value) {
    b.create<StoreOp>(vecView->getLoc(), transfer->getVector(),
                      vecView->getResult(0), ArrayRef<SSAValue *>{zero});
  }

  // 3. Emit the loop-nest.
  // TODO(ntv): Invert the mapping and indexing contiguously in the remote
  // memory.
  // TODO(ntv): Handle broadcast / slice properly.
  auto permutationMap = transfer->getPermutationMap();
  SetVector<ForStmt *> loops;
  SmallVector<SSAValue *, 8> accessIndices(transfer->getIndices());
  for (auto it : llvm::enumerate(transfer->getVectorType().getShape())) {
    auto composed = composeWithUnboundedMap(
        getAffineDimExpr(it.index(), b.getContext()), permutationMap);
    auto *forStmt = b.createFor(transfer->getLoc(), 0, it.value());
    loops.insert(forStmt);
    // Setting the insertion point to the innermost loop achieves nesting.
    b.setInsertionPointToStart(loops.back());
    if (composed == getAffineConstantExpr(0, b.getContext())) {
      transfer->emitWarning(
          "Redundant copy can be implemented as a vector broadcast");
    } else {
      auto dim = composed.template cast<AffineDimExpr>();
      assert(accessIndices.size() > dim.getPosition());
      accessIndices[dim.getPosition()] =
          ::add(&b, transfer->getLoc(), accessIndices[dim.getPosition()],
                loops.back());
    }
  }

  // 4. Emit memory operations within the loops.
  // TODO(ntv): SelectOp + padding value for load out-of-bounds.
  if (std::is_same<VectorTransferOpTy, VectorTransferReadOp>::value) {
    // VectorTransferReadOp.
    // a. read scalar from remote;
    // b. write scalar to local.
    auto scalarLoad = b.create<LoadOp>(transfer->getLoc(),
                                       transfer->getMemRef(), accessIndices);
    b.create<StoreOp>(
        transfer->getLoc(), scalarLoad->getResult(),
        tmpScalarAlloc->getResult(),
        functional::map([](SSAValue *val) { return val; }, loops));
  } else {
    // VectorTransferWriteOp.
    // a. read scalar from local;
    // b. write scalar to remote.
    auto scalarLoad = b.create<LoadOp>(
        transfer->getLoc(), tmpScalarAlloc->getResult(),
        functional::map([](SSAValue *val) { return val; }, loops));
    b.create<StoreOp>(transfer->getLoc(), scalarLoad->getResult(),
                      transfer->getMemRef(), accessIndices);
  }

  // 5. Read the vector from local storage in case of a vector_transfer_read.
  // TODO(ntv): This vector_load operation should be further lowered in the
  // case of GPUs.
  if (std::is_same<VectorTransferOpTy, VectorTransferReadOp>::value) {
    b.setInsertionPoint(cast<OperationStmt>(transfer->getOperation()));
    auto *vector = b.create<LoadOp>(transfer->getLoc(), vecView->getResult(0),
                                    ArrayRef<SSAValue *>{zero})
                       ->getResult();
    transfer->getVector()->replaceAllUsesWith(vector);
  }

  // 6. Free the local buffer.
  b.setInsertionPoint(cast<OperationStmt>(transfer->getOperation()));
  b.create<DeallocOp>(transfer->getLoc(), tmpScalarAlloc);

  // 7. It is now safe to erase the statement.
  transfer->erase();
}

PassResult LowerVectorTransfersPass::runOnMLFunction(MLFunction *f) {
  using matcher::Op;
  LLVM_DEBUG(dbgs() << "\nLowerVectorTransfersPass on MLFunction\n");
  LLVM_DEBUG(f->print(dbgs()));

  // Avoid any read/write ordering considerations: do it in 2 steps.
  // 1. vector_transfer_reads;
  auto filterReads = [](const Statement &stmt) {
    const auto &opStmt = cast<OperationStmt>(stmt);
    return opStmt.isa<VectorTransferReadOp>();
  };
  for (auto m : Op(filterReads).match(f)) {
    auto read = cast<OperationStmt>(m.first)->cast<VectorTransferReadOp>();
    // TODO(ntv): Drop &* once lowerAsLoops is detemplatized.
    lowerAsLoops(&*read);
  }

  // 2. vector_transfer_writes;
  auto filterWrites = [](const Statement &stmt) {
    const auto &opStmt = cast<OperationStmt>(stmt);
    return opStmt.isa<VectorTransferWriteOp>();
  };
  for (auto m : Op(filterWrites).match(f)) {
    auto write = cast<OperationStmt>(m.first)->cast<VectorTransferWriteOp>();
    // TODO(ntv): Drop &* once lowerAsLoops is detemplatized.
    lowerAsLoops(&*write);
  }

  return PassResult::Success;
}

FunctionPass *mlir::createLowerVectorTransfersPass() {
  return new LowerVectorTransfersPass();
}

static PassRegistration<LowerVectorTransfersPass>
    pass("lower-vector-transfers", "Materializes vector transfer ops to a "
                                   "proper abstraction for the hardware");

#undef DEBUG_TYPE
