//===- VectorToLoops.cpp - Conversion from Vector to mix of Loops and Std -===//
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

#include <type_traits>

#include "mlir/Conversion/VectorToLoops/ConvertVectorToLoops.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using vector::TransferReadOp;
using vector::TransferWriteOp;

namespace {

using vector_type_cast = edsc::intrinsics::ValueBuilder<vector::TypeCastOp>;

/// Implements lowering of TransferReadOp and TransferWriteOp to a
/// proper abstraction for the hardware.
///
/// For now, we only emit a simple loop nest that performs clipped pointwise
/// copies from a remote to a locally allocated memory.
///
/// Consider the case:
///
/// ```mlir
///    // Read the slice `%A[%i0, %i1:%i1+256, %i2:%i2+32]` into
///    // vector<32x256xf32> and pad with %f0 to handle the boundary case:
///    %f0 = constant 0.0f : f32
///    loop.for %i0 = 0 to %0 {
///      loop.for %i1 = 0 to %1 step %c256 {
///        loop.for %i2 = 0 to %2 step %c32 {
///          %v = vector.transfer_read %A[%i0, %i1, %i2], %f0
///               {permutation_map: (d0, d1, d2) -> (d2, d1)} :
///               memref<?x?x?xf32>, vector<32x256xf32>
///    }}}
/// ```
///
/// The rewriters construct loop and indices that access MemRef A in a pattern
/// resembling the following (while guaranteeing an always full-tile
/// abstraction):
///
/// ```mlir
///    loop.for %d2 = 0 to %c256 {
///      loop.for %d1 = 0 to %c32 {
///        %s = %A[%i0, %i1 + %d1, %i2 + %d2] : f32
///        %tmp[%d2, %d1] = %s
///      }
///    }
/// ```
///
/// In the current state, only a clipping transfer is implemented by `clip`,
/// which creates individual indexing expressions of the form:
///
/// ```mlir-dsc
///    auto condMax = i + ii < N;
///    auto max = select(condMax, i + ii, N - one)
///    auto cond = i + ii < zero;
///    select(cond, zero, max);
/// ```
///
/// In the future, clipping should not be the only way and instead we should
/// load vectors + mask them. Similarly on the write side, load/mask/store for
/// implementing RMW behavior.
///
/// Lowers TransferOp into a combination of:
///   1. local memory allocation;
///   2. perfect loop nest over:
///      a. scalar load/stores from local buffers (viewed as a scalar memref);
///      a. scalar store/load to original memref (with clipping).
///   3. vector_load/store
///   4. local memory deallocation.
/// Minor variations occur depending on whether a TransferReadOp or
/// a TransferWriteOp is rewritten.
template <typename TransferOpTy>
struct VectorTransferRewriter : public RewritePattern {
  explicit VectorTransferRewriter(MLIRContext *context)
      : RewritePattern(TransferOpTy::getOperationName(), 1, context) {}

  /// Used for staging the transfer in a local scalar buffer.
  MemRefType tmpMemRefType(TransferOpTy transfer) const {
    auto vectorType = transfer.getVectorType();
    return MemRefType::get(vectorType.getShape(), vectorType.getElementType(),
                           {}, 0);
  }

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override;
};

/// Analyzes the `transfer` to find an access dimension along the fastest remote
/// MemRef dimension. If such a dimension with coalescing properties is found,
/// `pivs` and `vectorView` are swapped so that the invocation of
/// LoopNestBuilder captures it in the innermost loop.
template <typename TransferOpTy>
void coalesceCopy(TransferOpTy transfer,
                  SmallVectorImpl<edsc::ValueHandle *> *pivs,
                  edsc::VectorView *vectorView) {
  // rank of the remote memory access, coalescing behavior occurs on the
  // innermost memory dimension.
  auto remoteRank = transfer.getMemRefType().getRank();
  // Iterate over the results expressions of the permutation map to determine
  // the loop order for creating pointwise copies between remote and local
  // memories.
  int coalescedIdx = -1;
  auto exprs = transfer.permutation_map().getResults();
  for (auto en : llvm::enumerate(exprs)) {
    auto dim = en.value().template dyn_cast<AffineDimExpr>();
    if (!dim) {
      continue;
    }
    auto memRefDim = dim.getPosition();
    if (memRefDim == remoteRank - 1) {
      // memRefDim has coalescing properties, it should be swapped in the last
      // position.
      assert(coalescedIdx == -1 && "Unexpected > 1 coalesced indices");
      coalescedIdx = en.index();
    }
  }
  if (coalescedIdx >= 0) {
    std::swap(pivs->back(), (*pivs)[coalescedIdx]);
    vectorView->swapRanges(pivs->size() - 1, coalescedIdx);
  }
}

/// Emits remote memory accesses that are clipped to the boundaries of the
/// MemRef.
template <typename TransferOpTy>
SmallVector<edsc::ValueHandle, 8> clip(TransferOpTy transfer,
                                       edsc::MemRefView &view,
                                       ArrayRef<edsc::IndexHandle> ivs) {
  using namespace mlir::edsc;
  using namespace edsc::op;
  using edsc::intrinsics::select;

  IndexHandle zero(index_t(0)), one(index_t(1));
  SmallVector<edsc::ValueHandle, 8> memRefAccess(transfer.indices());
  SmallVector<edsc::ValueHandle, 8> clippedScalarAccessExprs(
      memRefAccess.size(), edsc::IndexHandle());

  // Indices accessing to remote memory are clipped and their expressions are
  // returned in clippedScalarAccessExprs.
  for (unsigned memRefDim = 0; memRefDim < clippedScalarAccessExprs.size();
       ++memRefDim) {
    // Linear search on a small number of entries.
    int loopIndex = -1;
    auto exprs = transfer.permutation_map().getResults();
    for (auto en : llvm::enumerate(exprs)) {
      auto expr = en.value();
      auto dim = expr.template dyn_cast<AffineDimExpr>();
      // Sanity check.
      assert(
          (dim || expr.template cast<AffineConstantExpr>().getValue() == 0) &&
          "Expected dim or 0 in permutationMap");
      if (dim && memRefDim == dim.getPosition()) {
        loopIndex = en.index();
        break;
      }
    }

    // We cannot distinguish atm between unrolled dimensions that implement
    // the "always full" tile abstraction and need clipping from the other
    // ones. So we conservatively clip everything.
    auto N = view.ub(memRefDim);
    auto i = memRefAccess[memRefDim];
    if (loopIndex < 0) {
      auto N_minus_1 = N - one;
      auto select_1 = select(i < N, i, N_minus_1);
      clippedScalarAccessExprs[memRefDim] = select(i < zero, zero, select_1);
    } else {
      auto ii = ivs[loopIndex];
      auto i_plus_ii = i + ii;
      auto N_minus_1 = N - one;
      auto select_1 = select(i_plus_ii < N, i_plus_ii, N_minus_1);
      clippedScalarAccessExprs[memRefDim] =
          select(i_plus_ii < zero, zero, select_1);
    }
  }

  return clippedScalarAccessExprs;
}

/// Lowers TransferReadOp into a combination of:
///   1. local memory allocation;
///   2. perfect loop nest over:
///      a. scalar load from local buffers (viewed as a scalar memref);
///      a. scalar store to original memref (with clipping).
///   3. vector_load from local buffer (viewed as a memref<1 x vector>);
///   4. local memory deallocation.
///
/// Lowers the data transfer part of a TransferReadOp while ensuring no
/// out-of-bounds accesses are possible. Out-of-bounds behavior is handled by
/// clipping. This means that a given value in memory can be read multiple
/// times and concurrently.
///
/// Important notes about clipping and "full-tiles only" abstraction:
/// =================================================================
/// When using clipping for dealing with boundary conditions, the same edge
/// value will appear multiple times (a.k.a edge padding). This is fine if the
/// subsequent vector operations are all data-parallel but **is generally
/// incorrect** in the presence of reductions or extract operations.
///
/// More generally, clipping is a scalar abstraction that is expected to work
/// fine as a baseline for CPUs and GPUs but not for vector_load and DMAs.
/// To deal with real vector_load and DMAs, a "padded allocation + view"
/// abstraction with the ability to read out-of-memref-bounds (but still within
/// the allocated region) is necessary.
///
/// Whether using scalar loops or vector_load/DMAs to perform the transfer,
/// junk values will be materialized in the vectors and generally need to be
/// filtered out and replaced by the "neutral element". This neutral element is
/// op-dependent so, in the future, we expect to create a vector filter and
/// apply it to a splatted constant vector with the proper neutral element at
/// each ssa-use. This filtering is not necessary for pure data-parallel
/// operations.
///
/// In the case of vector_store/DMAs, Read-Modify-Write will be required, which
/// also have concurrency implications. Note that by using clipped scalar stores
/// in the presence of data-parallel only operations, we generate code that
/// writes the same value multiple time on the edge locations.
///
/// TODO(ntv): implement alternatives to clipping.
/// TODO(ntv): support non-data-parallel operations.

/// Performs the rewrite.
template <>
PatternMatchResult VectorTransferRewriter<TransferReadOp>::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  using namespace mlir::edsc;
  using namespace mlir::edsc::op;
  using namespace mlir::edsc::intrinsics;
  using IndexedValue =
      TemplatedIndexedValue<intrinsics::std_load, intrinsics::std_store>;

  TransferReadOp transfer = cast<TransferReadOp>(op);

  // 1. Setup all the captures.
  ScopedContext scope(rewriter, transfer.getLoc());
  IndexedValue remote(transfer.memref());
  MemRefView view(transfer.memref());
  VectorView vectorView(transfer.vector());
  SmallVector<IndexHandle, 8> ivs = makeIndexHandles(vectorView.rank());
  SmallVector<ValueHandle *, 8> pivs =
      makeHandlePointers(MutableArrayRef<IndexHandle>(ivs));
  coalesceCopy(transfer, &pivs, &vectorView);

  auto lbs = vectorView.getLbs();
  auto ubs = vectorView.getUbs();
  SmallVector<ValueHandle, 8> steps;
  steps.reserve(vectorView.getSteps().size());
  for (auto step : vectorView.getSteps())
    steps.push_back(constant_index(step));

  // 2. Emit alloc-copy-load-dealloc.
  ValueHandle tmp = alloc(tmpMemRefType(transfer));
  IndexedValue local(tmp);
  ValueHandle vec = vector_type_cast(tmp);
  LoopNestBuilder(pivs, lbs, ubs, steps)([&] {
    // Computes clippedScalarAccessExprs in the loop nest scope (ivs exist).
    local(ivs) = remote(clip(transfer, view, ivs));
  });
  ValueHandle vectorValue = std_load(vec);
  (dealloc(tmp)); // vexing parse

  // 3. Propagate.
  rewriter.replaceOp(op, vectorValue.getValue());
  return matchSuccess();
}

/// Lowers TransferWriteOp into a combination of:
///   1. local memory allocation;
///   2. vector_store to local buffer (viewed as a memref<1 x vector>);
///   3. perfect loop nest over:
///      a. scalar load from local buffers (viewed as a scalar memref);
///      a. scalar store to original memref (with clipping).
///   4. local memory deallocation.
///
/// More specifically, lowers the data transfer part while ensuring no
/// out-of-bounds accesses are possible. Out-of-bounds behavior is handled by
/// clipping. This means that a given value in memory can be written to multiple
/// times and concurrently.
///
/// See `Important notes about clipping and full-tiles only abstraction` in the
/// description of `readClipped` above.
///
/// TODO(ntv): implement alternatives to clipping.
/// TODO(ntv): support non-data-parallel operations.
template <>
PatternMatchResult VectorTransferRewriter<TransferWriteOp>::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  using namespace mlir::edsc;
  using namespace mlir::edsc::op;
  using namespace mlir::edsc::intrinsics;
  using IndexedValue =
      TemplatedIndexedValue<intrinsics::std_load, intrinsics::std_store>;

  TransferWriteOp transfer = cast<TransferWriteOp>(op);

  // 1. Setup all the captures.
  ScopedContext scope(rewriter, transfer.getLoc());
  IndexedValue remote(transfer.memref());
  MemRefView view(transfer.memref());
  ValueHandle vectorValue(transfer.vector());
  VectorView vectorView(transfer.vector());
  SmallVector<IndexHandle, 8> ivs = makeIndexHandles(vectorView.rank());
  SmallVector<ValueHandle *, 8> pivs =
      makeHandlePointers(MutableArrayRef<IndexHandle>(ivs));
  coalesceCopy(transfer, &pivs, &vectorView);

  auto lbs = vectorView.getLbs();
  auto ubs = vectorView.getUbs();
  SmallVector<ValueHandle, 8> steps;
  steps.reserve(vectorView.getSteps().size());
  for (auto step : vectorView.getSteps())
    steps.push_back(constant_index(step));

  // 2. Emit alloc-store-copy-dealloc.
  ValueHandle tmp = alloc(tmpMemRefType(transfer));
  IndexedValue local(tmp);
  ValueHandle vec = vector_type_cast(tmp);
  std_store(vectorValue, vec);
  LoopNestBuilder(pivs, lbs, ubs, steps)([&] {
    // Computes clippedScalarAccessExprs in the loop nest scope (ivs exist).
    remote(clip(transfer, view, ivs)) = local(ivs);
  });
  (dealloc(tmp)); // vexing parse...

  rewriter.eraseOp(op);
  return matchSuccess();
}
} // namespace

void mlir::populateVectorToAffineLoopsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<VectorTransferRewriter<vector::TransferReadOp>,
                  VectorTransferRewriter<vector::TransferWriteOp>>(context);
}
