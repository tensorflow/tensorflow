/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <iterator>
#include <memory>
#include <utility>

#include "deallocation/IR/deallocation_ops.h"
#include "deallocation/transforms/passes.h"
#include "deallocation/utils/util.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace deallocation {
namespace {

#define GEN_PASS_DEF_DEALLOCATIONTOSCFPASS
#include "deallocation/transforms/passes.h.inc"

LogicalResult rewriteRetain(RetainOp op, PatternRewriter& rewriter) {
  assert(!op.getAllocs().empty() && "run canonicalization first");

  // `dealloc` happens to lower to free, which accepts null pointers. We still
  // guard it with if, because this behavior is not documented and it makes
  // downstream passes simpler (because they can assume we never deallocate
  // null).

  // Note: The generated code has size O(|`allocs`| * |`retains`|). If there are
  // cases where this gets too big, we should lower it to a library call
  // instead.

  auto loc = op.getLoc();

  // Get the buffers of all `alloc` values.
  SmallVector<Value> remainingBuffersAndResult;
  for (Value alloc : op.getAllocs()) {
    if (alloc.getType().isa<UnrankedMemRefType>()) {
      remainingBuffersAndResult.push_back(alloc);
    } else {
      remainingBuffersAndResult.push_back(rewriter.create<memref::CastOp>(
          loc, getUnrankedMemrefType(alloc), alloc));
    }
  }
  llvm::copy(llvm::map_range(op.getAllocs(),
                             [&](Value alloc) -> Value {
                               return rewriter.create<GetBufferOp>(
                                   loc, rewriter.getIndexType(), alloc);
                             }),
             std::back_inserter(remainingBuffersAndResult));
  remainingBuffersAndResult.push_back({});

  Value null =
      rewriter.create<NullOp>(loc, getUnrankedMemrefType(op.getAllocs()[0]));
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> results;

  size_t nAllocs = op.getAllocs().size();
  for (auto [retainedIndex, retained] : llvm::enumerate(op.getRetained())) {
    auto retainedBuffer =
        rewriter.create<GetBufferOp>(loc, rewriter.getIndexType(), retained);

    remainingBuffersAndResult.back() = null;
    for (auto allocIndex : llvm::seq<size_t>(0, nAllocs)) {
      auto isSame = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, retainedBuffer,
          remainingBuffersAndResult[nAllocs + allocIndex]);

      // If the buffers are the same, remove the alloc from consideration for
      // future `retained` values.
      SmallVector<Value> yieldedIfSame{null, zero,
                                       remainingBuffersAndResult[allocIndex]};
      SmallVector<Value> yieldedIfDifferent{
          remainingBuffersAndResult[allocIndex],
          remainingBuffersAndResult[allocIndex + nAllocs],
          remainingBuffersAndResult.back()};

      auto ifOp =
          rewriter.create<scf::IfOp>(loc, TypeRange{ValueRange{yieldedIfSame}},
                                     isSame, /*withElseRegion=*/true);
      ifOp.getThenBodyBuilder().create<scf::YieldOp>(loc, yieldedIfSame);

      // Otherwise, keep the current results.
      ifOp.getElseBodyBuilder().create<scf::YieldOp>(loc, yieldedIfDifferent);

      remainingBuffersAndResult[allocIndex] = ifOp.getResult(0);
      remainingBuffersAndResult[allocIndex + nAllocs] = ifOp.getResult(1);
      remainingBuffersAndResult.back() = ifOp.getResult(2);
    }

    results.push_back(remainingBuffersAndResult.back());
  }

  // Deallocate any remaining buffers.
  for (auto index : llvm::seq<size_t>(0, op.getAllocs().size())) {
    auto nonZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne,
        remainingBuffersAndResult[index + op.getAllocs().size()], zero);
    rewriter.create<scf::IfOp>(loc, nonZero,
                               [&](OpBuilder& thenBuilder, Location loc) {
                                 thenBuilder.create<memref::DeallocOp>(
                                     loc, remainingBuffersAndResult[index]);
                                 thenBuilder.create<scf::YieldOp>(loc);
                               });
  }

  rewriter.replaceOp(op, results);

  return success();
}

struct DeallocationToScfPass
    : public impl::DeallocationToScfPassBase<DeallocationToScfPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add(rewriteRetain);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createDeallocationToScfPass() {
  return std::make_unique<DeallocationToScfPass>();
}

}  // namespace deallocation
}  // namespace mlir
