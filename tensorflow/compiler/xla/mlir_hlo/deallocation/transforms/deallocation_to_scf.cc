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

#include <memory>
#include <utility>

#include "deallocation/IR/deallocation_ops.h"
#include "deallocation/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace deallocation {
namespace {

#define GEN_PASS_DEF_DEALLOCATIONTOSCFPASS
#include "deallocation/transforms/passes.h.inc"

LogicalResult rewriteRetain(RetainOp op, PatternRewriter& rewriter) {
  assert(!op.getAllocs().empty() && "run canonicalization first");
  assert(op.getAllocs().size() == 1 && "not supported yet");
  assert(op.getRetained().size() <= 1 && "not supported yet");

  // dealloc happens to lower to free, which accepts null pointers. We still
  // guard it with if, because this behavior is not documented.

  auto loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, rewriter);
  Value alloc = op.getAllocs().front();
  Value returnValue = {};
  auto allocBuffer = b.create<GetBufferOp>(b.getIndexType(), alloc);
  auto zero = b.create<arith::ConstantIndexOp>(0);
  OpBuilder deallocBuilder = rewriter;
  if (op.getRetained().size() == 1) {
    auto retainedBuffer =
        b.create<GetBufferOp>(b.getIndexType(), op.getRetained().front());
    auto isSameBuffer = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                allocBuffer, retainedBuffer);
    auto ifOp =
        b.create<scf::IfOp>(TypeRange{alloc.getType()}, isSameBuffer, true);
    ifOp.getThenBodyBuilder().create<scf::YieldOp>(loc, ValueRange{alloc});
    auto null = ifOp.getElseBodyBuilder().create<deallocation::NullOp>(
        loc, alloc.getType());
    ifOp.getElseBodyBuilder().create<scf::YieldOp>(
        loc, ValueRange{null.getResult()});
    deallocBuilder = ifOp.getElseBodyBuilder();
    deallocBuilder.setInsertionPoint(null);

    returnValue = ifOp.getResult(0);
  }

  Value shouldDealloc = deallocBuilder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, allocBuffer, zero);
  deallocBuilder.create<scf::IfOp>(
      loc, shouldDealloc, [&](mlir::OpBuilder& thenB, mlir::Location loc) {
        thenB.create<memref::DeallocOp>(loc, alloc);
        thenB.create<scf::YieldOp>(loc);
      });

  if (returnValue) {
    rewriter.replaceOp(op, returnValue);
  } else {
    op.erase();
  }

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
