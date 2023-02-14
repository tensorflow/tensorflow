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
  assert(op.getRetained().empty() && "not supported yet");

  if (op.getRetained().empty()) {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto zero = b.create<arith::ConstantIndexOp>(0);
    for (auto alloc : op.getAllocs()) {
      auto buffer = b.create<GetBufferOp>(b.getIndexType(), alloc);
      auto isNonNull =
          b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, buffer, zero);
      b.create<scf::IfOp>(isNonNull,
                          [&](mlir::OpBuilder& thenB, mlir::Location loc) {
                            thenB.create<memref::DeallocOp>(loc, alloc);
                            thenB.create<scf::YieldOp>(loc);
                          });
    }
    op.erase();
  }

  return failure();
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
