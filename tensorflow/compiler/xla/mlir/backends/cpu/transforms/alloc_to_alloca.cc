/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_ALLOCTOALLOCAPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

// Chosen arbitrarily for now. Note: this is the number of elements, not the
// size in bytes.
constexpr int64_t kMaxAllocSizeToConvert = 256;

class AllocToAllocaPass
    : public impl::AllocToAllocaPassBase<AllocToAllocaPass> {
  void runOnOperation() override;
};

struct RewriteAllocs : OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter& rewriter) const override {
    auto parent = dyn_cast<func::FuncOp>(op->getParentOp());

    // Don't convert allocs inside loops. Allocas don't get released until the
    // function terminates, which can easily lead to stack overflows.
    if (!parent) {
      return failure();
    }

    // Only convert small allocs of statically known size.
    if (!op.getType().hasStaticShape()) {
      return failure();
    }
    int64_t size = 1;
    for (int64_t dim : op.getType().getShape()) {
      size *= dim;
    }
    if (size > kMaxAllocSizeToConvert) {
      return failure();
    }

    // When this pass runs, we can just assume all transitive uses of memref
    // type are aliases of this memref (e.g. memref.subview,
    // memref.collapse_shape, arith.select).
    // We just check if the parent's func.return takes any alias of the memref.
    llvm::SmallSet<Operation*, 8> seen{{op.getOperation()}};
    llvm::SmallVector<Operation*> worklist{op.getOperation()};
    while (!worklist.empty()) {
      auto* operation = worklist.back();
      worklist.pop_back();
      if (dyn_cast<func::ReturnOp>(operation)) {
        return failure();
      }
      for (Value result : operation->getResults()) {
        if (!result.getType().isa<MemRefType>()) {
          continue;
        }
        for (Operation* user : result.getUsers()) {
          if (seen.insert(user).second) {
            worklist.push_back(user);
          }
        }
      }
    }

    // The memref doesn't escape the function, so we can replace it.
    rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        op, op->getResultTypes(), op->getOperands(), op->getAttrs());
    return success();
  }
};

void AllocToAllocaPass::runOnOperation() {
  func::FuncOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  RewritePatternSet patterns(ctx);
  patterns.insert<RewriteAllocs>(ctx);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createAllocToAllocaPass() {
  return std::make_unique<AllocToAllocaPass>();
}

}  // namespace cpu
}  // namespace xla
