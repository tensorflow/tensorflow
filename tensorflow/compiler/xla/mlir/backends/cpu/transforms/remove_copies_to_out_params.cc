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
#include <optional>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"

namespace xla {
namespace cpu {
namespace {

using ::mlir::LogicalResult;
using ::mlir::Operation;
using ::mlir::OperationPass;
using ::mlir::PatternRewriter;
using ::mlir::RewritePatternSet;
using ::mlir::Value;

namespace memref = ::mlir::memref;
namespace func = ::mlir::func;

#define GEN_PASS_DEF_REMOVECOPIESTOOUTPARAMSPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

LogicalResult AllocRemoval(memref::CopyOp copy, PatternRewriter &rewriter) {
  Value from = copy.getSource();
  Value to = copy.getTarget();

  Operation *alloc =
      llvm::dyn_cast_or_null<memref::AllocOp>(from.getDefiningOp());
  if (!alloc) {
    return mlir::failure();
  }

  // Only match if we dealloc immediately after the copy.
  auto dealloc = llvm::dyn_cast_or_null<memref::DeallocOp>(copy->getNextNode());
  if (!dealloc || dealloc.getMemref() != from) {
    return mlir::failure();
  }

  // Only go up one level to grab the parent function; the match we're looking
  // for is at the very end of a function.
  auto func = llvm::dyn_cast_or_null<func::FuncOp>(copy->getParentOp());
  if (!func) {
    return mlir::failure();
  }

  // If the copy target is a function argument, use it directly.
  if (llvm::is_contained(func.getArguments(), to)) {
    rewriter.replaceAllUsesWith(from, to);
    rewriter.eraseOp(alloc);
    rewriter.eraseOp(dealloc);
    rewriter.eraseOp(copy);
    return mlir::success();
  }
  return mlir::failure();
}

LogicalResult AllocaRemoval(memref::CopyOp copy, PatternRewriter &rewriter) {
  Value from = copy.getSource();
  Value to = copy.getTarget();

  Operation *alloca =
      llvm::dyn_cast_or_null<memref::AllocaOp>(from.getDefiningOp());
  if (!alloca) {
    return mlir::failure();
  }

  // Only go up one level to grab the parent function; the match we're looking
  // for is at the very end of a function.
  auto func = llvm::dyn_cast_or_null<func::FuncOp>(copy->getParentOp());
  if (!func) {
    return mlir::failure();
  }

  // If the copy target is a function argument, use it directly.
  if (llvm::is_contained(func.getArguments(), to)) {
    rewriter.replaceAllUsesWith(from, to);
    rewriter.eraseOp(alloca);
    rewriter.eraseOp(copy);
    return mlir::success();
  }
  return mlir::failure();
}

class RemoveCopiesToOutParamsPass
    : public impl::RemoveCopiesToOutParamsPassBase<
          RemoveCopiesToOutParamsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(AllocRemoval);
    patterns.add(AllocaRemoval);
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveCopiesToOutParamsPass() {
  return std::make_unique<RemoveCopiesToOutParamsPass>();
}

}  // namespace cpu
}  // namespace xla
