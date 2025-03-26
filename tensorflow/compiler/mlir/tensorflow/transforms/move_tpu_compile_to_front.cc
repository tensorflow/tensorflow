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

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/topological_sort.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_MOVETPUCOMPILETOFRONTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct MoveTpuCompileToFrontPass
    : public impl::MoveTpuCompileToFrontPassBase<MoveTpuCompileToFrontPass> {
  void runOnOperation() override;
};

void MarkCompilationOps(Operation* func) {
  func->walk([&](Operation* op) {
    if (llvm::isa<TF::_TPUCompileMlirOp>(op)) {
      op->setAttr("_is_compilation", UnitAttr::get(func->getContext()));
      op = op->getParentOp();
      while (op && op != func) {
        op->setAttr("_wraps_compilation", UnitAttr::get(func->getContext()));
        op = op->getParentOp();
      }
    }
  });
}

void UnmarkCompilationOps(Operation* func) {
  func->walk([&](Operation* op) {
    while (op && op != func) {
      op->removeAttr("_is_compilation");
      op->removeAttr("_wraps_compilation");
      op = op->getParentOp();
    }
  });
}

int OutsideCompilationOrdering(Operation* predecessor, Operation* op) {
  // Actual compilations go first.
  if (op->hasAttr("_is_compilation")) return 2;
  // Followed by nested ops that contain compilations.
  if (op->hasAttr("_wraps_compilation")) return 1;
  // Followed by everything else.
  return 0;
}

void MoveTpuCompileToFrontPass::runOnOperation() {
  MarkCompilationOps(getOperation());
  getOperation().walk([](Operation* op) {
    for (Region& region : op->getRegions()) {
      for (Block& block : region.getBlocks()) {
        if (block.empty()) continue;
        auto ops = SortBlockTopologically(block, OutsideCompilationOrdering);
        // Replace the block with the reordered block.
        for (Operation* o : ops) {
          o->remove();
          block.push_back(o);
        }
      }
    }
  });
  UnmarkCompilationOps(getOperation());
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateMoveTpuCompileToFrontPass() {
  return std::make_unique<MoveTpuCompileToFrontPass>();
}

}  // namespace TF
}  // namespace mlir
