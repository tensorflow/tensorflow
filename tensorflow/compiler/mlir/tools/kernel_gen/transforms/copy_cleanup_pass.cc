/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Analysis/BufferViewFlowAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

// A pass to remove memref::AllocOps and memref::CopyOps ops if the only user
// of the copy is a linalg::GenericOp.
void RemoveMemrefCopy(FuncOp func) {
  llvm::SmallVector<memref::AllocOp, 8> allocs_to_remove;
  llvm::SmallVector<memref::CopyOp, 8> copies_to_remove;

  // Gather all allocs and copies which are consumed by linalg::GenericOp ops.
  func->walk([&](memref::AllocOp op) {
    memref::CopyOp copy;
    linalg::GenericOp generic;
    bool at_most_one_copy = true;
    bool at_most_one_generic = true;
    for (auto user : op->getUsers()) {
      if (auto copy_user = dyn_cast<memref::CopyOp>(user)) {
        if (copy) {
          at_most_one_copy = false;
        } else {
          copy = copy_user;
        }
      } else if (auto generic_user = dyn_cast<linalg::GenericOp>(user)) {
        if (generic) {
          at_most_one_generic = false;
        } else {
          generic = generic_user;
        }
      }
    }
    if (!copy || !at_most_one_copy) return;
    if (!generic || !at_most_one_generic) return;
    // The copy should have the alloc op as target.
    if (copy.getTarget() != op.getResult()) return;

    // The copy should be before the linalg::GenericOp.
    if (copy->getBlock() != generic->getBlock() ||
        !copy->isBeforeInBlock(generic)) {
      return;
    }
    allocs_to_remove.push_back(op);
    copies_to_remove.push_back(copy);
  });
  for (auto it : llvm::zip(allocs_to_remove, copies_to_remove)) {
    auto alloc_op = std::get<0>(it);
    auto copy_op = std::get<1>(it);
    auto source = copy_op.getSource().getDefiningOp();
    copy_op->erase();
    alloc_op->replaceAllUsesWith(source);
    alloc_op->erase();
  }
}
}  // namespace

struct CopyCleanupPass : public CopyCleanupPassBase<CopyCleanupPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, linalg::LinalgDialect>();
  }

  void runOnFunction() override { RemoveMemrefCopy(getFunction()); }
};

std::unique_ptr<FunctionPass> CreateCopyCleanupPass() {
  return std::make_unique<CopyCleanupPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
