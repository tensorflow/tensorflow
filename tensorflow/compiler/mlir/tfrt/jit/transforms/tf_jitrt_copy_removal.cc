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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

// -------------------------------------------------------------------------- //
// Remove redundant memref.copy operations
// -------------------------------------------------------------------------- //
struct LinalgTrivialCopyRemovalPass
    : public LinalgTrivialCopyRemovalBase<LinalgTrivialCopyRemovalPass> {
  void runOnOperation() override {
    mlir::func::FuncOp function = getOperation();

    mlir::SmallVector<mlir::Operation*> to_erase;
    function.walk([&to_erase](mlir::memref::CopyOp copy) {
      // Only match precise alloc/copy/dealloc triples.
      auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(copy->getPrevNode());
      auto dealloc =
          llvm::dyn_cast<mlir::memref::DeallocOp>(copy->getNextNode());

      if (!alloc || !dealloc) return;

      // Make sure the alloc and dealloc handle the operands of the copy.
      if (alloc.getResult() != copy.getTarget() ||
          dealloc.memref() != copy.getSource()) {
        return;
      }

      // Remember the operations to delete.
      to_erase.push_back(alloc);
      to_erase.push_back(dealloc);
      to_erase.push_back(copy);
      copy.getTarget().replaceAllUsesWith(copy.getSource());
    });

    for (auto op : to_erase) {
      op->erase();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateLinalgTrivialCopyRemovalPass() {
  return std::make_unique<LinalgTrivialCopyRemovalPass>();
}

}  // namespace tensorflow
