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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {
#define GEN_PASS_DEF_COPYCLEANUPPASS
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

// A pass to remove memref::AllocOps and memref::CopyOps ops.
//
// The idea behind this pass is to collect all patterns we are interested in in
// a single place. Eventually, this should be replaced by a generalized copy
// removal pass.

// Handles the pattern where an input operand of a linalg generic is copied
// even though the producer is not mutated.
void RemoveCopyIfTargetOnlyRead(func::FuncOp func) {
  llvm::SmallVector<memref::AllocOp, 8> allocs_to_remove;
  llvm::SmallVector<memref::CopyOp, 8> copies_to_remove;

  // Gather all allocs and copies which are only read and have an immutable
  // source.
  func->walk([&](memref::AllocOp op) {
    memref::CopyOp copy;
    MemoryEffectOpInterface reader;
    bool at_most_one_copy = true;
    bool at_most_one_read = true;
    for (auto user : op->getUsers()) {
      if (auto copy_user = dyn_cast<memref::CopyOp>(user)) {
        if (copy) {
          at_most_one_copy = false;
        } else {
          copy = copy_user;
        }
        continue;
      }
      if (auto effect_interface = cast<MemoryEffectOpInterface>(user)) {
        if (reader) {
          at_most_one_read = false;
        } else {
          reader = effect_interface;
        }
        SmallVector<MemoryEffects::EffectInstance, 2> effects;
        effect_interface.getEffectsOnValue(op.getResult(), effects);
        if (llvm::any_of(effects, [](MemoryEffects::EffectInstance it) {
              return !isa<MemoryEffects::Read>(it.getEffect());
            })) {
          at_most_one_read = false;
        }
        continue;
      }
      // We don't understand this use, be conservative.
      at_most_one_read = false;
    }
    if (!copy || !at_most_one_copy) return;
    if (!reader || !at_most_one_read) return;
    // The copy should have the alloc op as target.
    if (copy.getTarget() != op.getResult()) return;

    // The copy should be before the reading use.
    if (copy->getBlock() != reader->getBlock() ||
        !copy->isBeforeInBlock(reader)) {
      return;
    }

    // No write effects between copy and use. With aliasing information, this
    // could be made more precise but for now we have to be conservative. The
    // only thing we allow are writes to values that are allocated after the
    // copy, as the aliasing is clear in those cases.
    bool source_is_mutated = false;
    for (Operation *pos = copy->getNextNode(), *end = reader; pos != end;
         pos = pos->getNextNode()) {
      auto effect_interface = dyn_cast<MemoryEffectOpInterface>(pos);
      if (!effect_interface) {
        continue;
      }
      SmallVector<MemoryEffects::EffectInstance, 2> effects;
      effect_interface.getEffects<MemoryEffects::Write>(effects);
      for (auto effect : effects) {
        if (auto alloc = effect.getValue().getDefiningOp<memref::AllocOp>()) {
          if (alloc->getBlock() == copy->getBlock() &&
              copy->isBeforeInBlock(alloc)) {
            continue;
          }
        }
        source_is_mutated = true;
        break;
      }
    }
    if (source_is_mutated) return;

    op->replaceAllUsesWith(ValueRange{copy.getSource()});
    allocs_to_remove.push_back(op);
    copies_to_remove.push_back(copy);
  });
  llvm::for_each(allocs_to_remove, [](Operation *op) { op->erase(); });
  llvm::for_each(copies_to_remove, [](Operation *op) { op->erase(); });
}

// Handles the case where the last instructions of a function implements a copy
// back to a function argument.
void RemoveCopyIfTargetIsFunctionArg(func::FuncOp func) {
  // For now only support this on functions with a single block.
  if (!func.getBody().hasOneBlock()) return;

  llvm::SmallVector<memref::AllocOp> allocs_to_remove;
  llvm::SmallVector<memref::CopyOp> copies_to_remove;

  Block &body = func.getBody().front();
  for (auto &op : llvm::reverse(body.without_terminator())) {
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      auto block_arg = copy.getTarget().dyn_cast<BlockArgument>();
      if (!block_arg) break;
      if (!isa<func::FuncOp>(block_arg.getOwner()->getParentOp()) ||
          !block_arg.hasOneUse())
        break;
      auto alloc = copy.getSource().getDefiningOp<memref::AllocOp>();
      if (!alloc) break;
      alloc->replaceAllUsesWith(ValueRange{block_arg});
      allocs_to_remove.push_back(alloc);
      copies_to_remove.push_back(copy);
      continue;
    }
    break;
  }
  llvm::for_each(allocs_to_remove, [](Operation *op) { op->erase(); });
  llvm::for_each(copies_to_remove, [](Operation *op) { op->erase(); });
}

}  // namespace

struct CopyCleanupPass : public impl::CopyCleanupPassBase<CopyCleanupPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RemoveCopyIfTargetOnlyRead(getOperation());
    RemoveCopyIfTargetIsFunctionArg(getOperation());
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> CreateCopyCleanupPass() {
  return std::make_unique<CopyCleanupPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
