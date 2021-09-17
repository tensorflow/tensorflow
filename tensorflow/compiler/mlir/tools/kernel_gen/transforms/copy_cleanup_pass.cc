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
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

// A pass to remove memref::AllocOps and memref::CopyOps ops if the only user
// of the copy is only reading and the source is not mutated.
void RemoveCopyIfTargetOnlyRead(FuncOp func) {
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

    allocs_to_remove.push_back(op);
    copies_to_remove.push_back(copy);
  });
  for (auto it : llvm::zip(allocs_to_remove, copies_to_remove)) {
    auto alloc_op = std::get<0>(it);
    auto copy_op = std::get<1>(it);
    auto source = copy_op.getSource();
    copy_op->erase();
    alloc_op->replaceAllUsesWith(ValueRange{source});
    alloc_op->erase();
  }
}
}  // namespace

struct CopyCleanupPass : public CopyCleanupPassBase<CopyCleanupPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnFunction() override { RemoveCopyIfTargetOnlyRead(getFunction()); }
};

std::unique_ptr<FunctionPass> CreateCopyCleanupPass() {
  return std::make_unique<CopyCleanupPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
