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

#include <algorithm>
#include <memory>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_HOISTLOOPINVARIANTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Hoists loop invariant ops to the outside of the loop.
// This is similar to LoopInvariantCodeMotion pass, but it also hoists resource
// related ops, e.g., ReadVariableOp, if the variable is read only.
struct HoistLoopInvariantPass
    : public impl::HoistLoopInvariantPassBase<HoistLoopInvariantPass> {
  void runOnOperation() override;
};

// Get the resource handle of the given `op`.
ResourceHandle GetResourceHandle(Operation *op) {
  llvm::StringRef device;
  if (auto attr = op->getAttrOfType<StringAttr>("device")) {
    device = attr.getValue();
  }

  llvm::StringRef container;
  if (auto attr = op->getAttrOfType<StringAttr>("container")) {
    container = attr.getValue();
  }

  llvm::StringRef shared_name;
  if (auto attr = op->getAttrOfType<StringAttr>("shared_name")) {
    shared_name = attr.getValue();
  }

  return {container, shared_name, device, /*op=*/nullptr};
}

bool ResourceOpCanBeHoisted(
    Operation *op, Region *region,
    const llvm::DenseSet<ResourceHandle> &read_only_vars) {
  // If the op is ReadVariableOp and the variable is readonly, it can be
  // hoisted.
  auto read_var_op = llvm::dyn_cast<ReadVariableOp>(op);
  if (!read_var_op) return false;
  auto var_handle_op = llvm::dyn_cast_or_null<VarHandleOp>(
      read_var_op.getResource().getDefiningOp());
  if (!var_handle_op) return false;
  return read_only_vars.contains(GetResourceHandle(var_handle_op));
}

bool ShouldMoveOutOfRegion(
    Operation *op, Region *region,
    const llvm::DenseSet<ResourceHandle> &read_only_vars) {
  return ResourceOpCanBeHoisted(op, region, read_only_vars) ||
         (isMemoryEffectFree(op) && isSpeculatable(op));
}

bool OnlyHasReadEffect(Operation *op) {
  auto interface = llvm::dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) return false;
  return interface.onlyHasEffect<MemoryEffects::Read>();
}

// Get read only variable handles in `func`.
llvm::DenseSet<ResourceHandle> GetReadOnlyVariables(func::FuncOp func) {
  llvm::DenseMap<ResourceHandle, llvm::SmallVector<Operation *, 4>> resources;

  // Get all VarHandleOps in the function.
  func.walk([&](Operation *op) {
    if (llvm::isa<VarHandleOp>(op)) {
      resources[GetResourceHandle(op)].push_back(op);
    }
  });

  // Get read only variables by checking if their users only have read effect.
  llvm::DenseSet<ResourceHandle> read_only_vars;
  for (const auto &[resource_handle, var_handle_ops] : resources) {
    if (std::all_of(var_handle_ops.begin(), var_handle_ops.end(),
                    [](Operation *op) {
                      for (auto *user : op->getUsers()) {
                        if (!OnlyHasReadEffect(user)) return false;
                      }
                      return true;
                    })) {
      read_only_vars.insert(resource_handle);
    }
  }
  return read_only_vars;
}

size_t HoistLoopInvariantCode(
    LoopLikeOpInterface loopLike,
    const llvm::DenseSet<ResourceHandle> &read_only_vars) {
  return moveLoopInvariantCode(
      &loopLike.getLoopBody(),
      [&](Value value, Region *) {
        return loopLike.isDefinedOutsideOfLoop(value);
      },
      [&](Operation *op, Region *region) {
        return ShouldMoveOutOfRegion(op, region, read_only_vars);
      },
      [&](Operation *op, Region *) { loopLike.moveOutOfLoop(op); });
}

void HoistLoopInvariantPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Skip the pass if the function inputs contain any resource.
  for (const auto &type : func.getArgumentTypes()) {
    if (getElementTypeOrSelf(type).isa<ResourceType>()) return;
  }

  llvm::DenseSet<ResourceHandle> read_only_vars = GetReadOnlyVariables(func);

  func->walk([&](LoopLikeOpInterface loopLike) {
    HoistLoopInvariantCode(loopLike, read_only_vars);
  });
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateHoistLoopInvariantPass() {
  return std::make_unique<HoistLoopInvariantPass>();
}

}  // namespace TF
}  // namespace mlir
