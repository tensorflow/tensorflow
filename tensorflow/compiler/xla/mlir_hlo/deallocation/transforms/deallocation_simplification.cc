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
#include <optional>

#include "deallocation/IR/deallocation_ops.h"
#include "deallocation/transforms/passes.h"
#include "deallocation/utils/util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace deallocation {
namespace {

// Returns the value owned by the given ownership indicator. Returns null if it
// could not be determined.
Value getOwnedValue(Value v) {
  ValueRange vals;
  unsigned valueNum;
  if (auto bbarg = v.dyn_cast<BlockArgument>()) {
    vals = v.getParentBlock()->getArguments();
    valueNum = bbarg.getArgNumber();
  } else {
    vals = v.getDefiningOp()->getResults();
    valueNum = v.cast<OpResult>().getResultNumber();
  }

  int64_t num = llvm::count_if(vals.take_front(valueNum), [](Value it) {
    return it.getType().isa<OwnershipIndicatorType>();
  });

  auto memrefs = llvm::make_filter_range(
      vals, [](Value it) { return it.getType().isa<BaseMemRefType>(); });

  auto it = memrefs.begin();
  for (auto end = memrefs.end(); it != end && num > 0; ++it) {
    --num;
  }
  if (it == memrefs.end()) return {};
  return *it;
}

enum AllocNullability : uint32_t {
  UNDEFINED = 0,
  ALWAYS_NULL = 1,
  NEVER_NULL = 2,
  SOMETIMES_NULL = 3
};

AllocNullability operator|=(AllocNullability& lhs, AllocNullability rhs) {
  return lhs = static_cast<AllocNullability>(static_cast<uint32_t>(lhs) | rhs);
}

struct AllocInfo {
  AllocNullability nullability;
  // Set only if nullability is NEVER_NULL.
  Value nonNullValue;
};

// Returns the nullability of `v`. `pending` contains a set of `Values` we're
// already considering in the computation of some value's nullability. It is
// assumed that we will eventually take the maximum (logical or) of all
// nullability values in this set.
AllocInfo getAllocNullabilityImpl(Value v, llvm::DenseSet<Value>& pending) {
  if (llvm::isa_and_present<OwnOp>(v.getDefiningOp())) {
    return {NEVER_NULL, v.getDefiningOp()->getOperand(0)};
  }

  if (llvm::isa_and_present<NullOp>(v.getDefiningOp())) {
    return {ALWAYS_NULL, {}};
  }

  if (auto retain = llvm::dyn_cast_or_null<RetainOp>(v.getDefiningOp())) {
    // We start with ALWAYS_NULL because a retain without any allocs is null.
    // Also, because a retain with a non-null alloc can be null (otherwise, this
    // would have been cleaned up by `retainNoOp`).
    AllocNullability nullability = ALWAYS_NULL;
    for (auto alloc : retain.getAllocs()) {
      if (pending.insert(alloc).second) {
        // We can ignore the non-null value here, since the final outcome won't
        // be NEVER_NULL.
        nullability |= getAllocNullabilityImpl(alloc, pending).nullability;
      }
      if (nullability == SOMETIMES_NULL) break;
    }
    return {nullability, {}};
  }

  // Returns the nullability of an operand in each of the region's predecessors.
  auto getPredecessorNullability =
      [&](RegionBranchOpInterface rbi,
          std::optional<int64_t> successorRegionIndex,
          int64_t successorArgIndex) -> AllocInfo {
    AllocNullability nullability = UNDEFINED;
    for (const auto& pred : getPredecessorRegions(rbi, successorRegionIndex)) {
      Value operand = pred.getPredecessorOperand(successorArgIndex);
      // It is safe to skip values that are already being considered higher
      // up in the call stack, because we end up taking the maximum of all
      // nullability values.
      if (pending.insert(operand).second) {
        nullability |= getAllocNullabilityImpl(operand, pending).nullability;
      }
      if (nullability == SOMETIMES_NULL) break;
    }
    if (nullability == NEVER_NULL) {
      return {NEVER_NULL, getOwnedValue(v)};
    }
    return {nullability, {}};
  };

  // If `v` is a block argument, check all incoming edges.
  if (auto bbarg = v.dyn_cast<BlockArgument>()) {
    if (auto rbi = llvm::dyn_cast<RegionBranchOpInterface>(
            bbarg.getParentRegion()->getParentOp())) {
      return getPredecessorNullability(
          rbi, bbarg.getParentRegion()->getRegionNumber(),
          bbarg.getArgNumber());
    }
  }

  if (auto rbi =
          llvm::dyn_cast_or_null<RegionBranchOpInterface>(v.getDefiningOp())) {
    return getPredecessorNullability(rbi, std::nullopt,
                                     llvm::cast<OpResult>(v).getResultNumber());
  }

  // Something we don't understand.
  return {AllocNullability::SOMETIMES_NULL, {}};
}

bool allocIsNull(Value v) {
  llvm::DenseSet<Value> pendingChecks;
  return getAllocNullabilityImpl(v, pendingChecks).nullability == ALWAYS_NULL;
}

#define GEN_PASS_DEF_DEALLOCATIONSIMPLIFICATIONPASS
#include "deallocation/transforms/passes.h.inc"

struct DeallocationSimplificationPass
    : public impl::DeallocationSimplificationPassBase<
          DeallocationSimplificationPass> {
  void runOnOperation() override {
    getOperation()->walk([](RetainOp op) {
      OpBuilder b(op);
      // If all allocs are null, the result is null and there is nothing to
      // deallocate.
      if (llvm::all_of(op.getAllocs(), allocIsNull)) {
        auto null = b.create<NullOp>(op.getLoc());
        auto nulls = llvm::SmallVector<Value>(op.getNumResults(), null);
        op.replaceAllUsesWith(nulls);
        op.erase();
        return;
      }

      if (op.getRetained().empty() && op.getAllocs().size() == 1) {
        llvm::DenseSet<Value> pendingChecks;
        auto nullability =
            getAllocNullabilityImpl(op.getAllocs()[0], pendingChecks);
        if (nullability.nullability != NEVER_NULL ||
            !nullability.nonNullValue) {
          return;
        }

        b.setInsertionPoint(op);
        b.create<memref::DeallocOp>(op.getLoc(), nullability.nonNullValue);
        op.erase();
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createDeallocationSimplificationPass() {
  return std::make_unique<DeallocationSimplificationPass>();
}

}  // namespace deallocation
}  // namespace mlir
