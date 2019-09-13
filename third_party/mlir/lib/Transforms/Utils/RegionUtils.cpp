//===- RegionUtils.cpp - Region-related transformation utilities ----------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallSet.h"

using namespace mlir;

void mlir::replaceAllUsesInRegionWith(Value *orig, Value *replacement,
                                      Region &region) {
  for (IROperand &use : llvm::make_early_inc_range(orig->getUses())) {
    if (region.isAncestor(use.getOwner()->getParentRegion()))
      use.set(replacement);
  }
}

void mlir::visitUsedValuesDefinedAbove(
    Region &region, Region &limit,
    llvm::function_ref<void(OpOperand *)> callback) {
  assert(limit.isAncestor(&region) &&
         "expected isolation limit to be an ancestor of the given region");

  // Collect proper ancestors of `limit` upfront to avoid traversing the region
  // tree for every value.
  llvm::SmallPtrSet<Region *, 4> properAncestors;
  for (auto *reg = limit.getParentRegion(); reg != nullptr;
       reg = reg->getParentRegion()) {
    properAncestors.insert(reg);
  }

  region.walk([callback, &properAncestors](Operation *op) {
    for (OpOperand &operand : op->getOpOperands())
      // Callback on values defined in a proper ancestor of region.
      if (properAncestors.count(operand.get()->getParentRegion()))
        callback(&operand);
  });
}

void mlir::visitUsedValuesDefinedAbove(
    llvm::MutableArrayRef<Region> regions,
    llvm::function_ref<void(OpOperand *)> callback) {
  for (Region &region : regions)
    visitUsedValuesDefinedAbove(region, region, callback);
}

void mlir::getUsedValuesDefinedAbove(Region &region, Region &limit,
                                     llvm::SetVector<Value *> &values) {
  visitUsedValuesDefinedAbove(region, limit, [&](OpOperand *operand) {
    values.insert(operand->get());
  });
}

void mlir::getUsedValuesDefinedAbove(llvm::MutableArrayRef<Region> regions,
                                     llvm::SetVector<Value *> &values) {
  for (Region &region : regions)
    getUsedValuesDefinedAbove(region, region, values);
}
