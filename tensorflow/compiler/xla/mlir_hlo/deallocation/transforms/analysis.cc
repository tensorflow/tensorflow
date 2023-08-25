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
#include "deallocation/transforms/analysis.h"

#include <optional>

#include "deallocation/utils/util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace deallocation {

namespace {

bool isRestrictBbArg(Value value) {
  auto bbarg = llvm::dyn_cast<BlockArgument>(value);
  auto func =
      llvm::dyn_cast<func::FuncOp>(value.getParentBlock()->getParentOp());
  if (!bbarg || !func) return false;
  auto isRestrict = func.getArgAttrOfType<BoolAttr>(bbarg.getArgNumber(),
                                                    "deallocation.restrict");
  return isRestrict && isRestrict.getValue();
}

bool isMemref(Value v) { return llvm::isa<BaseMemRefType>(v.getType()); }

}  // namespace

void DeallocationAnalysis::collectBackingMemory(
    Value source, DenseSet<Value>& visited,
    breaks_if_you_move_ops::ValueSet& results) {
  if (!isMemref(source)) return;
  if (!visited.insert(source).second) return;

  auto type = getElementTypeOrSelf(source);
  if (auto bbarg = llvm::dyn_cast<BlockArgument>(source)) {
    results.insert(source);
    if (llvm::isa<func::FuncOp>(bbarg.getParentBlock()->getParentOp())) {
      if (!isRestrictBbArg(source)) {
        // Restrict bbargs can't alias anything else.
        for (auto arg : bbarg.getParentBlock()->getArguments()) {
          if (isMemref(arg) && getElementTypeOrSelf(arg.getType()) == type) {
            results.insert(arg);
          }
        }
      }
    } else if (auto rbi = llvm::dyn_cast<RegionBranchOpInterface>(
                   bbarg.getParentRegion()->getParentOp())) {
      for (const auto& edge : getPredecessorRegions(
               rbi, bbarg.getParentRegion()->getRegionNumber())) {
        if (bbarg.getArgNumber() >= edge.successorValueIndex &&
            static_cast<size_t>(bbarg.getArgNumber() -
                                edge.successorValueIndex) <=
                edge.getPredecessorOperands().size()) {
          Value dep = edge.getPredecessorOperand(bbarg.getArgNumber());
          collectBackingMemory(dep, visited, results);
        }
      }
    }
    return;
  }

  auto result = llvm::cast<OpResult>(source);
  if (auto rbi = llvm::dyn_cast<RegionBranchOpInterface>(result.getOwner())) {
    for (const auto& edge : getPredecessorRegions(rbi, std::nullopt)) {
      collectBackingMemory(edge.getPredecessorOperand(result.getResultNumber()),
                           visited, results);
    }
  }

  if (auto mem = llvm::dyn_cast<MemoryEffectOpInterface>(result.getOwner())) {
    if (mem.getEffectOnValue<MemoryEffects::Allocate>(result).has_value()) {
      results.insert(result);
    }
  }

  for (auto operand : result.getOwner()->getOperands()) {
    if (isMemref(operand) && getElementTypeOrSelf(operand) == type) {
      collectBackingMemory(operand, visited, results);
    }
  }
}

const breaks_if_you_move_ops::ValueSet& DeallocationAnalysis::getBackingMemory(
    Value source) {
  auto it = backingMemory.find(source);
  if (it != backingMemory.end()) return it->second;

  auto& results = backingMemory[source];
  DenseSet<Value> visited;
  collectBackingMemory(source, visited, results);
  return results;
}

}  // namespace deallocation
}  // namespace mlir
