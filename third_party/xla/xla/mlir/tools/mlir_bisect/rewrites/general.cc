/* Copyright 2023 The OpenXLA Authors. All Rights Reserved.

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

#include <cassert>
#include <cstdint>
#include <functional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_bisect/bisect_lib.h"
#include "xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"

namespace mlir {
namespace bisect {
namespace {

bool IsTerminator(Operation* op) {
  return op->hasTrait<OpTrait::IsTerminator>();
}

bool IsTopLevelOp(Operation* op) {
  return !op->getBlock()->back().mightHaveTrait<OpTrait::IsTerminator>();
}

SmallVector<std::function<OwningOpRef<ModuleOp>()>> EraseOpWithoutResults(
    BisectState& state, Operation* op) {
  // Only erase ops with results if they're unused.
  if (op->getNumResults() > 0 && !op->use_empty()) {
    return {};
  }

  // Don't erase entire functions, constants, terminators.
  if (IsTopLevelOp(op) || IsTerminator(op)) {
    return {};
  }

  SmallVector<std::function<OwningOpRef<ModuleOp>()>> ret;
  ret.push_back([op]() {
    auto [module, cloned_op] = CloneModuleFor(op);
    cloned_op->erase();
    return std::move(module);
  });
  return ret;
}

llvm::SmallVector<std::function<OwningOpRef<ModuleOp>()>> ReplaceOpWithConstant(
    BisectState& state, Operation* op) {
  llvm::SmallVector<std::function<OwningOpRef<ModuleOp>()>> result;
  if (op->hasTrait<OpTrait::ConstantLike>() || IsTopLevelOp(op) ||
      IsTerminator(op) || op->use_empty() || op->getNumResults() == 0) {
    return result;
  }

  auto mii = llvm::dyn_cast<MemoryEffectOpInterface>(op);
  if (mii && mii.hasEffect<MemoryEffects::Allocate>()) {
    // Don't replace allocs with constants.
    return result;
  }

  // Ops that are never executed won't be replaced here, but we have other
  // strategies that get rid of them (e.g. deleting the entire region).
  for (auto* execution : state.GetExecutions(op)) {
    assert(execution->results_size() == op->getNumResults() &&
           "unexpected number of results");

    result.push_back([execution, op]() -> OwningOpRef<ModuleOp> {
      auto [module_clone, op_clone] = CloneModuleFor(op);
      SmallVector<Value> results;
      OpBuilder b(op_clone);
      for (int64_t i = 0; i < op->getNumResults(); ++i) {
        auto type = op->getResultTypes()[i];
        auto value = *interpreter::TracedValueToValue(
            execution->results(static_cast<int>(i)));
        auto attribute = interpreter::ValueToAttribute(value, type);
        // We don't currently support tuples.
        if (attribute.size() != 1) {
          return nullptr;
        }
        op_clone->getResults()[i].replaceAllUsesWith(
            b.create<arith::ConstantOp>(
                op_clone->getLoc(), type,
                llvm::cast<TypedAttr>(attribute.front())));
      }
      return std::move(module_clone);
    });
  }
  return result;
}

llvm::SmallVector<std::function<OwningOpRef<ModuleOp>()>>
ReplaceOperandWithConstant(BisectState& state, Operation* op) {
  llvm::SmallVector<std::function<OwningOpRef<ModuleOp>()>> result;
  if (IsTopLevelOp(op) || op->getNumOperands() == 0) {
    return result;
  }

  for (auto* execution : state.GetExecutions(op)) {
    for (int64_t i = 0; i < op->getNumOperands(); ++i) {
      auto operand = op->getOperand(i);
      if (operand.getDefiningOp() &&
          operand.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {
        continue;
      }
      result.push_back([execution, i, op]() -> OwningOpRef<ModuleOp> {
        auto type = op->getOperandTypes()[i];
        auto value = *interpreter::TracedValueToValue(
            execution->args(static_cast<int>(i)));
        auto attribute = interpreter::ValueToAttribute(value, type);
        if (attribute.size() != 1) {
          return nullptr;
        }
        auto [module_clone, op_clone] = CloneModuleFor(op);
        OpBuilder b(op_clone);
        op_clone->setOperand(i, b.create<arith::ConstantOp>(
                                    op_clone->getLoc(), type,
                                    llvm::cast<TypedAttr>(attribute.front())));
        return std::move(module_clone);
      });
    }
  }
  return result;
}

// Replaces an op's result with some other value with the same type defined
// previously in the same region.
llvm::SmallVector<std::function<OwningOpRef<ModuleOp>()>> ReplaceOpWithValue(
    BisectState&, Operation* op) {
  llvm::SmallVector<std::function<OwningOpRef<ModuleOp>()>> ret;
  if (op->hasTrait<OpTrait::ConstantLike>() || IsTopLevelOp(op) ||
      IsTerminator(op)) {
    return ret;
  }

  // TODO(jreiffers): Consider bbargs.
  llvm::DenseMap<mlir::Type, llvm::SmallVector<std::pair<Operation*, int64_t>>>
      candidates_by_type;
  for (auto* pred = op->getPrevNode(); pred != nullptr;
       pred = pred->getPrevNode()) {
    for (auto [index, result] : llvm::enumerate(pred->getResults())) {
      candidates_by_type[result.getType()].emplace_back(pred, index);
    }
  }

  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    if (result.use_empty()) {
      continue;
    }

    for (auto [new_result_op, new_result_index] :
         candidates_by_type[result.getType()]) {
      ret.push_back(
          [op, i = index, j = new_result_index, result_op = new_result_op]() {
            auto [module_clone, op_clone] = CloneModuleFor(op);
            op_clone->getResults()[i].replaceAllUsesWith(
                FindInClone(result_op, module_clone.get())->getResults()[j]);
            return std::move(module_clone);
          });
    }
  }
  return ret;
}

REGISTER_MLIR_REDUCE_STRATEGY(EraseOpWithoutResults);
REGISTER_MLIR_REDUCE_STRATEGY(ReplaceOpWithConstant);
REGISTER_MLIR_REDUCE_STRATEGY(ReplaceOpWithValue);
REGISTER_MLIR_REDUCE_STRATEGY(ReplaceOperandWithConstant);

}  // namespace
}  // namespace bisect
}  // namespace mlir
