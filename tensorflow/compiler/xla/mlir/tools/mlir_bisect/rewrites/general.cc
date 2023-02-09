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

#include <functional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/tools/mlir_bisect/bisect_lib.h"
#include "tensorflow/compiler/xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"

namespace mlir {
namespace bisect {
namespace {

bool IsTerminator(Operation* op) {
  return op->hasTrait<OpTrait::IsTerminator>();
}

bool IsTopLevelOp(Operation* op) {
  return !op->getBlock()->back().mightHaveTrait<OpTrait::IsTerminator>();
}

SmallVector<OwningOpRef<ModuleOp>> EraseOpWithoutResults(BisectState& state,
                                                         Operation* op) {
  // Only erase ops with results if they're unused.
  if (op->getNumResults() > 0 && !op->use_empty()) {
    return {};
  }

  // Don't erase entire functions, constants, terminators.
  if (IsTopLevelOp(op) || IsTerminator(op)) {
    return {};
  }

  auto [module, cloned_op] = CloneModuleFor(op);
  cloned_op->erase();
  SmallVector<OwningOpRef<ModuleOp>> ret;
  ret.push_back(std::move(module));
  return ret;
}

llvm::SmallVector<OwningOpRef<ModuleOp>> ReplaceOpWithConstant(
    BisectState& state, Operation* op) {
  llvm::SmallVector<OwningOpRef<ModuleOp>> result;
  if (op->hasTrait<OpTrait::ConstantLike>() || IsTopLevelOp(op) ||
      IsTerminator(op) || op->use_empty() || op->getNumResults() == 0) {
    return result;
  }

  // Ops that are never executed won't be replaced here, but we have other
  // strategies that get rid of them (e.g. deleting the entire region).
  for (auto* execution : state.GetExecutions(op)) {
    assert(execution->results_size() == op->getNumResults() &&
           "unexpected number of results");

    auto [module_clone, op_clone] = CloneModuleFor(op);
    SmallVector<Value> results;
    OpBuilder b(op_clone);
    bool all_replaced = true;
    for (int64_t i = 0; i < op->getNumResults(); ++i) {
      auto type = op->getResultTypes()[i];
      auto value = *interpreter::TracedValueToValue(
          execution->results(static_cast<int>(i)));
      auto attribute = interpreter::ValueToAttribute(value, type);
      if (attribute.size() == 1) {
        op_clone->getResults()[i].replaceAllUsesWith(
            b.create<arith::ConstantOp>(op_clone->getLoc(), attribute.front(),
                                        type));
      } else {
        // We don't currently support tuples.
        all_replaced = false;
      }
    }
    if (all_replaced) {
      result.push_back(std::move(module_clone));
    }
  }
  return result;
}

llvm::SmallVector<OwningOpRef<ModuleOp>> ReplaceOperandWithConstant(
    BisectState& state, Operation* op) {
  llvm::SmallVector<OwningOpRef<ModuleOp>> result;
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
      auto type = op->getOperandTypes()[i];
      auto value = *interpreter::TracedValueToValue(
          execution->args(static_cast<int>(i)));
      auto attribute = interpreter::ValueToAttribute(value, type);
      if (attribute.size() == 1) {
        auto [module_clone, op_clone] = CloneModuleFor(op);
        OpBuilder b(op_clone);
        op_clone->setOperand(
            i, b.create<arith::ConstantOp>(op_clone->getLoc(),
                                           attribute.front(), type));
        result.push_back(std::move(module_clone));
      }
    }
  }
  return result;
}

// Replaces an op's result with some other value with the same type defined
// previously in the same region.
llvm::SmallVector<OwningOpRef<ModuleOp>> ReplaceOpWithValue(BisectState&,
                                                            Operation* op) {
  llvm::SmallVector<OwningOpRef<ModuleOp>> ret;
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
      auto [module_clone, op_clone] = CloneModuleFor(op);
      op_clone->getResults()[index].replaceAllUsesWith(
          FindInClone(new_result_op, module_clone.get())
              ->getResults()[new_result_index]);
      ret.push_back(std::move(module_clone));
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
