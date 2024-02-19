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

#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_dataflow.h"

#include <algorithm>
#include <vector>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#define DEBUG_TYPE "resource-dataflow"

namespace mlir {
namespace TF {

namespace {
constexpr char kCompositeDevice[] = "tf._composite_device";
}  // namespace

ResourceConstructingOps::ResourceConstructingOps(Operation *op) {
  if (op) ops.insert(op);
}

ResourceConstructingOps ResourceConstructingOps::getPessimisticValueState(
    MLIRContext *context) {
  return ResourceConstructingOps();
}
ResourceConstructingOps ResourceConstructingOps::getPessimisticValueState(
    Value value) {
  if (auto barg = value.dyn_cast<BlockArgument>()) {
    if (func::FuncOp func =
            dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp())) {
      SymbolTable symbol_table(func->getParentOfType<ModuleOp>());
      auto global_tensor = tf_saved_model::LookupBoundInputOfType<
          tf_saved_model::GlobalTensorOp>(func, barg.getArgNumber(),
                                          symbol_table);
      ResourceConstructingOps result(global_tensor);
      if (func.getArgAttr(barg.getArgNumber(), kCompositeDevice)) {
        result.is_on_composite_device = true;
      }
      return result;
    }
  } else if (auto vh = dyn_cast<TF::VarHandleOp>(value.getDefiningOp())) {
    return ResourceConstructingOps(vh);
  } else if (auto it = dyn_cast<TF::IteratorOp>(value.getDefiningOp())) {
    return ResourceConstructingOps(it);
  }
  return ResourceConstructingOps();
}

ResourceConstructingOps ResourceConstructingOps::join(
    const ResourceConstructingOps &lhs, const ResourceConstructingOps &rhs) {
  // Take union of both sets of possible GlobalTensorOp values that can be
  // referenced here.
  ResourceConstructingOps ret;
  ret.ops.insert(lhs.ops.begin(), lhs.ops.end());
  ret.ops.insert(rhs.ops.begin(), rhs.ops.end());
  ret.is_on_composite_device =
      lhs.is_on_composite_device || rhs.is_on_composite_device;
  return ret;
}

void ResourceConstructingOps::print(raw_ostream &os) const {
  llvm::interleaveComma(ops, os << "[");
  if (is_on_composite_device) {
    os << " COMPOSITE";
  }
  os << "]";
}

void ResourceDataflowAnalysis::visitOperation(Operation *op,
                                              ArrayRef<const StateT *> operands,
                                              ArrayRef<StateT *> results) {
  LLVM_DEBUG(llvm::dbgs() << "ResAn: Visiting operation: " << *op << "\n");

  if (auto cast = dyn_cast<TF::CastOp>(op)) {
    join(results[0], *operands[0]);
  } else if (auto while_op = dyn_cast<TF::WhileRegionOp>(op)) {
    for (auto &region : while_op->getRegions()) {
      for (auto [arg, value] :
           llvm::zip(region.getArguments(), while_op->getOperands())) {
        join(getLatticeElement(arg), *getLatticeElement(value));
      }
    }
  } else if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
    func::FuncOp cond = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        while_op, while_op.getCondAttr());
    func::FuncOp body = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        while_op, while_op.getBodyAttr());
    for (auto &arg : while_op->getOpOperands()) {
      BlockArgument cond_arg = cond.getArgument(arg.getOperandNumber());
      join(getLatticeElement(cond_arg), *getLatticeElement(arg.get()));
      BlockArgument body_arg = body.getArgument(arg.getOperandNumber());
      join(getLatticeElement(body_arg), *getLatticeElement(arg.get()));
    }
  } else if (auto graph = dyn_cast<tf_executor::GraphOp>(op)) {
    for (auto &arg : graph.GetFetch()->getOpOperands()) {
      if (arg.getOperandNumber() < graph.getNumResults()) {
        auto result = graph.getResult(arg.getOperandNumber());
        join(getLatticeElement(result), *getLatticeElement(arg.get()));
      }
    }
  } else if (auto island = dyn_cast<tf_executor::IslandOp>(op)) {
    for (auto &arg : island.GetYield()->getOpOperands()) {
      auto result = island.getResult(arg.getOperandNumber());
      join(getLatticeElement(result), *getLatticeElement(arg.get()));
      // getLatticeElement(arg.get())->print(llvm::errs());
    }
  } else {
    setAllToEntryStates(results);
  }
}

}  // namespace TF
}  // namespace mlir
