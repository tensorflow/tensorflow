/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_SPMD_EXPANDERS_TERMINATOR_IFRT_SPMD_EXPANDER_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_SPMD_EXPANDERS_TERMINATOR_IFRT_SPMD_EXPANDER_H_

#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/sharding_param.h"

namespace xla::ifrt {

// SPMD expander for `mlir::func::ReturnOp`-like operations.
template <typename OpT>
class TerminatorIfrtSpmdExpander
    : public xla::ifrt::IfrtSpmdExpandable::ExternalModel<
          TerminatorIfrtSpmdExpander<OpT>, OpT> {
 public:
  mlir::FailureOr<mlir::Operation*> SpmdExpand(mlir::Operation* op) const;

  mlir::FailureOr<llvm::DenseMap<int, ShardingParam>> ComputeShardingForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, ShardingParam>& input_shardings) const;

  mlir::FailureOr<llvm::DenseMap<int, ShardingParam>> ComputeShardingBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, ShardingParam>& output_shardings) const;
};

template <typename OpT>
mlir::FailureOr<mlir::Operation*> TerminatorIfrtSpmdExpander<OpT>::SpmdExpand(
    mlir::Operation* op) const {
  mlir::Operation* parent_op = op->getBlock()->getParentOp();
  if (!parent_op) return mlir::success();

  auto output_types = llvm::to_vector<8>(op->getOperandTypes());
  if (auto function = llvm::dyn_cast<mlir::func::FuncOp>(parent_op)) {
    // Update function output type to have local shape.
    mlir::FunctionType new_func_type = mlir::FunctionType::get(
        function.getContext(), function.getFunctionType().getInputs(),
        output_types);
    function.setType(new_func_type);

    // Update function callsite operations to reflect local output shapes.
    std::optional<mlir::SymbolTable::UseRange> function_uses =
        mlir::SymbolTable::getSymbolUses(
            function,
            &(function->getParentOfType<mlir::ModuleOp>().getBodyRegion()));
    if (!function_uses) return mlir::success();
    for (const mlir::SymbolTable::SymbolUse& function_use : *function_uses) {
      mlir::Operation* callsite_op = function_use.getUser();
      if (!callsite_op) continue;

      for (const auto& output_type_and_index : llvm::enumerate(output_types)) {
        int index = output_type_and_index.index();
        const mlir::Type& type = output_type_and_index.value();
        callsite_op->getResult(index).setType(type);
      }
    }
  } else {
    for (const auto& output_type_and_index : llvm::enumerate(output_types)) {
      int index = output_type_and_index.index();
      const mlir::Type& type = output_type_and_index.value();
      parent_op->getResult(index).setType(type);
    }
  }
  return op;
}

template <typename OpT>
mlir::FailureOr<llvm::DenseMap<int, ShardingParam>>
TerminatorIfrtSpmdExpander<OpT>::ComputeShardingForward(
    mlir::Operation* op,
    const llvm::DenseMap<int, ShardingParam>& input_shardings) const {
  // TODO(b/261623129): implement this method when sharding propagation pass is
  // implemented.
  op->emitOpError("Interface method `ComputeShardingForward` not implemented.");
  return mlir::failure();
}

template <typename OpT>
mlir::FailureOr<llvm::DenseMap<int, ShardingParam>>
TerminatorIfrtSpmdExpander<OpT>::ComputeShardingBackward(
    mlir::Operation* op,
    const llvm::DenseMap<int, ShardingParam>& output_shardings) const {
  // TODO(b/261623129): implement this method when sharding propagation pass is
  // implemented.
  op->emitOpError(
      "Interface method `ComputeShardingBackward` not implemented.");
  return mlir::failure();
}

}  // namespace xla::ifrt

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_SPMD_EXPANDERS_TERMINATOR_IFRT_SPMD_EXPANDER_H_
