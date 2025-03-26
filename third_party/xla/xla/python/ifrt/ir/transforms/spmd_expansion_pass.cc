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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla::ifrt {

namespace {

#define GEN_PASS_DEF_SPMDEXPANSIONPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

// Returns FuncOps called by `op`.
std::optional<llvm::SmallVector<mlir::func::FuncOp>> MaybeFindFunctions(
    mlir::Operation* op) {
  mlir::SymbolTableCollection symbol_table;
  auto uses = mlir::SymbolTable::getSymbolUses(op);
  if (!uses.has_value()) {
    return std::nullopt;
  }

  llvm::SmallVector<mlir::func::FuncOp> called_funcs;
  for (const mlir::SymbolTable::SymbolUse& use : *uses) {
    // Since not all call-like ops implement `CallOpInterface` (e.g.,
    // `tf.BatchFunction`), we rely on symbol references to detect call ops.
    mlir::func::FuncOp called_func =
        symbol_table.lookupNearestSymbolFrom<mlir::func::FuncOp>(
            use.getUser(), use.getSymbolRef());
    if (called_func != nullptr) {
      called_funcs.push_back(called_func);
    }
  }
  if (called_funcs.empty()) {
    return std::nullopt;
  }
  return called_funcs;
}

// A general iterator that visits a FuncOp's body in topological order. Note
// that this does not visit the given FuncOp itself. Function ops are visited
// exactly once if functions are used in multiple call sites.
class TopologicalIterator {
 public:
  explicit TopologicalIterator(mlir::func::FuncOp main_func);

  // Returns whether there is any further ops to visit.
  bool hasNext();

  // Returns the next op to visit in the topological ordering. Returns
  // a nullptr if there is no next op to visit.
  mlir::Operation* next();

 private:
  // Stack to keep track of ops to visit.
  llvm::SmallVector<mlir::Operation*, 4> ops_to_visit_;

  // Keep track of functions we are walking. This is needed to avoid recursive
  // function calls.
  llvm::SmallDenseSet<mlir::StringRef, 4> funcs_visited_in_call_stack_;

  // Keep track of all visit functions. This is to guarantee that
  // functions are visited exactly once if functions are used at multiple
  // call sites.
  llvm::SmallDenseSet<mlir::StringRef, 4> funcs_visited_;
};

TopologicalIterator::TopologicalIterator(mlir::func::FuncOp main_func)
    : ops_to_visit_{&main_func.front().front()} {
  funcs_visited_.insert(main_func.getName());
  funcs_visited_in_call_stack_.insert(main_func.getName());
}

mlir::Operation* TopologicalIterator::next() {
  if (!hasNext()) return nullptr;

  auto* op = ops_to_visit_.pop_back_val();
  auto* next_op = op->getNextNode();
  if (next_op) ops_to_visit_.push_back(next_op);

  // If this is a function call op, push the first op of the function body so
  // that the function body is converted before the call site.
  std::optional<llvm::SmallVector<mlir::func::FuncOp>> funcs =
      MaybeFindFunctions(op);
  if (funcs.has_value()) {
    for (auto& func : *funcs) {
      mlir::StringRef func_name = func.getName();
      if (funcs_visited_.contains(func_name)) {
        return next();
      }
      ops_to_visit_.push_back(&(func.front().front()));
      funcs_visited_.insert(func_name);
    }
  }

  // If we have reached the end of a function body, remove the function from
  // our active set.
  if (!next_op && !funcs_visited_in_call_stack_.empty())
    if (auto func = op->getParentOfType<mlir::func::FuncOp>())
      funcs_visited_in_call_stack_.erase(func.getName());

  // If the op contains regions, push the first op of the region to stack.
  for (auto& region : op->getRegions()) {
    ops_to_visit_.push_back(&region.front().front());
  }

  return op;
}

bool TopologicalIterator::hasNext() { return !ops_to_visit_.empty(); }

// Updates `function` input signature operand at `argument_index` with
// `new_shape`.
void UpdateFunctionInputShape(const int argument_index,
                              mlir::RankedTensorType new_arg_type,
                              mlir::func::FuncOp function) {
  mlir::FunctionType func_type = function.getFunctionType();
  auto input_types = llvm::to_vector<8>(func_type.getInputs());
  input_types[argument_index] = new_arg_type;
  auto new_func_type = mlir::FunctionType::get(
      function.getContext(), input_types, func_type.getResults());
  function.setType(new_func_type);
  function.getBody().getArgument(argument_index).setType(new_arg_type);
}

// Updates argument shapes of `function` based on `ifrt.sharding` attribute.
mlir::LogicalResult UpdateFunctionArgsUsingSharding(
    mlir::func::FuncOp function) {
  // TODO(b/261623129): need to support resource types for sub-functions as they
  // can have resource type as input.
  for (int i = 0; i < function.getNumArguments(); ++i) {
    auto arg_sharding_attr =
        function.getArgAttrOfType<IfrtShardingAttrInterface>(
            i, kIfrtShardingAttrName);
    if (arg_sharding_attr == nullptr) {
      return function.emitOpError() << "requires `" << kIfrtShardingAttrName
                                    << "` attribute on arg " << i;
    }

    auto value = function.getFunctionType().getInput(i);

    mlir::RankedTensorType ranked_type =
        mlir::dyn_cast<mlir::RankedTensorType>(value);
    if (ranked_type == nullptr) {
      return function.emitOpError()
             << "requires `mlir::RankedTensorType` for arg " << i;
    }

    llvm::ArrayRef<int64_t> arg_shape = ranked_type.getShape();
    absl::StatusOr<llvm::SmallVector<int64_t>> arg_local_shape =
        arg_sharding_attr.LocalShapeFromGlobalShape(arg_shape);
    if (!arg_local_shape.ok()) {
      return function.emitOpError() << arg_local_shape.status().message();
    }
    mlir::RankedTensorType new_arg_type = mlir::RankedTensorType::get(
        *arg_local_shape, ranked_type.getElementType());
    UpdateFunctionInputShape(i, new_arg_type, function);
  }
  return mlir::success();
}

class SpmdExpansionPass
    : public impl::SpmdExpansionPassBase<SpmdExpansionPass> {
 public:
  using impl::SpmdExpansionPassBase<SpmdExpansionPass>::SpmdExpansionPassBase;

  mlir::LogicalResult initialize(mlir::MLIRContext* context) override {
    return mlir::success();
  }

  void runOnOperation() override;

 private:
  mlir::LogicalResult spmdExpand(mlir::func::FuncOp func_op);
};

// Given SPMD expanded `function_operands` to `function`, update the function
// signature to reflect the local shape of `function_operands`.
mlir::LogicalResult UpdateFunctionWithLocalInputShapes(
    mlir::MutableArrayRef<mlir::OpOperand> function_operands,
    mlir::func::FuncOp function) {
  for (auto& operand : function_operands) {
    const int index = operand.getOperandNumber();
    auto arg_type =
        mlir::dyn_cast<mlir::RankedTensorType>(operand.get().getType());
    if (!arg_type) continue;

    llvm::ArrayRef<int64_t> arg_local_shape = arg_type.getShape();
    mlir::RankedTensorType new_arg_type =
        mlir::RankedTensorType::get(arg_local_shape, arg_type.getElementType());
    UpdateFunctionInputShape(index, new_arg_type, function);
  }
  return mlir::success();
}

mlir::LogicalResult SpmdExpansionPass::spmdExpand(mlir::func::FuncOp func_op) {
  if (mlir::failed(UpdateFunctionArgsUsingSharding(func_op))) {
    return mlir::failure();
  }

  TopologicalIterator iterator(func_op);
  while (iterator.hasNext()) {
    mlir::Operation* op = iterator.next();
    std::optional<llvm::SmallVector<mlir::func::FuncOp>> funcs =
        MaybeFindFunctions(op);
    if (funcs.has_value()) {
      for (auto& func : *funcs) {
        if (mlir::failed(
                UpdateFunctionWithLocalInputShapes(op->getOpOperands(), func)))
          return mlir::failure();
      }
    }

    if (auto spmd_interface_op =
            llvm::dyn_cast<xla::ifrt::IfrtSpmdExpandable>(op)) {
      // Name the following variable with suffix because we need a second named
      // variable for the unwrapped value.
      auto expanded_op_or = spmd_interface_op.SpmdExpand();
      if (mlir::failed(expanded_op_or) || *expanded_op_or == nullptr) {
        // Sometimes the op may been erased but the expanded op will be set.
        // In this case we should emit the error on the expanded op.
        mlir::Operation* emit_op = op;
        if (mlir::succeeded(expanded_op_or) && *expanded_op_or != nullptr) {
          emit_op = *expanded_op_or;
        }
        return emit_op->emitError("Error while computing SPMD expansion");
      }
    } else {
      return op->emitOpError(
          "has not implemented the `IfrtSpmdExpandable` interface.");
    }
  }
  return mlir::success();
}

void SpmdExpansionPass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();
  // Skip single-device case.
  auto num_devices =
      module_op->getAttrOfType<mlir::IntegerAttr>(kIfrtNumDevicesAttrName);
  if (num_devices == nullptr) {
    module_op->emitOpError()
        << "`" << module_op.getName()->str() << "` requires `"
        << kIfrtNumDevicesAttrName << "` attribute.";
    return signalPassFailure();
  }
  if (num_devices.getInt() == 1) {
    return;
  }

  // Find the entry function to start SPMD expansion.
  auto entry_function_attr =
      module_op->getAttrOfType<mlir::StringAttr>(kIfrtEntryFunctionAttrName);
  std::string entry_function_name =
      (entry_function_attr == nullptr) ? "main" : entry_function_attr.str();
  auto entry_function =
      module_op.lookupSymbol<mlir::func::FuncOp>(entry_function_name);
  if (entry_function == nullptr) {
    module_op->emitOpError()
        << "cannot find entry function `" << entry_function_name << "`";
    return signalPassFailure();
  }

  if (mlir::failed(spmdExpand(entry_function))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateSpmdExpansionPass() {
  return std::make_unique<SpmdExpansionPass>();
}

}  // namespace xla::ifrt
