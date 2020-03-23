/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This pass promotes resource reads in the main function to input arguments
// of the function. It also promotes resource writes in the main function to
// outputs of the main function. If a resource may be updated by the main
// function, the corresponding input and output arguments are alias.
//
// The information of variable identification and input-output alising is
// recorded as named attributes of the input arguments:
//
//  . 'tf.resource_name' matches 'shared_name' of VarHandleOp, which represents
//    the identifier of the resource corresponding to the input argument.
//
//  . 'tf.aliasing_output' is the index of the function output that is an alias
//    of the input argument. This attribute is not added if there is no output
//    alias for the input argument.
//
// Assumption of this pass:
//  . Compound resource operations have already been decomposed.
//  . Dead functions have already been removed, as resource arguments in dead
//    functions can cause the pass to fail.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

constexpr char kResourceFunctionMsg[] =
    "expects function level resource argument";
constexpr char kInvalidResourceMsg[] =
    "expects resource to be a VarHandleOp or function argument";

// Records the input argument index and the current live value for a resource
// variable.
struct ResourceInfo {
  int64_t input_index;
  Value live_value;
};

using ArgOrName = llvm::PointerUnion<BlockArgument, Attribute>;
using ResourceMap = llvm::SmallDenseMap<ArgOrName, ResourceInfo>;

LogicalResult VerifyNoPotentialNestedResourceAccesses(ModuleOp module) {
  auto result = module.walk([&](FuncOp func) -> WalkResult {
    // Skip main function as resources can be passed in as arguments.
    if (func.getName() == "main") return WalkResult::advance();

    for (auto type : func.getType().getInputs())
      if (getElementTypeOrSelf(type).isa<TF::ResourceType>())
        return func.emitError("potential nested resource accesses in function");

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

LogicalResult PromoteResourcesToArguments(FuncOp function) {
  Block& block = function.front();

  auto return_op = llvm::dyn_cast_or_null<ReturnOp>(block.getTerminator());
  if (!return_op)
    return function.emitError(
        "expects 'main' function to have a MLIR ReturnOp");

  ResourceMap resource_map;
  auto argument_types = llvm::to_vector<4>(function.getType().getInputs());

  // Loop through the resource arguments in the function and store a mapping
  // from that argument to its index and itself as the current live value.
  for (BlockArgument& func_arg : function.getArguments()) {
    auto resource_type =
        getElementTypeOrSelf(func_arg.getType()).dyn_cast<TF::ResourceType>();
    if (!resource_type) continue;
    if (resource_type.getSubtypes().size() != 1)
      return function.emitError()
             << "expects resource type of argument " << func_arg.getArgNumber()
             << " to have one subtype, got " << resource_type;

    for (auto* user : func_arg.getUsers())
      if (!llvm::isa<TF::ReadVariableOp>(user) &&
          !llvm::isa<TF::AssignVariableOp>(user))
        return function.emitError()
               << "expects users of resource argument "
               << func_arg.getArgNumber()
               << " to be 'tf.ReadVariableOp' or 'tf.AssignVariableOp'";

    Type arg_type = resource_type.getSubtypes().front();
    func_arg.setType(arg_type);
    resource_map[func_arg] = {func_arg.getArgNumber(), func_arg};
    argument_types[func_arg.getArgNumber()] = arg_type;
  }

  // Loop through the VarHandleOp in the function. When the first VarHandleOp
  // for a resource variable is encountered, create a new function argument and
  // add an entry to the resource_map to record the information.
  for (auto var_handle_op : block.getOps<TF::VarHandleOp>()) {
    if (resource_map.count(var_handle_op.shared_nameAttr())) continue;

    auto resource_type =
        getElementTypeOrSelf(var_handle_op.getType()).cast<TF::ResourceType>();
    if (resource_type.getSubtypes().size() != 1)
      return var_handle_op.emitOpError()
             << "expects resource type to have one subtype, got "
             << resource_type;

    Type arg_type = resource_type.getSubtypes().front();
    BlockArgument arg = block.addArgument(arg_type);
    resource_map[var_handle_op.shared_nameAttr()] = {
        static_cast<int64_t>(argument_types.size()), arg};
    argument_types.push_back(arg_type);
  }

  if (resource_map.empty()) return success();

  // We initially assign the argument for a resource as the live value for the
  // resource. We then walk through the operations in the function in their
  // lexical order, to update the live value for the resource when we see a
  // store to the resource and replace reads of the resource with uses of its
  // live value.
  for (Operation& op : llvm::make_early_inc_range(block)) {
    if (auto read_op = llvm::dyn_cast<TF::ReadVariableOp>(&op)) {
      if (auto func_arg = read_op.resource().dyn_cast<BlockArgument>()) {
        if (func_arg.getOwner() != &block)
          return read_op.emitOpError(kResourceFunctionMsg);

        read_op.value().replaceAllUsesWith(resource_map[func_arg].live_value);
      } else if (auto var_handle_op = llvm::dyn_cast<TF::VarHandleOp>(
                     read_op.resource().getDefiningOp())) {
        read_op.value().replaceAllUsesWith(
            resource_map[var_handle_op.shared_nameAttr()].live_value);
      } else {
        return read_op.emitOpError(kInvalidResourceMsg);
      }

      read_op.erase();
    } else if (auto write_op = llvm::dyn_cast<TF::AssignVariableOp>(&op)) {
      if (auto func_arg = write_op.resource().dyn_cast<BlockArgument>()) {
        if (func_arg.getOwner() != &block)
          return write_op.emitOpError(kResourceFunctionMsg);

        resource_map[func_arg].live_value = write_op.value();
      } else if (auto var_handle_op = llvm::dyn_cast<TF::VarHandleOp>(
                     write_op.resource().getDefiningOp())) {
        resource_map[var_handle_op.shared_nameAttr()].live_value =
            write_op.value();
      } else {
        return read_op.emitOpError(kInvalidResourceMsg);
      }

      write_op.erase();
    }
  }

  const int64_t num_results_before = function.getNumResults();
  auto return_operands = llvm::to_vector<4>(return_op.getOperands());
  return_operands.reserve(num_results_before + resource_map.size());
  auto result_types = llvm::to_vector<4>(return_op.getOperandTypes());
  result_types.reserve(num_results_before + resource_map.size());
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> input_output_alias;
  input_output_alias.reserve(resource_map.size());

  // Collect new return values and mapping from resource input index to output
  // alias. If the last live value is itself (argument), then that live value
  // will not be returned as the resource is unmodified.
  for (auto& resource : resource_map) {
    int64_t input_index = resource.getSecond().input_index;
    Value live_value = resource.getSecond().live_value;
    auto live_arg = live_value.dyn_cast<BlockArgument>();
    if (live_arg && live_arg.getOwner() == &block &&
        live_arg.getArgNumber() == input_index)
      continue;

    return_operands.push_back(live_value);
    result_types.push_back(live_value.getType());
    input_output_alias.push_back(
        {input_index, num_results_before + input_output_alias.size()});
  }

  // Erase all VarHandleOp.
  for (Operation& op : llvm::make_early_inc_range(function.front())) {
    auto var_handle_op = llvm::dyn_cast<TF::VarHandleOp>(op);
    if (!var_handle_op) continue;
    if (!var_handle_op.use_empty()) {
      // SmallSet will use a vector when there is only one element and use
      // std::set when there are more than one elements. This ensures that
      // the operations in the error message are ordered.
      llvm::SmallSet<std::string, 2> unique_operations;
      llvm::for_each(
          var_handle_op.getOperation()->getUsers(), [&](Operation* user) {
            unique_operations.insert(user->getName().getStringRef().str());
          });

      return var_handle_op.emitOpError(
                 "expects no uses but used by operations: ")
             << llvm::join(unique_operations.begin(), unique_operations.end(),
                           ", ");
    }

    op.erase();
  }

  // Rewrite return if more results need to be returned by the function.
  OpBuilder builder(return_op);
  if (!input_output_alias.empty()) {
    builder.create<ReturnOp>(return_op.getLoc(), return_operands);
    return_op.erase();
  }

  // Update function argument and result types with new resource subtypes.
  function.setType(builder.getFunctionType(argument_types, result_types));

  // Add resource_name attribute to the input argument for the resources.
  for (auto& resource : resource_map) {
    if (auto attr = resource.getFirst().dyn_cast<Attribute>()) {
      function.setArgAttr(resource.getSecond().input_index, "tf.resource_name",
                          attr);
    }
  }

  // Add aliasing_output attribute to the input argument for the resources that
  // are updated by the function.
  for (auto& input_output : input_output_alias)
    function.setArgAttr(input_output.first, "tf.aliasing_output",
                        builder.getI64IntegerAttr(input_output.second));

  return success();
}

class PromoteResourcesToArgsPass
    : public ModulePass<PromoteResourcesToArgsPass> {
 public:
  void runOnModule() override;
};

void PromoteResourcesToArgsPass::runOnModule() {
  ModuleOp module = getModule();
  FuncOp main_func = module.lookupSymbol<FuncOp>("main");
  if (!main_func) return;

  // This routine should only be called when control flow operations are still
  // represented with TF IfOp and WhileOp operations. In this case, there should
  // be only one basic blocks in the MLIR representation.
  if (!has_single_element(main_func.getBlocks())) {
    main_func.emitError() << "expects 'main' function to have 1 block, got "
                          << main_func.getBlocks().size();
    return signalPassFailure();
  }

  if (failed(ResourceLiftingForFunctionalControlFlow(main_func)) ||
      failed(VerifyNoPotentialNestedResourceAccesses(module)) ||
      failed(PromoteResourcesToArguments(main_func)))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> CreatePromoteResourcesToArgsPass() {
  return std::make_unique<PromoteResourcesToArgsPass>();
}

static PassRegistration<PromoteResourcesToArgsPass> pass(
    "tf-promote-resources-to-args",
    "Promote resources reads/writes to function inputs/outputs.");

}  // namespace TF
}  // namespace mlir
