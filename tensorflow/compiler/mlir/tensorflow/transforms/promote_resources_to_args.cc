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

// This pass promotes resource accesses in the main function to input arguments
// and outputs of the main function.
//
// Two types of resources are supported:
// (1) A function argument of TF::ResourceType type.
// (2) A VarHandleOp in the function.
//
// After the pass,
//
//  . The function will have an input argument for each resource that is
//    already provided as an input argument or is read. The type of the input
//    argument will become the shape of the value represented by the resource.
//
//  . The function will have an output for each resource that is written. The
//    type of the output will become the shape of the resource.
//
// The information of variable identification and input-output alising is
// recorded as named attributes of the input argument or output:
//
//  . 'tf.resource_name' matches 'shared_name' of VarHandleOp, which represents
//    the identifier of the corresponding resource. This attribute is added to
//    an input argument if the initial value of the resource is read, or to the
//    output if the initial value is not read.
//
//  . 'tf.aliasing_output' is the index of the function output that is an alias
//    of the input argument. This attribute is added only to the input argument
//    when the initial value of the corresponding resource is read, and the
//    resource is written later.
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
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
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
//
// . If the input argument already exists or has been added, input_index is the
//   index of the function, and live_value_or_type tracks the live value of the
//   resource.
//
// . If the input argument has not been added in the pass, input_index is
//   kInputUnassigned, live_value_or_type represents the type of the resource.
//   (a) If this resource is read, add a new argument whose type is obtained
//       from live_value_or_type, and input_index and live_value_or_type will be
//       updated to reference the new argument.
//   (b) If this resource is written, live_value_or_type will track the new
//       value of the resource. input_index will remain to be kInputUnassigned.
struct ResourceInfo {
  static constexpr int64_t kInputUnassigned = -1;
  int64_t input_index;
  llvm::PointerUnion<Value, Type> live_value_or_type;
};

using ArgOrName = llvm::PointerUnion<BlockArgument, Attribute>;
using ResourceMap = llvm::SmallDenseMap<ArgOrName, ResourceInfo>;

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
  // for a resource variable is encountered, add an entry to the resource_map to
  // record the information. Do not add a new function argument yet.
  for (auto var_handle_op : block.getOps<TF::VarHandleOp>()) {
    if (resource_map.count(var_handle_op.shared_nameAttr())) continue;

    auto resource_type =
        getElementTypeOrSelf(var_handle_op.getType()).cast<TF::ResourceType>();
    if (resource_type.getSubtypes().size() != 1)
      return var_handle_op.emitOpError()
             << "expects resource type to have one subtype, got "
             << resource_type;

    resource_map[var_handle_op.shared_nameAttr()] = {
        ResourceInfo::kInputUnassigned, resource_type.getSubtypes().front()};
  }

  if (resource_map.empty()) return success();

  // We initially assign the argument for a resource as the live value for the
  // resource. We then walk through the operations in the function in their
  // lexical order, to update the live value for the resource when we see a
  // store to the resource and replace reads of the resource with uses of its
  // live value. For the reads, if the resource does not have a live value yet,
  // we add a new argument and use it as the live value.
  for (Operation& op : llvm::make_early_inc_range(block)) {
    if (auto read_op = llvm::dyn_cast<TF::ReadVariableOp>(&op)) {
      if (auto func_arg = read_op.resource().dyn_cast<BlockArgument>()) {
        if (func_arg.getOwner() != &block)
          return read_op.emitOpError(kResourceFunctionMsg);

        // resource_map[func_arg] is always a Value when func_arg is a
        // BlockArgument.
        read_op.value().replaceAllUsesWith(
            resource_map[func_arg].live_value_or_type.get<Value>());
      } else if (auto var_handle_op = llvm::dyn_cast<TF::VarHandleOp>(
                     read_op.resource().getDefiningOp())) {
        ResourceInfo& info = resource_map[var_handle_op.shared_nameAttr()];
        if (auto live_value = info.live_value_or_type.dyn_cast<Value>()) {
          read_op.value().replaceAllUsesWith(live_value);
        } else {
          auto arg_type = info.live_value_or_type.get<Type>();
          BlockArgument arg = block.addArgument(arg_type);
          info.input_index = argument_types.size();
          info.live_value_or_type = arg;
          argument_types.push_back(arg_type);
          read_op.value().replaceAllUsesWith(arg);
        }
      } else {
        return read_op.emitOpError(kInvalidResourceMsg);
      }

      read_op.erase();
    } else if (auto write_op = llvm::dyn_cast<TF::AssignVariableOp>(&op)) {
      if (auto func_arg = write_op.resource().dyn_cast<BlockArgument>()) {
        if (func_arg.getOwner() != &block)
          return write_op.emitOpError(kResourceFunctionMsg);

        resource_map[func_arg].live_value_or_type = write_op.value();
      } else if (auto var_handle_op = llvm::dyn_cast<TF::VarHandleOp>(
                     write_op.resource().getDefiningOp())) {
        resource_map[var_handle_op.shared_nameAttr()].live_value_or_type =
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
  llvm::SmallVector<std::pair<int64_t, Attribute>, 4> output_only_resources;
  output_only_resources.reserve(resource_map.size());
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> input_output_alias;
  input_output_alias.reserve(resource_map.size());

  // Collect new return values and either (a) output-only resource attributes
  // (if the resource is not promoted to an argument) or (b) mapping from
  // resource input index to output alias (if the resource has been promoted to
  // an argument). If the last live value is itself (argument), then that live
  // value will not be returned as the resource is unmodified.
  for (auto& resource : resource_map) {
    int64_t input_index = resource.getSecond().input_index;
    auto live_value = resource.getSecond().live_value_or_type.dyn_cast<Value>();
    if (input_index == ResourceInfo::kInputUnassigned) {
      if (!live_value) continue;

      output_only_resources.push_back(
          {return_operands.size(), resource.getFirst().dyn_cast<Attribute>()});
    } else {
      // live_value is not nullptr because any input-assigned resource has a
      // Value as live_value.
      auto live_arg = live_value.dyn_cast<BlockArgument>();
      if (live_arg && live_arg.getOwner() == &block &&
          live_arg.getArgNumber() == input_index)
        continue;

      input_output_alias.push_back({input_index, return_operands.size()});
    }
    return_operands.push_back(live_value);
    result_types.push_back(live_value.getType());
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
  if (!output_only_resources.empty() || !input_output_alias.empty()) {
    builder.create<ReturnOp>(return_op.getLoc(), return_operands);
    return_op.erase();
  }

  // Update function argument and result types with new resource subtypes.
  function.setType(builder.getFunctionType(argument_types, result_types));

  // Add resource_name attribute to the input argument for the resources.
  for (auto& resource : resource_map) {
    if (auto attr = resource.getFirst().dyn_cast<Attribute>()) {
      int64_t input_index = resource.getSecond().input_index;
      if (input_index != ResourceInfo::kInputUnassigned)
        function.setArgAttr(input_index, "tf.resource_name", attr);
    }
  }
  // Add resource_name attribute to the output for the resources.
  for (auto& resource : output_only_resources)
    function.setResultAttr(resource.first, "tf.resource_name", resource.second);

  // Add aliasing_output attribute to the input argument for the resources that
  // are updated by the function.
  for (auto& input_output : input_output_alias)
    function.setArgAttr(input_output.first, "tf.aliasing_output",
                        builder.getI64IntegerAttr(input_output.second));

  return success();
}

class PromoteResourcesToArgsPass
    : public PassWrapper<PromoteResourcesToArgsPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override;
};

void PromoteResourcesToArgsPass::runOnOperation() {
  ModuleOp module = getOperation();
  FuncOp main_func = module.lookupSymbol<FuncOp>("main");
  if (!main_func) return;

  // This routine should only be called when control flow operations are still
  // represented with TF IfOp and WhileOp operations. In this case, there should
  // be only one basic blocks in the MLIR representation.
  if (!hasSingleElement(main_func.getBlocks())) {
    main_func.emitError() << "expects 'main' function to have 1 block, got "
                          << main_func.getBlocks().size();
    return signalPassFailure();
  }

  if (failed(ResourceLiftingForFunctionalControlFlow(main_func)) ||
      failed(PromoteResourcesToArguments(main_func)))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreatePromoteResourcesToArgsPass() {
  return std::make_unique<PromoteResourcesToArgsPass>();
}

static PassRegistration<PromoteResourcesToArgsPass> pass(
    "tf-promote-resources-to-args",
    "Promote resources reads/writes to function inputs/outputs.");

}  // namespace TF
}  // namespace mlir
