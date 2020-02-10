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
// function, the corresponding input and output arguments are alias. This
// aliasing information is recorded as a named attribute tf.aliasing_output of
// the input arguments.
//
// Assumption of this pass:
//  . Compound resource operations have already been decomposed.
//  . Dead functions have already been removed, as resource arguments in dead
//    functions can cause the pass to fail.
//
// TODO(bixia): This pass currently reports any error when it sees ResourceType
//   as function arguments. That is, this pass assumes resource reads/writes in
//   functions called by the main function, such as through TF IfOp and WhileOp,
//   have already been functionalized. This functionalization can be achieved by
//   either finishing cl/281636304 or enhancing PromoteResourcesToArguments
//   here.

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace {

// Records the input argument index and the current live value for a resource
// variable.
struct ResourceInfo {
  int64_t input_index;
  Value live_value;
};

using ResourceMap = llvm::SmallDenseMap<llvm::StringRef, ResourceInfo>;

LogicalResult VerifyNoPotentialNestedResourceAccesses(ModuleOp module) {
  LogicalResult result = success();
  module.walk([&](FuncOp func) {
    for (auto type : func.getType().getInputs()) {
      if (getElementTypeOrSelf(type).isa<TF::ResourceType>()) {
        result =
            func.emitError("potential nested resource accesses in function");
        break;
      }
    }
  });

  return result;
}

LogicalResult PromoteResourcesToArguments(FuncOp function) {
  // This routine should only be called when control flow operations are still
  // represented with TF IfOp and WhileOp operations. In this case, there should
  // be only one basic blocks in the MLIR representation.
  if (!has_single_element(function.getBlocks())) {
    return function.emitError()
           << "expect the function to have 1 block while it has "
           << function.getBlocks().size();
  }

  ResourceMap resource_map;
  std::vector<Type> new_input_types = function.getType().getInputs().vec();
  int64_t input_num = function.getNumArguments();

  // Loop through the VarHandleOp in the function. When the first VarHandleOp
  // for a resource variable is encountered, create a new function argument and
  // add an entry to the resource_map to record the information.
  for (auto var_handle_op : function.front().getOps<TF::VarHandleOp>()) {
    if (resource_map.count(var_handle_op.shared_name())) {
      continue;
    }

    auto resource_type =
        getElementTypeOrSelf(var_handle_op.getType()).cast<TF::ResourceType>();
    if (!resource_type || resource_type.getSubtypes().size() != 1) {
      return var_handle_op.emitError("unrecognized resource type");
    }
    Type arg_type = resource_type.getSubtypes().front();
    BlockArgument arg = function.front().addArgument(arg_type);
    new_input_types.push_back(arg_type);
    resource_map[var_handle_op.shared_name()] = {input_num++, arg};
  }

  if (resource_map.empty()) {
    return success();
  }

  // We initially assign the argument for a resource as the live value for the
  // resource. We then walk through the operations in the function in their
  // lexical order, to update the live value for the resource when we see a
  // store to the resource and replace reads of the resource with uses of its
  // live value.
  for (Operation& op : llvm::make_early_inc_range(function.front())) {
    if (auto read_op = llvm::dyn_cast<TF::ReadVariableOp>(&op)) {
      auto var_handle_op =
          llvm::dyn_cast<TF::VarHandleOp>(read_op.resource().getDefiningOp());
      if (!var_handle_op) {
        return read_op.emitError("resource is not VarHandleOp");
      }
      read_op.value().replaceAllUsesWith(
          resource_map[var_handle_op.shared_name()].live_value);
      read_op.erase();
    } else if (auto write_op = llvm::dyn_cast<TF::AssignVariableOp>(&op)) {
      auto var_handle_op =
          llvm::dyn_cast<TF::VarHandleOp>(write_op.resource().getDefiningOp());
      if (!var_handle_op) {
        return write_op.emitError("resource is not VarHandleOp");
      }
      resource_map[var_handle_op.shared_name()].live_value = write_op.value();
      write_op.erase();
    }
  }

  auto return_op = llvm::dyn_cast<ReturnOp>(function.front().getTerminator());
  if (!return_op) {
    return function.emitError("the function doesn't have an MLIR ReturnOp");
  }

  int64_t output_num = return_op.getNumOperands();
  llvm::SmallVector<Value, 4> new_return_operands(return_op.getOperands());
  std::vector<std::pair<int64_t, int64_t>> input_output_alias;
  std::vector<Type> new_return_types = function.getType().getResults().vec();

  // If the live value of a resource is not an argument, then the resource is
  // updated by the function. Add the resource live value to the ReturnOp of the
  // function and record the input-output aliasing.
  for (Operation& op : function.front()) {
    if (auto var_handle_op = llvm::dyn_cast<TF::VarHandleOp>(&op)) {
      ResourceInfo& resource_info = resource_map[var_handle_op.shared_name()];
      Value live_value = resource_info.live_value;
      if (!live_value.isa<BlockArgument>()) {
        new_return_operands.push_back(live_value);
        input_output_alias.push_back(
            std::make_pair(resource_info.input_index, output_num++));
        new_return_types.push_back(live_value.getType());
      }
    }
  }

  // Erase all VarHandleOp.
  for (Operation& op : llvm::make_early_inc_range(function.front())) {
    if (llvm::isa<TF::VarHandleOp>(&op)) {
      op.erase();
    }
  }

  OpBuilder builder(return_op);
  function.setType(builder.getFunctionType(new_input_types, new_return_types));

  if (input_output_alias.empty()) {
    return success();
  }

  builder.create<ReturnOp>(return_op.getLoc(), new_return_operands);
  return_op.erase();

  // Add aliasing_output attribute to the input argument for the resources that
  // are updated by the function.
  for (auto input_output : input_output_alias) {
    function.setArgAttr(input_output.first, "tf.aliasing_output",
                        builder.getI64IntegerAttr(input_output.second));
  }

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
  if (!main_func) {
    return;
  }

  if (failed(VerifyNoPotentialNestedResourceAccesses(module)) ||
      failed(PromoteResourcesToArguments(main_func))) {
    return signalPassFailure();
  }
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
