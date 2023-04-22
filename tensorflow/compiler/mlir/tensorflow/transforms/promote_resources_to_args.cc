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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TF {
namespace {

constexpr char kResourceFunctionMsg[] =
    "expects function level resource argument";
constexpr char kInvalidResourceMsg[] =
    "expects resource to be a VarHandleOp or function argument";
constexpr char kResourceNameArgAttr[] = "tf.resource_name";

// Checks if a function has only one block.
mlir::LogicalResult CheckSingleBlockFunction(FuncOp function) {
  if (!llvm::hasSingleElement(function)) {
    return function.emitError()
           << "expects function '" << function.getName()
           << "' to have 1 block, got " << function.getBlocks().size();
  }
  return success();
}

// Collects names of users of a resource that are not `tf.ReadVariableOp` and
// not `tf.AssignVariableOp`.
llvm::SmallSet<llvm::StringRef, 1> GetCompositeResourceUserNames(
    Value resource) {
  // SmallSet will use a vector when there is only one element and use std::set
  // when there are more than one elements. This ensures that the operations in
  // the error message are ordered.
  llvm::SmallSet<llvm::StringRef, 1> composite_users;
  for (Operation* user : resource.getUsers())
    if (!llvm::isa<TF::ReadVariableOp, TF::AssignVariableOp>(user))
      composite_users.insert(user->getName().getStringRef());

  return composite_users;
}

// Checks that the only users of `tf.VarHandleOp` are
// `tf.ReadVariableOp` and `tf.AssignVariableOp`.
mlir::LogicalResult ValidateVarHandle(TF::VarHandleOp var_handle_op) {
  auto composite_ops = GetCompositeResourceUserNames(var_handle_op);
  if (!composite_ops.empty())
    return var_handle_op.emitOpError()
           << "expects users to be 'tf.ReadVariableOp' or "
              "'tf.AssignVariableOp', got ["
           << llvm::join(composite_ops.begin(), composite_ops.end(), ", ")
           << "]";

  return success();
}

// Checks if resource argument has a valid resource subtype and its users are of
// `tf.ReadVariableOp` and `tf.AssignVariableOp` only.
mlir::LogicalResult ValidateResourceArgument(FuncOp function,
                                             BlockArgument resource_arg,
                                             TF::ResourceType resource_type) {
  if (resource_type.getSubtypes().size() != 1)
    return function.emitError()
           << "expects resource type of argument "
           << resource_arg.getArgNumber() << " to have one subtype, got "
           << resource_type;

  auto composite_ops = GetCompositeResourceUserNames(resource_arg);
  if (!composite_ops.empty())
    return function.emitError()
           << "expects users of resource argument "
           << resource_arg.getArgNumber()
           << " to be 'tf.ReadVariableOp' or 'tf.AssignVariableOp', got ["
           << llvm::join(composite_ops.begin(), composite_ops.end(), ", ")
           << "]";

  return success();
}

bool VariableIsInitialized(TF::VarHandleOp var_handle_op) {
  auto is_variable_initialized =
      var_handle_op->getAttrOfType<BoolAttr>("_is_initialized");
  // Assume variable is initialized if attribute is not set.
  // There are paths that doesn't mark the variables. All variables
  // that doesn't have the attribute will be promoted.
  return !is_variable_initialized || is_variable_initialized.getValue();
}

// Adds resource arguments for every unique (name) variable handle. Associated
// `tf.VarHandleOp` are removed from the function. Variable shared names are
// returned in `var_handle_shared_names` based on the ordering of added resource
// arguments.
mlir::LogicalResult PromoteVarHandlesToArguments(
    FuncOp function, bool add_validation,
    llvm::SmallVectorImpl<std::string>* var_handle_shared_names) {
  Block& block = function.front();
  auto func_type = function.getType();

  auto func_arg_types = llvm::to_vector<4>(func_type.getInputs());
  llvm::SmallDenseMap<llvm::StringRef, int> var_arg_index_by_name;
  for (auto var_handle_op :
       llvm::make_early_inc_range(block.getOps<TF::VarHandleOp>())) {
    if (add_validation && failed(ValidateVarHandle(var_handle_op)))
      return failure();
    // In the case of variables that are not initialized at graph creation
    // then we keep them as VarHandleOps.
    if (!VariableIsInitialized(var_handle_op)) continue;

    llvm::StringRef name = var_handle_op.shared_nameAttr().getValue();
    auto it = var_arg_index_by_name.insert({name, func_arg_types.size()});
    if (it.second) {
      var_handle_shared_names->emplace_back(name);
      auto resource_type = var_handle_op.resource().getType();
      func_arg_types.push_back(resource_type);
      var_handle_op.resource().replaceAllUsesWith(
          block.addArgument(resource_type));
    } else {
      var_handle_op.resource().replaceAllUsesWith(
          block.getArgument(it.first->getSecond()));
    }
    var_handle_op.erase();
  }

  if (!var_handle_shared_names->empty())
    function.setType(FunctionType::get(function.getContext(), func_arg_types,
                                       func_type.getResults()));

  return success();
}

// Records the current live value for a resource variable and whether a read or
// write on the variable occurred.
struct ResourceInfo {
  Value live_value = nullptr;
  bool read = false;
  bool write = false;
};

LogicalResult PromoteResourcesToArguments(
    FuncOp function, llvm::ArrayRef<std::string> var_handle_shared_names) {
  Block& block = function.front();

  auto return_op = llvm::dyn_cast_or_null<ReturnOp>(block.getTerminator());
  if (!return_op)
    return function.emitError() << "expects function '" << function.getName()
                                << "' to have a MLIR ReturnOp";

  llvm::SmallVector<ResourceInfo, 4> resources(function.getNumArguments());
  auto argument_types = llvm::to_vector<4>(function.getType().getInputs());
  bool has_resources = false;
  auto add_resource_argument = [&](BlockArgument arg,
                                   TF::ResourceType resource_type) {
    Type arg_type = resource_type.getSubtypes().front();
    arg.setType(arg_type);
    resources[arg.getArgNumber()].live_value = arg;
    argument_types[arg.getArgNumber()] = arg_type;
    has_resources = true;
  };

  // Loop through the non `tf.VarHandleOp` resource arguments in the function,
  // validate its uses and subtype, and store a mapping from that argument to
  // itself as the current live value.
  auto func_args = function.getArguments().take_front(
      function.getNumArguments() - var_handle_shared_names.size());
  for (BlockArgument& func_arg : func_args) {
    auto resource_type =
        getElementTypeOrSelf(func_arg.getType()).dyn_cast<TF::ResourceType>();
    if (!resource_type) continue;
    if (failed(ValidateResourceArgument(function, func_arg, resource_type)))
      return failure();

    add_resource_argument(func_arg, resource_type);
  }

  // Loop through `tf.VarHandleOp` resource arguments in the function and store
  // a mapping from that argument to itself as the current live value. No
  // validations are necessary here as these arguments were validated prior to
  // being added.
  auto var_handle_args =
      function.getArguments().take_back(var_handle_shared_names.size());
  for (BlockArgument& var_handle_arg : var_handle_args) {
    auto resource_type =
        getElementTypeOrSelf(var_handle_arg.getType()).cast<TF::ResourceType>();
    add_resource_argument(var_handle_arg, resource_type);
  }

  if (!has_resources) return success();

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

        ResourceInfo& resource_info = resources[func_arg.getArgNumber()];
        resource_info.read = true;
        read_op.value().replaceAllUsesWith(resource_info.live_value);
      } else {
        return read_op.emitOpError(kInvalidResourceMsg);
      }

      read_op.erase();
    } else if (auto write_op = llvm::dyn_cast<TF::AssignVariableOp>(&op)) {
      if (auto func_arg = write_op.resource().dyn_cast<BlockArgument>()) {
        if (func_arg.getOwner() != &block)
          return write_op.emitOpError(kResourceFunctionMsg);

        ResourceInfo& resource_info = resources[func_arg.getArgNumber()];
        resource_info.write = true;
        resource_info.live_value = write_op.value();
      } else {
        return read_op.emitOpError(kInvalidResourceMsg);
      }

      write_op.erase();
    }
  }

  const int64_t num_results_before = function.getNumResults();
  auto return_operands = llvm::to_vector<4>(return_op.getOperands());
  auto result_types = llvm::to_vector<4>(return_op.getOperandTypes());
  llvm::SmallVector<std::pair<int64_t, llvm::StringRef>, 4>
      output_only_resources;
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> input_output_alias;

  // Collect new return values for variable writes and either (a) output-only
  // resource attributes (if the resource is not promoted to an argument) or (b)
  // mapping from resource input index to output alias (if the resource has been
  // promoted to an argument). Resource arguments that were originally
  // `tf.VarHandleOp` but not read are collected and then removed.
  OpBuilder builder(return_op);
  const int var_handles_start_idx =
      function.getNumArguments() - var_handle_shared_names.size();
  int new_argument_index = 0;
  llvm::SmallVector<int, 4> argument_indices_to_remove;
  for (auto resource_and_index : llvm::enumerate(resources)) {
    const auto& resource = resource_and_index.value();
    if (!resource.live_value) {
      // Ignore non resource arguments.
      ++new_argument_index;
      continue;
    }

    const int64_t index = resource_and_index.index();
    const bool is_var_handle = index >= var_handles_start_idx;
    if (resource.write) {
      if (!is_var_handle || resource.read) {
        input_output_alias.push_back(
            {new_argument_index, return_operands.size()});
      } else if (is_var_handle) {
        output_only_resources.push_back(
            {return_operands.size(),
             var_handle_shared_names[index - var_handles_start_idx]});
      }
      return_operands.push_back(resource.live_value);
      result_types.push_back(resource.live_value.getType());
    }

    if (is_var_handle && !resource.read) {
      assert(block.getArgument(index).getUses().empty());
      argument_indices_to_remove.push_back(index);
    } else {
      if (is_var_handle) {
        // Add resource_name attribute to VarHandleOp read.
        function.setArgAttr(
            new_argument_index, kResourceNameArgAttr,
            builder.getStringAttr(
                var_handle_shared_names[index - var_handles_start_idx]));
      }
      ++new_argument_index;
    }
  }

  // Remove unread var handle arguments.
  for (int argument_index_to_remove :
       llvm::reverse(argument_indices_to_remove)) {
    block.eraseArgument(argument_index_to_remove);
    argument_types.erase(argument_types.begin() + argument_index_to_remove);
  }

  // Rewrite return if there are variable writes.
  const int return_operands_size = return_operands.size();
  if (return_operands_size > num_results_before) {
    builder.create<ReturnOp>(return_op.getLoc(), return_operands);
    return_op.erase();
  }

  // Update function argument and result types with new resource subtypes.
  function.setType(builder.getFunctionType(argument_types, result_types));

  // Add resource_name attribute to the output for the resources.
  for (auto& resource : output_only_resources)
    function.setResultAttr(resource.first, kResourceNameArgAttr,
                           builder.getStringAttr(resource.second));

  // Add aliasing_output attribute to the input argument for the resources that
  // are updated by the function.
  for (auto& input_output : input_output_alias)
    function.setArgAttr(input_output.first, "tf.aliasing_output",
                        builder.getI64IntegerAttr(input_output.second));

  return success();
}

class PromoteResourcesToArgsPass
    : public PromoteResourcesToArgsPassBase<PromoteResourcesToArgsPass> {
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
  if (failed(CheckSingleBlockFunction(main_func))) return signalPassFailure();

  llvm::SmallVector<std::string, 4> var_handle_shared_names;
  if (failed(ResourceLiftingForFunctionalControlFlow(main_func)) ||
      failed(PromoteVarHandlesToArguments(main_func, /*add_validation=*/true,
                                          &var_handle_shared_names)) ||
      failed(PromoteResourcesToArguments(main_func, var_handle_shared_names)))
    return signalPassFailure();
}

class PromoteVarHandlesToArgsPass
    : public PromoteVarHandlesToArgsPassBase<PromoteVarHandlesToArgsPass> {
 public:
  void runOnOperation() override;
};

void PromoteVarHandlesToArgsPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* context = module.getContext();
  for (auto function : module.getOps<FuncOp>()) {
    if (failed(CheckSingleBlockFunction(function))) return signalPassFailure();

    llvm::SmallVector<std::string, 4> var_handle_shared_names;
    (void)PromoteVarHandlesToArguments(function, /*add_validation=*/false,
                                       &var_handle_shared_names);

    // Add resource names for each `tf.VarHandleOp` that were promoted to
    // resource arguments.
    const int var_handle_args_offset =
        function.getNumArguments() - var_handle_shared_names.size();
    for (auto var_name_and_index : llvm::enumerate(var_handle_shared_names))
      function.setArgAttr(var_name_and_index.index() + var_handle_args_offset,
                          kResourceNameArgAttr,
                          StringAttr::get(context, var_name_and_index.value()));
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreatePromoteResourcesToArgsPass() {
  return std::make_unique<PromoteResourcesToArgsPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreatePromoteVarHandlesToArgsPass() {
  return std::make_unique<PromoteVarHandlesToArgsPass>();
}

}  // namespace TF
}  // namespace mlir
