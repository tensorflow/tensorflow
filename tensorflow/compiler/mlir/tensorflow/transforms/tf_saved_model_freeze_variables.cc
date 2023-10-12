/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"

#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/session_utils.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {
namespace {

// Attribute name that specifies the input shapes of a function.
constexpr StringRef kTfInputShapesAttr = "tf._input_shapes";

// Build and returns ElementsAttr which holds the data in 'tensor'.
ElementsAttr GetTensorValueAsElementsAttr(const tensorflow::Tensor& tensor,
                                          OpBuilder builder) {
  tensorflow::StatusOr<ElementsAttr> tensor_attr_or =
      tensorflow::ConvertTensor(tensor, &builder);
  if (!tensor_attr_or.ok()) return nullptr;
  return tensor_attr_or.value();
}

// Creates a constant op that holds 'tensor_elements'.
TF::ConstOp GetConstOpFromElementsAttr(ElementsAttr tensor_elements,
                                       OpBuilder builder, Location loc) {
  return builder.create<TF::ConstOp>(loc, tensor_elements);
}

// Returns ElementsAttr which has the value held by 'resource_tensor'.
ElementsAttr GetTensorValueAsElementsAttr(
    TF::VarHandleOp var_handle_op, const tensorflow::Tensor& resource_tensor,
    const tensorflow::DeviceMgr* mgr, OpBuilder builder) {
  if (resource_tensor.dtype() != tensorflow::DT_RESOURCE) {
    return GetTensorValueAsElementsAttr(resource_tensor, builder);
  }

  auto handle = resource_tensor.scalar<tensorflow::ResourceHandle>()();
  auto* var_ptr = tf_saved_model::GetVariableFromSession(var_handle_op,
                                                         handle.device(), mgr);
  if (!var_ptr) {
    return nullptr;
  }
  tensorflow::core::RefCountPtr<tensorflow::Var> var(var_ptr);
  auto* tensor = var_ptr->tensor();

  return GetTensorValueAsElementsAttr(*tensor, builder);
}

// Replace usage of 'read_variable_op' with 'value'.
void PropagateUsage(TF::ReadVariableOp read_variable_op, ElementsAttr value) {
  OpBuilder builder(read_variable_op);
  read_variable_op->getResult(0).replaceAllUsesWith(
      GetConstOpFromElementsAttr(value, builder, read_variable_op->getLoc()));
}

// Propagates a resource usage across the graph where
// 'user_op' uses a resource and is passed to this op at 'argument_index'.
// This resource should be replaced by 'value'.
// Output params:
// - work_list: Is updated with new regions to process that is called
//   by 'user_op';
// - arguments_to_erase: Captures updates to the graph - which arguments
//   to remove from the op;
void PropagateUsage(
    Operation* user_op, int argument_index, ElementsAttr value,
    llvm::SmallVector<std::pair<Region*, int>, 4>* work_list,
    llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>*
        arguments_to_erase) {
  if (auto read_variable_op = dyn_cast<TF::ReadVariableOp>(user_op)) {
    (*arguments_to_erase)[read_variable_op];
    PropagateUsage(read_variable_op, value);
  } else if (auto call = dyn_cast<CallOpInterface>(user_op)) {
    (*arguments_to_erase)[call].push_back(argument_index);
    if (auto func = dyn_cast<func::FuncOp>(call.resolveCallable())) {
      (*arguments_to_erase)[func].push_back(argument_index);
      work_list->push_back(std::make_pair(&func.getRegion(), argument_index));
    }
  } else if (auto if_op = dyn_cast<TF::IfOp>(user_op)) {
    (*arguments_to_erase)[if_op].push_back(argument_index);
    for (auto callee : {if_op.then_function(), if_op.else_function()}) {
      (*arguments_to_erase)[callee].push_back(argument_index - 1);
      work_list->push_back(
          std::make_pair(&callee.getBody(), argument_index - 1));
    }
  } else if (auto if_op = dyn_cast<TF::IfRegionOp>(user_op)) {
    (*arguments_to_erase)[if_op].push_back(argument_index);
    for (auto callee : {&if_op.getThenBranch(), &if_op.getElseBranch()}) {
      work_list->push_back(std::make_pair(callee, argument_index));
    }
  } else if (auto while_op = dyn_cast<TF::WhileOp>(user_op)) {
    (*arguments_to_erase)[while_op].push_back(argument_index);
    for (auto callee : {while_op.cond_function(), while_op.body_function()}) {
      (*arguments_to_erase)[callee].push_back(argument_index);
      work_list->push_back(std::make_pair(&callee.getBody(), argument_index));
    }
  } else if (auto while_op = dyn_cast<TF::WhileRegionOp>(user_op)) {
    (*arguments_to_erase)[while_op].push_back(argument_index);
    for (auto callee : {&while_op.getCond(), &while_op.getBody()}) {
      work_list->push_back(std::make_pair(callee, argument_index));
    }
  } else if (auto batch_func_op = dyn_cast<TF::BatchFunctionOp>(user_op)) {
    (*arguments_to_erase)[batch_func_op].push_back(argument_index);
    // Add the called function to the work list.
    func::FuncOp func_op = batch_func_op.func();
    (*arguments_to_erase)[func_op].push_back(argument_index);
    work_list->push_back({&func_op.getRegion(), argument_index});
  }
}

// An override that takes region.
void PropagateUsage(
    Region* region, ElementsAttr value, int argument_index,
    llvm::SmallVector<std::pair<Region*, int>, 4>* work_list,
    llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>*
        arguments_to_erase) {
  auto arg = region->getArgument(argument_index);
  for (auto& usage : arg.getUses()) {
    auto* user_op = usage.getOwner();
    int operand_index = usage.getOperandNumber();
    PropagateUsage(user_op, operand_index, value, work_list,
                   arguments_to_erase);
  }
}

// Traces usage of 'var_handle_op' and replaces it's usage with constant value
// 'value'.
// All op operands updates are captured in 'arguments_to_erase'.
void ReplaceVarWithConstant(
    TF::VarHandleOp var_handle_op, ElementsAttr value,
    llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>*
        arguments_to_erase) {
  llvm::SmallVector<std::pair<Region*, int>, 4> work_list;
  for (auto& usage : var_handle_op->getUses()) {
    auto* user_op = usage.getOwner();
    int operand_index = usage.getOperandNumber();
    PropagateUsage(user_op, operand_index, value, &work_list,
                   arguments_to_erase);
  }
  // Container to mark visited regions to avoid infinite loop.
  llvm::DenseSet<std::pair<Region*, int>> visited;
  while (!work_list.empty()) {
    auto work_item = work_list.pop_back_val();
    if (visited.contains(work_item)) continue;
    PropagateUsage(work_item.first, value, work_item.second, &work_list,
                   arguments_to_erase);
    visited.insert(work_item);
  }
}

// Returns ID for identifying a resource.
std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef> GetResourceKey(
    Operation* op) {
  llvm::StringRef device;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("device")) {
    device = attr.getValue();
  }

  llvm::StringRef container;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("container")) {
    container = attr.getValue();
  }

  llvm::StringRef shared_name;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("shared_name")) {
    shared_name = attr.getValue();
  }

  return std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>{
      device, container, shared_name};
}

// Remove the initialization of the variables in 'var_handle_ops' from
// the session init function 'session_init_func'
void RemoveVariablesInitializations(
    const llvm::SmallVector<TF::VarHandleOp, 4>& var_handle_ops,
    func::FuncOp session_init_func) {
  // We identify the variables using (device, container, shared_name) of the
  // resource. Capture them here and use them to identify the useless
  // initializations.
  llvm::SetVector<std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>>
      variables;
  for (auto var_handle_op : var_handle_ops)
    variables.insert(GetResourceKey(var_handle_op));

  llvm::SmallVector<Operation*, 4> work_list;
  for (auto var_handle_op : session_init_func.getOps<TF::VarHandleOp>()) {
    if (variables.count(GetResourceKey(var_handle_op)))
      work_list.push_back(var_handle_op);
  }

  // Capture list of ops to be erased by traversing usage starting from
  // the VarHandle ops.
  llvm::SetVector<Operation*> erase_list;
  while (!work_list.empty()) {
    auto* operation = work_list.pop_back_val();
    erase_list.insert(operation);
    for (auto& use : operation->getUses()) {
      if (erase_list.count(use.getOwner())) continue;
      work_list.push_back(use.getOwner());
    }
  }

  for (auto* op : erase_list) {
    op->dropAllUses();
    op->erase();
  }
}

// Updates terminator op arguments of 'func' after removing arguments
// specified in 'arguments_to_erase'.
template <typename T>
void UpdateTerminatorArguments(T& func,
                               const ArrayRef<unsigned> arguments_to_erase,
                               llvm::BitVector& erase_indices) {
  auto terminator = func.front().getTerminator();
  int num_operands = terminator->getNumOperands();
  erase_indices.resize(num_operands);
  for (auto arg_index : arguments_to_erase) {
    auto argument = func.getArgument(arg_index);
    for (auto& use : argument.getUses()) {
      if (llvm::isa<func::ReturnOp, TF::YieldOp>(use.getOwner())) {
        erase_indices.set(use.getOperandNumber());
      }
    }
    func.getArgument(arg_index).dropAllUses();
  }
  if (llvm::isa<func::ReturnOp, TF::YieldOp>(func.front().getTerminator())) {
    terminator->eraseOperands(erase_indices);
  }
}

// Updates 'while_op' signatures based on which arguments should be removed
// in 'arguments_to_erase'.
template <typename T, typename U>
T GetUpdatedWhileOp(T while_op, const U& argument_types,
                    const llvm::SmallVector<unsigned, 4>& arguments_to_erase) {
  OpBuilder builder(while_op);
  llvm::SmallVector<Type, 4> new_operand_types;
  llvm::SmallVector<Value> new_operands;
  auto operands = while_op->getOperands();
  const int num_operands = while_op->getNumOperands();
  llvm::BitVector skip_indices(num_operands);
  for (int i : arguments_to_erase) skip_indices.set(i);
  for (int i = 0; i < num_operands; ++i) {
    if (!skip_indices.test(i)) {
      new_operand_types.emplace_back(argument_types[i]);
      new_operands.emplace_back(operands[i]);
    }
  }
  auto new_while_op = builder.create<T>(while_op->getLoc(), new_operand_types,
                                        new_operands, while_op->getAttrs());
  int new_index = 0;
  for (int i = 0; i < num_operands; ++i) {
    if (!skip_indices.test(i)) {
      while_op->getResult(i).replaceAllUsesWith(
          new_while_op->getResult(new_index++));
    }
  }
  return new_while_op;
}

// Erases function arguments indexed at `args_to_erase`. Also applies the
// changes to any relevant function attributes accordingly.
void EraseFuncOpArguments(func::FuncOp func_op,
                          const ArrayRef<unsigned> args_to_erase) {
  BitVector args_to_erase_bit_vector(func_op.getNumArguments());
  for (const unsigned i : args_to_erase) args_to_erase_bit_vector.set(i);

  func_op.eraseArguments(args_to_erase_bit_vector);

  // Erases entries in "tf._input_shapes" attribute of `func_op` that correspond
  // to the erased arguments.
  if (auto input_shapes_attr =
          func_op->getAttrOfType<ArrayAttr>(kTfInputShapesAttr);
      input_shapes_attr) {
    // Construct a new array of input shapes excluding the input shapes of the
    // erased arguments.
    SmallVector<Attribute> updated_input_shapes_attr;
    for (const unsigned i : args_to_erase_bit_vector.flip().set_bits()) {
      updated_input_shapes_attr.emplace_back(input_shapes_attr[i]);
    }

    // Replaces the attribute with the updated "#tf_type.shape" array.
    // Builder builder(func_op.getContext());
    func_op->setAttr(
        kTfInputShapesAttr,
        ArrayAttr::get(func_op.getContext(), updated_input_shapes_attr));
  }
}

// Validates func ops. Returns `failure` if the function is invalid.
LogicalResult ValidateFuncOp(func::FuncOp func_op) {
  auto input_shapes_attr =
      func_op->getAttrOfType<ArrayAttr>(kTfInputShapesAttr);
  if (!input_shapes_attr) return success();

  if (input_shapes_attr.size() != func_op.getNumArguments()) {
    return func_op->emitError(
               "Number of arguments and 'tf._input_shapes' "
               "attribute size do not match. ")
           << "Num args: " << func_op.getNumArguments()
           << ", tf._input_shapes size: " << input_shapes_attr.size();
  }

  return success();
}

// Validates ModuleOp. Returns `failure` if the module op is invalid.
LogicalResult ValidateModule(ModuleOp module_op) {
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    if (failed(ValidateFuncOp(func_op))) {
      return failure();
    }
  }
  return success();
}

}  // namespace

LogicalResult FreezeVariables(ModuleOp module, tensorflow::Session* session) {
  if (failed(ValidateModule(module))) return failure();

  const tensorflow::DeviceMgr* mgr = nullptr;
  auto status = session->LocalDeviceManager(&mgr);
  if (!status.ok()) {
    module->emitError(
        absl::StrCat("failed to fetch device manager: ", status.message()));
    return failure();
  }

  SmallVector<func::FuncOp, 2> session_init_funcs =
      tf_saved_model::GetInitializerFunctions(module);
  func::FuncOp session_init_func =
      session_init_funcs.empty() ? nullptr : session_init_funcs[0];

  TF::ResourceAnalyzer analyzer(module, /*skip_session_init=*/true);
  llvm::SmallVector<TF::VarHandleOp, 4> variables;
  // Capture list of all read only variables.
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func == session_init_func) continue;
    for (auto var_handle_op : func.getOps<TF::VarHandleOp>()) {
      if (!analyzer.IsPotentiallyWritten(var_handle_op.getResource())) {
        variables.push_back(var_handle_op);
      }
    }
  }

  // Fetch the values to replace the VarHandleOps with.
  auto resource_tensors_or =
      tf_saved_model::GetResourcesFromSession(variables, session);
  if (!resource_tensors_or.ok()) {
    module->emitError(resource_tensors_or.status().message().data());
    return failure();
  }

  auto* context = module.getContext();
  OpBuilder builder(context);
  // Note: We can't modify the graph while navigating through it, as erasing
  // invalidate pointers.
  // So instead we capture all the updates in the below map, and then
  // process them after.

  // Container to hold all update actions on ops.
  // Key: Operation to update.
  // Value: optional list of argument indices to delete from this op.
  // Note that we use MapVector because we want to iterate on the same order
  // of insertion.
  llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>
      arguments_to_erase;
  for (auto [var_handle_op, resource_tensor] :
       llvm::zip(variables, resource_tensors_or.value())) {
    builder.setInsertionPointAfterValue(var_handle_op);
    auto elements_attr = GetTensorValueAsElementsAttr(
        var_handle_op, resource_tensor, mgr, builder);
    ReplaceVarWithConstant(var_handle_op, elements_attr, &arguments_to_erase);
  }

  // All updates to different ops are captured in 'arguments_to_erase'.
  // Now loop on them and based on each item type update accordingly.
  for (auto& [user_op, args_to_erase] : arguments_to_erase) {
    if (auto func = dyn_cast<func::FuncOp>(user_op)) {
      // To update a function we will need to:
      // 1) Remove the unused arguments from the function itself.
      //    1-2) Remove func attributes corresponding to the removed arguments.
      // 2) Remove any returns that are not needed from the function terminator
      //    op in the function.
      // 3) Update function result to match the terminator.
      llvm::BitVector result_indices_to_erase;
      UpdateTerminatorArguments(func, args_to_erase, result_indices_to_erase);
      EraseFuncOpArguments(func, args_to_erase);

      func.eraseResults(result_indices_to_erase);
    } else if (auto read_var = dyn_cast<TF::ReadVariableOp>(user_op)) {
      // Read variables was already replaced by constant op. Just remove the op.
      read_var->erase();
    } else if (auto while_op = dyn_cast<TF::WhileOp>(user_op)) {
      GetUpdatedWhileOp<TF::WhileOp>(
          while_op, while_op.cond_function().getArgumentTypes(), args_to_erase);
      while_op->erase();
    } else if (auto while_op = dyn_cast<TF::WhileRegionOp>(user_op)) {
      auto new_while_op = GetUpdatedWhileOp(
          while_op, while_op.getCond().getArgumentTypes(), args_to_erase);
      new_while_op.getCond().takeBody(while_op.getCond());
      new_while_op.getBody().takeBody(while_op.getBody());
      llvm::BitVector erase_indices;
      UpdateTerminatorArguments(new_while_op.getBody(), args_to_erase,
                                erase_indices);
      llvm::BitVector body_bit_vector(
          new_while_op.getBody().front().getNumArguments());
      for (auto i : args_to_erase) body_bit_vector.set(i);
      new_while_op.getBody().front().eraseArguments(body_bit_vector);
      llvm::BitVector cond_bit_vector(
          new_while_op.getCond().front().getNumArguments());
      for (auto i : args_to_erase) cond_bit_vector.set(i);
      new_while_op.getCond().front().eraseArguments(cond_bit_vector);
      while_op->erase();
    } else if (auto batch_func_op = dyn_cast<TF::BatchFunctionOp>(user_op)) {
      llvm::BitVector erase_indices(user_op->getNumOperands());
      for (auto operand_index : args_to_erase) {
        erase_indices.set(operand_index);
      }
      batch_func_op.eraseArguments(erase_indices);
    } else {
      llvm::BitVector erase_indices(user_op->getNumOperands());
      for (auto operand_index : args_to_erase) {
        erase_indices.set(operand_index);
      }
      user_op->eraseOperands(erase_indices);
    }
  }

  // Remove initialization of unused variables.
  if (session_init_func)
    RemoveVariablesInitializations(variables, session_init_func);

  // Remove the unused VarHandleOp.
  for (auto var_handle_op : variables) {
    if (var_handle_op) var_handle_op->erase();
  }
  return success();
}

}  // namespace tf_saved_model
}  // namespace mlir
