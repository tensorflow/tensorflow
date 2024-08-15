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

#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_utils.h"
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
  absl::StatusOr<ElementsAttr> tensor_attr_or =
      tensorflow::ConvertTensor(tensor, &builder);
  if (!tensor_attr_or.ok()) return nullptr;
  return tensor_attr_or.value();
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
    if (failed(ReplaceVarWithConstant(var_handle_op.getResource().getUses(),
                                      elements_attr, &arguments_to_erase))) {
      return failure();
    }
  }

  if (failed(EraseObsoleteResourceUses(arguments_to_erase))) {
    return failure();
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
