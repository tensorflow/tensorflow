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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/savedmodel_passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/session_utils.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {
namespace {

void InitializeVariable(TF::VarHandleOp var_handle_op,
                        tensorflow::Tensor* tensor,
                        func::FuncOp session_init_func, OpBuilder builder) {
  tensorflow::StatusOr<ElementsAttr> tensor_attr_or =
      tensorflow::ConvertTensor(*tensor, &builder);
  assert(tensor_attr_or.ok() && "Expect valid tensor");
  ElementsAttr tensor_attr = tensor_attr_or.value();

  builder.setInsertionPointToStart(&session_init_func.getBlocks().front());
  auto var_handle_op_in_init = var_handle_op->clone();
  builder.insert(var_handle_op_in_init);
  auto const_op = builder.create<mlir::arith::ConstantOp>(
      session_init_func.getLoc(), tensor_attr.getType(), tensor_attr);

  builder.create<TF::AssignVariableOp>(
      session_init_func.getLoc(), llvm::ArrayRef<mlir::Type>{},
      llvm::ArrayRef<mlir::Value>{var_handle_op_in_init->getResult(0),
                                  const_op.getResult()});
}

constexpr char kTfSavedModelExportedNameAttr[] =
    "tf_saved_model.exported_names";

func::FuncOp CreateSessionInitFunc(ModuleOp module) {
  constexpr char kSessionInitFuncName[] = "SessionInitializerFunction";

  mlir::OpBuilder builder(module.getBodyRegion());
  auto func_type =
      FunctionType::get(module.getContext(), /*inputs=*/{}, /*results=*/{});
  auto func = builder.create<func::FuncOp>(module->getLoc(),
                                           kSessionInitFuncName, func_type);
  func->setAttr(kTfSavedModelExportedNameAttr,
                builder.getStrArrayAttr({kSessionInitFuncName}));
  func.setVisibility(mlir::func::FuncOp::Visibility::Public);
  auto func_builder = OpBuilder::atBlockBegin(func.addEntryBlock());
  func_builder.create<mlir::func::ReturnOp>(func.getLoc());
  // In cases where there is a session initializer op with empty initializer,
  // replace the session initializer with the new one that points to the session
  // initializer func.
  SessionInitializerOp session_init_op = GetSessionInitializerOp(module);
  auto new_session_init_op =
      builder.create<tf_saved_model::SessionInitializerOp>(
          module->getLoc(), builder.getArrayAttr(SymbolRefAttr::get(
                                builder.getContext(), kSessionInitFuncName)));
  if (session_init_op) {
    session_init_op->replaceAllUsesWith(new_session_init_op);
    session_init_op->erase();
  }
  return func;
}

func::FuncOp GetOrCreateSessionInitFunc(ModuleOp module) {
  SessionInitializerOp session_init_op = GetSessionInitializerOp(module);
  if (!session_init_op) return CreateSessionInitFunc(module);

  SymbolTable symbol_table(module);
  if (!session_init_op.initializers().empty()) {
    func::FuncOp init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
        session_init_op.initializers()[0].cast<FlatSymbolRefAttr>().getValue());
    return init_func_op;
  }
  return CreateSessionInitFunc(module);
}

}  // namespace

LogicalResult InitializeVariablesInSessionInitializer(
    ModuleOp module, tensorflow::Session* session) {
  const tensorflow::DeviceMgr* mgr = nullptr;
  auto status = session->LocalDeviceManager(&mgr);
  if (!status.ok()) {
    module->emitError("failed to fetch device manager: " +
                      status.error_message());
    return failure();
  }

  // Fetch all VarHandleOp.
  llvm::StringSet<> variable_names;
  llvm::SmallVector<TF::VarHandleOp, 4> var_ops;
  for (auto func_op : module.getOps<func::FuncOp>()) {
    for (auto var_handle_op : func_op.getOps<TF::VarHandleOp>()) {
      auto variable_name = GetVariableName(var_handle_op);
      if (variable_names.count(variable_name)) continue;
      var_ops.emplace_back(var_handle_op);
      variable_names.insert(variable_name);
    }
  }

  // Get resources from Session.
  auto resource_tensors_or = GetResourcesFromSession(var_ops, session);
  if (!resource_tensors_or.ok()) {
    module->emitError(resource_tensors_or.status().message().data());
    return failure();
  }

  auto session_init_func = GetOrCreateSessionInitFunc(module);
  OpBuilder builder(session_init_func.getContext());

  for (auto var_and_tensor : llvm::zip(var_ops, resource_tensors_or.value())) {
    auto& var_op = std::get<0>(var_and_tensor);
    auto& resource_tensor = std::get<1>(var_and_tensor);
    if (resource_tensor.dtype() != tensorflow::DT_RESOURCE) {
      InitializeVariable(var_op, &resource_tensor, session_init_func, builder);
      continue;
    }

    auto handle = resource_tensor.scalar<tensorflow::ResourceHandle>()();
    auto* var_ptr = GetVariableFromSession(var_op, handle.device(), mgr);
    if (!var_ptr) {
      // If no value in session, then just skip this variable.
      // This can happen if the variable is not saved in checkpoint.
      // For example, when the variable is created on every call.
      continue;
    }
    tensorflow::core::RefCountPtr<tensorflow::Var> var(var_ptr);
    auto* tensor = var_ptr->tensor();

    InitializeVariable(var_op, tensor, session_init_func, builder);
  }
  return success();
}

}  // namespace tf_saved_model
}  // namespace mlir
