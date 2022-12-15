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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/save_variables.h"

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace quantization {
namespace {

using ::mlir::func::FuncOp;
using ::mlir::tf_saved_model::GetSessionInitializerOp;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;
using ::mlir::tf_saved_model::kTfSavedModelInitializerTypeAttr;
using ::mlir::tf_saved_model::SessionInitializerOp;

// Gets the initializer function whose initializer_type attribute matches
// `type`. Returns a null operation if it doesn't exist.
FuncOp GetInitializerFunction(
    SessionInitializerOp session_init_op, mlir::SymbolTable symbol_table,
    llvm::StringRef type = kTfSavedModelInitializerRestoreType) {
  auto session_init_symbols = session_init_op.getInitializers()
                                  .getAsValueRange<mlir::FlatSymbolRefAttr>();
  if (session_init_symbols.empty()) {
    LOG(INFO) << "No session initializers exist in 'initializers' attribute of "
                 "SessionInitializerOp. No variables are saved to checkpoint.";
    return {};
  }

  FuncOp session_init_func_type_restore_op{};
  for (const auto init_sym : session_init_symbols) {
    auto init_func_op = symbol_table.lookup<FuncOp>(init_sym);

    if (auto init_type = init_func_op->getAttrOfType<mlir::StringAttr>(
            kTfSavedModelInitializerTypeAttr);
        init_type && init_type == type) {
      session_init_func_type_restore_op = init_func_op;
    }
  }

  return session_init_func_type_restore_op;
}

// Adds the tensor that initializes the variable through the provided
// `assign_var_op` to the `bundle_writer` for saving to checkpoint.
absl::Status AddTensorToBundleWriter(mlir::TF::AssignVariableOp assign_var_op,
                                     BundleWriter& bundle_writer) {
  auto resource_operand = assign_var_op.getOperand(0);
  auto var_handle_op =
      llvm::dyn_cast<mlir::TF::VarHandleOp>(resource_operand.getDefiningOp());
  if (!var_handle_op) {
    assign_var_op->emitRemark(
        "Operand idx 0 is not a tf.VarHandleOp. The initializing tensor is not "
        "saved to checkpoint.");
    return absl::OkStatus();
  }

  auto assigned_value_operand = assign_var_op.getOperand(1);
  auto const_op =
      llvm::dyn_cast<mlir::TF::ConstOp>(assigned_value_operand.getDefiningOp());
  if (!const_op) {
    assign_var_op->emitRemark(
        "Operand idx 1 is not a tf.ConstOp. The initializing tensor is not "
        "saved to checkpoint.");
    return absl::OkStatus();
  }

  Tensor const_tensor{};
  if (tsl::Status status = mlir::tfg::ConvertToTensor(
          /*attr=*/const_op.getValue(), /*output_tensor=*/&const_tensor);
      !status.ok()) {
    return tsl::ToAbslStatus(status);
  }

  return tsl::ToAbslStatus(bundle_writer.Add(
      /*key=*/var_handle_op.getSharedName(), const_tensor));
}

}  // namespace

absl::Status SaveVariablesToCheckpoint(const absl::string_view prefix,
                                       mlir::ModuleOp module_op) {
  SessionInitializerOp session_init_op = GetSessionInitializerOp(module_op);
  if (!session_init_op) {
    LOG(INFO) << "SessionInitializerOp does not exist. No variables are saved "
                 "to checkpoint.";
    return absl::OkStatus();
  }

  // Only the "tf.AssignVariableOp" patterns inside this initializer function
  // will be searched.
  FuncOp session_init_func_type_restore_op =
      GetInitializerFunction(session_init_op, mlir::SymbolTable(module_op));
  if (!session_init_func_type_restore_op) {
    LOG(INFO) << "No session initializer function with type 'restore_op'. No "
                 "variables are saved to checkpoint.";
    return absl::OkStatus();
  }

  BundleWriter bundle_writer(Env::Default(), prefix);
  if (!bundle_writer.status().ok()) {
    return tsl::ToAbslStatus(bundle_writer.status());
  }

  for (auto assign_variable_op :
       session_init_func_type_restore_op.getOps<mlir::TF::AssignVariableOp>()) {
    if (const absl::Status save_variable_status =
            AddTensorToBundleWriter(assign_variable_op, bundle_writer);
        !save_variable_status.ok()) {
      return save_variable_status;
    }
  }

  return tsl::ToAbslStatus(bundle_writer.Finish());
}

}  // namespace quantization
}  // namespace tensorflow
