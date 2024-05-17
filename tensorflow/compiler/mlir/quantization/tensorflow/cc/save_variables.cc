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

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace quantization {
namespace {

using ::mlir::func::FuncOp;
using ::mlir::tf_saved_model::GetInitializerFunction;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;

// Adds the tensor that initializes the variable through the provided
// `assign_var_op` to the `bundle_writer` for saving to checkpoint. Returns the
// shared name of the variable if a variable is saved successfully. If the
// variable is not saved, returns an empty string.
absl::StatusOr<std::string> AddTensorToBundleWriter(
    mlir::TF::AssignVariableOp assign_var_op, BundleWriter& bundle_writer) {
  auto resource_operand = assign_var_op.getOperand(0);
  auto var_handle_op =
      llvm::dyn_cast<mlir::TF::VarHandleOp>(resource_operand.getDefiningOp());
  if (!var_handle_op) {
    assign_var_op->emitRemark(
        "Operand idx 0 is not a tf.VarHandleOp. The initializing tensor is not "
        "saved to checkpoint.");
    return "";
  }

  auto assigned_value_operand = assign_var_op.getOperand(1);
  auto const_op =
      llvm::dyn_cast<mlir::TF::ConstOp>(assigned_value_operand.getDefiningOp());
  if (!const_op) {
    assign_var_op->emitRemark(
        "Operand idx 1 is not a tf.ConstOp. The initializing tensor is not "
        "saved to checkpoint.");
    return "";
  }

  Tensor const_tensor{};
  if (const absl::Status status = mlir::tfg::ConvertToTensor(
          /*attr=*/const_op.getValue(), /*output_tensor=*/&const_tensor);
      !status.ok()) {
    return status;
  }

  if (!bundle_writer.Add(/*key=*/var_handle_op.getSharedName(), const_tensor)
           .ok()) {
    return bundle_writer.status();
  }

  return var_handle_op.getSharedName().str();
}

}  // namespace

absl::StatusOr<std::vector<std::string>> SaveVariablesToCheckpoint(
    const absl::string_view prefix, mlir::ModuleOp module_op) {
  // Only the "tf.AssignVariableOp" patterns inside this initializer function
  // will be searched.
  FuncOp session_init_func_type_restore_op = GetInitializerFunction(
      module_op, /*initializer_type=*/kTfSavedModelInitializerRestoreType);
  if (!session_init_func_type_restore_op) {
    LOG(INFO) << "No session initializer function with type 'restore_op'. No "
                 "variables are saved to checkpoint.";
    return std::vector<std::string>{};
  }

  BundleWriter bundle_writer(Env::Default(), prefix);
  if (!bundle_writer.status().ok()) {
    return bundle_writer.status();
  }

  std::vector<std::string> saved_variable_shared_names;
  for (auto assign_variable_op :
       session_init_func_type_restore_op.getOps<mlir::TF::AssignVariableOp>()) {
    if (const absl::StatusOr<std::string> variable_shared_name =
            AddTensorToBundleWriter(assign_variable_op, bundle_writer);
        !variable_shared_name.ok()) {
      return variable_shared_name.status();
    } else if (!variable_shared_name->empty()) {
      // Empty string means the variable isn't applicable for saving.
      saved_variable_shared_names.emplace_back(
          std::move(*variable_shared_name));
      VLOG(1) << "Saved a variable with shared_name: " << *variable_shared_name;
    }
  }

  // Exit early if no variables are added.
  if (saved_variable_shared_names.empty()) {
    LOG(INFO) << "No variables are saved to checkpoint";
    return saved_variable_shared_names;
  }

  if (!bundle_writer.Finish().ok()) {
    return bundle_writer.status();
  }

  return saved_variable_shared_names;
}

}  // namespace quantization
}  // namespace tensorflow
