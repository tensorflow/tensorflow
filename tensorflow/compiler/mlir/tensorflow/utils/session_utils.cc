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
#include "tensorflow/compiler/mlir/tensorflow/utils/session_utils.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/compiler/mlir/utils/string_container_utils.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/resource_var.h"

namespace mlir {
namespace tf_saved_model {

std::string GetVariableName(TF::VarHandleOp var_handle_op) {
  // In some cases the shared_name attribute doesn't have the same
  // tensor name in the model, so we first try to use the location
  // then fallback to shared_name attribute.
  if (auto loc = var_handle_op->getLoc().dyn_cast<NameLoc>())
    return loc.getName().str();
  return var_handle_op.getSharedName().str();
}

std::vector<std::string> GetVariableNames(
    llvm::ArrayRef<TF::VarHandleOp> var_handle_ops) {
  std::vector<std::string> names;
  names.reserve(var_handle_ops.size());
  for (auto var_handle_op : var_handle_ops)
    names.push_back(GetVariableName(var_handle_op));
  return names;
}

tensorflow::Var* GetVariableFromSession(mlir::TF::VarHandleOp var_handle_op,
                                        llvm::StringRef device_name,
                                        const tensorflow::DeviceMgr* mgr) {
  tensorflow::Device* device = nullptr;
  if (!mgr || !mgr->LookupDevice(StringRefToView(device_name), &device).ok())
    return nullptr;
  tensorflow::Var* var_ptr = nullptr;
  const auto& container = var_handle_op.getContainer().str();
  auto status = device->resource_manager()->Lookup(
      (container.empty() ? device->resource_manager()->default_container()
                         : container),
      var_handle_op.getSharedName().str(), &var_ptr);
  if (!device || !status.ok()) return nullptr;
  return var_ptr;
}

absl::StatusOr<std::vector<tensorflow::Tensor>> GetResourcesFromSession(
    llvm::ArrayRef<TF::VarHandleOp> var_handle_ops,
    tensorflow::Session* session) {
  if (!session)
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Null Session provided.");
  std::vector<tensorflow::Tensor> resource_tensors;
  auto variable_names = GetVariableNames(var_handle_ops);
  if (variable_names.empty()) return resource_tensors;

  auto status = session->Run({}, variable_names, {}, &resource_tensors);
  if (!status.ok())
    return absl::Status(absl::StatusCode::kInternal, status.message());
  return resource_tensors;
}
}  // namespace tf_saved_model
}  // namespace mlir
