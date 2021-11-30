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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"

namespace mlir {
namespace TF {

llvm::StringRef GetDeviceOrEmpty(Operation *op) {
  if (auto device_attr = op->getAttrOfType<StringAttr>("device"))
    return device_attr.getValue();
  return llvm::StringRef();
}

// Returns resource handle value and id for resource op based on attributes. If
// a resource handle is anonymous, a new id is always returned.
ResourceHandleValueAndId GetResourceHandleValueAndIdBase(
    llvm::StringRef container, llvm::StringRef shared_name,
    llvm::StringRef device, Value resource,
    llvm::SmallDenseMap<ResourceHandle, int64_t> &resource_handle_id_map,
    int64_t &next_id) {
  // Always create a new ID for anonymous handle.
  if (IsResourceHandleAnonymous(shared_name)) return {resource, next_id++};

  ResourceHandle handle(container, shared_name, device, /*op=*/nullptr);
  auto emplace_res = resource_handle_id_map.try_emplace(handle, next_id);
  // New ID created, increment next_id.
  if (emplace_res.second) ++next_id;
  return {resource, emplace_res.first->second};
}

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.cc.inc"
}  // namespace TF
}  // namespace mlir
