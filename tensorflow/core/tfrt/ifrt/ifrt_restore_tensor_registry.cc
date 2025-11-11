/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/tsl/concurrency/future.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace ifrt_serving {

absl::Status IfrtRestoreTensorRegistry::TryRegister(
    absl::string_view name, RestoredTensorInfo restored_tensor_info) {
  absl::MutexLock lock(mutex_);
  auto& info = restored_tensors_[name];
  if (info.tensor_future.IsValid()) {
    return absl::AlreadyExistsError(
        absl::StrCat("Variable '", name, "' already registered."));
  }
  info = std::move(restored_tensor_info);
  return absl::OkStatus();
}

tsl::Future<tensorflow::Tensor> IfrtRestoreTensorRegistry::GetRestoredTensor(
    absl::string_view name) const {
  absl::MutexLock lock(mutex_);
  auto it = restored_tensors_.find(name);
  if (it == restored_tensors_.end()) {
    return tsl::Future<tensorflow::Tensor>(
        absl::NotFoundError(absl::StrCat("Variable '", name, "' not found.")));
  }

  return it->second.tensor_future;
}

absl::Status IfrtRestoreTensorRegistry::SetUsedByHost(absl::string_view name) {
  absl::MutexLock lock(mutex_);
  auto it = restored_tensors_.find(name);
  if (it == restored_tensors_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Variable '", name, "' not found."));
  }

  it->second.used_by_host = true;
  return absl::OkStatus();
}

void IfrtRestoreTensorRegistry::Freeze() {
  absl::MutexLock lock(mutex_);
  tsl::Future<tensorflow::Tensor> release_tensor_future(
      absl::UnavailableError("Tensor is already release."));
  for (auto& [name, info] : restored_tensors_) {
    if (!info.used_by_host) {
      // Release the tensor by replacing the future containing the tensor with
      // an future containing a status.
      info.tensor_future = release_tensor_future;
    }
  }
}

absl::StatusOr<DtypeAndShape> IfrtRestoreTensorRegistry::GetDtypeAndShape(
    absl::string_view name) const {
  absl::MutexLock lock(mutex_);
  auto it = restored_tensors_.find(name);
  if (it == restored_tensors_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Variable '", name, "' not found."));
  }

  return it->second.dtype_and_shape;
}

}  // namespace ifrt_serving
}  // namespace tensorflow
