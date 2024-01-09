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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_DEVICE_VARIABLES_TABLE_H_
#define TENSORFLOW_CORE_TFRT_UTILS_DEVICE_VARIABLES_TABLE_H_

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime

namespace tfrt {

// A variable table that keeps track of the device copies of host tensors.
// The same variable can have multiple copies on devices (e.g., on different TPU
// cores), and hence they are differenticated via `copy_index`.
// The table maps from <host tensor, copy_index> to device tensor.
template <typename HostTensorType, typename DeviceTensorType>
class DeviceVariablesTable {
 public:
  virtual ~DeviceVariablesTable() { ClearDeviceVariablesTable(); }

  void AddOrUpdateDeviceVariable(
      const HostTensorType& host_tensor, int copy_index,
      AsyncValueRef<DeviceTensorType> device_tensor) {
    absl::MutexLock lock(&device_variables_mu_);
    device_variables_table_.insert_or_assign(
        std::make_pair(GetHostTensorDataPtr(host_tensor), copy_index),
        std::move(device_tensor));
  }

  AsyncValueRef<DeviceTensorType> GetDeviceVariable(
      const HostTensorType& host_tensor, int copy_index) {
    absl::ReaderMutexLock lock(&device_variables_mu_);
    auto it = device_variables_table_.find(
        std::make_pair(GetHostTensorDataPtr(host_tensor), copy_index));
    return it != device_variables_table_.end()
               ? it->second.CopyRef()
               : AsyncValueRef<DeviceTensorType>();
  }

  AsyncValueRef<DeviceTensorType> GetOrAddDeviceVariable(
      const HostTensorType& host_tensor, int copy_index,
      llvm::unique_function<void(AsyncValueRef<DeviceTensorType>)> creator) {
    absl::ReleasableMutexLock lock(&device_variables_mu_);
    auto it = device_variables_table_.find(
        std::make_pair(GetHostTensorDataPtr(host_tensor), copy_index));
    if (it != device_variables_table_.end()) return it->second.CopyRef();

    auto device_tensor = MakeUnconstructedAsyncValueRef<DeviceTensorType>();
    device_variables_table_.insert(
        {std::make_pair(GetHostTensorDataPtr(host_tensor), copy_index),
         device_tensor.CopyRef()});
    lock.Release();
    creator(device_tensor.CopyRef());
    return device_tensor;
  }

  void ClearDeviceVariablesTable() {
    absl::MutexLock lock(&device_variables_mu_);
    device_variables_table_.clear();
  }

  int size() {
    absl::ReaderMutexLock lock(&device_variables_mu_);
    return device_variables_table_.size();
  }

 protected:
  // Get the host tensor data pointer, which is used as a part of the table key.
  virtual const void* GetHostTensorDataPtr(
      const HostTensorType& host_tensor) = 0;

 private:
  absl::Mutex device_variables_mu_;

  // A map from <host tensor data, copy_index> to device tensor.
  absl::flat_hash_map<std::pair<const void*, int>,
                      AsyncValueRef<DeviceTensorType>>
      device_variables_table_ ABSL_GUARDED_BY(device_variables_mu_);
};

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_UTILS_DEVICE_VARIABLES_TABLE_H_
