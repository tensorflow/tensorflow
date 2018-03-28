/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_tensor_info.h"

namespace tensorflow {

const XlaTensorInfo* XlaTensorInfoManager::GetTensorInfo(
    const void* device_ptr) const {
  mutex_lock lock(lock_);
  auto iterator = tensor_infos_.find(device_ptr);
  return (iterator == tensor_infos_.end()) ? nullptr
                                           : tensor_infos_.at(device_ptr).get();
}

XlaTensorInfo* XlaTensorInfoManager::GetOrCreateTensorInfo(
    const void* device_ptr) {
  mutex_lock lock(lock_);
  auto iterator = tensor_infos_.find(device_ptr);
  if (iterator != tensor_infos_.end()) {
    return iterator->second.get();
  }
  auto iterator_and_inserted =
      tensor_infos_.emplace(device_ptr, MakeUnique<XlaTensorInfo>());
  CHECK(iterator_and_inserted.second);
  return iterator_and_inserted.first->second.get();
}

const XlaTensorInfo* XlaTensorInfoManager::GetTensorInfo(const Tensor& tensor) {
  return GetTensorInfo(tensor.tensor_data().data());
}

XlaTensorInfo* XlaTensorInfoManager::GetOrCreateTensorInfo(
    const Tensor& tensor) {
  return GetOrCreateTensorInfo(tensor.tensor_data().data());
}

void XlaTensorInfoManager::DeallocateRaw(void* ptr) {
  wrapped()->DeallocateRaw(ptr);
  mutex_lock lock(lock_);
  tensor_infos_.erase(ptr);
}

}  // namespace tensorflow
