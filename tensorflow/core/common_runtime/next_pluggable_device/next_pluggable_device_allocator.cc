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

#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_allocator.h"

#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"

namespace tensorflow {

NextPluggableDeviceAllocator::NextPluggableDeviceAllocator(int device_ordinal)
    : device_ordinal_(device_ordinal) {
  api_ = TfnpdApi();
  device_allocator_ = api_->TFNPD_DeviceAllocatorCreate(device_ordinal_);
  TF_StringView device_allocator_name =
      api_->TFNPD_DeviceAllocatorName(device_allocator_);
  device_allocator_name_ = device_allocator_name.data;
  allocates_opaque_handle_ =
      api_->TFNPD_DeviceAllocatorAllocatesOpaqueHandle(device_allocator_);
}

NextPluggableDeviceAllocator::~NextPluggableDeviceAllocator() {
  api_->TFNPD_DeviceAllocatorDelete(device_allocator_);
}

void* NextPluggableDeviceAllocator::AllocateRaw(size_t alignment,
                                                size_t num_bytes) {
  return api_->TFNPD_DeviceAllocateRaw(device_allocator_, alignment, num_bytes);
}

void NextPluggableDeviceAllocator::DeallocateRaw(void* ptr) {
  api_->TFNPD_DeviceDeallocateRaw(device_allocator_, ptr);
}

}  // namespace tensorflow
