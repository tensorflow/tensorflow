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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_SIMPLE_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_SIMPLE_ALLOCATOR_H_

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class PluggableDeviceSimpleAllocator : public Allocator {
 public:
  explicit PluggableDeviceSimpleAllocator(DeviceMemAllocator* sub_allocator);
  ~PluggableDeviceSimpleAllocator() override = default;

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  bool TracksAllocationSizes() const override { return false; }
  string Name() override { return "Simple allocator"; }
  std::optional<AllocatorStats> GetStats() override;

  AllocatorMemoryType GetMemoryType() const override {
    return sub_allocator_->GetMemoryType();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PluggableDeviceSimpleAllocator);
  std::unique_ptr<SubAllocator> sub_allocator_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_SIMPLE_ALLOCATOR_H_
