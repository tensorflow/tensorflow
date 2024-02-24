/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "xla/stream_executor/stream_executor.h"
#include "tsl/framework/allocator.h"
#include "tsl/framework/device_id.h"
#include "tsl/platform/macros.h"

namespace tensorflow {

// An allocator that wraps a GPU allocator and adds debugging
// functionality that verifies that users do not write outside their
// allocated memory.
class GPUDebugAllocator : public tsl::Allocator {
 public:
  explicit GPUDebugAllocator(tsl::Allocator* allocator,
                             tsl::PlatformDeviceId platform_device_id);
  ~GPUDebugAllocator() override;
  std::string Name() override { return "gpu_debug"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  bool TracksAllocationSizes() const override;
  size_t RequestedSize(const void* ptr) const override;
  size_t AllocatedSize(const void* ptr) const override;
  int64_t AllocationId(const void* ptr) const override;
  std::optional<tsl::AllocatorStats> GetStats() override;
  bool ClearStats() override;

  // For testing.
  bool CheckHeader(void* ptr);
  bool CheckFooter(void* ptr);

 private:
  tsl::Allocator* base_allocator_ = nullptr;  // owned

  se::StreamExecutor* stream_exec_;  // Not owned.

  GPUDebugAllocator(const GPUDebugAllocator&) = delete;
  void operator=(const GPUDebugAllocator&) = delete;
};

// An allocator that wraps a GPU allocator and resets the memory on
// allocation and free to 'NaN', helping to identify cases where the
// user forgets to initialize the memory.
class GPUNanResetAllocator : public tsl::Allocator {
 public:
  explicit GPUNanResetAllocator(tsl::Allocator* allocator,
                                tsl::PlatformDeviceId platform_device_id);
  ~GPUNanResetAllocator() override;
  std::string Name() override { return "gpu_nan_reset"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  size_t RequestedSize(const void* ptr) const override;
  size_t AllocatedSize(const void* ptr) const override;
  std::optional<tsl::AllocatorStats> GetStats() override;
  bool ClearStats() override;

  tsl::AllocatorMemoryType GetMemoryType() const override {
    return base_allocator_->GetMemoryType();
  }

 private:
  tsl::Allocator* base_allocator_ = nullptr;  // owned

  se::StreamExecutor* stream_exec_;  // Not owned.

  GPUNanResetAllocator(const GPUNanResetAllocator&) = delete;
  void operator=(const GPUNanResetAllocator&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_
