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
#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// An allocator that wraps a GPU allocator and adds debugging
// functionality that verifies that users do not write outside their
// allocated memory.
class GPUDebugAllocator : public Allocator {
 public:
  explicit GPUDebugAllocator(Allocator* allocator,
                             PlatformDeviceId platform_device_id);
  ~GPUDebugAllocator() override;
  string Name() override { return "gpu_debug"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  bool TracksAllocationSizes() const override;
  size_t RequestedSize(const void* ptr) const override;
  size_t AllocatedSize(const void* ptr) const override;
  int64 AllocationId(const void* ptr) const override;
  absl::optional<AllocatorStats> GetStats() override;
  bool ClearStats() override;

  // For testing.
  bool CheckHeader(void* ptr);
  bool CheckFooter(void* ptr);

 private:
  Allocator* base_allocator_ = nullptr;  // owned

  se::StreamExecutor* stream_exec_;  // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(GPUDebugAllocator);
};

// An allocator that wraps a GPU allocator and resets the memory on
// allocation and free to 'NaN', helping to identify cases where the
// user forgets to initialize the memory.
class GPUNanResetAllocator : public Allocator {
 public:
  explicit GPUNanResetAllocator(Allocator* allocator,
                                PlatformDeviceId platform_device_id);
  ~GPUNanResetAllocator() override;
  string Name() override { return "gpu_nan_reset"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  size_t RequestedSize(const void* ptr) const override;
  size_t AllocatedSize(const void* ptr) const override;
  absl::optional<AllocatorStats> GetStats() override;
  bool ClearStats() override;

 private:
  Allocator* base_allocator_ = nullptr;  // owned

  se::StreamExecutor* stream_exec_;  // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(GPUNanResetAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEBUG_ALLOCATOR_H_
