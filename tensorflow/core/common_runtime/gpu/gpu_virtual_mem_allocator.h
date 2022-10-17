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

// CUDA virtual memory API is only available in CUDA versions greater than 10.2.

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VIRTUAL_MEM_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VIRTUAL_MEM_ALLOCATOR_H_

#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/tsl/framework/device_id.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"
#endif

#if CUDA_VERSION >= 10020

namespace tensorflow {

// GpuVirtualMemAllocator is a SubAllocator for use with BFCAllocator which
// provides contiguous allocations with each call to Alloc. This is done by
// reserving a large chunk of virtual addresses at construction and then mapping
// physical memory pages to this virtual address range as requested.
//
// This class is not thread-safe.
class GpuVirtualMemAllocator : public SubAllocator {
 public:
  static stream_executor::port::StatusOr<
      std::unique_ptr<GpuVirtualMemAllocator>>
  Create(const std::vector<Visitor>& alloc_visitors,
         const std::vector<Visitor>& free_visitors,
         stream_executor::gpu::GpuContext& gpu_context,
         tsl::PlatformDeviceId gpu_id, size_t virtual_address_space_size,
         const std::vector<tsl::PlatformDeviceId>& peer_gpu_ids);
  ~GpuVirtualMemAllocator() override;

  // Allocates memory at least as large as requested by num_bytes. Will be
  // aligned to the min allocation granularity (typically 2MiB).
  // alignment is ignored by this allocator.
  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override;

  // Frees should only happen at the end of the contiguous memory allocations or
  // else we introduce pointless fragmentation...But, this is supported. If the
  // allocation happens at the end, then the next_alloc_offset_ is moved back,
  // otherwise a hole is created.
  //
  // Holes are not re-used, all allocations continue to come at the end of the
  // next_alloc_offset_. To accommodate this, the virtual_address_space_size
  // should be much larger than the max physical size of the allocator.
  //
  // In practice, since the BFC allocator coalesces adjacent AllocationRegions,
  // this free function should never be invoked.
  void Free(void* ptr, size_t num_bytes) override;

  bool SupportsCoalescing() const override { return true; }

 private:
  GpuVirtualMemAllocator(
      const std::vector<Visitor>& alloc_visitors,
      const std::vector<Visitor>& free_visitors,
      stream_executor::gpu::GpuContext& gpu_context,
      tsl::PlatformDeviceId gpu_id,
      std::vector<stream_executor::gpu::GpuDeviceHandle> access_device_handles,
      stream_executor::gpu::GpuDriver::VmemSpan vmem, size_t granularity);

  stream_executor::gpu::GpuContext& gpu_context_;
  tsl::PlatformDeviceId gpu_id_;

  // Peer access is configured at mmap time so the allocator must be aware of
  // all gpus that may want to read the memory. This list also includes the
  // above gpu_id_ to facilitate the invocation of the GpuDriver::MapMemory
  // function.
  const std::vector<stream_executor::gpu::GpuDeviceHandle> access_gpu_handles_;

  // The virtual memory span held by this allocator.
  stream_executor::gpu::GpuDriver::VmemSpan vmem_;
  // The next offset from the vmem base address that will be allocated. This
  // corresponds to the size of physically pinned memory if holes haven't been
  // created with "free".
  size_t next_alloc_offset_ = 0;

  // Smallest allocation as determined by CUDA.
  const size_t granularity_;

  struct Mapping {
    stream_executor::gpu::GpuDevicePtr va;
    stream_executor::gpu::GpuDriver::GenericMemoryHandle physical;
  };
  // List of mappings, sorted by va.
  std::vector<Mapping> mappings_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuVirtualMemAllocator);
};

}  // namespace tensorflow

#endif  // CUDA_VERSION >= 10200

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VIRTUAL_MEM_ALLOCATOR_H_
