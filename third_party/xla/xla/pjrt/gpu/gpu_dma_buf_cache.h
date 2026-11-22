/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_GPU_GPU_DMA_BUF_CACHE_H_
#define XLA_PJRT_GPU_GPU_DMA_BUF_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/dma_buf.h"
#include "xla/tsl/framework/allocator.h"

namespace xla {

// Caches exported Linux dma-buf file descriptors for GPU suballocator chunks.
//
// Chunks are recorded via `alloc_visitor()` and invalidated via
// `free_visitor()`. When `GetOrExport()` is called for a buffer contained in a
// tracked chunk, the entire chunk is lazily exported once and shared across all
// buffers in that chunk. Buffers outside any single tracked chunk fall back to
// an uncached export owned by the returned `ExportedDmaBuf`. Instances must be
// owned by `std::shared_ptr` before calling `alloc_visitor()` or
// `free_visitor()`.
class GpuDmaBufCache : public std::enable_shared_from_this<GpuDmaBufCache> {
 public:
  struct ExportedDmaBuf {
    // Shared ownership of the underlying dma-buf handle.
    std::shared_ptr<stream_executor::gpu::DmaBuf> dma_buf;

    // Byte offset of the requested buffer from the start of `dma_buf`.
    uint64_t offset = 0;
  };

  using ExportFn =
      absl::FunctionRef<absl::StatusOr<stream_executor::gpu::DmaBuf>(
          const stream_executor::DeviceAddressBase&)>;

  GpuDmaBufCache() = default;
  GpuDmaBufCache(const GpuDmaBufCache&) = delete;
  GpuDmaBufCache& operator=(const GpuDmaBufCache&) = delete;

  // Returns suballocator visitors that track chunk lifetimes.
  tsl::SubAllocator::Visitor alloc_visitor();
  tsl::SubAllocator::Visitor free_visitor();

  // Returns a cached dma-buf covering `buffer_range` (if inside a tracked
  // chunk), or falls back to an uncached export via `export_fn`.
  absl::StatusOr<ExportedDmaBuf> GetOrExport(
      int32_t device_ordinal,
      const stream_executor::DeviceAddressBase& buffer_range,
      ExportFn export_fn);

 private:
  struct Chunk {
    // Full suballocator chunk address range.
    stream_executor::DeviceAddressBase range;

    // Unique identifier for this chunk allocation instance.
    uint64_t id = 0;

    // Lazily exported dma-buf for `range`, or nullptr if not yet exported.
    std::shared_ptr<stream_executor::gpu::DmaBuf> dma_buf;
  };

  void RecordAlloc(void* ptr, int32_t device_ordinal, size_t bytes);
  void RecordFree(void* ptr, int32_t device_ordinal, size_t bytes);

  // Protects `chunks_` and `next_id_`.
  absl::Mutex mu_;

  // Monotonically increasing chunk allocation ID.
  uint64_t next_id_ ABSL_GUARDED_BY(mu_) = 0;

  // Tracked suballocator chunks keyed by device ordinal. A flat vector with
  // linear scan is intentional: default BFC preallocates a single chunk per
  // device, dynamic BFC grows chunks exponentially (tens of entries at most),
  // and non-BFC allocators (e.g., VMM) do not register alloc visitors and fall
  // back to uncached exports.
  absl::flat_hash_map<int32_t, std::vector<Chunk>> chunks_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_GPU_DMA_BUF_CACHE_H_
