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

#include "xla/pjrt/gpu/gpu_dma_buf_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/dma_buf.h"
#include "xla/tsl/framework/allocator.h"

namespace xla {

namespace {

bool RangeContains(const stream_executor::DeviceAddressBase& outer,
                   const stream_executor::DeviceAddressBase& inner) {
  if (inner.opaque() == nullptr || inner.size() == 0) {
    return false;
  }
  const uintptr_t outer_start = reinterpret_cast<uintptr_t>(outer.opaque());
  const uintptr_t inner_start = reinterpret_cast<uintptr_t>(inner.opaque());
  if (inner_start < outer_start) {
    return false;
  }
  const uint64_t rel_offset = static_cast<uint64_t>(inner_start - outer_start);
  return rel_offset <= outer.size() &&
         inner.size() <= outer.size() - rel_offset;
}

absl::StatusOr<GpuDmaBufCache::ExportedDmaBuf> ExportUncached(
    const stream_executor::DeviceAddressBase& buffer_range,
    GpuDmaBufCache::ExportFn export_fn) {
  ABSL_ASSIGN_OR_RETURN(stream_executor::gpu::DmaBuf dma_buf,
                   export_fn(buffer_range));
  const uint64_t offset = dma_buf.offset();
  return GpuDmaBufCache::ExportedDmaBuf{
      /*dma_buf=*/std::make_shared<stream_executor::gpu::DmaBuf>(
          std::move(dma_buf)),
      /*offset=*/offset,
  };
}

}  // namespace

tsl::SubAllocator::Visitor GpuDmaBufCache::alloc_visitor() {
  return [self = shared_from_this()](void* ptr, int32_t device_ordinal,
                                     size_t bytes) {
    self->RecordAlloc(ptr, device_ordinal, bytes);
  };
}

tsl::SubAllocator::Visitor GpuDmaBufCache::free_visitor() {
  return [self = shared_from_this()](void* ptr, int32_t device_ordinal,
                                     size_t bytes) {
    self->RecordFree(ptr, device_ordinal, bytes);
  };
}

void GpuDmaBufCache::RecordAlloc(void* ptr, int32_t device_ordinal,
                                 size_t bytes) {
  if (ptr == nullptr || bytes == 0) {
    return;
  }
  absl::MutexLock lock(mu_);
  chunks_[device_ordinal].push_back(Chunk{
      /*range=*/stream_executor::DeviceAddressBase(ptr, bytes),
      /*id=*/++next_id_,
      /*dma_buf=*/nullptr,
  });
}

void GpuDmaBufCache::RecordFree(void* ptr, int32_t device_ordinal,
                                size_t /*bytes*/) {
  if (ptr == nullptr) {
    return;
  }
  // Collect removed dma_bufs and drop them after releasing `mu_`.
  std::vector<std::shared_ptr<stream_executor::gpu::DmaBuf>> to_release;
  {
    absl::MutexLock lock(mu_);
    auto it = chunks_.find(device_ordinal);
    if (it == chunks_.end()) {
      return;
    }
    std::vector<Chunk>& device_chunks = it->second;
    device_chunks.erase(
        std::remove_if(device_chunks.begin(), device_chunks.end(),
                       [&](Chunk& chunk) {
                         if (chunk.range.opaque() == ptr) {
                           if (chunk.dma_buf != nullptr) {
                             to_release.push_back(std::move(chunk.dma_buf));
                           }
                           return true;
                         }
                         return false;
                       }),
        device_chunks.end());
  }
}

absl::StatusOr<GpuDmaBufCache::ExportedDmaBuf> GpuDmaBufCache::GetOrExport(
    int32_t device_ordinal,
    const stream_executor::DeviceAddressBase& buffer_range,
    ExportFn export_fn) {
  const uintptr_t buffer_start =
      reinterpret_cast<uintptr_t>(buffer_range.opaque());

  std::optional<stream_executor::DeviceAddressBase> chunk_range;
  uint64_t chunk_id = 0;
  {
    absl::MutexLock lock(mu_);
    auto it = chunks_.find(device_ordinal);
    if (it != chunks_.end()) {
      for (const Chunk& chunk : it->second) {
        if (RangeContains(chunk.range, buffer_range)) {
          if (chunk.dma_buf != nullptr) {
            const uintptr_t chunk_start =
                reinterpret_cast<uintptr_t>(chunk.range.opaque());
            const uint64_t offset =
                static_cast<uint64_t>(buffer_start - chunk_start) +
                chunk.dma_buf->offset();
            return ExportedDmaBuf{
                /*dma_buf=*/chunk.dma_buf,
                /*offset=*/offset,
            };
          }
          chunk_range = chunk.range;
          chunk_id = chunk.id;
          break;
        }
      }
    }
  }

  // Fallback for untracked / multi-chunk / non-BFC allocations.
  if (!chunk_range.has_value()) {
    return ExportUncached(buffer_range, export_fn);
  }

  // Export the full chunk outside `mu_` to avoid holding locks across driver
  // syscalls.
  ABSL_ASSIGN_OR_RETURN(stream_executor::gpu::DmaBuf exported_chunk,
                   export_fn(*chunk_range));
  auto candidate =
      std::make_shared<stream_executor::gpu::DmaBuf>(std::move(exported_chunk));

  std::shared_ptr<stream_executor::gpu::DmaBuf> result_dma_buf;
  // Declared outside the lock scope so closing a losing fd happens after `mu_`
  // is released.
  std::shared_ptr<stream_executor::gpu::DmaBuf> discarded;
  {
    absl::MutexLock lock(mu_);
    auto it = chunks_.find(device_ordinal);
    if (it != chunks_.end()) {
      for (Chunk& chunk : it->second) {
        if (chunk.id == chunk_id) {
          if (chunk.dma_buf != nullptr) {
            result_dma_buf = chunk.dma_buf;
            discarded = std::move(candidate);
          } else {
            chunk.dma_buf = candidate;
            result_dma_buf = std::move(candidate);
          }
          break;
        }
      }
    }
  }

  if (result_dma_buf != nullptr) {
    const uintptr_t chunk_start =
        reinterpret_cast<uintptr_t>(chunk_range->opaque());
    const uint64_t offset = static_cast<uint64_t>(buffer_start - chunk_start) +
                            result_dma_buf->offset();
    return ExportedDmaBuf{
        /*dma_buf=*/std::move(result_dma_buf),
        /*offset=*/offset,
    };
  }

  // The chunk was freed concurrently; drop the chunk-wide export and fall back
  // to exporting `buffer_range` directly.
  candidate.reset();
  return ExportUncached(buffer_range, export_fn);
}

}  // namespace xla
