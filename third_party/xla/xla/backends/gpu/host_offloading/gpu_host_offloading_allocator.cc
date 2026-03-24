/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/host_offloading/gpu_host_offloading_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/core/host_offloading/host_offloading_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"
#include "xla/util.h"
#include "tsl/platform/numbers.h"

namespace xla::gpu {
namespace {
constexpr size_t k1GB = 1024 * 1024 * 1024;

class GpuHostOffloadingAllocator : public HostOffloadingAllocator {
 public:
  explicit GpuHostOffloadingAllocator(
      stream_executor::StreamExecutor* executor);
  ~GpuHostOffloadingAllocator() override = default;

  absl::StatusOr<std::unique_ptr<Buffer>> AllocateTransferBuffer(
      size_t num_bytes) final;

  absl::StatusOr<std::unique_ptr<Buffer>> AllocateStagingBuffer(
      size_t num_bytes) final;

 private:
  tsl::BFCAllocator transfer_allocator_;
};

// Staging buffers are regular host buffers allocated with operator new.
class StagingBuffer : public HostOffloadingAllocator::Buffer {
 public:
  StagingBuffer(void* data, size_t num_bytes);
  ~StagingBuffer() final;

  absl::Span<uint8_t> data() const final;

 private:
  void* data_;
  size_t num_bytes_;
};

StagingBuffer::StagingBuffer(void* data, size_t num_bytes)
    : data_(data), num_bytes_(num_bytes) {}

StagingBuffer::~StagingBuffer() {
  ::operator delete(data_, std::align_val_t{xla::cpu::Align()});
}

absl::Span<uint8_t> StagingBuffer::data() const {
  return absl::MakeSpan(static_cast<uint8_t*>(data_), num_bytes_);
}

// Transfer buffer allocated from the pre-mapped memory chunk.
class TransferBuffer : public HostOffloadingAllocator::Buffer {
 public:
  TransferBuffer(tsl::BFCAllocator* allocator, void* data, size_t num_bytes);
  ~TransferBuffer() final;

  absl::Span<uint8_t> data() const final;

 private:
  tsl::BFCAllocator* allocator_;
  void* data_;
  size_t num_bytes_;
};

TransferBuffer::TransferBuffer(tsl::BFCAllocator* allocator, void* data,
                               size_t num_bytes)
    : allocator_(allocator), data_(data), num_bytes_(num_bytes) {
  VLOG(3) << absl::StreamFormat(
      "Allocated transfer buffer: %p, size: %s", data_,
      tsl::strings::HumanReadableNumBytes(num_bytes_));
}

TransferBuffer::~TransferBuffer() {
  VLOG(3) << absl::StreamFormat(
      "Free transfer buffer: %p, size: %s", data_,
      tsl::strings::HumanReadableNumBytes(num_bytes_));
  allocator_->DeallocateRaw(data_);
}

absl::Span<uint8_t> TransferBuffer::data() const {
  return absl::MakeSpan(static_cast<uint8_t*>(data_), num_bytes_);
}

// StreamExecutor allocates pinned memory that can be used for direct transfers
// between device and host. We rely on BFC allocator to allocate buffers
// requested by the user.
class TransferBufferSubAllocator : public tsl::SubAllocator {
 public:
  explicit TransferBufferSubAllocator(
      stream_executor::StreamExecutor* executor);

  void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) final;
  void Free(void* ptr, size_t num_bytes) final;

  bool SupportsCoalescing() const final { return true; }

 private:
  stream_executor::StreamExecutor* executor_;
  absl::flat_hash_map<void*, std::unique_ptr<stream_executor::MemoryAllocation>>
      allocated_buffers_;
};

TransferBufferSubAllocator::TransferBufferSubAllocator(
    stream_executor::StreamExecutor* executor)
    : tsl::SubAllocator({}, {}), executor_(executor) {}

void* TransferBufferSubAllocator::Alloc(size_t alignment, size_t num_bytes,
                                        size_t* bytes_received) {
  auto allocation = executor_->HostMemoryAllocate(num_bytes);

  if (!allocation.ok()) {
    LOG(ERROR) << "Failed to allocate host memory for transfer buffer: "
               << allocation.status();
    return nullptr;
  }

  void* opaque = allocation.value()->address().opaque();
  allocated_buffers_[allocation.value()->address().opaque()] =
      std::move(*allocation);
  *bytes_received = num_bytes;

  return opaque;
}

void TransferBufferSubAllocator::Free(void* ptr, size_t num_bytes) {
  auto it = allocated_buffers_.find(ptr);
  CHECK(it != allocated_buffers_.end()) << "Buffer not found: " << ptr;
  allocated_buffers_.erase(it);
}

GpuHostOffloadingAllocator::GpuHostOffloadingAllocator(
    stream_executor::StreamExecutor* executor)
    : transfer_allocator_(
          std::make_unique<TransferBufferSubAllocator>(executor),
          /*total_memory=*/256 * k1GB, /*name=*/"HostOffloading",
          tsl::BFCAllocator::Options{.allow_retry_on_failure = false}) {}

absl::StatusOr<std::unique_ptr<HostOffloadingAllocator::Buffer>>
GpuHostOffloadingAllocator::AllocateTransferBuffer(size_t num_bytes) {
  if (void* data = transfer_allocator_.AllocateRaw(cpu::Align(), num_bytes)) {
    return std::make_unique<TransferBuffer>(&transfer_allocator_, data,
                                            num_bytes);
  }

  return Internal(
      "Failed to allocate %d bytes for host offloading transfer buffer",
      num_bytes);
}

absl::StatusOr<std::unique_ptr<HostOffloadingAllocator::Buffer>>
GpuHostOffloadingAllocator::AllocateStagingBuffer(size_t num_bytes) {
  return std::make_unique<StagingBuffer>(
      ::operator new(num_bytes, std::align_val_t{xla::cpu::Align()}),
      num_bytes);
}

}  // namespace

std::unique_ptr<HostOffloadingAllocator> CreateGpuHostOffloadingAllocator(
    stream_executor::StreamExecutor* executor) {
  return std::make_unique<GpuHostOffloadingAllocator>(executor);
}
}  // namespace xla::gpu
