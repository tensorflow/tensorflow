/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_transfer_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "llvm/IR/DataLayout.h"
#include "xla/literal.h"
#include "xla/service/compiler.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/service/gpu/infeed_manager.h"
#include "xla/service/gpu/outfeed_manager.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

// TODO(b/30467474) Once GPU infeed implementation settles, consider
// folding back the cpu and gpu infeed implementations into a generic
// one if possible.
GpuTransferManager::GpuTransferManager(se::Platform::Id id,
                                       unsigned pointer_size)
    : GenericTransferManager(id, pointer_size) {}

absl::Status GpuTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  return gpu::GetOrCreateInfeedManager(executor)->TransferLiteralToInfeed(
      executor, literal);
}

absl::Status GpuTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, MutableBorrowingLiteral literal) {
  return gpu::GetOrCreateOutfeedManager(executor)->TransferLiteralFromOutfeed(
      executor, literal);
}

absl::Status GpuTransferManager::EnsurePinnedBuffersAllocated(
    se::StreamExecutor* executor) {
  if (pinned_chunk_ != nullptr) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(pinned_chunk_,
                      executor->HostMemoryAllocate(kPinnedChunkBytes));
  pinned_chunk_se_ = executor;

  static_assert(kPinnedChunkBytes % kPinnedBufferBytes == 0,
                "assumption of loop below");
  char* base = reinterpret_cast<char*>(pinned_chunk_->opaque());
  for (char* buf = base; buf < base + kPinnedChunkBytes;
       buf += kPinnedBufferBytes) {
    pinned_buffers_.push_back(buf);
  }

  return absl::OkStatus();
}

absl::Status GpuTransferManager::ReadDynamicShapes(
    se::Stream* stream, const ShapedBuffer* device_buffer,
    Shape* device_shape) {
  DCHECK(device_shape->is_dynamic());
  Shape original_device_shape = *device_shape;

  TF_ASSIGN_OR_RETURN(auto compiler,
                      Compiler::GetForPlatform(stream->parent()->platform()));
  auto shape_size_fn = compiler->ShapeSizeBytesFunction();

  // First, figure out which parts of `device_shape` are dynamic and where the
  // dynamic shapes live in GPU memory.  We'll copy the bytes at the
  // DeviceMemoryBase into the Shape*'s dimensions.
  std::vector<std::pair<se::DeviceMemoryBase, Shape*>> copies;

  TF_RETURN_IF_ERROR(device_buffer->buffers().ForEachElementWithStatus(
      [&](const ShapeIndex& index,
          const se::DeviceMemoryBase& buffer) -> absl::Status {
        const Shape& buffer_shape =
            ShapeUtil::GetSubshape(*device_shape, index);
        if (buffer_shape.IsTuple()) {
          return absl::OkStatus();
        }
        Shape& device_sub_shape =
            *ShapeUtil::GetMutableSubshape(device_shape, index);
        if (device_sub_shape.is_static()) {
          return absl::OkStatus();
        }

        // Read the dynamic shape metadata from the device stream.  The dynamic
        // shape itself is stored at the end of the buffer.
        Shape buffer_shape_static = ShapeUtil::MakeStaticShape(buffer_shape);
        const int64_t offset = shape_size_fn(buffer_shape_static);
        int64_t metadata_size = shape_size_fn(buffer_shape) - offset;
        if (metadata_size == 0) {
          return InvalidArgument("Dynamic shape metadata size should not be 0");
        }

        auto buffer_8 = se::DeviceMemory<uint8_t>(buffer);
        auto metadata_buffer = buffer_8.GetSlice(offset, metadata_size);
        copies.push_back(std::make_pair(metadata_buffer, &device_sub_shape));

        return absl::OkStatus();
      }));

  // Check out pinned memory for each buffer we want to copy.  If there aren't
  // enough pinned buffers available, or if one of our buffers is so big it
  // doesn't fit, allocate an entry for it in fallback_buffers.
  std::vector<int32_t*> h2d_memcpy_dsts;
  std::vector<void*> checked_out_buffers;
  std::vector<std::unique_ptr<char[]>> fallback_buffers;

  // Return checked-out buffers at the end of this function.
  absl::Cleanup cleanup = [&] {
    absl::MutexLock lock(&mu_);
    pinned_buffers_.insert(pinned_buffers_.end(), checked_out_buffers.begin(),
                           checked_out_buffers.end());
  };

  {
    absl::MutexLock lock(&mu_);
    TF_RETURN_IF_ERROR(EnsurePinnedBuffersAllocated(stream->parent()));

    for (const auto& src_dst : copies) {
      se::DeviceMemoryBase src = src_dst.first;
      if (!pinned_buffers_.empty() && src.size() <= kPinnedBufferBytes) {
        void* buf = pinned_buffers_.back();
        pinned_buffers_.pop_back();
        checked_out_buffers.push_back(buf);
        h2d_memcpy_dsts.push_back(reinterpret_cast<int32_t*>(buf));
      } else {
        LOG_FIRST_N(WARNING, 10)
            << "Unable to copy dynamic shape buffer of size " << src.size()
            << " to host using pinned memory.  Falling back to unpinned "
               "memory, which may be slow.";
        fallback_buffers.push_back(std::make_unique<char[]>(src.size()));
        h2d_memcpy_dsts.push_back(
            reinterpret_cast<int32_t*>(fallback_buffers.back().get()));
      }
    }
  }

  // Copy into the h2d_memcpy_dsts.
  for (int i = 0; i < copies.size(); i++) {
    se::DeviceMemoryBase src = copies[i].first;
    void* dst = h2d_memcpy_dsts[i];
    TF_RETURN_IF_ERROR(stream->Memcpy(dst, src, src.size()));
  }

  // Wait for all the async copies to complete, then write into device_shape.
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  for (int i = 0; i < copies.size(); i++) {
    Shape* dst_shape = copies[i].second;
    int32_t* dst = h2d_memcpy_dsts[i];
    for (int64_t j = 0; j < dst_shape->rank(); j++) {
      dst_shape->mutable_dimensions()[j] = dst[j];
    }
  }

  device_shape->clear_dynamic_dimensions();
  TF_RET_CHECK(ShapeUtil::DynamicShapeIsCompatible(*device_shape,
                                                   original_device_shape));
  return absl::OkStatus();
}

// Chunks `size` into chunks of `chunk_size` and calls `callback` for each.
static absl::Status ForEachChunk(
    size_t size, size_t chunk_size,
    absl::FunctionRef<absl::Status(size_t chunk_offset, size_t chunk_size)>
        callback) {
  int64_t num_chunks = CeilOfRatio(size, chunk_size);

  for (int64_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
    TF_RETURN_IF_ERROR(callback(
        /*chunk_offset=*/chunk_index * chunk_size,
        /*chunk_size=*/std::min(chunk_size, size - chunk_index * chunk_size)));
  }
  return absl::OkStatus();
}

absl::Status GpuTransferManager::TransferBufferFromDevice(
    se::Stream* stream, const se::DeviceMemoryBase& source, int64_t size,
    void* destination) {
  if (source.size() < size) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Source allocation on device not large enough for data transfer: "
        "%d < %d",
        source.size(), size));
  }

  VLOG(5) << "Transfer buffer from device: size="
          << tsl::strings::HumanReadableNumBytes(size);

  TF_ASSIGN_OR_RETURN(auto staging_buffer,
                      GetOrCreateStagingBuffer(stream->parent()));

  absl::MutexLock lock(&staging_buffer->mutex);
  void* staging = staging_buffer->allocation->opaque();

  // Transfer chunk of data from device to destination via staging buffer.
  auto transfer_chunk = [&](size_t chunk_offset,
                            size_t chunk_size) -> absl::Status {
    VLOG(5) << "Transfer buffer chunk from device: offset=" << chunk_offset
            << " size=" << tsl::strings::HumanReadableNumBytes(chunk_size);

    se::DeviceMemoryBase chunk = source.GetByteSlice(chunk_offset, chunk_size);
    TF_RETURN_IF_ERROR(stream->Memcpy(staging, chunk, chunk_size));

    void* dst = reinterpret_cast<char*>(destination) + chunk_offset;
    return stream->DoHostCallback(
        [=] { std::memcpy(dst, staging, chunk_size); });
  };

  TF_RETURN_IF_ERROR(stream->WaitFor(staging_buffer->transfer_completed.get()));
  TF_RETURN_IF_ERROR(ForEachChunk(size, kStagingBufferSize, transfer_chunk));
  TF_RETURN_IF_ERROR(
      stream->RecordEvent(staging_buffer->transfer_completed.get()));

  return absl::OkStatus();
}

absl::Status GpuTransferManager::TransferBufferToDevice(
    se::Stream* stream, int64_t size, const void* source,
    se::DeviceMemoryBase* destination) {
  if (destination->size() < size) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Destination allocation on device not large enough for data transfer: "
        "%d < %d",
        destination->size(), size));
  }

  VLOG(5) << "Transfer buffer to device: size="
          << tsl::strings::HumanReadableNumBytes(size);

  TF_ASSIGN_OR_RETURN(auto staging_buffer,
                      GetOrCreateStagingBuffer(stream->parent()));

  absl::MutexLock lock(&staging_buffer->mutex);
  void* staging = staging_buffer->allocation->opaque();

  // Transfer chunk of data from device to destination.
  auto transfer_chunk = [&](size_t chunk_offset, size_t chunk_size) {
    VLOG(5) << "Transfer buffer chunk to device: offset=" << chunk_offset
            << " size=" << tsl::strings::HumanReadableNumBytes(chunk_size);

    const void* src = reinterpret_cast<const char*>(source) + chunk_offset;
    TF_RETURN_IF_ERROR(
        stream->DoHostCallback([=] { std::memcpy(staging, src, chunk_size); }));

    auto chunk = destination->GetByteSlice(chunk_offset, chunk_size);
    return stream->Memcpy(&chunk, staging, chunk_size);
  };

  TF_RETURN_IF_ERROR(stream->WaitFor(staging_buffer->transfer_completed.get()));
  TF_RETURN_IF_ERROR(ForEachChunk(size, kStagingBufferSize, transfer_chunk));
  TF_RETURN_IF_ERROR(
      stream->RecordEvent(staging_buffer->transfer_completed.get()));

  return absl::OkStatus();
}

GpuTransferManager::StagingBuffer::StagingBuffer(
    std::unique_ptr<se::MemoryAllocation> allocation,
    std::unique_ptr<se::Event> transfer_completed)
    : allocation(std::move(allocation)),
      transfer_completed(std::move(transfer_completed)) {}

absl::StatusOr<GpuTransferManager::StagingBuffer*>
GpuTransferManager::GetOrCreateStagingBuffer(se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);
  if (auto it = staging_buffers_.find(executor); it != staging_buffers_.end()) {
    return &it->second;
  }

  VLOG(3) << absl::StreamFormat(
      "Allocate staging buffer of %s for executor %p (device_ordinal=%d)",
      tsl::strings::HumanReadableNumBytes(kStagingBufferSize), executor,
      executor->device_ordinal());

  TF_ASSIGN_OR_RETURN(auto staging_buffer,
                      executor->HostMemoryAllocate(kStagingBufferSize));

  auto transfer_completed = std::make_unique<se::Event>(executor);
  if (!transfer_completed->Init()) {
    return absl::InternalError("Failed to initialize transfer completed event");
  }

  auto emplaced = staging_buffers_.try_emplace(
      executor, std::move(staging_buffer), std::move(transfer_completed));
  return &emplaced.first->second;
}

}  // namespace gpu
}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreateNVPTXTransferManager() {
  return std::make_unique<xla::gpu::GpuTransferManager>(
      /*id=*/stream_executor::cuda::kCudaPlatformId,
      /*pointer_size=*/llvm::DataLayout(xla::gpu::nvptx::DataLayout())
          .getPointerSize(0 /* default address space */));
}

static std::unique_ptr<xla::TransferManager> CreateAMDGPUTransferManager() {
  return std::make_unique<xla::gpu::GpuTransferManager>(
      /*id=*/stream_executor::rocm::kROCmPlatformId,
      /*pointer_size=*/llvm::DataLayout(xla::gpu::amdgpu::DataLayout())
          .getPointerSize(0 /* default address space */));
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      stream_executor::cuda::kCudaPlatformId, &CreateNVPTXTransferManager);
  xla::TransferManager::RegisterTransferManager(
      stream_executor::rocm::kROCmPlatformId, &CreateAMDGPUTransferManager);
  return true;
}

static bool module_initialized = InitModule();
