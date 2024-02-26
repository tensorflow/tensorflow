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

#ifndef XLA_SERVICE_GPU_GPU_TRANSFER_MANAGER_H_
#define XLA_SERVICE_GPU_GPU_TRANSFER_MANAGER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/literal.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// An implementation of the XLA GenericTransferManager that
// handles GPU-specific infeed.
class GpuTransferManager : public GenericTransferManager {
 public:
  GpuTransferManager(se::Platform::Id id, unsigned pointer_size);

  absl::Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                       const LiteralSlice& literal) override;
  absl::Status TransferLiteralFromOutfeed(
      se::StreamExecutor* executor, MutableBorrowingLiteral literal) override;
  absl::Status ReadDynamicShapes(se::Stream* stream,
                                 const ShapedBuffer* device_buffer,
                                 Shape* device_shape) override;

 private:
  // We use a fixed-size staging buffers and split transfer into multiple
  // operations if literal does not fit into it.
  static constexpr int64_t kStagingBufferSize = 128 * 1024 * 1024;

  // We use host memory allocation (pinned host memory) as a staging buffer for
  // transfering literals to and from device. We keep a separate staging
  // allocation per device so we don't need to do cross-device synchronization.
  // All transfers to and from a device are ordered via stream dependencies.
  struct StagingBuffer {
    StagingBuffer(std::unique_ptr<se::MemoryAllocation> allocation,
                  std::unique_ptr<se::Event> transfer_completed);

    absl::Mutex mutex;
    std::unique_ptr<se::MemoryAllocation> allocation ABSL_GUARDED_BY(mutex);
    std::unique_ptr<se::Event> transfer_completed ABSL_GUARDED_BY(mutex);
  };

  GpuTransferManager(const GpuTransferManager&) = delete;
  GpuTransferManager& operator=(const GpuTransferManager&) = delete;

  bool PackSubbyteTypes() const override { return true; }

  // Returns or creates the staging buffer for the given executor.
  absl::StatusOr<StagingBuffer*> GetOrCreateStagingBuffer(
      se::StreamExecutor* executor);

  absl::Status TransferBufferFromDevice(se::Stream* stream,
                                        const se::DeviceMemoryBase& source,
                                        int64_t size,
                                        void* destination) override;

  absl::Status TransferBufferToDevice(
      se::Stream* stream, int64_t size, const void* source,
      se::DeviceMemoryBase* destination) override;

  // TODO(ezhulenev): Unify this with staged buffers for transfering literals.

  // This class keeps a pool of pinned memory
  // (StreamExecutor::HostMemoryAllocate()) that serves ReadDynamicShapes().
  // This is a bit of a hack: Callers like TensorFlow already have a full pinned
  // memory allocator, and we could in theory use it here and elsewhere in XLA.
  // But because GpuTransferManager is a singleton, we can't really access that.
  //
  // To keep things relatively simple, our allocator does the following.
  //
  //  - Allocate one chunk of 128 KiB pinned memory.
  //  - Divide each chunk into 128-byte buffers.
  //  - During ReadDynamicShapes(), "check out" one buffer for each dynamic
  //    subshape.  Copy one subshape into one buffer.  If it doesn't fit or
  //    there are no free buffers, fall back to an unpinned memcpy.
  //
  // A 128-byte buffer is large enough to hold a shape of rank 128/sizeof(int32)
  // = 32, which is much larger than we normally see in XLA programs.  A 128 KiB
  // chunk is large enough to hold 128 KiB/128B = 1024 dynamically-shaped
  // buffers, which is also way larger than we should need, even if we're
  // running multiple programs in parallel.
  //
  // This pool is lazily initialized on first use.  It would be better to
  // initialize it in the constructor, but doing so poses a challenge in the
  // presence of multiple GPUs.  We need a StreamExecutor in order to allocate
  // pinned memory.  We don't care which GPU's SE we use, because SE allocates
  // pinned memory with the PORTABLE flag, making it available to all CUDA
  // contexts.  But we do need to avoid calling platform->ExecutorForDevice for
  // a device that we're not "supposed" to use, because this will create a CUDA
  // context for that device, consuming significant resources on the GPU,
  // b/228207839.
  //
  // Lazy initialization works around this, because at that point we have a
  // stream, and therefore we have an already-initialized StreamExecutor.
  absl::Status EnsurePinnedBuffersAllocated(se::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static constexpr int64_t kPinnedChunkBytes = 128 * 1024;
  static constexpr int64_t kPinnedBufferBytes = 128;

  absl::Mutex mu_;

  // The StreamExecutor on which our pinned memory was allocated.  We use this
  // when freeing the pinned memory.  Lazily initialized.
  se::StreamExecutor* pinned_chunk_se_ ABSL_GUARDED_BY(mu_) = nullptr;

  // Chunk of pinned memory of size kPinnedChunkBytes.  The pointers in
  // pinned_buffers_ point into this chunk.  Lazily initialized.
  std::unique_ptr<se::MemoryAllocation> pinned_chunk_ ABSL_GUARDED_BY(mu_);

  // Host buffers for reading dynamic shapes.  Each buffer has size
  // kPinnedBufferBytes.  Lazily initialized.
  std::vector<void*> pinned_buffers_ ABSL_GUARDED_BY(mu_);

  // Staging buffers allocated for transfers to and from device.
  absl::Mutex mutex_;
  absl::node_hash_map<se::StreamExecutor*, StagingBuffer> staging_buffers_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_TRANSFER_MANAGER_H_
