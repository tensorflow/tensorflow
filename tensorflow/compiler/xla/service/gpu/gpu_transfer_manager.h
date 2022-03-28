/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_TRANSFER_MANAGER_H_

#include <vector>

#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// An implementation of the XLA GenericTransferManager that
// handles GPU-specific infeed.
class GpuTransferManager : public GenericTransferManager {
 public:
  GpuTransferManager(se::Platform::Id id, unsigned pointer_size);
  ~GpuTransferManager() override {}

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;
  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    MutableBorrowingLiteral literal) override;
  Status ReadDynamicShapes(se::Stream* stream, ShapedBuffer* device_buffer,
                           Shape* device_shape) override;

 private:
  GpuTransferManager(const GpuTransferManager&) = delete;
  GpuTransferManager& operator=(const GpuTransferManager&) = delete;

  // Pool of pinned memory (StreamExecutor::HostMemoryAllocate()) that serves
  // ReadDynamicShapes().  This is a bit of a hack: Callers like TensorFlow
  // already have a full pinned memory allocator, and we could in theory use it
  // here and elsewhere in XLA.  But because GpuTransferManager is a singleton,
  // we can't really access that.
  //
  // To keep things relatively simple, our allocator does the following.
  //
  //  - Allocate one chunk of 128 KiB pinned memory per GPU.
  //  - Divide each chunk into 128-byte buffers.
  //  - During ReadDynamicShapes(), check out one buffer for each dynamic
  //    subshape.  Copy one subshape into one buffer.  If it doesn't fit or
  //    there are no free buffers, fall back to an unpinned memcpy.
  //
  // A 128-byte buffer is large enough to hold a shape of rank 128/sizeof(int32)
  // = 32, which is much larger than we normally see in XLA programs.  A 128 KiB
  // chunk is large enough to hold 128 KiB/128B = 1024 dynamically-shaped
  // buffers, which is also way larger than we should need, even if we're
  // running multiple programs in parallel.
  static constexpr int64_t kPinnedChunkBytes = 128 * 1024;
  static constexpr int64_t kPinnedBufferBytes = 128;

  // One mutex for each GPU's pinned buffers.  These are never null; they just
  // have to be unique_ptr's because Mutex is not copyable or movable.
  std::vector<std::unique_ptr<absl::Mutex>> pinned_buffer_mutexes_;

  // Host buffers for each device.  Each buffer has size kPinnedBufferBytes.
  std::vector<std::vector<void*>> pinned_buffers_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_TRANSFER_MANAGER_H_
