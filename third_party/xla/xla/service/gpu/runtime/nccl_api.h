/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_API_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_API_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// NcclApi
//===----------------------------------------------------------------------===//

// NcclApi hides implementation detail of collective operations built on top of
// NCCL library so that no other parts of XLA should include nccl.h header
// directly (or indirectly).

class NcclApi : public GpuCollectives {
 public:
  virtual ~NcclApi() = default;

  // Returns a default NcclApi for a current process. Can be a real one based on
  // NCCL or a stub if XLA compiled without NCCL or CUDA support.
  static NcclApi* Default();

  // Returns true if XLA is compiled with NCCL support, otherwise returns false.
  // If false, Default() will return a stub implementation.
  static bool HasNcclSupport();

  // Forward declarations of opaque structs corresponding to underlying platform
  // types (also defined as opaque structs).
  struct NcclPersistentPlanAllocator;
  struct NcclRegisteredBuffer;

  // Convenience handles for defining API functions.
  using NcclPersistentPlanAllocatorHandle = NcclPersistentPlanAllocator*;
  using NcclRegisteredBufferHandle = NcclRegisteredBuffer*;

  // Persistent plan allocator allows to pass XLA memory allocator to NCCL to
  // allocate device memory for persistent execution plans for NCCL operations
  // captured into CUDA graphs. It relies on NCCL patch that is not part of
  // upstream NCCL.
  class PersistentPlanAllocator
      : public tsl::ReferenceCounted<PersistentPlanAllocator> {
   public:
    PersistentPlanAllocator(int64_t device_ordinal,
                            se::DeviceMemoryAllocator* allocator,
                            se::Stream* stream);
    ~PersistentPlanAllocator();

    // Allocates new device memory buffer and copies `size` bytes from `src`
    // into it (NCCL persistent execution plan for a collective operation).
    absl::StatusOr<se::DeviceMemoryBase> AllocateAndInitialize(void* src,
                                                               size_t size);
    absl::Status Deallocate(se::DeviceMemoryBase mem);

    NcclPersistentPlanAllocatorHandle handle() const { return handle_; }

   private:
    NcclPersistentPlanAllocatorHandle handle_;  // owned

    int64_t device_ordinal_;
    se::DeviceMemoryAllocator* allocator_;
    se::Stream* stream_;
  };

  // RAII helper to set NCCL persistent plan `allocator` for `comm`.
  class ScopedPersistentPlanAllocator {
   public:
    ScopedPersistentPlanAllocator(
        Communicator* comm,
        tsl::RCReference<PersistentPlanAllocator> allocator);
    ~ScopedPersistentPlanAllocator();

   private:
    Communicator* comm_;
    NcclPersistentPlanAllocatorHandle recover_;
    tsl::RCReference<PersistentPlanAllocator> allocator_;
  };

  // Reduce buffers of length `count` in `send_buff` using `reduction_kind`
  // reduction and leaves identical copies of the result on each `recv_buff`.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce
  virtual absl::Status AllReduce(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 ReductionKind reduction_kind,
                                 Communicator* comm, se::Stream* stream) = 0;

  // Copy data in `send_buff` from the root GPU to the `recv_buff` on
  // all GPUs.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast
  virtual absl::Status Broadcast(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count, size_t root,
                                 Communicator* comm, se::Stream* stream) = 0;
  // Reduce data in `send_buff` from all GPUs using the `reduction_kind`
  // operation and leave the reduced result scattered over the devices so that
  // the `recv_buff` on rank `i` will contain the i-th block of the result.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreducescatter
  virtual absl::Status ReduceScatter(se::DeviceMemoryBase send_buffer,
                                     se::DeviceMemoryBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     Communicator* comm,
                                     se::Stream* stream) = 0;

  // Gather `count` values from all GPUs into recv_buffer, receiving data from
  // rank `i` at offset `i * sendcount`.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather
  virtual absl::Status AllGather(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 Communicator* comm, se::Stream* stream) = 0;

  // Send data from `send_buff` to rank `peer`.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend
  virtual absl::Status Send(se::DeviceMemoryBase send_buffer,
                            PrimitiveType dtype, size_t count, int32_t peer,
                            Communicator* comm, se::Stream* stream) = 0;
  // Send a pointer `ptr` to rank `peer`.
  virtual absl::Status SendPtrToPeer(void* ptr, int32_t peer,
                                     Communicator* comm,
                                     se::Stream* stream) = 0;

  // Receive data from rank `peer` into `recv_buff`.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclrecv
  virtual absl::Status Recv(se::DeviceMemoryBase recv_buffer,
                            PrimitiveType dtype, size_t count, int32_t peer,
                            Communicator* comm, se::Stream* stream) = 0;
  // Receive a pointer from rank `peer` into `ptr`.
  virtual absl::Status RecvPtrFromPeer(void* ptr, int32_t peer,
                                       Communicator* comm,
                                       se::Stream* stream) = 0;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_API_H_
