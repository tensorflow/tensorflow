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
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/platform/logging.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// NcclApi
//===----------------------------------------------------------------------===//

// NcclApi hides implementation detail of collective operations built on top of
// NCCL library so that no other parts of XLA should include nccl.h header
// directly (or indirectly).

class NcclApi {
 public:
  virtual ~NcclApi() = default;

  // Communicator configuration.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig
  struct Config {
    bool split_share = false;
    int64_t max_nchannels = 0;
  };

  // Returns a default NcclApi for a current process. Can be a real one based on
  // NCCL or a stub if XLA compiled without NCCL or CUDA support.
  static NcclApi* Default();

  // Forward declarations of opaque structs corresponding to underlying platform
  // types (also defined as opaque structs).
  struct NcclComm;
  struct NcclPersistentPlanAllocator;
  struct NcclRegisteredBuffer;

  // Convenience handles for defining API functions.
  using NcclCommHandle = NcclComm*;
  using NcclPersistentPlanAllocatorHandle = NcclPersistentPlanAllocator*;
  using NcclRegisteredBufferHandle = NcclRegisteredBuffer*;

  // RAII handle for NCCL communicator.
  struct NcclCommDeleter {
    void operator()(NcclCommHandle comm) {
      if (auto destroyed = api->CommDestroy(comm); !destroyed.ok())
        LOG(ERROR) << "Failed to destroy communicator: " << destroyed;
    }
    NcclApi* api;
  };

  using OwnedNcclComm = std::unique_ptr<NcclComm, NcclCommDeleter>;

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
        NcclCommHandle comm,
        tsl::RCReference<PersistentPlanAllocator> allocator);
    ~ScopedPersistentPlanAllocator();

   private:
    NcclCommHandle comm_;
    NcclPersistentPlanAllocatorHandle recover_;
    tsl::RCReference<PersistentPlanAllocator> allocator_;
  };

  struct DeviceRank {
    DeviceRank(se::StreamExecutor* device, int32_t rank)
        : device(device), rank(rank) {}

    se::StreamExecutor* device;
    int32_t rank;
  };

  // Returns a slice of device memory `buff` containing `count` values of data
  // type `dtype` starting from `offset`.
  static se::DeviceMemoryBase Slice(se::DeviceMemoryBase buff,
                                    PrimitiveType dtype, size_t offset,
                                    size_t count) {
    size_t multiplier = ShapeUtil::ByteSizeOfPrimitiveType(dtype);
    return buff.GetByteSlice(offset * multiplier, count * multiplier);
  }

  // Creates a new unique clique id.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclgetuniqueid
  virtual absl::StatusOr<NcclCliqueId> GetUniqueId() = 0;

  // Creates new communicators for given devices.
  //
  // This API doesn't have a corresponding API in NCCL and implemented as
  // multiple calls to ncclCommInitRank within a single group.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitrank
  virtual absl::StatusOr<std::vector<OwnedNcclComm>> CommInitRanks(
      int32_t nranks, const NcclCliqueId& clique_id,
      absl::Span<const DeviceRank> ranks, const Config& config) = 0;

  // Creates new communicators by splitting `comms`.
  //
  // This API doesn't have a corresponding API in NCCL and implemented as
  // multiple calls to ncclCommSplit within a single group.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommsplit
  virtual absl::StatusOr<std::vector<OwnedNcclComm>> CommSplit(
      absl::Span<const NcclCommHandle> comms, int32_t color,
      absl::Span<const int32_t> keys, std::optional<Config> config) = 0;

  // Abort any uncompleted operations and destroys the communicator. Frees
  // resources that are allocated to a communicator object comm.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommabort
  virtual absl::Status CommAbort(NcclCommHandle comm) = 0;

  // Finalize a communicator object comm.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommdestroy
  virtual absl::Status CommFinalize(NcclCommHandle comm) = 0;

  // Destroy a communicator object comm.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommdestroy
  virtual absl::Status CommDestroy(NcclCommHandle comm) = 0;

  // Returns the number of ranks in the NCCL communicator comm.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommcount
  virtual absl::StatusOr<int32_t> CommCount(NcclCommHandle comm) = 0;

  // Queries the progress and potential errors of asynchronous operations
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommgetasyncerror
  virtual absl::Status CommGetAsyncError(NcclCommHandle comm) = 0;

  // Starts a group call.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupstart
  virtual absl::Status GroupStart() = 0;

  // Ends a group call.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupend
  virtual absl::Status GroupEnd() = 0;

  // Reduce buffers of length `count` in `send_buff` using `reduction_kind`
  // reduction and leaves identical copies of the result on each `recv_buff`.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce
  virtual absl::Status AllReduce(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 ReductionKind reduction_kind,
                                 NcclCommHandle comm, se::Stream* stream) = 0;

  // Copy data in `send_buff` from the root GPU to the `recv_buff` on
  // all GPUs.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast
  virtual absl::Status Broadcast(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count, size_t root,
                                 NcclCommHandle comm, se::Stream* stream) = 0;
  // Reduce data in `send_buff` from all GPUs using the `reduction_kind`
  // operation and leave the reduced result scattered over the devices so that
  // the `recv_buff` on rank `i` will contain the i-th block of the result.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreducescatter
  virtual absl::Status ReduceScatter(se::DeviceMemoryBase send_buffer,
                                     se::DeviceMemoryBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     NcclCommHandle comm,
                                     se::Stream* stream) = 0;

  // Gather `count` values from all GPUs into recv_buffer, receiving data from
  // rank `i` at offset `i * sendcount`.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather
  virtual absl::Status AllGather(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 NcclCommHandle comm, se::Stream* stream) = 0;

  // Send data from `send_buff` to rank `peer`.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend
  virtual absl::Status Send(se::DeviceMemoryBase send_buffer,
                            PrimitiveType dtype, size_t count, int32_t peer,
                            NcclCommHandle comm, se::Stream* stream) = 0;

  // Receive data from rank `peer` into `recv_buff`.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclrecv
  virtual absl::Status Recv(se::DeviceMemoryBase recv_buffer,
                            PrimitiveType dtype, size_t count, int32_t peer,
                            NcclCommHandle comm, se::Stream* stream) = 0;

  // Register `buffer` with communicator `comm` for zero-copy communication.
  // Returned handle can be used for future unregistration.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommregister
  virtual absl::StatusOr<NcclRegisteredBufferHandle> RegisterBuffer(
      NcclCommHandle comm, se::DeviceMemoryBase buffer) = 0;

  // Deregister buffer represented by `handle` from communicator `comm`.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommderegister
  virtual absl::StatusOr<NcclRegisteredBufferHandle> DeregisterBuffer(
      NcclCommHandle comm, NcclRegisteredBufferHandle handle) = 0;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_API_H_
