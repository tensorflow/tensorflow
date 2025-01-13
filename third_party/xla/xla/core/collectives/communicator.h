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

#ifndef XLA_CORE_COLLECTIVES_COMMUNICATOR_H_
#define XLA_CORE_COLLECTIVES_COMMUNICATOR_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Collective communicator defines the set of communicating XLA processes.
class Communicator {
 public:
  virtual ~Communicator() = default;

  // An executor is an abstraction for the underlying resource where collective
  // operations are executed. For example on GPU backend it could be a device
  // stream, and on CPU backend it could be a thread pool.
  class Executor {
   public:
    virtual ~Executor() = default;
  };

  // An RAII handle for buffers registered with the communicator. Child classes
  // are responsible for unregistering the buffer when the handle is destroyed.
  class RegisteredBufferHandle {
   public:
    virtual ~RegisteredBufferHandle() = default;
    virtual absl::Status Unregister() = 0;
  };

  // Register `buffer` for efficient collective operations (i.e. on NCCL backend
  // it registers the buffer for zero-copy collective operations).
  virtual absl::StatusOr<std::unique_ptr<RegisteredBufferHandle>>
  RegisterBuffer(stream_executor::DeviceMemoryBase buffer) {
    return Unimplemented("User-managed buffer registration is not supported");
  }

  // Abort any uncompleted operations and destroys the underlying communicator
  // object. It is undefined behavior to use the communicator after calling
  // this method.
  virtual absl::Status Abort() {
    return Unimplemented("Aborting communicator is not implemented");
  }

  // Checks the health of the communicator. It might return an error from the
  // previously launched asynchronous collective operations, and it does not
  // have to wait for the completion of scheduled operations.
  virtual absl::Status HealthCheck() const { return absl::OkStatus(); }

  // Reduce buffers of length `count` in `send_buff` using `reduction_kind`
  // reduction and leaves identical copies of the result on each `recv_buff`.
  virtual absl::Status AllReduce(stream_executor::DeviceMemoryBase send_buffer,
                                 stream_executor::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 ReductionKind reduction_kind,
                                 const Executor& executor) = 0;

  // Copy data in `send_buff` from the root device to the `recv_buff` on
  // all other devices.
  virtual absl::Status Broadcast(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count, RankId root,
                                 const Executor& executor) = 0;

  // Reduce data in `send_buff` from all devices using the `reduction_kind`
  // operation and leave the reduced result scattered over the devices so that
  // the `recv_buff` on rank `i` will contain the i-th block of the result.
  virtual absl::Status ReduceScatter(se::DeviceMemoryBase send_buffer,
                                     se::DeviceMemoryBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     const Executor& executor) = 0;

  // Gather `count` values from all devices into `recv_buffer`, receiving data
  // from rank `i` at offset `i * sendcount`.
  virtual absl::Status AllGather(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 const Executor& executor) = 0;

  // Sends data from `send_buffer` to `target_ranks` and receives data from
  // `source_rank` into `recv_buffer`. If `source_rank` is not specified, the
  // output is filled with zeros.
  virtual absl::Status CollectivePermute(se::DeviceMemoryBase send_buffer,
                                         se::DeviceMemoryBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         std::optional<RankId> source_rank,
                                         absl::Span<const RankId> target_ranks,
                                         const Executor& executor) = 0;

  // Sends `count` values from `send_buffers` to other ranks and receives data
  // from other ranks into `recv_buffers`.
  virtual absl::Status AllToAll(
      absl::Span<const se::DeviceMemoryBase> send_buffers,
      absl::Span<const se::DeviceMemoryBase> recv_buffers, PrimitiveType dtype,
      size_t count, const Executor& executor) = 0;

  // Send data from `send_buff` to rank `peer`.
  virtual absl::Status Send(se::DeviceMemoryBase send_buffer,
                            PrimitiveType dtype, size_t count, RankId peer,
                            const Executor& executor) = 0;

  // Receive data from rank `peer` into `recv_buff`.
  virtual absl::Status Recv(se::DeviceMemoryBase recv_buffer,
                            PrimitiveType dtype, size_t count, RankId peer,
                            const Executor& executor) = 0;

  // Returns the number of ranks in the communicator.
  virtual absl::StatusOr<size_t> NumRanks() const = 0;

  // Returns a human-readable description of the communicator.
  virtual std::string ToString() const = 0;
};

inline std::ostream& operator<<(std::ostream& os, const Communicator& comm) {
  return os << comm.ToString();
}

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_COMMUNICATOR_H_
