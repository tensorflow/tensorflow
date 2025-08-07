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

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Collective communicator defines the set of communicating XLA processes.
//
// Returned async value signals that the communicator has successfully
// launched the operation on the underlying executor.
// Completion of the operation depends on the backend implementation. i.e. on
// GPU the async value becomes available when the operation is scheduled on the
// device stream, and on CPU it becomes available when the operation is
// completed.
class Communicator {
 public:
  using Event = tsl::Chain;

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

  // Register `buffer` for efficient collective operations (i.e. on NVSHMEM
  // backend it registers the buffer for unregistered nvshmem buffers).
  virtual absl::Status RegisterBuffer(void* addr, size_t length) {
    return absl::UnimplementedError(
        "User-managed buffer registration is not supported");
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

  // This is a barrier operation that blocks all participating
  // ranks from proceeding.
  virtual absl::Status Barrier(const Executor& executor) {
    return Unimplemented("Barrier is not implemented");
  }

  // Reduce buffers of length `count` in `send_buff` using `reduction_kind`
  // reduction and leaves identical copies of the result on each `recv_buff`.
  virtual tsl::AsyncValueRef<Event> AllReduce(
      stream_executor::DeviceMemoryBase send_buffer,
      stream_executor::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
      size_t count, ReductionKind reduction_kind, const Executor& executor) = 0;

  // Copy data in `send_buff` from the root device to the `recv_buff` on
  // all other devices.
  virtual tsl::AsyncValueRef<Event> Broadcast(se::DeviceMemoryBase send_buffer,
                                              se::DeviceMemoryBase recv_buffer,
                                              PrimitiveType dtype, size_t count,
                                              RankId root,
                                              const Executor& executor) = 0;

  // Reduce data in `send_buff` from all devices using the `reduction_kind`
  // operation and leave the reduced result scattered over the devices so that
  // the `recv_buff` on rank `i` will contain the i-th block of the result.
  virtual tsl::AsyncValueRef<Event> ReduceScatter(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
      const Executor& executor) = 0;

  // Gather `count` values from all devices into `recv_buffer`, receiving data
  // from rank `i` at offset `i * sendcount`.
  virtual tsl::AsyncValueRef<Event> AllGather(se::DeviceMemoryBase send_buffer,
                                              se::DeviceMemoryBase recv_buffer,
                                              PrimitiveType dtype, size_t count,
                                              const Executor& executor) = 0;

  // Sends data from `send_buffer` to `target_ranks` and receives data from
  // `source_rank` into `recv_buffer`. If `source_rank` is not specified, the
  // output is filled with zeros.
  virtual tsl::AsyncValueRef<Event> CollectivePermute(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) = 0;

  // Sends `count` values from `send_buffers` to other ranks and receives data
  // from other ranks into `recv_buffers`.
  virtual tsl::AsyncValueRef<Event> AllToAll(
      absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) = 0;

  // Send data from `send_buff` to rank `peer`.
  virtual tsl::AsyncValueRef<Event> Send(se::DeviceMemoryBase send_buffer,
                                         PrimitiveType dtype, size_t count,
                                         RankId peer,
                                         const Executor& executor) = 0;

  // Receive data from rank `peer` into `recv_buff`.
  virtual tsl::AsyncValueRef<Event> Recv(se::DeviceMemoryBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         RankId peer,
                                         const Executor& executor) = 0;

  // Send data from `send_buff` to rank `recv_buff` (one-way send).
  virtual tsl::AsyncValueRef<Event> Send(se::DeviceMemoryBase recv_buffer,
                                         se::DeviceMemoryBase send_buffer,
                                         PrimitiveType dtype, size_t count,
                                         RankId peer,
                                         const Executor& executor) {
    return Unimplemented("One-way send is not implemented");
  }

  // Receive data from rank `peer` into `recv_buff` (one-way recv).
  virtual tsl::AsyncValueRef<Event> Recv(se::DeviceMemoryBase recv_buffer,
                                         se::DeviceMemoryBase send_buffer,
                                         PrimitiveType dtype, size_t count,
                                         RankId peer,
                                         const Executor& executor) {
    return Unimplemented("One-way recv is not implemented");
  }

  // Returns the number of ranks in the communicator.
  virtual absl::StatusOr<size_t> NumRanks() const = 0;

  // Returns the current rank number in the communicator.
  virtual absl::StatusOr<size_t> CurrentRank() {
    return Unimplemented("CurrentRank is not implemented");
  }

  // Returns a human-readable description of the communicator.
  virtual std::string ToString() const = 0;

  // Guarantees completion of all operations on symmetric data objects which
  // makes the updates visible to all other PEs.
  virtual absl::Status Quiet(const Executor& executor) {
    return Unimplemented("Quiet is not implemented");
  }

  // Guarantees ordering of delivery of all previous operations on symmetric
  // data objects.
  virtual absl::Status Fence() {
    return Unimplemented("Fence is not implemented");
  }

 protected:
  // Returns an `Event` that is always available.
  static tsl::AsyncValueRef<Event> OkEvent() {
    return tsl::MakeAvailableAsyncValueRef<Event>();
  }
};

inline std::ostream& operator<<(std::ostream& os, const Communicator& comm) {
  return os << comm.ToString();
}

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_COMMUNICATOR_H_
