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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COMMUNICATOR_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

class NvshmemCollectives;

// XLA collectives communicator wrapping an NVSHMEM communicator.
class NvshmemCommunicator : public Communicator {
 public:
  explicit NvshmemCommunicator(NvshmemCollectives* collectives);
  // Since the communicator is hardcoded to use pre-defined node
  // team for now, we don't need to call nvshmem_team_destroy on it.
  // If user-defined comms are used, we need to add a destructor to
  // call nvshmem_team_destroy on each one.

  // NvshmemCommunicator is not copyable or movable.
  NvshmemCommunicator(const NvshmemCommunicator&) = delete;
  NvshmemCommunicator(NvshmemCommunicator&&) = delete;
  NvshmemCommunicator& operator=(const NvshmemCommunicator&) = delete;
  NvshmemCommunicator& operator=(NvshmemCommunicator&&) = delete;

  absl::Status Abort() final;
  absl::StatusOr<size_t> NumRanks() const final;
  absl::StatusOr<size_t> CurrentRank() final;

  absl::Status Barrier(const Executor& executor) final;

  tsl::AsyncValueRef<Event> AllReduce(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      ReductionKind reduction_kind,
                                      const Executor& executor) final;

  tsl::AsyncValueRef<Event> Broadcast(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      RankId root,
                                      const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  tsl::AsyncValueRef<Event> ReduceScatter(se::DeviceMemoryBase send_buffer,
                                          se::DeviceMemoryBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          ReductionKind reduction_kind,
                                          const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  tsl::AsyncValueRef<Event> AllGather(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  tsl::AsyncValueRef<Event> AllToAll(
      absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  tsl::AsyncValueRef<Event> CollectivePermute(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  tsl::AsyncValueRef<Event> Send(se::DeviceMemoryBase send_buffer,
                                 PrimitiveType dtype, size_t count, RankId peer,
                                 const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  tsl::AsyncValueRef<Event> Recv(se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count, RankId peer,
                                 const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  absl::Status RegisterBuffer(void* addr, size_t length) final;

  tsl::AsyncValueRef<Event> Send(se::DeviceMemoryBase recv_buffer,
                                 se::DeviceMemoryBase send_buffer,
                                 PrimitiveType dtype, size_t count, RankId peer,
                                 const Executor& executor) final;

  tsl::AsyncValueRef<Event> Recv(se::DeviceMemoryBase recv_buffer,
                                 se::DeviceMemoryBase send_buffer,
                                 PrimitiveType dtype, size_t count, RankId peer,
                                 const Executor& executor) final;

  absl::Status Quiet(const Executor& executor) final;

  absl::Status Fence() final;

  std::string ToString() const final;

 private:
  absl::Status P2P(absl::string_view op_name, PrimitiveType type,
                   se::DeviceMemoryBase recv_buffer,
                   se::DeviceMemoryBase send_buffer, size_t count, RankId peer,
                   const Executor& executor);

  static absl::StatusOr<se::Stream*> ToStream(const Executor& executor);

  NvshmemCollectives* collectives_;  // Parent NvshmemCollectives instance
  bool aborted_ = false;             // Has Abort() been called?
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COMMUNICATOR_H_
