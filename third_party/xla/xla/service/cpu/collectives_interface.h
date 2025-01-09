/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_COLLECTIVES_INTERFACE_H_
#define XLA_SERVICE_CPU_COLLECTIVES_INTERFACE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

namespace internal {

// An adapter from a shared_ptr<Communicator> to a Communicator.
class CommunicatorWrapper final : public Communicator {
 public:
  explicit CommunicatorWrapper(std::shared_ptr<Communicator> comm)
      : comm_(std::move(comm)) {}

  absl::Status AllReduce(stream_executor::DeviceMemoryBase send_buffer,
                         stream_executor::DeviceMemoryBase recv_buffer,
                         PrimitiveType dtype, size_t count,
                         ReductionKind reduction_kind,
                         const Executor& executor) final {
    return comm_->AllReduce(send_buffer, recv_buffer, dtype, count,
                            reduction_kind, executor);
  }

  absl::Status Broadcast(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, RankId root,
                         const Executor& executor) final {
    return comm_->Broadcast(send_buffer, recv_buffer, dtype, count, root,
                            executor);
  }

  absl::Status ReduceScatter(se::DeviceMemoryBase send_buffer,
                             se::DeviceMemoryBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             ReductionKind reduction_kind,
                             const Executor& executor) final {
    return comm_->ReduceScatter(send_buffer, recv_buffer, dtype, count,
                                reduction_kind, executor);
  }

  absl::Status AllGather(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, const Executor& executor) final {
    return comm_->AllGather(send_buffer, recv_buffer, dtype, count, executor);
  }

  absl::Status CollectivePermute(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 std::optional<RankId> source_rank,
                                 absl::Span<const RankId> target_ranks,
                                 const Executor& executor) final {
    return comm_->CollectivePermute(send_buffer, recv_buffer, dtype, count,
                                    source_rank, target_ranks, executor);
  }

  absl::Status AllToAll(absl::Span<const se::DeviceMemoryBase> send_buffers,
                        absl::Span<const se::DeviceMemoryBase> recv_buffers,
                        PrimitiveType dtype, size_t count,
                        const Executor& executor) final {
    return comm_->AllToAll(send_buffers, recv_buffers, dtype, count, executor);
  }

  absl::Status Send(se::DeviceMemoryBase send_buffer, PrimitiveType dtype,
                    size_t count, RankId peer, const Executor& executor) final {
    return comm_->Send(send_buffer, dtype, count, peer, executor);
  }

  absl::Status Recv(se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                    size_t count, RankId peer, const Executor& executor) final {
    return comm_->Recv(recv_buffer, dtype, count, peer, executor);
  }

  absl::StatusOr<size_t> NumRanks() const final { return comm_->NumRanks(); }

  std::string ToString() const final { return comm_->ToString(); }

 private:
  std::shared_ptr<Communicator> comm_;
};

}  // namespace internal

class CollectivesInterface : public CpuCollectives {
 public:
  virtual ~CollectivesInterface() = default;

  // Builds a context for a collective group.
  // Args:
  //  devices: the devices participating in this collective.
  //  rank: the rank of this process.
  virtual absl::StatusOr<std::shared_ptr<Communicator>> GetCommunicator(
      absl::Span<GlobalDeviceId const> devices, int rank) = 0;

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(int32_t nranks, const CliqueKey& clique_key,
                      const std::optional<CliqueId>& clique_id,
                      absl::Span<const DeviceRank> ranks,
                      const Config& config) final {
    // We expect to create CPU communicators lazily one at a time.
    if (ranks.size() != 1) {
      return InvalidArgument("Expected 1 rank, got %d", ranks.size());
    }

    TF_ASSIGN_OR_RETURN(auto comm, GetCommunicator(clique_key.devices(),
                                                   ranks[0].rank.value()));

    std::vector<std::unique_ptr<Communicator>> comms;
    comms.reserve(1);
    comms.push_back(std::make_unique<internal::CommunicatorWrapper>(comm));
    return comms;
  }
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_COLLECTIVES_INTERFACE_H_
