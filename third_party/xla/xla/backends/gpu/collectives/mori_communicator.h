/* Copyright 2026 The OpenXLA Authors.
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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_MORI_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_MORI_COMMUNICATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

class MoriCollectives;

// XLA collectives communicator wrapping a MORI communicator.
class MoriCommunicator : public GpuCommunicator {
 public:
  constexpr static uint32_t kMaxTeams = 24;

  friend class MoriCollectives;
  ~MoriCommunicator() override;

  static absl::StatusOr<std::unique_ptr<MoriCommunicator>> Create(
      MoriCollectives* coll, std::shared_ptr<CancellationToken> cancel,
      int rank, absl::Span<const int> rank_to_pe);

  // MoriCommunicator is not copyable or movable.
  MoriCommunicator(const MoriCommunicator&) = delete;
  MoriCommunicator(MoriCommunicator&&) = delete;
  MoriCommunicator& operator=(const MoriCommunicator&) = delete;
  MoriCommunicator& operator=(MoriCommunicator&&) = delete;

  absl::Status Abort() final;
  absl::StatusOr<size_t> NumRanks() const final;
  absl::StatusOr<size_t> CurrentRank() final;

  absl::Status Barrier(const Executor& executor) final;

  Future<> GroupExecute(absl::AnyInvocable<absl::Status() &&> group) final;

  Future<> AllReduce(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, ReductionKind reduction_kind,
                     const Executor& executor) final;

  Future<> Broadcast(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, RankId root, const Executor& executor) final;

  Future<> ReduceScatter(se::DeviceAddressBase send_buffer,
                         se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) final;

  Future<> AllGather(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, const Executor& executor) final;

  Future<> AllToAll(absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
                    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
                    PrimitiveType dtype, size_t count,
                    const Executor& executor) final;

  Future<> CollectivePermute(se::DeviceAddressBase send_buffer,
                             se::DeviceAddressBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             std::optional<RankId> source_rank,
                             absl::Span<const RankId> target_ranks,
                             const Executor& executor) final;

  Future<> Send(se::DeviceAddressBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final {
    return absl::UnimplementedError("Not implemented");
  }

  Future<> Recv(se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final {
    return absl::UnimplementedError("Not implemented");
  }

  Future<> Send(se::DeviceAddressBase recv_buffer,
                se::DeviceAddressBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  Future<> Recv(se::DeviceAddressBase recv_buffer,
                se::DeviceAddressBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  // Polls the communicator until any pending non-blocking operations are done
  // or aborted.
  absl::Status PollUntilDone() const;

 private:
  absl::Status GroupLaunch(absl::FunctionRef<absl::Status()> group);

  absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               const Executor& executor) final;

  absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count, RankId root,
                               const Executor& executor) final {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status LaunchReduceScatter(se::DeviceAddressBase send_buffer,
                                   se::DeviceAddressBase recv_buffer,
                                   PrimitiveType dtype, size_t count,
                                   ReductionKind reduction_kind,
                                   const Executor& executor) final;

  absl::Status LaunchAllGather(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               const Executor& executor) final;

  absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) final {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status LaunchCollectivePermute(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       std::optional<RankId> source_rank,
                                       absl::Span<const RankId> target_ranks,
                                       const Executor& executor) final;

  absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status Quiet(const Executor& executor) final;

  absl::Status Fence() final;

  std::string ToString() const final;

 private:
  // Executes f on executor_, or calls f directly if executor_ is null.
  Future<> Execute(absl::AnyInvocable<absl::Status() &&> f) const;

  MoriCommunicator(MoriCollectives* coll,
                   std::shared_ptr<CancellationToken> cancel)
      : collectives_(coll), cancel_(std::move(cancel)) {}

  enum class P2PType : int32_t { Send, Recv };

  absl::Status P2P(P2PType p2p_type, PrimitiveType type,
                   se::DeviceAddressBase recv_buffer,
                   se::DeviceAddressBase send_buffer, size_t count, RankId peer,
                   const Executor& executor);

  static absl::StatusOr<se::Stream*> ToStream(const Executor& executor);

  MoriCollectives* collectives_;  // Parent MoriCollectives instance

  // This communicator's participant set (NOT the global MORI clique). `rank_`
  // is this rank within the collective, `num_ranks_` the participant count, and
  // `rank_to_pe_dev_` a device array mapping collective rank -> global MORI PE.
  int rank_ = 0;
  int num_ranks_ = 0;
  // Should all pending collectives cancel?
  std::shared_ptr<CancellationToken> cancel_;
  bool aborted_ = false;  // Has Abort() been called?
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MORI_COMMUNICATOR_H_
