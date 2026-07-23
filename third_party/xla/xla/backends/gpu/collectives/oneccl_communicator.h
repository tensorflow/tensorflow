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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_COMMUNICATOR_H_

#include <memory>
#include <optional>
#include <string>

#include "oneapi/ccl.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
class OnecclCommunicator : public GpuCommunicator {
 public:
  static absl::StatusOr<std::unique_ptr<OnecclCommunicator>> Create(
      absl::AnyInvocable<absl::StatusOr<onecclComm_t>()> make_comm,
      bool is_async, tsl::Env& env = *tsl::Env::Default());
  ~OnecclCommunicator() override;

  absl::StatusOr<size_t> NumRanks() const final { return absl::OkStatus(); };
  Future<> GroupExecute(absl::AnyInvocable<absl::Status() &&> group) {
    return absl::OkStatus();
  }

  Future<> AllReduce(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, ReductionKind reduction_kind,
                     const Executor& executor) final {
    return absl::OkStatus();
  }
  Future<> Broadcast(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, RankId root,
                     const Executor& executor) final {
    return absl::OkStatus();
  }

  Future<> ReduceScatter(se::DeviceAddressBase send_buffer,
                         se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) final {
    return absl::OkStatus();
  }

  Future<> AllGather(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, const Executor& executor) final {
    return absl::OkStatus();
  }

  Future<> Send(se::DeviceAddressBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final {
    return absl::OkStatus();
  }

  Future<> Recv(se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final {
    return absl::OkStatus();
  }

  std::string ToString() const final;

  Future<> AllToAll(absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
                    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
                    PrimitiveType dtype, size_t count,
                    const Executor& executor) final {
    return absl::OkStatus();
  }

  Future<> CollectivePermute(se::DeviceAddressBase send_buffer,
                             se::DeviceAddressBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             std::optional<RankId> source_rank,
                             absl::Span<const RankId> target_ranks,
                             const Executor& executor) final {
    return absl::OkStatus();
  }

 private:
  absl::Status GroupStart() { return absl::OkStatus(); }
  absl::Status GroupEnd() { return absl::OkStatus(); }

  absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               const Executor& executor) final {
    return absl::OkStatus();
  }

  absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count, RankId root,
                               const Executor& executor) final {
    return absl::OkStatus();
  }

  absl::Status LaunchReduceScatter(se::DeviceAddressBase send_buffer,
                                   se::DeviceAddressBase recv_buffer,
                                   PrimitiveType dtype, size_t count,
                                   ReductionKind reduction_kind,
                                   const Executor& executor) final {
    return absl::OkStatus();
  }

  absl::Status LaunchAllGather(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               const Executor& executor) final {
    return absl::OkStatus();
  }

  absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) final {
    return absl::OkStatus();
  }

  absl::Status LaunchCollectivePermute(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       std::optional<RankId> source_rank,
                                       absl::Span<const RankId> target_ranks,
                                       const Executor& executor) final {
    return absl::OkStatus();
  }

  absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final {
    return absl::OkStatus();
  }

  absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final {
    return absl::OkStatus();
  }

  // Executes f on executor_, or calls f directly if executor_ is null.
  Future<> Execute(absl::AnyInvocable<absl::Status() &&> f) const;
  explicit OnecclCommunicator(onecclComm_t comm,
                              std::unique_ptr<tsl::Executor> executor)
      : comm_(comm), executor_(std::move(executor)) {
    LOG(INFO) << "Created oneCCL communicator" << *this;
  }
  onecclComm_t comm_;
  std::unique_ptr<tsl::Executor> executor_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_COMMUNICATOR_H_
