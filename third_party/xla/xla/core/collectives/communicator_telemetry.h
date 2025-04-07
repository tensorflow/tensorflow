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

#ifndef XLA_CORE_COLLECTIVES_COMMUNICATOR_TELEMETRY_H_
#define XLA_CORE_COLLECTIVES_COMMUNICATOR_TELEMETRY_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {

class CommunicatorTelemetry : public Communicator {
 public:
  explicit CommunicatorTelemetry(std::unique_ptr<Communicator> communicator)
      : communicator_(std::move(communicator)) {}
  ~CommunicatorTelemetry() override = default;

  // CommunicatorTelemetry is not copyable or movable.
  CommunicatorTelemetry(const CommunicatorTelemetry&) = delete;
  CommunicatorTelemetry(CommunicatorTelemetry&&) = delete;
  CommunicatorTelemetry& operator=(const CommunicatorTelemetry&) = delete;
  CommunicatorTelemetry& operator=(CommunicatorTelemetry&&) = delete;

  absl::Status Abort() final;
  absl::Status HealthCheck() const final;
  absl::StatusOr<size_t> NumRanks() const final;

  absl::StatusOr<std::unique_ptr<RegisteredBufferHandle>> RegisterBuffer(
      se::DeviceMemoryBase buffer) final;

  absl::Status MeasureAndReportCollectiveLatency(
      absl::string_view collective_name,
      std::optional<ReductionKind> reduction_kind, PrimitiveType dtype,
      size_t count, absl::AnyInvocable<absl::Status(void)> collective_fn);

  absl::Status AllReduce(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) final;

  absl::Status Broadcast(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, RankId root,
                         const Executor& executor) final;

  absl::Status ReduceScatter(se::DeviceMemoryBase send_buffer,
                             se::DeviceMemoryBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             ReductionKind reduction_kind,
                             const Executor& executor) final;

  absl::Status AllGather(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, const Executor& executor) final;

  absl::Status AllToAll(absl::Span<const se::DeviceMemoryBase> send_buffers,
                        absl::Span<const se::DeviceMemoryBase> recv_buffers,
                        PrimitiveType dtype, size_t count,
                        const Executor& executor) final;

  absl::Status CollectivePermute(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 std::optional<RankId> source_rank,
                                 absl::Span<const RankId> target_ranks,
                                 const Executor& executor) final;

  absl::Status Send(se::DeviceMemoryBase send_buffer, PrimitiveType dtype,
                    size_t count, RankId peer, const Executor& executor) final;

  absl::Status Recv(se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                    size_t count, RankId peer, const Executor& executor) final;
  // Returns a human-readable description of the communicator.
  std::string ToString() const final;

  const Communicator* GetCommunicator() const { return communicator_.get(); }

 private:
  std::unique_ptr<Communicator> communicator_;
};

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_COMMUNICATOR_TELEMETRY_H_
