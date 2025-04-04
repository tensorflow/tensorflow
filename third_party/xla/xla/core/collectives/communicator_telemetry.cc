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

#include "xla/core/collectives/communicator_telemetry.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/core/collectives/collectives_telemetry_consumer_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {

absl::Status CommunicatorTelemetry::Abort() { return communicator_->Abort(); }

absl::Status CommunicatorTelemetry::HealthCheck() const {
  return communicator_->HealthCheck();
}

absl::StatusOr<size_t> CommunicatorTelemetry::NumRanks() const {
  return communicator_->NumRanks();
}

absl::StatusOr<std::unique_ptr<Communicator::RegisteredBufferHandle>>
CommunicatorTelemetry::RegisterBuffer(se::DeviceMemoryBase buffer) {
  return communicator_->RegisterBuffer(buffer);
}

absl::Status CommunicatorTelemetry::MeasureAndReportCollectiveLatency(
    absl::string_view collective_name,
    std::optional<ReductionKind> reduction_kind, PrimitiveType dtype,
    size_t count, absl::AnyInvocable<absl::Status(void)> collective_fn) {
  absl::Time start_time = absl::Now();
  auto status = collective_fn();
  absl::Duration duration = absl::Now() - start_time;

  CollectivesTelemetryConsumerRegistry::Get()
      .value()
      ->ConsumeCollectiveTelemetry(collective_name, reduction_kind, dtype,
                                   count, duration);

  return status;
}

absl::Status CommunicatorTelemetry::AllReduce(se::DeviceMemoryBase send_buffer,
                                              se::DeviceMemoryBase recv_buffer,
                                              PrimitiveType dtype, size_t count,
                                              ReductionKind reduction_kind,
                                              const Executor& executor) {
  return MeasureAndReportCollectiveLatency(
      "AllReduce", reduction_kind, dtype, count,
      [this, &send_buffer, &recv_buffer, &dtype, &count, &reduction_kind,
       &executor]() {
        return communicator_->AllReduce(send_buffer, recv_buffer, dtype, count,
                                        reduction_kind, executor);
      });
}

absl::Status CommunicatorTelemetry::Broadcast(se::DeviceMemoryBase send_buffer,
                                              se::DeviceMemoryBase recv_buffer,
                                              PrimitiveType dtype, size_t count,
                                              RankId root,
                                              const Executor& executor) {
  return MeasureAndReportCollectiveLatency(
      "Broadcast", std::nullopt, dtype, count,
      [this, &send_buffer, &recv_buffer, &dtype, &count, &root, &executor]() {
        return communicator_->Broadcast(send_buffer, recv_buffer, dtype, count,
                                        root, executor);
      });
}
absl::Status CommunicatorTelemetry::ReduceScatter(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  return MeasureAndReportCollectiveLatency(
      "ReduceScatter", reduction_kind, dtype, count,
      [this, &send_buffer, &recv_buffer, &dtype, &count, &reduction_kind,
       &executor]() {
        return communicator_->ReduceScatter(send_buffer, recv_buffer, dtype,
                                            count, reduction_kind, executor);
      });
}

absl::Status CommunicatorTelemetry::AllGather(se::DeviceMemoryBase send_buffer,
                                              se::DeviceMemoryBase recv_buffer,
                                              PrimitiveType dtype, size_t count,
                                              const Executor& executor) {
  return MeasureAndReportCollectiveLatency(
      "AllGather", std::nullopt, dtype, count,
      [this, &send_buffer, &recv_buffer, &dtype, &count, &executor]() {
        return communicator_->AllGather(send_buffer, recv_buffer, dtype, count,
                                        executor);
      });
}

absl::Status CommunicatorTelemetry::AllToAll(
    absl::Span<const se::DeviceMemoryBase> send_buffers,
    absl::Span<const se::DeviceMemoryBase> recv_buffers, PrimitiveType dtype,
    size_t count, const Executor& executor) {
  return MeasureAndReportCollectiveLatency(
      "AllToAll", std::nullopt, dtype, count,
      [this, &send_buffers, &recv_buffers, &dtype, &count, &executor]() {
        return communicator_->AllToAll(send_buffers, recv_buffers, dtype, count,
                                       executor);
      });
}

absl::Status CommunicatorTelemetry::CollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  return MeasureAndReportCollectiveLatency(
      "CollectivePermute", std::nullopt, dtype, count,
      [this, &send_buffer, &recv_buffer, &dtype, &count, &source_rank,
       &target_ranks, &executor]() {
        return communicator_->CollectivePermute(send_buffer, recv_buffer, dtype,
                                                count, source_rank,
                                                target_ranks, executor);
      });
}

absl::Status CommunicatorTelemetry::Send(se::DeviceMemoryBase send_buffer,
                                         PrimitiveType dtype, size_t count,
                                         RankId peer,
                                         const Executor& executor) {
  return MeasureAndReportCollectiveLatency(
      "Send", std::nullopt, dtype, count,
      [this, &send_buffer, &dtype, &count, &peer, &executor]() {
        return communicator_->Send(send_buffer, dtype, count, peer, executor);
      });
}

absl::Status CommunicatorTelemetry::Recv(se::DeviceMemoryBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         RankId peer,
                                         const Executor& executor) {
  return MeasureAndReportCollectiveLatency(
      "Recv", std::nullopt, dtype, count,
      [this, &recv_buffer, &dtype, &count, &peer, &executor]() {
        return communicator_->Recv(recv_buffer, dtype, count, peer, executor);
      });
}

std::string CommunicatorTelemetry::ToString() const {
  return communicator_->ToString();
}

}  // namespace xla
