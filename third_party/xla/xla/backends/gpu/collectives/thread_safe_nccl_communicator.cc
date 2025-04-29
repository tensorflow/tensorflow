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

#include "xla/backends/gpu/collectives/thread_safe_nccl_communicator.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/nccl_communicator.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

namespace xla::gpu {

absl::StatusOr<std::unique_ptr<ThreadSafeNcclCommunicator>>
ThreadSafeNcclCommunicator::Create(
    absl::AnyInvocable<absl::StatusOr<ncclComm_t>() &&> f, tsl::Env& env) {
  auto c = absl::WrapUnique(new ThreadSafeNcclCommunicator(env));
  ncclComm_t comm = nullptr;
  TF_RETURN_IF_ERROR(c->worker_thread_.Run([&comm, &f]() {
    TF_ASSIGN_OR_RETURN(comm, std::move(f)());
    return PollUntilDone(comm);
  }));
  c->communicator_ = std::make_unique<NcclCommunicator>(comm);
  return c;
}

absl::Status ThreadSafeNcclCommunicator::GroupStart() {
  return worker_thread_.Run([this]() { return communicator_->GroupStart(); });
}

absl::Status ThreadSafeNcclCommunicator::GroupEnd() {
  return worker_thread_.Run([this]() { return communicator_->GroupEnd(); });
}

absl::Status ThreadSafeNcclCommunicator::Abort() {
  return worker_thread_.Run([this]() { return communicator_->Abort(); });
}

absl::Status ThreadSafeNcclCommunicator::HealthCheck() const {
  return worker_thread_.Run([this]() { return communicator_->HealthCheck(); });
}

absl::StatusOr<size_t> ThreadSafeNcclCommunicator::NumRanks() const {
  absl::StatusOr<size_t> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([this, &result]() {
    result = communicator_->NumRanks();
    return absl::OkStatus();
  }));
  return result;
}

absl::StatusOr<std::unique_ptr<Communicator::RegisteredBufferHandle>>
ThreadSafeNcclCommunicator::RegisterBuffer(se::DeviceMemoryBase buffer) {
  absl::StatusOr<std::unique_ptr<Communicator::RegisteredBufferHandle>> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([&, this]() {
    result = communicator_->RegisterBuffer(buffer);
    return absl::OkStatus();
  }));
  return result;
}

tsl::AsyncValueRef<NcclCommunicator::Event>
ThreadSafeNcclCommunicator::AllReduce(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      ReductionKind reduction_kind,
                                      const Executor& executor) {
  tsl::AsyncValueRef<NcclCommunicator::Event> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([&, this]() {
    result = communicator_->AllReduce(send_buffer, recv_buffer, dtype, count,
                                      reduction_kind, executor);
    return absl::OkStatus();
  }));
  return result;
}

tsl::AsyncValueRef<NcclCommunicator::Event>
ThreadSafeNcclCommunicator::Broadcast(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      RankId root, const Executor& executor) {
  tsl::AsyncValueRef<NcclCommunicator::Event> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([&, this]() {
    result = communicator_->Broadcast(send_buffer, recv_buffer, dtype, count,
                                      root, executor);
    return absl::OkStatus();
  }));
  return result;
}

tsl::AsyncValueRef<NcclCommunicator::Event>
ThreadSafeNcclCommunicator::ReduceScatter(se::DeviceMemoryBase send_buffer,
                                          se::DeviceMemoryBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          ReductionKind reduction_kind,
                                          const Executor& executor) {
  tsl::AsyncValueRef<NcclCommunicator::Event> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([&, this]() {
    result = communicator_->ReduceScatter(send_buffer, recv_buffer, dtype,
                                          count, reduction_kind, executor);
    return absl::OkStatus();
  }));
  return result;
}

tsl::AsyncValueRef<NcclCommunicator::Event>
ThreadSafeNcclCommunicator::AllGather(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      const Executor& executor) {
  tsl::AsyncValueRef<NcclCommunicator::Event> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([&, this]() {
    result = communicator_->AllGather(send_buffer, recv_buffer, dtype, count,
                                      executor);
    return absl::OkStatus();
  }));
  return result;
}

tsl::AsyncValueRef<NcclCommunicator::Event>
ThreadSafeNcclCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  tsl::AsyncValueRef<NcclCommunicator::Event> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([&, this]() {
    result = communicator_->AllToAll(send_buffers, recv_buffers, dtype, count,
                                     executor);
    return absl::OkStatus();
  }));
  return result;
}

tsl::AsyncValueRef<NcclCommunicator::Event>
ThreadSafeNcclCommunicator::CollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  tsl::AsyncValueRef<NcclCommunicator::Event> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([&, this]() {
    result =
        communicator_->CollectivePermute(send_buffer, recv_buffer, dtype, count,
                                         source_rank, target_ranks, executor);
    return absl::OkStatus();
  }));
  return result;
}

tsl::AsyncValueRef<NcclCommunicator::Event> ThreadSafeNcclCommunicator::Send(
    se::DeviceMemoryBase send_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  tsl::AsyncValueRef<NcclCommunicator::Event> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([&, this]() {
    result = communicator_->Send(send_buffer, dtype, count, peer, executor);
    return absl::OkStatus();
  }));
  return result;
}

tsl::AsyncValueRef<NcclCommunicator::Event> ThreadSafeNcclCommunicator::Recv(
    se::DeviceMemoryBase recv_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  tsl::AsyncValueRef<NcclCommunicator::Event> result;
  TF_RETURN_IF_ERROR(worker_thread_.Run([&, this]() {
    result = communicator_->Recv(recv_buffer, dtype, count, peer, executor);
    return absl::OkStatus();
  }));
  return result;
}

}  // namespace xla::gpu
