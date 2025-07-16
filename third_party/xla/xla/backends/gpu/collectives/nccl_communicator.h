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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COMMUNICATOR_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"

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

// XLA collectives communicator wrapping an NCCL communicator.
class NcclCommunicator : public GpuCommunicator {
 public:
  // Creates a NCCL communicator.
  //
  // make_comm should construct and return a new ncclComm_t. For example, it
  // could call ncclCommInitRank. make_comm should not return a ncclComm_t that
  // was created by a different thread.
  //
  // If is_async is true, all collective methods (e.g., AllReduce) are performed
  // asynchronously on a separate thread. Otherwise, they are performed
  // synchronously on the calling thread.
  static absl::StatusOr<std::unique_ptr<NcclCommunicator>> Create(
      absl::AnyInvocable<absl::StatusOr<ncclComm_t>()> make_comm,
      bool is_async = false, tsl::Env& env = *tsl::Env::Default());

  ~NcclCommunicator() override;

  // NcclCommunicator is not copyable or movable.
  NcclCommunicator(const NcclCommunicator&) = delete;
  NcclCommunicator(NcclCommunicator&&) = delete;
  NcclCommunicator& operator=(const NcclCommunicator&) = delete;
  NcclCommunicator& operator=(NcclCommunicator&&) = delete;

  absl::Status Abort() final;
  absl::Status HealthCheck() const final;
  absl::StatusOr<size_t> NumRanks() const final;

  absl::StatusOr<std::unique_ptr<RegisteredBufferHandle>> RegisterBuffer(
      se::DeviceMemoryBase buffer) final;

  tsl::AsyncValueRef<Communicator::Event> GroupExecute(
      absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) final;

  tsl::AsyncValueRef<Event> AllReduce(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      ReductionKind reduction_kind,
                                      const Executor& executor) final;

  tsl::AsyncValueRef<Event> Broadcast(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      RankId root,
                                      const Executor& executor) final;

  tsl::AsyncValueRef<Event> ReduceScatter(se::DeviceMemoryBase send_buffer,
                                          se::DeviceMemoryBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          ReductionKind reduction_kind,
                                          const Executor& executor) final;

  tsl::AsyncValueRef<Event> AllGather(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      const Executor& executor) final;

  tsl::AsyncValueRef<Event> AllToAll(
      absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) final;

  tsl::AsyncValueRef<Event> CollectivePermute(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) final;

  tsl::AsyncValueRef<Event> Send(se::DeviceMemoryBase send_buffer,
                                 PrimitiveType dtype, size_t count, RankId peer,
                                 const Executor& executor) final;

  tsl::AsyncValueRef<Event> Recv(se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count, RankId peer,
                                 const Executor& executor) final;

  std::string ToString() const final;

  ncclComm_t comm() const { return comm_; }

 private:
  class NcclRegisteredBufferHandle;

  explicit NcclCommunicator(ncclComm_t comm,
                            std::unique_ptr<tsl::AsyncValue::Executor> executor)
      : comm_(comm), executor_(std::move(executor)) {
    VLOG(1) << "Created " << *this;
  }

  absl::Status GroupStart();
  absl::Status GroupEnd();

  absl::Status LaunchAllReduce(se::DeviceMemoryBase send_buffer,
                               se::DeviceMemoryBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               const Executor& executor) final;

  absl::Status LaunchBroadcast(se::DeviceMemoryBase send_buffer,
                               se::DeviceMemoryBase recv_buffer,
                               PrimitiveType dtype, size_t count, RankId root,
                               const Executor& executor) final;

  absl::Status LaunchReduceScatter(se::DeviceMemoryBase send_buffer,
                                   se::DeviceMemoryBase recv_buffer,
                                   PrimitiveType dtype, size_t count,
                                   ReductionKind reduction_kind,
                                   const Executor& executor) final;

  absl::Status LaunchAllGather(se::DeviceMemoryBase send_buffer,
                               se::DeviceMemoryBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               const Executor& executor) final;

  absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) final;

  absl::Status LaunchCollectivePermute(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       std::optional<RankId> source_rank,
                                       absl::Span<const RankId> target_ranks,
                                       const Executor& executor) final;

  absl::Status LaunchSend(se::DeviceMemoryBase send_buffer, PrimitiveType dtype,
                          size_t count, RankId peer,
                          const Executor& executor) final;

  absl::Status LaunchRecv(se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                          size_t count, RankId peer,
                          const Executor& executor) final;

  // Polls the communicator until any pending non-blocking operations are "done"
  // or aborted.
  absl::Status PollUntilDone() const;

  // Executes f on executor_, or calls f directly if executor_ is null.
  tsl::AsyncValueRef<Event> Execute(absl::AnyInvocable<absl::Status()> f) const;

  // Executes f on executor_, or calls f directly if executor_ is null.
  template <typename T>
  tsl::AsyncValueRef<T> Execute(
      absl::AnyInvocable<absl::StatusOr<T>()> f) const;

  // Underlying NCCL communicator.
  ncclComm_t comm_;

  // If not null, used to execute methods.
  //
  // NCCL communicators (instances of ncclComm_t) are not thread safe. Thus,
  // multiple threads cannot concurrently access the same ncclComm_t. This is
  // not surprising. What is very surprising is that multiple threads cannot
  // serially access the same ncclComm_t. In fact, a ncclComm_t must be created
  // by, live on, and be destroyed by a single thread. A ncclComm_t cannot be
  // accessed by any thread except the one that created it. To accomplish this,
  // we perform all comm_ operations on executor_, if it is not null.
  //
  // Concretely, the lack of thread safety comes from the fact that the NCCL
  // code uses thread-local variables that do not work properly when a
  // ncclComm_t is accessed from multiple threads. Emperically, the lack of
  // thread safety only manifests as buggy behavior when using non-blocking
  // communicators.
  std::unique_ptr<tsl::AsyncValue::Executor> executor_;

  // Should all pending collectives cancel?
  std::atomic_bool canceling_ = false;

  // Has comm_ been aborted?
  bool aborted_ = false;

  // Nesting level of current NCCL group
  int group_nesting_level_ = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COMMUNICATOR_H_
