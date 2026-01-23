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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"

#if NCCL_VERSION_CODE >= 22800
// Device initiated collective operations were added in NCCL 2.28.0.
#include "third_party/nccl/nccl_device.h"
#endif  // NCCL_VERSION_CODE >= 22800

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
      se::StreamExecutor* stream_executor,
      absl::AnyInvocable<absl::StatusOr<ncclComm_t>()> make_comm,
      std::shared_ptr<CancellationToken> cancel, bool is_async = false,
      tsl::Env& env = *tsl::Env::Default());

  ~NcclCommunicator() override;

  // NcclCommunicator is not copyable or movable.
  NcclCommunicator(const NcclCommunicator&) = delete;
  NcclCommunicator(NcclCommunicator&&) = delete;
  NcclCommunicator& operator=(const NcclCommunicator&) = delete;
  NcclCommunicator& operator=(NcclCommunicator&&) = delete;

  absl::Status Abort() final;
  absl::Status HealthCheck() const final;
  absl::StatusOr<size_t> NumRanks() const final;

  PlatformCommunicatorHandle platform_comm() const final {
    return PlatformCommunicatorHandle{comm_};
  }

  bool SupportsDeviceComm() const final;

  absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>> CreateDeviceComm(
      const GpuDeviceCommunicator::Requirements& requirements) final;

  absl::StatusOr<std::unique_ptr<SymmetricMemory>> CreateSymmetricMemory(
      se::DeviceAddressBase addr) final;

  // Since each XLA buffer is a slice into a larger BFCAllocator chunk, first
  // get the base address of buffer. We will use the base address to keep track
  // of which chunks we have registered.
  absl::Status RegisterBufferOnce(se::DeviceAddressBase buffer_range,
                                  int device_ordinal,
                                  bool use_symmetric_buffer) final;

  Future<> GroupExecute(
      absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) final;

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
                size_t count, RankId peer, const Executor& executor) final;

  Future<> Recv(se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  std::string ToString() const final;

  ncclComm_t comm() const { return comm_; }

  se::StreamExecutor* stream_executor() const { return stream_executor_; }

 private:
  absl::StatusOr<std::unique_ptr<RegisteredBufferHandle>> RegisterBuffer(
      se::DeviceAddressBase buffer, int device_ordinal,
      bool use_symmetric_buffer);

  class NcclRegisteredBufferHandle;

  NcclCommunicator(se::StreamExecutor* stream_executor, ncclComm_t comm,
                   std::unique_ptr<tsl::Executor> executor,
                   std::shared_ptr<CancellationToken> cancel)
      : stream_executor_(stream_executor),
        comm_(comm),
        executor_(std::move(executor)),
        cancel_(std::move(cancel)) {
    VLOG(1) << "Created NCCL communicator" << *this << " on device ordinal "
            << stream_executor_->device_ordinal();
  }

  absl::Status GroupStart();
  absl::Status GroupEnd();

  absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               const Executor& executor) final;

  absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count, RankId root,
                               const Executor& executor) final;

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
      PrimitiveType dtype, size_t count, const Executor& executor) final;

  absl::Status LaunchCollectivePermute(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       std::optional<RankId> source_rank,
                                       absl::Span<const RankId> target_ranks,
                                       const Executor& executor) final;

  absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final;

  absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final;

  // Polls the communicator until any pending non-blocking operations are "done"
  // or aborted.
  absl::Status PollUntilDone() const;

  // Executes f on executor_, or calls f directly if executor_ is null.
  Future<> Execute(absl::AnyInvocable<absl::Status() &&> f) const;

  // Executes f on executor_, or calls f directly if executor_ is null.
  template <typename T>
  Future<T> Execute(absl::AnyInvocable<absl::StatusOr<T>() &&> f) const;

  absl::Status ExecuteAwait(absl::AnyInvocable<absl::Status() &&> f) const {
    return Execute(std::move(f)).Await();
  }

  template <typename T>
  absl::StatusOr<T> ExecuteAwait(
      absl::AnyInvocable<absl::StatusOr<T>() &&> f) const {
    return Execute<T>(std::move(f)).Await();
  }

  // The stream executor (underlying GPU device) on which this communicator is
  // instantiated. We need to know the stream executor to be able to active
  // context for all operations that create or destroy device comms.
  se::StreamExecutor* stream_executor_;

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
  std::unique_ptr<tsl::Executor> executor_;

  // Should all pending collectives cancel?
  std::shared_ptr<CancellationToken> cancel_;

  // Has comm_ been aborted?
  bool aborted_ = false;

  // Nesting level of current NCCL group
  int group_nesting_level_ = 0;

  // Keep track of which communicators we have registered for already.
  // Each ncclMemAlloc'd buffer needs to be registered once per comm.
  struct RegisteredBuffers {
    absl::Mutex mu;
    // Buffer range to the registered buffer handle.
    absl::flat_hash_map<void*,
                        std::unique_ptr<Communicator::RegisteredBufferHandle>>
        range_to_handle ABSL_GUARDED_BY(mu);
  };
  RegisteredBuffers registered_buffers_;
};

//===----------------------------------------------------------------------===//
// NCCL device communicator
//===----------------------------------------------------------------------===//

#if NCCL_VERSION_CODE >= 22800

// A device-side NCCL communicator.
class NcclDeviceCommunicator : public GpuDeviceCommunicator {
 public:
  ~NcclDeviceCommunicator() override;

  NcclDeviceCommunicator(NcclDeviceCommunicator&&) = delete;
  NcclDeviceCommunicator& operator=(NcclDeviceCommunicator&&) = delete;

  // Creates a new instance of a NCCL device communicator from the given host
  // communicator object.
  static absl::StatusOr<std::unique_ptr<NcclDeviceCommunicator>> CreateFrom(
      const NcclCommunicator& comm, const Requirements& requirements);

  PlatformCommunicatorHandle platform_comm() const final;

  std::string ToString() const final;

  PackedKernelArg PackKernelArg() const final;

 private:
  NcclDeviceCommunicator(const NcclCommunicator* comm, ncclDevComm dev_comm);

  const NcclCommunicator* comm_;
  ncclDevComm dev_comm_;
};

#endif  // NCCL_VERSION_CODE >= 22800

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COMMUNICATOR_H_
