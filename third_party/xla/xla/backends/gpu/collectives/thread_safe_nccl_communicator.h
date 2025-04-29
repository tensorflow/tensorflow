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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_THREAD_SAFE_NCCL_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_THREAD_SAFE_NCCL_COMMUNICATOR_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/nccl_communicator.h"
#include "xla/backends/gpu/collectives/worker_thread.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
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

// A thread-safe wrapper around a NcclCommunicator.
//
// The NCCL documentation [1] is a bit terse, but NCCL communicators
// (concretely, instances of ncclComm_t) have odd thread safety properties. A
// NCCL communicator cannot be used by multiple threads, even serially. A NCCL
// communicator must be created by, live on, and be destroyed by a single
// thread.
//
// ThreadSafeNcclCommunicator enforces these strict threading requirements.
// ThreadSafeNcclCommunicator is thread-safe, and internally it dispatches all
// operations to a single long-lived thread.
//
// [1]:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
class ThreadSafeNcclCommunicator : public Communicator {
 public:
  // Constructs a ThreadSafeNcclCommunicator that uses the ncclComm_t returned
  // by f. f and all operations performed on the resulting ncclComm_t are
  // executed on the same long-lived thread.
  static absl::StatusOr<std::unique_ptr<ThreadSafeNcclCommunicator>> Create(
      absl::AnyInvocable<absl::StatusOr<ncclComm_t>() &&> f,
      tsl::Env& env = *tsl::Env::Default());

  // ThreadSafeNcclCommunicator is not copyable or movable.
  ThreadSafeNcclCommunicator(const ThreadSafeNcclCommunicator&) = delete;
  ThreadSafeNcclCommunicator(ThreadSafeNcclCommunicator&&) = delete;
  ThreadSafeNcclCommunicator& operator=(const ThreadSafeNcclCommunicator&) =
      delete;
  ThreadSafeNcclCommunicator& operator=(ThreadSafeNcclCommunicator&&) = delete;

  // These methods dispatch to the underlying NcclCommunicator.
  absl::Status GroupStart();
  absl::Status GroupEnd();
  absl::Status Abort() override;
  absl::Status HealthCheck() const override;
  absl::StatusOr<size_t> NumRanks() const override;
  absl::StatusOr<std::unique_ptr<RegisteredBufferHandle>> RegisterBuffer(
      se::DeviceMemoryBase buffer) override;
  tsl::AsyncValueRef<NcclCommunicator::Event> AllReduce(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
      const Executor& executor) override;
  tsl::AsyncValueRef<NcclCommunicator::Event> Broadcast(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, RankId root,
      const Executor& executor) override;
  tsl::AsyncValueRef<NcclCommunicator::Event> ReduceScatter(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
      const Executor& executor) override;
  tsl::AsyncValueRef<NcclCommunicator::Event> AllGather(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, const Executor& executor) override;
  tsl::AsyncValueRef<NcclCommunicator::Event> AllToAll(
      absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) override;
  tsl::AsyncValueRef<NcclCommunicator::Event> CollectivePermute(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) override;
  tsl::AsyncValueRef<NcclCommunicator::Event> Send(
      se::DeviceMemoryBase send_buffer, PrimitiveType dtype, size_t count,
      RankId peer, const Executor& executor) override;
  tsl::AsyncValueRef<NcclCommunicator::Event> Recv(
      se::DeviceMemoryBase recv_buffer, PrimitiveType dtype, size_t count,
      RankId peer, const Executor& executor) override;
  std::string ToString() const override { return communicator_->ToString(); }
  ncclComm_t comm() const { return communicator_->comm(); }

 private:
  explicit ThreadSafeNcclCommunicator(tsl::Env& env)
      : worker_thread_(
            env, /*thread_name=*/"ThreadSafeNcclCommunicator-WorkerThread") {}

  // Thread on which all operations are performed.
  mutable WorkerThread worker_thread_;

  // Underlying communicator.
  std::unique_ptr<NcclCommunicator> communicator_;  // underlying communicator
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_THREAD_SAFE_NCCL_COMMUNICATOR_H_
