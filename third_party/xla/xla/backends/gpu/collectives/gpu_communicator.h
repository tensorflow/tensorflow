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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_

#include <cstddef>
#include <optional>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::gpu {

// GpuCommunicator extends Communicator with synchronous versions of the
// collective methods.
//
// For example, the Communicator::AllReduce method (which is asynchronous and
// returns an AsyncValueRef<Event>) has a corresponding
// GpuCommunicator::LaunchAllReduce method (which is synchronous and returns an
// absl::Status).
class GpuCommunicator : public Communicator {
 public:
  ~GpuCommunicator() override = default;

  // Executes f in a group. f should invoke synchronous collective methods like
  // LaunchAllReduce and not asynchronous collective methods like AllReduce.
  virtual tsl::AsyncValueRef<Communicator::Event> GroupExecute(
      absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) = 0;

  virtual absl::Status LaunchAllReduce(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       ReductionKind reduction_kind,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchBroadcast(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       RankId root,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchReduceScatter(se::DeviceMemoryBase send_buffer,
                                           se::DeviceMemoryBase recv_buffer,
                                           PrimitiveType dtype, size_t count,
                                           ReductionKind reduction_kind,
                                           const Executor& executor) = 0;

  virtual absl::Status LaunchAllGather(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) = 0;

  virtual absl::Status LaunchCollectivePermute(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) = 0;

  virtual absl::Status LaunchSend(se::DeviceMemoryBase send_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) = 0;

  virtual absl::Status LaunchRecv(se::DeviceMemoryBase recv_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_
