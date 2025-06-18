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

#ifndef XLA_BACKENDS_CPU_RUNTIME_COLLECTIVE_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_COLLECTIVE_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class CollectiveThunk : public Thunk {
  using Thunk::Thunk;

 public:
  enum class CollectiveKind {
    kAllGather,
    kAllReduce,
    kAllToAll,
    kCollectivePermute,
    kReduceScatter,
  };

  static absl::string_view CollectiveKindToString(CollectiveKind kind);

  // Parameters of the collective operation behind the collective thunk. We rely
  // on them to construct the rendezvous key and to find a thunk "location" in
  // the collective operation "clique" (group of communicating devices).
  struct OpParams {
    int64_t op_id;
    bool has_channel_id;
    std::optional<bool> use_global_device_ids;
    std::vector<ReplicaGroup> group;
  };

  // Source and destination buffers for the collective operation.
  struct OpBuffers {
    std::vector<BufferAllocation::Slice> source_buffers;
    std::vector<Shape> source_shapes;

    std::vector<BufferAllocation::Slice> destination_buffers;
    std::vector<Shape> destination_shapes;
  };

  // Resources used by the collective operation.
  struct OpResources {
    std::shared_ptr<Resource> communicator_resource;
  };

  // Device memory resolved for the collective operation buffers.
  struct OpDeviceMemory {
    absl::InlinedVector<se::DeviceMemoryBase, 4> source;
    absl::InlinedVector<se::DeviceMemoryBase, 4> destination;
  };

  CollectiveThunk(CollectiveKind collective_kind, Thunk::Info info,
                  OpParams op_params, OpBuffers op_buffers,
                  OpResources op_resources);

  const OpParams& op_params() const { return op_params_; }

  const OpBuffers& op_buffers() const { return op_buffers_; }

  const OpResources& op_resources() const { return op_resources_; }

  CollectiveKind collective_kind() const { return collective_kind_; }

  // Resolves operation's device memory from the buffers and buffer allocations.
  absl::StatusOr<OpDeviceMemory> GetOpDeviceMemory(const ExecuteParams& params);

  BufferUses buffer_uses() const final;
  ResourceUses resource_uses() const final;

  // Callback for collective thunk implementations.
  using Callback = absl::AnyInvocable<tsl::AsyncValueRef<Communicator::Event>(
      const RendezvousKey& key, Communicator& comm)>;

  static bool IsDataTypeSupportedByCollectiveReduce(PrimitiveType datatype);

  absl::Duration DefaultCollectiveTimeout();

  absl::StatusOr<RendezvousKey> GetRendezvousKey(
      const Thunk::CollectiveExecuteParams& params);

  absl::StatusOr<int32_t> RankInGlobalDevices(const RendezvousKey& key,
                                              GlobalDeviceId device);

  // Acquires collective communicator for the given parameters and executes the
  // user provided callback with acquired rendezvous key, rank and communicator.
  tsl::AsyncValueRef<ExecuteEvent> ExecuteWithCommunicator(
      const Thunk::CollectiveExecuteParams* params, Callback callback);

  const BufferAllocation::Slice& source_buffer(int64_t index) const;
  absl::Span<const BufferAllocation::Slice> source_buffers() const;

  const Shape& source_shape(int64_t index) const;

  const BufferAllocation::Slice& destination_buffer(int64_t index) const;
  absl::Span<const BufferAllocation::Slice> destination_buffers() const;

  const Shape& destination_shape(int64_t index) const;

 private:
  OpParams op_params_;
  OpBuffers op_buffers_;
  OpResources op_resources_;
  CollectiveKind collective_kind_;
};

std::ostream& operator<<(std::ostream& os,
                         CollectiveThunk::CollectiveKind kind);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_COLLECTIVE_THUNK_H_
