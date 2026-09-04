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

#include "xla/backends/gpu/runtime/collective_reduce_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

CollectiveReduceThunk::CollectiveReduceThunk(ThunkInfo thunk_info,
                                             AllReduceConfig config,
                                             std::vector<Buffer> buffers,
                                             bool has_dynamic_root)
    : AllReduceReduceScatterThunkBase(Thunk::kCollectiveReduce, thunk_info,
                                      std::move(config), std::move(buffers)),
      has_dynamic_root_(has_dynamic_root) {}

CollectiveReduceThunk::CollectiveReduceThunk(
    ThunkInfo thunk_info, const HloCollectiveReduceInstruction* inst,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled, bool has_dynamic_root)
    : AllReduceReduceScatterThunkBase(Thunk::kCollectiveReduce, thunk_info,
                                      GetAllReduceConfigInst(inst),
                                      std::move(buffers)),
      has_dynamic_root_(has_dynamic_root) {}

/*static*/ absl::Status CollectiveReduceThunk::CheckImplementable(
    const HloCollectiveReduceInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&]() -> absl::Status {
    // The trailing S32 root operand (dynamic root) is not reduced, so exclude
    // it from operand validation.
    int64_t num_data_operands = inst->has_dynamic_root()
                                    ? inst->operand_count() - 1
                                    : inst->operand_count();
    for (int64_t i = 0; i < num_data_operands; ++i) {
      ABSL_RETURN_IF_ERROR(
          IsValidOperand(inst->operand(i)->shape(), Thunk::kCollectiveReduce));
    }
    if (!MatchReductionComputation(inst->called_computations().front())
             .has_value()) {
      return absl::UnimplementedError("Unrecognized reduction computation");
    }
    return absl::OkStatus();
  }();
  return AddOpDescription<CollectiveReduceThunk>(status, inst, replica_count,
                                                 partition_count);
}

/*static*/ CollectiveOpGroupMode CollectiveReduceThunk::GetGroupMode(
    const HloCollectiveReduceInstruction* inst) {
  return GetAllReduceConfigInst(inst).config.group_mode;
}

absl::Status CollectiveReduceThunk::InitializeCollective(
    const InitializeParams& params, const GpuCliqueKey& clique_key) {
  if (!has_dynamic_root_) {
    return absl::OkStatus();
  }
  se::StreamExecutor* executor = params.executor;
  absl::MutexLock lock(mutex_);
  std::unique_ptr<CollectiveReduceMetadata>& metadata =
      per_executor_metadata_[executor];
  if (metadata == nullptr) {
    metadata = std::make_unique<CollectiveReduceMetadata>();
  }
  if (metadata->reduce_roots == nullptr) {
    // The last buffer holds the runtime-selected root ranks (one S32 per
    // reduce); all other buffers are the data being reduced.
    metadata->num_roots = buffers().size() - 1;
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<se::MemoryAllocation> alloc,
        executor->HostMemoryAllocate(metadata->num_roots * sizeof(int32_t)));
    metadata->reduce_roots = std::move(alloc);
  }
  return absl::OkStatus();
}

absl::Status CollectiveReduceThunk::RunCollective(
    const ExecuteParams& params, const GpuCliqueKey& clique_key,
    se::Stream& stream, Communicator& comm) {
  ABSL_ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params.buffer_allocations, buffers(),
                                          config_.config.operand_element_type));
  CollectiveReduceMetadata* metadata = nullptr;
  {
    absl::MutexLock lock(mutex_);
    metadata = per_executor_metadata_[stream.parent()].get();
  }
  return RunCollectiveReduce(config_.reduction_kind, device_buffers, stream,
                             comm, metadata, has_dynamic_root_);
}

absl::StatusOr<std::unique_ptr<CollectiveReduceThunk>>
CollectiveReduceThunk::FromProto(
    ThunkInfo thunk_info, const CollectiveReduceThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ABSL_ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  CollectiveConfig config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  ABSL_ASSIGN_OR_RETURN(ReductionKind reduction_kind,
                   FromReductionKindProto(thunk_proto.reduction_kind()));

  return std::make_unique<CollectiveReduceThunk>(
      std::move(thunk_info), AllReduceConfig{config, reduction_kind},
      std::move(buffers), thunk_proto.has_dynamic_root());
}

absl::StatusOr<ThunkProto> CollectiveReduceThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CollectiveReduceThunkProto* thunk_proto =
      proto.mutable_collective_reduce_thunk();

  for (const Buffer& buffer : buffers()) {
    ABSL_ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  thunk_proto->set_reduction_kind(ToReductionKindProto(config_.reduction_kind));
  thunk_proto->set_has_dynamic_root(has_dynamic_root_);

  return proto;
}

absl::Status RunCollectiveReduce(ReductionKind reduction_kind,
                                 std::vector<DeviceBufferPair>& buffers,
                                 se::Stream& stream, Communicator& comm,
                                 CollectiveReduceMetadata* metadata,
                                 bool has_dynamic_root) {
  int device_ordinal = stream.parent()->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal) << "Performing collective-reduce";

  if (has_dynamic_root && metadata) {
    // Stage the runtime-selected root ranks on the host so we can pick a root
    // per reduce below.
    DeviceBufferPair& roots_device_buffer = buffers.back();
    CHECK(metadata->reduce_roots != nullptr);
    ABSL_RETURN_IF_ERROR(stream.Memcpy(metadata->reduce_roots->address().opaque(),
                                  roots_device_buffer.source_buffer,
                                  roots_device_buffer.source_buffer.size()));
    if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
      return absl::InternalError(
          absl::StrFormat("Failed to copy dynamic roots on stream %p: %s",
                          &stream, blocked.message()));
    }
  }

  // With a dynamic root, the last buffer only carries the per-reduce root ranks
  // and must not itself be reduced.
  const int64_t num_reduces =
      (has_dynamic_root && metadata) ? buffers.size() - 1 : buffers.size();
  auto* gpu_comm = absl::down_cast<GpuCommunicator*>(&comm);

  if (has_dynamic_root && metadata) {
    ABSL_ASSIGN_OR_RETURN(size_t num_ranks, comm.NumRanks());
    int32_t* roots_ptr =
        reinterpret_cast<int32_t*>(metadata->reduce_roots->address().opaque());
    for (int64_t i = 0; i < num_reduces; ++i) {
      if (roots_ptr[i] < 0 || static_cast<size_t>(roots_ptr[i]) >= num_ranks) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "collective-reduce dynamic root %d at index %d is out of range "
            "[0, %d)",
            roots_ptr[i], i, num_ranks));
      }
    }
  }

  Future<> future = gpu_comm->GroupExecute([&]() -> absl::Status {
    RankId root(0);
    for (int64_t i = 0; i < num_reduces; ++i) {
      DeviceBufferPair& buffer = buffers[i];
      if (has_dynamic_root && metadata) {
        int32_t* roots_ptr = reinterpret_cast<int32_t*>(
            metadata->reduce_roots->address().opaque());
        root = RankId(roots_ptr[i]);
      }
      ABSL_RETURN_IF_ERROR(gpu_comm->LaunchReduce(
          buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
          buffer.element_count, reduction_kind, root,
          GpuCollectives::On(stream)));
    }
    return absl::OkStatus();
  });
  ABSL_RETURN_IF_ERROR(future.Await());
  XLA_VLOG_DEVICE(3, device_ordinal) << "Done performing collective-reduce";
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
