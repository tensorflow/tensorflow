/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/all_reduce_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/rendezvous.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

constexpr int64_t kMaxOneShotAllReduceSizeBytes = 256 * 1024;

// Contains the values that are passed between host threads with rendezvous.
struct RendezvousValue {
  RankId rank;
  se::DeviceMemoryBase input_buffer;
  se::Event* start_event;
  se::Event* end_event;

  bool operator<(const RendezvousValue& other) const {
    return rank < other.rank;
  }
};

// Executes the rendezvous before the kernel start.
// Inserts CUDA events into the stream to ensure that all devices have reached
// the start event before the kernel starts.
absl::StatusOr<std::shared_ptr<std::vector<RendezvousValue>>>
RendezvousBeforeKernelStart(const GpuCliqueKey& clique_key, RankId rank,
                            int64_t num_ranks,
                            const se::DeviceMemoryBase& input_buffer,
                            se::Stream& stream, se::Event* start_event,
                            se::Event* end_event) {
  RendezvousValue rendezvous_value;
  rendezvous_value.rank = rank;
  rendezvous_value.input_buffer = input_buffer;
  rendezvous_value.start_event = start_event;
  rendezvous_value.end_event = end_event;

  // Record that this device has started executing the kernel. We do
  // this before the rendezvous to make sure that RecordEvent is called before
  // WaitFor on another stream.
  TF_RETURN_IF_ERROR(stream.RecordEvent(start_event));

  auto rendezvous_fn = [](absl::Span<const RendezvousValue* const> values) {
    std::vector<RendezvousValue> values_copy;
    for (const auto& value : values) {
      values_copy.push_back(*value);
    }
    // Sort to make sure that values are in the same order as the devices are
    // ordered in the communicator.
    absl::c_sort(values_copy);
    return values_copy;
  };

  std::string start_rendezvous_key =
      absl::StrFormat("start one-shot all-reduce for rank %d, clique %s",
                      rank.value(), clique_key.ToString());
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<std::vector<RendezvousValue>> rendezvous_values,
      Rendezvous<std::vector<RendezvousValue>>(
          /*name=*/start_rendezvous_key, /*key=*/clique_key,
          /*value=*/rendezvous_value, /*num_threads=*/num_ranks,
          rendezvous_fn));

  // Wait for all devices to reach the start event. This indicates that all
  // output buffers are ready for transfer.
  for (auto& value : *rendezvous_values) {
    TF_RETURN_IF_ERROR(stream.WaitFor(value.start_event));
  }

  return rendezvous_values;
}

// Executes the rendezvous after the kernel finish. Waits for all devices to
// reach the end event.
absl::Status RendezvousAfterKernelFinish(
    const GpuCliqueKey& clique_key, RankId rank, int64_t num_ranks,
    se::Stream& stream, se::Event* end_event,
    const std::shared_ptr<std::vector<RendezvousValue>>& rendezvous_values) {
  // Record that this device has finished executing the kernel.
  TF_RETURN_IF_ERROR(stream.RecordEvent(end_event));

  // Do another rendezvous to make sure that we call RecordEvent for end_event
  // before WaitFor on another stream.
  std::string finish_rendezvous_key =
      absl::StrFormat("finish one-shot all-reduce for rank %d, clique %s",
                      rank.value(), clique_key.ToString());
  TF_RETURN_IF_ERROR(Rendezvous(/*name=*/finish_rendezvous_key,
                                /*key=*/clique_key,
                                /*num_threads=*/num_ranks));

  // Wait for all devices to reach the end event. This indicates that all
  // updates from other devices have arrived.
  for (auto& value : *rendezvous_values) {
    TF_RETURN_IF_ERROR(stream.WaitFor(value.end_event));
  }

  return absl::OkStatus();
}

absl::Status RunOneShotAllReduce(const GpuCliqueKey& clique_key, RankId rank,
                                 std::vector<DeviceBufferPair>& buffers,
                                 se::Stream& stream, Communicator* comm,
                                 se::DeviceMemoryBase local_buffer,
                                 se::Event* start_event, se::Event* end_event) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing one-shot all-reduce from device ordinal: "
          << device_ordinal;

  // TODO(b/407736956): Support variadic all-reduce.
  if (buffers.size() > 1) {
    return absl::UnimplementedError(
        "One-shot kernel does not support variadic all-reduce");
  }
  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  const DeviceBufferPair& buffer = buffers[0];

  // Buffer assignment aliases the source buffer to the destination buffer. This
  // works for NCCL implementation, but for one-shot kernel, input and output
  // buffers should be different. We do not have enough information at buffer
  // assignement time to change aliasing, so we allocate a new device buffer
  // ourselves and copy the data to it.
  // TODO(b/407736956): Fuse the copy into the one-shot kernel.
  TF_RETURN_IF_ERROR(stream.MemcpyD2D(&local_buffer, buffer.source_buffer,
                                      buffer.source_buffer.size()));

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<std::vector<RendezvousValue>> rendezvous_values,
      RendezvousBeforeKernelStart(clique_key, rank, num_ranks, local_buffer,
                                  stream, start_event, end_event));

  absl::InlinedVector<se::DeviceMemoryBase, 4> input_ptrs;
  for (auto& value : *rendezvous_values) {
    input_ptrs.push_back(value.input_buffer);
  }

  TF_RETURN_IF_ERROR(RunAllReduceKernel(&stream, buffer.element_type,
                                        input_ptrs, buffer.destination_buffer,
                                        num_ranks, buffer.element_count));

  TF_RETURN_IF_ERROR(RendezvousAfterKernelFinish(
      clique_key, rank, num_ranks, stream, end_event, rendezvous_values));

  return absl::OkStatus();
}

absl::Status CheckImplementableInst(const HloInstruction* inst,
                                    Thunk::Kind reduction_op) {
  for (HloInstruction* operand : inst->operands()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand->shape(), reduction_op));
  }

  if (!MatchReductionComputation(inst->called_computations().front())
           .has_value()) {
    return absl::UnimplementedError("Unrecognized reduction computation");
  }

  return absl::OkStatus();
}

template <typename HloInstType>
AllReduceConfig GetAllReduceConfigInst(HloInstType* inst) {
  std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(inst->called_computations().front());
  CHECK(reduction_kind.has_value());

  AllReduceConfig config;
  config.config = GetCollectiveConfig(inst, inst->use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename HloInstType>
CollectiveOpGroupMode GetGroupModeInst(HloInstType* inst) {
  return GetAllReduceConfigInst(inst).config.group_mode;
}

}  // namespace

absl::Status RunAllReduce(GpuCollectives* collectives,
                          ReductionKind reduction_kind,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_ASSIGN_OR_RETURN(GpuCommunicator * gpu_comm, collectives->TryCast(comm));
  tsl::AsyncValueRef<Communicator::Event> event =
      gpu_comm->GroupExecute([reduction_kind, &buffers,
                              &stream](GpuCommunicator* comm) -> absl::Status {
        for (DeviceBufferPair& buffer : buffers) {
          TF_RETURN_IF_ERROR(comm->LaunchAllReduce(
              buffer.source_buffer, buffer.destination_buffer,
              buffer.element_type, buffer.element_count, reduction_kind,
              GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });
  tsl::BlockUntilReady(event);
  if (event.IsError()) {
    return event.GetError();
  }
  return absl::OkStatus();
}

AllReduceReduceScatterThunkBase::AllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, AllReduceConfig config,
    std::vector<Buffer> buffers, bool is_sync)
    : CollectiveThunk(kind, thunk_info, is_sync, AsyncStreamKind::kCollective),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

AllReduceStartThunk::AllReduceStartThunk(ThunkInfo thunk_info,
                                         const HloAllReduceInstruction* inst,
                                         std::vector<Buffer> buffers,
                                         bool p2p_memcpy_enabled)
    : AllReduceReduceScatterThunkBase(
          Thunk::kAllReduceStart, thunk_info, GetAllReduceConfigInst(inst),
          std::move(buffers), IsGPUSyncCollective(*inst)),
      one_shot_kernel_enabled_(
          inst->GetModule()
              ->config()
              .debug_options()
              .xla_gpu_unsupported_use_all_reduce_one_shot_kernel()) {}

absl::Status AllReduceStartThunk::CheckImplementable(
    const HloAllReduceInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<AllReduceStartThunk>(
      CheckImplementableInst(inst, Thunk::kAllReduceStart), inst, replica_count,
      partition_count);
}

CollectiveOpGroupMode AllReduceStartThunk::GetGroupMode(
    const HloAllReduceInstruction* inst) {
  return GetGroupModeInst(inst);
}

absl::StatusOr<bool> AllReduceStartThunk::ShouldUseOneShotAllReduceKernel(
    const GpuCliqueKey& clique_key,
    const CollectiveCliques* collective_cliques) {
  if (!one_shot_kernel_enabled_) {
    return false;
  }

  // TODO(b/407736956): Support variadic all-reduce.
  if (buffers_.size() != 1) {
    return false;
  }

  int64_t num_elements = buffers_[0].element_count;
  PrimitiveType element_type = config().operand_element_type[0];

  int64_t input_size_bytes =
      num_elements * ShapeUtil::ByteSizeOfPrimitiveType(element_type);

  // One-shot all-reduce is only beneficial for small inputs.
  if (input_size_bytes > kMaxOneShotAllReduceSizeBytes) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool peer_access_enabled,
                      collective_cliques->peer_access_enabled(clique_key));

  // Check that peer access is enabled.
  if (!peer_access_enabled) {
    return false;
  }

  return IsAllReduceKernelSupported(clique_key.num_local_participants(),
                                    num_elements,
                                    config().operand_element_type[0]);
}

absl::Status AllReduceStartThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(collectives, *params.collective_params,
                      config().replica_groups, config().group_mode,
                      GetAsyncStreamKind()));

  TF_ASSIGN_OR_RETURN(
      bool use_one_shot_kernel,
      ShouldUseOneShotAllReduceKernel(clique_key, params.collective_cliques));

  if (use_one_shot_kernel) {
    absl::MutexLock lock(&mutex_);

    if (!local_buffer_allocs_.contains(params.executor)) {
      int64_t max_size = 0;
      for (auto buffer : buffers_) {
        max_size = std::max(max_size, buffer.source_buffer.size());
      }

      se::DeviceMemoryHandle local_buffer_alloc(
          params.executor, params.executor->Allocate(max_size));

      local_buffer_allocs_.emplace(params.executor,
                                   std::move(local_buffer_alloc));
    }

    if (!start_events_.contains(params.executor)) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Event> event,
                          params.executor->CreateEvent());
      start_events_.emplace(params.executor, std::move(event));
    }

    if (!end_events_.contains(params.executor)) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Event> event,
                          params.executor->CreateEvent());
      end_events_.emplace(params.executor, std::move(event));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> AllReduceStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));

  TF_ASSIGN_OR_RETURN(bool use_one_shot_kernel,
                      ShouldUseOneShotAllReduceKernel(
                          comm_handle.clique_key, params.collective_cliques));

  if (use_one_shot_kernel) {
    se::Event* start_event = nullptr;
    se::Event* end_event = nullptr;
    se::DeviceMemoryBase local_buffer;
    {
      absl::MutexLock lock(&mutex_);
      local_buffer = local_buffer_allocs_[stream.parent()].memory();
      start_event = start_events_[stream.parent()].get();
      end_event = end_events_[stream.parent()].get();
    }

    std::optional<RankId> rank =
        comm_handle.clique_key.rank(params.collective_params->global_device_id);

    TF_RETURN_IF_ERROR(RunOneShotAllReduce(
        comm_handle.clique_key, *rank, device_buffers, stream, comm_handle.comm,
        local_buffer, start_event, end_event));
    return false;
  }

  TF_RETURN_IF_ERROR(RunAllReduce(collectives, config_.reduction_kind,
                                  device_buffers, stream, comm_handle.comm));
  return true;
}

ReduceScatterStartThunk::ReduceScatterStartThunk(
    ThunkInfo thunk_info, const HloReduceScatterInstruction* inst,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled)
    : AllReduceReduceScatterThunkBase(
          Thunk::kReduceScatterStart, thunk_info, GetAllReduceConfigInst(inst),
          std::move(buffers), IsGPUSyncCollective(*inst)) {}

/*static*/ absl::Status ReduceScatterStartThunk::CheckImplementable(
    const HloReduceScatterInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<ReduceScatterStartThunk>(
      CheckImplementableInst(inst, Thunk::kReduceScatterStart), inst,
      replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode ReduceScatterStartThunk::GetGroupMode(
    const HloReduceScatterInstruction* inst) {
  return GetGroupModeInst(inst);
}

absl::StatusOr<bool> ReduceScatterStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  TF_RETURN_IF_ERROR(RunReduceScatter(collectives, config_.reduction_kind,
                                      device_buffers, stream,
                                      comm_handle.comm));
  return true;
}

absl::Status RunReduceScatter(GpuCollectives* collectives,
                              ReductionKind reduction_kind,
                              std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  TF_ASSIGN_OR_RETURN(GpuCommunicator * gpu_comm, collectives->TryCast(comm));
  tsl::AsyncValueRef<Communicator::Event> event =
      gpu_comm->GroupExecute([num_ranks, reduction_kind, &buffers,
                              &stream](GpuCommunicator* comm) -> absl::Status {
        for (DeviceBufferPair& buffer : buffers) {
          // buffer.element_count is the source buffers element count. For
          // ncclReduceScatter, we need the destination buffers element count.
          TF_RET_CHECK(buffer.element_count % num_ranks == 0)
              << "Source buffer was not an exact multiple of the number of "
                 "participants.";

          TF_RETURN_IF_ERROR(comm->LaunchReduceScatter(
              buffer.source_buffer, buffer.destination_buffer,
              buffer.element_type, buffer.element_count / num_ranks,
              reduction_kind, GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });
  tsl::BlockUntilReady(event);
  if (event.IsError()) {
    return event.GetError();
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
