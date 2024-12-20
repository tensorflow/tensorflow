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

#include "xla/service/gpu/runtime/nccl_ragged_all_to_all_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// RaggedAllToAll has 4 operands with ragged tensor metadata: input_offsets,
// send_sizes, output_offsets, and recv_sizes.
constexpr int64_t kNumRaggedMetadataOperands = 4;

NcclRaggedAllToAllConfig GetNcclRaggedAllToAllConfig(
    const HloRaggedAllToAllInstruction* instr) {
  NcclRaggedAllToAllConfig config;
  config.config = GetNcclCollectiveConfig(instr, std::nullopt);
  config.num_ragged_rows = instr->operand(2)->shape().dimensions(0);
  config.ragged_row_element_size =
      ShapeUtil::ElementsIn(instr->shape()) / instr->shape().dimensions(0);
  return config;
}

// A wrapper around an raw data buffer that indexes values based on the
// PrimitiveType that is stored in the buffer.
class IntegerOperandData {
 public:
  IntegerOperandData(PrimitiveType element_type, void* data)
      : element_type_(element_type), data_(data) {}

  int64_t get(int i) const {
    switch (element_type_) {
      case PrimitiveType::S32:
      case PrimitiveType::U32:
        return reinterpret_cast<int32_t*>(data_)[i];
      case PrimitiveType::S64:
      case PrimitiveType::U64:
        return reinterpret_cast<int64_t*>(data_)[i];
      default:
        LOG(FATAL) << "Unsupported element type: " << element_type_;
    }
  }

  int64_t operator[](int i) const { return get(i); }

 private:
  PrimitiveType element_type_;
  void* data_;
};

// Loads the offsets and sizes of the input and output ragged tensors from
// device memory.
//
// The parameter `ragged_metadata_allocs` is a vector of pointers to the buffers
// in the host memory allocated by StreamExecutor to copy data from the device
// memory.
absl::StatusOr<std::vector<IntegerOperandData>> LoadRaggedTensorMetadata(
    se::Stream& stream, const std::vector<DeviceBufferPair>& buffers,
    const std::vector<int64_t*>& ragged_metadata_allocs) {
  std::vector<IntegerOperandData> indices;
  for (int i = 0; i < kNumRaggedMetadataOperands; ++i) {
    TF_RETURN_IF_ERROR(stream.Memcpy(ragged_metadata_allocs[i],
                                     buffers[i + 2].source_buffer,
                                     buffers[i + 2].source_buffer.size()));
    indices.push_back(IntegerOperandData(buffers[i + 2].element_type,
                                         ragged_metadata_allocs[i]));
  }

  // Wait for the copies to complete.
  if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to complete all kernels launched on stream %p: %s", &stream,
        blocked.message()));
  }

  return indices;
}

}  // namespace

NcclRaggedAllToAllStartThunk::NcclRaggedAllToAllStartThunk(
    ThunkInfo thunk_info, const HloRaggedAllToAllInstruction* instr,
    std::vector<NcclCollectiveThunk::Buffer> buffers, bool p2p_memcpy_enabled)
    : NcclCollectiveThunk(Thunk::kNcclAllToAllStart, thunk_info,
                          IsSyncCollective(instr)),
      config_(GetNcclRaggedAllToAllConfig(instr)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status NcclRaggedAllToAllStartThunk::CheckImplementable(
    const HloRaggedAllToAllInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&instr]() -> absl::Status {
    for (HloInstruction* operand : instr->operands()) {
      Shape shape = operand->shape();
      TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kNcclRaggedAllToAll));
    }
    return absl::OkStatus();
  };
  return AddOpDescription<NcclRaggedAllToAllStartThunk>(
      status(), instr, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclRaggedAllToAllStartThunk::GetGroupMode(
    const HloRaggedAllToAllInstruction* instr) {
  return GetNcclRaggedAllToAllConfig(instr).config.group_mode;
}

absl::Status NcclRaggedAllToAllStartThunk::Initialize(
    const InitializeParams& params) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::Initialize(params));

  // Allocate temp buffers in the host memory to load the sizes and offsets of
  // ragged tensors from device memory.
  absl::MutexLock lock(&mutex_);
  if (!host_buffer_allocs_.contains(params.executor)) {
    std::vector<std::unique_ptr<se::MemoryAllocation>> allocs;
    for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<se::MemoryAllocation> alloc,
                          params.executor->HostMemoryAllocate(
                              config_.num_ragged_rows * sizeof(int64_t)));
      allocs.push_back(std::move(alloc));
    }
    host_buffer_allocs_.emplace(params.executor, std::move(allocs));
  }

  if (!device_buffer_allocs_.contains(params.executor)) {
    se::DeviceMemoryBase output_offsets_device_buffer =
        params.executor->Allocate(config_.num_ragged_rows * sizeof(int64_t));

    if (output_offsets_device_buffer.is_null()) {
      return absl::InternalError("Failed to allocate output offsets buffer.");
    }

    device_buffer_allocs_.emplace(params.executor,
                                  output_offsets_device_buffer);
  }

  return absl::OkStatus();
}

absl::Status NcclRaggedAllToAllStartThunk::Cleanup(
    const CleanupParams& params) {
  absl::MutexLock lock(&mutex_);

  if (device_buffer_allocs_.contains(params.executor)) {
    se::DeviceMemoryBase alloc =
        device_buffer_allocs_.extract(params.executor).mapped();
    params.executor->Deallocate(&alloc);
  }

  return absl::OkStatus();
}

absl::Status NcclRaggedAllToAllStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));

  // Get buffer allocs to load sizes and offsets of ragged tensors from device
  // memory.
  std::vector<int64_t*> ragged_metadata_allocs(4);
  se::DeviceMemoryBase output_offsets_device_buffer;
  {
    absl::MutexLock lock(&mutex_);
    auto it = host_buffer_allocs_.find(stream.parent());
    CHECK(it != host_buffer_allocs_.end());

    for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
      ragged_metadata_allocs[i] =
          reinterpret_cast<int64_t*>(it->second[i]->opaque());
    }

    auto jt = device_buffer_allocs_.find(stream.parent());
    CHECK(jt != device_buffer_allocs_.end());
    output_offsets_device_buffer = jt->second;
  }

  return xla::gpu::RunRaggedAllToAll(
      collectives, config_.ragged_row_element_size, device_buffers, stream,
      comm_handle.comm, ragged_metadata_allocs, output_offsets_device_buffer);
}

AsyncStreamKind NcclRaggedAllToAllStartThunk::GetAsyncStreamKind() const {
  return AsyncStreamKind::kCollective;
}

// Runs AllToAll on a buffer that contains ragged tensor metadata.
absl::Status RunAllToAllOnIndexBuffer(
    GpuCollectives* collectives, const se::DeviceMemoryBase& source_buffer,
    const se::DeviceMemoryBase& destination_buffer, PrimitiveType element_type,
    se::Stream& stream, Communicator* comm) {
  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  TF_RETURN_IF_ERROR(collectives->GroupStart());
  for (int peer = 0; peer < num_ranks; ++peer) {
    se::DeviceMemoryBase send_slice = collectives->Slice(
        source_buffer, element_type, /*offset=*/peer, /*count=*/1);
    se::DeviceMemoryBase recv_slice = collectives->Slice(
        destination_buffer, element_type, /*offset=*/peer, /*count=*/1);

    TF_RETURN_IF_ERROR(comm->Send(send_slice, element_type, /*count=*/1, peer,
                                  GpuCollectives::On(stream)));

    TF_RETURN_IF_ERROR(comm->Recv(recv_slice, element_type, /*count=*/1, peer,
                                  GpuCollectives::On(stream)));
  }

  TF_RETURN_IF_ERROR(collectives->GroupEnd());
  return stream.BlockHostUntilDone();
}

absl::Status RunRaggedAllToAll(
    GpuCollectives* collectives, int64_t ragged_row_element_size,
    const std::vector<DeviceBufferPair>& original_buffers, se::Stream& stream,
    Communicator* comm, const std::vector<int64_t*>& ragged_metadata_allocs,
    const se::DeviceMemoryBase& output_offsets_device_buffer) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing ragged-all-to-all from device ordinal: "
          << device_ordinal;
  TF_RETURN_IF_ERROR(MaybeRegisterBuffers(collectives, stream.parent(),
                                          original_buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  std::vector<DeviceBufferPair> buffers = original_buffers;

  // `output_offsets` of the RaggedAllToAll instruction are sharded in a way,
  // that `output_offset[i]` is an offset in the i-th peer output buffer. To
  // make it work for NCCL model with send/recv, we need to know offsets in the
  // local output buffer. To get the correct offsets we perform an AllToAll on
  // the output_offsets buffer.
  DeviceBufferPair& output_offsets_buffer_pair = buffers[4];
  TF_RETURN_IF_ERROR(RunAllToAllOnIndexBuffer(
      collectives, output_offsets_buffer_pair.source_buffer,
      output_offsets_device_buffer, output_offsets_buffer_pair.element_type,
      stream, comm));
  output_offsets_buffer_pair.source_buffer = output_offsets_device_buffer;

  TF_ASSIGN_OR_RETURN(
      std::vector<IntegerOperandData> ragged_metadata,
      LoadRaggedTensorMetadata(stream, buffers, ragged_metadata_allocs));

  const IntegerOperandData& input_offsets = ragged_metadata[0];
  const IntegerOperandData& send_sizes = ragged_metadata[1];
  const IntegerOperandData& output_offsets = ragged_metadata[2];
  const IntegerOperandData& recv_sizes = ragged_metadata[3];

  TF_RETURN_IF_ERROR(collectives->GroupStart());

  const DeviceBufferPair& data_buffer = buffers[0];
  for (int peer = 0; peer < num_ranks; ++peer) {
    se::DeviceMemoryBase send_slice =
        collectives->Slice(data_buffer.source_buffer, data_buffer.element_type,
                           input_offsets[peer] * ragged_row_element_size,
                           send_sizes[peer] * ragged_row_element_size);

    se::DeviceMemoryBase recv_slice = collectives->Slice(
        data_buffer.destination_buffer, data_buffer.element_type,
        output_offsets[peer] * ragged_row_element_size,
        recv_sizes[peer] * ragged_row_element_size);

    TF_RETURN_IF_ERROR(comm->Send(send_slice, data_buffer.element_type,
                                  send_sizes[peer] * ragged_row_element_size,
                                  peer, GpuCollectives::On(stream)));

    TF_RETURN_IF_ERROR(comm->Recv(recv_slice, data_buffer.element_type,
                                  recv_sizes[peer] * ragged_row_element_size,
                                  peer, GpuCollectives::On(stream)));
  }

  return collectives->GroupEnd();
}

}  // namespace gpu
}  // namespace xla
