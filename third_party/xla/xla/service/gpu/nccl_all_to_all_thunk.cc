/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/nccl_all_to_all_thunk.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using mlir::lmhlo_gpu::AllToAllStartOp;

namespace {

NcclAllToAllConfig GetNcclAllToAllConfig(AllToAllStartOp op) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfigForMlir(op, std::nullopt);
  config.has_split_dimension = op.getSplitDimension().has_value();
  return config;
}

NcclAllToAllConfig GetNcclAllToAllConfig(const HloAllToAllInstruction* instr) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfig(instr, std::nullopt);
  config.has_split_dimension = instr->split_dimension().has_value();
  return config;
}

}  // namespace

NcclAllToAllStartThunk::NcclAllToAllStartThunk(
    ThunkInfo thunk_info, AllToAllStartOp op,
    std::vector<NcclCollectiveThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllToAllStart, thunk_info,
                          op.getIsSync()),
      config_(GetNcclAllToAllConfig(op)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

NcclAllToAllStartThunk::NcclAllToAllStartThunk(
    ThunkInfo thunk_info, const HloAllToAllInstruction* instr,
    std::vector<NcclCollectiveThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllToAllStart, thunk_info,
                          IsSyncCollective(instr)),
      config_(GetNcclAllToAllConfig(instr)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status NcclAllToAllStartThunk::CheckImplementable(
    AllToAllStartOp op, int64_t replica_count, int64_t partition_count) {
  auto status = [&]() -> absl::Status {
    TF_RETURN_IF_ERROR(NcclCollectiveThunk::CheckImplementable());
    std::optional<uint64_t> split_dim = op.getSplitDimension();
    for (mlir::Value operand : op.getInputs()) {
      TF_RETURN_IF_ERROR(IsValidOperand(operand, Thunk::kNcclAllToAll));
      Shape shape = GetShape(operand);
      if (split_dim &&
          !ShapeUtil::IsEffectivelyMostMajorDimension(shape, *split_dim)) {
        return absl::UnimplementedError(absl::Substitute(
            "all-to-all split dim $0 is not the most major in input shape $1",
            *split_dim, shape.ToString(/*print_layout=*/true)));
      }
    }
    return absl::OkStatus();
  };
  return AddOpDescription<NcclAllToAllStartThunk>(status(), op, replica_count,
                                                  partition_count);
}

/*static*/ absl::Status NcclAllToAllStartThunk::CheckImplementable(
    const HloAllToAllInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&instr]() -> absl::Status {
    TF_RETURN_IF_ERROR(NcclCollectiveThunk::CheckImplementable());
    std::optional<uint64_t> split_dim = instr->split_dimension();
    for (HloInstruction* operand : instr->operands()) {
      Shape shape = operand->shape();
      TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kNcclAllToAll));
      if (split_dim &&
          !ShapeUtil::IsEffectivelyMostMajorDimension(shape, *split_dim)) {
        return absl::UnimplementedError(absl::Substitute(
            "all-to-all split dim $0 is not the most major in input shape $1",
            *split_dim, shape.ToString(/*print_layout=*/true)));
      }
    }
    return absl::OkStatus();
  };
  return AddOpDescription<NcclAllToAllStartThunk>(
      status(), instr, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclAllToAllStartThunk::GetGroupMode(
    AllToAllStartOp op) {
  return GetNcclAllToAllConfig(op).config.group_mode;
}
/*static*/ CollectiveOpGroupMode NcclAllToAllStartThunk::GetGroupMode(
    const HloAllToAllInstruction* instr) {
  return GetNcclAllToAllConfig(instr).config.group_mode;
}

absl::Status NcclAllToAllStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream, ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return xla::gpu::RunAllToAll(config_.has_split_dimension, device_buffers,
                               stream, comm);
}

absl::Status RunAllToAll(bool has_split_dimension,
                         std::vector<DeviceBufferPair>& buffers,
                         se::Stream& stream, ncclComm_t comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-to-all from device ordinal: " << device_ordinal;

  TF_ASSIGN_OR_RETURN(
      int32_t num_participants,
      NcclApi::CommCount(reinterpret_cast<NcclApi::NcclCommHandle>(comm)));

  TF_RETURN_IF_ERROR(NcclApi::GroupStart());

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    for (DeviceBufferPair& buffer : buffers) {
      TF_RET_CHECK(buffer.element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";

      size_t chunk_elements = buffer.element_count / num_participants;

      for (int peer = 0; peer < num_participants; ++peer) {
        TF_ASSIGN_OR_RETURN(
            se::DeviceMemoryBase send_slice,
            NcclApi::Slice(buffer.source_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements));

        TF_ASSIGN_OR_RETURN(
            se::DeviceMemoryBase recv_slice,
            NcclApi::Slice(buffer.destination_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements));

        TF_RETURN_IF_ERROR(NcclApi::Send(
            send_slice, buffer.element_type, chunk_elements, peer,
            reinterpret_cast<NcclApi::NcclCommHandle>(comm), &stream));

        TF_RETURN_IF_ERROR(NcclApi::Recv(
            recv_slice, buffer.element_type, chunk_elements, peer,
            reinterpret_cast<NcclApi::NcclCommHandle>(comm), &stream));
      }
    }
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];

      TF_RETURN_IF_ERROR(NcclApi::Send(
          buffer.source_buffer, buffer.element_type, buffer.element_count, i,
          reinterpret_cast<NcclApi::NcclCommHandle>(comm), &stream));

      TF_RETURN_IF_ERROR(NcclApi::Recv(
          buffer.destination_buffer, buffer.element_type, buffer.element_count,
          i, reinterpret_cast<NcclApi::NcclCommHandle>(comm), &stream));
    }
  }

  return NcclApi::GroupEnd();
}

}  // namespace gpu
}  // namespace xla
