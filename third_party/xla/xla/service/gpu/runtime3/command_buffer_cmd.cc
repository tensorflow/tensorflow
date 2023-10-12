/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime3/command_buffer_cmd.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "xla/service/buffer_assignment.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

/*static*/ StatusOr<CommandBufferCmdSequence> CommandBufferCmdSequence::Create(
    se::StreamExecutor* executor) {
  TF_ASSIGN_OR_RETURN(auto command_buffer, se::CommandBuffer::Create(executor));
  return CommandBufferCmdSequence(std::move(command_buffer));
}

CommandBufferCmdSequence::CommandBufferCmdSequence(
    se::CommandBuffer command_buffer)
    : command_buffer_(std::move(command_buffer)) {}

void CommandBufferCmdSequence::Append(std::unique_ptr<CommandBufferCmd> cmd) {
  commands_.push_back(std::move(cmd));
}

Status CommandBufferCmdSequence::Record(
    const CommandBufferCmd::RecordParams& params) {
  for (auto& cmd : commands_) {
    TF_RETURN_IF_ERROR(cmd->Record(params, command_buffer_));
  }
  return command_buffer_.Finalize();
}

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

MemcpyDeviceToDeviceCmd::MemcpyDeviceToDeviceCmd(BufferAllocation::Slice dst,
                                                 BufferAllocation::Slice src,
                                                 int64_t num_bytes)
    : dst_(dst), src_(src), num_bytes_(num_bytes) {}

Status MemcpyDeviceToDeviceCmd::Record(const RecordParams& params,
                                       se::CommandBuffer& command_buffer) {
  VLOG(5) << "MemcpyDeviceToDeviceCmd: dst=" << dst_ << ", src=" << src_
          << ", num_bytes=" << num_bytes_;
  se::DeviceMemoryBase dst = params.buffer_allocations->GetDeviceAddress(dst_);
  se::DeviceMemoryBase src = params.buffer_allocations->GetDeviceAddress(src_);
  return command_buffer.MemcpyDeviceToDevice(&dst, src, num_bytes_);
}

}  // namespace xla::gpu
