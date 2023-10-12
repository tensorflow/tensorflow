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

#ifndef XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_CMD_H_
#define XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_CMD_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream_executor_pimpl.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CommandBufferCmd
//===----------------------------------------------------------------------===//

// CommandBufferCmd is an abstract command that creates or updates command
// buffer by recording commands into it.
class CommandBufferCmd {
 public:
  // Run time parameters required for recording commands into the command
  // buffer. For example when we emit command buffer cmd sequence from an HLO
  // module, we only know the buffer slices required for HLO operations, but the
  // concrete device pointers become available only at run time.
  struct RecordParams {
    const BufferAllocations* buffer_allocations;
  };

  // Records command into the command buffer.
  virtual Status Record(const RecordParams& params,
                        se::CommandBuffer& command_buffer) = 0;

  virtual ~CommandBufferCmd() = default;
};

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

// A sequence of command buffer commands that create or update a command buffer.
// You can think of CommandBufferCmdSequence as a mini interpreter whose sole
// purpose is to manipulate command buffers at run time.
class CommandBufferCmdSequence {
 public:
  static StatusOr<CommandBufferCmdSequence> Create(
      se::StreamExecutor* executor);

  void Append(std::unique_ptr<CommandBufferCmd> cmd);

  template <typename T, typename... Args>
  void Emplace(Args... args) {
    Append(std::make_unique<T>(std::forward<Args>(args)...));
  }

  // Records all commands added to a sequence into the command buffer.
  Status Record(const CommandBufferCmd::RecordParams& params);

  se::CommandBuffer& command_buffer() { return command_buffer_; }

 private:
  explicit CommandBufferCmdSequence(se::CommandBuffer command_buffer);

  std::vector<std::unique_ptr<CommandBufferCmd>> commands_;
  se::CommandBuffer command_buffer_;
};

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

class MemcpyDeviceToDeviceCmd : public CommandBufferCmd {
 public:
  MemcpyDeviceToDeviceCmd(BufferAllocation::Slice dst,
                          BufferAllocation::Slice src, int64_t num_bytes);

  Status Record(const RecordParams& params,
                se::CommandBuffer& command_buffer) override;

 private:
  BufferAllocation::Slice dst_;
  BufferAllocation::Slice src_;
  int64_t num_bytes_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_CMD_H_
