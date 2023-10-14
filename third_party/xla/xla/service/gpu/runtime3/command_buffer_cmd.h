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
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_pimpl.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CommandBufferCmd
//===----------------------------------------------------------------------===//

// CommandBufferCmd is an abstract command that creates or updates command
// buffer by recording commands into it.
class CommandBufferCmd {
 public:
  using ExecutableSource = Thunk::ExecutableSource;

  // Run time parameters required for recording commands into the command
  // buffer. For example when we emit command buffer cmd sequence from an HLO
  // module, we only know the buffer slices required for HLO operations, but the
  // concrete device pointers become available only at run time.
  struct RecordParams {
    const BufferAllocations* buffer_allocations;
  };

  // Prepares a command for recording on a given executor. We split it into a
  // separate function to allow expensive initialization (e.g. device kernel
  // loading) to happen before a command buffer thunk execution.
  virtual Status Initialize(se::StreamExecutor* executor,
                            ExecutableSource source) {
    return OkStatus();
  }

  // Records command into the command buffer.
  virtual Status Record(const RecordParams& params,
                        se::CommandBuffer* command_buffer) = 0;

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
  CommandBufferCmdSequence() = default;

  void Append(std::unique_ptr<CommandBufferCmd> cmd);

  template <typename T, typename... Args>
  void Emplace(Args... args) {
    Append(std::make_unique<T>(std::forward<Args>(args)...));
  }

  // Initialized all commands added to a sequence.
  Status Initialize(se::StreamExecutor* executor,
                    CommandBufferCmd::ExecutableSource source);

  // Records all commands added to a sequence into the given command buffer.
  Status Record(const CommandBufferCmd::RecordParams& params,
                se::CommandBuffer* command_buffer);

 private:
  std::vector<std::unique_ptr<CommandBufferCmd>> commands_;
};

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

class LaunchCmd : public CommandBufferCmd {
 public:
  LaunchCmd(std::string kernel_name,
            absl::Span<const BufferAllocation::Slice> args,
            LaunchDimensions dims, int64_t shmem_bytes);

  Status Initialize(se::StreamExecutor* executor,
                    ExecutableSource source) override;

  Status Record(const RecordParams& params,
                se::CommandBuffer* command_buffer) override;

 private:
  using OwnedKernel = std::unique_ptr<se::KernelBase>;

  std::string kernel_name_;
  std::vector<BufferAllocation::Slice> args_;
  LaunchDimensions dims_;
  int64_t shmem_bytes_;

  absl::flat_hash_map<se::StreamExecutor*, OwnedKernel> kernels_;
};

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

class MemcpyDeviceToDeviceCmd : public CommandBufferCmd {
 public:
  MemcpyDeviceToDeviceCmd(BufferAllocation::Slice dst,
                          BufferAllocation::Slice src, int64_t num_bytes);

  Status Record(const RecordParams& params,
                se::CommandBuffer* command_buffer) override;

 private:
  BufferAllocation::Slice dst_;
  BufferAllocation::Slice src_;
  int64_t num_bytes_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_CMD_H_
