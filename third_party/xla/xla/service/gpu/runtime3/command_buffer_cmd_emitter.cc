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

#include "xla/service/gpu/runtime3/command_buffer_cmd_emitter.h"

#include <memory>
#include <utility>

#include "xla/service/gpu/copy_thunk.h"
#include "xla/service/gpu/kernel_thunk.h"
#include "xla/service/gpu/runtime3/command_buffer_cmd.h"
#include "xla/service/gpu/thunk.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

StatusOr<std::unique_ptr<CommandBufferCmd>> ConvertToCommand(
    const Thunk& thunk) {
  switch (thunk.kind()) {
    // TODO(anlunx): Support other thunk kinds.
    case Thunk::Kind::kKernel: {
      auto& kernel_thunk = static_cast<const KernelThunk&>(thunk);
      auto kernel_cmd = std::make_unique<LaunchCmd>(
          kernel_thunk.kernel_name(), kernel_thunk.arguments(),
          kernel_thunk.launch_dimensions(), kernel_thunk.shmem_bytes());
      return kernel_cmd;
    }
    case Thunk::Kind::kCopy: {
      auto& copy_thunk = static_cast<const DeviceToDeviceCopyThunk&>(thunk);
      auto copy_cmd = std::make_unique<MemcpyDeviceToDeviceCmd>(
          copy_thunk.destination(), copy_thunk.source(),
          copy_thunk.size_bytes());
      return copy_cmd;
    }
    default:
      return InternalError("Unsupported thunk kind");
  }
}

}  // namespace

StatusOr<CommandBufferCmdSequence> ConvertToCommands(
    const ThunkSequence& sequence) {
  CommandBufferCmdSequence cmd_sequence;
  for (const std::unique_ptr<Thunk>& thunk : sequence) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<CommandBufferCmd> cmd,
                        ConvertToCommand(*thunk));
    cmd_sequence.Append(std::move(cmd));
  }
  return cmd_sequence;
}

}  // namespace xla::gpu
