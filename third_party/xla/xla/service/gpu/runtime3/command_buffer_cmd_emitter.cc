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
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "xla/service/gpu/gemm_thunk.h"
#include "xla/service/gpu/nccl_all_gather_thunk.h"
#include "xla/service/gpu/nccl_all_reduce_thunk.h"
#include "xla/service/gpu/runtime3/command_buffer_cmd.h"
#include "xla/service/gpu/runtime3/copy_thunk.h"
#include "xla/service/gpu/runtime3/kernel_thunk.h"
#include "xla/service/gpu/runtime3/memset_thunk.h"
#include "xla/service/gpu/runtime3/sequential_thunk.h"
#include "xla/service/gpu/runtime3/while_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

// Appends command(s) converted from `thunk` to `cmd_sequence`.
static Status AppendCommands(CommandBufferCmdSequence& cmd_sequence,
                             const Thunk& thunk, bool force_barriers);

// Appends command(s) converted from `sequence` to `cmd_sequence`.
static Status AppendCommands(CommandBufferCmdSequence& cmd_sequence,
                             const ThunkSequence& sequence,
                             bool force_barriers);

//===----------------------------------------------------------------------===//
// Conversions from Thunk to Command
//===----------------------------------------------------------------------===//

using Command = std::unique_ptr<CommandBufferCmd>;

static auto ArgsAccess(const std::vector<bool>& written) {
  absl::InlinedVector<CommandBufferCmd::MemoryAccess, 4> args_access;
  args_access.reserve(written.size());
  for (bool w : written) {
    args_access.push_back(w ? CommandBufferCmd::MemoryAccess::kWrite
                            : CommandBufferCmd::MemoryAccess::kRead);
  }
  return args_access;
}

static StatusOr<Command> Convert(const KernelThunk& thunk) {
  return std::make_unique<LaunchCmd>(
      thunk.kernel_name(), thunk.arguments(), ArgsAccess(thunk.written()),
      thunk.launch_dimensions(), thunk.shmem_bytes());
}

static StatusOr<Command> Convert(const CustomKernelThunk& thunk) {
  return std::make_unique<CustomKernelLaunchCmd>(
      thunk.arguments(), ArgsAccess(thunk.written()), thunk.custom_kernel());
}

static StatusOr<Command> Convert(const DeviceToDeviceCopyThunk& thunk) {
  return std::make_unique<MemcpyDeviceToDeviceCmd>(
      thunk.destination(), thunk.source(), thunk.size_bytes());
}

static StatusOr<Command> Convert(const MemzeroThunk& thunk) {
  return std::make_unique<MemzeroCmd>(thunk.destination());
}

static StatusOr<Command> Convert(const Memset32BitValueThunk& thunk) {
  return std::make_unique<Memset32Cmd>(thunk.destination(), thunk.value());
}

static StatusOr<Command> Convert(const WhileThunk& thunk, bool force_barriers) {
  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdSequence cond_cmds,
      ConvertToCommands(thunk.condition_thunk_sequence()->thunks(),
                        force_barriers));
  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdSequence body_cmds,
      ConvertToCommands(thunk.body_thunk_sequence()->thunks(), force_barriers));
  return std::make_unique<WhileCmd>(thunk.condition_result_buffer(),
                                    std::move(cond_cmds), std::move(body_cmds));
}

static StatusOr<Command> Convert(const GemmThunk& thunk) {
  if (!thunk.workspace().has_value()) {
    return absl::InternalError(
        "Gemm thunk does not contain a workspace buffer");
  }
  return std::make_unique<GemmCmd>(
      thunk.config(), thunk.lhs_buffer(), thunk.rhs_buffer(),
      thunk.output_buffer(), thunk.workspace().value(), thunk.deterministic());
}

static StatusOr<Command> Convert(const NcclAllReduceStartThunk& thunk) {
  return std::make_unique<AllReduceCmd>(thunk.config(), thunk.reduction_kind(),
                                        thunk.buffers());
}

static StatusOr<Command> Convert(const NcclReduceScatterStartThunk& thunk) {
  return std::make_unique<ReduceScatterCmd>(
      thunk.config(), thunk.reduction_kind(), thunk.buffers());
}

static StatusOr<Command> Convert(const NcclAllGatherStartThunk& thunk) {
  return std::make_unique<AllGatherCmd>(thunk.config(), thunk.buffers());
}

//===----------------------------------------------------------------------===//

template <typename ThunkType>
static StatusOr<Command> Convert(const Thunk& thunk) {
  return Convert(static_cast<const ThunkType&>(thunk));
}

template <typename ThunkType>
static StatusOr<Command> Convert(const Thunk& thunk, bool force_barriers) {
  return Convert(static_cast<const ThunkType&>(thunk), force_barriers);
}

static Status AppendCommands(CommandBufferCmdSequence& cmd_sequence,
                             const Thunk& thunk, bool force_barriers) {
  auto append = [&](StatusOr<Command> command) -> Status {
    if (command.ok()) {
      cmd_sequence.Append(std::move(*command));
      return OkStatus();
    }
    return command.status();
  };

  switch (thunk.kind()) {
    case Thunk::Kind::kKernel:
      return append(Convert<KernelThunk>(thunk));
    case Thunk::Kind::kCustomKernel:
      return append(Convert<CustomKernelThunk>(thunk));
    case Thunk::Kind::kCopy:
      return append(Convert<DeviceToDeviceCopyThunk>(thunk));
    case Thunk::Kind::kMemzero:
      return append(Convert<MemzeroThunk>(thunk));
    case Thunk::Kind::kMemset32BitValue:
      return append(Convert<Memset32BitValueThunk>(thunk));
    case Thunk::Kind::kWhile:
      return append(Convert<WhileThunk>(thunk, force_barriers));
    case Thunk::Kind::kGemm:
      return append(Convert<GemmThunk>(thunk));
    case Thunk::Kind::kNcclAllReduceStart:
      return append(Convert<NcclAllReduceStartThunk>(thunk));
    case Thunk::Kind::kNcclReduceScatterStart:
      return append(Convert<NcclReduceScatterStartThunk>(thunk));
    case Thunk::Kind::kNcclAllGatherStart:
      return append(Convert<NcclAllGatherStartThunk>(thunk));

    // Sequential thunk does not have any special semantics and we simply inline
    // all nested thunks into command buffer.
    case Thunk::Kind::kSequential:
      return AppendCommands(cmd_sequence,
                            static_cast<const SequentialThunk&>(thunk).thunks(),
                            force_barriers);

    // Currently all collective operations recorded on the tracing stream and do
    // not need to have a separate done command.
    case Thunk::Kind::kNcclAllReduceDone:
    case Thunk::Kind::kNcclReduceScatterDone:
    case Thunk::Kind::kNcclAllGatherDone:
      return OkStatus();

    default:
      return InternalError("Unsupported thunk kind: %s",
                           Thunk::KindToString(thunk.kind()));
  }
}

static Status AppendCommands(CommandBufferCmdSequence& cmd_sequence,
                             const ThunkSequence& sequence,
                             bool force_barriers) {
  for (const std::unique_ptr<Thunk>& thunk : sequence)
    TF_RETURN_IF_ERROR(AppendCommands(cmd_sequence, *thunk, force_barriers));
  return OkStatus();
}

StatusOr<CommandBufferCmdSequence> ConvertToCommands(
    const ThunkSequence& sequence, bool force_barriers) {
  CommandBufferCmdSequence cmd_sequence(force_barriers);
  TF_RETURN_IF_ERROR(AppendCommands(cmd_sequence, sequence, force_barriers));
  return cmd_sequence;
}

}  // namespace xla::gpu
