/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

// Appends command(s) converted from `thunk` to `cmd_sequence`.
static absl::Status AppendCommands(CommandBufferCmdSequence& cmd_sequence,
                                   const Thunk& thunk,
                                   const ConvertToCommandsOptions& options);

// Appends command(s) converted from `sequence` to `cmd_sequence`.
static absl::Status AppendCommands(CommandBufferCmdSequence& cmd_sequence,
                                   const ThunkSequence& sequence,
                                   const ConvertToCommandsOptions& options);

//===----------------------------------------------------------------------===//
// Conversions from Thunk to Command
//===----------------------------------------------------------------------===//

using Command = std::unique_ptr<CommandBufferCmd>;

static auto ArgsAccess(const std::vector<bool>& written) {
  absl::InlinedVector<BufferUse::MemoryAccess, 4> args_access;
  args_access.reserve(written.size());
  for (bool w : written) {
    args_access.push_back(w ? BufferUse::MemoryAccess::kWrite
                            : BufferUse::MemoryAccess::kRead);
  }
  return args_access;
}

static absl::StatusOr<Command> Convert(const KernelThunk& thunk) {
  return std::make_unique<LaunchCmd>(
      thunk.execution_stream_id(), thunk.kernel_name(), thunk.arguments(),
      ArgsAccess(thunk.written()), thunk.launch_dimensions(),
      thunk.shmem_bytes());
}

static absl::StatusOr<Command> Convert(const CustomKernelThunk& thunk) {
  return std::make_unique<CustomKernelLaunchCmd>(
      thunk.execution_stream_id(), thunk.arguments(),
      ArgsAccess(thunk.written()), thunk.custom_kernel());
}

static absl::StatusOr<Command> Convert(const DeviceToDeviceCopyThunk& thunk) {
  return std::make_unique<MemcpyDeviceToDeviceCmd>(
      thunk.execution_stream_id(), thunk.destination(), thunk.source(),
      thunk.size_bytes());
}

static absl::StatusOr<Command> Convert(const MemzeroThunk& thunk) {
  return std::make_unique<MemzeroCmd>(thunk.execution_stream_id(),
                                      thunk.destination());
}

static absl::StatusOr<Command> Convert(const Memset32BitValueThunk& thunk) {
  return std::make_unique<Memset32Cmd>(thunk.execution_stream_id(),
                                       thunk.destination(), thunk.value());
}

static absl::StatusOr<Command> Convert(
    const WhileThunk& thunk, const ConvertToCommandsOptions& options) {
  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdExecutor cond_cmds,
      ConvertToCommands(thunk.condition_thunk_sequence()->thunks(), options));
  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdExecutor body_cmds,
      ConvertToCommands(thunk.body_thunk_sequence()->thunks(), options));

  return std::make_unique<WhileCmd>(thunk.execution_stream_id(),
                                    thunk.condition_result_buffer(),
                                    std::move(cond_cmds), std::move(body_cmds));
}

static absl::StatusOr<Command> Convert(const GemmThunk& thunk) {
  if (!thunk.workspace().has_value()) {
    return absl::InternalError(
        "Gemm thunk does not contain a workspace buffer");
  }
  return std::make_unique<GemmCmd>(
      thunk.execution_stream_id(), thunk.config(), thunk.lhs_buffer(),
      thunk.rhs_buffer(), thunk.output_buffer(), thunk.workspace().value(),
      thunk.deterministic());
}

static absl::StatusOr<Command> Convert(const CublasLtMatmulThunk& thunk) {
  if (!thunk.workspace().has_value()) {
    return absl::InternalError(
        "Gemm thunk does not contain a workspace buffer");
  }
  return std::make_unique<CublasLtCmd>(
      thunk.execution_stream_id(), thunk.config(), thunk.epilogue(),
      thunk.algorithm_idx(), thunk.a_buffer(), thunk.b_buffer(),
      thunk.c_buffer(), thunk.d_buffer(), thunk.bias_buffer(),
      thunk.aux_buffer(), thunk.a_scale_buffer(), thunk.b_scale_buffer(),
      thunk.c_scale_buffer(), thunk.d_scale_buffer(), thunk.d_amax_buffer(),
      thunk.workspace().value());
}

static absl::StatusOr<Command> Convert(
    const ConditionalThunk& thunk, const ConvertToCommandsOptions& options) {
  std::vector<CommandBufferCmdExecutor> branch_cmds;
  branch_cmds.reserve(thunk.branch_thunks().size());
  if (thunk.branch_index_is_bool()) {
    // For boolean predicates, we need to convert the branches in reverse order
    // because the first branch is the "false" branch and the second is "true"
    CHECK_EQ(thunk.branch_thunks().size(), 2);
    TF_ASSIGN_OR_RETURN(
        branch_cmds.emplace_back(),
        ConvertToCommands(thunk.branch_thunks()[1]->thunks(), options));
    TF_ASSIGN_OR_RETURN(
        branch_cmds.emplace_back(),
        ConvertToCommands(thunk.branch_thunks()[0]->thunks(), options));
  } else {
    for (auto& branch_thunk : thunk.branch_thunks()) {
      TF_ASSIGN_OR_RETURN(CommandBufferCmdExecutor cmds,
                          ConvertToCommands(branch_thunk->thunks(), options));
      branch_cmds.emplace_back(std::move(cmds));
    }
  }
  return std::make_unique<CaseCmd>(
      thunk.execution_stream_id(), thunk.branch_index_buffer(),
      thunk.branch_index_is_bool(), std::move(branch_cmds));
}

static absl::StatusOr<Command> Convert(const AllReduceStartThunk& thunk) {
  return std::make_unique<AllReduceCmd>(
      thunk.nccl_execution_stream_id(), thunk.execution_stream_id(),
      thunk.config(), thunk.reduction_kind(), thunk.buffers());
}

static absl::StatusOr<Command> Convert(const ReduceScatterStartThunk& thunk) {
  return std::make_unique<ReduceScatterCmd>(
      thunk.nccl_execution_stream_id(), thunk.execution_stream_id(),
      thunk.config(), thunk.reduction_kind(), thunk.buffers());
}

static absl::StatusOr<Command> Convert(const AllToAllStartThunk& thunk) {
  return std::make_unique<AllToAllCmd>(
      thunk.nccl_execution_stream_id(), thunk.execution_stream_id(),
      thunk.config(), thunk.has_split_dimension(), thunk.buffers());
}

static absl::StatusOr<Command> Convert(const AllGatherStartThunk& thunk) {
  return std::make_unique<AllGatherCmd>(thunk.nccl_execution_stream_id(),
                                        thunk.execution_stream_id(),
                                        thunk.config(), thunk.buffers());
}

static absl::StatusOr<Command> Convert(
    const DynamicSliceThunk& thunk, const ConvertToCommandsOptions& options) {
  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdExecutor embedded_cmds,
      ConvertToCommands(thunk.get_embedded_thunk()->thunks(), options));

  auto& thunk_fake_allocations = thunk.get_fake_allocations();
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations;
  for (auto it = thunk_fake_allocations.begin();
       it != thunk_fake_allocations.end(); ++it) {
    fake_allocations.push_back(std::make_unique<BufferAllocation>(**it));
  }
  return std::make_unique<DynamicSliceFusionCmd>(
      thunk.execution_stream_id(), std::move(embedded_cmds),
      thunk.get_arguments(), std::move(fake_allocations), thunk.get_offsets(),
      thunk.get_orig_shapes(), thunk.get_sliced_shapes(),
      thunk.get_offset_byte_sizes());
}

static absl::StatusOr<Command> Convert(const PartitionIdThunk& thunk) {
  return std::make_unique<ComputationIdCmd>(thunk.execution_stream_id(),
                                            thunk.dest(),
                                            ComputationIdCmd::Kind::kPartition);
}

static absl::StatusOr<Command> Convert(const ReplicaIdThunk& thunk) {
  return std::make_unique<ComputationIdCmd>(thunk.execution_stream_id(),
                                            thunk.dest(),
                                            ComputationIdCmd::Kind::kReplica);
}

static absl::StatusOr<Command> Convert(const CustomCallThunk& thunk) {
  if (auto bundle = thunk.bundle(); bundle.has_value()) {
    return std::make_unique<CustomCallCmd>(
        thunk.execution_stream_id(), thunk.target_name(), bundle->execute,
        thunk.operands(), thunk.results(), thunk.attributes(),
        /*called_computation=*/nullptr);  // TODO(b/342285364)
  } else {
    return std::make_unique<CustomCallCmd>(
        thunk.execution_stream_id(), thunk.target_name(), thunk.call_target(),
        thunk.operands(), thunk.results(), thunk.opaque());
  }
}

static absl::StatusOr<Command> Convert(const CuDnnThunk& thunk) {
  return std::make_unique<CuDnnCmd>(thunk.execution_stream_id(),
                                    thunk.arguments(), thunk.graph());
}

//===----------------------------------------------------------------------===//
static absl::StatusOr<Command> CopyMetadata(absl::StatusOr<Command> cmd,
                                            const Thunk& thunk) {
  if (cmd.ok()) {
    (*cmd)->set_profile_annotation(thunk.profile_annotation());
    return cmd;
  }
  return cmd;
}

template <typename ThunkType, typename... Args>
static absl::StatusOr<Command> Convert(const Thunk& thunk, Args&&... args) {
  return CopyMetadata(Convert(static_cast<const ThunkType&>(thunk),
                              std::forward<Args>(args)...),
                      thunk);
}

static absl::Status AppendCommands(CommandBufferCmdSequence& cmd_sequence,
                                   const Thunk& thunk,
                                   const ConvertToCommandsOptions& options) {
  auto append = [&](absl::StatusOr<Command> command) -> absl::Status {
    if (command.ok()) {
      cmd_sequence.push_back(std::move(*command));
      return absl::OkStatus();
    }
    return command.status();
  };

  switch (thunk.kind()) {
    case Thunk::Kind::kConditional:
      return append(Convert<ConditionalThunk>(thunk, options));
    case Thunk::Kind::kCopy:
      return append(Convert<DeviceToDeviceCopyThunk>(thunk));
    case Thunk::Kind::kCustomCall:
      return append(Convert<CustomCallThunk>(thunk));
    case Thunk::Kind::kCustomKernel:
      return append(Convert<CustomKernelThunk>(thunk));
    case Thunk::Kind::kKernel:
      return append(Convert<KernelThunk>(thunk));
    case Thunk::Kind::kGemm:
      return append(Convert<GemmThunk>(thunk));
    case Thunk::Kind::kCublasLtMatmul:
      return append(Convert<CublasLtMatmulThunk>(thunk));
    case Thunk::Kind::kMemset32BitValue:
      return append(Convert<Memset32BitValueThunk>(thunk));
    case Thunk::Kind::kMemzero:
      return append(Convert<MemzeroThunk>(thunk));
    case Thunk::Kind::kAllGatherStart:
      return append(Convert<AllGatherStartThunk>(thunk));
    case Thunk::Kind::kAllReduceStart:
      return append(Convert<AllReduceStartThunk>(thunk));
    case Thunk::Kind::kReduceScatterStart:
      return append(Convert<ReduceScatterStartThunk>(thunk));
    case Thunk::Kind::kAllToAllStart:
      return append(Convert<AllToAllStartThunk>(thunk));
    case Thunk::Kind::kPartitionId:
      return append(Convert<PartitionIdThunk>(thunk));
    case Thunk::Kind::kReplicaId:
      return append(Convert<ReplicaIdThunk>(thunk));
    case Thunk::Kind::kWhile:
      return append(Convert<WhileThunk>(thunk, options));
    case Thunk::Kind::kCuDnn:
      return append(Convert<CuDnnThunk>(thunk));
    case Thunk::Kind::kDynamicSlice:
      return append(Convert<DynamicSliceThunk>(thunk, options));

    // Sequential thunk does not have any special semantics and we simply inline
    // all nested thunks into command buffer.
    case Thunk::Kind::kSequential:
      return AppendCommands(cmd_sequence,
                            static_cast<const SequentialThunk&>(thunk).thunks(),
                            options);

    // Thunks that simply wait for stream events are no-op in the command buffer
    // context, as we convert async thunks to command dependency graph.
    case Thunk::Kind::kAllGatherDone:
    case Thunk::Kind::kAllReduceDone:
    case Thunk::Kind::kReduceScatterDone:
    case Thunk::Kind::kAllToAllDone:
    case Thunk::Kind::kWaitForStreams:
      return absl::OkStatus();

    case Thunk::Kind::kCommandBuffer:
      return Internal(
          "Error trying to emit command for a CommandBufferThunk. Input HLO "
          "must already contain command buffers and XLA should not run command "
          "buffer scheduling pass the second time. It it happens in the test, "
          "try explicitly disabling command buffers in tested HLO module.");

    default:
      return Internal("Unsupported thunk kind: %s",
                      Thunk::KindToString(thunk.kind()));
  }
}

static absl::Status AppendCommands(CommandBufferCmdSequence& cmd_sequence,
                                   const ThunkSequence& sequence,
                                   const ConvertToCommandsOptions& options) {
  for (const std::unique_ptr<Thunk>& thunk : sequence)
    TF_RETURN_IF_ERROR(AppendCommands(cmd_sequence, *thunk, options));
  return absl::OkStatus();
}

absl::StatusOr<CommandBufferCmdExecutor> ConvertToCommands(
    const ThunkSequence& sequence, const ConvertToCommandsOptions& options) {
  CommandBufferCmdSequence cmd_sequence;
  TF_RETURN_IF_ERROR(AppendCommands(cmd_sequence, sequence, options));
  return CommandBufferCmdExecutor::Create(std::move(cmd_sequence),
                                          options.synchronization_mode);
}

}  // namespace xla::gpu
