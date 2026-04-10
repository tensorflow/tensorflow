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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_memcpy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {
// A context for tracking thunks to commands conversion details.
struct ConversionContext {
  std::vector<Command::ResourceUses> extra_resources;
};
}  // namespace

// Appends command(s) converted from `sequence` to `cmd_sequence`.
static absl::Status AppendCommands(ConversionContext& ctx,
                                   CommandSequence& cmd_sequence,
                                   const ThunkSequence& sequence,
                                   const ConvertToCommandsOptions& options);

//===----------------------------------------------------------------------===//
// Conversions from Thunk to Command
//===----------------------------------------------------------------------===//

static auto ArgsAccess(const std::vector<bool>& written) {
  absl::InlinedVector<BufferUse::MemoryAccess, 4> args_access;
  args_access.reserve(written.size());
  for (bool w : written) {
    args_access.push_back(w ? BufferUse::MemoryAccess::kWrite
                            : BufferUse::MemoryAccess::kRead);
  }
  return args_access;
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const KernelThunk& thunk) {
  return std::make_unique<LaunchCmd>(
      thunk.kernel_name(), thunk.arguments(), ArgsAccess(thunk.written()),
      thunk.launch_dimensions(), thunk.shmem_bytes(), thunk.tma_metadata(),
      thunk.use_pdl());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const CustomKernelThunk& thunk) {
  return std::make_unique<CustomKernelLaunchCmd>(
      thunk.arguments(), ArgsAccess(thunk.written()), thunk.custom_kernel());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const DeviceToDeviceCopyThunk& thunk) {
  return std::make_unique<MemcpyDeviceToDeviceCmd>(
      thunk.destination(), thunk.source(), thunk.size_bytes());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const DynamicMemcpyThunk& thunk) {
  return std::make_unique<DynamicSliceCopyFusionCmd>(
      thunk.source(), thunk.destination(), thunk.mem_size(), thunk.offsets());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const MemzeroThunk& thunk) {
  return std::make_unique<MemzeroCmd>(thunk.destination());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const Memset32BitValueThunk& thunk) {
  return std::make_unique<Memset32Cmd>(thunk.destination(), thunk.value());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const WhileThunk& thunk, const ConvertToCommandsOptions& options) {
  VLOG(1) << "WhileThunk: " << thunk.profile_annotation();
  TF_ASSIGN_OR_RETURN(
      CommandExecutor cond_cmds,
      ConvertToCommands(thunk.condition_executor().thunks(), options));
  TF_ASSIGN_OR_RETURN(
      CommandExecutor body_cmds,
      ConvertToCommands(thunk.body_executor().thunks(), options));

  return std::make_unique<WhileCmd>(
      thunk.condition_result_buffer(), std::move(cond_cmds),
      std::move(body_cmds), thunk.trip_count(), options.enable_loop_unroll);
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const GemmThunk& thunk) {
  return std::make_unique<GemmCmd>(thunk.config(), thunk.lhs_buffer(),
                                   thunk.rhs_buffer(), thunk.output_buffer(),
                                   thunk.workspace(), thunk.deterministic());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const CublasLtMatmulThunk& thunk) {
  if (!thunk.workspace().has_value()) {
    return absl::InternalError(
        "Gemm thunk does not contain a workspace buffer");
  }
  return std::make_unique<CublasLtCmd>(thunk);
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const ConditionalThunk& thunk, const ConvertToCommandsOptions& options) {
  std::vector<CommandExecutor> branch_cmds;
  branch_cmds.reserve(thunk.branch_executors().size());
  if (thunk.branch_index_is_bool()) {
    // For boolean predicates, we need to convert the branches in reverse order
    // because the first branch is the "false" branch and the second is "true"
    CHECK_EQ(thunk.branch_executors().size(), 2);
    TF_ASSIGN_OR_RETURN(
        branch_cmds.emplace_back(),
        ConvertToCommands(thunk.branch_executors()[1].thunks(), options));
    TF_ASSIGN_OR_RETURN(
        branch_cmds.emplace_back(),
        ConvertToCommands(thunk.branch_executors()[0].thunks(), options));
  } else {
    for (auto& branch_thunk : thunk.branch_executors()) {
      TF_ASSIGN_OR_RETURN(CommandExecutor cmds,
                          ConvertToCommands(branch_thunk.thunks(), options));
      branch_cmds.emplace_back(std::move(cmds));
    }
  }
  return std::make_unique<CaseCmd>(thunk.branch_index_buffer(),
                                   std::move(branch_cmds));
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const AllReduceThunk& thunk) {
  return std::make_unique<AllReduceCmd>(thunk.config(), thunk.reduction_kind(),
                                        thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const ReduceScatterThunk& thunk) {
  return std::make_unique<ReduceScatterCmd>(
      thunk.config(), thunk.reduction_kind(), thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const AllToAllThunk& thunk) {
  return std::make_unique<AllToAllCmd>(
      thunk.config(), thunk.has_split_dimension(), thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const AllGatherThunk& thunk) {
  return std::make_unique<AllGatherCmd>(thunk.config(), thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const CollectiveBroadcastThunk& thunk) {
  return std::make_unique<CollectiveBroadcastCmd>(thunk.config(),
                                                  thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const CollectivePermuteThunk& thunk) {
  return std::make_unique<CollectivePermuteCmd>(
      thunk.config(), thunk.p2p_config(), thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const RaggedAllToAllThunk& thunk) {
  return std::make_unique<RaggedAllToAllCmd>(thunk.ragged_all_to_all_config(),
                                             thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const RecvThunk& thunk) {
  return std::make_unique<RecvCmd>(thunk.config(), thunk.p2p_config(),
                                   thunk.buffer());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const SendThunk& thunk) {
  return std::make_unique<SendCmd>(thunk.config(), thunk.p2p_config(),
                                   thunk.buffer());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const DynamicSliceThunk& thunk, const ConvertToCommandsOptions& options) {
  TF_ASSIGN_OR_RETURN(
      CommandExecutor embedded_cmds,
      ConvertToCommands(thunk.get_embedded_executor().thunks(), options));

  auto& thunk_fake_allocations = thunk.get_fake_allocations();
  std::vector<BufferAllocation> fake_allocations;
  for (auto it = thunk_fake_allocations.begin();
       it != thunk_fake_allocations.end(); ++it) {
    fake_allocations.push_back(BufferAllocation(*it));
  }
  return std::make_unique<DynamicSliceFusionCmd>(
      std::move(embedded_cmds), thunk.get_arguments(),
      std::move(fake_allocations), thunk.get_offsets(), thunk.get_orig_shapes(),
      thunk.get_sliced_shapes(), thunk.offset_primitive_types(),
      thunk.get_offset_function());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const CustomCallThunk& thunk) {
  if (auto bundle = thunk.bundle(); bundle.has_value()) {
    return std::make_unique<CustomCallCmd>(
        thunk.target_name(), bundle->execute, thunk.operands(), thunk.results(),
        *thunk.call_frame(), thunk.thunk_info().thunk_id,
        thunk.execution_state(),
        /*called_computation=*/nullptr);  // TODO(b/342285364)
  }
  return std::make_unique<CustomCallCmd>(thunk.target_name(),
                                         thunk.call_target(), thunk.operands(),
                                         thunk.results(), thunk.opaque());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const CuDnnThunk& thunk) {
  return std::make_unique<CuDnnCmd>(thunk.arguments(), thunk.graph());
}

//===----------------------------------------------------------------------===//
static absl::StatusOr<std::unique_ptr<Command>> CopyMetadata(
    absl::StatusOr<std::unique_ptr<Command>> cmd, const Thunk& thunk) {
  if (cmd.ok()) {
    (*cmd)->set_profile_annotation(thunk.profile_annotation());
    return cmd;
  }
  return cmd;
}

// Takes Thunk& (non-const) rather than const Thunk& so that thunks which
// also implement Command can be appended as borrowed Command* without
// const_cast (which is banned). The thunks in ThunkSequence are non-const
// (unique_ptr<Thunk>), so callers always have a non-const reference available.
template <typename ThunkType, typename... Args>
static absl::StatusOr<std::unique_ptr<Command>> Convert(Thunk& thunk,
                                                        Args&&... args) {
  return CopyMetadata(
      Convert(static_cast<ThunkType&>(thunk), std::forward<Args>(args)...),
      thunk);
}

static absl::Status AppendCommands(ConversionContext& ctx,
                                   CommandSequence& cmd_sequence, Thunk& thunk,
                                   const ConvertToCommandsOptions& options) {
  auto append =
      [&](absl::StatusOr<std::unique_ptr<Command>> command) -> absl::Status {
    if (!command.ok()) {
      return command.status();
    }

    cmd_sequence.Append(std::move(*command));
    return absl::OkStatus();
  };

  switch (thunk.kind()) {
    case Thunk::Kind::kConditional:
      return append(Convert<ConditionalThunk>(thunk, options));
    case Thunk::Kind::kCopy:
      if (dynamic_cast<const DynamicMemcpyThunk*>(&thunk)) {
        return append(Convert<DynamicMemcpyThunk>(thunk));
      } else {
        return append(Convert<DeviceToDeviceCopyThunk>(thunk));
      }
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
    case Thunk::Kind::kAllGather:
      return append(Convert<AllGatherThunk>(thunk));
    case Thunk::Kind::kAllReduce:
      return append(Convert<AllReduceThunk>(thunk));
    case Thunk::Kind::kReduceScatter:
      return append(Convert<ReduceScatterThunk>(thunk));
    case Thunk::Kind::kAllToAll:
      return append(Convert<AllToAllThunk>(thunk));
    case Thunk::Kind::kCollectiveBroadcast:
      return append(Convert<CollectiveBroadcastThunk>(thunk));
    case Thunk::Kind::kCollectivePermute:
      return append(Convert<CollectivePermuteThunk>(thunk));
    case Thunk::Kind::kRaggedAllToAll:
      return append(Convert<RaggedAllToAllThunk>(thunk));
    case Thunk::Kind::kRecv:
      return append(Convert<RecvThunk>(thunk));
    case Thunk::Kind::kSend:
      return append(Convert<SendThunk>(thunk));
    // These thunks implement Command directly; append borrowed pointers.
    case Thunk::Kind::kPartitionId:
      cmd_sequence.Append(static_cast<PartitionIdThunk*>(&thunk));
      return absl::OkStatus();
    case Thunk::Kind::kReplicaId:
      cmd_sequence.Append(static_cast<ReplicaIdThunk*>(&thunk));
      return absl::OkStatus();
    case Thunk::Kind::kWhile:
      return append(Convert<WhileThunk>(thunk, options));
    case Thunk::Kind::kCuDnn:
      return append(Convert<CuDnnThunk>(thunk));
    case Thunk::Kind::kDynamicSlice:
      return append(Convert<DynamicSliceThunk>(thunk, options));

    // Sequential thunk does not have any special semantics and we simply inline
    // all nested thunks into command buffer.
    case Thunk::Kind::kSequential:
      return AppendCommands(ctx, cmd_sequence,
                            static_cast<const SequentialThunk&>(thunk).thunks(),
                            options);

    // Async start thunks inline their nested thunk sequence into the command
    // buffer. Command buffers rely on DAG structure for dependencies.
    case Thunk::Kind::kAsyncStart: {
      auto& start = static_cast<const AsyncStartThunk&>(thunk);
      return AppendCommands(ctx, cmd_sequence, start.thunks(), options);
    }

    // Async done thunks are no-ops in command buffers. Create an empty
    // command only if needed as a dependency node.
    case Thunk::Kind::kAsyncDone: {
      if (thunk.control_predecessors().empty()) {
        return absl::OkStatus();
      }
      return append(absl::StatusOr<std::unique_ptr<Command>>(
          std::make_unique<EmptyCmd>()));
    }

    case Thunk::Kind::kCommandBuffer:
      return Internal(
          "Error trying to emit command for a CommandBufferThunk. Input HLO "
          "must already contain command buffers and XLA should not run command "
          "buffer scheduling pass the second time. If it happens in the test, "
          "try explicitly disabling command buffers in tested HLO module.");

    default:
      return Internal("Unsupported thunk kind: %s",
                      Thunk::KindToString(thunk.kind()));
  }
}

static absl::Status AppendCommands(ConversionContext& ctx,
                                   CommandSequence& cmd_sequence,
                                   const ThunkSequence& sequence,
                                   const ConvertToCommandsOptions& options) {
  absl::flat_hash_map<const Thunk*, int64_t> thunk_to_index;
  absl::flat_hash_map<int64_t, std::vector<int64_t>>
      concurrent_region_id_to_thunk_indices;
  std::vector<int64_t> concurrent_region_ids;
  for (const std::unique_ptr<Thunk>& thunk : sequence) {
    TF_RETURN_IF_ERROR(AppendCommands(ctx, cmd_sequence, *thunk, options));
    int64_t index = cmd_sequence.size() - 1;
    thunk_to_index[thunk.get()] = index;
    if (thunk->concurrent_region_id().has_value()) {
      int64_t concurrent_region_id = thunk->concurrent_region_id().value();
      concurrent_region_id_to_thunk_indices[concurrent_region_id].push_back(
          index);
      if (concurrent_region_ids.empty() ||
          concurrent_region_id > concurrent_region_ids.back()) {
        concurrent_region_ids.push_back(concurrent_region_id);
      }
      CHECK_GE(concurrent_region_id, concurrent_region_ids.back())
          << "Concurrent region ids are not monotonic.";
    }
  }

  // Ensure extra_resources is sized to cover all commands added so far
  // (including those added by nested AppendCommands calls).
  ctx.extra_resources.resize(cmd_sequence.size());

  // Add dependencies between concurrent regions to serialize them.
  for (int64_t i = 1; i < concurrent_region_ids.size(); ++i) {
    int64_t concurrent_region_id = concurrent_region_ids[i - 1];
    int64_t next_concurrent_region_id = concurrent_region_ids[i];
    for (int64_t thunk_index :
         concurrent_region_id_to_thunk_indices[concurrent_region_id]) {
      for (int64_t next_thunk_index :
           concurrent_region_id_to_thunk_indices[next_concurrent_region_id]) {
        ctx.extra_resources[thunk_index].push_back(
            ResourceUse::Read(cmd_sequence[next_thunk_index]->token()));
      }
    }
  }

  // Convert thunk control dependencies to token resource dependency, where the
  // predecessor has the token write, and control successor does the token read.
  for (const std::unique_ptr<Thunk>& thunk : sequence) {
    for (const Thunk* control_predecessor : thunk->control_predecessors()) {
      ctx.extra_resources[thunk_to_index[control_predecessor]].push_back(
          ResourceUse::Read(
              cmd_sequence[thunk_to_index[thunk.get()]]->token()));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<CommandExecutor> ConvertToCommands(
    const ThunkSequence& sequence, const ConvertToCommandsOptions& options) {
  VLOG(3) << absl::StreamFormat(
      "Convert thunk sequence to command executor: synchronization_mode=%v",
      options.synchronization_mode);
  ConversionContext ctx;
  CommandSequence cmd_sequence;
  TF_RETURN_IF_ERROR(AppendCommands(ctx, cmd_sequence, sequence, options));
  return CommandExecutor::Create(std::move(cmd_sequence),
                                 options.synchronization_mode,
                                 std::move(ctx.extra_resources));
}

}  // namespace xla::gpu
