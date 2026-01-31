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

#include "xla/backends/gpu/runtime/command_buffer_cmd.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_memcpy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tensor_map.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

namespace {
// Indvar is a thread-local map that stores the induction variable for each
// dynamic slice thunk. The same thunk object in the memory is shared by
// multiple replicas of the same computation. So, each replica should have its
// own tracking of the induction variable (threadlocal). With threadlocal, we
// cannot embed this inside the dynamic slice thunk object, and so we have a
// static map. There could be multiple dynamic slice thunks in the same module,
// and so we need a map to store the induction variable for each thunk. The
// usage of threadlocal in this context is similar to `LoopCounters` in
// while_thunk.cc (b/343294327).
Literal& Indvar(DynamicSliceFusionCmd* cmd) {
  static thread_local absl::flat_hash_map<DynamicSliceFusionCmd*, Literal>
      indvar_map;
  return indvar_map[cmd];
}
}  // namespace

static absl::string_view ReductionKindString(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::MAX:
      return "max";
    case ReductionKind::MIN:
      return "min";
    case ReductionKind::PRODUCT:
      return "product";
    case ReductionKind::SUM:
      return "sum";
  }
}

// Create a callback to create a command buffer from a command sequence.
static se::CommandBuffer::CreateCommands CreateCommands(
    const CommandBufferCmdExecutor* commands,
    const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer,
             absl::Span<const se::CommandBuffer::Command* const> dependencies) {
    return commands->RecordCreate(*execute_params, *record_params,
                                  command_buffer, dependencies);
  };
}

// Create callbacks to create a command buffer from command sequences.
static std::vector<se::CommandBuffer::CreateCommands> CreateCommands(
    absl::Span<const CommandBufferCmdExecutor> commands,
    const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  std::vector<se::CommandBuffer::CreateCommands> create_commands;
  for (const CommandBufferCmdExecutor& cmd : commands) {
    create_commands.push_back(
        CreateCommands(&cmd, execute_params, record_params));
  }
  return create_commands;
}

// Create a callback to update a command buffer with command sequence.
static se::CommandBuffer::UpdateCommands UpdateCommands(
    const CommandBufferCmdExecutor* commands,
    const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->RecordUpdate(*execute_params, *record_params,
                                  command_buffer);
  };
}

// Create callbacks to update a command buffer with command sequence.
static std::vector<se::CommandBuffer::UpdateCommands> UpdateCommands(
    absl::Span<const CommandBufferCmdExecutor> commands,
    const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  std::vector<se::CommandBuffer::UpdateCommands> update_commands;
  for (const CommandBufferCmdExecutor& cmd : commands) {
    update_commands.push_back(
        UpdateCommands(&cmd, execute_params, record_params));
  }
  return update_commands;
}

//===----------------------------------------------------------------------===//
// Command::RecordAction helpers.
//===----------------------------------------------------------------------===//

using CreateCommand =
    absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>(
        absl::Span<const se::CommandBuffer::Command* const> dependencies)>;

using UpdateCommand =
    absl::FunctionRef<absl::Status(const se::CommandBuffer::Command* command)>;

// Handles a record action by calling one of the user-provided functions.
static absl::StatusOr<const se::CommandBuffer::Command*> Handle(
    Command::RecordAction action, CreateCommand create_command,
    UpdateCommand update_command) {
  if (auto* create = std::get_if<Command::RecordCreate>(&action)) {
    return create_command(create->dependencies);
  }

  if (auto* update = std::get_if<Command::RecordUpdate>(&action)) {
    TF_RETURN_IF_ERROR(update_command(update->command));
    return update->command;
  }

  return Internal("Invalid record action");
}

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

TracedCommandBuffer::TracedCommandBuffer(const Command* trace_cmd,
                                         Command::BufferUseVector buffers,
                                         int64_t capacity)
    : trace_cmd_(trace_cmd), capacity_(capacity), entries_(capacity) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";  // NOLINT
  // Collect unique buffer allocation indices in a set first and convert to
  // vector as flat hash set iteration has measurable overheads.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices;
  for (auto& buffer : buffers) {
    allocs_indices.insert(buffer.slice().index());
  }
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::GetOrTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
    se::StreamPriority priority) {
  // Collect memory addresses for relevant allocations.
  absl::InlinedVector<se::DeviceAddressBase, 4> allocs;
  allocs.reserve(allocs_indices_.size());
  for (auto& index : allocs_indices_) {
    allocs.emplace_back(buffer_allocation->GetDeviceAddress(index));
  }

  // Moves entry at `i` position to front and moves entries in `[0, i)` range
  // one element to the right. Returns reference to the first entry.
  auto shift_right = [&](size_t i) -> Entry& {
    if (i == 0) {
      return entries_[0];
    }

    Entry entry = std::move(entries_[i]);
    do {
      entries_[i] = std::move(entries_[i - 1]);
    } while (--i > 0);

    return entries_[0] = std::move(entry);
  };

  for (size_t i = 0; i < capacity_; ++i) {
    // Found entry for a given allocations, move it to front and return a
    // pointer to cached command buffer.
    if (ABSL_PREDICT_TRUE(absl::c_equal(entries_[i].recorded_allocs, allocs) &&
                          entries_[i].command_buffer)) {
      VLOG(6) << "Command buffer trace cache hit for command "
              << trace_cmd_->ToString();
      return shift_right(i).command_buffer.get();
    }

    // Create a new entry by calling a user-provided tracing function, move it
    // to front and return a pointer to cached command buffer.
    if (entries_[i].command_buffer == nullptr) {
      TF_ASSIGN_OR_RETURN(
          entries_[i].command_buffer,
          se::TraceCommandBufferFactory::Create(executor, stream, trace));
      entries_[i].recorded_allocs.assign(allocs.begin(), allocs.end());
      if (priority != se::StreamPriority::Default) {
        TF_RETURN_IF_ERROR(entries_[i].command_buffer->SetPriority(priority));
      }
      VLOG(6) << "Command buffer trace cache create new item for command "
              << trace_cmd_->ToString();
      return shift_right(i).command_buffer.get();
    }
  }

  // Create a new entry by calling a user-provided tracing function, replace
  // the last entry with it, move it to front and return a pointer to cached
  // command buffer.
  TF_ASSIGN_OR_RETURN(
      entries_[capacity_ - 1].command_buffer,
      se::TraceCommandBufferFactory::Create(executor, stream, trace));
  entries_[capacity_ - 1].recorded_allocs.assign(allocs.begin(), allocs.end());
  VLOG(6) << "Command buffer trace cache does replacement for command "
          << trace_cmd_->ToString();
  return shift_right(capacity_ - 1).command_buffer.get();
}

//===----------------------------------------------------------------------===//
// TracedCommandBufferCmd
//===----------------------------------------------------------------------===//

TracedCommandBufferCmd::TracedCommandBufferCmd(CommandType cmd_type)
    : Command(cmd_type) {}

absl::StatusOr<const se::CommandBuffer::Command*>
TracedCommandBufferCmd::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  auto traced_cmd = record_params.state.GetOrCreate<TracedCommandBuffer>(
      this, command_buffer, [&] {
        const auto& debug_options = xla::GetDebugOptionsFromFlags();
        return std::make_unique<TracedCommandBuffer>(
            this, buffers(), debug_options.xla_cmd_buffer_trace_cache_size());
      });

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      traced_cmd->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace, priority()));

  VLOG(5) << "Record traced command into command buffer: " << command_buffer;
  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateChildCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateChildCommand(command, *nested_cmd);
      });
}

//===----------------------------------------------------------------------===//
// EmptyCmd
//===----------------------------------------------------------------------===//

EmptyCmd::EmptyCmd() : Command(CommandType::kEmptyCmd) {}

absl::StatusOr<const se::CommandBuffer::Command*> EmptyCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateEmptyCmd(dependencies, priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        // Empty command is not updatable.
        return absl::OkStatus();
      });
}

//===----------------------------------------------------------------------===//
// ComputationId
//===----------------------------------------------------------------------===//

ComputationIdCmd::ComputationIdCmd(BufferAllocation::Slice dest, Kind kind)
    : Command(CommandType::kComputationIdCmd), dest_(dest), kind_(kind) {}

Command::BufferUseVector ComputationIdCmd::buffers() const {
  return {BufferUse::Write(dest_, ShapeUtil::MakeShape(S32, {}))};
}

absl::StatusOr<const se::CommandBuffer::Command*> ComputationIdCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dest_);

  GlobalDeviceId global_device_id =
      execute_params.collective_params->global_device_id;
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID logical_id,
      execute_params.collective_params->device_assn->LogicalIdForDevice(
          global_device_id));

  uint32_t value = static_cast<uint32_t>(kind_ == Kind::kReplica
                                             ? logical_id.replica_id
                                             : logical_id.computation_id);

  VLOG(5) << "ComputationIdCmd"
          << ": kind=" << (kind_ == Kind::kReplica ? "replica" : "partition")
          << "; value=" << value;
  VLOG(5) << "  Id: " << dest_ << " (" << dst.opaque() << ")";

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateMemset(&dst, value, /*num_elements=*/1,
                                            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateMemset(command, &dst, value,
                                            /*num_elements=*/1);
      });
}

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

LaunchCmd::LaunchCmd(
    std::string kernel_name, absl::Span<const BufferAllocation::Slice> args,
    absl::Span<const BufferUse::MemoryAccess> args_access,
    LaunchDimensions dims, int64_t shmem_bytes,
    std::optional<stream_executor::gpu::TmaMetadata> tma_metadata)
    : Command(CommandType::kLaunchCmd),
      kernel_name_(std::move(kernel_name)),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      dims_(dims),
      shmem_bytes_(shmem_bytes),
      tma_metadata_(std::move(tma_metadata)) {}

absl::Status LaunchCmd::Initialize(const Thunk::InitializeParams& params) {
  {
    absl::MutexLock lock(mutex_);
    if (kernels_.contains(params.executor)) {
      return absl::OkStatus();
    }
  }

  std::unique_ptr<se::Kernel> kernel;
  if (!params.src.binary.empty()) {
    TF_ASSIGN_OR_RETURN(
        kernel, CreateKernel(kernel_name_, args_.size(), params.src.binary,
                             params.executor, shmem_bytes_));

  } else {
    TF_ASSIGN_OR_RETURN(
        kernel, CreateKernel(kernel_name_, args_.size(), params.src.text,
                             params.executor, shmem_bytes_));
  }

  absl::MutexLock lock(mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> LaunchCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << "; shmem_bytes=" << shmem_bytes_;

  se::StreamExecutor* executor = execute_params.stream->parent();
  se::Kernel* kernel = [&] {
    absl::MutexLock lock(mutex_);
    return kernels_[executor].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Kernel not loaded on a command buffer executor: ", kernel_name_));
  }

  absl::InlinedVector<se::KernelArg, 4> kernel_args_variant;
  stream_executor::gpu::TmaMetadata tma_metadata =
      tma_metadata_.value_or(se::gpu::TmaMetadata{});
  for (int idx = 0; idx < args_.size(); ++idx) {
    const BufferAllocation::Slice& arg = args_[idx];
    se::DeviceAddressBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();

    if (auto it = tma_metadata.arg_index_to_tma_info.find(idx);
        it != tma_metadata.arg_index_to_tma_info.end()) {
      // TMA descriptor argument.
      stream_executor::gpu::TmaDescriptor tma_desc = it->second;
      TF_ASSIGN_OR_RETURN(se::TensorMap tensor_map,
                          executor->CreateTensorMap(tma_desc, buf.opaque()));
      VLOG(5) << "  Using TensorMap for arg #" << idx << ": "
              << tma_desc.ToString();
      kernel_args_variant.push_back(std::move(tensor_map));
    } else {
      // Buffer argument.
      kernel_args_variant.push_back(buf);
    }
  }

  auto kernel_args = se::PackKernelArgs(kernel_args_variant, shmem_bytes_);

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateLaunch(
            dims_.thread_counts_per_block(), dims_.block_counts(), *kernel,
            *kernel_args, dependencies, priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateLaunch(
            command, dims_.thread_counts_per_block(), dims_.block_counts(),
            *kernel, *kernel_args);
      });
}

Command::BufferUseVector LaunchCmd::buffers() const {
  BufferUseVector buffers;
  for (int32_t i = 0; i < args_.size(); ++i) {
    buffers.emplace_back(args_[i], args_access_[i]);
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// CustomKernelLaunchCmd
//===----------------------------------------------------------------------===//

CustomKernelLaunchCmd::CustomKernelLaunchCmd(
    absl::Span<const BufferAllocation::Slice> args,
    absl::Span<const BufferUse::MemoryAccess> args_access,
    CustomKernel custom_kernel)
    : Command(CommandType::kCustomKernelLaunchCmd),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      custom_kernel_(std::move(custom_kernel)) {}

absl::Status CustomKernelLaunchCmd::Initialize(
    const Thunk::InitializeParams& params) {
  {
    absl::MutexLock lock(mutex_);
    if (kernels_.contains(params.executor)) {
      return absl::OkStatus();
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::Kernel> kernel,
      params.executor->LoadKernel(custom_kernel_.kernel_spec()));

  absl::MutexLock lock(mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> CustomKernelLaunchCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  VLOG(5) << "CustomKernelLaunchCmd: custom_kernel=" << custom_kernel_.name();

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(mutex_);
    return kernels_[execute_params.stream->parent()].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(
        absl::StrCat("Custom kernel not loaded on a command buffer executor: ",
                     custom_kernel_.name()));
  }

  absl::InlinedVector<se::DeviceAddressBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceAddressBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    buffers.push_back(buf);
  }

  se::KernelArgsDeviceAddressArray kernel_args(
      buffers, custom_kernel_.shared_memory_bytes());

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateLaunch(
            custom_kernel_.thread_dims(), custom_kernel_.block_dims(), *kernel,
            kernel_args, dependencies, priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateLaunch(
            command, custom_kernel_.thread_dims(), custom_kernel_.block_dims(),
            *kernel, kernel_args);
      });
}

Command::BufferUseVector CustomKernelLaunchCmd::buffers() const {
  BufferUseVector buffers;
  for (int32_t i = 0; i < args_.size(); ++i) {
    buffers.emplace_back(args_[i], args_access_[i]);
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

MemcpyDeviceToDeviceCmd::MemcpyDeviceToDeviceCmd(ShapedSlice dst,
                                                 ShapedSlice src,
                                                 int64_t num_bytes)
    : Command(CommandType::kMemcpyDeviceToDeviceCmd),
      dst_(dst),
      src_(src),
      num_bytes_(num_bytes) {
  CHECK_EQ(ShapeUtil::ByteSizeOfElements(src_.shape),
           ShapeUtil::ByteSizeOfElements(dst_.shape));
  CHECK_LE(num_bytes, dst_.slice.size());
  CHECK_LE(num_bytes, src_.slice.size());
  CHECK_GE(src_.slice.size(), ShapeUtil::ByteSizeOf(src_.shape));
}

absl::StatusOr<const se::CommandBuffer::Command*>
MemcpyDeviceToDeviceCmd::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                RecordAction record_action,
                                se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_.slice);
  se::DeviceAddressBase src =
      execute_params.buffer_allocations->GetDeviceAddress(src_.slice);

  VLOG(5) << "MemcpyDeviceToDeviceCmd: num_bytes = " << num_bytes_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << src_ << " (" << src.opaque() << ")";

  if (num_bytes_ == 0) {
    VLOG(5) << "Skip recording MemcpyDeviceToDeviceCmd command of 0 bytes";
    return nullptr;
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateMemcpyD2D(&dst, src, num_bytes_,
                                               dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateMemcpyD2D(command, &dst, src, num_bytes_);
      });
}

Command::BufferUseVector MemcpyDeviceToDeviceCmd::buffers() const {
  return {BufferUse::Write(dst_.slice, dst_.shape),
          BufferUse::Read(src_.slice, src_.shape)};
}

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

MemzeroCmd::MemzeroCmd(ShapedSlice dst)
    : Command(CommandType::kMemzeroCmd), dst_(dst) {}

absl::StatusOr<const se::CommandBuffer::Command*> MemzeroCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_.slice);

  VLOG(5) << "MemzeroCmd:";
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.slice.size() == 0) {
    VLOG(5) << "Skip recording MemzeroCmd command of 0 bytes";
    return nullptr;
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateMemset(&dst, uint8_t{0},
                                            /*num_elements=*/dst_.slice.size(),
                                            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateMemset(command, &dst, uint8_t{0},
                                            /*num_elements=*/dst_.slice.size());
      });
}

Command::BufferUseVector MemzeroCmd::buffers() const {
  return {BufferUse::Write(dst_.slice, dst_.shape)};
}

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

Memset32Cmd::Memset32Cmd(BufferAllocation::Slice dst, uint32_t bit_pattern)
    : Command(CommandType::kMemset32Cmd),
      dst_(dst),
      bit_pattern_(bit_pattern) {}

absl::StatusOr<const se::CommandBuffer::Command*> Memset32Cmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "Memset32Cmd: bit_pattern=" << bit_pattern_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording Memset32Cmd command of 0 bytes";
    return nullptr;
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateMemset(
            &dst, bit_pattern_,
            /*num_elements=*/dst_.size() / sizeof(uint32_t), dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateMemset(
            command, &dst, bit_pattern_,
            /*num_elements=*/dst_.size() / sizeof(uint32_t));
      });
}

Command::BufferUseVector Memset32Cmd::buffers() const {
  return {BufferUse::Write(dst_, ShapeUtil::MakeShape(U32, {}))};
}

//===----------------------------------------------------------------------===//
// ChildCmd
//===----------------------------------------------------------------------===//

ChildCmd::ChildCmd(CommandBufferCmdExecutor child_commands)
    : Command(CommandType::kChildCmd),
      child_commands_(std::move(child_commands)) {}

bool ChildCmd::requires_initialization() {
  return child_commands_.requires_initialization();
}

bool ChildCmd::force_update() { return child_commands_.force_update(); }

Command::BufferUseVector ChildCmd::buffers() const {
  return {child_commands_.buffers().begin(), child_commands_.buffers().end()};
}

absl::Status ChildCmd::Initialize(const Thunk::InitializeParams& params) {
  TF_RETURN_IF_ERROR(child_commands_.Initialize(params));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> ChildCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  VLOG(5) << "Record ChildCmd " << child_commands_.size() << " commands";

  auto record_fn = [&](se::CommandBuffer* command_buffer) {
    return child_commands_
        .RecordCreate(execute_params, record_params, command_buffer,
                      /*dependencies=*/{})
        .status();
  };

  auto update_fn = [&](se::CommandBuffer* command_buffer) {
    return child_commands_.RecordUpdate(execute_params, record_params,
                                        command_buffer);
  };

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateChildCommand(record_fn, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateChildCommand(command, update_fn);
      });
}

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

CaseCmd::CaseCmd(ShapedSlice index,
                 std::vector<CommandBufferCmdExecutor> branches)
    : Command(CommandType::kCaseCmd),
      index_(index),
      index_is_bool_(index.shape.element_type() == PRED),
      branches_(std::move(branches)) {}

absl::Status CaseCmd::Initialize(const Thunk::InitializeParams& params) {
  for (auto& branch : branches_) {
    TF_RETURN_IF_ERROR(branch.Initialize(params));
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> CaseCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase index =
      execute_params.buffer_allocations->GetDeviceAddress(index_.slice);

  VLOG(5) << "CaseCmd:";
  VLOG(5) << "  index: " << index_ << " (" << index.opaque() << ")";

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        if (index_is_bool_) {
          return command_buffer->CreateCase(
              se::DeviceAddress<bool>(index),
              CreateCommands(branches_, &execute_params, &record_params),
              dependencies);
        }
        return command_buffer->CreateCase(
            se::DeviceAddress<int32_t>(index),
            CreateCommands(branches_, &execute_params, &record_params),
            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        if (index_is_bool_) {
          return command_buffer->UpdateCase(
              command, se::DeviceAddress<bool>(index),
              UpdateCommands(branches_, &execute_params, &record_params));
        }
        return command_buffer->UpdateCase(
            command, se::DeviceAddress<int32_t>(index),
            UpdateCommands(branches_, &execute_params, &record_params));
      });
}

bool CaseCmd::requires_initialization() {
  return absl::c_any_of(
      branches_, [](const auto& seq) { return seq.requires_initialization(); });
}

bool CaseCmd::force_update() {
  return absl::c_any_of(branches_,
                        [](const auto& seq) { return seq.force_update(); });
}

Command::BufferUseVector CaseCmd::buffers() const {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(BufferUse::Read(index_.slice, index_.shape));
  for (auto& branch : branches_) {
    buffers.insert(branch.buffers().begin(), branch.buffers().end());
  }
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

WhileCmd::WhileCmd(BufferAllocation::Slice pred,
                   CommandBufferCmdExecutor cond_commands,
                   CommandBufferCmdExecutor body_commands,
                   std::optional<int64_t> trip_count, bool enable_loop_unroll)
    : Command(CommandType::kWhileCmd),
      pred_(pred),
      cond_commands_(std::move(cond_commands)),
      body_commands_(std::move(body_commands)),
      trip_count_(trip_count),
      enable_loop_unroll_(enable_loop_unroll) {}

absl::Status WhileCmd::Initialize(const Thunk::InitializeParams& params) {
  TF_RETURN_IF_ERROR(cond_commands_.Initialize(params));
  TF_RETURN_IF_ERROR(body_commands_.Initialize(params));
  if (enable_loop_unroll_ && body_commands_.support_loop_unroll() &&
      cond_commands_.support_loop_unroll() && trip_count_.has_value()) {
    is_unrolled_loop_ = true;
  }
  VLOG(3) << "WhileCmd::Initialize: enable_loop_unroll_=" << enable_loop_unroll_
          << ", body_support=" << body_commands_.support_loop_unroll()
          << ", cond_support=" << cond_commands_.support_loop_unroll()
          << ", trip_count=" << trip_count_.value_or(-1)
          << ", is_unrolled_loop_=" << is_unrolled_loop_;
  return absl::OkStatus();
}

absl::Status WhileCmd::Prepare(const Thunk::PrepareParams& params) {
  TF_RETURN_IF_ERROR(cond_commands_.Prepare(params));
  TF_RETURN_IF_ERROR(body_commands_.Prepare(params));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> WhileCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  VLOG(5) << "WhileCmd: cond_commands=" << cond_commands_.size()
          << " body_commands=" << body_commands_.size();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";
  VLOG(5) << "  trip_count: " << trip_count_.value_or(-1)
          << " (unroll: " << is_unrolled_loop_ << ")";
  if (is_unrolled_loop_) {
    // When the loop is unrolled, we need to record the body commands for
    // `trip_count` times into child_command_buffer, and implement the While
    // command as a child command.
    //
    // Unroll the while loop body for `trip_count` times.
    // Unrolled execution sequence: cond -> body -> cond -> body -> ...
    // In the unrolled pattern, we still need to run the cond commands because
    // body commands might depends on the value of index variable that is
    // updated by condition commands.

    auto record_fn =
        [&](se::CommandBuffer* child_command_buffer) -> absl::Status {
      VLOG(3) << "Recording unrolled loop with trip_count: "
              << trip_count_.value();

      Command::RecordParams new_record_params = record_params;
      std::vector<const se::CommandBuffer::Command*> dependencies;

      for (int64_t i = 0; i < trip_count_.value(); ++i) {
        CommandExecutor::RecordId record_id(i);
        new_record_params.unroll_iteration = i;
        TF_ASSIGN_OR_RETURN(dependencies,
                            cond_commands_.RecordCreate(
                                execute_params, new_record_params,
                                child_command_buffer, dependencies, record_id));
        TF_ASSIGN_OR_RETURN(dependencies,
                            body_commands_.RecordCreate(
                                execute_params, new_record_params,
                                child_command_buffer, dependencies, record_id));
      }

      return absl::OkStatus();
    };

    auto update_fn =
        [&](se::CommandBuffer* child_command_buffer) -> absl::Status {
      VLOG(3) << "Updating unrolled loop with trip_count: "
              << trip_count_.value();

      Command::RecordParams new_record_params = record_params;

      for (int64_t i = 0; i < trip_count_.value(); ++i) {
        CommandExecutor::RecordId record_id(i);
        new_record_params.unroll_iteration = i;
        TF_RETURN_IF_ERROR(
            cond_commands_.RecordUpdate(execute_params, new_record_params,
                                        child_command_buffer, record_id));
        TF_RETURN_IF_ERROR(
            body_commands_.RecordUpdate(execute_params, new_record_params,
                                        child_command_buffer, record_id));
      }

      return absl::OkStatus();
    };

    return Handle(
        std::move(record_action),
        [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
          return command_buffer->CreateChildCommand(record_fn, dependencies);
        },
        [&](const se::CommandBuffer::Command* command) {
          return command_buffer->UpdateChildCommand(command, update_fn);
        });
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateWhile(
            se::DeviceAddress<bool>(pred),
            CreateCommands(&cond_commands_, &execute_params, &record_params),
            CreateCommands(&body_commands_, &execute_params, &record_params),
            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateWhile(
            command, se::DeviceAddress<bool>(pred),
            UpdateCommands(&cond_commands_, &execute_params, &record_params),
            UpdateCommands(&body_commands_, &execute_params, &record_params));
      });
}

bool WhileCmd::requires_initialization() {
  return (cond_commands_.requires_initialization() ||
          body_commands_.requires_initialization());
}

bool WhileCmd::force_update() {
  return cond_commands_.force_update() || body_commands_.force_update();
}

Command::BufferUseVector WhileCmd::buffers() const {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(BufferUse::Read(pred_, ShapeUtil::MakeShape(PRED, {})));
  buffers.insert(cond_commands_.buffers().begin(),
                 cond_commands_.buffers().end());
  buffers.insert(body_commands_.buffers().begin(),
                 body_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

GemmCmd::GemmCmd(GemmConfig config, const BufferAllocation::Slice& lhs_buffer,
                 const BufferAllocation::Slice& rhs_buffer,
                 const BufferAllocation::Slice& output_buffer,
                 std::optional<BufferAllocation::Slice> workspace,
                 bool deterministic)
    : TracedCommandBufferCmd(CommandType::kGemmCmd),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      workspace_(workspace),
      deterministic_(deterministic) {}

absl::Status GemmCmd::Initialize(const Thunk::InitializeParams& params) {
  if (!params.stream->parent()->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support for GemmCmd");
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> GemmCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase lhs =
      execute_params.buffer_allocations->GetDeviceAddress(lhs_buffer_);
  se::DeviceAddressBase rhs =
      execute_params.buffer_allocations->GetDeviceAddress(rhs_buffer_);
  se::DeviceAddressBase out =
      execute_params.buffer_allocations->GetDeviceAddress(output_buffer_);

  se::DeviceAddressBase workspace(/*opaque=*/nullptr, /*size=*/0);
  if (workspace_.has_value()) {
    workspace =
        execute_params.buffer_allocations->GetDeviceAddress(workspace_.value());
  }

  VLOG(5) << "GemmCmd: deterministic=" << deterministic_;
  VLOG(5) << "  Lhs: " << lhs_buffer_ << " (" << lhs.opaque() << ")";
  VLOG(5) << "  Lhs: " << rhs_buffer_ << " (" << rhs.opaque() << ")";
  VLOG(5) << "  Out: " << output_buffer_ << " (" << out.opaque() << ")";
  VLOG(5) << "  Workspace: " << workspace.opaque();

  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer,
                             [&](se::Stream* stream) {
                               return RunGemm(config_, lhs, rhs, out, workspace,
                                              deterministic_, stream);
                             });
}

Command::BufferUseVector GemmCmd::buffers() const {
  Command::BufferUseVector res{
      BufferUse::Read(lhs_buffer_, config_.lhs_layout.ToShape()),
      BufferUse::Read(rhs_buffer_, config_.rhs_layout.ToShape()),
      BufferUse::Write(output_buffer_, config_.output_layout.ToShape()),
  };
  if (workspace_.has_value()) {
    res.push_back(BufferUse::Write(
        *workspace_, ShapeUtil::MakeShape(S8, {workspace_->size()})));
  }
  return res;
}

//===----------------------------------------------------------------------===//
// CublasLtCmd
//===----------------------------------------------------------------------===//

CublasLtCmd::CublasLtCmd(const CublasLtMatmulThunk& matmul_thunk)
    : TracedCommandBufferCmd(CommandType::kCublasLtCmd),
      CublasLtMatmulThunk(matmul_thunk) {}

absl::StatusOr<const se::CommandBuffer::Command*> CublasLtCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  // This call is required to make sure matmul plan is already created and
  // cached before recording the command buffer.
  TF_RETURN_IF_ERROR(GetCachedMatmulPlan(execute_params).status());

  VLOG(5) << "CublasLtCmd:";
  VLOG(5) << "  a_buffer: " << a_.ToString();
  VLOG(5) << "  b_buffer: " << b_.ToString();
  VLOG(5) << "  c_buffer: " << c_.ToString();
  VLOG(5) << "  d_buffer: " << d_.ToString();
  VLOG(5) << "  bias_buffer: " << bias_.ToString();
  VLOG(5) << "  aux_buffer: " << aux_.ToString();
  VLOG(5) << "  a_scale_buffer: " << a_scale_.ToString();
  VLOG(5) << "  b_scale_buffer: " << b_scale_.ToString();
  VLOG(5) << "  c_scale_buffer: " << c_scale_.ToString();
  VLOG(5) << "  d_scale_buffer: " << d_scale_.ToString();
  VLOG(5) << "  d_amax_buffer: " << d_amax_.ToString();
  // workspace buffer is guaranteed to be non-null here.
  VLOG(5) << "  workspace_buffer: " << workspace_->ToString();

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return ExecuteOnStreamInternal(stream, execute_params);
      });
}

Command::BufferUseVector CublasLtCmd::buffers() const {
  BufferUseVector buffer_usage;
  buffer_usage.reserve(13);
  buffer_usage.push_back(BufferUse::Read(a_));
  buffer_usage.push_back(BufferUse::Read(b_));
  buffer_usage.push_back(BufferUse::Read(c_));
  buffer_usage.push_back(BufferUse::Write(d_));
  buffer_usage.push_back(BufferUse::Write(*workspace_));

  if (bias_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(bias_));
  }
  if (a_scale_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(a_scale_));
  }
  if (b_scale_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(b_scale_));
  }
  if (c_scale_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(c_scale_));
  }
  if (d_scale_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(d_scale_));
  }
  if (aux_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Write(aux_));
  }
  if (d_amax_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(d_amax_));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CuDnnCmd
//===----------------------------------------------------------------------===//

CuDnnCmd::CuDnnCmd(absl::Span<const ShapedSlice> args,
                   const std::shared_ptr<se::dnn::LazyDnnGraph> graph)
    : TracedCommandBufferCmd(CommandType::kCuDnnCmd),
      args_(args.cbegin(), args.cend()),
      graph_(graph) {}

absl::Status CuDnnCmd::Initialize(const Thunk::InitializeParams& params) {
  if (!params.stream->parent()->AsDnn()) {
    return absl::InternalError("Failed to initialize DNN support for CuDnnCmd");
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> CuDnnCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  CHECK(graph_ != nullptr);
  std::vector<se::DeviceAddressBase> operands;
  operands.reserve(args_.size());
  for (const ShapedSlice& arg : args_) {
    se::DeviceAddressBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg.slice);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    operands.push_back(buf);
  }
  TF_ASSIGN_OR_RETURN(
      const bool supports_explicit,
      graph_->get()->SupportsExplicitCommandBufferConstruction());
  if (supports_explicit) {
    return Handle(
        std::move(record_action),
        [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
          return command_buffer->CreateDnnGraphCommand(
              *graph_->get(), *execute_params.stream,
              absl::Span<se::DeviceAddressBase>(operands), dependencies);
        },
        [&](const se::CommandBuffer::Command* command) {
          return command_buffer->UpdateDnnGraphCommand(
              command, *graph_->get(), *execute_params.stream,
              absl::Span<se::DeviceAddressBase>(operands));
        });
  }
  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return graph_->get()->Execute(
            *stream, absl::Span<se::DeviceAddressBase>(operands),
            execute_params.collective_params->local_device_id.value());
      });
}

Command::BufferUseVector CuDnnCmd::buffers() const {
  Command::BufferUseVector buffer_usage;
  buffer_usage.reserve(args_.size());
  for (int i = 0; i < args_.size() - 1; ++i) {
    buffer_usage.push_back(BufferUse::Read(args_[i].slice, args_[i].shape));
  }
  buffer_usage.push_back(
      BufferUse::Write(args_.back().slice, args_.back().shape));
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CustomCallCmd
//===----------------------------------------------------------------------===//

absl::StatusOr<const se::CommandBuffer::Command*> CustomCallCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  if (handler_ == nullptr) {
    return RecordLegacyCustomCall(execute_params, record_params,
                                  std::move(record_action), command_buffer);
  }
  return RecordXlaFfiCall(execute_params, record_params,
                          std::move(record_action), command_buffer);
}

namespace {
// Records each buffer associated with each slice into the provided vector.
// Returns an error if any of the slices is missing a buffer allocation.
absl::Status GetBuffers(const Thunk::ExecuteParams& execute_params,
                        absl::Span<const NullableShapedSlice> slices,
                        std::vector<void*>& buffers, absl::string_view label) {
  for (int i = 0; i < slices.size(); ++i) {
    if (!slices[i].has_value()) {
      buffers.push_back(nullptr);
      VLOG(5) << label << i << ": null";
      continue;
    }

    if (!slices[i]->slice.allocation()) {
      return absl::InternalError("custom call input missing buffer allocation");
    }

    auto buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slices[i]->slice)
            .opaque();
    VLOG(5) << label << i << ": " << slices[i]->slice << " (" << buffer << ")";
    buffers.push_back(buffer);
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<const se::CommandBuffer::Command*>
CustomCallCmd::RecordLegacyCustomCall(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());

  VLOG(5) << "CustomCallCmd: target_name=" << target_name_;
  TF_RETURN_IF_ERROR(
      GetBuffers(execute_params, operands_, buffers, "  Operand "));
  TF_RETURN_IF_ERROR(
      GetBuffers(execute_params, results_, buffers, "  Result "));

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            XlaCustomCallStatus custom_call_status;
            call_target_(stream, buffers.data(), opaque_.data(), opaque_.size(),
                         &custom_call_status);
            auto message = CustomCallStatusGetMessage(&custom_call_status);
            if (message) {
              return absl::InternalError(
                  absl::StrCat("CustomCall failed: ", *message));
            }
            return absl::OkStatus();
          }));

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateChildCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateChildCommand(command, *nested_cmd);
      });
}

absl::StatusOr<const se::CommandBuffer::Command*>
CustomCallCmd::RecordXlaFfiCall(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                RecordAction record_action,
                                se::CommandBuffer* command_buffer) {
  // TODO(ezhulenev): This is not the most optimal approach, as we'll be doing
  // a lot of extra allocation on every call. We have to keep attributes
  // separate from arguments, as they do not change after thunk is
  // constructed.
  ffi::CallFrameBuilder builder(operands_.size(), results_.size());

  VLOG(5) << "CustomCallCmd: target_name=" << target_name_;

  absl::InlinedVector<se::DeviceAddressBase, 4> arguments;
  arguments.reserve(operands_.size());

  for (int i = 0; i < operands_.size(); ++i) {
    const NullableShapedSlice& slice = operands_[i];
    if (!slice.has_value()) {
      arguments.push_back(se::DeviceAddressBase{});
      continue;
    }

    se::DeviceAddressBase buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slice->slice);
    VLOG(5) << "  Operand " << i << ": " << slice->slice << " ("
            << buffer.opaque() << ")";
    arguments.push_back(buffer);
  }

  absl::InlinedVector<se::DeviceAddressBase, 4> results;
  results.reserve(results_.size());

  for (int i = 0; i < results_.size(); ++i) {
    const NullableShapedSlice& slice = results_[i];
    if (!slice.has_value()) {
      results.push_back(se::DeviceAddressBase{});
      continue;
    }

    se::DeviceAddressBase buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slice->slice);
    VLOG(5) << "  Result " << i << ": " << slice->slice << " ("
            << buffer.opaque() << ")";
    results.push_back(buffer);
  }

  // Borrow the FFI call frame from the object pool and update with the actual
  // device memory addresses.
  TF_ASSIGN_OR_RETURN(auto call_frame, call_frames_->GetOrCreate());
  TF_RETURN_IF_ERROR(call_frame->UpdateWithBuffers(arguments, results));

  RunId run_id = execute_params.collective_params->run_id;

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            ffi::CallOptions options = {
                run_id,
                execute_params.buffer_allocations->device_ordinal(),
                ffi::CallOptions::GpuOptions{
                    stream,
                    execute_params.buffer_allocations->memory_allocator()},
                /*called_computation=*/nullptr,  // TODO(b/342285364)
                execute_params.ffi_execution_context,
                execution_state_.get()};
            return ffi::Call(handler_, *call_frame, options);
          }));

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateChildCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateChildCommand(command, *nested_cmd);
      });
}

Command::BufferUseVector CustomCallCmd::buffers() const {
  Command::BufferUseVector buffer_usage;
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<ShapedSlice>& slice : slices) {
      if (slice.has_value()) {
        buffer_usage.push_back(BufferUse::Write(slice->slice));
      }
    }
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

CollectiveCmd::CollectiveCmd(
    CommandType cmd_type, CollectiveConfig config,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : AsyncStartCommand(cmd_type, se::StreamPriority::Highest),
      config_(std::move(config)),
      async_events_(std::move(async_events)) {}

absl::Status CollectiveCmd::Prepare(const Thunk::PrepareParams& params) {
  TF_RET_CHECK(params.collective_params != nullptr);
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));
  return params.collective_clique_requests->RequestClique(clique_key);
}

absl::StatusOr<const se::CommandBuffer::Command*>
CollectiveCmd::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::CommandBuffer> nested_cmd,
                      se::TraceCommandBufferFactory::Create(
                          execute_params.stream->parent(),
                          execute_params.command_buffer_trace_stream, trace));

  if (priority() != se::StreamPriority::Default) {
    TF_RETURN_IF_ERROR(nested_cmd->SetPriority(priority()));
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateChildCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateChildCommand(command, *nested_cmd);
      });
}

//===----------------------------------------------------------------------===//
// CollectiveDoneCmd
//===----------------------------------------------------------------------===//

CollectiveDoneCmd::CollectiveDoneCmd(
    const AsyncStartCommand* async_start,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : AsyncDoneCommand(async_start), async_events_(std::move(async_events)) {}

absl::StatusOr<const se::CommandBuffer::Command*> CollectiveDoneCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateEmptyCmd(dependencies, priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        return absl::OkStatus();
      });
}

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

AllReduceCmd::AllReduceCmd(
    CollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(CommandType::kAllReduceCmd, std::move(config),
                    std::move(async_events)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> AllReduceCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "AllReduceCmd: reduction=" << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllReduceCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunAllReduce(reduction_kind_, device_buffers, *stream, *comm,
                            config().use_symmetric_buffer);
      });
}

Command::BufferUseVector AllReduceCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

ReduceScatterCmd::ReduceScatterCmd(
    CollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(CommandType::kReduceScatterCmd, std::move(config),
                    std::move(async_events)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> ReduceScatterCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "ReduceScatterCmd: reduction=" << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "ReduceScatterCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(execute_params, record_params, record_action,
                             command_buffer, [&](se::Stream* stream) {
                               return RunReduceScatter(
                                   reduction_kind_, device_buffers, *stream,
                                   *comm, config().use_symmetric_buffer);
                             });
}

Command::BufferUseVector ReduceScatterCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllToAllCmd
//===----------------------------------------------------------------------===//

AllToAllCmd::AllToAllCmd(
    CollectiveConfig config, bool has_split_dimension,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(CommandType::kAllToAllCmd, std::move(config),
                    std::move(async_events)),
      has_split_dimension_(has_split_dimension),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> AllToAllCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "AllToAllCmd, has_split_dimension=" << has_split_dimension_;

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllToAllCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  // MemCpy case is not currently supported in CommandBuffer.
  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunAllToAll(has_split_dimension_, device_buffers, *stream, *comm,
                           config().use_symmetric_buffer);
      });
}

Command::BufferUseVector AllToAllCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

AllGatherCmd::AllGatherCmd(
    CollectiveConfig config, absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(CommandType::kAllGatherCmd, std::move(config),
                    std::move(async_events)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> AllGatherCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "AllGatherCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllGatherCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunAllGather(device_buffers, *stream, *comm,
                            config().use_symmetric_buffer);
      });
}

Command::BufferUseVector AllGatherCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

CollectiveBroadcastCmd::CollectiveBroadcastCmd(
    CollectiveConfig config, absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(CommandType::kCollectiveBroadcastCmd, std::move(config),
                    std::move(async_events)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*>
CollectiveBroadcastCmd::Record(const Thunk::ExecuteParams& execute_params,
                               const RecordParams& record_params,
                               RecordAction record_action,
                               se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "CollectiveBroadcastCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "CollectiveBroadcastCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunCollectiveBroadcast(device_buffers, *stream, *comm);
      });
}

Command::BufferUseVector CollectiveBroadcastCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// RecvCmd
//===----------------------------------------------------------------------===//

RecvCmd::RecvCmd(CollectiveConfig config, P2PConfig p2p_config,
                 const CollectiveThunk::Buffer& buffer,
                 std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(CommandType::kRecvCmd, std::move(config),
                    std::move(async_events)),
      p2p_config_(std::move(p2p_config)),
      buffer_(buffer) {}

absl::StatusOr<const se::CommandBuffer::Command*> RecvCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  DeviceBufferPair device_buffer_pair{
      config().operand_element_type[0],
      buffer_.element_count,
      execute_params.buffer_allocations->GetDeviceAddress(
          buffer_.source_buffer.slice),
      execute_params.buffer_allocations->GetDeviceAddress(
          buffer_.destination_buffer.slice),
      buffer_.source_memory_space,
      buffer_.destination_memory_space};

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "RecvCmd:";

  XLA_VLOG_DEVICE(5, device_ordinal)
      << "  Src: " << buffer_.source_buffer << " ("
      << device_buffer_pair.source_buffer.opaque() << ")";
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "  Dst: " << buffer_.destination_buffer << " ("
      << device_buffer_pair.destination_buffer.opaque() << ")";

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "RecvCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  GlobalDeviceId global_device_id =
      execute_params.collective_params->global_device_id;

  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID current_logical_id,
      execute_params.collective_params->device_assn->LogicalIdForDevice(
          global_device_id));

  const int64_t current_id =
      config().group_mode ==
              CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  std::string device_string =
      CollectiveThunk::GetDeviceString(*execute_params.collective_params);

  P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(p2p_config_.id_to_source_target, current_id);

  bool should_run = false;
  switch (p2p_config_.validation_kind) {
    case P2PConfig::ValidationKind::kValid:
      should_run = true;
      break;
    case P2PConfig::ValidationKind::kInvalid:
      should_run = false;
      break;
    case P2PConfig::ValidationKind::kConditional:
      return absl::UnimplementedError(
          "Conditional validation is not supported in RecvCmd CommandBuffer");
  }

  if (!should_run) {
    VLOG(3) << "[" << device_ordinal << "] Skipping Recv";
    return nullptr;
  }

  const std::optional<int64_t> source_id = source_target.source;
  std::function<absl::Status(se::Stream*)> trace = [&](se::Stream* stream) {
    return RunRecv(device_buffer_pair, *stream, *comm, current_id, source_id,
                   device_string);
  };

  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer, trace);
}

Command::BufferUseVector RecvCmd::buffers() const {
  BufferUseVector buffer_usage;
  buffer_usage.emplace_back(BufferUse::Read(buffer_.source_buffer.slice,
                                            buffer_.source_buffer.shape));
  buffer_usage.emplace_back(BufferUse::Write(buffer_.destination_buffer.slice,
                                             buffer_.destination_buffer.shape));
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// SendCmd
//===----------------------------------------------------------------------===//

SendCmd::SendCmd(CollectiveConfig config, P2PConfig p2p_config,
                 const CollectiveThunk::Buffer& buffer,
                 std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(CommandType::kSendCmd, std::move(config),
                    std::move(async_events)),
      p2p_config_(std::move(p2p_config)),
      buffer_(buffer) {}

absl::StatusOr<const se::CommandBuffer::Command*> SendCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  DeviceBufferPair device_buffer_pair{
      config().operand_element_type[0],
      buffer_.element_count,
      execute_params.buffer_allocations->GetDeviceAddress(
          buffer_.source_buffer.slice),
      execute_params.buffer_allocations->GetDeviceAddress(
          buffer_.destination_buffer.slice),
      buffer_.source_memory_space,
      buffer_.destination_memory_space};

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "SendCmd:";

  XLA_VLOG_DEVICE(5, device_ordinal)
      << "  Src: " << buffer_.source_buffer << " ("
      << device_buffer_pair.source_buffer.opaque() << ")";
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "  Dst: " << buffer_.destination_buffer << " ("
      << device_buffer_pair.destination_buffer.opaque() << ")";

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "SendCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  GlobalDeviceId global_device_id =
      execute_params.collective_params->global_device_id;

  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID current_logical_id,
      execute_params.collective_params->device_assn->LogicalIdForDevice(
          global_device_id));

  const int64_t current_id =
      config().group_mode ==
              CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  std::string device_string =
      CollectiveThunk::GetDeviceString(*execute_params.collective_params);

  P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(p2p_config_.id_to_source_target, current_id);

  bool should_run = false;
  switch (p2p_config_.validation_kind) {
    case P2PConfig::ValidationKind::kValid:
      should_run = true;
      break;
    case P2PConfig::ValidationKind::kInvalid:
      should_run = false;
      break;
    case P2PConfig::ValidationKind::kConditional:
      return absl::UnimplementedError(
          "Conditional validation is not supported in SendCmd CommandBuffer");
  }

  std::optional<int64_t> target_id = source_target.target;
  if (!target_id || !should_run) {
    VLOG(3) << "[" << device_ordinal << "] Skipping Send";
    return nullptr;
  }

  std::function<absl::Status(se::Stream*)> trace = [&](se::Stream* stream) {
    return RunSend(device_buffer_pair, *stream, *comm, current_id, *target_id,
                   device_string);
  };

  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer, trace);
}

Command::BufferUseVector SendCmd::buffers() const {
  BufferUseVector buffer_usage;
  buffer_usage.emplace_back(BufferUse::Read(buffer_.source_buffer.slice,
                                            buffer_.source_buffer.shape));
  buffer_usage.emplace_back(BufferUse::Write(buffer_.destination_buffer.slice,
                                             buffer_.destination_buffer.shape));
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectivePermuteCmd
//===----------------------------------------------------------------------===//

CollectivePermuteCmd::CollectivePermuteCmd(
    CollectiveConfig config, P2PConfig p2p_config,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(CommandType::kCollectivePermuteCmd, std::move(config),
                    std::move(async_events)),
      p2p_config_(std::move(p2p_config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> CollectivePermuteCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "CollectivePermuteCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "CollectivePermuteCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  std::string device_string =
      CollectiveThunk::GetDeviceString(*execute_params.collective_params);
  bool use_symmetric_buffer = config().use_symmetric_buffer;

  TF_ASSIGN_OR_RETURN(
      const int64_t current_id,
      GetCollectiveCurrentId(execute_params.collective_params, p2p_config_));

  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(p2p_config_.id_to_source_target, current_id);

  // MemCpy case is not currently supported in CommandBuffer.
  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunCollectivePermute(source_target, device_buffers, *stream,
                                    *comm, device_string, current_id,
                                    /*use_memcpy=*/false,
                                    /*recv_ptr_map=*/nullptr,
                                    use_symmetric_buffer);
      });
}

Command::BufferUseVector CollectivePermuteCmd::buffers() const {
  BufferUseVector buffer_usage;
  for (const CollectiveThunk::Buffer& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

DynamicSliceFusionCmd::DynamicSliceFusionCmd(
    CommandBufferCmdExecutor embedded_commands,
    std::vector<std::optional<BufferAllocation::Slice>> arguments,
    std::vector<BufferAllocation> fake_allocations,
    std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets,
    std::vector<std::optional<Shape>> orig_shapes,
    std::vector<std::optional<Shape>> sliced_shapes,
    std::vector<std::optional<PrimitiveType>> offset_primitive_types,
    std::optional<
        const DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata*>
        offset_as_function_of_indvar_metadata)
    : Command(CommandType::kDynamicSliceFusionCmd),
      embedded_commands_(std::move(embedded_commands)),
      fake_allocations_(std::move(fake_allocations)),
      offset_as_function_of_indvar_metadata_(
          std::move(offset_as_function_of_indvar_metadata)) {
  // Zip all arguments together to create a list of SliceDef.
  for (auto [arg, offset, orig_shape, sliced_shape, offset_primitive_type] :
       llvm::zip_equal(arguments, offsets, orig_shapes, sliced_shapes,
                       offset_primitive_types)) {
    slices_.push_back(DynamicSliceThunk::SliceDef{
        std::move(arg),
        std::move(offset),
        std::move(orig_shape),
        std::move(sliced_shape),
        std::move(offset_primitive_type),
    });
  }

  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    embeded_to_origin_slice_map_[argument_idx] = slice.embedded_thunk_argument;
  }

  // Find how many offsets we might have to transfer from device to host and
  // pre-compute host allocation requirements.
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    offsets_allocs_base_.push_back(offsets_allocs_size_);
    if (slice.sliced_shape.has_value()) {
      offsets_allocs_size_ +=
          slice.sliced_shape->dimensions().size() * sizeof(int64_t);
    }
  }
}

// Force update the command when there is any non-constant value slice offset,
// because the memory address might changed if the offset is loop
// iterator or operator outputs even if the parent command's memory pointers
// do not change.
bool DynamicSliceFusionCmd::requires_initialization() {
  return !absl::c_all_of(slices_, [](const DynamicSliceThunk::SliceDef& slice) {
    if (!slice.offsets.has_value()) {
      return true;
    }
    return absl::c_all_of(slice.offsets.value(),
                          [](DynamicSliceThunk::Offset offset) {
                            return std::holds_alternative<int64_t>(offset);
                          });
  });
}

absl::Status DynamicSliceFusionCmd::Initialize(
    const Thunk::InitializeParams& params) {
  TF_RETURN_IF_ERROR(embedded_commands_.Initialize(params));
  absl::MutexLock lock(mutex_);
  if (offsets_allocs_.contains(params.executor)) {
    return absl::OkStatus();
  }

  XLA_VLOG_DEVICE(2, params.executor->device_ordinal())
      << "Allocate " << offsets_allocs_size_
      << " bytes for transferring offsets on executor: " << params.executor;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::MemoryAllocation> allocation,
      params.executor->HostMemoryAllocate(offsets_allocs_size_));
  offsets_allocs_.emplace(params.executor, std::move(allocation));
  return absl::OkStatus();
}

absl::Status DynamicSliceFusionCmd::Prepare(
    const Thunk::PrepareParams& params) {
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    VLOG(3) << "DynamicSliceFusionCmd: slice: " << slice.ToString();
    if (slice.offsets.has_value()) {
      TF_RET_CHECK(slice.embedded_thunk_argument.has_value());
      TF_RET_CHECK(slice.orig_shape.has_value());
      TF_RET_CHECK(slice.sliced_shape.has_value());
      TF_RET_CHECK(slice.offset_primitive_type.has_value());
      TF_RET_CHECK(slice.orig_shape->IsArray());
      TF_RET_CHECK(slice.sliced_shape->IsArray());
      TF_RET_CHECK(slice.offsets->size() ==
                   slice.orig_shape->dimensions().size());
      TF_RET_CHECK(slice.sliced_shape->dimensions().size() ==
                   slice.orig_shape->dimensions().size());
    }
  }
  TF_RETURN_IF_ERROR(embedded_commands_.Prepare(params));
  if (offset_as_function_of_indvar_metadata_.has_value()) {
    Indvar(this) =
        HloEvaluator()
            .Evaluate(
                *offset_as_function_of_indvar_metadata_.value()->indvar_init,
                {})
            .value();
    VLOG(3) << "Indvar init module: "
            << offset_as_function_of_indvar_metadata_.value()
                   ->indvar_init->ToString();
    VLOG(3) << "Indvar update module: "
            << offset_as_function_of_indvar_metadata_.value()
                   ->indvar_update->ToString();
    VLOG(3) << "Indvar value initialized to :" << Indvar(this).ToString();
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> DynamicSliceFusionCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::Stream& stream = *execute_params.stream;

  const BufferAllocations& orig_allocations =
      *execute_params.buffer_allocations;
  absl::InlinedVector<se::DeviceAddressBase, 8> slice_buffers(
      slices_.size(), se::DeviceAddressBase());

  // Get memory allocation for copying offsets from device.
  int64_t* offsets_alloc = [&] {
    absl::MutexLock lock(mutex_);
    return reinterpret_cast<int64_t*>(
        offsets_allocs_.at(stream.parent())->opaque());
  }();

  auto offset_value = [&](int64_t arg_idx, int64_t offset_idx) -> int64_t& {
    return offsets_alloc[offsets_allocs_base_.at(arg_idx) + offset_idx];
  };

  VLOG(2) << "Execute dynamic slice thunk: slices=" << slices_.size();
  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    // Skip arguments that do not have buffer slices (tokens).
    if (!slice.embedded_thunk_argument.has_value()) {
      continue;
    }

    // `argument_buffer` will contain the original offset for slice
    // `argument_slice` within `orig_allocations`
    se::DeviceAddressBase argument_buffer =
        orig_allocations.GetDeviceAddress(*slice.embedded_thunk_argument);

    // If argument is not sliced, just use the original buffer.
    if (!slice.offsets.has_value()) {
      slice_buffers[argument_idx] = argument_buffer;
      continue;
    }

    const Shape& src_shape = *slice.orig_shape;
    const Shape& dst_shape = *slice.sliced_shape;

    absl::InlinedVector<int64_t, 4> slice_starts;
    slice_starts.reserve(dst_shape.dimensions().size());

    // Number of issues d2h transfers to copy offset values from device to
    // host.
    int64_t num_transfers = 0;

    // Get offset for `argument_idx`-th argument, which has
    // `dst_shape.dimensions_size()` components.
    for (auto [offset_idx, values] : llvm::enumerate(llvm::zip(
             *slice.offsets, src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [offset, src_dim, dst_dim] = values;
      if (int64_t* const_offset = std::get_if<int64_t>(&offset)) {
        // Forward slice offsets that are known constant values
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: constant offset = " << *const_offset;
        offset_value(argument_idx, offset_idx) = *const_offset;

      } else if (HloModule** offset_module = std::get_if<HloModule*>(&offset)) {
        TF_ASSIGN_OR_RETURN(
            Literal offset,
            HloEvaluator().Evaluate(**offset_module, {&Indvar(this)}));
        auto offset_int = LiteralUtil::LiteralAsScalarInt64(offset);
        if (offset_int.has_value()) {
          offset_value(argument_idx, offset_idx) = *offset_int;
        } else {
          return absl::InternalError(
              absl::StrFormat("Unhandled type returned from offset module: %s",
                              offset.shape().ToString()));
        }
        VLOG(2) << "Offset value = " << offset_value(argument_idx, offset_idx);
      } else {
        // Transfer slice offset value from device to host.
        auto alloc_slice = std::get<BufferAllocation::Slice>(offset);
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: transfer offset from device " << alloc_slice.ToString();

        se::DeviceAddressBase offset_src =
            orig_allocations.GetDeviceAddress(alloc_slice);
        int64_t* offset_dst = &offset_value(argument_idx, offset_idx);

        // Copy the `offset_idx`-th component of the offset for the
        // `argument_idx`-th argument from device to host.
        TF_RETURN_IF_ERROR(stream.Memcpy(
            offset_dst, offset_src,
            ShapeUtil::ByteSizeOfPrimitiveType(*slice.offset_primitive_type)));
        ++num_transfers;
      }
    }

    // Wait for the completion of all transfers.
    if (num_transfers > 0) {
      VLOG(2) << "Wait for completion of " << num_transfers << " transfer";
      TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
    }

    // Clamp start indices:
    // start_indices[i] = min(max(start_indices[i], 0),
    //                        operand.dimension_size[i] - size_indices[i])
    for (auto [offset_idx, values] : llvm::enumerate(
             llvm::zip(src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [src_dim, dst_dim] = values;
      int64_t start_index =
          std::min(std::max(offset_value(argument_idx, offset_idx), int64_t{0}),
                   src_dim - dst_dim);
      VLOG(2) << "arg idx: " << argument_idx << " offset_idx " << offset_idx
              << " with offset_value " << offset_value(argument_idx, offset_idx)
              << " start_idx: " << start_index << " src_dim: " << src_dim
              << " dst_dim:" << dst_dim;
      slice_starts.push_back(start_index);
    }

    // Compute new slice. No need to copy the content to new buffers as we can
    // reuse the original buffers since slices are contiguous.
    int64_t new_size = ShapeUtil::ByteSizeOf(dst_shape);

    int64_t new_offset = 0;
    for (auto [start, stride] :
         llvm::zip(slice_starts, *ShapeUtil::ByteStrides(src_shape))) {
      new_offset += start * stride;
    }

    VLOG(3) << "Create sliced argument " << argument_idx << " of shape "
            << slice.sliced_shape->ToString()
            << " by slicing argument of shape " << slice.orig_shape->ToString()
            << " at offset " << new_offset << " with " << new_size;
    slice_buffers[argument_idx] =
        argument_buffer.GetByteSlice(new_offset, new_size);
  }

  // Safe to create a local BufferAllocations here since buffers are only
  // slices of bigger ones allocated elsewhere.
  BufferAllocations slice_allocations(slice_buffers,
                                      orig_allocations.device_ordinal(),
                                      orig_allocations.memory_allocator());

  VLOG(3) << "DynamicSliceFusionCmd: new slice_allocations: "
          << slice_allocations.ToString();

  Thunk::ExecuteParams new_params =
      Thunk::ExecuteParams::CloneWithNewAllocations(execute_params,
                                                    slice_allocations);

  // TODO(b/406370928): Instead of creating a nested command buffer on every
  // call we should create it once and update it. CommandBufferThunk state
  // manager relies on command buffer pointer as an identity for command
  // buffers, and it means that command buffer commands sequence should not
  // create ephemeral command buffers at run time.
  TF_ASSIGN_OR_RETURN(auto nested_command_buffer,
                      execute_params.stream->parent()->CreateCommandBuffer(
                          se::CommandBuffer::Mode::kNested));

  CommandStateManager state;
  RecordParams nested_record_params = {state, std::nullopt, false};
  TF_RETURN_IF_ERROR(embedded_commands_.Record(new_params, nested_record_params,
                                               nested_command_buffer.get()));

  // For command buffer instantiation ran by CommandBufferThunk::Initialize, we
  // must not step the Indvar, because it is not a real run.
  if (offset_as_function_of_indvar_metadata_.has_value() &&
      command_buffer->state() == se::CommandBuffer::State::kUpdate) {
    Indvar(this) =
        HloEvaluator()
            .Evaluate(
                *offset_as_function_of_indvar_metadata_.value()->indvar_update,
                {&Indvar(this)})
            .value();
    VLOG(2) << "Update Indvar = " << Indvar(this).ToString();
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateChildCommand(

            *nested_command_buffer, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateChildCommand(command,
                                                  *nested_command_buffer);
      });
}

Command::BufferUseVector DynamicSliceFusionCmd::buffers() const {
  Command::BufferUseVector buffers;
  auto embed_buffers = embedded_commands_.buffers();
  for (const auto& buffer_usage : embed_buffers) {
    buffers.emplace_back(
        *embeded_to_origin_slice_map_.at(buffer_usage.slice().index()),
        buffer_usage.access());
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// DynamicSliceCopyFusionCmd
//===----------------------------------------------------------------------===//

DynamicSliceCopyFusionCmd::DynamicSliceCopyFusionCmd(
    const ShapedSlice& source_buffer, const ShapedSlice& destination_buffer,
    uint64_t mem_size, DynamicMemcpyThunk::Offsets offsets)
    : Command(CommandType::kDynamicSliceCopyFusionCmd),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size),
      offsets_(offsets) {}

absl::StatusOr<const se::CommandBuffer::Command*>
DynamicSliceCopyFusionCmd::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  RecordAction record_action,
                                  se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase src_data =
      execute_params.buffer_allocations->GetDeviceAddress(source_buffer_.slice);
  se::DeviceAddressBase dst_data =
      execute_params.buffer_allocations->GetDeviceAddress(
          destination_buffer_.slice);

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies)
          -> absl::StatusOr<const se::CommandBuffer::Command*> {
        int64_t src_offset = offsets_.src_offsets[0];
        int64_t dst_offset = offsets_.dst_offsets[0];
        auto src_with_offset = src_data.GetByteSlice(src_offset, mem_size_);
        auto dst_with_offset = dst_data.GetByteSlice(dst_offset, mem_size_);
        VLOG(3) << "Create DynamicSliceCopyFusionCmd with Memcpy of size "
                << mem_size_ << " from " << src_with_offset.opaque()
                << " (offset " << src_offset << ") to "
                << dst_with_offset.opaque() << " (offset " << dst_offset
                << "), dependends_on_loop: " << offsets_.depends_on_loop;
        return command_buffer->CreateMemcpyD2D(
            &dst_with_offset, src_with_offset, mem_size_, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        int64_t iteration_index = 0;
        if (offsets_.depends_on_loop) {
          if (WhileThunk::RunningWhileThunkLoop()) {
            TF_ASSIGN_OR_RETURN(iteration_index,
                                WhileThunk::CurrentLoopIteration());
          } else {
            iteration_index = record_params.unroll_iteration;
          }
        }
        int64_t src_offset = offsets_.src_offsets[iteration_index];
        int64_t dst_offset = offsets_.dst_offsets[iteration_index];
        auto src_with_offset = src_data.GetByteSlice(src_offset, mem_size_);
        auto dst_with_offset = dst_data.GetByteSlice(dst_offset, mem_size_);

        VLOG(3) << "Update DynamicSliceCopyFusionCmd with Memcpy of size "
                << mem_size_ << " from " << src_with_offset.opaque()
                << " (offset " << src_offset << ") to "
                << dst_with_offset.opaque() << " (offset " << dst_offset
                << "), iteration_index: " << iteration_index;
        return command_buffer->UpdateMemcpyD2D(command, &dst_with_offset,
                                               src_with_offset, mem_size_);
      });
}

Command::BufferUseVector DynamicSliceCopyFusionCmd::buffers() const {
  Command::BufferUseVector buffers;
  buffers.emplace_back(
      BufferUse::Read(source_buffer_.slice, source_buffer_.shape));
  buffers.emplace_back(
      BufferUse::Write(destination_buffer_.slice, destination_buffer_.shape));
  return buffers;
}

}  // namespace xla::gpu
