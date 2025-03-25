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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

using MemoryAccess = BufferUse::MemoryAccess;

std::string CommandBufferCmdString(CommandBufferCmdType type) {
  switch (type) {
#define CASE_CMD_STRING(enum_name, cmd_name, ...) \
  case CommandBufferCmdType::enum_name:           \
    return cmd_name;
    COMMAND_BUFFER_CMD_LIST(CASE_CMD_STRING)
#undef CASE_CMD_STRING
    default:
      return "UnknownCmd";
  }
}

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

//===----------------------------------------------------------------------===//
// CommandBufferCmd
//===----------------------------------------------------------------------===//

CommandBufferCmd::State* CommandBufferCmd::StateManager::GetOrNull(
    const CommandBufferCmd* cmd) {
  if (auto it = state_.find(cmd); it != state_.end()) {
    return it->second.get();
  }
  return nullptr;
}

CommandBufferCmd::State* CommandBufferCmd::StateManager::GetOrCreate(
    const CommandBufferCmd* cmd,
    absl::FunctionRef<std::unique_ptr<State>()> create) {
  if (auto it = state_.find(cmd); it != state_.end()) {
    return it->second.get();
  }
  return state_.try_emplace(cmd, create()).first->second.get();
}

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

CommandBufferCmdSequence::CommandBufferCmdSequence(
    SynchronizationMode synchronization_mode)
    : synchronization_mode_(synchronization_mode) {}

void CommandBufferCmdSequence::Append(std::unique_ptr<CommandBufferCmd> cmd) {
  if (cmd->IsBarrier() && commands_.back()->IsBarrier()) {
    VLOG(2) << "Skipping barrier command as last command is barrier ";
    return;
  }

  for (const BufferUse& buffer : cmd->buffers()) {
    buffers_.insert(buffer);
    allocs_indices_.insert(buffer.slice().index());
  }

  if (synchronization_mode_ == SynchronizationMode::kSerialize) {
    if (!commands_.empty()) {
      cmd->add_dependency(commands_.back().get());
    }
    commands_.push_back(std::move(cmd));
    VLOG(3) << "Append command in serialize mode: "
            << commands_.back()->ToString();
    return;
  }

  for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
    // Add dependency to the latest barrier command.
    if ((*it)->IsBarrier()) {
      cmd->add_dependency(it->get());
      break;
    }

    // Barrier command depends on all previous commands since last barrier.
    if (cmd->IsBarrier()) {
      cmd->add_dependency(it->get());
      continue;
    }

    // Add depencency that has read/write conflict with commands since last
    // barrier.
    if (absl ::c_any_of((*it)->buffers(), [&](const auto& prev_buffer) {
          return absl::c_any_of(cmd->buffers(), [&](const auto& cur_buffer) {
            return cur_buffer.slice().OverlapsWith(prev_buffer.slice()) &&
                   (prev_buffer.access() == MemoryAccess::kWrite ||
                    (prev_buffer.access() == MemoryAccess::kRead &&
                     cur_buffer.access() == MemoryAccess::kWrite));
          });
        })) {
      cmd->add_dependency(it->get());
    }
  }

  // If current command is a collective command, add a dependency to the
  // last previous collective command if there any. This is to avoid concurrent
  // collective operators which are very easy to get deadlock.
  if (cmd->IsCollective()) {
    for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
      if ((*it)->IsBarrier()) break;
      if ((*it)->IsCollective()) {
        cmd->add_dependency(it->get());
        break;
      }
    }
  }
  commands_.push_back(std::move(cmd));
}

absl::Status CommandBufferCmdSequence::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequestsInterface& resource_requests) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}

absl::Status CommandBufferCmdSequence::Initialize(
    const Thunk::InitializeParams& params,
    CommandBufferCmd::StateManager& state) {
  for (const auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Initialize(params, state));
  }
  return absl::OkStatus();
}

std::unique_ptr<CommandBufferCmdSequence> CommandBufferCmdSequence::Clone()
    const {
  auto cloned_sequence =
      std::make_unique<CommandBufferCmdSequence>(synchronization_mode_);
  for (const auto& command : commands_) {
    cloned_sequence->Append(command->Clone());
  }
  return cloned_sequence;
}

absl::Status CommandBufferCmdSequence::Record(
    const Thunk::ExecuteParams& execute_params,
    const CommandBufferCmd::RecordParams& record_params,
    se::CommandBuffer* command_buffer) {
  VLOG(3) << "Record CommandBufferCmdSequence: \n" << ToString();

  if (!created_) {
    if (commands_.size() == 1 && commands_.front()->IsNestedCommandBuffer()) {
      VLOG(3) << "Append an empty command if CommandBufferCmdSequence contains "
                 "only one nested command buffer";
      commands_.push_back(std::make_unique<EmptyCmd>(
          CommandBufferCmd::DependencyCmdSet{commands_.back().get()}));
    }
  }

  uint64_t start_micros = tsl::Env::Default()->NowMicros();
  for (const std::unique_ptr<CommandBufferCmd>& command : commands_) {
    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command->profile_annotation());
    TF_RETURN_IF_ERROR(command->Record(execute_params, record_params,
                                       command_buffer, !created_));
  }
  created_ = true;
  TF_RETURN_IF_ERROR(command_buffer->Finalize());
  uint64_t end_micros = tsl::Env::Default()->NowMicros();

  VLOG(3) << "Recorded " << commands_.size()
          << " commands into command buffer in " << (end_micros - start_micros)
          << " μs.";

  return absl::OkStatus();
}

const absl::flat_hash_set<BufferUse>& CommandBufferCmdSequence::buffers()
    const {
  return buffers_;
}

const absl::flat_hash_set<BufferAllocation::Index>&
CommandBufferCmdSequence::allocs_indices() const {
  return allocs_indices_;
}

std::string CommandBufferCmdSequence::ToString() const {
  std::string result;
  for (const auto& command : commands_) {
    absl::StrAppend(&result, command->ToString(), "\n");
  }
  return result;
}

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

TracedCommandBuffer::TracedCommandBuffer(
    const CommandBufferCmd* trace_cmd,
    CommandBufferCmd::BufferUseVector buffers, int64_t capacity)
    : trace_cmd_(trace_cmd), capacity_(capacity), entries_(capacity) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";  // NOLINT
  // Collect unique buffer allocation indices in a set first and convert to
  // vector as flat hash set iteration has measurable overheads.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices;
  for (auto& buffer : buffers) allocs_indices.insert(buffer.slice().index());
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::GetOrTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  // Collect memory addresses for relevant allocations.
  absl::InlinedVector<se::DeviceMemoryBase, 4> allocs;
  allocs.reserve(allocs_indices_.size());
  for (auto& index : allocs_indices_) {
    allocs.emplace_back(buffer_allocation->GetDeviceAddress(index));
  }

  // Moves entry at `i` position to front and moves entries in `[0, i)` range
  // one element to the right. Returns reference to the first entry.
  auto shift_right = [&](size_t i) -> Entry& {
    if (i == 0) return entries_[0];

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
      VLOG(6) << "Command buffer trace cache create new item for command "
              << trace_cmd_->ToString();
      return shift_right(i).command_buffer.get();
    }
  }

  // Create a new entry by calling a user-provided tracing function, replace the
  // last entry with it, move it to front and return a pointer to cached command
  // buffer.
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

TracedCommandBufferCmd::TracedCommandBufferCmd(CommandBufferCmdType cmd_type)
    : CommandBufferCmd(cmd_type) {}

absl::Status TracedCommandBufferCmd::RecordTracedCommandBuffer(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create, absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  auto traced_cmd_buffer =
      record_params.state.GetOrCreate<TracedCommandBuffer>(this, [&] {
        const auto& debug_options = xla::GetDebugOptionsFromFlags();
        return std::make_unique<TracedCommandBuffer>(
            this, buffers(), debug_options.xla_cmd_buffer_trace_cache_size());
      });

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd_buffer,
      traced_cmd_buffer->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace));

  if (create) {
    TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateChildNode(
                                   ToDependentNodes(), *nested_cmd_buffer));
  } else {
    TF_RETURN_IF_ERROR(
        command_buffer->UpdateChildNode(node_, *nested_cmd_buffer));
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// ComputationId
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): PTX kernel should be replaced with CUDA C++ kernel but
// today we accidentally try to build them without CUDA support. We need to
// clean our build and testing infrastructure first.

// PTX kernel compiled from:
//
// __global__ void memset32(int64_t n, uint32_t value, uint32_t* dst)
// {
//   int i = blockIdx.x*blockDim.x + threadIdx.x;
//   if (i < n) dst[i] = value;
// }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
inline constexpr absl::string_view kMemset32Kernel = R"(
.version 4.0
.target sm_50
.address_size 64

.visible .entry memset32(
        .param .u64 memset32_param_0,
        .param .u32 memset32_param_1,
        .param .u64 memset32_param_2
)
{
        .reg .pred      %p<2>;
        .reg .b32       %r<6>;
        .reg .b64       %rd<7>;
        .loc    1 3 0

        ld.param.u64    %rd3, [memset32_param_0];
        ld.param.u32    %r1, [memset32_param_1];
        ld.param.u64    %rd2, [memset32_param_2];
        .loc    1 5 3
        mov.u32         %r2, %ctaid.x;
        mov.u32         %r3, %ntid.x;
        mov.u32         %r4, %tid.x;
        mad.lo.s32      %r5, %r2, %r3, %r4;
        .loc    1 6 3
        cvt.s64.s32     %rd1, %r5;
        setp.ge.s64     %p1, %rd1, %rd3;
        @%p1 bra        $L__BB0_2;

        .loc    1 5 3
        cvta.to.global.u64      %rd4, %rd2;
        .loc    1 6 3
        shl.b64         %rd5, %rd1, 2;
        add.s64         %rd6, %rd4, %rd5;
        st.global.u32   [%rd6], %r1;

$L__BB0_2:
        .loc    1 7 1
        ret;

})";

ComputationIdCmd::ComputationIdCmd(BufferAllocation::Slice dest, Kind kind)
    : CommandBufferCmd(CommandBufferCmdType::kComputationIdCmd),
      dest_(dest),
      kind_(kind) {}

CommandBufferCmd::BufferUseVector ComputationIdCmd::buffers() {
  return {{dest_, MemoryAccess::kWrite}};
}

absl::Status ComputationIdCmd::Initialize(const Thunk::InitializeParams& params,
                                          StateManager& state) {
  auto cuda_cc = std::get_if<stream_executor::CudaComputeCapability>(
      &params.executor->GetDeviceDescription().gpu_compute_capability());
  if (cuda_cc != nullptr && memset_kernel_ == nullptr) {
    TF_ASSIGN_OR_RETURN(memset_kernel_,
                        CreateKernel("memset32", 3, kMemset32Kernel,
                                     /*cubin_data=*/{}, params.executor,
                                     /*shared_mem_bytes=*/0));
  }
  return absl::OkStatus();
}

absl::Status ComputationIdCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dest_);

  GlobalDeviceId global_device_id =
      execute_params.collective_params->global_device_id;
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID logical_id,
      execute_params.collective_params->device_assn->LogicalIdForDevice(
          global_device_id));

  uint32_t value = kind_ == Kind::kReplica ? logical_id.replica_id
                                           : logical_id.computation_id;

  VLOG(5) << "ComputationIdCmd"
          << ": kind=" << (kind_ == Kind::kReplica ? "replica" : "partition")
          << "; value=" << value << "; dst=" << dest_ << " (" << dst.opaque()
          << ")";
  auto cuda_cc = std::get_if<stream_executor::CudaComputeCapability>(
      &execute_params.stream->parent()
           ->GetDeviceDescription()
           .gpu_compute_capability());

  if (cuda_cc != nullptr) {
    if (memset_kernel_ == nullptr) {
      return absl::InternalError("Memset kernel not loaded.");
    }

    auto args = se::PackKernelArgs(/*shmem_bytes=*/0, int64_t{1}, value, dst);
    if (create) {
      TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateLaunchNode(
                                     ToDependentNodes(), se::ThreadDim(1),
                                     se::BlockDim(1), *memset_kernel_, *args));
    } else {
      TF_RETURN_IF_ERROR(command_buffer->UpdateLaunchNode(
          node_, se::ThreadDim(1), se::BlockDim(1), *memset_kernel_, *args));
    }
  } else {
    if (create) {
      TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateMemsetNode(
                                     ToDependentNodes(), dst, value,
                                     /*num_elements=*/1));
    } else {
      TF_RETURN_IF_ERROR(command_buffer->UpdateMemsetNode(node_, dst, value,
                                                          /*num_elements=*/1));
    }
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//
LaunchCmd::LaunchCmd(std::string kernel_name,
                     absl::Span<const BufferAllocation::Slice> args,
                     absl::Span<const MemoryAccess> args_access,
                     LaunchDimensions dims, int64_t shmem_bytes)
    : CommandBufferCmd(CommandBufferCmdType::kLaunchCmd),
      kernel_name_(kernel_name),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      dims_(dims),
      shmem_bytes_(shmem_bytes) {}

absl::Status LaunchCmd::Initialize(const Thunk::InitializeParams& params,
                                   StateManager& state) {
  if (kernel_ == nullptr) {
    TF_ASSIGN_OR_RETURN(
        kernel_,
        CreateKernel(kernel_name_, args_.size(), params.src.text,
                     params.src.binary, params.executor, shmem_bytes_));
  }
  return absl::OkStatus();
}

absl::Status LaunchCmd::Record(const Thunk::ExecuteParams& execute_params,
                               const RecordParams& record_params,
                               se::CommandBuffer* command_buffer, bool create) {
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << "; kernel_=" << reinterpret_cast<void*>(kernel_.get())
          << "; shmem_bytes=" << shmem_bytes_
          << "; dependencies=" << dependencies().size()
          << "; buffers=" << BufferUseVectorToString(buffers());

  if (kernel_ == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Kernel not loaded on a command buffer executor: ", kernel_name_));
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    buffers.push_back(buf);
  }

  TF_ASSIGN_OR_RETURN(auto kernel_args,
                      se::PackKernelArgs(buffers, shmem_bytes_));

  if (create) {
    TF_ASSIGN_OR_RETURN(node_,
                        command_buffer->CreateLaunchNode(
                            ToDependentNodes(), dims_.thread_counts_per_block(),
                            dims_.block_counts(), *kernel_, *kernel_args));
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateLaunchNode(
        node_, dims_.thread_counts_per_block(), dims_.block_counts(), *kernel_,
        *kernel_args));
  }
  return absl::OkStatus();
}

CommandBufferCmd::BufferUseVector LaunchCmd::buffers() {
  BufferUseVector buffers;
  for (int32_t i = 0; i < args_.size(); ++i) {
    buffers.emplace_back(args_[i], args_access_[i]);
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// ChildCmd
//===----------------------------------------------------------------------===//

ChildCmd::ChildCmd(std::unique_ptr<CommandBufferCmdSequence> child_cmds)
    : CommandBufferCmd(CommandBufferCmdType::kChildCmd),
      child_cmds_(std::move(child_cmds)) {}

absl::Status ChildCmd::Initialize(const Thunk::InitializeParams& params,
                                  StateManager& state) {
  return child_cmds_->Initialize(params, state);
}

absl::Status ChildCmd::Record(const Thunk::ExecuteParams& execute_params,
                              const RecordParams& record_params,
                              se::CommandBuffer* command_buffer, bool create) {
  if (create) {
    TF_ASSIGN_OR_RETURN(child_command_buffer_,
                        execute_params.stream->parent()->CreateCommandBuffer(
                            se::CommandBuffer::Mode::kNested));
  }

  TF_RETURN_IF_ERROR(child_cmds_->Record(execute_params, record_params,
                                         child_command_buffer_.get()));

  if (create) {
    TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateChildNode(
                                   ToDependentNodes(), *child_command_buffer_));
  } else {
    TF_RETURN_IF_ERROR(
        command_buffer->UpdateChildNode(node_, *child_command_buffer_));
  }
  return absl::OkStatus();
}

CommandBufferCmd::BufferUseVector ChildCmd::buffers() {
  BufferUseVector buffers(child_cmds_->buffers().begin(),
                          child_cmds_->buffers().end());
  return buffers;
}

//===----------------------------------------------------------------------===//
// CustomKernelLaunchCmd
//===----------------------------------------------------------------------===//

CustomKernelLaunchCmd::CustomKernelLaunchCmd(
    absl::Span<const BufferAllocation::Slice> args,
    absl::Span<const MemoryAccess> args_access, CustomKernel custom_kernel)
    : CommandBufferCmd(CommandBufferCmdType::kCustomKernelLaunchCmd),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      custom_kernel_(custom_kernel) {}

absl::Status CustomKernelLaunchCmd::Initialize(
    const Thunk::InitializeParams& params, StateManager& state) {
  if (kernel_ == nullptr) {
    TF_ASSIGN_OR_RETURN(
        kernel_, params.executor->LoadKernel(custom_kernel_.kernel_spec()));
  }
  return absl::OkStatus();
}

absl::Status CustomKernelLaunchCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
  VLOG(5) << "CustomKernelLaunchCmd: custom_kernel=" << custom_kernel_.name();

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    buffers.push_back(buf);
  }

  se::KernelArgsDeviceMemoryArray kernel_args(
      buffers, custom_kernel_.shared_memory_bytes());

  if (create) {
    TF_ASSIGN_OR_RETURN(
        node_, command_buffer->CreateLaunchNode(
                   ToDependentNodes(), custom_kernel_.thread_dims(),
                   custom_kernel_.block_dims(), *kernel_, kernel_args));
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateLaunchNode(
        node_, custom_kernel_.thread_dims(), custom_kernel_.block_dims(),
        *kernel_, kernel_args));
  }
  return absl::OkStatus();
}

CommandBufferCmd::BufferUseVector CustomKernelLaunchCmd::buffers() {
  BufferUseVector buffers;
  for (int32_t i = 0; i < args_.size(); ++i) {
    buffers.emplace_back(args_[i], args_access_[i]);
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

MemcpyDeviceToDeviceCmd::MemcpyDeviceToDeviceCmd(BufferAllocation::Slice dst,
                                                 BufferAllocation::Slice src,
                                                 int64_t num_bytes)
    : CommandBufferCmd(CommandBufferCmdType::kMemcpyDeviceToDeviceCmd),
      dst_(dst),
      src_(src),
      num_bytes_(num_bytes) {}

absl::Status MemcpyDeviceToDeviceCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);
  se::DeviceMemoryBase src =
      execute_params.buffer_allocations->GetDeviceAddress(src_);

  VLOG(5) << "MemcpyDeviceToDeviceCmd: num_bytes = " << num_bytes_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << src_ << " (" << src.opaque() << ")";

  if (num_bytes_ == 0) {
    VLOG(5) << "Replacing MemcpyDeviceToDeviceCmd command of 0 bytes with an "
               "barrier command to keep the original dependencies";
    if (create) {
      TF_ASSIGN_OR_RETURN(node_,
                          command_buffer->CreateEmptyNode(ToDependentNodes()));
    }
  } else {
    if (create) {
      TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateMemcpyD2DNode(
                                     ToDependentNodes(), dst, src, num_bytes_));
    } else {
      TF_RETURN_IF_ERROR(
          command_buffer->UpdateMemcpyD2DNode(node_, dst, src, num_bytes_));
    }
  }
  return absl::OkStatus();
}

CommandBufferCmd::BufferUseVector MemcpyDeviceToDeviceCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}, {src_, MemoryAccess::kRead}};
}

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

MemzeroCmd::MemzeroCmd(BufferAllocation::Slice dst)
    : CommandBufferCmd(CommandBufferCmdType::kMemzeroCmd), dst_(dst) {}

absl::Status MemzeroCmd::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                se::CommandBuffer* command_buffer,
                                bool create) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "Recording MemzeroCmd, Dst: " << dst_ << " (" << dst.opaque()
          << ")";

  if (dst_.size() == 0) {
    if (create) {
      VLOG(5)
          << "Replacing MemzeroCmd command of 0 bytes with a barrier command "
             "to keep the original dependencie";
      TF_ASSIGN_OR_RETURN(node_,
                          command_buffer->CreateEmptyNode(ToDependentNodes()));
    } else {
      // No update operation for empty node
      return absl::OkStatus();
    }
  } else {
    if (create) {
      TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateMemsetNode(
                                     ToDependentNodes(), dst, uint8_t{0},
                                     /*num_elements=*/dst_.size()));
    } else {
      TF_RETURN_IF_ERROR(
          command_buffer->UpdateMemsetNode(node_, dst, uint8_t{0},
                                           /*num_elements=*/dst_.size()));
    }
  }
  return absl::OkStatus();
}

CommandBufferCmd::BufferUseVector MemzeroCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

Memset32Cmd::Memset32Cmd(BufferAllocation::Slice dst, uint32_t bit_pattern)
    : CommandBufferCmd(CommandBufferCmdType::kMemset32Cmd),
      dst_(dst),
      bit_pattern_(bit_pattern) {}

absl::Status Memset32Cmd::Record(const Thunk::ExecuteParams& execute_params,
                                 const RecordParams& record_params,
                                 se::CommandBuffer* command_buffer,
                                 bool create) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "Memset32Cmd: bit_pattern=" << bit_pattern_ << "  Dst: " << dst_
          << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Replacing Memset32Cmd command of 0 bytes with a barrier "
               "command to keep the original dependencies";
    TF_ASSIGN_OR_RETURN(node_,
                        command_buffer->CreateEmptyNode(ToDependentNodes()));
  }
  if (create) {
    TF_ASSIGN_OR_RETURN(node_,
                        command_buffer->CreateMemsetNode(
                            ToDependentNodes(), dst, bit_pattern_,
                            /*num_elements=*/dst_.size() / sizeof(uint32_t)));
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateMemsetNode(
        node_, dst, bit_pattern_,
        /*num_elements=*/dst_.size() / sizeof(uint32_t)));
  }
  return absl::OkStatus();
}

CommandBufferCmd::BufferUseVector Memset32Cmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// IfCmd
//===----------------------------------------------------------------------===//

IfCmd::IfCmd(BufferAllocation::Slice pred,
             std::unique_ptr<CommandBufferCmdSequence> then_commands)
    : CommandBufferCmd(CommandBufferCmdType::kIfCmd),
      pred_(pred),
      then_commands_(std::move(then_commands)) {}

absl::Status IfCmd::Initialize(const Thunk::InitializeParams& params,
                               StateManager& state) {
  return then_commands_->Initialize(params, state);
}

absl::Status IfCmd::Record(const Thunk::ExecuteParams& execute_params,
                           const RecordParams& record_params,
                           se::CommandBuffer* command_buffer, bool create) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);
  VLOG(5) << "Recording IfCmd, pred: " << pred_ << " (" << pred.opaque() << ")";
  if (create) {
    TF_ASSIGN_OR_RETURN(then_cond_handle_,
                        command_buffer->CreateConditionalHandle());
    TF_ASSIGN_OR_RETURN(set_cond_handle_kernel_node_,
                        command_buffer->CreateIfConditionNode(
                            ToDependentNodes(), then_cond_handle_,
                            se::DeviceMemory<bool>(pred)));
    TF_ASSIGN_OR_RETURN(
        auto cond_node_result,
        command_buffer->CreateConditionalNode(
            Dependencies{set_cond_handle_kernel_node_}, then_cond_handle_,
            se::CommandBuffer::ConditionType::kIf));
    then_command_buffer_ = std::move(cond_node_result.command_buffer);
    then_cond_node_ = cond_node_result.node_handle;
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateIfConditionNode(
        set_cond_handle_kernel_node_, then_cond_handle_,
        se::DeviceMemory<bool>(pred)));
  }
  return then_commands_->Record(execute_params, record_params,
                                then_command_buffer_.get());
}

bool IfCmd::force_update() { return then_commands_->force_update(); }

CommandBufferCmd::BufferUseVector IfCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(pred_, MemoryAccess::kRead);
  buffers.insert(then_commands_->buffers().begin(),
                 then_commands_->buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// IfElseCmd
//===----------------------------------------------------------------------===//

IfElseCmd::IfElseCmd(BufferAllocation::Slice pred,
                     std::unique_ptr<CommandBufferCmdSequence> then_commands,
                     std::unique_ptr<CommandBufferCmdSequence> else_commands)
    : CommandBufferCmd(CommandBufferCmdType::kIfElseCmd),
      pred_(pred),
      then_commands_(std::move(then_commands)),
      else_commands_(std::move(else_commands)) {}

absl::Status IfElseCmd::Initialize(const Thunk::InitializeParams& params,
                                   StateManager& state) {
  TF_RETURN_IF_ERROR(then_commands_->Initialize(params, state));
  TF_RETURN_IF_ERROR(else_commands_->Initialize(params, state));
  return absl::OkStatus();
}

absl::Status IfElseCmd::Record(const Thunk::ExecuteParams& execute_params,
                               const RecordParams& record_params,
                               se::CommandBuffer* command_buffer, bool create) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  VLOG(5) << "Recording IfElseCmd, pred: " << pred_ << " (" << pred.opaque()
          << ")";

  if (create) {
    TF_ASSIGN_OR_RETURN(then_cond_handle_,
                        command_buffer->CreateConditionalHandle());
    TF_ASSIGN_OR_RETURN(else_cond_handle_,
                        command_buffer->CreateConditionalHandle());
    TF_ASSIGN_OR_RETURN(set_cond_handle_kernel_node_,
                        command_buffer->CreateIfElseConditionNode(
                            ToDependentNodes(), then_cond_handle_,
                            else_cond_handle_, se::DeviceMemory<bool>(pred)));
    TF_ASSIGN_OR_RETURN(
        auto then_cond_node_result,
        command_buffer->CreateConditionalNode(
            Dependencies{set_cond_handle_kernel_node_}, then_cond_handle_,
            se::CommandBuffer::ConditionType::kIf));
    TF_ASSIGN_OR_RETURN(
        auto else_cond_node_result,
        command_buffer->CreateConditionalNode(
            Dependencies{set_cond_handle_kernel_node_}, else_cond_handle_,
            se::CommandBuffer::ConditionType::kIf));
    then_command_buffer_ = std::move(then_cond_node_result.command_buffer);
    else_command_buffer_ = std::move(else_cond_node_result.command_buffer);
    then_cond_node_ = then_cond_node_result.node_handle;
    else_cond_node_ = else_cond_node_result.node_handle;
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateIfElseConditionNode(
        set_cond_handle_kernel_node_, then_cond_handle_, else_cond_handle_,
        se::DeviceMemory<bool>(pred)));
  }
  TF_RETURN_IF_ERROR(then_commands_->Record(execute_params, record_params,
                                            then_command_buffer_.get()));
  return else_commands_->Record(execute_params, record_params,
                                else_command_buffer_.get());
}

bool IfElseCmd::force_update() {
  return (then_commands_->force_update() || else_commands_->force_update());
}

CommandBufferCmd::BufferUseVector IfElseCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(pred_, MemoryAccess::kRead);
  buffers.insert(then_commands_->buffers().begin(),
                 then_commands_->buffers().end());
  buffers.insert(else_commands_->buffers().begin(),
                 else_commands_->buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

CaseCmd::CaseCmd(
    BufferAllocation::Slice index, bool index_is_bool,
    std::vector<std::unique_ptr<CommandBufferCmdSequence>> branches_commands)
    : CommandBufferCmd(CommandBufferCmdType::kCaseCmd),
      index_(index),
      index_is_bool_(index_is_bool),
      branches_commands_(std::move(branches_commands)) {
  if (VLOG_IS_ON(5)) {
    for (int i = 0; i < branches_commands_.size(); ++i) {
      VLOG(5) << "Branch " << i << ": \n" << branches_commands_[i]->ToString();
    }
  }
}

absl::Status CaseCmd::Record(const Thunk::ExecuteParams& execute_params,
                             const RecordParams& record_params,
                             se::CommandBuffer* command_buffer, bool create) {
  se::DeviceMemoryBase index_memory_base =
      execute_params.buffer_allocations->GetDeviceAddress(index_);

  VLOG(5) << "CaseCmd, index: " << index_ << " (" << index_memory_base.opaque()
          << ") index_is_bool: " << index_is_bool_;

  int64_t num_branches = branches_commands_.size();
  int64_t set_case_handle_batches_num =
      (num_branches + kBranchBatchSize - 1) / kBranchBatchSize;

  if (create) {
    case_branch_handles_.reserve(set_case_handle_batches_num *
                                 kBranchBatchSize);
    for (int64_t i = 0; i < num_branches; ++i) {
      TF_ASSIGN_OR_RETURN(case_branch_handles_.emplace_back(),
                          command_buffer->CreateConditionalHandle());
      VLOG(5) << "Case branch " << i
              << " with handle:  " << case_branch_handles_.back();
    }
    case_branch_handles_.resize(set_case_handle_batches_num * kBranchBatchSize);
    int64_t batch_offset = 0;
    for (int64_t i = 0; i < set_case_handle_batches_num; ++i) {
      batch_offset += i * kBranchBatchSize;
      bool enable_conditional_default = (i == set_case_handle_batches_num - 1);
      TF_ASSIGN_OR_RETURN(
          set_case_handle_kernel_nodes_.emplace_back(),
          command_buffer->CreateCaseConditionNode(
              ToDependentNodes(),
              std::array<CommandBufferConditionalHandle, 8>{
                  case_branch_handles_[batch_offset + 0],
                  case_branch_handles_[batch_offset + 1],
                  case_branch_handles_[batch_offset + 2],
                  case_branch_handles_[batch_offset + 3],
                  case_branch_handles_[batch_offset + 4],
                  case_branch_handles_[batch_offset + 5],
                  case_branch_handles_[batch_offset + 6],
                  case_branch_handles_[batch_offset + 7]},
              se::DeviceMemory<uint8_t>(index_memory_base), index_is_bool_,
              batch_offset, static_cast<int32_t>(num_branches),
              enable_conditional_default));
    }

    branch_command_buffers_.reserve(num_branches);
    for (int64_t i = 0; i < num_branches; ++i) {
      TF_ASSIGN_OR_RETURN(
          auto case_branch_node_result,
          command_buffer->CreateConditionalNode(
              Dependencies(set_case_handle_kernel_nodes_.begin(),
                           set_case_handle_kernel_nodes_.end()),
              case_branch_handles_[i], se::CommandBuffer::ConditionType::kIf));
      branch_command_buffers_.emplace_back(
          std::move(case_branch_node_result.command_buffer));
      cond_nodes_.push_back(case_branch_node_result.node_handle);
    }
  } else {
    int64_t batch_offset = 0;
    for (int64_t i = 0; i < set_case_handle_batches_num; ++i) {
      batch_offset += i * kBranchBatchSize;
      bool enable_conditional_default = (i == set_case_handle_batches_num - 1);
      TF_RETURN_IF_ERROR(command_buffer->UpdateCaseConditionNode(
          set_case_handle_kernel_nodes_[i],
          std::array<CommandBufferConditionalHandle, 8>{
              case_branch_handles_[batch_offset + 0],
              case_branch_handles_[batch_offset + 1],
              case_branch_handles_[batch_offset + 2],
              case_branch_handles_[batch_offset + 3],
              case_branch_handles_[batch_offset + 4],
              case_branch_handles_[batch_offset + 5],
              case_branch_handles_[batch_offset + 6],
              case_branch_handles_[batch_offset + 7]},
          se::DeviceMemory<uint8_t>(index_memory_base), index_is_bool_,
          batch_offset, static_cast<int32_t>(num_branches),
          enable_conditional_default));
    }
  }

  for (int64_t i = 0; i < num_branches; ++i) {
    TF_RETURN_IF_ERROR(branches_commands_[i]->Record(
        execute_params, record_params, branch_command_buffers_[i].get()));
  }
  return absl::OkStatus();
}

absl::Status CaseCmd::Initialize(const Thunk::InitializeParams& params,
                                 StateManager& state) {
  for (auto& branch : branches_commands_) {
    TF_RETURN_IF_ERROR(branch->Initialize(params, state));
  }
  return absl::OkStatus();
}

bool CaseCmd::force_update() {
  return absl::c_any_of(branches_commands_,
                        [](const auto& seq) { return seq->force_update(); });
}

CommandBufferCmd::BufferUseVector CaseCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(index_, MemoryAccess::kRead);
  for (auto& branch : branches_commands_) {
    buffers.insert(branch->buffers().begin(), branch->buffers().end());
  }
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// SetForConditionCmd
//===----------------------------------------------------------------------===//

SetForConditionCmd::SetForConditionCmd(
    CommandBufferConditionalHandle* cond_handle,
    BufferAllocation::Slice loop_counter, int32_t num_iterations)
    : CommandBufferCmd(CommandBufferCmdType::kSetForConditionCmd),
      cond_handle_(cond_handle),
      loop_counter_(loop_counter),
      num_iterations_(num_iterations) {}

absl::Status SetForConditionCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
  se::DeviceMemoryBase loop_counter =
      execute_params.buffer_allocations->GetDeviceAddress(loop_counter_);
  if (create) {
    TF_ASSIGN_OR_RETURN(
        node_, command_buffer->CreateForConditionNode(
                   ToDependentNodes(), *cond_handle_,
                   se::DeviceMemory<int32_t>(loop_counter), num_iterations_));
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateForConditionNode(
        node_, *cond_handle_, se::DeviceMemory<int32_t>(loop_counter),
        num_iterations_));
  }
  return absl::OkStatus();
}

CommandBufferCmd::BufferUseVector SetForConditionCmd::buffers() {
  return {{loop_counter_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// ForCmd
//===----------------------------------------------------------------------===//

ForCmd::ForCmd(int32_t num_iterations, BufferAllocation::Slice loop_counter,
               std::unique_ptr<CommandBufferCmdSequence> body_commands)
    : CommandBufferCmd(CommandBufferCmdType::kForCmd),
      num_iterations_(num_iterations),
      loop_counter_(loop_counter),
      body_commands_(std::move(body_commands)) {
  body_and_predict_commands_ = std::make_unique<CommandBufferCmdSequence>();
  body_and_predict_commands_->Append(
      std::make_unique<ChildCmd>(body_commands_->Clone()));
  body_and_predict_commands_->Append(std::make_unique<SetForConditionCmd>(
      &cond_handle_, loop_counter_, num_iterations_));
}

absl::Status ForCmd::Initialize(const Thunk::InitializeParams& params,
                                StateManager& state) {
  return body_and_predict_commands_->Initialize(params, state);
}

absl::Status ForCmd::Record(const Thunk::ExecuteParams& execute_params,
                            const RecordParams& record_params,
                            se::CommandBuffer* command_buffer, bool create) {
  se::DeviceMemoryBase loop_counter =
      execute_params.buffer_allocations->GetDeviceAddress(loop_counter_);

  VLOG(5) << "ForCmd: num_iterations=" << num_iterations_
          << "; body_commands=" << body_and_predict_commands_->size()
          << "  loop_counter: " << loop_counter_ << " ("
          << loop_counter.opaque() << ")";

  if (create) {
    TF_ASSIGN_OR_RETURN(
        initialize_counter_node_,
        command_buffer->CreateMemsetNode(ToDependentNodes(), loop_counter,
                                         static_cast<uint32_t>(0),
                                         /*num_elements=*/1));

    TF_ASSIGN_OR_RETURN(cond_handle_,
                        command_buffer->CreateConditionalHandle());
    TF_ASSIGN_OR_RETURN(
        set_cond_handle_node_,
        command_buffer->CreateForConditionNode(
            Dependencies{initialize_counter_node_}, cond_handle_,
            se::DeviceMemory<int32_t>(loop_counter), num_iterations_));
    TF_ASSIGN_OR_RETURN(auto cond_node_result,
                        command_buffer->CreateConditionalNode(
                            Dependencies{set_cond_handle_node_}, cond_handle_,
                            se::CommandBuffer::ConditionType::kWhile));
    body_command_buffer_ = std::move(cond_node_result.command_buffer);
    cond_node_ = cond_node_result.node_handle;
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateMemsetNode(
        initialize_counter_node_, loop_counter, static_cast<uint32_t>(0),
        /*num_elements=*/1));
    TF_RETURN_IF_ERROR(command_buffer->UpdateForConditionNode(
        set_cond_handle_node_, cond_handle_,
        se::DeviceMemory<int32_t>(loop_counter), num_iterations_));
  }
  return body_and_predict_commands_->Record(execute_params, record_params,
                                            body_command_buffer_.get());
}

bool ForCmd::force_update() { return body_commands_->force_update(); }

CommandBufferCmd::BufferUseVector ForCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(loop_counter_, MemoryAccess::kWrite);
  buffers.insert(body_commands_->buffers().begin(),
                 body_commands_->buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// SetWhileConditionCmd
//===----------------------------------------------------------------------===//

SetWhileConditionCmd::SetWhileConditionCmd(
    CommandBufferConditionalHandle* cond_handle, BufferAllocation::Slice pred)
    : CommandBufferCmd(CommandBufferCmdType::kSetWhileConditionCmd),
      cond_handle_(cond_handle),
      pred_(pred) {}

absl::Status SetWhileConditionCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);
  if (create) {
    TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateWhileConditionNode(
                                   ToDependentNodes(), *cond_handle_,
                                   se::DeviceMemory<bool>(pred)));
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateWhileConditionNode(
        node_, *cond_handle_, se::DeviceMemory<bool>(pred)));
  }
  return absl::OkStatus();
}

CommandBufferCmd::BufferUseVector SetWhileConditionCmd::buffers() {
  return {{pred_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

WhileCmd::WhileCmd(BufferAllocation::Slice pred,
                   std::unique_ptr<CommandBufferCmdSequence> cond_commands,
                   std::unique_ptr<CommandBufferCmdSequence> body_commands)
    : CommandBufferCmd(CommandBufferCmdType::kWhileCmd),
      pred_(pred),
      cond_commands_(std::move(cond_commands)),
      body_commands_(std::move(body_commands)) {}

absl::Status WhileCmd::Record(const Thunk::ExecuteParams& execute_params,
                              const RecordParams& record_params,
                              se::CommandBuffer* command_buffer, bool create) {
  if (create) {
    TF_ASSIGN_OR_RETURN(cond_handle_,
                        command_buffer->CreateConditionalHandle());

    initialize_commands_.Append(
        std::make_unique<ChildCmd>(cond_commands_->Clone()));
    initialize_commands_.Append(
        std::make_unique<SetWhileConditionCmd>(&cond_handle_, pred_));
    TF_ASSIGN_OR_RETURN(initialize_command_buffer_,
                        execute_params.stream->parent()->CreateCommandBuffer(
                            se::CommandBuffer::Mode::kNested));

    TF_RETURN_IF_ERROR(initialize_commands_.Record(
        execute_params, record_params, initialize_command_buffer_.get()));

    TF_ASSIGN_OR_RETURN(initialize_while_handle_node_,
                        command_buffer->CreateChildNode(
                            ToDependentNodes(), *initialize_command_buffer_));

    loop_commands_.Append(std::make_unique<ChildCmd>(body_commands_->Clone()));
    loop_commands_.Append(std::make_unique<ChildCmd>(cond_commands_->Clone()));
    loop_commands_.Append(
        std::make_unique<SetWhileConditionCmd>(&cond_handle_, pred_));

    TF_ASSIGN_OR_RETURN(
        auto cond_node_results,
        command_buffer->CreateConditionalNode(
            Dependencies{initialize_while_handle_node_}, cond_handle_,
            se::CommandBuffer::ConditionType::kWhile));
    loop_command_buffer_ = std::move(cond_node_results.command_buffer);
    cond_node_ = cond_node_results.node_handle;
    TF_RETURN_IF_ERROR(loop_commands_.Record(execute_params, record_params,
                                             loop_command_buffer_.get()));
  } else {
    TF_RETURN_IF_ERROR(initialize_commands_.Record(
        execute_params, record_params, initialize_command_buffer_.get()));

    TF_RETURN_IF_ERROR(loop_commands_.Record(execute_params, record_params,
                                             loop_command_buffer_.get()));

    TF_RETURN_IF_ERROR(command_buffer->UpdateChildNode(
        initialize_while_handle_node_, *initialize_command_buffer_));
  }
  return absl::OkStatus();
}

absl::Status WhileCmd::Initialize(const Thunk::InitializeParams& params,
                                  StateManager& state) {
  TF_RETURN_IF_ERROR(body_commands_->Initialize(params, state));
  return cond_commands_->Initialize(params, state);
}

bool WhileCmd::force_update() {
  return (cond_commands_->force_update() || body_commands_->force_update());
}

CommandBufferCmd::BufferUseVector WhileCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(pred_, MemoryAccess::kWrite);
  buffers.insert(cond_commands_->buffers().begin(),
                 cond_commands_->buffers().end());
  buffers.insert(body_commands_->buffers().begin(),
                 body_commands_->buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

GemmCmd::GemmCmd(GemmConfig config, const BufferAllocation::Slice& lhs_buffer,
                 const BufferAllocation::Slice& rhs_buffer,
                 const BufferAllocation::Slice& output_buffer,
                 const BufferAllocation::Slice& workspace, bool deterministic)
    : TracedCommandBufferCmd(CommandBufferCmdType::kGemmCmd),
      config_(config),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      workspace_(workspace),
      deterministic_(deterministic) {}

absl::Status GemmCmd::Initialize(const Thunk::InitializeParams& params,
                                 StateManager& state) {
  if (!params.stream->parent()->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support for GemmCmd");
  }
  return absl::OkStatus();
}

absl::Status GemmCmd::Record(const Thunk::ExecuteParams& execute_params,
                             const RecordParams& record_params,
                             se::CommandBuffer* command_buffer, bool create) {
  se::DeviceMemoryBase lhs =
      execute_params.buffer_allocations->GetDeviceAddress(lhs_buffer_);
  se::DeviceMemoryBase rhs =
      execute_params.buffer_allocations->GetDeviceAddress(rhs_buffer_);
  se::DeviceMemoryBase out =
      execute_params.buffer_allocations->GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase workspace =
      execute_params.buffer_allocations->GetDeviceAddress(workspace_);

  VLOG(5) << "GemmCmd: deterministic=" << deterministic_;
  VLOG(5) << "  Lhs: " << lhs_buffer_ << " (" << lhs.opaque() << ")";
  VLOG(5) << "  Lhs: " << rhs_buffer_ << " (" << rhs.opaque() << ")";
  VLOG(5) << "  Out: " << output_buffer_ << " (" << out.opaque() << ")";
  VLOG(5) << "  Workspace: " << workspace_ << " (" << workspace.opaque() << ")";

  return RecordTracedCommandBuffer(
      execute_params, record_params, command_buffer, create,
      [&](se::Stream* stream) {
        return RunGemm(config_, lhs, rhs, out, workspace, deterministic_,
                       stream);
      });
}

CommandBufferCmd::BufferUseVector GemmCmd::buffers() {
  return {{lhs_buffer_, MemoryAccess::kRead},
          {rhs_buffer_, MemoryAccess::kRead},
          {output_buffer_, MemoryAccess::kWrite},
          {workspace_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// CublasLtCmd
//===----------------------------------------------------------------------===//

CublasLtCmd::CublasLtCmd(
    GemmConfig gemm_config, se::gpu::BlasLt::Epilogue epilogue,
    int64_t algorithm_idx, BufferAllocation::Slice a_buffer,
    BufferAllocation::Slice b_buffer, BufferAllocation::Slice c_buffer,
    BufferAllocation::Slice d_buffer,
    BufferAllocation::Slice bias_buffer /* may be null */,
    BufferAllocation::Slice aux_buffer /* may be null */,
    BufferAllocation::Slice a_scale_buffer /* may be null */,
    BufferAllocation::Slice b_scale_buffer /* may be null */,
    BufferAllocation::Slice c_scale_buffer /* may be null */,
    BufferAllocation::Slice d_scale_buffer /* may be null */,
    BufferAllocation::Slice d_amax_buffer /* may be null */,
    BufferAllocation::Slice workspace_buffer)
    : TracedCommandBufferCmd(CommandBufferCmdType::kCublasLtCmd),
      gemm_config_(gemm_config),
      epilogue_(epilogue),
      algorithm_idx_(algorithm_idx),
      a_buffer_(a_buffer),
      b_buffer_(b_buffer),
      c_buffer_(c_buffer),
      d_buffer_(d_buffer),
      bias_buffer_(bias_buffer),
      aux_buffer_(aux_buffer),
      a_scale_buffer_(a_scale_buffer),
      b_scale_buffer_(b_scale_buffer),
      c_scale_buffer_(c_scale_buffer),
      d_scale_buffer_(d_scale_buffer),
      d_amax_buffer_(d_amax_buffer),
      workspace_buffer_(workspace_buffer) {}

absl::StatusOr<se::gpu::BlasLt::MatmulPlan*> CublasLtCmd::GetMatmulPlan(
    const se::Stream* stream) {
  auto it = matmul_plans_cache_.find(stream);
  if (it != matmul_plans_cache_.end()) return it->second.get();
  TF_ASSIGN_OR_RETURN(auto plan, se::gpu::BlasLt::GetMatmulPlan(
                                     stream, gemm_config_, epilogue_));
  auto [it_insert, _] = matmul_plans_cache_.emplace(stream, std::move(plan));
  return it_insert->second.get();
}

absl::StatusOr<se::gpu::BlasLt::MatmulAlgorithm>
CublasLtCmd::GetMatmulAlgorithm(const se::Stream* stream,
                                const se::gpu::BlasLt::MatmulPlan* plan,
                                int64_t max_workspace) {
  auto it = matmul_algorithm_cache_.find(plan);
  if (it != matmul_algorithm_cache_.end()) return it->second;
  TF_ASSIGN_OR_RETURN(
      auto algorithms,
      plan->GetAlgorithms(stream, /*max_algorithm_count*/ 128,
                          /*max_workspace_size*/ max_workspace));
  TF_RET_CHECK(algorithm_idx_ >= 0 && algorithm_idx_ < algorithms.size());
  auto [it_insert, _] =
      matmul_algorithm_cache_.emplace(plan, algorithms[algorithm_idx_]);
  return it_insert->second;
}

absl::Status CublasLtCmd::Initialize(const Thunk::InitializeParams& params,
                                     StateManager& state) {
  if (!params.stream->parent()->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support for GemmCmd");
  }
  // Populate plan and algorithm cache;
  TF_ASSIGN_OR_RETURN(auto plan, GetMatmulPlan(params.stream));
  TF_RETURN_IF_ERROR(
      GetMatmulAlgorithm(params.stream, plan, workspace_buffer_.size())
          .status());
  return absl::OkStatus();
}

absl::Status CublasLtCmd::Record(const Thunk::ExecuteParams& execute_params,
                                 const RecordParams& record_params,
                                 se::CommandBuffer* command_buffer,
                                 bool create) {
  TF_ASSIGN_OR_RETURN(auto plan, GetMatmulPlan(execute_params.stream));
  TF_ASSIGN_OR_RETURN(auto algorithm,
                      GetMatmulAlgorithm(execute_params.stream, plan,
                                         workspace_buffer_.size()));

  const BufferAllocations& allocs = *execute_params.buffer_allocations;

  se::DeviceMemoryBase bias, a_scale, b_scale, c_scale, d_scale, aux, d_amax;
  if (bias_buffer_.allocation() != nullptr) {
    bias = allocs.GetDeviceAddress(bias_buffer_);
  }
  if (a_scale_buffer_.allocation() != nullptr) {
    a_scale = allocs.GetDeviceAddress(a_scale_buffer_);
  }
  if (b_scale_buffer_.allocation() != nullptr) {
    b_scale = allocs.GetDeviceAddress(b_scale_buffer_);
  }
  if (c_scale_buffer_.allocation() != nullptr) {
    c_scale = allocs.GetDeviceAddress(c_scale_buffer_);
  }
  if (d_scale_buffer_.allocation() != nullptr) {
    d_scale = allocs.GetDeviceAddress(d_scale_buffer_);
  }
  if (d_amax_buffer_.allocation() != nullptr) {
    d_amax = allocs.GetDeviceAddress(d_amax_buffer_);
  }
  if (aux_buffer_.allocation() != nullptr) {
    aux = allocs.GetDeviceAddress(aux_buffer_);
  }

  VLOG(5) << "Recording CublasLtCmd ";
  VLOG(5) << "  a_buffer: " << a_buffer_.ToString();
  VLOG(5) << "  b_buffer: " << b_buffer_.ToString();
  VLOG(5) << "  c_buffer: " << c_buffer_.ToString();
  VLOG(5) << "  d_buffer: " << d_buffer_.ToString();
  VLOG(5) << "  bias_buffer: " << bias_buffer_.ToString();
  VLOG(5) << "  aux_buffer: " << aux_buffer_.ToString();
  VLOG(5) << "  a_scale_buffer: " << a_scale_buffer_.ToString();
  VLOG(5) << "  b_scale_buffer: " << b_scale_buffer_.ToString();
  VLOG(5) << "  c_scale_buffer: " << c_scale_buffer_.ToString();
  VLOG(5) << "  d_scale_buffer: " << d_scale_buffer_.ToString();
  VLOG(5) << "  d_amax_buffer: " << d_amax_buffer_.ToString();
  VLOG(5) << "  workspace_buffer: " << workspace_buffer_.ToString();

  return RecordTracedCommandBuffer(
      execute_params, record_params, command_buffer, create,
      [&](se::Stream* stream) {
        return plan->ExecuteOnStream(
            stream, allocs.GetDeviceAddress(a_buffer_),
            allocs.GetDeviceAddress(b_buffer_),
            allocs.GetDeviceAddress(c_buffer_),
            allocs.GetDeviceAddress(d_buffer_), bias, aux, a_scale, b_scale,
            c_scale, d_scale, d_amax, algorithm,
            allocs.GetDeviceAddress(workspace_buffer_));
      });
}

CommandBufferCmd::BufferUseVector CublasLtCmd::buffers() {
  BufferUseVector buffer_use;
  buffer_use.reserve(13);
  buffer_use.push_back({a_buffer_, MemoryAccess::kRead});
  buffer_use.push_back({b_buffer_, MemoryAccess::kRead});
  buffer_use.push_back({c_buffer_, MemoryAccess::kRead});
  buffer_use.push_back({d_buffer_, MemoryAccess::kWrite});
  buffer_use.push_back({workspace_buffer_, MemoryAccess::kWrite});

  if (bias_buffer_.allocation() != nullptr) {
    buffer_use.push_back({bias_buffer_, MemoryAccess::kRead});
  }
  if (a_scale_buffer_.allocation() != nullptr) {
    buffer_use.push_back({a_scale_buffer_, MemoryAccess::kRead});
  }
  if (b_scale_buffer_.allocation() != nullptr) {
    buffer_use.push_back({b_scale_buffer_, MemoryAccess::kRead});
  }
  if (c_scale_buffer_.allocation() != nullptr) {
    buffer_use.push_back({c_scale_buffer_, MemoryAccess::kRead});
  }
  if (d_scale_buffer_.allocation() != nullptr) {
    buffer_use.push_back({d_scale_buffer_, MemoryAccess::kRead});
  }
  if (aux_buffer_.allocation() != nullptr) {
    buffer_use.push_back({aux_buffer_, MemoryAccess::kWrite});
  }
  if (d_amax_buffer_.allocation() != nullptr) {
    buffer_use.push_back({d_amax_buffer_, MemoryAccess::kRead});
  }
  return buffer_use;
}

//===----------------------------------------------------------------------===//
// CuDnnCmd
//===----------------------------------------------------------------------===//

CuDnnCmd::CuDnnCmd(absl::Span<const BufferAllocation::Slice> args,
                   const std::shared_ptr<se::dnn::LazyDnnGraph> graph)
    : TracedCommandBufferCmd(CommandBufferCmdType::kCuDnnCmd),
      args_(args.cbegin(), args.cend()),
      graph_(graph) {}

absl::Status CuDnnCmd::Initialize(const Thunk::InitializeParams& params,
                                  StateManager&) {
  if (!params.stream->parent()->AsDnn()) {
    return absl::InternalError("Failed to initialize DNN support for CuDnnCmd");
  }
  return absl::OkStatus();
}

absl::Status CuDnnCmd::Record(const Thunk::ExecuteParams& execute_params,
                              const RecordParams& record_params,
                              se::CommandBuffer* command_buffer, bool create) {
  CHECK(graph_ != nullptr);
  std::vector<se::DeviceMemoryBase> operands;
  operands.reserve(args_.size());
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    operands.push_back(buf);
  }

  return RecordTracedCommandBuffer(
      execute_params, record_params, command_buffer, create,
      [&](se::Stream* stream) {
        return graph_->get()->Execute(
            *stream, absl::Span<se::DeviceMemoryBase>(operands),
            execute_params.collective_params->local_device_ordinal);
      });
}

CommandBufferCmd::BufferUseVector CuDnnCmd::buffers() {
  CommandBufferCmd::BufferUseVector buffer_use;
  buffer_use.reserve(args_.size());
  for (int i = 0; i < args_.size() - 1; ++i) {
    buffer_use.push_back({args_[i], MemoryAccess::kRead});
  }
  buffer_use.push_back({args_.back(), MemoryAccess::kWrite});
  return buffer_use;
}

//===----------------------------------------------------------------------===//
// CustomCallCmd
//===----------------------------------------------------------------------===//

absl::Status CustomCallCmd::Record(const Thunk::ExecuteParams& execute_params,
                                   const RecordParams& record_params,
                                   se::CommandBuffer* command_buffer,
                                   bool create) {
  if (handler_ == nullptr) {
    return RecordLegacyCustomCall(execute_params, record_params, command_buffer,
                                  create);
  }
  return RecordXlaFfiCall(execute_params, record_params, command_buffer,
                          create);
}

namespace {
// Records each buffer associated with each slice into the provided vector.
// Returns an error if any of the slices is missing a buffer allocation.
absl::Status GetBuffers(
    const Thunk::ExecuteParams& execute_params,
    absl::Span<const std::optional<CustomCallCmd::Slice>> slices,
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

absl::Status CustomCallCmd::RecordLegacyCustomCall(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
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

  if (create) {
    TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateChildNode(
                                   ToDependentNodes(), *nested_cmd));
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateChildNode(node_, *nested_cmd));
  }
  return absl::OkStatus();
}

absl::Status CustomCallCmd::RecordXlaFfiCall(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
  // TODO(ezhulenev): This is not the most optimal approach, as we'll be doing
  // a lot of extra allocation on every call. We have to keep attributes
  // separate from arguments, as they do not change after thunk is
  // constructed.
  ffi::CallFrameBuilder builder(operands_.size(), results_.size());

  VLOG(5) << "CustomCallCmd: target_name=" << target_name_;

  for (int i = 0; i < operands_.size(); ++i) {
    const std::optional<Slice>& slice = operands_[i];
    // TODO(ezhulenev): Add a token argument type to XLA:FFI.
    if (!slice.has_value()) {
      return Internal("FFI handlers do not support tokens (yet)!");
    }

    if (!slice->slice.allocation())
      return Internal("custom call input missing buffer allocation");

    se::DeviceMemoryBase buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slice->slice);
    VLOG(5) << "  Operand " << i << ": " << slice->slice << " ("
            << buffer.opaque() << ")";
    builder.AddBufferArg(buffer, slice->shape.element_type(),
                         slice->shape.dimensions());
  }

  for (int i = 0; i < results_.size(); ++i) {
    const std::optional<Slice>& slice = results_[i];
    // TODO(ezhulenev): Add a token argument type to XLA:FFI.
    if (!slice.has_value()) {
      return Internal("FFI handlers do not support tokens (yet)!");
    }

    if (!slice->slice.allocation())
      return Internal("custom call input missing buffer allocation");

    se::DeviceMemoryBase buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slice->slice);
    VLOG(5) << "  Result " << i << ": " << slice->slice << " ("
            << buffer.opaque() << ")";
    builder.AddBufferRet(buffer, slice->shape.element_type(),
                         slice->shape.dimensions());
  }

  ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Append(attributes_);
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

  RunId run_id = execute_params.collective_params->run_id;

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            ffi::CallOptions options = {
                run_id, execute_params.buffer_allocations->device_ordinal(),
                ffi::CallOptions::GpuOptions{
                    stream,
                    execute_params.buffer_allocations->memory_allocator()},
                /*called_computation=*/nullptr,  // TODO(b/342285364)
                execute_params.ffi_execution_context};
            return ffi::Call(handler_, call_frame, options);
          }));

  if (create) {
    TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateChildNode(
                                   ToDependentNodes(), *nested_cmd));
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateChildNode(node_, *nested_cmd));
  }
  return absl::OkStatus();
}

CommandBufferCmd::BufferUseVector CustomCallCmd::buffers() {
  CommandBufferCmd::BufferUseVector buffer_use;
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<Slice>& slice : slices) {
      if (!slice.has_value()) continue;
      buffer_use.push_back({slice->slice, MemoryAccess::kWrite});
    }
  }
  return buffer_use;
}

//===----------------------------------------------------------------------===//
// BarrierCmd
//===----------------------------------------------------------------------===//

BarrierCmd::BarrierCmd()
    : CommandBufferCmd(CommandBufferCmdType::kBarrierCmd) {}

absl::Status BarrierCmd::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                se::CommandBuffer* command_buffer,
                                bool create) {
  VLOG(5) << "Record BarrierCmd";
  if (create) {
    TF_ASSIGN_OR_RETURN(node_,
                        command_buffer->CreateEmptyNode(ToDependentNodes()));
  }
  return absl::OkStatus();
}

BarrierCmd::BufferUseVector BarrierCmd::buffers() { return {}; }

//===----------------------------------------------------------------------===//
// EmptyCmd
//===----------------------------------------------------------------------===//

EmptyCmd::EmptyCmd(DependencyCmdSet dependencies)
    : CommandBufferCmd(CommandBufferCmdType::kEmptyCmd) {
  for (const CommandBufferCmd* cmd : dependencies) {
    add_dependency(cmd);
  }
}

absl::Status EmptyCmd::Record(const Thunk::ExecuteParams& execute_params,
                              const RecordParams& record_params,
                              se::CommandBuffer* command_buffer, bool create) {
  VLOG(5) << "Record EmptyCmd,dependencies: "
          << DependencySetToString(dependencies());
  if (create) {
    TF_ASSIGN_OR_RETURN(node_,
                        command_buffer->CreateEmptyNode(ToDependentNodes()));
  }
  return absl::OkStatus();
}

EmptyCmd::BufferUseVector EmptyCmd::buffers() { return {}; }

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

CollectiveCmd::CollectiveCmd(CommandBufferCmdType cmd_type,
                             CollectiveConfig config)
    : CommandBufferCmd(cmd_type), config_(config) {}

absl::Status CollectiveCmd::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequestsInterface& resource_requests) {
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(params));
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(collectives, *params.collective_params,
                      config_.replica_groups, config_.group_mode,
                      GetAsyncStreamKind()));
  TF_ASSIGN_OR_RETURN(
      size_t num_local_participants,
      GetNumLocalParticipants(*params.collective_params, config_.replica_groups,
                              config_.group_mode));
  return resource_requests.AddClique(clique_key, num_local_participants);
}

absl::Status CollectiveCmd::RecordTracedCommandBuffer(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create, absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  if (execute_params.mock_collectives) {
    if (create) {
      // Treat mock collectives as a barrier with the same dependencies.
      TF_ASSIGN_OR_RETURN(node_,
                          command_buffer->CreateEmptyNode(ToDependentNodes()));
    }
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::CommandBuffer> nested_cmd,
                      se::TraceCommandBufferFactory::Create(
                          execute_params.stream->parent(),
                          execute_params.command_buffer_trace_stream, trace));
  if (create) {
    TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateChildNode(
                                   ToDependentNodes(), *nested_cmd));
  } else {
    TF_RETURN_IF_ERROR(command_buffer->UpdateChildNode(node_, *nested_cmd));
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

AllReduceCmd::AllReduceCmd(CollectiveConfig config,
                           ReductionKind reduction_kind,
                           absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kAllReduceCmd, config),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllReduceCmd::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  se::CommandBuffer* command_buffer,
                                  bool create) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config_.operand_element_type));

  VLOG(5) << "AllReduceCmd: reduction=" << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllReduceCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));

  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config_.replica_groups,
              config_.group_mode, GetAsyncStreamKind()));

  return RecordTracedCommandBuffer(
      execute_params, record_params, command_buffer, create,
      [&](se::Stream* stream) {
        return RunAllReduce(collectives, reduction_kind_, device_buffers,
                            *stream, comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector AllReduceCmd::buffers() {
  BufferUseVector buffer_use;
  for (auto& buffer : buffers_) {
    buffer_use.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_use.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_use;
}

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

ReduceScatterCmd::ReduceScatterCmd(
    CollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kReduceScatter, config),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status ReduceScatterCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config_.operand_element_type));

  VLOG(5) << "ReduceScatterCmd: reduction="
          << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "ReduceScatterCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));

  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config_.replica_groups,
              config_.group_mode, GetAsyncStreamKind()));

  return RecordTracedCommandBuffer(
      execute_params, record_params, command_buffer, create,
      [&](se::Stream* stream) {
        return RunReduceScatter(collectives, reduction_kind_, device_buffers,
                                *stream, comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector ReduceScatterCmd::buffers() {
  BufferUseVector buffer_use;
  for (auto& buffer : buffers_) {
    buffer_use.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_use.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_use;
}

//===----------------------------------------------------------------------===//
// AllToAllCmd
//===----------------------------------------------------------------------===//

AllToAllCmd::AllToAllCmd(CollectiveConfig config, bool has_split_dimension,
                         absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kAllToAll, config),
      has_split_dimension_(has_split_dimension),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllToAllCmd::Record(const Thunk::ExecuteParams& execute_params,
                                 const RecordParams& record_params,
                                 se::CommandBuffer* command_buffer,
                                 bool create) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config_.operand_element_type));

  VLOG(5) << "Record AllToAllCmd, has_split_dimension=" << has_split_dimension_;

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "ReduceScatterCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));
  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config_.replica_groups,
              config_.group_mode, GetAsyncStreamKind()));

  return RecordTracedCommandBuffer(
      execute_params, record_params, command_buffer, create,
      [&](se::Stream* stream) {
        return RunAllToAll(collectives, has_split_dimension_, device_buffers,
                           *stream, comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector AllToAllCmd::buffers() {
  BufferUseVector buffer_use;
  for (auto& buffer : buffers_) {
    buffer_use.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_use.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_use;
}

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

AllGatherCmd::AllGatherCmd(CollectiveConfig config,
                           absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kAllGatherCmd, config),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllGatherCmd::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  se::CommandBuffer* command_buffer,
                                  bool create) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config_.operand_element_type));

  if VLOG_IS_ON (5) {
    VLOG(5) << "Recording AllGatherCmd.";
    for (size_t i = 0; i < device_buffers.size(); ++i) {
      VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
              << device_buffers[i].source_buffer.opaque() << ")";
      VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
              << device_buffers[i].destination_buffer.opaque() << ")";
    }
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllGatherCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));

  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config_.replica_groups,
              config_.group_mode, GetAsyncStreamKind()));

  return RecordTracedCommandBuffer(
      execute_params, record_params, command_buffer, create,
      [&](se::Stream* stream) {
        return RunAllGather(collectives, device_buffers, *stream,
                            comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector AllGatherCmd::buffers() {
  BufferUseVector buffer_use;
  for (auto& buffer : buffers_) {
    buffer_use.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_use.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_use;
}

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

CollectiveBroadcastCmd::CollectiveBroadcastCmd(
    CollectiveConfig config, absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kCollectiveBroadcastCmd, config),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status CollectiveBroadcastCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config_.operand_element_type));

  VLOG(5) << "Recording CollectiveBroadcastCmd";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "CollectiveBroadcastCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(execute_params));

  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *execute_params.collective_params,
              *execute_params.collective_cliques, config_.replica_groups,
              config_.group_mode, GetAsyncStreamKind()));

  return RecordTracedCommandBuffer(
      execute_params, record_params, command_buffer, create,
      [&](se::Stream* stream) {
        return RunCollectiveBroadcast(device_buffers, *stream, comm_handle.comm,
                                      collectives);
      });
}

CommandBufferCmd::BufferUseVector CollectiveBroadcastCmd::buffers() {
  BufferUseVector buffer_use;
  for (auto& buffer : buffers_) {
    buffer_use.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_use.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_use;
}

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

DynamicSliceFusionCmd::DynamicSliceFusionCmd(
    std::unique_ptr<CommandBufferCmdSequence> embedded_commands,
    std::vector<std::optional<BufferAllocation::Slice>> arguments,
    std::vector<std::unique_ptr<BufferAllocation>> fake_allocations,
    std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets,
    std::vector<std::optional<Shape>> orig_shapes,
    std::vector<std::optional<Shape>> sliced_shapes,
    std::vector<std::optional<uint64_t>> offset_byte_sizes)
    : CommandBufferCmd(CommandBufferCmdType::kDynamicSliceFusionCmd),
      embedded_commands_(std::move(embedded_commands)),
      arguments_(arguments),
      fake_allocations_(std::move(fake_allocations)),
      offsets_(offsets),
      orig_shapes_(orig_shapes),
      sliced_shapes_(sliced_shapes),
      offset_byte_sizes_(offset_byte_sizes) {
  // Zip all arguments together to create a list of SliceDef.
  for (auto [arg, offset, orig_shape, sliced_shape, offset_byte_size] :
       llvm::zip_equal(arguments_, offsets_, orig_shapes_, sliced_shapes_,
                       offset_byte_sizes)) {
    slices_.push_back(DynamicSliceThunk::SliceDef{
        arg,
        offset,
        orig_shape,
        sliced_shape,
        offset_byte_size,
    });
  }

  VLOG(0) << "slices_ = " << slices_.size();

  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    embeded_to_origin_slice_map_[argument_idx] = slice.embedded_thunk_argument;
  }

  // Find how many offsets we might have to transfer from device to host and
  // pre-compute host allocation requirements.
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    offsets_allocs_base_.push_back(offsets_allocs_size_);
    if (slice.sliced_shape.has_value()) {
      offsets_allocs_size_ +=
          slice.sliced_shape->dimensions_size() * sizeof(int64_t);
    }
  }
}

// Force update the command when there is any non-constant value slice offset,
// because the memory address might changed if the offset is loop
// iterator or operator outputs even if the parent command's memory pointers
// do not change.
bool DynamicSliceFusionCmd::force_update() {
  return !absl::c_all_of(slices_, [](const DynamicSliceThunk::SliceDef& slice) {
    if (!slice.offsets.has_value()) return true;
    return absl::c_all_of(slice.offsets.value(),
                          [](DynamicSliceThunk::Offset offset) {
                            return std::holds_alternative<int64_t>(offset);
                          });
  });
}

absl::Status DynamicSliceFusionCmd::Initialize(
    const Thunk::InitializeParams& params, StateManager& state) {
  TF_RETURN_IF_ERROR(embedded_commands_->Initialize(params, state));
  if (offsets_alloc_ == nullptr) {
    VLOG(2) << "Allocate " << offsets_allocs_size_
            << " bytes for transferring offsets on executor: "
            << params.executor;
    TF_ASSIGN_OR_RETURN(offsets_alloc_, params.executor->HostMemoryAllocate(
                                            offsets_allocs_size_));
  }
  return absl::OkStatus();
}

absl::Status DynamicSliceFusionCmd::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequestsInterface& resource_requests) {
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    if (slice.offsets.has_value()) {
      TF_RET_CHECK(slice.embedded_thunk_argument.has_value());
      TF_RET_CHECK(slice.orig_shape.has_value());
      TF_RET_CHECK(slice.sliced_shape.has_value());
      TF_RET_CHECK(slice.offset_byte_size.has_value());

      TF_RET_CHECK(slice.orig_shape->IsArray());
      TF_RET_CHECK(slice.sliced_shape->IsArray());

      TF_RET_CHECK(slice.offsets->size() ==
                   slice.orig_shape->dimensions_size());
      TF_RET_CHECK(slice.sliced_shape->dimensions_size() ==
                   slice.orig_shape->dimensions_size());
    }
  }
  TF_RETURN_IF_ERROR(embedded_commands_->Prepare(params, resource_requests));
  return absl::OkStatus();
}

absl::Status DynamicSliceFusionCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    bool create) {
  se::Stream& stream = *execute_params.stream;

  const BufferAllocations& orig_allocations =
      *execute_params.buffer_allocations;
  absl::InlinedVector<se::DeviceMemoryBase, 8> slice_buffers(
      slices_.size(), se::DeviceMemoryBase());

  // Get memory allocation for copying offsets from device.
  int64_t* offsets_alloc = [&] {
    return reinterpret_cast<int64_t*>(offsets_alloc_->opaque());
  }();

  auto offset_value = [&](int64_t arg_idx, int64_t offset_idx) -> int64_t& {
    return offsets_alloc[offsets_allocs_base_.at(arg_idx) + offset_idx];
  };

  VLOG(2) << "Execute address computation thunk: slices=" << slices_.size();
  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    // Skip arguments that do not have buffer slices (tokens).
    if (!slice.embedded_thunk_argument.has_value()) {
      continue;
    }

    // `argument_buffer` will contain the original offset for slice
    // `argument_slice` within `orig_allocations`
    se::DeviceMemoryBase argument_buffer =
        orig_allocations.GetDeviceAddress(*slice.embedded_thunk_argument);

    // If argument is not sliced, just use the original buffer.
    if (!slice.offsets.has_value()) {
      slice_buffers[argument_idx] = argument_buffer;
      continue;
    }

    const Shape& src_shape = *slice.orig_shape;
    const Shape& dst_shape = *slice.sliced_shape;

    absl::InlinedVector<int64_t, 4> slice_starts;
    slice_starts.reserve(dst_shape.dimensions_size());

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

      } else {
        // Transfer slice offset value from device to host.
        auto alloc_slice = std::get<BufferAllocation::Slice>(offset);
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: transfer offset from device " << alloc_slice.ToString();

        se::DeviceMemoryBase offset_src =
            orig_allocations.GetDeviceAddress(alloc_slice);
        int64_t* offset_dst = &offset_value(argument_idx, offset_idx);

        // Copy the `offset_idx`-th component of the offset for the
        // `argument_idx`-th argument from device to host.
        TF_RETURN_IF_ERROR(
            stream.Memcpy(offset_dst, offset_src, *slice.offset_byte_size));
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

    VLOG(2) << "Create sliced argument " << argument_idx << " of shape "
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

  Thunk::ExecuteParams new_params =
      Thunk::ExecuteParams::CloneWithNewAllocations(execute_params,
                                                    slice_allocations);

  if (create) {
    TF_ASSIGN_OR_RETURN(child_command_buffer_,
                        execute_params.stream->parent()->CreateCommandBuffer(
                            se::CommandBuffer::Mode::kNested));
  }

  TF_RETURN_IF_ERROR(embedded_commands_->Record(new_params, record_params,
                                                child_command_buffer_.get()));

  if (create) {
    TF_ASSIGN_OR_RETURN(node_, command_buffer->CreateChildNode(
                                   ToDependentNodes(), *child_command_buffer_));
  } else {
    TF_RETURN_IF_ERROR(
        command_buffer->UpdateChildNode(node_, *child_command_buffer_));
  }

  return absl::OkStatus();
}

std::unique_ptr<CommandBufferCmd> DynamicSliceFusionCmd::Clone() const {
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations_clone;
  for (auto& fake_allocation : fake_allocations_) {
    fake_allocations_clone.push_back(
        std::make_unique<BufferAllocation>(*fake_allocation));
  }
  return std::make_unique<DynamicSliceFusionCmd>(
      embedded_commands_->Clone(), arguments_,
      std::move(fake_allocations_clone), offsets_, orig_shapes_, sliced_shapes_,
      offset_byte_sizes_);
}

CommandBufferCmd::BufferUseVector DynamicSliceFusionCmd::buffers() {
  CommandBufferCmd::BufferUseVector buffers;
  auto embed_buffers = embedded_commands_->buffers();
  for (auto buffer_use : embed_buffers) {
    CHECK(embeded_to_origin_slice_map_[buffer_use.slice().index()].has_value());
    buffers.emplace_back(
        embeded_to_origin_slice_map_[buffer_use.slice().index()].value(),
        buffer_use.access());
  }
  return buffers;
}

}  // namespace xla::gpu
