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
#include "absl/synchronization/mutex.h"
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

// Creates command buffer builder from a cmd sequence.
static se::CommandBuffer::Builder CreateBuilder(
    CommandBufferCmdSequence* commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->Record(*execute_params, *record_params, command_buffer,
                            CommandBufferCmdSequence::RecordMode::kConditional);
  };
}

// Creates command buffer builders from a span of cmd sequences.
static std::vector<se::CommandBuffer::Builder> CreateBuilders(
    absl::Span<CommandBufferCmdSequence> commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  std::vector<se::CommandBuffer::Builder> builders;
  for (CommandBufferCmdSequence& cmd : commands) {
    builders.push_back(CreateBuilder(&cmd, execute_params, record_params));
  }
  return builders;
}

// Creates command buffer execution scope builder from a cmd sequence.
static se::CommandBuffer::Builder CreateExecutionScopeBuilder(
    CommandBufferCmdSequence* commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    CommandBufferCmd::RecordParams params = *record_params;
    return commands->Record(*execute_params, params, command_buffer,
                            CommandBufferCmdSequence::RecordMode::kConditional);
  };
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
  for (const BufferUse& buffer : cmd->buffers()) {
    buffers_.insert(buffer);
    allocs_indices_.insert(buffer.slice().index());
  }

  commands_.push_back({std::move(cmd)});
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
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command->Initialize(params, state));
  }
  return absl::OkStatus();
}

static absl::string_view RecordModeString(
    CommandBufferCmdSequence::RecordMode mode) {
  switch (mode) {
    case CommandBufferCmdSequence::RecordMode::kExclusive:
      return "exclusive";
    case CommandBufferCmdSequence::RecordMode::kConditional:
      return "conditional";
  }
}

absl::Status CommandBufferCmdSequence::Record(
    const Thunk::ExecuteParams& execute_params,
    const CommandBufferCmd::RecordParams& record_params,
    se::CommandBuffer* command_buffer, RecordMode mode) {
  VLOG(3) << "Record " << commands_.size() << " commands into command buffer"
          << "; mode=" << RecordModeString(mode);
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  if (mode == RecordMode::kExclusive) {
    if (command_buffer->state() == se::CommandBuffer::State::kFinalized) {
      TF_RETURN_IF_ERROR(command_buffer->Update());
    }
  }

  // Track the number of commands recorded between barriers.
  int64_t num_recorded_commands = 0;

  for (std::unique_ptr<CommandBufferCmd>& command : commands_) {
    if (execute_params.mock_collectives &&
        dynamic_cast<CollectiveCmd*>(command.get())) {
      continue;
    }

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command->profile_annotation());

    TF_RETURN_IF_ERROR(
        command->Record(execute_params, record_params, command_buffer));
    ++num_recorded_commands;
  }

  if (mode == RecordMode::kExclusive) {
    TF_RETURN_IF_ERROR(command_buffer->Finalize());
  }

  uint64_t end_micros = tsl::Env::Default()->NowMicros();
  VLOG(3) << "Recorded " << commands_.size()
          << " commands into command buffer in " << (end_micros - start_micros)
          << " μs; mode=" << RecordModeString(mode);

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

TracedCommandBufferCmd::TracedCommandBufferCmd(
    CommandBufferCmdType cmd_type, ExecutionStreamId execution_stream_id)
    : CommandBufferCmd(cmd_type, execution_stream_id) {}

absl::Status TracedCommandBufferCmd::AddTracedCommandBuffer(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  auto traced_cmd =
      record_params.state.GetOrCreate<TracedCommandBuffer>(this, [&] {
        const auto& debug_options = xla::GetDebugOptionsFromFlags();
        return std::make_unique<TracedCommandBuffer>(
            this, buffers(), debug_options.xla_cmd_buffer_trace_cache_size());
      });

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      traced_cmd->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace));

  VLOG(5) << "Add nested command buffer";
  return command_buffer->AddNestedCommandBuffer(*nested_cmd, {}).status();
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

ComputationIdCmd::ComputationIdCmd(ExecutionStreamId execution_stream_id,
                                   BufferAllocation::Slice dest, Kind kind)
    : CommandBufferCmd(CommandBufferCmdType::kComputationIdCmd,
                       execution_stream_id),
      dest_(dest),
      kind_(kind) {}

CommandBufferCmd::BufferUseVector ComputationIdCmd::buffers() {
  return {{dest_, MemoryAccess::kWrite}};
}

absl::Status ComputationIdCmd::Initialize(const Thunk::InitializeParams& params,
                                          StateManager& state) {
  auto cuda_cc = std::get_if<stream_executor::CudaComputeCapability>(
      &params.executor->GetDeviceDescription().gpu_compute_capability());
  if (cuda_cc != nullptr) {
    {
      absl::MutexLock lock(&mutex_);
      if (memset_kernels_.contains(params.executor)) return absl::OkStatus();
    }

    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                        CreateKernel("memset32", 3, kMemset32Kernel,
                                     /*cubin_data=*/{}, params.executor,
                                     /*shared_mem_bytes=*/0));

    absl::MutexLock lock(&mutex_);
    memset_kernels_.emplace(params.executor, std::move(kernel));
  }
  return absl::OkStatus();
}

absl::Status ComputationIdCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
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
          << "; value=" << value;
  VLOG(5) << "  Id: " << dest_ << " (" << dst.opaque() << ")";

  auto cuda_cc = std::get_if<stream_executor::CudaComputeCapability>(
      &execute_params.stream->parent()
           ->GetDeviceDescription()
           .gpu_compute_capability());

  if (cuda_cc != nullptr) {
    se::Kernel* memset_kernel = [&] {
      absl::MutexLock lock(&mutex_);
      return memset_kernels_[execute_params.stream->parent()].get();
    }();

    if (memset_kernel == nullptr) {
      return absl::InternalError(
          "Memset kernel not loaded on a command buffer executor");
    }

    auto args = se::PackKernelArgs(/*shmem_bytes=*/0, int64_t{1}, value, dst);
    return command_buffer
        ->Launch(se::ThreadDim(1), se::BlockDim(1), *memset_kernel, *args, {})
        .status();
  } else {
    return command_buffer->Memset(&dst, value, /*num_elements=*/1, {}).status();
  }
}

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

LaunchCmd::LaunchCmd(ExecutionStreamId execution_stream_id,
                     std::string kernel_name,
                     absl::Span<const BufferAllocation::Slice> args,
                     absl::Span<const MemoryAccess> args_access,
                     LaunchDimensions dims, int64_t shmem_bytes)
    : CommandBufferCmd(CommandBufferCmdType::kLaunchCmd, execution_stream_id),
      kernel_name_(std::move(kernel_name)),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      dims_(dims),
      shmem_bytes_(shmem_bytes) {}

absl::Status LaunchCmd::Initialize(const Thunk::InitializeParams& params,
                                   StateManager& state) {
  {
    absl::MutexLock lock(&mutex_);
    if (kernels_.contains(params.executor)) return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::Kernel> kernel,
      CreateKernel(kernel_name_, args_.size(), params.src.text,
                   params.src.binary, params.executor, shmem_bytes_));

  absl::MutexLock lock(&mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::Status LaunchCmd::Record(const Thunk::ExecuteParams& execute_params,
                               const RecordParams& record_params,
                               se::CommandBuffer* command_buffer) {
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << "; shmem_bytes=" << shmem_bytes_;

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return kernels_[execute_params.stream->parent()].get();
  }();

  if (kernel == nullptr) {
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

  return command_buffer
      ->Launch(dims_.thread_counts_per_block(), dims_.block_counts(), *kernel,
               *kernel_args, {})
      .status();
}

CommandBufferCmd::BufferUseVector LaunchCmd::buffers() {
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
    ExecutionStreamId execution_stream_id,
    absl::Span<const BufferAllocation::Slice> args,
    absl::Span<const MemoryAccess> args_access, CustomKernel custom_kernel)
    : CommandBufferCmd(CommandBufferCmdType::kCustomKernelLaunchCmd,
                       execution_stream_id),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      custom_kernel_(std::move(custom_kernel)) {}

absl::Status CustomKernelLaunchCmd::Initialize(
    const Thunk::InitializeParams& params, StateManager& state) {
  {
    absl::MutexLock lock(&mutex_);
    if (kernels_.contains(params.executor)) return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::Kernel> kernel,
      params.executor->LoadKernel(custom_kernel_.kernel_spec()));

  absl::MutexLock lock(&mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::Status CustomKernelLaunchCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  VLOG(5) << "CustomKernelLaunchCmd: custom_kernel=" << custom_kernel_.name();

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return kernels_[execute_params.stream->parent()].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(
        absl::StrCat("Custom kernel not loaded on a command buffer executor: ",
                     custom_kernel_.name()));
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    buffers.push_back(buf);
  }

  se::KernelArgsDeviceMemoryArray kernel_args(
      buffers, custom_kernel_.shared_memory_bytes());

  return command_buffer
      ->Launch(custom_kernel_.thread_dims(), custom_kernel_.block_dims(),
               *kernel, kernel_args, {})
      .status();
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

MemcpyDeviceToDeviceCmd::MemcpyDeviceToDeviceCmd(
    ExecutionStreamId execution_stream_id, BufferAllocation::Slice dst,
    BufferAllocation::Slice src, int64_t num_bytes)
    : CommandBufferCmd(CommandBufferCmdType::kMemcpyDeviceToDeviceCmd,
                       execution_stream_id),
      dst_(dst),
      src_(src),
      num_bytes_(num_bytes) {}

absl::Status MemcpyDeviceToDeviceCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);
  se::DeviceMemoryBase src =
      execute_params.buffer_allocations->GetDeviceAddress(src_);

  VLOG(5) << "MemcpyDeviceToDeviceCmd: num_bytes = " << num_bytes_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << src_ << " (" << src.opaque() << ")";

  if (num_bytes_ == 0) {
    VLOG(5) << "Skip recording MemcpyDeviceToDeviceCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer->MemcpyDeviceToDevice(&dst, src, num_bytes_, {})
      .status();
}

CommandBufferCmd::BufferUseVector MemcpyDeviceToDeviceCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}, {src_, MemoryAccess::kRead}};
}

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

MemzeroCmd::MemzeroCmd(ExecutionStreamId execution_stream_id,
                       BufferAllocation::Slice dst)
    : CommandBufferCmd(CommandBufferCmdType::kMemzeroCmd, execution_stream_id),
      dst_(dst) {}

absl::Status MemzeroCmd::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "MemzeroCmd:";
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording MemzeroCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer
      ->Memset(&dst, uint8_t{0},
               /*num_elements=*/dst_.size(), {})
      .status();
}

CommandBufferCmd::BufferUseVector MemzeroCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

Memset32Cmd::Memset32Cmd(ExecutionStreamId execution_stream_id,
                         BufferAllocation::Slice dst, uint32_t bit_pattern)
    : CommandBufferCmd(CommandBufferCmdType::kMemset32Cmd, execution_stream_id),
      dst_(dst),
      bit_pattern_(bit_pattern) {}

absl::Status Memset32Cmd::Record(const Thunk::ExecuteParams& execute_params,
                                 const RecordParams& record_params,
                                 se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "Memset32Cmd: bit_pattern=" << bit_pattern_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording Memset32Cmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer
      ->Memset(&dst, bit_pattern_,
               /*num_elements=*/dst_.size() / sizeof(uint32_t), {})
      .status();
}

CommandBufferCmd::BufferUseVector Memset32Cmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

CaseCmd::CaseCmd(ExecutionStreamId execution_stream_id,
                 BufferAllocation::Slice index, bool index_is_bool,
                 std::vector<CommandBufferCmdSequence> branches_commands)
    : CommandBufferCmd(CommandBufferCmdType::kCaseCmd, execution_stream_id),
      index_(index),
      index_is_bool_(index_is_bool),
      branches_commands_(std::move(branches_commands)) {}

absl::Status CaseCmd::Initialize(const Thunk::InitializeParams& params,
                                 StateManager& state) {
  for (auto& branch : branches_commands_) {
    TF_RETURN_IF_ERROR(branch.Initialize(params, state));
  }
  return absl::OkStatus();
}

absl::Status CaseCmd::Record(const Thunk::ExecuteParams& execute_params,
                             const RecordParams& record_params,
                             se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase index =
      execute_params.buffer_allocations->GetDeviceAddress(index_);

  VLOG(5) << "CaseCmd:";
  VLOG(5) << "  index: " << index_ << " (" << index.opaque() << ")";

  if (index_is_bool_) {
    return command_buffer->Case(
        se::DeviceMemory<bool>(index),
        CreateBuilders(absl::MakeSpan(branches_commands_), &execute_params,
                       &record_params));
  } else {
    return command_buffer->Case(
        se::DeviceMemory<int32_t>(index),
        CreateBuilders(absl::MakeSpan(branches_commands_), &execute_params,
                       &record_params));
  }
}

bool CaseCmd::force_update() {
  return absl::c_any_of(branches_commands_,
                        [](const auto& seq) { return seq.force_update(); });
}

CommandBufferCmd::BufferUseVector CaseCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(index_, MemoryAccess::kRead);
  for (auto& branch : branches_commands_) {
    buffers.insert(branch.buffers().begin(), branch.buffers().end());
  }
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// ForCmd
//===----------------------------------------------------------------------===//

ForCmd::ForCmd(ExecutionStreamId execution_stream_id, int32_t num_iterations,
               BufferAllocation::Slice loop_counter,
               CommandBufferCmdSequence body_commands)
    : CommandBufferCmd(CommandBufferCmdType::kForCmd, execution_stream_id),
      num_iterations_(num_iterations),
      loop_counter_(loop_counter),
      body_commands_(std::move(body_commands)) {}

absl::Status ForCmd::Initialize(const Thunk::InitializeParams& params,
                                StateManager& state) {
  return body_commands_.Initialize(params, state);
}

absl::Status ForCmd::Record(const Thunk::ExecuteParams& execute_params,
                            const RecordParams& record_params,
                            se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase loop_counter =
      execute_params.buffer_allocations->GetDeviceAddress(loop_counter_);

  VLOG(5) << "ForCmd: num_iterations=" << num_iterations_
          << "; body_commands=" << body_commands_.size();
  VLOG(5) << "  loop_counter: " << loop_counter_ << " ("
          << loop_counter.opaque() << ")";

  return command_buffer->For(
      num_iterations_, se::DeviceMemory<int32_t>(loop_counter),
      CreateBuilder(&body_commands_, &execute_params, &record_params));
}

bool ForCmd::force_update() { return body_commands_.force_update(); }

CommandBufferCmd::BufferUseVector ForCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(loop_counter_, MemoryAccess::kWrite);
  buffers.insert(body_commands_.buffers().begin(),
                 body_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

WhileCmd::WhileCmd(ExecutionStreamId execution_stream_id,
                   BufferAllocation::Slice pred,
                   CommandBufferCmdSequence cond_commands,
                   CommandBufferCmdSequence body_commands)
    : CommandBufferCmd(CommandBufferCmdType::kWhileCmd, execution_stream_id),
      pred_(pred),
      cond_commands_(std::move(cond_commands)),
      body_commands_(std::move(body_commands)) {}

absl::Status WhileCmd::Initialize(const Thunk::InitializeParams& params,
                                  StateManager& state) {
  TF_RETURN_IF_ERROR(cond_commands_.Initialize(params, state));
  return body_commands_.Initialize(params, state);
}

absl::Status WhileCmd::Record(const Thunk::ExecuteParams& execute_params,
                              const RecordParams& record_params,
                              se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  VLOG(5) << "WhileCmd: cond_commands=" << cond_commands_.size()
          << " body_commands=" << body_commands_.size();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";

  return command_buffer->While(
      se::DeviceMemory<bool>(pred),
      CreateExecutionScopeBuilder(&cond_commands_, &execute_params,
                                  &record_params),
      CreateBuilder(&body_commands_, &execute_params, &record_params));
}

bool WhileCmd::force_update() {
  return (cond_commands_.force_update() || body_commands_.force_update());
}

CommandBufferCmd::BufferUseVector WhileCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(pred_, MemoryAccess::kWrite);
  buffers.insert(cond_commands_.buffers().begin(),
                 cond_commands_.buffers().end());
  buffers.insert(body_commands_.buffers().begin(),
                 body_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

GemmCmd::GemmCmd(ExecutionStreamId execution_stream_id, GemmConfig config,
                 const BufferAllocation::Slice& lhs_buffer,
                 const BufferAllocation::Slice& rhs_buffer,
                 const BufferAllocation::Slice& output_buffer,
                 const BufferAllocation::Slice& workspace, bool deterministic)
    : TracedCommandBufferCmd(CommandBufferCmdType::kGemmCmd,
                             execution_stream_id),
      config_(std::move(config)),
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
                             se::CommandBuffer* command_buffer) {
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

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
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
    ExecutionStreamId execution_stream_id, GemmConfig gemm_config,
    se::gpu::BlasLt::Epilogue epilogue, int64_t algorithm_idx,
    BufferAllocation::Slice a_buffer, BufferAllocation::Slice b_buffer,
    BufferAllocation::Slice c_buffer, BufferAllocation::Slice d_buffer,
    BufferAllocation::Slice bias_buffer /* may be null */,
    BufferAllocation::Slice aux_buffer /* may be null */,
    BufferAllocation::Slice a_scale_buffer /* may be null */,
    BufferAllocation::Slice b_scale_buffer /* may be null */,
    BufferAllocation::Slice c_scale_buffer /* may be null */,
    BufferAllocation::Slice d_scale_buffer /* may be null */,
    BufferAllocation::Slice d_amax_buffer /* may be null */,
    BufferAllocation::Slice workspace_buffer)
    : TracedCommandBufferCmd(CommandBufferCmdType::kCublasLtCmd,
                             execution_stream_id),
      gemm_config_(std::move(gemm_config)),
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
                                 se::CommandBuffer* command_buffer) {
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

  VLOG(5) << "CublasLtCmd:";
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

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
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
  BufferUseVector buffer_usage;
  buffer_usage.reserve(13);
  buffer_usage.push_back({a_buffer_, MemoryAccess::kRead});
  buffer_usage.push_back({b_buffer_, MemoryAccess::kRead});
  buffer_usage.push_back({c_buffer_, MemoryAccess::kRead});
  buffer_usage.push_back({d_buffer_, MemoryAccess::kWrite});
  buffer_usage.push_back({workspace_buffer_, MemoryAccess::kWrite});

  if (bias_buffer_.allocation() != nullptr) {
    buffer_usage.push_back({bias_buffer_, MemoryAccess::kRead});
  }
  if (a_scale_buffer_.allocation() != nullptr) {
    buffer_usage.push_back({a_scale_buffer_, MemoryAccess::kRead});
  }
  if (b_scale_buffer_.allocation() != nullptr) {
    buffer_usage.push_back({b_scale_buffer_, MemoryAccess::kRead});
  }
  if (c_scale_buffer_.allocation() != nullptr) {
    buffer_usage.push_back({c_scale_buffer_, MemoryAccess::kRead});
  }
  if (d_scale_buffer_.allocation() != nullptr) {
    buffer_usage.push_back({d_scale_buffer_, MemoryAccess::kRead});
  }
  if (aux_buffer_.allocation() != nullptr) {
    buffer_usage.push_back({aux_buffer_, MemoryAccess::kWrite});
  }
  if (d_amax_buffer_.allocation() != nullptr) {
    buffer_usage.push_back({d_amax_buffer_, MemoryAccess::kRead});
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CuDnnCmd
//===----------------------------------------------------------------------===//

CuDnnCmd::CuDnnCmd(ExecutionStreamId execution_stream_id,
                   absl::Span<const BufferAllocation::Slice> args,
                   const std::shared_ptr<se::dnn::LazyDnnGraph> graph)
    : TracedCommandBufferCmd(CommandBufferCmdType::kCuDnnCmd,
                             execution_stream_id),
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
                              se::CommandBuffer* command_buffer) {
  CHECK(graph_ != nullptr);
  std::vector<se::DeviceMemoryBase> operands;
  operands.reserve(args_.size());
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    operands.push_back(buf);
  }

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return graph_->get()->Execute(
            *stream, absl::Span<se::DeviceMemoryBase>(operands),
            execute_params.collective_params->local_device_ordinal);
      });
}

CommandBufferCmd::BufferUseVector CuDnnCmd::buffers() {
  CommandBufferCmd::BufferUseVector buffer_usage;
  buffer_usage.reserve(args_.size());
  for (int i = 0; i < args_.size() - 1; ++i) {
    buffer_usage.push_back({args_[i], MemoryAccess::kRead});
  }
  buffer_usage.push_back({args_.back(), MemoryAccess::kWrite});
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CustomCallCmd
//===----------------------------------------------------------------------===//

absl::Status CustomCallCmd::Record(const Thunk::ExecuteParams& execute_params,
                                   const RecordParams& record_params,
                                   se::CommandBuffer* command_buffer) {
  if (handler_ == nullptr) {
    return RecordLegacyCustomCall(execute_params, record_params,
                                  command_buffer);
  }
  return RecordXlaFfiCall(execute_params, record_params, command_buffer);
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
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
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

  return command_buffer->AddNestedCommandBuffer(*nested_cmd, {}).status();
}

absl::Status CustomCallCmd::RecordXlaFfiCall(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
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

  return command_buffer->AddNestedCommandBuffer(*nested_cmd, {}).status();
}

CommandBufferCmd::BufferUseVector CustomCallCmd::buffers() {
  CommandBufferCmd::BufferUseVector buffer_usage;
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<Slice>& slice : slices) {
      if (!slice.has_value()) continue;
      buffer_usage.push_back({slice->slice, MemoryAccess::kWrite});
    }
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

CollectiveCmd::CollectiveCmd(CommandBufferCmdType cmd_type,
                             ExecutionStreamId execution_stream_id,
                             ExecutionStreamId async_from_stream_id,
                             CollectiveConfig config)
    : CommandBufferCmd(cmd_type, execution_stream_id),
      async_from_stream_id_(async_from_stream_id),
      config_(std::move(config)) {}

absl::Status CollectiveCmd::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequestsInterface& resource_requests) {
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      Thunk::GetGpuCollectives(params));
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(collectives, *params.collective_params,
                      config().replica_groups, config().group_mode,
                      GetAsyncStreamKind()));
  return resource_requests.AddClique(clique_key);
}

absl::Status CollectiveCmd::AddTracedCommandBuffer(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::CommandBuffer> nested_cmd,
                      se::TraceCommandBufferFactory::Create(
                          execute_params.stream->parent(),
                          execute_params.command_buffer_trace_stream, trace));

  return command_buffer->AddNestedCommandBuffer(*nested_cmd, {}).status();
}

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

AllReduceCmd::AllReduceCmd(ExecutionStreamId execution_stream_id,
                           ExecutionStreamId async_from_stream_id,
                           CollectiveConfig config,
                           ReductionKind reduction_kind,
                           absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kAllReduceCmd, execution_stream_id,
                    async_from_stream_id, std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllReduceCmd::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

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
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return RunAllReduce(collectives, reduction_kind_, device_buffers,
                            *stream, comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector AllReduceCmd::buffers() {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

ReduceScatterCmd::ReduceScatterCmd(
    ExecutionStreamId execution_stream_id,
    ExecutionStreamId async_from_stream_id, CollectiveConfig config,
    ReductionKind reduction_kind,
    absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kReduceScatter, execution_stream_id,
                    async_from_stream_id, std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status ReduceScatterCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

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
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return RunReduceScatter(collectives, reduction_kind_, device_buffers,
                                *stream, comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector ReduceScatterCmd::buffers() {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllToAllCmd
//===----------------------------------------------------------------------===//

AllToAllCmd::AllToAllCmd(ExecutionStreamId execution_stream_id,
                         ExecutionStreamId async_from_stream_id,
                         CollectiveConfig config, bool has_split_dimension,
                         absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kAllToAll, execution_stream_id,
                    async_from_stream_id, std::move(config)),
      has_split_dimension_(has_split_dimension),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllToAllCmd::Record(const Thunk::ExecuteParams& execute_params,
                                 const RecordParams& record_params,
                                 se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  VLOG(5) << "AllToAllCmd, has_split_dimension=" << has_split_dimension_;

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
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return RunAllToAll(collectives, has_split_dimension_, device_buffers,
                           *stream, comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector AllToAllCmd::buffers() {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

AllGatherCmd::AllGatherCmd(ExecutionStreamId execution_stream_id,
                           ExecutionStreamId async_from_stream_id,
                           CollectiveConfig config,
                           absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kAllGatherCmd, execution_stream_id,
                    async_from_stream_id, std::move(config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllGatherCmd::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  VLOG(5) << "AllGatherCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
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
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return RunAllGather(collectives, device_buffers, *stream,
                            comm_handle.comm);
      });
}

CommandBufferCmd::BufferUseVector AllGatherCmd::buffers() {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

CollectiveBroadcastCmd::CollectiveBroadcastCmd(
    ExecutionStreamId execution_stream_id,
    ExecutionStreamId async_from_stream_id, CollectiveConfig config,
    absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandBufferCmdType::kCollectiveBroadcastCmd,
                    execution_stream_id, async_from_stream_id,
                    std::move(config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status CollectiveBroadcastCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  VLOG(5) << "CollectiveBroadcastCmd:";

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
              *execute_params.collective_cliques, config().replica_groups,
              config().group_mode, GetAsyncStreamKind()));

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return RunCollectiveBroadcast(device_buffers, *stream, comm_handle.comm,
                                      collectives);
      });
}

CommandBufferCmd::BufferUseVector CollectiveBroadcastCmd::buffers() {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

DynamicSliceFusionCmd::DynamicSliceFusionCmd(
    ExecutionStreamId execution_stream_id,
    std::unique_ptr<CommandBufferCmdSequence> embedded_commands,
    std::vector<std::optional<BufferAllocation::Slice>> arguments,
    std::vector<std::unique_ptr<BufferAllocation>> fake_allocations,
    std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets,
    std::vector<std::optional<Shape>> orig_shapes,
    std::vector<std::optional<Shape>> sliced_shapes,
    std::vector<std::optional<uint64_t>> offset_byte_sizes)
    : CommandBufferCmd(CommandBufferCmdType::kDynamicSliceFusionCmd,
                       execution_stream_id),
      embedded_commands_(std::move(embedded_commands)),
      fake_allocations_(std::move(fake_allocations)) {
  // Zip all arguments together to create a list of SliceDef.
  for (auto [arg, offset, orig_shape, sliced_shape, offset_byte_size] :
       llvm::zip_equal(arguments, offsets, orig_shapes, sliced_shapes,
                       offset_byte_sizes)) {
    slices_.push_back(DynamicSliceThunk::SliceDef{
        std::move(arg),
        std::move(offset),
        std::move(orig_shape),
        std::move(sliced_shape),
        std::move(offset_byte_size),
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
  absl::MutexLock lock(&mutex_);
  if (offsets_allocs_.contains(params.executor)) return absl::OkStatus();

  VLOG(2) << "Allocate " << offsets_allocs_size_
          << " bytes for transferring offsets on executor: " << params.executor;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::MemoryAllocation> allocation,
      params.executor->HostMemoryAllocate(offsets_allocs_size_));
  offsets_allocs_.emplace(params.executor, std::move(allocation));
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
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  se::Stream& stream = *execute_params.stream;

  const BufferAllocations& orig_allocations =
      *execute_params.buffer_allocations;
  absl::InlinedVector<se::DeviceMemoryBase, 8> slice_buffers(
      slices_.size(), se::DeviceMemoryBase());

  // Get memory allocation for copying offsets from device.
  int64_t* offsets_alloc = [&] {
    absl::MutexLock lock(&mutex_);
    return reinterpret_cast<int64_t*>(
        offsets_allocs_.at(stream.parent())->opaque());
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
  auto nested_command_buffer =
      execute_params.stream->parent()
          ->CreateCommandBuffer(se::CommandBuffer::Mode::kNested)
          .value();
  TF_RETURN_IF_ERROR(embedded_commands_->Record(new_params, record_params,
                                                nested_command_buffer.get()));
  return command_buffer->AddNestedCommandBuffer(*nested_command_buffer, {})
      .status();
}

CommandBufferCmd::BufferUseVector DynamicSliceFusionCmd::buffers() {
  CommandBufferCmd::BufferUseVector buffers;
  auto embed_buffers = embedded_commands_->buffers();
  for (auto buffer_usage : embed_buffers) {
    CHECK(
        embeded_to_origin_slice_map_[buffer_usage.slice().index()].has_value());
    buffers.emplace_back(
        embeded_to_origin_slice_map_[buffer_usage.slice().index()].value(),
        buffer_usage.access());
  }
  return buffers;
}

}  // namespace xla::gpu
