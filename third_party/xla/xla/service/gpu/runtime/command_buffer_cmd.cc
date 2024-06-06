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

#include "xla/service/gpu/runtime/command_buffer_cmd.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/annotation.h"
#include "xla/service/gpu/runtime/nccl_all_gather_thunk.h"
#include "xla/service/gpu/runtime/nccl_all_reduce_thunk.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/gpu/runtime/nccl_collective_broadcast_thunk.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_factory.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#endif

namespace xla::gpu {

using ExecutionScopeId = se::CommandBuffer::ExecutionScopeId;
using MemoryAccess = CommandBufferCmd::MemoryAccess;

static std::string_view ReductionKindString(ReductionKind kind) {
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
static se::CommandBuffer::ExecutionScopeBuilder CreateExecutionScopeBuilder(
    CommandBufferCmdSequence* commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  return [=](ExecutionScopeId id, se::CommandBuffer* command_buffer) {
    CommandBufferCmd::RecordParams params = *record_params;
    params.execution_scope_id = id;
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

se::CommandBuffer::ExecutionScopeId CommandBufferCmd::GetExecutionScope(
    const RecordParams& record_params,
    ExecutionStreamId execution_stream_id) const {
  uint64_t base = record_params.execution_scope_id.value();
  uint64_t offset = execution_stream_id.value();
  return se::CommandBuffer::ExecutionScopeId(base + offset);
}

se::CommandBuffer::ExecutionScopeId CommandBufferCmd::GetExecutionScope(
    const RecordParams& record_params) const {
  return GetExecutionScope(record_params, execution_stream_id_);
}

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

CommandBufferCmdSequence::CommandBufferCmdSequence(
    SynchronizationMode synchronization_mode)
    : synchronization_mode_(synchronization_mode) {}

void CommandBufferCmdSequence::Append(std::unique_ptr<CommandBufferCmd> cmd) {
  for (const CommandBufferCmd::BufferUsage& buffer : cmd->buffers()) {
    buffers_.insert(buffer);
    allocs_indices_.insert(buffer.slice.index());
  }

  ExecutionStreamId execution_stream_id = cmd->execution_stream_id();
  CommandBufferCmd::BufferUsageVector buffers = cmd->buffers();
  bool requires_barrier = HasConflicts(execution_stream_id, buffers);

  // Always add barriers between commands if we want to serialize execution.
  if (synchronization_mode_ == SynchronizationMode::kSerialize &&
      !commands_.empty()) {
    requires_barrier = true;
  }

  // If the first recorded command is implemented as a nested command buffer we
  // force a barrier before recording the next command as a workaround for CUDA
  // graph bug, where child CUDA graph must be a single CUDA graph root node.
  if (commands_.size() == 1 && commands_.front().cmd->IsNestedCommandBuffer()) {
    requires_barrier = true;
  }

  if (requires_barrier) ClearTrackedBuffers(execution_stream_id);

  commands_.push_back({std::move(cmd), requires_barrier});
  TrackBuffers(execution_stream_id, buffers);
}

absl::Status CommandBufferCmdSequence::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequests& resource_requests) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command.cmd->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}

absl::Status CommandBufferCmdSequence::Initialize(
    const Thunk::InitializeParams& params,
    CommandBufferCmd::StateManager& state) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command.cmd->Initialize(params, state));
  }
  return absl::OkStatus();
}

bool CommandBufferCmdSequence::HasConflicts(
    ExecutionStreamId execution_stream_id,
    const CommandBufferCmd::BufferUsageVector& buffers) {
  auto& rwset = read_write_sets_[execution_stream_id];

  // Returns true if slice overlaps with any of the slices in read set.
  auto read_overlap = [&](const BufferAllocation::Slice& slice) {
    if (rwset.read.contains(slice)) return true;
    for (auto& read : rwset.read)
      if (read.OverlapsWith(slice)) return true;
    return false;
  };

  // Returns true if slice overlaps with any of the slices in write set.
  auto write_overlap = [&](const BufferAllocation::Slice& slice) {
    if (rwset.write.contains(slice)) return true;
    for (auto& write : rwset.write)
      if (write.OverlapsWith(slice)) return true;
    return false;
  };

  return absl::c_any_of(buffers, [&](const auto& buffer) {
    return buffer.access == MemoryAccess::kWrite
               ? write_overlap(buffer.slice) || read_overlap(buffer.slice)
               : write_overlap(buffer.slice);
  });
}

void CommandBufferCmdSequence::TrackBuffers(
    ExecutionStreamId execution_stream_id,
    const CommandBufferCmd::BufferUsageVector& buffers) {
  auto& rwset = read_write_sets_[execution_stream_id];
  for (const CommandBufferCmd::BufferUsage& buffer : buffers) {
    if (buffer.access == MemoryAccess::kWrite) rwset.write.insert(buffer.slice);
    if (buffer.access == MemoryAccess::kRead) rwset.read.insert(buffer.slice);
  }
}

void CommandBufferCmdSequence::ClearTrackedBuffers(
    ExecutionStreamId execution_stream_id) {
  read_write_sets_[execution_stream_id] = ReadWriteSet();
}

static std::string_view RecordModeString(
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

  const ModuleAnnotations* annotations = GetCurrentModuleAnnotations();

  // Track the number of commands recorded between barriers.
  absl::flat_hash_map<ExecutionScopeId, int64_t> num_recorded_commands;

  for (auto& command : commands_) {
    ExecutionScopeId execution_scope_id =
        command.cmd->GetExecutionScope(record_params);
    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(annotations, command.cmd->profile_annotation());

    if (command.requires_barrier) {
      VLOG(3) << "Add command buffer barrier after "
              << num_recorded_commands[execution_scope_id]
              << " recorded commands into the execution scope #"
              << execution_scope_id.value();
      TF_RETURN_IF_ERROR(command_buffer->Barrier(execution_scope_id));
      num_recorded_commands.erase(execution_scope_id);
    }
    VLOG(5) << " Record command buffer with scope id "
            << execution_scope_id.value();

    TF_RETURN_IF_ERROR(
        command.cmd->Record(execute_params, record_params, command_buffer));
    ++num_recorded_commands[execution_scope_id];
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

const absl::flat_hash_set<CommandBufferCmd::BufferUsage>&
CommandBufferCmdSequence::buffers() const {
  return buffers_;
}

const absl::flat_hash_set<BufferAllocation::Index>&
CommandBufferCmdSequence::allocs_indices() const {
  return allocs_indices_;
}

std::vector<bool> CommandBufferCmdSequence::barriers() const {
  std::vector<bool> barriers;
  absl::c_transform(commands_, std::back_inserter(barriers),
                    [](auto& command) { return command.requires_barrier; });
  return barriers;
}

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

TracedCommandBuffer::TracedCommandBuffer(
    CommandBufferCmd::BufferUsageVector buffers, int64_t capacity)
    : capacity_(capacity), entries_(capacity) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";  // NOLINT
  // Collect unique buffer allocation indices in a set first and convert to
  // vector as flat hash set iteration has measurable overheads.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices;
  for (auto& buffer : buffers) allocs_indices.insert(buffer.slice.index());
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
      return shift_right(i).command_buffer.get();
    }

    // Create a new entry by calling a user-provided tracing function, move it
    // to front and return a pointer to cached command buffer.
    if (entries_[i].command_buffer == nullptr) {
      TF_ASSIGN_OR_RETURN(
          entries_[i].command_buffer,
          se::TraceCommandBufferFactory::Create(executor, stream, trace));
      entries_[i].recorded_allocs.assign(allocs.begin(), allocs.end());
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
  return shift_right(capacity_ - 1).command_buffer.get();
}

//===----------------------------------------------------------------------===//
// TracedCommandBufferCmd
//===----------------------------------------------------------------------===//

TracedCommandBufferCmd::TracedCommandBufferCmd(
    ExecutionStreamId execution_stream_id)
    : CommandBufferCmd(execution_stream_id) {}

absl::Status TracedCommandBufferCmd::AddTracedCommandBuffer(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  auto traced_cmd = record_params.state.GetOrCreate<TracedCommandBuffer>(
      this, [&] { return std::make_unique<TracedCommandBuffer>(buffers()); });

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      traced_cmd->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace));

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "Add nested command buffer to execution scope: "
          << execution_scope_id.value();
  return command_buffer->AddNestedCommandBuffer(execution_scope_id,
                                                *nested_cmd);
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
inline constexpr std::string_view kMemset32Kernel = R"(
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
    : CommandBufferCmd(execution_stream_id), dest_(dest), kind_(kind) {}

CommandBufferCmd::BufferUsageVector ComputationIdCmd::buffers() {
  return {{dest_, MemoryAccess::kWrite}};
}

absl::Status ComputationIdCmd::Initialize(const Thunk::InitializeParams& params,
                                          StateManager& state) {
#if defined(GOOGLE_CUDA)
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
#endif  // GOOGLE_CUDA
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

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "ComputationIdCmd"
          << ": kind=" << (kind_ == Kind::kReplica ? "replica" : "partition")
          << "; value=" << value
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  Id: " << dest_ << " (" << dst.opaque() << ")";

#if defined(GOOGLE_CUDA)
  se::Kernel* memset_kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return memset_kernels_[execute_params.stream->parent()].get();
  }();

  if (memset_kernel == nullptr) {
    return absl::InternalError(
        "Memset kernel not loaded on a command buffer executor");
  }

  auto args = se::PackKernelArgs(/*shmem_bytes=*/0, int64_t{1}, value, dst);
  return command_buffer->Launch(execution_scope_id, se::ThreadDim(1),
                                se::BlockDim(1), *memset_kernel, *args);
#else
  return command_buffer->Memset(execution_scope_id, &dst, value,
                                /*num_elements=*/1);
#endif  // GOOGLE_CUDA
}

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

LaunchCmd::LaunchCmd(ExecutionStreamId execution_stream_id,
                     std::string kernel_name,
                     absl::Span<const BufferAllocation::Slice> args,
                     absl::Span<const MemoryAccess> args_access,
                     LaunchDimensions dims, int64_t shmem_bytes)
    : CommandBufferCmd(execution_stream_id),
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
  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << "; shmem_bytes=" << shmem_bytes_
          << "; execution_scope_id=" << execution_scope_id.value();

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

  return command_buffer->Launch(execution_scope_id,
                                dims_.thread_counts_per_block(),
                                dims_.block_counts(), *kernel, *kernel_args);
}

CommandBufferCmd::BufferUsageVector LaunchCmd::buffers() {
  BufferUsageVector buffers;
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
    : CommandBufferCmd(execution_stream_id),
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
      se::KernelFactory::Create(params.executor, custom_kernel_.kernel_spec()));

  absl::MutexLock lock(&mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::Status CustomKernelLaunchCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "CustomKernelLaunchCmd: custom_kernel=" << custom_kernel_.name()
          << "; execution_scope_id=" << execution_scope_id.value();

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

  return command_buffer->Launch(
      execution_scope_id, custom_kernel_.thread_dims(),
      custom_kernel_.block_dims(), *kernel, kernel_args);
}

CommandBufferCmd::BufferUsageVector CustomKernelLaunchCmd::buffers() {
  BufferUsageVector buffers;
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
    : CommandBufferCmd(execution_stream_id),
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

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "MemcpyDeviceToDeviceCmd: num_bytes = " << num_bytes_
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << src_ << " (" << src.opaque() << ")";

  if (num_bytes_ == 0) {
    VLOG(5) << "Skip recording MemcpyDeviceToDeviceCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer->MemcpyDeviceToDevice(execution_scope_id, &dst, src,
                                              num_bytes_);
}

CommandBufferCmd::BufferUsageVector MemcpyDeviceToDeviceCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}, {src_, MemoryAccess::kRead}};
}

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

MemzeroCmd::MemzeroCmd(ExecutionStreamId execution_stream_id,
                       BufferAllocation::Slice dst)
    : CommandBufferCmd(execution_stream_id), dst_(dst) {}

absl::Status MemzeroCmd::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "MemzeroCmd: execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording MemzeroCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer->Memset(execution_scope_id, &dst, uint8_t{0},
                                /*num_elements=*/dst_.size());
}

CommandBufferCmd::BufferUsageVector MemzeroCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

Memset32Cmd::Memset32Cmd(ExecutionStreamId execution_stream_id,
                         BufferAllocation::Slice dst, uint32_t bit_pattern)
    : CommandBufferCmd(execution_stream_id),
      dst_(dst),
      bit_pattern_(bit_pattern) {}

absl::Status Memset32Cmd::Record(const Thunk::ExecuteParams& execute_params,
                                 const RecordParams& record_params,
                                 se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "Memset32Cmd: bit_pattern=" << bit_pattern_
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording Memset32Cmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer->Memset(
      execution_scope_id, &dst, bit_pattern_,
      /*num_elements=*/dst_.size() / sizeof(uint32_t));
}

CommandBufferCmd::BufferUsageVector Memset32Cmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// IfCmd
//===----------------------------------------------------------------------===//

IfCmd::IfCmd(ExecutionStreamId execution_stream_id,
             BufferAllocation::Slice pred,
             CommandBufferCmdSequence then_commands)
    : CommandBufferCmd(execution_stream_id),
      pred_(pred),
      then_commands_(std::move(then_commands)) {}

absl::Status IfCmd::Initialize(const Thunk::InitializeParams& params,
                               StateManager& state) {
  return then_commands_.Initialize(params, state);
}

absl::Status IfCmd::Record(const Thunk::ExecuteParams& execute_params,
                           const RecordParams& record_params,
                           se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "IfCmd: execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";

  return command_buffer->If(
      execution_scope_id, se::DeviceMemory<bool>(pred),
      CreateBuilder(&then_commands_, &execute_params, &record_params));
}

bool IfCmd::force_update() { return then_commands_.force_update(); }

CommandBufferCmd::BufferUsageVector IfCmd::buffers() {
  absl::flat_hash_set<CommandBufferCmd::BufferUsage> buffers;
  buffers.emplace(pred_, MemoryAccess::kRead);
  buffers.insert(then_commands_.buffers().begin(),
                 then_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// IfElseCmd
//===----------------------------------------------------------------------===//

IfElseCmd::IfElseCmd(ExecutionStreamId execution_stream_id,
                     BufferAllocation::Slice pred,
                     CommandBufferCmdSequence then_commands,
                     CommandBufferCmdSequence else_commands)
    : CommandBufferCmd(execution_stream_id),
      pred_(pred),
      then_commands_(std::move(then_commands)),
      else_commands_(std::move(else_commands)) {}

absl::Status IfElseCmd::Initialize(const Thunk::InitializeParams& params,
                                   StateManager& state) {
  TF_RETURN_IF_ERROR(then_commands_.Initialize(params, state));
  TF_RETURN_IF_ERROR(else_commands_.Initialize(params, state));
  return absl::OkStatus();
}

absl::Status IfElseCmd::Record(const Thunk::ExecuteParams& execute_params,
                               const RecordParams& record_params,
                               se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "IfElseCmd: execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";

  return command_buffer->IfElse(
      execution_scope_id, se::DeviceMemory<bool>(pred),
      CreateBuilder(&then_commands_, &execute_params, &record_params),
      CreateBuilder(&else_commands_, &execute_params, &record_params));
}

bool IfElseCmd::force_update() {
  return (then_commands_.force_update() || else_commands_.force_update());
}

CommandBufferCmd::BufferUsageVector IfElseCmd::buffers() {
  absl::flat_hash_set<CommandBufferCmd::BufferUsage> buffers;
  buffers.emplace(pred_, MemoryAccess::kRead);
  buffers.insert(then_commands_.buffers().begin(),
                 then_commands_.buffers().end());
  buffers.insert(else_commands_.buffers().begin(),
                 else_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

CaseCmd::CaseCmd(ExecutionStreamId execution_stream_id,
                 BufferAllocation::Slice index,
                 std::vector<CommandBufferCmdSequence> branches_commands)
    : CommandBufferCmd(execution_stream_id),
      index_(index),
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

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "CaseCmd: execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  index: " << index_ << " (" << index.opaque() << ")";

  return command_buffer->Case(execution_scope_id,
                              se::DeviceMemory<int32_t>(index),
                              CreateBuilders(absl::MakeSpan(branches_commands_),
                                             &execute_params, &record_params));
}

bool CaseCmd::force_update() {
  return absl::c_any_of(branches_commands_,
                        [](const auto& seq) { return seq.force_update(); });
}

CommandBufferCmd::BufferUsageVector CaseCmd::buffers() {
  absl::flat_hash_set<CommandBufferCmd::BufferUsage> buffers;
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
    : CommandBufferCmd(execution_stream_id),
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

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "ForCmd: num_iterations=" << num_iterations_
          << "; body_commands=" << body_commands_.size()
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  loop_counter: " << loop_counter_ << " ("
          << loop_counter.opaque() << ")";

  return command_buffer->For(
      execution_scope_id, num_iterations_,
      se::DeviceMemory<int32_t>(loop_counter),
      CreateBuilder(&body_commands_, &execute_params, &record_params));
}

bool ForCmd::force_update() { return body_commands_.force_update(); }

CommandBufferCmd::BufferUsageVector ForCmd::buffers() {
  absl::flat_hash_set<CommandBufferCmd::BufferUsage> buffers;
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
    : CommandBufferCmd(execution_stream_id),
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

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "WhileCmd: cond_commands=" << cond_commands_.size()
          << " body_commands=" << body_commands_.size()
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";

  return command_buffer->While(
      execution_scope_id, se::DeviceMemory<bool>(pred),
      CreateExecutionScopeBuilder(&cond_commands_, &execute_params,
                                  &record_params),
      CreateBuilder(&body_commands_, &execute_params, &record_params));
}

bool WhileCmd::force_update() {
  return (cond_commands_.force_update() || body_commands_.force_update());
}

CommandBufferCmd::BufferUsageVector WhileCmd::buffers() {
  absl::flat_hash_set<CommandBufferCmd::BufferUsage> buffers;
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
    : TracedCommandBufferCmd(execution_stream_id),
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

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "GemmCmd: deterministic=" << deterministic_
          << "; execution_scope_id=" << execution_scope_id.value();
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

CommandBufferCmd::BufferUsageVector GemmCmd::buffers() {
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
    : TracedCommandBufferCmd(execution_stream_id),
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
    const stream_executor::Stream* stream) {
  auto it = matmul_plans_cache_.find(stream);
  if (it != matmul_plans_cache_.end()) return it->second.get();
  TF_ASSIGN_OR_RETURN(auto plan, se::gpu::BlasLt::GetMatmulPlan(
                                     stream, gemm_config_, epilogue_));
  auto [it_insert, _] = matmul_plans_cache_.emplace(stream, std::move(plan));
  return it_insert->second.get();
}

absl::StatusOr<se::gpu::BlasLt::MatmulAlgorithm>
CublasLtCmd::GetMatmulAlgorithm(const se::gpu::BlasLt::MatmulPlan* plan,
                                int64_t max_workspace) {
  auto it = matmul_algorithm_cache_.find(plan);
  if (it != matmul_algorithm_cache_.end()) return it->second;
  TF_ASSIGN_OR_RETURN(
      auto algorithms,
      plan->GetAlgorithms(/*max_algorithm_count*/ 128,
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
  TF_ASSIGN_OR_RETURN(plan_, GetMatmulPlan(params.stream));
  TF_ASSIGN_OR_RETURN(algorithm_,
                      GetMatmulAlgorithm(plan_, workspace_buffer_.size()));
  return absl::OkStatus();
}

absl::Status CublasLtCmd::Record(const Thunk::ExecuteParams& execute_params,
                                 const RecordParams& record_params,
                                 se::CommandBuffer* command_buffer) {
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

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);

  VLOG(5) << "CublasLtCmd with execution_scope_id: "
          << execution_scope_id.value();
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
        return plan_->ExecuteOnStream(
            stream, allocs.GetDeviceAddress(a_buffer_),
            allocs.GetDeviceAddress(b_buffer_),
            allocs.GetDeviceAddress(c_buffer_),
            allocs.GetDeviceAddress(d_buffer_), bias, aux, a_scale, b_scale,
            c_scale, d_scale, d_amax, algorithm_,
            allocs.GetDeviceAddress(workspace_buffer_));
      });
}

CommandBufferCmd::BufferUsageVector CublasLtCmd::buffers() {
  BufferUsageVector buffer_usage;
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
    : TracedCommandBufferCmd(execution_stream_id),
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
            *stream, absl::Span<se::DeviceMemoryBase>(operands));
      });
}

CommandBufferCmd::BufferUsageVector CuDnnCmd::buffers() {
  CommandBufferCmd::BufferUsageVector buffer_usage;
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

absl::Status CustomCallCmd::RecordLegacyCustomCall(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<Slice>& slice : slices) {
      if (!slice.has_value()) {
        buffers.push_back(nullptr);
        continue;
      }

      if (!slice->slice.allocation()) {
        return absl::InternalError(
            "custom call input missing buffer allocation");
      }

      buffers.push_back(
          execute_params.buffer_allocations->GetDeviceAddress(slice->slice)
              .opaque());
    }
  }

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "CustomCallCmd: execution_scope_id=" << execution_scope_id.value();
  for (int i = 0; i < operands_.size(); ++i) {
    if (operands_[i].has_value()) {
      VLOG(5) << "  Operand " << i << ": " << operands_[i]->slice << " ("
              << buffers[i] << ")";
    } else {
      VLOG(5) << "  Operand " << i << ": null";
    }
  }
  for (int i = 0; i < results_.size(); ++i) {
    if (results_[i].has_value()) {
      VLOG(5) << "  Result " << i << ": " << results_[i]->slice << " ("
              << buffers[operands_.size() + i] << ")";
    } else {
      VLOG(5) << "  Result " << i << ": null";
    }
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            se::gpu::GpuStreamHandle gpu_stream =
                se::gpu::AsGpuStreamValue(stream);
            XlaCustomCallStatus custom_call_status;
            call_target_(gpu_stream, buffers.data(), opaque_.data(),
                         opaque_.size(), &custom_call_status);
            auto message = CustomCallStatusGetMessage(&custom_call_status);
            if (message) {
              return absl::InternalError(
                  absl::StrCat("CustomCall failed: ", *message));
            }
            return absl::OkStatus();
          }));

  return command_buffer->AddNestedCommandBuffer(execution_scope_id,
                                                *nested_cmd);
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Custom calls on GPU are not supported in this configuration. Please "
      "build with --config=cuda or --config=rocm");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

absl::Status CustomCallCmd::RecordXlaFfiCall(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  // TODO(ezhulenev): This is not the most optimal approach, as we'll be doing
  // a lot of extra allocation on every call. We have to keep attributes
  // separate from arguments, as they do not change after thunk is constructed.
  ffi::CallFrameBuilder builder;

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "CustomCallCmd: execution_scope_id=" << execution_scope_id.value();

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
    builder.AddBufferArg(buffer, slice->shape.element_type(),
                         slice->shape.dimensions());
  }

  ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Append(attributes_);
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            ffi::CallOptions options = {
                execute_params.buffer_allocations->device_ordinal(),
                execute_params.stream,
                execute_params.buffer_allocations->memory_allocator(),
                /*called_computation=*/nullptr,  // TODO(b/342285364)
                execute_params.ffi_execution_context};
            return ffi::Call(handler_, call_frame, options);
          }));

  return command_buffer->AddNestedCommandBuffer(execution_scope_id,
                                                *nested_cmd);
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Custom calls on GPU are not supported in this configuration. Please "
      "build with --config=cuda or --config=rocm");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

CommandBufferCmd::BufferUsageVector CustomCallCmd::buffers() {
  CommandBufferCmd::BufferUsageVector buffer_usage;
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<Slice>& slice : slices) {
      if (!slice.has_value()) continue;
      buffer_usage.push_back({slice->slice, MemoryAccess::kWrite});
    }
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// BarrierCmd
//===----------------------------------------------------------------------===//

BarrierCmd::BarrierCmd(ExecutionStreamId execution_stream_id,
                       ExecutionStreamId from_stream_id)
    : CommandBufferCmd(execution_stream_id), from_stream_id_(from_stream_id) {}

absl::Status BarrierCmd::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                se::CommandBuffer* command_buffer) {
  VLOG(5) << "BarrierCmd from stream " << from_stream_id_.value()
          << " to stream " << execution_stream_id().value();
  if (from_stream_id_ != execution_stream_id()) {
    TF_RETURN_IF_ERROR(command_buffer->Barrier(
        CommandBufferCmd::GetExecutionScope(record_params, from_stream_id_),
        CommandBufferCmd::GetExecutionScope(record_params,
                                            execution_stream_id())));
  }
  return absl::OkStatus();
}

BarrierCmd::BufferUsageVector BarrierCmd::buffers() { return {}; }

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

CollectiveCmd::CollectiveCmd(ExecutionStreamId execution_stream_id,
                             ExecutionStreamId async_from_stream_id,
                             NcclApi* nccl_api, NcclCollectiveConfig config)
    : CommandBufferCmd(execution_stream_id),
      async_from_stream_id_(async_from_stream_id),
      nccl_api_(nccl_api),
      config_(std::move(config)) {}

absl::Status CollectiveCmd::BarrierIfAsync(
    se::CommandBuffer* command_buffer, se::StreamExecutor* executor,
    const CommandBufferCmd::RecordParams& record_params) {
  if (IsAsync()) {
    TF_RETURN_IF_ERROR(
        command_buffer->Barrier(CommandBufferCmd::GetExecutionScope(
                                    record_params, async_from_stream_id_),
                                CommandBufferCmd::GetExecutionScope(
                                    record_params, execution_stream_id())));
    VLOG(5) << "Insert Async barrier from stream "
            << async_from_stream_id_.value() << " to stream "
            << execution_stream_id().value();
  }
  return absl::OkStatus();
}

absl::Status CollectiveCmd::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequests& resource_requests) {
  const Thunk::CollectiveExecuteParams* collectives = params.collective_params;

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(collectives->global_device_id,
                              *collectives->device_assn,
                              config().replica_groups, config().group_mode));

  std::vector<GlobalDeviceId> local_devices;
  if (collectives->global_device_id_map) {
    local_devices.reserve(collectives->global_device_id_map->size());
    for (const auto& entry : *collectives->global_device_id_map) {
      local_devices.push_back(entry.second);
    }
  }

  size_t num_local_participants = GetNumLocalParticipants(
      participants,
      collectives->global_device_id_map ? &local_devices : nullptr);

  return resource_requests.AddClique(
      NcclCliqueKey(std::move(participants), nccl_stream_id(),
                    GetAsyncStreamKind()),
      num_local_participants);
}

absl::Status CollectiveCmd::AddTracedCommandBuffer(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::CommandBuffer> nested_cmd,
                      se::TraceCommandBufferFactory::Create(
                          execute_params.stream->parent(),
                          execute_params.command_buffer_trace_stream, trace));

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  return command_buffer->AddNestedCommandBuffer(execution_scope_id,
                                                *nested_cmd);
}

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

AllReduceCmd::AllReduceCmd(
    ExecutionStreamId execution_stream_id,
    ExecutionStreamId async_from_stream_id, NcclApi* nccl_api,
    NcclCollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : CollectiveCmd(execution_stream_id, async_from_stream_id, nccl_api,
                    std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllReduceCmd::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  se::CommandBuffer* command_buffer) {
  TF_RETURN_IF_ERROR(BarrierIfAsync(
      command_buffer, execute_params.stream->parent(), record_params));

  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "AllReduceCmd: reduction=" << ReductionKindString(reduction_kind_)
          << "; execution_scope_id=" << execution_scope_id.value();

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

  TF_ASSIGN_OR_RETURN(
      NcclCommHandleWrapper comm_handle,
      GetNcclComm(*execute_params.collective_params,
                  *execute_params.collective_cliques, config().replica_groups,
                  config().group_mode, nccl_stream_id(), GetAsyncStreamKind()));
  NcclApi::NcclCommHandle comm = comm_handle.comm_handle;
  // Use custom allocator for persistent execution plans.
  NcclApi::ScopedPersistentPlanAllocator scoped_allocator(
      comm, tsl::MakeRef<NcclApi::PersistentPlanAllocator>(
                execute_params.buffer_allocations->device_ordinal(),
                execute_params.buffer_allocations->memory_allocator(),
                execute_params.stream));

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return RunAllReduce(nccl_api(), reduction_kind_, device_buffers,
                            *stream, comm);
      });
}

CommandBufferCmd::BufferUsageVector AllReduceCmd::buffers() {
  BufferUsageVector buffer_usage;
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
    ExecutionStreamId async_from_stream_id, NcclApi* nccl_api,
    NcclCollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : CollectiveCmd(execution_stream_id, async_from_stream_id, nccl_api,
                    std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status ReduceScatterCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  TF_RETURN_IF_ERROR(BarrierIfAsync(
      command_buffer, execute_params.stream->parent(), record_params));

  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "ReduceScatterCmd: reduction="
          << ReductionKindString(reduction_kind_)
          << "; execution_scope_id=" << execution_scope_id.value();

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

  TF_ASSIGN_OR_RETURN(
      NcclCommHandleWrapper comm_handle,
      GetNcclComm(*execute_params.collective_params,
                  *execute_params.collective_cliques, config().replica_groups,
                  config().group_mode, nccl_stream_id(), GetAsyncStreamKind()));
  NcclApi::NcclCommHandle comm = comm_handle.comm_handle;
  // Use custom allocator for persistent execution plans.
  NcclApi::ScopedPersistentPlanAllocator scoped_allocator(
      comm, tsl::MakeRef<NcclApi::PersistentPlanAllocator>(
                execute_params.buffer_allocations->device_ordinal(),
                execute_params.buffer_allocations->memory_allocator(),
                execute_params.stream));

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return RunReduceScatter(nccl_api(), reduction_kind_, device_buffers,
                                *stream, comm);
      });
}

CommandBufferCmd::BufferUsageVector ReduceScatterCmd::buffers() {
  BufferUsageVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

AllGatherCmd::AllGatherCmd(
    ExecutionStreamId execution_stream_id,
    ExecutionStreamId async_from_stream_id, NcclApi* nccl_api,
    NcclCollectiveConfig config,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : CollectiveCmd(execution_stream_id, async_from_stream_id, nccl_api,
                    std::move(config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllGatherCmd::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  se::CommandBuffer* command_buffer) {
  TF_RETURN_IF_ERROR(BarrierIfAsync(
      command_buffer, execute_params.stream->parent(), record_params));

  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "AllGatherCmd: execution_scope_id=" << execution_scope_id.value();

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

  TF_ASSIGN_OR_RETURN(
      NcclCommHandleWrapper comm_handle,
      GetNcclComm(*execute_params.collective_params,
                  *execute_params.collective_cliques, config().replica_groups,
                  config().group_mode, nccl_stream_id(), GetAsyncStreamKind()));
  NcclApi::NcclCommHandle comm = comm_handle.comm_handle;
  // Use custom allocator for persistent execution plans.
  NcclApi::ScopedPersistentPlanAllocator scoped_allocator(
      comm, tsl::MakeRef<NcclApi::PersistentPlanAllocator>(
                execute_params.buffer_allocations->device_ordinal(),
                execute_params.buffer_allocations->memory_allocator(),
                execute_params.stream));

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return RunAllGather(nccl_api(), device_buffers, *stream, comm);
      });
}

CommandBufferCmd::BufferUsageVector AllGatherCmd::buffers() {
  BufferUsageVector buffer_usage;
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
    ExecutionStreamId async_from_stream_id, NcclApi* nccl_api,
    NcclCollectiveConfig config,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : CollectiveCmd(execution_stream_id, async_from_stream_id, nccl_api,
                    std::move(config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status CollectiveBroadcastCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  TF_RETURN_IF_ERROR(BarrierIfAsync(
      command_buffer, execute_params.stream->parent(), record_params));

  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "CollectiveBroadcastCmd: execution_scope_id="
          << execution_scope_id.value();

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

  TF_ASSIGN_OR_RETURN(
      NcclCommHandleWrapper comm_handle,
      GetNcclComm(*execute_params.collective_params,
                  *execute_params.collective_cliques, config().replica_groups,
                  config().group_mode, nccl_stream_id(), GetAsyncStreamKind()));
  NcclApi::NcclCommHandle comm = comm_handle.comm_handle;
  // Use custom allocator for persistent execution plans.
  NcclApi::ScopedPersistentPlanAllocator scoped_allocator(
      comm, tsl::MakeRef<NcclApi::PersistentPlanAllocator>(
                execute_params.buffer_allocations->device_ordinal(),
                execute_params.buffer_allocations->memory_allocator(),
                execute_params.stream));

  return AddTracedCommandBuffer(
      execute_params, record_params, command_buffer, [&](se::Stream* stream) {
        return RunCollectiveBroadcast(device_buffers, *stream, comm,
                                      nccl_api());
      });
}

CommandBufferCmd::BufferUsageVector CollectiveBroadcastCmd::buffers() {
  BufferUsageVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(buffer.source_buffer, MemoryAccess::kRead);
    buffer_usage.emplace_back(buffer.destination_buffer, MemoryAccess::kWrite);
  }
  return buffer_usage;
}

}  // namespace xla::gpu
