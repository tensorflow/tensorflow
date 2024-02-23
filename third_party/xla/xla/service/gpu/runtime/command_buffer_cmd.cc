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
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/annotation.h"
#include "xla/service/gpu/runtime/nccl_all_gather_thunk.h"
#include "xla/service/gpu/runtime/nccl_all_reduce_thunk.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/thunk.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/concurrency/ref_count.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#endif

namespace xla::gpu {

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

// Creates condition command buffer builder from a cmd sequence.
static se::CommandBuffer::Builder ConditionBuilder(
    CommandBufferCmdSequence* commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->Record(*execute_params, *record_params, command_buffer,
                            CommandBufferCmdSequence::RecordMode::kConditional);
  };
}

// Creates condition command buffer builders from a span of cmd sequences.
static std::vector<se::CommandBuffer::Builder> ConditionBuilders(
    absl::Span<CommandBufferCmdSequence> commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  std::vector<se::CommandBuffer::Builder> builders;
  for (CommandBufferCmdSequence& cmd : commands) {
    builders.push_back(ConditionBuilder(&cmd, execute_params, record_params));
  }
  return builders;
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

  // Track the number of commands recorded between barriers.
  int64_t num_recorded_commands = 0;

  const ModuleAnnotations* annotations = GetCurrentModuleAnnotations();
  for (auto& command : commands_) {
    if (command.requires_barrier) {
      VLOG(3) << "Add command buffer barrier after " << num_recorded_commands
              << " recorded commands";
      TF_RETURN_IF_ERROR(
          command_buffer->Barrier(execute_params.stream->parent()));
      num_recorded_commands = 0;
    }
    auto annotation =
        GetKernelAnnotation(annotations, command.cmd->profile_annotation());
    TF_RETURN_IF_ERROR(
        command.cmd->Record(execute_params, record_params, command_buffer));
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
      TF_ASSIGN_OR_RETURN(entries_[i].command_buffer,
                          se::CommandBuffer::Trace(executor, stream, trace));
      entries_[i].recorded_allocs.assign(allocs.begin(), allocs.end());
      return shift_right(i).command_buffer.get();
    }
  }

  // Create a new entry by calling a user-provided tracing function, replace the
  // last entry with it, move it to front and return a pointer to cached command
  // buffer.
  TF_ASSIGN_OR_RETURN(entries_[capacity_ - 1].command_buffer,
                      se::CommandBuffer::Trace(executor, stream, trace));
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

  return command_buffer->AddNestedCommandBuffer(*nested_cmd);
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

  se::Kernel* memset_kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return memset_kernels_[execute_params.stream->parent()].get();
  }();

  if (memset_kernel == nullptr) {
    return absl::InternalError(
        "Memset kernel not loaded on a command buffer executor");
  }

  auto args = se::PackKernelArgs(/*shmem_bytes=*/0, int64_t{1}, value, dst);
  return command_buffer->Launch(se::ThreadDim(1), se::BlockDim(1),
                                *memset_kernel, *args);
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
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << ", shmem_bytes=" << shmem_bytes_;

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

  return command_buffer->Launch(dims_.thread_counts_per_block(),
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
      se::Kernel::Create(params.executor, custom_kernel_.kernel_spec()));

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

  return command_buffer->Launch(custom_kernel_.thread_dims(),
                                custom_kernel_.block_dims(), *kernel,
                                kernel_args);
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

  VLOG(5) << "MemcpyDeviceToDeviceCmd: num_bytes = " << num_bytes_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << src_ << " (" << src.opaque() << ")";

  if (num_bytes_ == 0) {
    VLOG(5) << "Skip recording MemcpyDeviceToDeviceCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer->MemcpyDeviceToDevice(&dst, src, num_bytes_);
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

  VLOG(5) << "MemzeroCmd:";
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording MemzeroCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer->Memset(&dst, uint8_t{0}, /*num_elements=*/dst_.size());
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

  VLOG(5) << "Memset32Cmd: bit_pattern=" << bit_pattern_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording Memset32Cmd command of 0 bytes";
    return absl::OkStatus();
  }

  size_t num_elements = dst_.size() / sizeof(uint32_t);
  return command_buffer->Memset(&dst, bit_pattern_, num_elements);
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

  return command_buffer->If(
      execute_params.stream->parent(), se::DeviceMemory<bool>(pred),
      ConditionBuilder(&then_commands_, &execute_params, &record_params));
}

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

  return command_buffer->IfElse(
      execute_params.stream->parent(), se::DeviceMemory<bool>(pred),
      ConditionBuilder(&then_commands_, &execute_params, &record_params),
      ConditionBuilder(&else_commands_, &execute_params, &record_params));
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

  return command_buffer->Case(
      execute_params.stream->parent(), se::DeviceMemory<int32_t>(index),
      ConditionBuilders(absl::MakeSpan(branches_commands_), &execute_params,
                        &record_params));
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

  VLOG(5) << "ForCmd: num_iterations=" << num_iterations_
          << "; body_commands=" << body_commands_.size();
  VLOG(5) << "  loop_counter: " << loop_counter_ << " ("
          << loop_counter.opaque() << ")";

  return command_buffer->For(
      execute_params.stream->parent(), num_iterations_,
      se::DeviceMemory<int32_t>(loop_counter),
      ConditionBuilder(&body_commands_, &execute_params, &record_params));
}

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

  VLOG(5) << "WhileCmd: cond_commands=" << cond_commands_.size()
          << " body_commands=" << body_commands_.size();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";

  return command_buffer->While(
      execute_params.stream->parent(), se::DeviceMemory<bool>(pred),
      ConditionBuilder(&cond_commands_, &execute_params, &record_params),
      ConditionBuilder(&body_commands_, &execute_params, &record_params));
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
// AllocateCmd
//===----------------------------------------------------------------------===//

AllocateCmd::AllocateCmd(ExecutionStreamId execution_stream_id,
                         BufferAllocation allocation)
    : CommandBufferCmd(execution_stream_id), allocation_(allocation) {}

absl::Status AllocateCmd::Record(const Thunk::ExecuteParams& execute_params,
                                 const RecordParams& record_params,
                                 se::CommandBuffer* command_buffer) {
  // Memory allocation address is returned on graph creation, and there is no
  // update operation
  VLOG(2) << "AllocationCmd: index=" << allocation_.index();

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                      command_buffer->Allocate(allocation_.size()));
  return execute_params.buffer_allocations->AddExternalAllocation(
      allocation_.index(), buffer);
}

CommandBufferCmd::BufferUsageVector AllocateCmd::buffers() { return {}; }

//===----------------------------------------------------------------------===//
// FreeCmd
//===----------------------------------------------------------------------===//

FreeCmd::FreeCmd(ExecutionStreamId execution_stream_id,
                 BufferAllocation allocation)
    : CommandBufferCmd(execution_stream_id), allocation_(allocation) {}

absl::Status FreeCmd::Record(const Thunk::ExecuteParams& execute_params,
                             const RecordParams& record_params,
                             se::CommandBuffer* command_buffer) {
  VLOG(2) << "FreeCmd: index=" << allocation_.index();

  se::DeviceMemoryBase address =
      execute_params.buffer_allocations->GetDeviceAddress(allocation_.index());

  // Free is in the same command buffer
  TF_RETURN_IF_ERROR(command_buffer->Free(address));

  // Remove the buffer from external allocations.
  return execute_params.buffer_allocations->EraseExternalAllocation(
      allocation_.index());
}

CommandBufferCmd::BufferUsageVector FreeCmd::buffers() { return {}; }

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

CommandBufferCmd::BufferUsageVector GemmCmd::buffers() {
  return {{lhs_buffer_, MemoryAccess::kRead},
          {rhs_buffer_, MemoryAccess::kRead},
          {output_buffer_, MemoryAccess::kWrite},
          {workspace_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// CustomCallCmd
//===----------------------------------------------------------------------===//

absl::Status CustomCallCmd::Record(const Thunk::ExecuteParams& execute_params,
                                   const RecordParams& record_params,
                                   se::CommandBuffer* command_buffer) {
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

  VLOG(5) << "CustomCallCmd: ";
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
      se::CommandBuffer::Trace(
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
  return command_buffer->AddNestedCommandBuffer(*nested_cmd);
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
// CollectiveCmd
//===----------------------------------------------------------------------===//

CollectiveCmd::CollectiveCmd(ExecutionStreamId execution_stream_id,
                             NcclApi* nccl_api, NcclCollectiveConfig config)
    : TracedCommandBufferCmd(execution_stream_id),
      nccl_api_(nccl_api),
      config_(std::move(config)) {}

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
      NcclCliqueKey(std::move(participants), /*stream_id=*/0,
                    GetAsyncStreamKind()),
      num_local_participants);
}

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

AllReduceCmd::AllReduceCmd(
    ExecutionStreamId execution_stream_id, NcclApi* nccl_api,
    NcclCollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : CollectiveCmd(execution_stream_id, nccl_api, std::move(config)),
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

  // Today when recording collective operations into command buffers we always
  // use a sync mode and a stream id `0`.
  TF_ASSIGN_OR_RETURN(NcclApi::NcclCommHandle comm,
                      GetNcclComm(*execute_params.collective_params,
                                  *execute_params.collective_cliques,
                                  config().replica_groups, config().group_mode,
                                  /*stream_id=*/0, GetAsyncStreamKind()));

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
    ExecutionStreamId execution_stream_id, NcclApi* nccl_api,
    NcclCollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : CollectiveCmd(execution_stream_id, nccl_api, std::move(config)),
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

  // Today when recording collective operations into command buffers we always
  // use a sync mode and a stream id `0`.
  TF_ASSIGN_OR_RETURN(NcclApi::NcclCommHandle comm,
                      GetNcclComm(*execute_params.collective_params,
                                  *execute_params.collective_cliques,
                                  config().replica_groups, config().group_mode,
                                  /*stream_id=*/0, GetAsyncStreamKind()));

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
    ExecutionStreamId execution_stream_id, NcclApi* nccl_api,
    NcclCollectiveConfig config,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : CollectiveCmd(execution_stream_id, nccl_api, std::move(config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllGatherCmd::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  VLOG(5) << "AllGatherCmd";

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

  // Today when recording collective operations into command buffers we always
  // use a sync mode and a stream id `0`.
  TF_ASSIGN_OR_RETURN(NcclApi::NcclCommHandle comm,
                      GetNcclComm(*execute_params.collective_params,
                                  *execute_params.collective_cliques,
                                  config().replica_groups, config().group_mode,
                                  /*stream_id=*/0, GetAsyncStreamKind()));

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

}  // namespace xla::gpu
