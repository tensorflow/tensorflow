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

#include "xla/service/gpu/runtime3/command_buffer_cmd.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/nccl_all_gather_thunk.h"
#include "xla/service/gpu/nccl_all_reduce_thunk.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/status.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

#if XLA_ENABLE_XCCL
#include "xla/service/gpu/nccl_utils.h"
#endif  // XLA_ENABLE_XCCL

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
    const CommandBufferCmd::RecordParams* params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->Record(*params, command_buffer,
                            CommandBufferCmdSequence::RecordMode::kConditional);
  };
}

// Creates condition command buffer builders from a span of cmd sequences.
static std::vector<se::CommandBuffer::Builder> ConditionBuilders(
    absl::Span<CommandBufferCmdSequence> commands,
    const CommandBufferCmd::RecordParams* params) {
  std::vector<se::CommandBuffer::Builder> builders;
  for (CommandBufferCmdSequence& cmd : commands) {
    builders.push_back(ConditionBuilder(&cmd, params));
  }
  return builders;
}

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

CommandBufferCmdSequence::CommandBufferCmdSequence(bool force_barriers)
    : force_barriers_(force_barriers) {}

void CommandBufferCmdSequence::Append(std::unique_ptr<CommandBufferCmd> cmd) {
  for (const CommandBufferCmd::BufferUsage& buffer : cmd->buffers()) {
    buffers_.insert(buffer);
    allocs_indices_.insert(buffer.slice.index());
  }

  CommandBufferCmd::BufferUsageVector buffers = cmd->buffers();
  bool requires_barrier = HasConflicts(buffers);

  // Always add barriers between commands if we want to linearize execution.
  if (force_barriers_ && !commands_.empty()) {
    requires_barrier = true;
  }

  // If the first recorded command is implemented as a nested command buffer we
  // force a barrier before recording the next command as a workaround for CUDA
  // graph bug, where child CUDA graph must be a single CUDA graph root node.
  if (commands_.size() == 1 && commands_.front().cmd->IsNestedCommandBuffer()) {
    requires_barrier = true;
  }

  if (requires_barrier) ClearTrackedBuffers();

  commands_.emplace_back(std::move(cmd), requires_barrier);
  TrackBuffers(buffers);
}

Status CommandBufferCmdSequence::Initialize(
    se::StreamExecutor* executor, CommandBufferCmd::ExecutableSource source) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command.cmd->Initialize(executor, source));
  }
  return OkStatus();
}

bool CommandBufferCmdSequence::HasConflicts(
    const CommandBufferCmd::BufferUsageVector& buffers) {
  // Returns true if slice overlaps with any of the slices in read set.
  auto read_overlap = [&](const BufferAllocation::Slice& slice) {
    if (read_set_.contains(slice)) return true;
    for (auto& read : read_set_)
      if (read.OverlapsWith(slice)) return true;
    return false;
  };

  // Returns true if slice overlaps with any of the slices in write set.
  auto write_overlap = [&](const BufferAllocation::Slice& slice) {
    if (write_set_.contains(slice)) return true;
    for (auto& write : write_set_)
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
    const CommandBufferCmd::BufferUsageVector& buffers) {
  for (auto& buffer : buffers) {
    if (buffer.access == MemoryAccess::kWrite) write_set_.insert(buffer.slice);
    if (buffer.access == MemoryAccess::kRead) read_set_.insert(buffer.slice);
  }
}

void CommandBufferCmdSequence::ClearTrackedBuffers() {
  read_set_.clear();
  write_set_.clear();
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

Status CommandBufferCmdSequence::Record(
    const CommandBufferCmd::RecordParams& params,
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

  for (auto& command : commands_) {
    if (command.requires_barrier) {
      VLOG(3) << "Add command buffer barrier after " << num_recorded_commands
              << " recorded commands";
      TF_RETURN_IF_ERROR(command_buffer->Barrier(params.executor));
      num_recorded_commands = 0;
    }

    TF_RETURN_IF_ERROR(command.cmd->Record(params, command_buffer));
    ++num_recorded_commands;
  }

  if (mode == RecordMode::kExclusive) {
    TF_RETURN_IF_ERROR(command_buffer->Finalize());
  }

  uint64_t end_micros = tsl::Env::Default()->NowMicros();
  VLOG(3) << "Recorded " << commands_.size()
          << " commands into command buffer in " << (end_micros - start_micros)
          << " μs; mode=" << RecordModeString(mode);

  return OkStatus();
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
// LaunchCmd
//===----------------------------------------------------------------------===//

LaunchCmd::LaunchCmd(std::string kernel_name,
                     absl::Span<const BufferAllocation::Slice> args,
                     absl::Span<const MemoryAccess> args_access,
                     LaunchDimensions dims, int64_t shmem_bytes)
    : kernel_name_(std::move(kernel_name)),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      dims_(dims),
      shmem_bytes_(shmem_bytes) {}

Status LaunchCmd::Initialize(se::StreamExecutor* executor,
                             ExecutableSource source) {
  {
    absl::MutexLock lock(&mutex_);
    if (kernels_.contains(executor)) return OkStatus();
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                      CreateKernel(kernel_name_, args_.size(), source.text,
                                   source.binary, executor, shmem_bytes_));

  absl::MutexLock lock(&mutex_);
  kernels_.emplace(executor, std::move(kernel));
  return OkStatus();
}

Status LaunchCmd::Record(const RecordParams& params,
                         se::CommandBuffer* command_buffer) {
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << ", shmem_bytes=" << shmem_bytes_;

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return kernels_[params.executor].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Kernel not loaded on a command buffer executor: ", kernel_name_));
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
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
    absl::Span<const BufferAllocation::Slice> args,
    absl::Span<const MemoryAccess> args_access, CustomKernel custom_kernel)
    : args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      custom_kernel_(std::move(custom_kernel)) {}

Status CustomKernelLaunchCmd::Initialize(se::StreamExecutor* executor,
                                         ExecutableSource source) {
  {
    absl::MutexLock lock(&mutex_);
    if (kernels_.contains(executor)) return OkStatus();
  }

  auto kernel = std::make_unique<se::Kernel>(executor);
  TF_RETURN_IF_ERROR(
      executor->GetKernel(custom_kernel_.kernel_spec(), kernel.get()));

  absl::MutexLock lock(&mutex_);
  kernels_.emplace(executor, std::move(kernel));
  return OkStatus();
}

Status CustomKernelLaunchCmd::Record(const RecordParams& params,
                                     se::CommandBuffer* command_buffer) {
  VLOG(5) << "CustomKernelLaunchCmd: custom_kernel=" << custom_kernel_.name();

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return kernels_[params.executor].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(
        absl::StrCat("Custom kernel not loaded on a command buffer executor: ",
                     custom_kernel_.name()));
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
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

MemcpyDeviceToDeviceCmd::MemcpyDeviceToDeviceCmd(BufferAllocation::Slice dst,
                                                 BufferAllocation::Slice src,
                                                 int64_t num_bytes)
    : dst_(dst), src_(src), num_bytes_(num_bytes) {}

Status MemcpyDeviceToDeviceCmd::Record(const RecordParams& params,
                                       se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst = params.buffer_allocations->GetDeviceAddress(dst_);
  se::DeviceMemoryBase src = params.buffer_allocations->GetDeviceAddress(src_);

  VLOG(5) << "MemcpyDeviceToDeviceCmd: num_bytes = " << num_bytes_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << src_ << " (" << src.opaque() << ")";

  if (num_bytes_ == 0) {
    VLOG(5) << "Skip recording MemcpyDeviceToDeviceCmd command of 0 bytes";
    return OkStatus();
  }

  return command_buffer->MemcpyDeviceToDevice(&dst, src, num_bytes_);
}

CommandBufferCmd::BufferUsageVector MemcpyDeviceToDeviceCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}, {src_, MemoryAccess::kRead}};
}

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

MemzeroCmd::MemzeroCmd(BufferAllocation::Slice dst) : dst_(dst) {}

Status MemzeroCmd::Record(const RecordParams& params,
                          se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst = params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "MemzeroCmd:";
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording MemzeroCmd command of 0 bytes";
    return OkStatus();
  }

  return command_buffer->Memset(&dst, uint8_t{0}, /*num_elements=*/dst_.size());
}

CommandBufferCmd::BufferUsageVector MemzeroCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

Memset32Cmd::Memset32Cmd(BufferAllocation::Slice dst, uint32_t bit_pattern)
    : dst_(dst), bit_pattern_(bit_pattern) {}

Status Memset32Cmd::Record(const RecordParams& params,
                           se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst = params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "Memset32Cmd: bit_pattern=" << bit_pattern_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording Memset32Cmd command of 0 bytes";
    return OkStatus();
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

IfCmd::IfCmd(BufferAllocation::Slice pred,
             CommandBufferCmdSequence then_commands)
    : pred_(pred), then_commands_(std::move(then_commands)) {}

Status IfCmd::Initialize(se::StreamExecutor* executor,
                         ExecutableSource source) {
  return then_commands_.Initialize(executor, source);
}

Status IfCmd::Record(const RecordParams& params,
                     se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      params.buffer_allocations->GetDeviceAddress(pred_);

  return command_buffer->If(params.executor, se::DeviceMemory<bool>(pred),
                            ConditionBuilder(&then_commands_, &params));
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

IfElseCmd::IfElseCmd(BufferAllocation::Slice pred,
                     CommandBufferCmdSequence then_commands,
                     CommandBufferCmdSequence else_commands)
    : pred_(pred),
      then_commands_(std::move(then_commands)),
      else_commands_(std::move(else_commands)) {}

Status IfElseCmd::Initialize(se::StreamExecutor* executor,
                             ExecutableSource source) {
  TF_RETURN_IF_ERROR(then_commands_.Initialize(executor, source));
  TF_RETURN_IF_ERROR(else_commands_.Initialize(executor, source));
  return OkStatus();
}

Status IfElseCmd::Record(const RecordParams& params,
                         se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      params.buffer_allocations->GetDeviceAddress(pred_);

  return command_buffer->IfElse(params.executor, se::DeviceMemory<bool>(pred),
                                ConditionBuilder(&then_commands_, &params),
                                ConditionBuilder(&else_commands_, &params));
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

CaseCmd::CaseCmd(BufferAllocation::Slice index,
                 std::vector<CommandBufferCmdSequence> branches_commands)
    : index_(index), branches_commands_(std::move(branches_commands)) {}

Status CaseCmd::Initialize(se::StreamExecutor* executor,
                           ExecutableSource source) {
  for (auto& branch : branches_commands_) {
    TF_RETURN_IF_ERROR(branch.Initialize(executor, source));
  }
  return OkStatus();
}

Status CaseCmd::Record(const RecordParams& params,
                       se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase index =
      params.buffer_allocations->GetDeviceAddress(index_);

  return command_buffer->Case(
      params.executor, se::DeviceMemory<int32_t>(index),
      ConditionBuilders(absl::MakeSpan(branches_commands_), &params));
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

ForCmd::ForCmd(int32_t num_iterations, BufferAllocation::Slice loop_counter,
               CommandBufferCmdSequence body_commands)
    : num_iterations_(num_iterations),
      loop_counter_(loop_counter),
      body_commands_(std::move(body_commands)) {}

Status ForCmd::Initialize(se::StreamExecutor* executor,
                          ExecutableSource source) {
  return body_commands_.Initialize(executor, source);
}

Status ForCmd::Record(const RecordParams& params,
                      se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase loop_counter =
      params.buffer_allocations->GetDeviceAddress(loop_counter_);

  return command_buffer->For(params.executor, num_iterations_,
                             se::DeviceMemory<int32_t>(loop_counter),
                             ConditionBuilder(&body_commands_, &params));
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

WhileCmd::WhileCmd(BufferAllocation::Slice pred,
                   CommandBufferCmdSequence cond_commands,
                   CommandBufferCmdSequence body_commands)
    : pred_(pred),
      cond_commands_(std::move(cond_commands)),
      body_commands_(std::move(body_commands)) {}

Status WhileCmd::Initialize(se::StreamExecutor* executor,
                            ExecutableSource source) {
  TF_RETURN_IF_ERROR(cond_commands_.Initialize(executor, source));
  return body_commands_.Initialize(executor, source);
}

Status WhileCmd::Record(const RecordParams& params,
                        se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      params.buffer_allocations->GetDeviceAddress(pred_);

  return command_buffer->While(params.executor, se::DeviceMemory<bool>(pred),
                               ConditionBuilder(&cond_commands_, &params),
                               ConditionBuilder(&body_commands_, &params));
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

AllocateCmd::AllocateCmd(BufferAllocation allocation)
    : allocation_(allocation) {}

Status AllocateCmd::Record(const RecordParams& params,
                           se::CommandBuffer* command_buffer) {
  // Memory allocation address is returned on graph creation, and there is no
  // update operation
  VLOG(2) << "AllocationCmd: index=" << allocation_.index();

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                      command_buffer->Allocate(allocation_.size()));
  return params.buffer_allocations->AddExternalAllocation(allocation_.index(),
                                                          buffer);
}

CommandBufferCmd::BufferUsageVector AllocateCmd::buffers() { return {}; }

//===----------------------------------------------------------------------===//
// FreeCmd
//===----------------------------------------------------------------------===//

FreeCmd::FreeCmd(BufferAllocation allocation) : allocation_(allocation) {}

Status FreeCmd::Record(const RecordParams& params,
                       se::CommandBuffer* command_buffer) {
  VLOG(2) << "FreeCmd: index=" << allocation_.index();

  se::DeviceMemoryBase address =
      params.buffer_allocations->GetDeviceAddress(allocation_.index());

  // Free is in the same command buffer
  TF_RETURN_IF_ERROR(command_buffer->Free(address));

  // Remove the buffer from external allocations.
  return params.buffer_allocations->EraseExternalAllocation(
      allocation_.index());
}

CommandBufferCmd::BufferUsageVector FreeCmd::buffers() { return {}; }

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

GemmCmd::GemmCmd(GemmConfig config, const BufferAllocation::Slice& lhs_buffer,
                 const BufferAllocation::Slice& rhs_buffer,
                 const BufferAllocation::Slice& output_buffer,
                 const BufferAllocation::Slice& workspace, bool deterministic)
    : config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      workspace_(workspace),
      deterministic_(deterministic) {}

Status GemmCmd::Initialize(se::StreamExecutor* executor,
                           ExecutableSource source) {
  if (!executor->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support for GemmCmd");
  }
  return OkStatus();
}

Status GemmCmd::Record(const RecordParams& params,
                       se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase lhs =
      params.buffer_allocations->GetDeviceAddress(lhs_buffer_);
  se::DeviceMemoryBase rhs =
      params.buffer_allocations->GetDeviceAddress(rhs_buffer_);
  se::DeviceMemoryBase out =
      params.buffer_allocations->GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase workspace =
      params.buffer_allocations->GetDeviceAddress(workspace_);

  VLOG(5) << "GemmCmd: deterministic=" << deterministic_;
  VLOG(5) << "  Lhs: " << lhs_buffer_ << " (" << lhs.opaque() << ")";
  VLOG(5) << "  Lhs: " << rhs_buffer_ << " (" << rhs.opaque() << ")";
  VLOG(5) << "  Out: " << output_buffer_ << " (" << out.opaque() << ")";
  VLOG(5) << "  Workspace: " << workspace_ << " (" << workspace.opaque() << ")";

  TF_ASSIGN_OR_RETURN(
      auto nested_buffer,
      se::CommandBuffer::Trace(
          params.executor, params.trace_stream, [&](se::Stream* stream) {
            return RunGemm(config_, lhs, rhs, out, workspace, deterministic_,
                           stream);
          }));

  return command_buffer->AddNestedCommandBuffer(nested_buffer);
}

CommandBufferCmd::BufferUsageVector GemmCmd::buffers() {
  return {{lhs_buffer_, MemoryAccess::kRead},
          {rhs_buffer_, MemoryAccess::kRead},
          {output_buffer_, MemoryAccess::kWrite},
          {workspace_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

AllReduceCmd::AllReduceCmd(
    NcclCollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : config_(std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

Status AllReduceCmd::Record(const RecordParams& params,
                            se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params.buffer_allocations, buffers_,
                             config_.operand_element_type));

  VLOG(5) << "AllReduceCmd: reduction=" << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (params.nccl_params == nullptr) {
    return absl::InvalidArgumentError("AllReduceCmd requires nccl_params");
  }

#if XLA_ENABLE_XCCL
  // Today when recording collective operations into command buffers we always
  // use a sync mode and a stream id `0`, and enable clique optimization.
  TF_ASSIGN_OR_RETURN(
      NcclComm::Lock comm,
      LockNcclComm(*params.nccl_params, config_.replica_groups,
                   config_.group_mode, config_.op_id, /*stream_id=*/0,
                   /*enable_clique_optimization=*/true));

  TF_ASSIGN_OR_RETURN(
      auto nested_buffer,
      se::CommandBuffer::Trace(
          params.executor, params.trace_stream, [&](se::Stream* stream) {
            return RunAllReduce(reduction_kind_, device_buffers, *stream,
                                *comm);
          }));

  return command_buffer->AddNestedCommandBuffer(nested_buffer);
#else   // XLA_ENABLE_XCCL
  return absl::UnimplementedError(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
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
    NcclCollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : config_(std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

Status ReduceScatterCmd::Record(const RecordParams& params,
                                se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params.buffer_allocations, buffers_,
                             config_.operand_element_type));

  VLOG(5) << "ReduceScatterCmd: reduction="
          << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (params.nccl_params == nullptr) {
    return absl::InvalidArgumentError("ReduceScatterCmd requires nccl_params");
  }

#if XLA_ENABLE_XCCL
  // Today when recording collective operations into command buffers we always
  // use a sync mode and a stream id `0`, and enable clique optimization.
  TF_ASSIGN_OR_RETURN(
      NcclComm::Lock comm,
      LockNcclComm(*params.nccl_params, config_.replica_groups,
                   config_.group_mode, config_.op_id, /*stream_id=*/0,
                   /*enable_clique_optimization=*/true));

  TF_ASSIGN_OR_RETURN(
      auto nested_buffer,
      se::CommandBuffer::Trace(
          params.executor, params.trace_stream, [&](se::Stream* stream) {
            return RunReduceScatter(reduction_kind_, device_buffers, *stream,
                                    *comm);
          }));

  return command_buffer->AddNestedCommandBuffer(nested_buffer);
#else   // XLA_ENABLE_XCCL
  return absl::UnimplementedError(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
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
    NcclCollectiveConfig config,
    absl::Span<const NcclCollectiveThunk::Buffer> buffers)
    : config_(std::move(config)), buffers_(buffers.begin(), buffers.end()) {}

Status AllGatherCmd::Record(const RecordParams& params,
                            se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params.buffer_allocations, buffers_,
                             config_.operand_element_type));

  VLOG(5) << "AllGatherCmd";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    VLOG(5) << "  Src: " << buffers_[i].source_buffer << " ("
            << device_buffers[i].source_buffer.opaque() << ")";
    VLOG(5) << "  Dst: " << buffers_[i].destination_buffer << " ("
            << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (params.nccl_params == nullptr) {
    return absl::InvalidArgumentError("AllGatherCmd requires nccl_params");
  }

#if XLA_ENABLE_XCCL
  // Today when recording collective operations into command buffers we always
  // use a sync mode and a stream id `0`, and enable clique optimization.
  TF_ASSIGN_OR_RETURN(
      NcclComm::Lock comm,
      LockNcclComm(*params.nccl_params, config_.replica_groups,
                   config_.group_mode, config_.op_id, /*stream_id=*/0,
                   /*enable_clique_optimization=*/true));

  TF_ASSIGN_OR_RETURN(
      auto nested_buffer,
      se::CommandBuffer::Trace(
          params.executor, params.trace_stream, [&](se::Stream* stream) {
            return RunAllGather(device_buffers, *stream, *comm);
          }));

  return command_buffer->AddNestedCommandBuffer(nested_buffer);
#else   // XLA_ENABLE_XCCL
  return absl::UnimplementedError(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
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
