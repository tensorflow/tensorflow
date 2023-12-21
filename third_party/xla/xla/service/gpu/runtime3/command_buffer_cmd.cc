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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/status.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

using MemoryAccess = CommandBufferCmd::MemoryAccess;

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

void CommandBufferCmdSequence::Append(std::unique_ptr<CommandBufferCmd> cmd) {
  for (const CommandBufferCmd::BufferUsage& buffer : cmd->buffers()) {
    buffers_.insert(buffer);
    allocs_indices_.insert(buffer.slice.index());
  }
  commands_.push_back(std::move(cmd));
}

Status CommandBufferCmdSequence::Initialize(
    se::StreamExecutor* executor, CommandBufferCmd::ExecutableSource source) {
  for (auto& cmd : commands_) {
    TF_RETURN_IF_ERROR(cmd->Initialize(executor, source));
  }
  return OkStatus();
}

Status CommandBufferCmdSequence::Record(
    const CommandBufferCmd::RecordParams& params,
    se::CommandBuffer* command_buffer, RecordMode mode) {
  if (mode == RecordMode::kExclusive) {
    if (command_buffer->state() == se::CommandBuffer::State::kFinalized) {
      TF_RETURN_IF_ERROR(command_buffer->Update());
    }
  }

  // We track read and write sets of all commands recorded into the command
  // buffer to detect conflicts and insert explicit barriers. This is likely not
  // the most efficient algorithm to track buffer aliasing and read/write
  // conflicts, but XLA optimizes for peak memory allocation and we almost never
  // have a long chains of independent HLO operations writing into
  // non-overlapping buffer slices, so here we prefer simplicity.
  absl::flat_hash_set<BufferAllocation::Slice> read_set;
  absl::flat_hash_set<BufferAllocation::Slice> write_set;

  auto track_buffers = [&](const CommandBufferCmd::BufferUsageVector& buffers) {
    for (auto& buffer : buffers) {
      if (buffer.access == MemoryAccess::kWrite) write_set.insert(buffer.slice);
      if (buffer.access == MemoryAccess::kRead) read_set.insert(buffer.slice);
    }
  };

  // Returns true if slice overlaps with any of the slices in read set.
  auto read_overlap = [&](const BufferAllocation::Slice& slice) {
    if (read_set.contains(slice)) return true;
    for (auto& read : read_set)
      if (read.OverlapsWith(slice)) return true;
    return false;
  };

  // Returns true if slice overlaps with any of the slices in write set.
  auto write_overlap = [&](const BufferAllocation::Slice& slice) {
    if (write_set.contains(slice)) return true;
    for (auto& write : write_set)
      if (write.OverlapsWith(slice)) return true;
    return false;
  };

  auto has_conflict = [&](const CommandBufferCmd::BufferUsageVector& buffers) {
    bool conflict = absl::c_any_of(buffers, [&](const auto& buffer) {
      return buffer.access == MemoryAccess::kWrite
                 ? write_overlap(buffer.slice) || read_overlap(buffer.slice)
                 : write_overlap(buffer.slice);
    });
    if (conflict) {
      write_set.clear();
      read_set.clear();
    }
    return conflict;
  };

  for (auto& cmd : commands_) {
    CommandBufferCmd::BufferUsageVector buffers = cmd->buffers();
    if (has_conflict(buffers)) {
      TF_RETURN_IF_ERROR(command_buffer->Barrier(params.executor));
    }
    track_buffers(buffers);
    TF_RETURN_IF_ERROR(cmd->Record(params, command_buffer));
  }

  if (mode == RecordMode::kExclusive) {
    TF_RETURN_IF_ERROR(command_buffer->Finalize());
  }

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
  if (kernels_.contains(executor)) return OkStatus();

  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                      CreateKernel(kernel_name_, args_.size(), source.text,
                                   source.binary, executor, shmem_bytes_));

  kernels_.emplace(executor, std::move(kernel));
  return OkStatus();
}

Status LaunchCmd::Record(const RecordParams& params,
                         se::CommandBuffer* command_buffer) {
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << ", shmem_bytes=" << shmem_bytes_;

  se::Kernel* kernel = kernels_[params.executor].get();
  if (kernel == nullptr) {
    return absl::InternalError(
        "Kernel not loaded on a command buffer executor");
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
  if (kernels_.contains(executor)) {
    return OkStatus();
  }

  auto kernel = std::make_unique<se::Kernel>(executor);
  TF_RETURN_IF_ERROR(
      executor->GetKernel(custom_kernel_.kernel_spec(), kernel.get()));

  kernels_.emplace(executor, std::move(kernel));
  return OkStatus();
}

Status CustomKernelLaunchCmd::Record(const RecordParams& params,
                                     se::CommandBuffer* command_buffer) {
  VLOG(5) << "CustomKernelLaunchCmd: custom_kernel=" << custom_kernel_.name();

  se::Kernel* kernel = kernels_[params.executor].get();
  if (kernel == nullptr) {
    return absl::InternalError(
        "Kernel not loaded on a command buffer executor");
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

  VLOG(5) << "MemcpyDeviceToDeviceCmd: dst=" << dst_ << " (" << dst.opaque()
          << "), src=" << src_ << " (" << src.opaque()
          << "), num_bytes=" << num_bytes_;

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
  VLOG(5) << "GemmCmd: lhs=" << lhs_buffer_ << ", rhs=" << rhs_buffer_
          << ", output=" << output_buffer_
          << ", deterministic=" << deterministic_;

  se::DeviceMemoryBase lhs =
      params.buffer_allocations->GetDeviceAddress(lhs_buffer_);
  se::DeviceMemoryBase rhs =
      params.buffer_allocations->GetDeviceAddress(rhs_buffer_);
  se::DeviceMemoryBase out =
      params.buffer_allocations->GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase workspace =
      params.buffer_allocations->GetDeviceAddress(workspace_);

  TF_ASSIGN_OR_RETURN(
      auto nested_buffer,
      se::CommandBuffer::Trace(params.executor, [&](se::Stream* stream) {
        return RunGemm(config_, lhs, rhs, out, workspace, deterministic_,
                       stream);
      }));

  return command_buffer->AddNestedCommandBuffer(nested_buffer);
}

CommandBufferCmd::BufferUsageVector GemmCmd::buffers() {
  return {{lhs_buffer_, MemoryAccess::kRead},
          {rhs_buffer_, MemoryAccess::kRead},
          {output_buffer_, MemoryAccess::kWrite}};
}

}  // namespace xla::gpu
