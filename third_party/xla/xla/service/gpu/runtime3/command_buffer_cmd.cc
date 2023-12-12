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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime3/command_buffer_allocations.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/status.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

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
  for (BufferAllocation::Slice& slice : cmd->slices()) {
    slices_.insert(slice);
    allocs_indices_.insert(slice.index());
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

  for (auto& cmd : commands_) {
    TF_RETURN_IF_ERROR(cmd->Record(params, command_buffer));
  }

  if (mode == RecordMode::kExclusive) {
    TF_RETURN_IF_ERROR(command_buffer->Finalize());
  }

  return OkStatus();
}

// Returns buffer allocation slices referenced by commands in this sequence.
const absl::flat_hash_set<BufferAllocation::Slice>&
CommandBufferCmdSequence::slices() const {
  return slices_;
}

// Returns buffer allocations indices referenced by commands in this sequence.
const absl::flat_hash_set<BufferAllocation::Index>&
CommandBufferCmdSequence::allocs_indices() const {
  return allocs_indices_;
}

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

LaunchCmd::LaunchCmd(std::string kernel_name,
                     absl::Span<const BufferAllocation::Slice> args,
                     LaunchDimensions dims, int64_t shmem_bytes)
    : kernel_name_(std::move(kernel_name)),
      args_(args.begin(), args.end()),
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

  LaunchDimensions::Dim3D thread_counts = dims_.thread_counts_per_block();
  LaunchDimensions::Dim3D block_counts = dims_.block_counts();

  return command_buffer->Launch(
      se::ThreadDim(thread_counts.x, thread_counts.y, thread_counts.z),
      se::BlockDim(block_counts.x, block_counts.y, block_counts.z), *kernel,
      *kernel_args);
}

CommandBufferCmd::Slices LaunchCmd::slices() {
  return CommandBufferCmd::Slices(args_.begin(), args_.end());
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

  VLOG(5) << "MemcpyDeviceToDeviceCmd: dst=" << dst_ << "("
          << reinterpret_cast<void*>(dst.opaque()) << ")"
          << ", src=" << src_ << "(" << reinterpret_cast<void*>(src.opaque())
          << ")"
          << ", num_bytes=" << num_bytes_;
  return command_buffer->MemcpyDeviceToDevice(&dst, src, num_bytes_);
}

CommandBufferCmd::Slices MemcpyDeviceToDeviceCmd::slices() {
  return {dst_, src_};
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

CommandBufferCmd::Slices IfCmd::slices() {
  absl::flat_hash_set<BufferAllocation::Slice> slices = {pred_};
  slices.insert(then_commands_.slices().begin(), then_commands_.slices().end());
  return {slices.begin(), slices.end()};
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

CommandBufferCmd::Slices IfElseCmd::slices() {
  absl::flat_hash_set<BufferAllocation::Slice> slices = {pred_};
  slices.insert(then_commands_.slices().begin(), then_commands_.slices().end());
  slices.insert(else_commands_.slices().begin(), else_commands_.slices().end());
  return {slices.begin(), slices.end()};
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

CommandBufferCmd::Slices CaseCmd::slices() {
  absl::flat_hash_set<BufferAllocation::Slice> slices = {index_};
  for (auto& branch : branches_commands_) {
    slices.insert(branch.slices().begin(), branch.slices().end());
  }
  return {slices.begin(), slices.end()};
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

CommandBufferCmd::Slices ForCmd::slices() {
  absl::flat_hash_set<BufferAllocation::Slice> slices = {loop_counter_};
  slices.insert(body_commands_.slices().begin(), body_commands_.slices().end());
  return {slices.begin(), slices.end()};
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

CommandBufferCmd::Slices WhileCmd::slices() {
  absl::flat_hash_set<BufferAllocation::Slice> slices = {pred_};
  slices.insert(cond_commands_.slices().begin(), cond_commands_.slices().end());
  slices.insert(body_commands_.slices().begin(), body_commands_.slices().end());
  return {slices.begin(), slices.end()};
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

CommandBufferCmd::Slices AllocateCmd::slices() { return {}; }

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

CommandBufferCmd::Slices FreeCmd::slices() { return {}; }

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

GemmCmd::GemmCmd(GemmConfig config, const BufferAllocation::Slice& lhs_buffer,
                 const BufferAllocation::Slice& rhs_buffer,
                 const BufferAllocation::Slice& output_buffer,
                 bool deterministic)
    : config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
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

  se::DeviceMemoryBase workspace(nullptr, 0);

  se::DeviceMemoryBase lhs =
      params.buffer_allocations->GetDeviceAddress(lhs_buffer_);

  se::DeviceMemoryBase rhs =
      params.buffer_allocations->GetDeviceAddress(rhs_buffer_);

  se::DeviceMemoryBase out =
      params.buffer_allocations->GetDeviceAddress(output_buffer_);

  TF_ASSIGN_OR_RETURN(
      auto nested_buffer,
      se::CommandBuffer::Trace(params.executor, [&](se::Stream* stream) {
        return RunGemm(config_, lhs, rhs, out, workspace, deterministic_,
                       stream);
      }));

  return command_buffer->AddNestedCommandBuffer(nested_buffer);
}

CommandBufferCmd::Slices GemmCmd::slices() {
  return {lhs_buffer_, rhs_buffer_, output_buffer_};
}

}  // namespace xla::gpu
