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

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
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
  VLOG(5) << "MemcpyDeviceToDeviceCmd: dst=" << dst_ << ", src=" << src_
          << ", num_bytes=" << num_bytes_;
  se::DeviceMemoryBase dst = params.buffer_allocations->GetDeviceAddress(dst_);
  se::DeviceMemoryBase src = params.buffer_allocations->GetDeviceAddress(src_);
  return command_buffer->MemcpyDeviceToDevice(&dst, src, num_bytes_);
}

CommandBufferCmd::Slices MemcpyDeviceToDeviceCmd::slices() {
  return {dst_, src_};
}

//===----------------------------------------------------------------------===//
// IfCmd
//===----------------------------------------------------------------------===//

IfCmd::IfCmd(BufferAllocation::Slice pred, CommandBufferCmdSequence then_cmds)
    : pred_(pred), then_cmds_(std::move(then_cmds)) {}

Status IfCmd::Initialize(se::StreamExecutor* executor,
                         ExecutableSource source) {
  return then_cmds_.Initialize(executor, source);
}

Status IfCmd::Record(const RecordParams& params,
                     se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      params.buffer_allocations->GetDeviceAddress(pred_);

  return command_buffer->If(
      params.executor, se::DeviceMemory<bool>(pred),
      [&](se::CommandBuffer* then_cmd_buffer) {
        return then_cmds_.Record(
            params, then_cmd_buffer,
            CommandBufferCmdSequence::RecordMode::kConditional);
      });
}

CommandBufferCmd::Slices IfCmd::slices() {
  auto& slices = then_cmds_.slices();
  return {slices.begin(), slices.end()};
}

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
