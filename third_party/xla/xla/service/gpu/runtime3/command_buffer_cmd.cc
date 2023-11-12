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

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/status.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

void CommandBufferCmdSequence::Append(std::unique_ptr<CommandBufferCmd> cmd) {
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
    se::CommandBuffer* command_buffer) {
  if (command_buffer->state() == se::CommandBuffer::State::kFinalized) {
    TF_RETURN_IF_ERROR(command_buffer->Update());
  }
  for (auto& cmd : commands_) {
    TF_RETURN_IF_ERROR(cmd->Record(params, command_buffer));
  }
  return command_buffer->Finalize();
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

  se::Kernel* kernel = kernels_[command_buffer->executor()].get();
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

  const BufferAllocations& allocs = *params.buffer_allocations;
  se::DeviceMemoryBase workspace(nullptr, 0);

  TF_ASSIGN_OR_RETURN(
      auto nested_buffer,
      stream_executor::CommandBuffer::Trace(
          command_buffer->executor(), [&](stream_executor::Stream* stream) {
            return RunGemm(config_, allocs.GetDeviceAddress(lhs_buffer_),
                           allocs.GetDeviceAddress(rhs_buffer_),
                           allocs.GetDeviceAddress(output_buffer_), workspace,
                           deterministic_, stream);
          }));
  return command_buffer->AddNestedCommandBuffer(nested_buffer);
}

}  // namespace xla::gpu
