/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

KernelThunk::KernelThunk(
    tensorflow::gtl::ArraySlice<BufferAllocation::Slice> io_buffers,
    const string& kernel_name, const HloInstruction* hlo_instruction)
    : Thunk(Kind::kKernel, hlo_instruction),
      io_buffers_(io_buffers.begin(), io_buffers.end()),
      kernel_name_(kernel_name) {}

tensorflow::Status KernelThunk::Initialize(const GpuExecutable& executable) {
  tensorflow::mutex_lock lock(mutex_);
  if (loader_spec_) {
    // Already initialized by another thread.
    return tensorflow::Status::OK();
  }

  loader_spec_.reset(new se::MultiKernelLoaderSpec(io_buffers_.size() + 1));
  tensorflow::StringPiece ptx = executable.ptx();
  // Convert tensorflow::StringPiece to se::port::StringPiece because
  // StreamExecutor uses the latter.
  loader_spec_->AddCudaPtxInMemory(
      se::port::StringPiece(ptx.data(), ptx.size()), kernel_name_);
  return tensorflow::Status::OK();
}

void KernelThunk::SetLaunchDimensions(const LaunchDimensions& launch_dims) {
  tensorflow::mutex_lock lock(mutex_);
  launch_dimensions_ = launch_dims;
}

tensorflow::Status KernelThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  // Load the kernel.
  se::StreamExecutor* executor = stream->parent();
  LaunchDimensions launch_dimensions;
  const se::KernelBase* kernel = nullptr;
  {
    tensorflow::mutex_lock lock(mutex_);
    auto it = kernel_cache_.find(executor);
    if (kernel_cache_.end() == it) {
      it = kernel_cache_.emplace(executor, se::KernelBase(executor)).first;
      if (!executor->GetKernel(*loader_spec_, &it->second)) {
        return InternalError("Unable to load kernel %s", kernel_name_.c_str());
      }
    }
    launch_dimensions = launch_dimensions_;
    kernel = &it->second;
  }

  // Launch the kernel with potentially multiple blocks and threads.
  static constexpr int kKernelArgsLimit = 1024;
  auto kernel_args = MakeUnique<se::KernelArgsArray<kKernelArgsLimit>>();
  for (const BufferAllocation::Slice io_buffer : io_buffers_) {
    kernel_args->add_device_memory_argument(
        buffer_allocations.GetDeviceAddress(io_buffer));
  }
  kernel_args->add_device_memory_argument(
      buffer_allocations.GetTempBufferBase());
  if (!stream->parent()->Launch(
          stream, se::ThreadDim(launch_dimensions.threads_per_block()),
          se::BlockDim(launch_dimensions.block_count()), *kernel,
          *kernel_args)) {
    return InternalError("Unable to launch kernel %s", kernel_name_.c_str());
  }
  return tensorflow::Status::OK();
}

}  // namespace gpu
}  // namespace xla
