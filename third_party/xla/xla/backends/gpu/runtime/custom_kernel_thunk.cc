/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/print_buffer_contents.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

CustomKernelThunk::CustomKernelThunk(
    Thunk::ThunkInfo thunk_info, CustomKernel custom_kernel,
    const emitters::KernelArguments& kernel_arguments)
    : Thunk(Kind::kCustomKernel, std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentBufferSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()),
      custom_kernel_(std::move(custom_kernel)) {}

std::string CustomKernelThunk::ToString(int indent) const {
  return custom_kernel_.ToString();
}

absl::Status CustomKernelThunk::Initialize(const InitializeParams& params) {
  absl::MutexLock lock(mutex_);

  if (!kernel_cache_.contains(params.executor)) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::Kernel> kernel,
        params.executor->LoadKernel(custom_kernel_.kernel_spec()));
    kernel_cache_.emplace(params.executor, std::move(kernel));
  }

  return absl::OkStatus();
}

absl::Status CustomKernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(mutex_);
    return kernel_cache_[executor].get();
  }();

  int device_ordinal = executor->device_ordinal();
  VLOG(3) << "[" << device_ordinal << "] Launching "
          << custom_kernel_.ToString() << " as device kernel "
          << kernel->name();

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(3) << "[" << device_ordinal << "]  Arg: alloc #" << arg.index()
            << ", offset: " << arg.offset() << ": " << buf.opaque() << " ("
            << buf.size() << "B)";
    buffer_args.push_back(buf);
  }

  if (VLOG_IS_ON(100)) {
    absl::InlinedVector<se::KernelArgument, 4> kernel_args;
    for (const se::DeviceMemoryBase& arg : buffer_args) {
      kernel_args.push_back(arg);
    }
    PrintBufferContents(params.stream, kernel_args);
  }

  se::KernelArgsDeviceMemoryArray args(buffer_args,
                                       custom_kernel_.shared_memory_bytes());

  return kernel->Launch(custom_kernel_.thread_dims(),
                        custom_kernel_.block_dims(),
                        custom_kernel_.cluster_dims(), params.stream, args);
}

Thunk::BufferUses CustomKernelThunk::buffer_uses() const {
  Thunk::BufferUses buffers;
  buffers.reserve(args_.size());
  for (int i = 0; i < args_.size(); ++i) {
    // We assume that any buffer is either an input or an output of the
    // kernel, and inout buffers are represented as 2 separate arguments.
    if (written_[i]) {
      buffers.push_back(BufferUse::Write(args_[i]));
    } else {
      buffers.push_back(BufferUse::Read(args_[i]));
    }
  }
  return buffers;
}

}  // namespace gpu
}  // namespace xla
