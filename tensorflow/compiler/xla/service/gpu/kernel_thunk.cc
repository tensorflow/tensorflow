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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/kernel.h"

namespace xla {
namespace gpu {

KernelThunk::KernelThunk(ThunkInfo thunk_info,
                         absl::Span<const BufferAllocation* const> args,
                         const std::string& kernel_name,
                         const LaunchDimensions& launch_dimensions)
    : Thunk(Kind::kKernel, thunk_info),
      args_(args.begin(), args.end()),
      kernel_name_(kernel_name),
      launch_dimensions_(launch_dimensions) {}

std::string KernelThunk::ToStringExtra(int indent) const {
  return " ,kernel = " + kernel_name_;
}

Status KernelThunk::Initialize(const GpuExecutable& executable,
                               se::StreamExecutor* executor) {
  tensorflow::mutex_lock lock(mutex_);

  // Load the kernel into the device if necessary.
  //
  // We could alternatively do this within ExecuteOnStream, but doing it here
  // lets the time spent loading the kernel not count towards our execution
  // profiles.
  auto it = kernel_cache_.find(executor);
  if (kernel_cache_.end() == it) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::KernelBase> kernel,
        CreateKernel(kernel_name_, args_.size(), executable.text(),
                     executable.binary(), executor));

    kernel_cache_.emplace(executor, std::move(kernel));
  }

  return Status::OK();
}

static void PrintBufferContents(
    se::Stream* stream, absl::Span<const se::DeviceMemoryBase> buffer_args) {
  int input_idx = 0;
  for (const se::DeviceMemoryBase& buf : buffer_args) {
    auto host_buffer = absl::make_unique<char[]>(buf.size());
    CHECK(stream->ThenMemcpy(host_buffer.get(), buf, buf.size()).ok());
    CHECK(stream->BlockHostUntilDone().ok());

    std::string buffer_contents;
    for (int i = 0; i < buf.size(); i++) {
      absl::StrAppendFormat(&buffer_contents, "%x ",
                            static_cast<unsigned>(host_buffer[i]));
    }
    VLOG(100) << "BUF(" << input_idx++ << ") = " << buffer_contents;
  }
}

Status KernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  // Load the kernel.
  se::StreamExecutor* executor = params.stream->parent();
  LaunchDimensions launch_dimensions;
  const se::KernelBase* kernel = nullptr;

  {
    tensorflow::mutex_lock lock(mutex_);
    auto it = kernel_cache_.find(executor);
    CHECK(it != kernel_cache_.end())
        << "Initialize() not called for StreamExecutor " << executor;
    launch_dimensions = launch_dimensions_;
    kernel = it->second.get();
  }

  VLOG(3) << "Launching " << kernel->name();
  absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
  for (const BufferAllocation* arg : args_) {
    se::DeviceMemoryBase buf =
        params.buffer_allocations->GetDeviceAddress(arg->index());
    VLOG(3) << "  Arg: alloc #" << arg->index() << ": " << buf.opaque() << "  ("
            << buf.size() << "B)";
    buffer_args.push_back(buf);
  }

  if (VLOG_IS_ON(100)) {
    PrintBufferContents(params.stream, buffer_args);
  }

  return ExecuteKernelOnStream(*kernel, buffer_args, launch_dimensions,
                               params.stream);
}

}  // namespace gpu
}  // namespace xla
