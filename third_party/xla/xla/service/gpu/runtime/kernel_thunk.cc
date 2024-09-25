/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/kernel_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

//===----------------------------------------------------------------------===//
// KernelThunk
//===----------------------------------------------------------------------===//

KernelThunk::KernelThunk(const HloInstruction* instr, std::string kernel_name,
                         absl::Span<const KernelArgument> kernel_arguments,
                         LaunchDimensions launch_dimensions,
                         std::optional<se::ClusterDim> cluster_dim,
                         int64_t shmem_bytes)
    : Thunk(Kind::kKernel, Thunk::ThunkInfo::WithProfileAnnotation(instr)),
      kernel_name_(std::move(kernel_name)),
      launch_dimensions_(std::move(launch_dimensions)),
      cluster_dim_(std::move(cluster_dim)),
      shmem_bytes_(shmem_bytes) {
  args_.reserve(kernel_arguments.size());
  written_.reserve(kernel_arguments.size());
  for (const auto& kernel_argument : kernel_arguments) {
    if (!kernel_argument.first_with_same_slice().has_value()) {
      args_.push_back(kernel_argument.slice());
      written_.push_back(kernel_argument.written());
    }
  }
}

std::string KernelThunk::ToString(int indent) const {
  return absl::StrFormat(
      ", kernel = %s, launch dimensions = %s, cluster_dim = %s", kernel_name_,
      launch_dimensions_.ToString(),
      cluster_dim_.has_value() ? cluster_dim_->ToString() : "nullopt");
}

absl::Status KernelThunk::Initialize(const InitializeParams& params) {
  absl::MutexLock lock(&mutex_);

  // Load the kernel into the device if necessary.
  //
  // We could alternatively do this within ExecuteOnStream, but doing it here
  // lets the time spent loading the kernel not count towards our execution
  // profiles.
  auto it = kernel_cache_.find(params.executor);
  if (kernel_cache_.end() == it) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::Kernel> kernel,
        CreateKernel(kernel_name_, args_.size(), params.src.text,
                     params.src.binary, params.executor, shmem_bytes_));

    kernel_cache_.emplace(params.executor, std::move(kernel));
  }

  return absl::OkStatus();
}

static void PrintBufferContents(
    se::Stream* stream, absl::Span<const se::DeviceMemoryBase> buffer_args) {
  int input_idx = 0;
  for (const se::DeviceMemoryBase& buf : buffer_args) {
    auto host_buffer = std::make_unique<char[]>(buf.size());
    CHECK_OK(stream->Memcpy(host_buffer.get(), buf, buf.size()));
    CHECK_OK(stream->BlockHostUntilDone());

    std::string buffer_contents;
    for (int i = 0; i < buf.size(); i++) {
      absl::StrAppendFormat(&buffer_contents, "%x ",
                            static_cast<unsigned>(host_buffer[i]));
    }
    VLOG(100) << "BUF(" << input_idx++ << ") = " << buffer_contents;
  }
}

absl::Status KernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  // Load the kernel.
  se::StreamExecutor* executor = params.stream->parent();
  LaunchDimensions launch_dimensions;
  std::optional<se::ClusterDim> cluster_dim;
  const se::Kernel* kernel = nullptr;

  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));

  {
    absl::MutexLock lock(&mutex_);
    auto it = kernel_cache_.find(executor);
    CHECK(it != kernel_cache_.end())
        << "Initialize() not called for StreamExecutor " << executor;
    launch_dimensions = launch_dimensions_;
    cluster_dim = cluster_dim_;
    kernel = it->second.get();
  }

  VLOG(3) << "Launching " << kernel->name();
  absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(3) << "  Arg: alloc #" << arg.index() << ", offset: " << arg.offset()
            << ": " << buf.opaque() << " (" << buf.size() << "B)";
    buffer_args.push_back(buf);
  }

  if (VLOG_IS_ON(100)) {
    PrintBufferContents(stream, buffer_args);
  }

  if (cluster_dim.has_value()) {
    return ExecuteKernelOnStream(*kernel, buffer_args, launch_dimensions,
                                 cluster_dim.value(), stream);
  } else {
    return ExecuteKernelOnStream(*kernel, buffer_args, launch_dimensions,
                                 stream);
  }
}

//===----------------------------------------------------------------------===//
// CustomKernelThunk
//===----------------------------------------------------------------------===//

CustomKernelThunk::CustomKernelThunk(
    const HloInstruction* instr, CustomKernel custom_kernel,
    absl::Span<const KernelArgument> kernel_arguments)
    : Thunk(Kind::kCustomKernel,
            Thunk::ThunkInfo::WithProfileAnnotation(instr)),
      custom_kernel_(std::move(custom_kernel)) {
  args_.reserve(kernel_arguments.size());
  written_.reserve(kernel_arguments.size());
  for (const auto& kernel_argument : kernel_arguments) {
    if (!kernel_argument.first_with_same_slice().has_value()) {
      args_.push_back(kernel_argument.slice());
      written_.push_back(kernel_argument.written());
    }
  }
}

std::string CustomKernelThunk::ToString(int indent) const {
  return custom_kernel_.ToString();
}

absl::Status CustomKernelThunk::Initialize(const InitializeParams& params) {
  absl::MutexLock lock(&mutex_);

  auto it = kernel_cache_.find(params.executor);
  if (kernel_cache_.end() == it) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::Kernel> kernel,
        params.executor->LoadKernel(custom_kernel_.kernel_spec()));
    kernel_cache_.emplace(params.executor, std::move(kernel));
  }

  return absl::OkStatus();
}

absl::Status CustomKernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();

  const se::Kernel* kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return kernel_cache_[executor].get();
  }();

  VLOG(3) << "Launching " << custom_kernel_.ToString() << " as device kernel "
          << kernel->name();

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(3) << "  Arg: alloc #" << arg.index() << ", offset: " << arg.offset()
            << ": " << buf.opaque() << " (" << buf.size() << "B)";
    buffer_args.push_back(buf);
  }

  if (VLOG_IS_ON(100)) {
    PrintBufferContents(params.stream, buffer_args);
  }

  se::KernelArgsDeviceMemoryArray args(buffer_args,
                                       custom_kernel_.shared_memory_bytes());

  if (auto cluster = custom_kernel_.cluster_dims(); cluster.has_value()) {
    return params.stream->Launch(custom_kernel_.thread_dims(),
                                 custom_kernel_.block_dims(), *cluster, *kernel,
                                 args);
  } else {
    return params.stream->Launch(custom_kernel_.thread_dims(),
                                 custom_kernel_.block_dims(), *kernel, args);
  }
}

}  // namespace gpu
}  // namespace xla
