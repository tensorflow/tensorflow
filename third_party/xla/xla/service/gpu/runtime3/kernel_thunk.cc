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

#include "xla/service/gpu/runtime3/kernel_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

//===----------------------------------------------------------------------===//
// KernelThunk
//===----------------------------------------------------------------------===//

mlir::Value RemoveTransformingOperations(mlir::Value value) {
  mlir::Operation* defining_op = value.getDefiningOp();
  if (auto cast_op = llvm::isa<mlir::memref::ReinterpretCastOp,
                               mlir::memref::CollapseShapeOp>(defining_op)) {
    return defining_op->getOperand(0);
  }
  return value;
}

}  // namespace

KernelThunk::KernelThunk(
    std::variant<mlir::Operation*, const HloInstruction*> op,
    std::string kernel_name, absl::Span<const KernelArgument> kernel_arguments,
    LaunchDimensions launch_dimensions, int64_t shmem_bytes)
    : Thunk(Kind::kKernel, std::holds_alternative<mlir::Operation*>(op)
                               ? Thunk::ThunkInfo::WithProfileAnnotation(
                                     std::get<mlir::Operation*>(op))
                               : Thunk::ThunkInfo::WithProfileAnnotation(
                                     std::get<const HloInstruction*>(op))),
      kernel_name_(std::move(kernel_name)),
      launch_dimensions_(std::move(launch_dimensions)),
      shmem_bytes_(shmem_bytes) {
  args_.reserve(kernel_arguments.size());
  written_.reserve(kernel_arguments.size());
  for (const auto& kernel_argument : kernel_arguments) {
    if (!kernel_argument.first_with_same_slice().has_value()) {
      args_.push_back(kernel_argument.slice());
      written_.push_back(kernel_argument.written());
    }
  }

  if (std::holds_alternative<const HloInstruction*>(op)) {
    // Skip populating MLIR values_ if emitting from HLO.
    return;
  }

  values_.reserve(kernel_arguments.size());
  for (const auto& kernel_argument : kernel_arguments) {
    if (!kernel_argument.first_with_same_slice().has_value()) {
      values_.push_back(RemoveTransformingOperations(kernel_argument.value()));
    }
  }
}

std::string KernelThunk::ToStringExtra(int indent) const {
  return absl::StrFormat(", kernel = %s, launch dimensions = %s", kernel_name_,
                         launch_dimensions_.ToString());
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
    CHECK(stream->ThenMemcpy(host_buffer.get(), buf, buf.size()).ok());
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
  const se::Kernel* kernel = nullptr;

  {
    absl::MutexLock lock(&mutex_);
    auto it = kernel_cache_.find(executor);
    CHECK(it != kernel_cache_.end())
        << "Initialize() not called for StreamExecutor " << executor;
    launch_dimensions = launch_dimensions_;
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
    PrintBufferContents(params.stream, buffer_args);
  }

  return ExecuteKernelOnStream(*kernel, buffer_args, launch_dimensions,
                               params.stream);
}

//===----------------------------------------------------------------------===//
// CustomKernelThunk
//===----------------------------------------------------------------------===//

CustomKernelThunk::CustomKernelThunk(
    std::variant<mlir::Operation*, const HloInstruction*> instr,
    CustomKernel custom_kernel,
    absl::Span<const KernelArgument> kernel_arguments)
    : Thunk(Kind::kCustomKernel,
            std::holds_alternative<mlir::Operation*>(instr)
                ? Thunk::ThunkInfo::WithProfileAnnotation(
                      std::get<mlir::Operation*>(instr))
                : Thunk::ThunkInfo::WithProfileAnnotation(
                      std::get<const HloInstruction*>(instr))),
      custom_kernel_(std::move(custom_kernel)) {
  args_.reserve(kernel_arguments.size());
  written_.reserve(kernel_arguments.size());
  for (const auto& kernel_argument : kernel_arguments) {
    if (!kernel_argument.first_with_same_slice().has_value()) {
      args_.push_back(kernel_argument.slice());
      written_.push_back(kernel_argument.written());
    }
  }

  if (std::holds_alternative<const HloInstruction*>(instr)) {
    // Skip populating MLIR values_ if emitting from HLO.
    return;
  }

  values_.reserve(kernel_arguments.size());
  for (const auto& kernel_argument : kernel_arguments) {
    if (!kernel_argument.first_with_same_slice().has_value()) {
      values_.push_back(RemoveTransformingOperations(kernel_argument.value()));
    }
  }
}

std::string CustomKernelThunk::ToStringExtra(int indent) const {
  return custom_kernel_.ToString();
}

absl::Status CustomKernelThunk::Initialize(const InitializeParams& params) {
  absl::MutexLock lock(&mutex_);

  auto it = kernel_cache_.find(params.executor);
  if (kernel_cache_.end() == it) {
    auto kernel = std::make_unique<se::Kernel>(params.executor);
    TF_RETURN_IF_ERROR(
        params.executor->GetKernel(custom_kernel_.kernel_spec(), kernel.get()));
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
    return executor->Launch(params.stream, custom_kernel_.thread_dims(),
                            custom_kernel_.block_dims(), *cluster, *kernel,
                            args);
  } else {
    return executor->Launch(params.stream, custom_kernel_.thread_dims(),
                            custom_kernel_.block_dims(), *kernel, args);
  }
}

}  // namespace gpu
}  // namespace xla
