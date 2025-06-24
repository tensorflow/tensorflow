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

#include "xla/backends/gpu/runtime/kernel_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

//===----------------------------------------------------------------------===//
// KernelThunk
//===----------------------------------------------------------------------===//

KernelThunk::KernelThunk(
    Thunk::ThunkInfo thunk_info, std::string kernel_name,
    absl::Span<const emitters::KernelArgument> kernel_arguments,
    LaunchDimensions launch_dimensions,
    std::optional<se::ClusterDim> cluster_dim, int64_t shmem_bytes,
    std::optional<stream_executor::gpu::TmaMetadata> tma_metadata)
    : Thunk(Kind::kKernel, std::move(thunk_info)),
      kernel_name_(std::move(kernel_name)),
      launch_dimensions_(std::move(launch_dimensions)),
      cluster_dim_(std::move(cluster_dim)),
      shmem_bytes_(shmem_bytes),
      tma_metadata_(std::move(tma_metadata)) {
  args_.reserve(kernel_arguments.size());
  written_.reserve(kernel_arguments.size());
  for (const emitters::KernelArgument& kernel_argument : kernel_arguments) {
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

absl::StatusOr<ThunkProto> KernelThunk::ToProto() const {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, Thunk::ToProto());
  auto* kernel_proto = proto.mutable_kernel_thunk();
  for (const auto& arg : args_) {
    TF_ASSIGN_OR_RETURN(*kernel_proto->add_args(), arg.ToProto());
  }
  for (bool written : written_) {
    kernel_proto->add_written(written);
  }
  kernel_proto->set_kernel_name(kernel_name_);
  *kernel_proto->mutable_launch_dimensions() = launch_dimensions_.ToProto();
  if (cluster_dim_) {
    *kernel_proto->mutable_cluster_dim() = cluster_dim_->ToProto();
  }
  kernel_proto->set_shmem_bytes(shmem_bytes_);
  return proto;
}

absl::StatusOr<std::unique_ptr<KernelThunk>> KernelThunk::FromProto(
    ThunkInfo thunk_info, const KernelThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      LaunchDimensions::FromProto(proto.launch_dimensions()));
  std::optional<stream_executor::ClusterDim> cluster_dim;
  if (proto.has_cluster_dim()) {
    TF_ASSIGN_OR_RETURN(
        cluster_dim.emplace(),
        stream_executor::ClusterDim::FromProto(proto.cluster_dim()));
  }

  if (proto.written().size() != proto.args().size()) {
    return absl::InvalidArgumentError(
        "Proto fields `written` and `args` need to have the same cardinality.");
  }

  std::vector<emitters::KernelArgument> arguments;
  arguments.reserve(proto.args().size());
  for (int i = 0; i < proto.args().size(); ++i) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        BufferAllocation::Slice::FromProto(proto.args().at(i),
                                                           buffer_allocations));
    bool written = proto.written().at(i);
    arguments.push_back(emitters::KernelArgument{Shape{}, slice, written});
  }

  return std::make_unique<KernelThunk>(thunk_info, proto.kernel_name(),
                                       arguments, launch_dimensions,
                                       cluster_dim, proto.shmem_bytes());
}

absl::Status KernelThunk::Initialize(const InitializeParams& params) {
  absl::MutexLock lock(&mutex_);

  // Load the kernel into the device if necessary.
  //
  // We could alternatively do this within ExecuteOnStream, but doing it here
  // lets the time spent loading the kernel not count towards our execution
  // profiles.
  if (!kernel_cache_.contains(params.executor)) {
    std::unique_ptr<se::Kernel> kernel;
    if (!params.src.binary.empty()) {
      TF_ASSIGN_OR_RETURN(
          kernel, CreateKernel(kernel_name_, args_.size(), params.src.binary,
                               params.executor, shmem_bytes_));

    } else {
      TF_ASSIGN_OR_RETURN(
          kernel, CreateKernel(kernel_name_, args_.size(), params.src.text,
                               params.executor, shmem_bytes_));
    }

    kernel_cache_.emplace(params.executor, std::move(kernel));
  }

  return absl::OkStatus();
}

static void PrintBufferContents(
    se::Stream* stream, absl::Span<const se::KernelArgument> kernel_args) {
  int input_idx = 0;
  for (const se::KernelArgument& arg : kernel_args) {
    if (std::holds_alternative<se::DeviceMemoryBase>(arg)) {
      se::DeviceMemoryBase buf = std::get<se::DeviceMemoryBase>(arg);

      auto host_buffer = std::make_unique<char[]>(buf.size());
      CHECK_OK(stream->Memcpy(host_buffer.get(), buf, buf.size()));
      CHECK_OK(stream->BlockHostUntilDone());

      std::string buffer_contents;
      for (int i = 0; i < buf.size(); ++i) {
        absl::StrAppendFormat(&buffer_contents, "%x ",
                              static_cast<unsigned>(host_buffer[i]));
      }
      VLOG(100) << "BUF(" << input_idx++ << ") = " << buffer_contents;
    } else {
      se::TensorMap tensor_map = std::get<se::TensorMap>(arg);
      VLOG(100) << "TENSOR_MAP(" << input_idx++ << ") = ";
      for (std::byte element : tensor_map.storage) {
        VLOG(100) << absl::StrFormat("%x ", static_cast<unsigned>(element));
      }
    }
  }
}

absl::Status KernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  // Load the kernel.
  se::StreamExecutor* executor = params.stream->parent();
  se::Kernel* kernel = nullptr;

  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));

  {
    absl::MutexLock lock(&mutex_);
    auto it = kernel_cache_.find(executor);
    CHECK(it != kernel_cache_.end())
        << "Initialize() not called for StreamExecutor " << executor;
    kernel = it->second.get();
  }

  VLOG(3) << "Launching " << kernel->name();
  absl::InlinedVector<std::variant<se::DeviceMemoryBase, se::TensorMap>, 4>
      kernel_args;
  stream_executor::gpu::TmaMetadata tma_metadata =
      tma_metadata_.value_or(stream_executor::gpu::TmaMetadata{});
  for (const auto& [idx, arg] : llvm::enumerate(args_)) {
    se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(3) << "  Arg: alloc #" << arg.index() << ", offset: " << arg.offset()
            << ": " << buf.opaque() << " (" << buf.size() << "B)";

    if (auto it = tma_metadata.arg_index_to_tma_info.find(idx);
        it != tma_metadata.arg_index_to_tma_info.end()) {
      // TMA descriptor argument.
      stream_executor::gpu::TmaDescriptor tma_desc = it->second;
      TF_ASSIGN_OR_RETURN(se::TensorMap tensor_map,
                          executor->CreateTensorMap(tma_desc, buf.opaque()));
      VLOG(3) << "  Using TensorMap for arg #" << arg.index() << ": "
              << tma_desc.ToString();
      kernel_args.push_back(std::move(tensor_map));
    } else {
      // Buffer argument.
      kernel_args.push_back(buf);
    }
  }

  if (VLOG_IS_ON(100)) {
    PrintBufferContents(stream, kernel_args);
  }

  return ExecuteKernelOnStream(
      *kernel,
      absl::Span<std::variant<se::DeviceMemoryBase, se::TensorMap>>(
          kernel_args.data(), kernel_args.size()),
      launch_dimensions_, cluster_dim_, stream);
}

//===----------------------------------------------------------------------===//
// CustomKernelThunk
//===----------------------------------------------------------------------===//

CustomKernelThunk::CustomKernelThunk(
    const HloInstruction* instr, CustomKernel custom_kernel,
    absl::Span<const emitters::KernelArgument> kernel_arguments)
    : Thunk(Kind::kCustomKernel,
            Thunk::ThunkInfo::WithProfileAnnotation(instr)),
      custom_kernel_(std::move(custom_kernel)) {
  args_.reserve(kernel_arguments.size());
  written_.reserve(kernel_arguments.size());
  for (const emitters::KernelArgument& kernel_argument : kernel_arguments) {
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
    absl::InlinedVector<se::KernelArgument, 4> kernel_args;
    for (const se::DeviceMemoryBase& arg : buffer_args) {
      kernel_args.push_back(arg);
    }
    PrintBufferContents(params.stream, kernel_args);
  }

  se::KernelArgsDeviceMemoryArray args(buffer_args,
                                       custom_kernel_.shared_memory_bytes());

  if (auto cluster = custom_kernel_.cluster_dims(); cluster.has_value()) {
    return kernel->Launch(custom_kernel_.thread_dims(),
                          custom_kernel_.block_dims(), *cluster, params.stream,
                          args);
  }
  return kernel->Launch(custom_kernel_.thread_dims(),
                        custom_kernel_.block_dims(), params.stream, args);
}

}  // namespace gpu
}  // namespace xla
