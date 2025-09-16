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
#include "xla/backends/gpu/runtime/thunk.pb.h"
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

KernelThunk::KernelThunk(Thunk::ThunkInfo thunk_info, std::string kernel_name,
                         const emitters::KernelArguments& kernel_arguments,
                         LaunchDimensions launch_dimensions,
                         std::optional<se::ClusterDim> cluster_dim,
                         int64_t shmem_bytes,
                         stream_executor::gpu::TmaMetadata tma_metadata)
    : Thunk(Kind::kKernel, std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentBufferSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()),
      kernel_name_(std::move(kernel_name)),
      launch_dimensions_(std::move(launch_dimensions)),
      cluster_dim_(std::move(cluster_dim)),
      shmem_bytes_(shmem_bytes),
      tma_metadata_(std::move(tma_metadata)) {}

std::string KernelThunk::ToString(int indent) const {
  return absl::StrFormat(
      ", kernel = %s, launch dimensions = %s, cluster_dim = %s", kernel_name_,
      launch_dimensions_.ToString(),
      cluster_dim_.has_value() ? cluster_dim_->ToString() : "nullopt");
}

absl::StatusOr<ThunkProto> KernelThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

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
  *kernel_proto->mutable_tma_metadata() = tma_metadata_.ToProto();
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
    emitters::KernelArgument argument{Shape{}, slice};
    argument.set_written(proto.written().at(i));
    arguments.push_back(std::move(argument));
  }

  TF_ASSIGN_OR_RETURN(
      stream_executor::gpu::TmaMetadata tma_metadata,
      stream_executor::gpu::TmaMetadata::FromProto(proto.tma_metadata()));

  return std::make_unique<KernelThunk>(
      thunk_info, proto.kernel_name(),
      emitters::KernelArguments(std::move(arguments)), launch_dimensions,
      cluster_dim, proto.shmem_bytes(), tma_metadata);
}

absl::Status KernelThunk::Initialize(const InitializeParams& params) {
  absl::MutexLock lock(mutex_);

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

void PrintBufferContents(se::Stream*, int input_idx, se::TensorMap tensor_map) {
  VLOG(100) << "TENSOR_MAP(" << input_idx << ") = ";
  for (std::byte element : tensor_map.storage) {
    VLOG(100) << absl::StrFormat("%x ", static_cast<unsigned>(element));
  }
}

void PrintBufferContents(se::Stream* stream, int input_idx,
                         se::DeviceMemoryBase buf) {
  auto host_buffer = std::make_unique<char[]>(buf.size());
  CHECK_OK(stream->Memcpy(host_buffer.get(), buf, buf.size()));
  CHECK_OK(stream->BlockHostUntilDone());

  std::string buffer_contents;
  for (int i = 0; i < buf.size(); ++i) {
    absl::StrAppendFormat(&buffer_contents, "%x ",
                          static_cast<unsigned>(host_buffer[i]));
  }
  VLOG(100) << "BUF(" << input_idx << ") = " << buffer_contents;
}

static void PrintBufferContents(
    se::Stream* stream, absl::Span<const se::KernelArgument> kernel_args) {
  for (const auto& [input_idx, arg] : llvm::enumerate(kernel_args)) {
    // pre-cpp-20-compat(P0588R1): Capturing structured bindings in lambdas is
    // ill-formed.
    std::visit(
        [&stream, &input_idx = input_idx](auto const& arg) {
          PrintBufferContents(stream, input_idx, arg);
        },
        arg);
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
    absl::MutexLock lock(mutex_);
    auto it = kernel_cache_.find(executor);
    CHECK(it != kernel_cache_.end())
        << "Initialize() not called for StreamExecutor " << executor;
    kernel = it->second.get();
  }

  int device_ordinal = executor->device_ordinal();
  VLOG(3) << "[" << device_ordinal << "] Launching " << kernel->name();
  absl::InlinedVector<se::KernelArgument, 4> kernel_args;
  for (const auto& [idx, arg] : llvm::enumerate(args_)) {
    se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(3) << "[" << device_ordinal << "] Arg: alloc #" << arg.index()
            << ", offset: " << arg.offset() << ": " << buf.opaque() << " ("
            << buf.size() << "B)";

    if (auto it = tma_metadata_.arg_index_to_tma_info.find(idx);
        it != tma_metadata_.arg_index_to_tma_info.end()) {
      // TMA descriptor argument.
      const se::gpu::TmaDescriptor& tma_desc = it->second;
      TF_ASSIGN_OR_RETURN(se::TensorMap tensor_map,
                          executor->CreateTensorMap(tma_desc, buf.opaque()));
      VLOG(3) << "[" << device_ordinal << "]  Using TensorMap for arg #" << idx
              << ": " << tma_desc.ToString();
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
      absl::Span<se::KernelArgument>(kernel_args.data(), kernel_args.size()),
      launch_dimensions_, cluster_dim_, stream);
}

//===----------------------------------------------------------------------===//
// CustomKernelThunk
//===----------------------------------------------------------------------===//

CustomKernelThunk::CustomKernelThunk(
    const HloInstruction* instr, CustomKernel custom_kernel,
    const emitters::KernelArguments& kernel_arguments)
    : Thunk(Kind::kCustomKernel,
            Thunk::ThunkInfo::WithProfileAnnotation(instr)),
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

}  // namespace gpu
}  // namespace xla
