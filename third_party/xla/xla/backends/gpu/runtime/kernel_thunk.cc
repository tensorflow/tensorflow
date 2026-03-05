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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/runtime/print_buffer_contents.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tensor_map.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;
using tsl::profiler::TraceMeLevel;

namespace xla {
namespace gpu {

KernelThunk::KernelThunk(Thunk::ThunkInfo thunk_info, std::string kernel_name,
                         const emitters::KernelArguments& kernel_arguments,
                         LaunchDimensions launch_dimensions,
                         std::optional<se::ClusterDim> cluster_dim,
                         int64_t shmem_bytes,
                         stream_executor::gpu::TmaMetadata tma_metadata,
                         std::vector<int64_t> zeroed_output_buffer_indices)
    : Thunk(Kind::kKernel, std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentShapedSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()),
      zeroed_output_buffer_indices_(std::move(zeroed_output_buffer_indices)),
      kernel_name_(std::move(kernel_name)),
      launch_dimensions_(std::move(launch_dimensions)),
      cluster_dim_(std::move(cluster_dim)),
      shmem_bytes_(shmem_bytes),
      tma_metadata_(std::move(tma_metadata)) {}

std::string KernelThunk::ToString(int indent) const {
  return absl::StrFormat(
      ", kernel = %s, profile_annotation = %s, launch dimensions = %s, "
      "cluster_dim = %s",
      kernel_name_, thunk_info().profile_annotation,
      launch_dimensions_.ToString(),
      cluster_dim_.has_value() ? cluster_dim_->ToString() : "nullopt");
}

absl::StatusOr<ThunkProto> KernelThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  auto* kernel_proto = proto.mutable_kernel_thunk();
  for (int i = 0; i < args_.size(); i++) {
    TF_ASSIGN_OR_RETURN(*kernel_proto->add_args(), args_[i].slice.ToProto());
    *kernel_proto->add_args_shape() = args_[i].shape.ToProto();
    kernel_proto->add_written(written_[i]);
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

  if (proto.written().size() != proto.args().size() ||
      proto.args().size() != proto.args_shape().size()) {
    return absl::InvalidArgumentError(
        "Proto fields `written`, `args` and `args_shape` need to have the same "
        "cardinality.");
  }

  std::vector<emitters::KernelArgument> arguments;
  arguments.reserve(proto.args().size());
  for (int i = 0; i < proto.args().size(); ++i) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        BufferAllocation::Slice::FromProto(proto.args().at(i),
                                                           buffer_allocations));
    TF_ASSIGN_OR_RETURN(Shape shape,
                        Shape::FromProto(proto.args_shape().at(i)));
    emitters::KernelArgument argument{shape, slice};
    argument.set_written(proto.written().at(i));
    arguments.push_back(std::move(argument));
  }

  TF_ASSIGN_OR_RETURN(
      stream_executor::gpu::TmaMetadata tma_metadata,
      stream_executor::gpu::TmaMetadata::FromProto(proto.tma_metadata()));

  std::vector<int64_t> zeroed_output_buffer_indices(
      proto.zeroed_output_buffer_indices().begin(),
      proto.zeroed_output_buffer_indices().end());

  return std::make_unique<KernelThunk>(
      thunk_info, proto.kernel_name(),
      emitters::KernelArguments(std::move(arguments)), launch_dimensions,
      cluster_dim, proto.shmem_bytes(), tma_metadata,
      std::move(zeroed_output_buffer_indices));
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

absl::Status KernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  TraceMe trace(
      [] { return TraceMeEncode("KernelThunk::ExecuteOnStream", {}); },
      /*level=*/TraceMeLevel::kVerbose);

  // Load the kernel.
  se::StreamExecutor* executor = params.stream->parent();
  se::Kernel* kernel = nullptr;

  se::Stream* stream = nullptr;
  {
    TraceMe trace(
        [] {
          return TraceMeEncode(
              "KernelThunk::ExecuteOnStream/GetStreamForExecution", {});
        },
        /*level=*/TraceMeLevel::kVerbose);
    TF_ASSIGN_OR_RETURN(
        stream, GetStreamForExecution(Thunk::execution_stream_id(), params));
  }

  for (int64_t index : zeroed_output_buffer_indices_) {
    se::DeviceAddressBase address =
        params.buffer_allocations->GetDeviceAddress(args_[index].slice);
    TF_RETURN_IF_ERROR(stream->MemZero(&address, address.size()));
  }

  {
    TraceMe trace(
        [] { return TraceMeEncode("KernelThunk::ExecuteOnStream/mutex", {}); },
        /*level=*/TraceMeLevel::kVerbose);
    absl::MutexLock lock(mutex_);
    TraceMe trace_find(
        [] {
          return TraceMeEncode("KernelThunk::ExecuteOnStream/mutex/find", {});
        },
        /*level=*/TraceMeLevel::kVerbose);
    auto it = kernel_cache_.find(executor);
    CHECK(it != kernel_cache_.end())
        << "Initialize() not called for StreamExecutor " << executor;
    kernel = it->second.get();
  }

  absl::InlinedVector<se::KernelArg, 4> kernel_args;
  {
    TraceMe trace(
        [] {
          return TraceMeEncode("KernelThunk::ExecuteOnStream/kernel_args", {});
        },
        /*level=*/TraceMeLevel::kVerbose);
    int device_ordinal = executor->device_ordinal();
    XLA_VLOG_DEVICE(3, device_ordinal) << "Launching " << kernel->name();
    for (const auto& [idx, arg] : llvm::enumerate(args_)) {
      se::DeviceAddressBase buf =
          params.buffer_allocations->GetDeviceAddress(arg.slice);
      XLA_VLOG_DEVICE(3, device_ordinal)
          << "Arg: alloc #" << arg.slice.index()
          << ", offset: " << arg.slice.offset() << ": " << buf.opaque() << " ("
          << buf.size() << "B)";

      if (auto it = tma_metadata_.arg_index_to_tma_info.find(idx);
          it != tma_metadata_.arg_index_to_tma_info.end()) {
        // TMA descriptor argument.
        const se::gpu::TmaDescriptor& tma_desc = it->second;
        TF_ASSIGN_OR_RETURN(se::TensorMap tensor_map,
                            executor->CreateTensorMap(tma_desc, buf.opaque()));
        XLA_VLOG_DEVICE(3, device_ordinal) << "Using TensorMap for arg #" << idx
                                           << ": " << tma_desc.ToString();
        kernel_args.push_back(std::move(tensor_map));
      } else {
        // Buffer argument.
        kernel_args.push_back(buf);
      }
    }
  }

  if (VLOG_IS_ON(100)) {
    PrintBufferContents(stream, kernel_args);
  }

  return ExecuteKernelOnStream(
      *kernel,
      absl::Span<se::KernelArg>(kernel_args.data(), kernel_args.size()),
      launch_dimensions_, cluster_dim_, stream);
}

Thunk::BufferUses KernelThunk::buffer_uses() const {
  Thunk::BufferUses buffers;
  buffers.reserve(args_.size());
  for (int i = 0; i < args_.size(); ++i) {
    // We assume that any buffer is either an input or an output of the
    // kernel, and inout buffers are represented as 2 separate arguments.
    if (written_[i]) {
      buffers.push_back(BufferUse::Write(args_[i].slice, args_[i].shape));
    } else {
      buffers.push_back(BufferUse::Read(args_[i].slice, args_[i].shape));
    }
  }
  return buffers;
}

}  // namespace gpu
}  // namespace xla
