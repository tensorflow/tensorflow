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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/print_buffer_contents.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tensor_map.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

KernelThunk::KernelThunk(Thunk::ThunkInfo thunk_info, std::string kernel_name,
                         const emitters::KernelArguments& kernel_arguments,
                         LaunchDimensions launch_dimensions,
                         std::optional<se::ClusterDim> cluster_dim,
                         int64_t shmem_bytes,
                         stream_executor::gpu::TmaMetadata tma_metadata,
                         std::vector<int64_t> zeroed_output_buffer_indices,
                         bool use_pdl)
    : Command(CommandType::kLaunchCmd, Kind::kKernel, std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentShapedSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()),
      zeroed_output_buffer_indices_(std::move(zeroed_output_buffer_indices)),
      kernel_name_(std::move(kernel_name)),
      launch_dimensions_(std::move(launch_dimensions)),
      cluster_dim_(std::move(cluster_dim)),
      shmem_bytes_(shmem_bytes),
      tma_metadata_(std::move(tma_metadata)),
      use_pdl_(use_pdl) {}

std::string KernelThunk::ToString(int indent) const {
  return absl::StrFormat(
      "kernel=%s, profile_annotation=%s, launch dimensions=%s, cluster_dim=%s",
      kernel_name_, thunk_info().profile_annotation,
      launch_dimensions_.ToString(),
      cluster_dim_.has_value() ? cluster_dim_->ToString() : "nullopt");
}

absl::StatusOr<ThunkProto> KernelThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  KernelThunkProto* kernel_proto = proto.mutable_kernel_thunk();
  for (int i = 0; i < args_.size(); i++) {
    ASSIGN_OR_RETURN(*kernel_proto->add_args(), args_[i].slice.ToProto());
    *kernel_proto->add_args_shape() = args_[i].shape.ToProto();
    kernel_proto->add_written(written_[i]);
  }
  kernel_proto->set_kernel_name(kernel_name_);
  *kernel_proto->mutable_launch_dimensions() = launch_dimensions_.ToProto();
  if (cluster_dim_) {
    *kernel_proto->mutable_cluster_dim() = cluster_dim_->ToProto();
  }
  kernel_proto->set_shmem_bytes(shmem_bytes_);
  kernel_proto->set_use_pdl(use_pdl_);
  *kernel_proto->mutable_tma_metadata() = tma_metadata_.ToProto();

  kernel_proto->mutable_zeroed_output_buffer_indices()->Assign(
      zeroed_output_buffer_indices_.begin(),
      zeroed_output_buffer_indices_.end());
  return proto;
}

absl::StatusOr<std::unique_ptr<KernelThunk>> KernelThunk::FromProto(
    ThunkInfo thunk_info, const KernelThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                   LaunchDimensions::FromProto(proto.launch_dimensions()));
  std::optional<stream_executor::ClusterDim> cluster_dim;
  if (proto.has_cluster_dim()) {
    ASSIGN_OR_RETURN(
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
    ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                     BufferAllocation::Slice::FromProto(proto.args().at(i),
                                                        buffer_allocations));
    ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(proto.args_shape().at(i)));
    emitters::KernelArgument argument{shape, slice};
    argument.set_written(proto.written().at(i));
    arguments.push_back(std::move(argument));
  }

  ASSIGN_OR_RETURN(
      stream_executor::gpu::TmaMetadata tma_metadata,
      stream_executor::gpu::TmaMetadata::FromProto(proto.tma_metadata()));

  std::vector<int64_t> zeroed_output_buffer_indices(
      proto.zeroed_output_buffer_indices().begin(),
      proto.zeroed_output_buffer_indices().end());

  return std::make_unique<KernelThunk>(
      thunk_info, proto.kernel_name(),
      emitters::KernelArguments(std::move(arguments)), launch_dimensions,
      cluster_dim, proto.shmem_bytes(), tma_metadata,
      std::move(zeroed_output_buffer_indices), proto.use_pdl());
}

/*static*/ std::unique_ptr<KernelThunk> KernelThunk::MakeKernelThunk(
    std::string kernel_name, absl::Span<const ShapedSlice> args,
    absl::Span<const BufferUse::MemoryAccess> args_access,
    LaunchDimensions dims, int64_t shmem_bytes,
    stream_executor::gpu::TmaMetadata tma_metadata) {
  std::vector<emitters::KernelArgument> kernel_args;
  kernel_args.reserve(args.size());
  for (int i = 0; i < static_cast<int>(args.size()); ++i) {
    emitters::KernelArgument arg(args[i].shape, args[i].slice);
    arg.set_written(args_access[i] != BufferUse::MemoryAccess::kRead);
    kernel_args.push_back(std::move(arg));
  }
  return std::make_unique<KernelThunk>(
      Thunk::ThunkInfo(), std::move(kernel_name),
      emitters::KernelArguments(std::move(kernel_args)), std::move(dims),
      /*cluster_dim=*/std::nullopt, shmem_bytes, std::move(tma_metadata));
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
      ASSIGN_OR_RETURN(
          kernel, CreateKernel(kernel_name_, args_.size(), params.src.binary,
                               params.executor, shmem_bytes_, use_pdl_));

    } else {
      ASSIGN_OR_RETURN(kernel,
                       CreateKernel(kernel_name_, args_.size(), params.src.text,
                                    params.executor, shmem_bytes_, use_pdl_));
    }

    kernel_cache_.emplace(params.executor, std::move(kernel));
  }

  return absl::OkStatus();
}

absl::StatusOr<KernelThunk::KernelWithArgs> KernelThunk::GetKernelAndArgs(
    const BufferAllocations& buffer_allocations,
    se::StreamExecutor* executor) const {
  se::Kernel* kernel;
  {
    absl::MutexLock lock(mutex_);
    auto it = kernel_cache_.find(executor);
    if (it == kernel_cache_.end() || it->second == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Kernel not loaded for executor (Initialize() not called): %s",
          kernel_name_));
    }
    kernel = it->second.get();
  }
  absl::InlinedVector<se::KernelArg, 4> kernel_args;
  for (int idx = 0; idx < args_.size(); ++idx) {
    se::DeviceAddressBase buf =
        buffer_allocations.GetDeviceAddress(args_[idx].slice);
    VLOG(5) << "  Arg #" << idx << ": " << args_[idx].slice << ": "
            << buf.opaque() << " (" << buf.size() << "B)";
    if (auto it = tma_metadata_.arg_index_to_tma_info.find(idx);
        it != tma_metadata_.arg_index_to_tma_info.end()) {
      const se::gpu::TmaDescriptor& tma_desc = it->second;
      ASSIGN_OR_RETURN(se::TensorMap tensor_map,
                       executor->CreateTensorMap(tma_desc, buf.opaque()));
      VLOG(5) << "  Using TensorMap for arg #" << idx << ": "
              << tma_desc.ToString();
      kernel_args.push_back(std::move(tensor_map));
    } else {
      kernel_args.push_back(buf);
    }
  }
  return KernelWithArgs{kernel, std::move(kernel_args)};
}

absl::Status KernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();

  for (int64_t index : zeroed_output_buffer_indices_) {
    se::DeviceAddressBase address =
        params.buffer_allocations->GetDeviceAddress(args_[index].slice);
    RETURN_IF_ERROR(stream->MemZero(&address, address.size()));
  }

  ASSIGN_OR_RETURN(auto kernel_with_args,
                   GetKernelAndArgs(*params.buffer_allocations, executor));
  auto& [kernel, kernel_args] = kernel_with_args;

  int device_ordinal = executor->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal) << "Launching " << kernel->name();

  if (VLOG_IS_ON(100)) {
    PrintBufferContents(stream, kernel_args);
  }

  return ExecuteKernelOnStream(
      *kernel,
      absl::Span<se::KernelArg>(kernel_args.data(), kernel_args.size()),
      launch_dimensions_, cluster_dim_, stream);
}

absl::StatusOr<const se::CommandBuffer::Command*> KernelThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::StreamExecutor* executor = execute_params.stream->parent();

  ASSIGN_OR_RETURN(
      auto kernel_with_args,
      GetKernelAndArgs(*execute_params.buffer_allocations, executor));
  auto& [kernel, kernel_args] = kernel_with_args;
  auto packed_args = se::PackKernelArgs(kernel_args, shmem_bytes_);

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateLaunch(
        launch_dimensions_.thread_counts_per_block(),
        launch_dimensions_.block_counts(), *kernel, *packed_args,
        create->dependencies, priority());
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(command_buffer->UpdateLaunch(
        update->command, launch_dimensions_.thread_counts_per_block(),
        launch_dimensions_.block_counts(), *kernel, *packed_args));
    return update->command;
  }
  return Internal("Invalid record action");
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
