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
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/print_buffer_contents.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tensor_map.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

CustomKernelThunk::CustomKernelThunk(
    Thunk::ThunkInfo thunk_info, CustomKernel custom_kernel,
    const emitters::KernelArguments& kernel_arguments, bool use_pdl,
    std::vector<int64_t> zeroed_output_buffer_indices,
    stream_executor::gpu::TmaMetadata tma_metadata)
    : Command(CommandType::kCustomKernelLaunchCmd, Kind::kCustomKernel,
              std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentShapedSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()),
      custom_kernel_(std::move(custom_kernel)),
      zeroed_output_buffer_indices_(std::move(zeroed_output_buffer_indices)),
      tma_metadata_(tma_metadata),
      use_pdl_(use_pdl) {}

std::string CustomKernelThunk::ToString(int indent) const {
  return custom_kernel_.ToString();
}

absl::Status CustomKernelThunk::Initialize(const InitializeParams& params) {
  absl::MutexLock lock(mutex_);

  if (!kernel_cache_.contains(params.executor)) {
    ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                     params.executor->LoadKernel(custom_kernel_.kernel_spec()));
    se::KernelMetadata m = kernel->metadata();
    m.set_shared_memory_bytes(custom_kernel_.shared_memory_bytes());
    kernel->set_metadata(m);
    kernel->set_use_pdl(use_pdl_);
    kernel_cache_.emplace(params.executor, std::move(kernel));
  }

  return absl::OkStatus();
}

absl::StatusOr<CustomKernelThunk::KernelWithArgs>
CustomKernelThunk::GetKernelAndArgs(const BufferAllocations& buffer_allocations,
                                    se::StreamExecutor* executor) const {
  se::Kernel* kernel;
  {
    absl::MutexLock lock(mutex_);
    auto it = kernel_cache_.find(executor);
    if (it == kernel_cache_.end() || it->second == nullptr) {
      return absl::InternalError(
          absl::StrCat("Custom kernel not loaded (Initialize() not called): ",
                       custom_kernel_.name()));
    }
    kernel = it->second.get();
  }

  absl::InlinedVector<se::KernelArg, 4> kernel_args;
  kernel_args.reserve(args_.size());
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

absl::Status CustomKernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = params.stream->parent();

  for (int64_t index : zeroed_output_buffer_indices_) {
    se::DeviceAddressBase address =
        params.buffer_allocations->GetDeviceAddress(args_[index].slice);
    RETURN_IF_ERROR(stream->MemZero(&address, address.size()));
  }

  ASSIGN_OR_RETURN(auto kernel_with_args,
                   GetKernelAndArgs(*params.buffer_allocations, executor));
  auto& [kernel, buffer_args] = kernel_with_args;

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "Launching " << custom_kernel_.ToString() << " as device kernel "
      << kernel->name();

  if (VLOG_IS_ON(100)) {
    PrintBufferContents(params.stream, buffer_args);
  }

  se::KernelArgsDeviceAddressArrayAdapter kernel_args =
      se::KernelArgsDeviceAddressArrayAdapter::Build(se::PackKernelArgs(
          buffer_args, custom_kernel_.shared_memory_bytes()));

  return kernel->Launch(
      custom_kernel_.thread_dims(), custom_kernel_.block_dims(),
      custom_kernel_.cluster_dims(), params.stream, kernel_args);
}

absl::StatusOr<const se::CommandBuffer::Command*> CustomKernelThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  VLOG(5) << "CustomKernelThunk::Record: custom_kernel="
          << custom_kernel_.name();

  ASSIGN_OR_RETURN(auto kernel_with_args,
                   GetKernelAndArgs(*execute_params.buffer_allocations,
                                    execute_params.stream->parent()));
  auto& [kernel, buffer_args] = kernel_with_args;

  se::KernelArgsDeviceAddressArrayAdapter kernel_args =
      se::KernelArgsDeviceAddressArrayAdapter::Build(se::PackKernelArgs(
          buffer_args, custom_kernel_.shared_memory_bytes()));

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateLaunch(
        custom_kernel_.thread_dims(), custom_kernel_.block_dims(), *kernel,
        kernel_args, create->dependencies, priority());
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(command_buffer->UpdateLaunch(
        update->command, custom_kernel_.thread_dims(),
        custom_kernel_.block_dims(), *kernel, kernel_args));
    return update->command;
  }
  return Internal("Invalid record action");
}

Thunk::BufferUses CustomKernelThunk::buffer_uses() const {
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

CustomKernelThunk::CustomKernelThunk(
    Thunk::ThunkInfo thunk_info, CustomKernel custom_kernel,
    std::vector<ShapedSlice> args, std::vector<bool> written,
    std::vector<int64_t> zeroed_output_buffer_indices,
    stream_executor::gpu::TmaMetadata tma_metadata, bool use_pdl)
    : Command(CommandType::kCustomKernelLaunchCmd, Kind::kCustomKernel,
              std::move(thunk_info)),
      args_(std::move(args)),
      written_(std::move(written)),
      custom_kernel_(std::move(custom_kernel)),
      zeroed_output_buffer_indices_(std::move(zeroed_output_buffer_indices)),
      tma_metadata_(tma_metadata),
      use_pdl_(use_pdl) {}

absl::StatusOr<ThunkProto> CustomKernelThunk::ToProto() const {
  ThunkProto thunk_proto;
  *thunk_proto.mutable_thunk_info() = thunk_info().ToProto();

  CustomKernelThunkProto* custom_kernel_thunk_proto =
      thunk_proto.mutable_custom_kernel_thunk();
  for (const ShapedSlice& arg : args_) {
    ASSIGN_OR_RETURN(*custom_kernel_thunk_proto->add_args(), arg.ToProto());
  }
  for (bool written : written_) {
    custom_kernel_thunk_proto->add_written(written);
  }
  ASSIGN_OR_RETURN(*custom_kernel_thunk_proto->mutable_custom_kernel(),
                   custom_kernel_.ToProto());

  custom_kernel_thunk_proto->mutable_zeroed_output_buffer_indices()->Assign(
      zeroed_output_buffer_indices_.begin(),
      zeroed_output_buffer_indices_.end());
  *custom_kernel_thunk_proto->mutable_tma_metadata() = tma_metadata_.ToProto();

  custom_kernel_thunk_proto->set_use_pdl(use_pdl_);
  return thunk_proto;
}

absl::StatusOr<std::unique_ptr<CustomKernelThunk>> CustomKernelThunk::FromProto(
    ThunkInfo thunk_info, const CustomKernelThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const std::optional<se::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver) {
  ASSIGN_OR_RETURN(
      CustomKernel custom_kernel,
      CustomKernel::FromProto(proto.custom_kernel(), symbol_resolver));
  std::vector<ShapedSlice> args;
  args.reserve(proto.args_size());
  for (const ShapedSliceProto& arg_proto : proto.args()) {
    ASSIGN_OR_RETURN(args.emplace_back(),
                     ShapedSlice::FromProto(arg_proto, buffer_allocations));
  }
  std::vector<bool> written{proto.written().begin(), proto.written().end()};

  std::vector<int64_t> zeroed_output_buffer_indices(
      proto.zeroed_output_buffer_indices().begin(),
      proto.zeroed_output_buffer_indices().end());

  ASSIGN_OR_RETURN(
      stream_executor::gpu::TmaMetadata tma_metadata,
      stream_executor::gpu::TmaMetadata::FromProto(proto.tma_metadata()));

  return absl::WrapUnique(new CustomKernelThunk(
      std::move(thunk_info), std::move(custom_kernel), args, std::move(written),
      std::move(zeroed_output_buffer_indices), std::move(tma_metadata),
      proto.use_pdl()));
}

}  // namespace gpu
}  // namespace xla
