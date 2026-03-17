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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/runtime/print_buffer_contents.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

CustomKernelThunk::CustomKernelThunk(
    Thunk::ThunkInfo thunk_info, CustomKernel custom_kernel,
    const emitters::KernelArguments& kernel_arguments)
    : Thunk(Kind::kCustomKernel, std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentShapedSlices()),
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

  absl::InlinedVector<se::DeviceAddressBase, 4> buffer_args;
  for (const ShapedSlice& arg : args_) {
    se::DeviceAddressBase buf =
        params.buffer_allocations->GetDeviceAddress(arg.slice);
    VLOG(3) << "[" << device_ordinal << "]  Arg: alloc #" << arg.slice.index()
            << ", offset: " << arg.slice.offset() << ": " << buf.opaque()
            << " (" << buf.size() << "B)";
    buffer_args.push_back(buf);
  }

  if (VLOG_IS_ON(100)) {
    absl::InlinedVector<se::KernelArg, 4> kernel_args;
    for (const se::DeviceAddressBase& arg : buffer_args) {
      kernel_args.push_back(arg);
    }
    PrintBufferContents(params.stream, kernel_args);
  }

  stream_executor::KernelArgsDeviceAddressArray args(
      buffer_args, custom_kernel_.shared_memory_bytes());

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
      buffers.push_back(BufferUse::Write(args_[i].slice, args_[i].shape));
    } else {
      buffers.push_back(BufferUse::Read(args_[i].slice, args_[i].shape));
    }
  }
  return buffers;
}

CustomKernelThunk::CustomKernelThunk(Thunk::ThunkInfo thunk_info,
                                     CustomKernel custom_kernel,
                                     std::vector<ShapedSlice> args,
                                     std::vector<bool> written)
    : Thunk(Kind::kCustomKernel, std::move(thunk_info)),
      args_(std::move(args)),
      written_(std::move(written)),
      custom_kernel_(std::move(custom_kernel)) {}

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
  return absl::WrapUnique(new CustomKernelThunk(std::move(thunk_info),
                                                std::move(custom_kernel), args,
                                                std::move(written)));
}

}  // namespace gpu
}  // namespace xla
