/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/nvtx_utils.h"

namespace stream_executor {
namespace gpu {

namespace {
void InternalHostCallback(void* data) {
  auto* callback = reinterpret_cast<absl::AnyInvocable<void() &&>*>(data);
  std::move (*callback)();
  delete callback;
}
}  // namespace


Stream::PlatformSpecificHandle GpuStream::platform_specific_handle() const {
  PlatformSpecificHandle handle;
  handle.stream = gpu_stream_;
  return handle;
}

absl::Status GpuStream::Memset32(DeviceMemoryBase* location, uint32_t pattern,
                                 uint64_t size) {
  CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
        size % 4 == 0);
  return GpuDriver::AsynchronousMemsetUint32(
      parent_->gpu_context(),
      reinterpret_cast<GpuDevicePtr>(location->opaque()), pattern, size / 4,
      gpu_stream());
}

absl::Status GpuStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return Memset32(location, 0x0, size);
  } else {
    return GpuDriver::AsynchronousMemsetUint8(
        parent_->gpu_context(),
        reinterpret_cast<GpuDevicePtr>(location->opaque()), 0x0, size,
        gpu_stream());
  }
}

absl::Status GpuStream::Memcpy(DeviceMemoryBase* gpu_dst,
                               const DeviceMemoryBase& gpu_src, uint64_t size) {
  return GpuDriver::AsynchronousMemcpyD2D(
      parent_->gpu_context(),
      reinterpret_cast<GpuDevicePtr>(const_cast<void*>(gpu_dst->opaque())),
      reinterpret_cast<GpuDevicePtr>(const_cast<void*>(gpu_src.opaque())), size,
      gpu_stream());
}

absl::Status GpuStream::Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                               uint64_t size) {
  return GpuDriver::AsynchronousMemcpyH2D(
      parent_->gpu_context(), reinterpret_cast<GpuDevicePtr>(gpu_dst->opaque()),
      host_src, size, gpu_stream());
}

absl::Status GpuStream::Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                               uint64_t size) {
  return GpuDriver::AsynchronousMemcpyD2H(
      parent_->gpu_context(), host_dst,
      reinterpret_cast<GpuDevicePtr>(const_cast<void*>(gpu_src.opaque())), size,
      gpu_stream());
}

absl::Status GpuStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  auto callback_ptr =
      new absl::AnyInvocable<void() &&>([cb = std::move(callback)]() mutable {
        absl::Status s = std::move(cb)();
        if (!s.ok()) {
          LOG(WARNING) << "Host callback failed: " << s;
        }
      });
  return GpuDriver::AddStreamCallback(parent_->gpu_context(), gpu_stream(),
                                      InternalHostCallback, callback_ptr);
}

GpuStream::~GpuStream() {
  GpuDriver::DestroyStream(parent_->gpu_context(), gpu_stream_);
}

void GpuStream::set_name(absl::string_view name) {
  name_ = name;
  tsl::profiler::NameStream(
      reinterpret_cast<tsl::profiler::StreamHandle>(gpu_stream()), name_);
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>>
GpuStream::CreateEventBasedTimer(bool use_delay_kernel) {
  return parent_->CreateEventBasedTimer(this, use_delay_kernel);
}

absl::Status GpuStream::Launch(const ThreadDim& thread_dims,
                               const BlockDim& block_dims, const Kernel& kernel,
                               const KernelArgs& args) {
  return Launch(thread_dims, block_dims, std::nullopt, kernel, args);
}

absl::Status GpuStream::Launch(const ThreadDim& thread_dims,
                               const BlockDim& block_dims,
                               const ClusterDim& cluster_dims,
                               const Kernel& kernel, const KernelArgs& args) {
  return Launch(thread_dims, block_dims, std::make_optional(cluster_dims),
                kernel, args);
}

absl::Status GpuStream::Launch(const ThreadDim& thread_dims,
                               const BlockDim& block_dims,
                               const std::optional<ClusterDim>& cluster_dims,
                               const Kernel& kernel, const KernelArgs& args) {
  const GpuKernel* gpu_kernel = AsGpuKernel(&kernel);
  GpuFunctionHandle function = gpu_kernel->gpu_function();

  // Launch kernels with packed arguments.
  auto launch = [this, &kernel, &cluster_dims, &thread_dims, &block_dims,
                 &function](const KernelArgsPackedArrayBase& packed) {
    int32_t expected_number_of_arguments =
        kernel.Arity() + (packed.number_of_shared_bytes() > 0);

    CHECK_EQ(expected_number_of_arguments, packed.number_of_arguments())
        << "Kernel " << kernel.name() << " has " << packed.number_of_arguments()
        << " arguments, but expected " << expected_number_of_arguments
        << "; arity=" << kernel.Arity()
        << "; number_of_shared_bytes=" << packed.number_of_shared_bytes();

    void** params = const_cast<void**>(packed.argument_addresses().data());

    if (cluster_dims.has_value()) {
      return GpuDriver::LaunchKernel(
          parent_->gpu_context(), kernel.name(), function, cluster_dims->x,
          cluster_dims->y, cluster_dims->z, block_dims.x, block_dims.y,
          block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
          packed.number_of_shared_bytes(), gpu_stream(), params,
          /*extra=*/nullptr);
    } else {
      return GpuDriver::LaunchKernel(
          parent_->gpu_context(), kernel.name(), function, block_dims.x,
          block_dims.y, block_dims.z, thread_dims.x, thread_dims.y,
          thread_dims.z, packed.number_of_shared_bytes(), gpu_stream(), params,
          /*extra=*/nullptr);
    }
  };

  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return launch(*packed);
  }

  // For device memory array we rely on a custom kernel arguments packing.
  if (auto* device_mem = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
    auto& pack = kernel.args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    TF_ASSIGN_OR_RETURN(auto packed, pack(kernel, *device_mem));
    return launch(*packed);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

GpuStream* AsGpuStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return static_cast<GpuStream*>(stream);
}

GpuStreamHandle AsGpuStreamValue(Stream* stream) {
  DCHECK(stream != nullptr);
  return AsGpuStream(stream)->gpu_stream();
}

}  // namespace gpu
}  // namespace stream_executor
