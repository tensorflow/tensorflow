/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_kernel.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
int GetGpuStreamPriority(StreamExecutor* executor,
                         stream_executor::StreamPriority stream_priority) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  if (stream_priority == stream_executor::StreamPriority::Default) {
    return 0;
  }
  int lowest, highest;
  hipError_t res = wrap::hipDeviceGetStreamPriorityRange(&lowest, &highest);
  if (res != hipSuccess) {
    LOG(ERROR)
        << "Could not query stream priority range. Returning default priority.";
    return 0;
  }
  return stream_priority == stream_executor::StreamPriority::Highest ? highest
                                                                     : lowest;
}

absl::StatusOr<hipStream_t> CreateStream(StreamExecutor* executor,
                                         int priority) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  hipStream_t stream;
  if (priority == 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipStreamCreateWithFlags(&stream, hipStreamDefault),
        "Failed to create stream"));  // switch to hipStreamNonBlocking?
  } else {
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipStreamCreateWithPriority(&stream, hipStreamDefault, priority),
        "Failed to create stream"));  // switch to hipStreamNonBlocking?
  }

  VLOG(2) << "successfully created stream " << stream << " for device "
          << executor->device_ordinal() << " on thread";
  return stream;
}

absl::Status RecordEvent(StreamExecutor* executor, hipEvent_t event,
                         hipStream_t stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  hipError_t res = wrap::hipEventRecord(event, stream);
  switch (res) {
    case hipSuccess:
      return absl::OkStatus();
    case hipErrorDeinitialized:
    case hipErrorNotInitialized:
      return absl::FailedPreconditionError(
          absl::StrFormat("error recording ROCM event on stream %p: %s", stream,
                          ToString(res).c_str()));
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("error recording ROCM event on stream %p: %s", stream,
                          ToString(res).c_str()));
  }
}

absl::Status WaitStreamOnEvent(StreamExecutor* executor, hipStream_t stream,
                               hipEvent_t event) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipStreamWaitEvent(stream, event, 0 /* = flags */),
               "could not wait stream on event"));
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2H(StreamExecutor* executor, void* host_dst,
                                   hipDeviceptr_t gpu_src, uint64_t size,
                                   hipStream_t stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyDtoHAsync(host_dst, gpu_src, size, stream),
      absl::StrFormat(
          "failed to enqueue async memcpy from device to host: host dst: %p; "
          "Gpu src: %p; size: %llu=0x%llx",
          host_dst, absl::bit_cast<void*>(gpu_src), size, size)));

  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream
          << " device: " << executor->device_ordinal();
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyH2D(StreamExecutor* executor,
                                   hipDeviceptr_t gpu_dst, const void* host_src,
                                   uint64_t size, hipStream_t stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyHtoDAsync(gpu_dst, const_cast<void*>(host_src), size,
                               stream),
      absl::StrFormat(
          "failed to enqueue async memcpy from host to device: Gpu dst: %p; "
          "host src: %p; size: %llu=0x%llx",
          absl::bit_cast<void*>(gpu_dst), host_src, size, size)));

  VLOG(2) << "successfully enqueued async memcpy h2d of " << size
          << " bytes from " << host_src << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream
          << " device: " << executor->device_ordinal();
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2D(StreamExecutor* executor,
                                   hipDeviceptr_t gpu_dst,
                                   hipDeviceptr_t gpu_src, uint64_t size,
                                   hipStream_t stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream),
      absl::StrFormat("failed to enqueue async memcpy from device to device: "
                      "Gpu dst: %p ; Gpu src: %p ; size: %llu=0x%llx",
                      absl::bit_cast<void*>(gpu_dst),
                      absl::bit_cast<void*>(gpu_src), size, size)));

  VLOG(2) << "successfully enqueued async memcpy d2d of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream
          << " device: " << executor->device_ordinal();
  return absl::OkStatus();
}

absl::Status SynchronizeStream(StreamExecutor* executor, hipStream_t stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipStreamSynchronize(stream),
                              "Could not synchronize on ROCM stream"));
  VLOG(2) << "successfully synchronized stream " << stream << " on device "
          << executor->device_ordinal();
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<RocmStream>> RocmStream::Create(
    StreamExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  int stream_priority = [&]() {
    if (priority.has_value() && std::holds_alternative<int>(priority.value())) {
      return std::get<int>(priority.value());
    }
    return GetGpuStreamPriority(
        executor,
        std::get<StreamPriority>(priority.value_or(StreamPriority::Default)));
  }();
  TF_ASSIGN_OR_RETURN(auto stream_handle,
                      CreateStream(executor, stream_priority));

  TF_ASSIGN_OR_RETURN(auto completed_event,
                      RocmEvent::Create(executor,
                                        /*allow_timing=*/false));

  return std::unique_ptr<RocmStream>(new RocmStream(
      executor, std::move(completed_event), priority, stream_handle));
}

absl::Status RocmStream::WaitFor(Stream* other) {
  RocmStream* other_stream = static_cast<RocmStream*>(other);

  TF_RETURN_IF_ERROR(other_stream->RecordCompletedEvent());

  return WaitStreamOnEvent(executor_, stream_handle_,
                           other_stream->completed_event_.GetHandle());
}

absl::Status RocmStream::RecordEvent(Event* event) {
  return stream_executor::gpu::RecordEvent(
      executor_, static_cast<RocmEvent*>(event)->GetHandle(), stream_handle_);
}

absl::Status RocmStream::WaitFor(Event* event) {
  return WaitStreamOnEvent(executor_, stream_handle_,
                           static_cast<RocmEvent*>(event)->GetHandle());
}

absl::Status RocmStream::RecordCompletedEvent() {
  return RecordEvent(&completed_event_);
}

namespace {
void DestroyStream(StreamExecutor* executor, hipStream_t stream) {
  if (stream == nullptr) {
    return;
  }
  hipError_t res = wrap::hipStreamQuery(stream);
  if (res != hipSuccess) {
    LOG(ERROR) << "stream not idle on destroy: " << ToString(res);
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  res = wrap::hipStreamDestroy(stream);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to destroy ROCM stream for device "
               << executor->device_ordinal() << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << stream << " for device "
            << executor->device_ordinal();
  }
}
}  // namespace

RocmStream::~RocmStream() {
  BlockHostUntilDone().IgnoreError();
  executor_->DeallocateStream(this);

  DestroyStream(executor_, stream_handle_);
}

absl::Status RocmStream::Memset32(DeviceMemoryBase* location, uint32_t pattern,
                                  uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) != 0) {
    return absl::InvalidArgumentError("location must be 4 byte aligned.");
  }
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("size must be a multiple of 4 bytes.");
  }
  return ToStatus(wrap::hipMemsetD32Async(location->opaque(), pattern, size / 4,
                                          stream_handle_),
                  "Failed to memset memory");
}

absl::Status RocmStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return Memset32(location, 0x0, size);
  } else {
    std::unique_ptr<ActivateContext> activation = executor_->Activate();
    return ToStatus(
        wrap::hipMemsetAsync(location->opaque(), 0x0, size, stream_handle_),
        "Failed to enqueue async memset operation");
  }
}

absl::Status RocmStream::Memcpy(DeviceMemoryBase* gpu_dst,
                                const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2D(
      executor_, absl::bit_cast<hipDeviceptr_t>(gpu_dst->opaque()),
      absl::bit_cast<hipDeviceptr_t>(gpu_src.opaque()), size, stream_handle_);
}

absl::Status RocmStream::Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) {
  return AsynchronousMemcpyH2D(
      executor_, absl::bit_cast<hipDeviceptr_t>(gpu_dst->opaque()), host_src,
      size, stream_handle_);
}

absl::Status RocmStream::Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2H(executor_, host_dst,
                               absl::bit_cast<hipDeviceptr_t>(gpu_src.opaque()),
                               size, stream_handle_);
}

namespace {
void InternalHostCallback(void* data) {
  auto* callback = reinterpret_cast<absl::AnyInvocable<void() &&>*>(data);
  std::move (*callback)();
  delete callback;
}
}  // namespace

absl::Status RocmStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  auto callback_ptr =
      new absl::AnyInvocable<void() &&>([cb = std::move(callback)]() mutable {
        absl::Status s = std::move(cb)();
        if (!s.ok()) {
          LOG(WARNING) << "Host callback failed: " << s;
        }
      });
  return ToStatus(
      wrap::hipLaunchHostFunc(stream_handle_, (hipHostFn_t)InternalHostCallback,
                              callback_ptr),
      "unable to add host callback");
}

namespace {
absl::Status LaunchKernel(StreamExecutor* executor,
                          absl::string_view kernel_name, hipFunction_t function,
                          unsigned int grid_dim_x, unsigned int grid_dim_y,
                          unsigned int grid_dim_z, unsigned int block_dim_x,
                          unsigned int block_dim_y, unsigned int block_dim_z,
                          unsigned int shared_mem_bytes, hipStream_t stream,
                          void** kernel_params, void** extra) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << " smem: " << shared_mem_bytes
          << " func: " << (const void*)function;

  auto res = hipSuccess;
#if TF_ROCM_VERSION < 60200
  // for in-process kernel this function returns mangled kernel function name,
  // and null otherwise
  auto name = wrap::hipKernelNameRefByPtr((const void*)function, stream);
  if (name != nullptr) {
    res = wrap::hipLaunchKernel((const void*)function,
                                dim3(grid_dim_x, grid_dim_y, grid_dim_z),
                                dim3(block_dim_x, block_dim_y, block_dim_z),
                                kernel_params, shared_mem_bytes, stream);
  } else  // NOLINT(readability/braces)
#endif    // TF_ROCM_VERSION < 60200
  {
    res = wrap::hipModuleLaunchKernel(
        function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
        block_dim_z, shared_mem_bytes, stream, kernel_params, extra);
  }
  TF_RETURN_IF_ERROR(
      ToStatus(res, absl::StrCat("Failed to launch ROCm kernel: ", kernel_name,
                                 " with block dimensions: ", block_dim_x, "x",
                                 block_dim_y, "x", block_dim_z)));

  VLOG(2) << "successfully launched kernel";
  return absl::OkStatus();
}

absl::Status LaunchKernel(StreamExecutor* executor,
                          absl::string_view kernel_name, hipFunction_t function,
                          unsigned int cluster_dim_x,
                          unsigned int cluster_dim_y,
                          unsigned int cluster_dim_z, unsigned int grid_dim_x,
                          unsigned int grid_dim_y, unsigned int grid_dim_z,
                          unsigned int block_dim_x, unsigned int block_dim_y,
                          unsigned int block_dim_z,
                          unsigned int shared_mem_bytes, hipStream_t stream,
                          void** kernel_params, void** extra) {
  if (cluster_dim_x != 1 || cluster_dim_y != 1 || cluster_dim_z != 1)
    return absl::UnimplementedError("Not implemented for ROCm");
  return LaunchKernel(executor, kernel_name, function, grid_dim_x, grid_dim_y,
                      grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
                      shared_mem_bytes, stream, kernel_params, extra);
}

}  // namespace

absl::Status RocmStream::BlockHostUntilDone() {
  return SynchronizeStream(executor_, stream_handle_);
}

absl::Status RocmStream::Launch(const ThreadDim& thread_dims,
                                const BlockDim& block_dims,
                                const std::optional<ClusterDim>& cluster_dims,
                                const Kernel& kernel, const KernelArgs& args) {
  const RocmKernel* gpu_kernel = static_cast<const RocmKernel*>(&kernel);
  hipFunction_t function = gpu_kernel->gpu_function();

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
      return LaunchKernel(
          executor_, kernel.name(), function, cluster_dims->x, cluster_dims->y,
          cluster_dims->z, block_dims.x, block_dims.y, block_dims.z,
          thread_dims.x, thread_dims.y, thread_dims.z,
          packed.number_of_shared_bytes(), stream_handle_, params,
          /*extra=*/nullptr);
    } else {
      return LaunchKernel(
          executor_, kernel.name(), function, block_dims.x, block_dims.y,
          block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
          packed.number_of_shared_bytes(), stream_handle_, params,
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

}  // namespace stream_executor::gpu
