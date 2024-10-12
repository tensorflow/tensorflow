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
#include <utility>
#include <variant>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
int GetGpuStreamPriority(Context* context,
                         stream_executor::StreamPriority stream_priority) {
  ScopedActivateContext activation(context);
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

absl::StatusOr<hipStream_t> CreateStream(Context* context, int priority) {
  ScopedActivateContext activated(context);
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
          << context->device_ordinal() << " on thread";
  return stream;
}

absl::Status RecordEvent(Context* context, hipEvent_t event,
                         hipStream_t stream) {
  ScopedActivateContext activated{context};
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

absl::Status WaitStreamOnEvent(Context* context, hipStream_t stream,
                               hipEvent_t event) {
  ScopedActivateContext activation{context};
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipStreamWaitEvent(stream, event, 0 /* = flags */),
               "could not wait stream on event"));
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2H(Context* context, void* host_dst,
                                   hipDeviceptr_t gpu_src, uint64_t size,
                                   hipStream_t stream) {
  ScopedActivateContext activation{context};
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyDtoHAsync(host_dst, gpu_src, size, stream),
      absl::StrFormat(
          "failed to enqueue async memcpy from device to host: host dst: %p; "
          "Gpu src: %p; size: %llu=0x%llx",
          host_dst, absl::bit_cast<void*>(gpu_src), size, size)));

  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream
          << " device: " << context->device_ordinal();
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyH2D(Context* context, hipDeviceptr_t gpu_dst,
                                   const void* host_src, uint64_t size,
                                   hipStream_t stream) {
  ScopedActivateContext activation{context};
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
          << " device: " << context->device_ordinal();
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2D(Context* context, hipDeviceptr_t gpu_dst,
                                   hipDeviceptr_t gpu_src, uint64_t size,
                                   hipStream_t stream) {
  ScopedActivateContext activation{context};
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream),
      absl::StrFormat("failed to enqueue async memcpy from device to device: "
                      "Gpu dst: %p ; Gpu src: %p ; size: %llu=0x%llx",
                      absl::bit_cast<void*>(gpu_dst),
                      absl::bit_cast<void*>(gpu_src), size, size)));

  VLOG(2) << "successfully enqueued async memcpy d2d of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream
          << " device: " << context->device_ordinal();
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<RocmStream>> RocmStream::Create(
    GpuExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  int stream_priority = [&]() {
    if (priority.has_value() && std::holds_alternative<int>(priority.value())) {
      return std::get<int>(priority.value());
    }
    return GetGpuStreamPriority(
        executor->gpu_context(),
        std::get<StreamPriority>(priority.value_or(StreamPriority::Default)));
  }();
  TF_ASSIGN_OR_RETURN(auto stream_handle,
                      CreateStream(executor->gpu_context(), stream_priority));

  TF_ASSIGN_OR_RETURN(auto completed_event,
                      RocmEvent::Create(executor->gpu_context(),
                                        /*allow_timing=*/false));

  return std::unique_ptr<RocmStream>(new RocmStream(
      executor, std::move(completed_event), priority, stream_handle));
}

absl::Status RocmStream::WaitFor(Stream* other) {
  RocmStream* other_stream = static_cast<RocmStream*>(other);

  TF_RETURN_IF_ERROR(other_stream->RecordCompletedEvent());

  return WaitStreamOnEvent(executor_->gpu_context(), gpu_stream(),
                           other_stream->completed_event_.GetHandle());
}

absl::Status RocmStream::RecordEvent(Event* event) {
  return stream_executor::gpu::RecordEvent(
      executor_->gpu_context(), static_cast<RocmEvent*>(event)->GetHandle(),
      gpu_stream());
}

absl::Status RocmStream::WaitFor(Event* event) {
  return WaitStreamOnEvent(executor_->gpu_context(), gpu_stream(),
                           static_cast<RocmEvent*>(event)->GetHandle());
}

absl::Status RocmStream::RecordCompletedEvent() {
  return RecordEvent(&completed_event_);
}

RocmStream::~RocmStream() {
  BlockHostUntilDone().IgnoreError();
  executor_->DeallocateStream(this);
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
                                          gpu_stream()),
                  "Failed to memset memory");
}

absl::Status RocmStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return Memset32(location, 0x0, size);
  } else {
    ScopedActivateContext activation{executor_->gpu_context()};
    return ToStatus(
        wrap::hipMemsetAsync(location->opaque(), 0x0, size, gpu_stream()),
        "Failed to enqueue async memset operation");
  }
}

absl::Status RocmStream::Memcpy(DeviceMemoryBase* gpu_dst,
                                const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2D(
      executor_->gpu_context(),
      absl::bit_cast<hipDeviceptr_t>(gpu_dst->opaque()),
      absl::bit_cast<hipDeviceptr_t>(gpu_src.opaque()), size, gpu_stream());
}

absl::Status RocmStream::Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) {
  return AsynchronousMemcpyH2D(
      executor_->gpu_context(),
      absl::bit_cast<hipDeviceptr_t>(gpu_dst->opaque()), host_src, size,
      gpu_stream());
}

absl::Status RocmStream::Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2H(executor_->gpu_context(), host_dst,
                               absl::bit_cast<hipDeviceptr_t>(gpu_src.opaque()),
                               size, gpu_stream());
}
}  // namespace stream_executor::gpu
