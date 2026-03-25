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

#include "xla/stream_executor/cuda/cuda_stream.h"

#include <stdalign.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/host_callback_registry.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;
using tsl::profiler::TraceMeLevel;

namespace stream_executor {
namespace gpu {

namespace {
absl::Status WaitStreamOnEvent(StreamExecutor* executor, CUstream stream,
                               CUevent event) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  return cuda::ToStatus(cuStreamWaitEvent(stream, event, 0 /* = flags */));
}

absl::Status RecordGpuEvent(StreamExecutor* executor, CUevent event,
                            CUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  return cuda::ToStatus(cuEventRecord(event, stream),
                        "Error recording CUDA event");
}

absl::StatusOr<CUstream> CreateStream(StreamExecutor* executor, int priority) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  CUstream stream;
  // If the priority is 0, then use the previous api to create the stream with
  // the default priority for backward compatibility. Probably there is no
  // difference in using the new api call but leaving it as is for now.
  if (priority == 0) {
    TF_RETURN_IF_ERROR(
        cuda::ToStatus(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING)));
  } else {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, priority)));
  }

  VLOG(2) << "successfully created stream " << stream << " for executor "
          << executor << " on thread";
  return stream;
}

absl::StatusOr<bool> StreamIsCapturing(CUstream stream) {
  VLOG(2) << "Checking if stream " << stream << " is capturing";

  CUstreamCaptureStatus status;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuStreamIsCapturing(stream, &status),
                                    "Failed to check stream capturing status"));

  return status == CU_STREAM_CAPTURE_STATUS_ACTIVE;
}

absl::Status AsynchronousMemcpyD2H(StreamExecutor* executor, void* host_dst,
                                   CUdeviceptr gpu_src, uint64_t size,
                                   CUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMemcpyDtoHAsync(host_dst, gpu_src, size, stream)));

  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyH2D(StreamExecutor* executor,
                                   CUdeviceptr gpu_dst, const void* host_src,
                                   uint64_t size, CUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMemcpyHtoDAsync(gpu_dst, host_src, size, stream)));

  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " from " << host_src << " to " << absl::bit_cast<void*>(gpu_dst)
          << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2D(StreamExecutor* executor,
                                   CUdeviceptr gpu_dst, CUdeviceptr gpu_src,
                                   uint64_t size, CUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  // In graph capture mode we never have operations that access peer memory, so
  // we can always make a call to cuMemcpyDtoDAsync.
  TF_ASSIGN_OR_RETURN(bool is_capturing, StreamIsCapturing(stream));

  if ((gpu_dst == 0 || gpu_src == 0) || is_capturing) {
    // GetContextMap()->GetAnyContext() doesn't work when ptr == 0.
    // This happens when the size is 0.
    TF_RETURN_IF_ERROR(
        cuda::ToStatus(cuMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream)));
  } else {
    // Any context work here.
    CUcontext dst_context = CudaContext::GetContextMap()->GetAnyContext(
        absl::bit_cast<void*>(gpu_dst));
    CUcontext src_context = CudaContext::GetContextMap()->GetAnyContext(
        absl::bit_cast<void*>(gpu_src));

    if (dst_context == src_context) {
      // Since the CUDA context is the same, the src and dst are within the same
      // GPU. So we can use cuMemcpyDtoD.
      TF_RETURN_IF_ERROR(
          cuda::ToStatus(cuMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream)));
    } else {
      TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemcpyPeerAsync(
          gpu_dst, dst_context, gpu_src, src_context, size, stream)));
    }
  }

  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes"
          << " from " << absl::bit_cast<void*>(gpu_src) << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status SynchronizeStream(StreamExecutor* executor, CUstream stream) {
  TraceMe trace(
      [] { return TraceMeEncode("CudaStream::SynchronizeStream", {}); },
      /*level=*/TraceMeLevel::kVerbose);
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  CHECK(stream != nullptr);
  return cuda::ToStatus(cuStreamSynchronize(stream),
                        "Could not synchronize CUDA stream");
}

}  // namespace

CudaStream::CudaStream(
    CudaExecutor* executor, CudaEvent completed_event,
    std::optional<std::variant<StreamPriority, int>> priority,
    CUstream stream_handle)
    : StreamCommon(executor, priority),
      executor_(executor),
      completed_event_(std::move(completed_event)),
      stream_handle_(stream_handle),
      callback_registry_handle_(
          executor->GetHostCallbackRegistry()->CreateHandle(
              /*synchronization_callback=*/
              [this] { return SynchronizeStream(executor_, stream_handle_); },
              /*status_callback=*/[this] { return RefreshStatus(); })) {}

absl::StatusOr<std::unique_ptr<CudaStream>> CudaStream::Create(
    CudaExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  int stream_priority = [&]() {
    if (priority.has_value() && std::holds_alternative<int>(priority.value())) {
      return std::get<int>(priority.value());
    }
    std::unique_ptr<ActivateContext> activation = executor->Activate();
    return executor->GetGpuStreamPriority(
        std::get<StreamPriority>(priority.value_or(StreamPriority::Default)));
  }();
  TF_ASSIGN_OR_RETURN(auto stream_handle,
                      CreateStream(executor, stream_priority));

  TF_ASSIGN_OR_RETURN(auto completed_event,
                      CudaEvent::Create(executor,
                                        /*allow_timing=*/false));

  return std::unique_ptr<CudaStream>(new CudaStream(
      executor, std::move(completed_event), priority, stream_handle));
}

absl::Status CudaStream::WaitFor(Stream* other) {
  CudaStream* other_stream = static_cast<CudaStream*>(other);

  TF_RETURN_IF_ERROR(other_stream->RecordCompletedEvent());
  return WaitStreamOnEvent(executor_, stream_handle_,
                           other_stream->completed_event_.GetHandle());
}

absl::Status CudaStream::RecordEvent(Event* event) {
  return RecordGpuEvent(executor_, static_cast<CudaEvent*>(event)->GetHandle(),
                        stream_handle_);
}

absl::Status CudaStream::WaitFor(Event* event) {
  return WaitStreamOnEvent(executor_, stream_handle_,
                           static_cast<CudaEvent*>(event)->GetHandle());
}

absl::Status CudaStream::RecordCompletedEvent() {
  return RecordEvent(&completed_event_);
}

namespace {
void DestroyStream(StreamExecutor* executor, CUstream stream) {
  if (stream == nullptr) {
    return;
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  CUresult res = cuStreamQuery(stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "stream not idle on destroy: " << cuda::ToStatus(res);
  }

  auto status = cuda::ToStatus(cuStreamDestroy(stream));
  if (!status.ok()) {
    LOG(ERROR) << "failed to destroy CUDA stream for executor " << executor
               << ": " << status;
  } else {
    VLOG(2) << "successfully destroyed stream " << stream << " for executor "
            << executor;
  }
}

}  // namespace

CudaStream::~CudaStream() {
  // Ensure that all pending host callbacks are handled before the stream is
  // destroyed.
  callback_registry_handle_.reset();
  executor_->DeallocateStream(this);
  DestroyStream(executor_, stream_handle_);
}

absl::Status CudaStream::BlockHostUntilDone() {
  TraceMe trace(
      [] { return TraceMeEncode("CudaStream::BlockHostUntilDone", {}); },
      /*level=*/TraceMeLevel::kVerbose);
  // SynchronizeStream will wait for any pending host callbacks, but if the
  // stream is itself poisoned, it will fail without waiting. So we force fail
  // them to be called before returning.
  if (absl::Status status = SynchronizeStream(executor_, stream_handle_);
      !status.ok()) {
    callback_registry_handle_->FailAll(status);
    return status;
  }
  // TSAN complains of cuda host callbacks racing for data accessed
  // after BlockHostUntilDone even though they are synchronized. This
  // annotation establishes the happens-after relationship.
  (void)tsan_proxy_.load(std::memory_order_acquire);
  return absl::OkStatus();
}

absl::Status CudaStream::RefreshStatus() {
  TraceMe trace([] { return TraceMeEncode("CudaStream::RefreshStatus", {}); },
                /*level=*/TraceMeLevel::kVerbose);
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  const absl::StatusOr<bool> is_capturing = StreamIsCapturing(stream_handle_);
  // Stream querying is not allowed during graph capture.
  // Errors during `StreamIsCapturing` itself means we will use cuStreamQuery
  // after.
  if (is_capturing.ok() && *is_capturing) {
    return absl::OkStatus();
  }
  CUresult res = cuStreamQuery(stream_handle_);
  if (res == CUDA_SUCCESS || res == CUDA_ERROR_NOT_READY) {
    return absl::OkStatus();
  }
  return cuda::ToStatus(res, "Error querying CUDA stream status");
}

absl::Status CudaStream::Memset32(DeviceAddressBase* location, uint32_t pattern,
                                  uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) != 0) {
    return absl::InvalidArgumentError("location must be 4 byte aligned.");
  }
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("size must be a multiple of 4 bytes.");
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return cuda::ToStatus(
      cuMemsetD32Async(absl::bit_cast<CUdeviceptr>(location->opaque()), pattern,
                       size / 4, stream_handle_),
      "Failed to enqueue async memset operation");
}

absl::Status CudaStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) ==
          0 &&
      size % sizeof(uint32_t) == 0) {
    return Memset32(location, 0x0, size);
  } else {
    std::unique_ptr<ActivateContext> activation = executor_->Activate();
    return cuda::ToStatus(
        cuMemsetD8Async(absl::bit_cast<CUdeviceptr>(location->opaque()), 0x0,
                        size, stream_handle_),
        "Failed to enqueue async memset operation");
  }
}

absl::Status CudaStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2D(
      executor_, absl::bit_cast<CUdeviceptr>(gpu_dst->opaque()),
      absl::bit_cast<CUdeviceptr>(gpu_src.opaque()), size, stream_handle_);
}

absl::Status CudaStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const void* host_src, uint64_t size) {
  return AsynchronousMemcpyH2D(executor_,
                               absl::bit_cast<CUdeviceptr>(gpu_dst->opaque()),
                               host_src, size, stream_handle_);
}

absl::Status CudaStream::Memcpy(void* host_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2H(executor_, host_dst,
                               absl::bit_cast<CUdeviceptr>(gpu_src.opaque()),
                               size, stream_handle_);
}

absl::Status CudaStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  return DoHostCallbackWithStatus(std::move(callback), nullptr);
}

absl::Status CudaStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_cb) {
  auto enqueue_cb =
      [stream_handle = stream_handle_](
          HostCallbackRegistry::RegistryHandle::DeviceCb device_cb,
          void* data) -> absl::Status {
    return cuda::ToStatus(cuLaunchHostFunc(stream_handle, device_cb, data));
  };
  const auto annotate_cb = [this](auto&& cb) {
    return [this, cb = std::forward<decltype(cb)>(cb)](auto&&... args) mutable {
      // TSAN complains of cuda host callbacks racing for data accessed
      // after BlockHostUntilDone even though they are  synchronized. This
      // annotation establishes the happens-before relationship.
      auto cleanup = absl::MakeCleanup(
          [this] { tsan_proxy_.fetch_add(1, std::memory_order_release); });
      return std::forward<decltype(cb)>(cb)(
          std::forward<decltype(args)>(args)...);
    };
  };
  return callback_registry_handle_->AddCallback(
      annotate_cb(std::move(callback)),
      error_cb ? annotate_cb(std::move(error_cb)) : std::move(error_cb),
      enqueue_cb);
}

namespace {
absl::Status LaunchCudaKernel(
    StreamExecutor* executor, absl::string_view kernel_name,
    CUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes, CUstream stream,
    void** kernel_params, void** extra, std::optional<ClusterDim> cluster_dims,
    bool use_pdl) {
  TraceMe trace0([] { return TraceMeEncode("LaunchCudaKernel", {}); },
                 /*level=*/TraceMeLevel::kVerbose);

  std::unique_ptr<ActivateContext> activation = executor->Activate();

  if (VLOG_IS_ON(2)) {
    std::string msg = absl::StrCat("launching kernel: ", kernel_name);
    if (cluster_dims.has_value()) {
      absl::StrAppend(&msg, "; cdx: ", cluster_dims->x,
                      " cdy: ", cluster_dims->y, " cdz: ", cluster_dims->z);
    }
    absl::StrAppend(&msg, "; gdx: ", grid_dim_x, " gdy: ", grid_dim_y,
                    " gdz: ", grid_dim_z, " bdx: ", block_dim_x,
                    " bdy: ", block_dim_y, " bdz: ", block_dim_z,
                    "; shared_mem_bytes: ", shared_mem_bytes);
    VLOG(2) << msg;
  }

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_SHARED)));
  }

  auto to_status = [&](CUresult result) {
    if (result == CUDA_SUCCESS) {
      return absl::OkStatus();
    }
    std::string msg =
        absl::StrCat("Failed to launch CUDA kernel: ", kernel_name);
    if (cluster_dims.has_value()) {
      absl::StrAppend(&msg, "; cluster dims: ", cluster_dims->x, "x",
                      cluster_dims->y, "x", cluster_dims->z);
    }
    absl::StrAppend(&msg, "; block dims: ", block_dim_x, "x", block_dim_y, "x",
                    block_dim_z, "; grid dims: ", grid_dim_x, "x", grid_dim_y,
                    "x", grid_dim_z,
                    "; shared memory size: ", shared_mem_bytes);
    return cuda::ToStatus(result, msg);
  };

  if (cluster_dims.has_value() || use_pdl) {
    CUlaunchConfig launch_config;
    memset(&launch_config, 0, sizeof(launch_config));
    launch_config.blockDimX = block_dim_x;
    launch_config.blockDimY = block_dim_y;
    launch_config.blockDimZ = block_dim_z;
    launch_config.gridDimX = grid_dim_x;
    launch_config.gridDimY = grid_dim_y;
    launch_config.gridDimZ = grid_dim_z;
    launch_config.hStream = stream;
    launch_config.sharedMemBytes = shared_mem_bytes;

    absl::InlinedVector<CUlaunchAttribute, 2> attrs;

    if (cluster_dims.has_value()) {
      CUlaunchAttribute attr{};
      attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attr.value.clusterDim.x = static_cast<unsigned int>(cluster_dims->x);
      attr.value.clusterDim.y = static_cast<unsigned int>(cluster_dims->y);
      attr.value.clusterDim.z = static_cast<unsigned int>(cluster_dims->z);
      attrs.push_back(attr);
    }

    if (use_pdl) {
      CUlaunchAttribute attr{};
      attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attr.value.programmaticStreamSerializationAllowed = 1;
      attrs.push_back(attr);
    }

    launch_config.attrs = attrs.data();
    launch_config.numAttrs = attrs.size();

    TraceMe trace(
        [] { return TraceMeEncode("LaunchCudaKernel/cuLaunchKernelEx", {}); },
        /*level=*/TraceMeLevel::kVerbose);
    return to_status(
        cuLaunchKernelEx(&launch_config, function, kernel_params, extra));
  }

  TraceMe trace(
      [&] { return TraceMeEncode("LaunchCudaKernel/cuLaunchKernel", {}); },
      /*level=*/TraceMeLevel::kVerbose);
  return to_status(cuLaunchKernel(
      function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
      block_dim_z, shared_mem_bytes, stream, kernel_params, extra));
}

}  // namespace

absl::Status CudaStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes, bool use_pdl) {
  TraceMe trace([] { return TraceMeEncode("CudaStream::LaunchKernel", {}); },
                /*level=*/TraceMeLevel::kVerbose);

  return LaunchCudaKernel(executor_, name, static_cast<CUfunction>(function),
                          block_dims.x, block_dims.y, block_dims.z,
                          thread_dims.x, thread_dims.y, thread_dims.z,
                          shmem_bytes, stream_handle_, args,
                          /*extra=*/nullptr, cluster_dims, use_pdl);
}

void CudaStream::SetName(std::string name) {
  tsl::profiler::NameStream(
      absl::bit_cast<tsl::profiler::StreamHandle>(stream_handle_), name);
  StreamCommon::SetName(std::move(name));
}

}  // namespace gpu
}  // namespace stream_executor
