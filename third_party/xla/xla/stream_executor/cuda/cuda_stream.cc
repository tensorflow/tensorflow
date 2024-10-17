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

#include <cstdint>
#include <cstring>
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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/cuda/cuda_kernel.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/nvtx_utils.h"

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

int GetGpuStreamPriority(stream_executor::StreamPriority stream_priority) {
  if (stream_priority == stream_executor::StreamPriority::Default) {
    return 0;
  }
  int lowest, highest;
  auto status = cuda::ToStatus(cuCtxGetStreamPriorityRange(&lowest, &highest));
  if (!status.ok()) {
    LOG(ERROR)
        << "Could not query stream priority range. Returning default priority.";
    return 0;
  }
  return stream_priority == stream_executor::StreamPriority::Highest ? highest
                                                                     : lowest;
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

}  // namespace

absl::StatusOr<std::unique_ptr<CudaStream>> CudaStream::Create(
    StreamExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  int stream_priority = [&]() {
    if (priority.has_value() && std::holds_alternative<int>(priority.value())) {
      return std::get<int>(priority.value());
    }
    std::unique_ptr<ActivateContext> activation = executor->Activate();
    return GetGpuStreamPriority(
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

absl::Status SynchronizeStream(StreamExecutor* executor, CUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  CHECK(stream != nullptr);
  return cuda::ToStatus(cuStreamSynchronize(stream),
                        "Could not synchronize CUDA stream");
}
}  // namespace

CudaStream::~CudaStream() {
  BlockHostUntilDone().IgnoreError();
  executor_->DeallocateStream(this);

  DestroyStream(executor_, stream_handle_);
}

absl::Status CudaStream::BlockHostUntilDone() {
  return SynchronizeStream(executor_, stream_handle_);
}

absl::Status CudaStream::Memset32(DeviceMemoryBase* location, uint32_t pattern,
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

absl::Status CudaStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
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

absl::Status CudaStream::Memcpy(DeviceMemoryBase* gpu_dst,
                                const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2D(
      executor_, absl::bit_cast<CUdeviceptr>(gpu_dst->opaque()),
      absl::bit_cast<CUdeviceptr>(gpu_src.opaque()), size, stream_handle_);
}

absl::Status CudaStream::Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) {
  return AsynchronousMemcpyH2D(executor_,
                               absl::bit_cast<CUdeviceptr>(gpu_dst->opaque()),
                               host_src, size, stream_handle_);
}

absl::Status CudaStream::Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2H(executor_, host_dst,
                               absl::bit_cast<CUdeviceptr>(gpu_src.opaque()),
                               size, stream_handle_);
}

namespace {
void InternalHostCallback(void* data) {
  auto* callback = reinterpret_cast<absl::AnyInvocable<void() &&>*>(data);
  std::move (*callback)();
  delete callback;
}
}  // namespace

absl::Status CudaStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  auto callback_ptr =
      new absl::AnyInvocable<void() &&>([cb = std::move(callback)]() mutable {
        absl::Status s = (std::move(cb))();
        if (!s.ok()) {
          LOG(WARNING) << "Host callback failed: " << s;
        }
      });
  return cuda::ToStatus(
      cuLaunchHostFunc(stream_handle_, InternalHostCallback, callback_ptr));
}

namespace {
absl::Status LaunchKernel(StreamExecutor* executor,
                          absl::string_view kernel_name, CUfunction function,
                          unsigned int grid_dim_x, unsigned int grid_dim_y,
                          unsigned int grid_dim_z, unsigned int block_dim_x,
                          unsigned int block_dim_y, unsigned int block_dim_z,
                          unsigned int shared_mem_bytes, CUstream stream,
                          void** kernel_params, void** extra) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z
          << "; shared_mem_bytes: " << shared_mem_bytes;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return cuda::ToStatus(
      cuLaunchKernel(function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                     block_dim_y, block_dim_z, shared_mem_bytes, stream,
                     kernel_params, extra),
      absl::StrCat("Failed to launch CUDA kernel: ", kernel_name,
                   "; block dims: ", block_dim_x, "x", block_dim_y, "x",
                   block_dim_z, "; grid dims: ", grid_dim_x, "x", grid_dim_y,
                   "x", grid_dim_z,
                   "; shared memory size: ", shared_mem_bytes));
}

absl::Status LaunchKernel(StreamExecutor* executor,
                          absl::string_view kernel_name, CUfunction function,
                          unsigned int cluster_dim_x,
                          unsigned int cluster_dim_y,
                          unsigned int cluster_dim_z, unsigned int grid_dim_x,
                          unsigned int grid_dim_y, unsigned int grid_dim_z,
                          unsigned int block_dim_x, unsigned int block_dim_y,
                          unsigned int block_dim_z,
                          unsigned int shared_mem_bytes, CUstream stream,
                          void** kernel_params, void** extra) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  VLOG(2) << "launching kernel: " << kernel_name << "; cdx: " << cluster_dim_x
          << " cdy: " << cluster_dim_y << " cdz: " << cluster_dim_z
          << " gdx: " << grid_dim_x << " gdy: " << grid_dim_y
          << " gdz: " << grid_dim_z << " bdx: " << block_dim_x
          << " bdy: " << block_dim_y << " bdz: " << block_dim_z
          << "; shared_mem_bytes: " << shared_mem_bytes;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

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

  CUlaunchAttribute cluster_dims;
  memset(&cluster_dims, 0, sizeof(cluster_dims));
  cluster_dims.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  cluster_dims.value.clusterDim.x = cluster_dim_x;
  cluster_dims.value.clusterDim.y = cluster_dim_y;
  cluster_dims.value.clusterDim.z = cluster_dim_z;

  launch_config.attrs = &cluster_dims;
  launch_config.numAttrs = 1;

  return cuda::ToStatus(
      cuLaunchKernelEx(&launch_config, function, kernel_params, extra),
      absl::StrCat("Failed to launch CUDA kernel: ", kernel_name,
                   "; cluster dims: ", cluster_dim_x, "x", cluster_dim_y, "x",
                   cluster_dim_z, "; block dims: ", block_dim_x, "x",
                   block_dim_y, "x", block_dim_z, "; grid dims: ", grid_dim_x,
                   "x", grid_dim_y, "x", grid_dim_z,
                   "; shared memory size: ", shared_mem_bytes));
}

}  // namespace

absl::Status CudaStream::Launch(const ThreadDim& thread_dims,
                                const BlockDim& block_dims,
                                const std::optional<ClusterDim>& cluster_dims,
                                const Kernel& kernel, const KernelArgs& args) {
  const CudaKernel* gpu_kernel = static_cast<const CudaKernel*>(&kernel);
  CUfunction function = gpu_kernel->gpu_function();

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

void CudaStream::SetName(std::string name) {
  tsl::profiler::NameStream(
      absl::bit_cast<tsl::profiler::StreamHandle>(stream_handle_), name);
  StreamCommon::SetName(std::move(name));
}

}  // namespace gpu
}  // namespace stream_executor
