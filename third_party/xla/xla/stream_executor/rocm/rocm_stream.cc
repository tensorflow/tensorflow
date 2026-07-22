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
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_kernel.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {

// ---------------------------------------------------------------------------
// Process-level HIP stream handle cache
//
// hipStreamCreate on ROCm is expensive (~100 ms/stream). Instead of calling
// hipStreamDestroy in RocmStream::~RocmStream() and hipStreamCreate in
// RocmStream::Create(), idle handles are kept in this cache and reused.
//
// Cache key: (device_ordinal, creation_flags, creation_priority_int).
// Flags and priority are queried from the live stream via hipStreamGetFlags /
// hipStreamGetPriority before insertion, so a retrieved handle is guaranteed
// to match the creation parameters of the new stream.
//
// Safety invariants:
//   - A handle is inserted only after BlockHostUntilDone() confirms it is
//     fully idle (hipStreamSynchronize ran in RocmStream::~RocmStream()).
//   - hipStreamQuery is re-checked at insertion time; any stream in an error
//     state is destroyed rather than cached.
//   - The GPU context is activated (executor->Activate()) for all HIP calls.
//
// Intentional no-destructor: the singleton holds at most one vector of handles
// per (device, flags, priority) for the process lifetime.  absl::NoDestructor
// avoids running the destructor at program exit (which would call
// hipStreamDestroy after the driver may already be torn down).
//
// Thread safety: mu guards all map accesses.
struct HipStreamHandleCache {
  absl::Mutex mu;
  // Key: (device_ordinal, flags, priority_int)
  absl::btree_map<std::tuple<int, unsigned int, int>, std::vector<hipStream_t>>
      handles ABSL_GUARDED_BY(mu);
};

HipStreamHandleCache& GetHipStreamHandleCache() {
  static absl::NoDestructor<HipStreamHandleCache> cache;
  return *cache;
}

absl::StatusOr<hipStream_t> CreateStream(StreamExecutor* executor,
                                         int priority) {
  // XLA always creates streams with hipStreamDefault. This constant is used
  // as part of the cache key so that if the flag changes in the future the
  // cache will not silently return a handle with the wrong synchronisation
  // semantics.
  constexpr unsigned int kFlags = hipStreamDefault;

  // Check the cache for an idle handle with matching (device, flags, priority).
  {
    auto& cache = GetHipStreamHandleCache();
    absl::MutexLock lock(cache.mu);
    auto key = std::make_tuple(executor->device_ordinal(), kFlags, priority);
    auto it = cache.handles.find(key);
    if (it != cache.handles.end() && !it->second.empty()) {
      hipStream_t h = it->second.back();
      it->second.pop_back();
      VLOG(2) << "Reusing cached HIP stream " << h << " for device "
              << executor->device_ordinal() << " flags=" << kFlags
              << " priority=" << priority;
      return h;
    }
  }

  // Cold path: create a new HIP stream.
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  hipStream_t stream;
  if (priority == 0) {
    RETURN_IF_ERROR(ToStatus(
        hipStreamCreateWithFlags(&stream, hipStreamDefault),
        "Failed to create stream"));  // switch to hipStreamNonBlocking?
  } else {
    RETURN_IF_ERROR(ToStatus(
        hipStreamCreateWithPriority(&stream, hipStreamDefault, priority),
        "Failed to create stream"));  // switch to hipStreamNonBlocking?
  }

  VLOG(2) << "successfully created stream " << stream << " for device "
          << executor->device_ordinal() << " on thread";
  return stream;
}

absl::Status RecordEvent(StreamExecutor* executor, hipEvent_t event,
                         hipStream_t stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  hipError_t res = hipEventRecord(event, stream);
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
  RETURN_IF_ERROR(ToStatus(hipStreamWaitEvent(stream, event, 0 /* = flags */),
                           "could not wait stream on event"));
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2H(StreamExecutor* executor, void* host_dst,
                                   hipDeviceptr_t gpu_src, uint64_t size,
                                   hipStream_t stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  RETURN_IF_ERROR(ToStatus(
      hipMemcpyDtoHAsync(host_dst, gpu_src, size, stream),
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
  RETURN_IF_ERROR(ToStatus(
      hipMemcpyHtoDAsync(gpu_dst, const_cast<void*>(host_src), size, stream),
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
  RETURN_IF_ERROR(ToStatus(
      hipMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream),
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
  RETURN_IF_ERROR(ToStatus(hipStreamSynchronize(stream),
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
    return executor->GetGpuStreamPriority(
        std::get<StreamPriority>(priority.value_or(StreamPriority::Default)));
  }();
  ASSIGN_OR_RETURN(auto stream_handle, CreateStream(executor, stream_priority));

  ASSIGN_OR_RETURN(auto completed_event,
                   RocmEvent::Create(executor,
                                     /*allow_timing=*/false));

  return std::unique_ptr<RocmStream>(new RocmStream(
      executor, std::move(completed_event), priority, stream_handle));
}

absl::Status RocmStream::WaitFor(Stream* other) {
  RocmStream* other_stream = static_cast<RocmStream*>(other);

  RETURN_IF_ERROR(other_stream->RecordCompletedEvent());

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

  // Activate the device context for all HIP calls below.
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  // Verify the stream is fully idle before caching.
  // BlockHostUntilDone() ran in ~RocmStream() before this call, so under
  // normal conditions this query always succeeds. An error here indicates a
  // GPU fault or driver issue; destroy the stream rather than poisoning the
  // cache with a broken handle.
  hipError_t query_res = hipStreamQuery(stream);
  if (query_res != hipSuccess) {
    LOG(WARNING) << "stream not idle on destroy: " << ToString(query_res)
                 << " — destroying instead of caching";
    hipError_t res = hipStreamDestroy(stream);
    if (res != hipSuccess) {
      LOG(ERROR) << "failed to destroy ROCM stream for device "
                 << executor->device_ordinal() << ": " << ToString(res);
    }
    return;
  }

  // Query the stream's creation flags and priority so they can be used as the
  // cache key. This guarantees a retrieved handle always matches the exact
  // (flags, priority) the new stream would have been created with — even if
  // XLA is changed to use hipStreamNonBlocking or non-default priorities.
  unsigned int flags = 0;
  int stream_priority = 0;
  hipError_t flags_res = hipStreamGetFlags(stream, &flags);
  if (flags_res != hipSuccess) {
    LOG(WARNING) << "hipStreamGetFlags failed: " << ToString(flags_res)
                 << " — destroying stream " << stream << " instead of caching";
    hipError_t destroy_res = hipStreamDestroy(stream);
    if (destroy_res != hipSuccess) {
      LOG(ERROR) << "failed to destroy ROCM stream for device "
                 << executor->device_ordinal() << ": " << ToString(destroy_res);
    }
    return;
  }
  hipError_t prio_res = hipStreamGetPriority(stream, &stream_priority);
  if (prio_res != hipSuccess) {
    LOG(WARNING) << "hipStreamGetPriority failed: " << ToString(prio_res)
                 << " — destroying stream " << stream << " instead of caching";
    hipError_t destroy_res = hipStreamDestroy(stream);
    if (destroy_res != hipSuccess) {
      LOG(ERROR) << "failed to destroy ROCM stream for device "
                 << executor->device_ordinal() << ": " << ToString(destroy_res);
    }
    return;
  }

  // Insert the verified, idle handle into the cache for reuse.
  auto& cache = GetHipStreamHandleCache();
  absl::MutexLock lock(cache.mu);
  auto key =
      std::make_tuple(executor->device_ordinal(), flags, stream_priority);
  cache.handles[key].push_back(stream);
  VLOG(2) << "cached HIP stream " << stream << " for device "
          << executor->device_ordinal() << " flags=" << flags
          << " priority=" << stream_priority;
}
}  // namespace

RocmStream::~RocmStream() {
  BlockHostUntilDone().IgnoreError();
  executor_->DeallocateStream(this);

  DestroyStream(executor_, stream_handle_);
}

absl::Status RocmStream::Memset32(DeviceAddressBase* location, uint32_t pattern,
                                  uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) != 0) {
    return absl::InvalidArgumentError("location must be 4 byte aligned.");
  }
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("size must be a multiple of 4 bytes.");
  }
  return ToStatus(
      hipMemsetD32Async(location->opaque(), pattern, size / 4, stream_handle_),
      "Failed to memset memory");
}

absl::Status RocmStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return Memset32(location, 0x0, size);
  } else {
    std::unique_ptr<ActivateContext> activation = executor_->Activate();
    return ToStatus(
        hipMemsetAsync(location->opaque(), 0x0, size, stream_handle_),
        "Failed to enqueue async memset operation");
  }
}

absl::Status RocmStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2D(
      executor_, absl::bit_cast<hipDeviceptr_t>(gpu_dst->opaque()),
      absl::bit_cast<hipDeviceptr_t>(gpu_src.opaque()), size, stream_handle_);
}

absl::Status RocmStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const void* host_src, uint64_t size) {
  return AsynchronousMemcpyH2D(
      executor_, absl::bit_cast<hipDeviceptr_t>(gpu_dst->opaque()), host_src,
      size, stream_handle_);
}

absl::Status RocmStream::Memcpy(void* host_dst,
                                const DeviceAddressBase& gpu_src,
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
      hipLaunchHostFunc(stream_handle_, (hipHostFn_t)InternalHostCallback,
                        callback_ptr),
      "unable to add host callback");
}

namespace {
absl::Status LaunchRocmKernel(
    StreamExecutor* executor, absl::string_view kernel_name,
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes, hipStream_t stream,
    void** kernel_params, void** extra) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << " smem: " << shared_mem_bytes
          << " func: " << (const void*)function;

  auto res = hipModuleLaunchKernel(
      function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
      block_dim_z, shared_mem_bytes, stream, kernel_params, extra);
  RETURN_IF_ERROR(ToStatus(
      res, absl::StrCat("Failed to launch ROCm kernel: ", kernel_name,
                        "; grid: ", grid_dim_x, "x", grid_dim_y, "x",
                        grid_dim_z, "; block: ", block_dim_x, "x", block_dim_y,
                        "x", block_dim_z, "; shared_mem: ", shared_mem_bytes)));

  VLOG(2) << "successfully launched kernel";
  return absl::OkStatus();
}

absl::Status LaunchRocmKernel(
    StreamExecutor* executor, absl::string_view kernel_name,
    hipFunction_t function, unsigned int cluster_dim_x,
    unsigned int cluster_dim_y, unsigned int cluster_dim_z,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes, hipStream_t stream,
    void** kernel_params, void** extra) {
  if (cluster_dim_x != 1 || cluster_dim_y != 1 || cluster_dim_z != 1)
    return absl::UnimplementedError("Not implemented for ROCm");
  return LaunchRocmKernel(executor, kernel_name, function, grid_dim_x,
                          grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
                          block_dim_z, shared_mem_bytes, stream, kernel_params,
                          extra);
}

}  // namespace

absl::Status RocmStream::BlockHostUntilDone() {
  return SynchronizeStream(executor_, stream_handle_);
}

absl::Status RocmStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes, bool use_pdl) {
  if (cluster_dims.has_value()) {
    return LaunchRocmKernel(
        executor_, name, static_cast<hipFunction_t>(function), cluster_dims->x,
        cluster_dims->y, cluster_dims->z, block_dims.x, block_dims.y,
        block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z, shmem_bytes,
        stream_handle_, args,
        /*extra=*/nullptr);
  } else {
    return LaunchRocmKernel(
        executor_, name, static_cast<hipFunction_t>(function), block_dims.x,
        block_dims.y, block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
        shmem_bytes, stream_handle_, args,
        /*extra=*/nullptr);
  }
}

}  // namespace stream_executor::gpu
