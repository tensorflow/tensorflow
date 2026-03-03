/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_executor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/gpus/cuda/nvml/include/nvml.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_command_buffer.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_core_info_table.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/cuda/cuda_kernel.h"
#include "xla/stream_executor/cuda/cuda_memory_allocator.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/cuda_stream.h"
#include "xla/stream_executor/cuda/cuda_timer.h"
#include "xla/stream_executor/cuda/cuda_version_parser.h"
#include "xla/stream_executor/cuda/cudnn_api_wrappers.h"
#include "xla/stream_executor/cuda/tma_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/gpu/read_numa_node.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tensor_map.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/numbers.h"

namespace stream_executor {
namespace gpu {

namespace {
bool ShouldLaunchDelayKernel() {
  // Only launch the delay kernel if CUDA_LAUNCH_BLOCKING is not set to 1.
  static bool value = [] {
    const char* blocking = std::getenv("CUDA_LAUNCH_BLOCKING");
    return !blocking || absl::string_view{blocking} != "1";
  }();
  return value;
}

// CUDA driver routines may require a large amount of stack (particularly
// cuModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
tsl::thread::ThreadPool* GetDriverExecutor() {
  static tsl::thread::ThreadPool* const thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "cuda_driver", 1);
  return thread_pool;
}

// Loads ptx_contents with the CUDA driver's PTX JIT and return the resulting
// handle. Any error logs that are produced are logged internally.
absl::StatusOr<CUmodule> LoadPtx(Context* context, const char* ptx_contents) {
  absl::Notification notification;
  absl::Status returned_status = absl::OkStatus();
  CUmodule module;
  GetDriverExecutor()->Schedule(
      [context, ptx_contents, &module, &returned_status, &notification]() {
        ScopedActivateContext activation(context);
        void* ptx_data = const_cast<char*>(ptx_contents);
        static const unsigned int kLogBufferBytesLimit = 1024;
        unsigned int error_log_buffer_bytes = kLogBufferBytesLimit;
        unsigned int info_log_buffer_bytes = kLogBufferBytesLimit;
        absl::InlinedVector<char, 4> error_log_buffer(error_log_buffer_bytes);
        absl::InlinedVector<char, 4> info_log_buffer(info_log_buffer_bytes);
        bool log_verbose = true;
        CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                  CU_JIT_ERROR_LOG_BUFFER,
                                  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                                  CU_JIT_INFO_LOG_BUFFER, CU_JIT_LOG_VERBOSE};
        // Note that the driver API wants the contents of this values to be
        // stored in an array of void*s, so we coerce them accordingly.
        void* option_values[] = {
            absl::bit_cast<void*>(uintptr_t(error_log_buffer_bytes)),
            absl::bit_cast<void*>(error_log_buffer.data()),
            absl::bit_cast<void*>(uintptr_t(info_log_buffer_bytes)),
            absl::bit_cast<void*>(info_log_buffer.data()),
            absl::bit_cast<void*>(uintptr_t(log_verbose))};
        CHECK(TF_ARRAYSIZE(options) == TF_ARRAYSIZE(option_values));

        absl::Status status;
        status = cuda::ToStatus(cuModuleLoadDataEx(
            &module, ptx_data, TF_ARRAYSIZE(options), options, option_values));

        // The PTX JIT mutates the values in the option values array to reflect
        // the size of the logs it output; now that we've made the call, read
        // the values back out.
        error_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[0]);
        info_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[2]);
        CHECK_LE(error_log_buffer_bytes, kLogBufferBytesLimit);
        CHECK_LE(info_log_buffer_bytes, kLogBufferBytesLimit);

        if (!status.ok()) {
          XLA_LOG_DEVICE(ERROR, context->device_ordinal())
              << "failed to load PTX text as a module: " << status;
          // As a precaution for null termination of the API-provided value,
          // ensure that at least the last byte is null.
          error_log_buffer[error_log_buffer_bytes ? error_log_buffer_bytes - 1
                                                  : 0] = '\0';
          XLA_LOG_DEVICE(ERROR, context->device_ordinal())
              << "error log buffer (" << error_log_buffer_bytes
              << " bytes): " << error_log_buffer.data();
          if (absl::StrContains(error_log_buffer.data(),
                                "Register allocation failed")) {
            returned_status = absl::ResourceExhaustedError(absl::StrFormat(
                "[%d] Failed to load PTX text as a module (register "
                "allocation failed): %s",
                context->device_ordinal(), status.ToString()));
          } else {
            returned_status = status;
          }
          notification.Notify();
          return;
        }

        XLA_VLOG_DEVICE(3, context->device_ordinal())
            << "PTX compilation info log (" << info_log_buffer_bytes
            << " bytes): " << info_log_buffer.data();
        XLA_VLOG_DEVICE(3, context->device_ordinal())
            << "PTX compilation error log (" << error_log_buffer_bytes
            << " bytes): " << error_log_buffer.data();
        CHECK(module != nullptr);
        notification.Notify();
      });
  notification.WaitForNotification();

  TF_RETURN_IF_ERROR(returned_status);
  return module;
}

// Loads cubin_bytes with the CUDA driver's blob loading interface and stores
// the resulting handle in "module".
absl::StatusOr<CUmodule> LoadCubin(Context* context, const char* cubin_bytes) {
  ScopedActivateContext activation(context);
  CUmodule module;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuModuleLoadFatBinary(&module, cubin_bytes),
      absl::StrCat(xla::XlaFormatDevice(context->device_ordinal()),
                   "Failed to load in-memory CUBIN "
                   "(compiled for a different GPU?).")));
  return module;
}

// Retrieves a named kernel from a loaded module, and return the CUfunction
// handle on success. Neither kernel_name nor function may be null. No ownership
// is taken of kernel_name.
absl::StatusOr<CUfunction> GetModuleFunction(Context* context, CUmodule module,
                                             const char* kernel_name) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  cudaError_t cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return absl::InternalError(
        absl::StrCat(xla::XlaFormatDevice(context->device_ordinal()),
                     "There was an error before calling cuModuleGetFunction (",
                     cuda_error, "): ", cudaGetErrorName(cuda_error), " : ",
                     cudaGetErrorString(cuda_error)));
  }
  CUfunction function;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuModuleGetFunction(&function, module, kernel_name),
      absl::StrCat(xla::XlaFormatDevice(context->device_ordinal()),
                   "Failed to get module function")));
  return function;
}

// Retrieves a named global/constant symbol from a loaded module, and returns
// a device pointer and size of the symbol on success. symbol_name may not be
// null. At least one of dptr or bytes should not be null. No ownership is
// taken of symbol_name.
absl::Status GetModuleSymbol(Context* context, CUmodule module,
                             const char* symbol_name, CUdeviceptr* dptr,
                             size_t* bytes) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  return cuda::ToStatus(
      cuModuleGetGlobal(dptr, bytes, module, symbol_name),
      absl::StrCat("Failed to get symbol '", symbol_name, "'"));
}

// Unloads module from the current context via cuModuleUnload.
void UnloadCudaModule(Context* context, CUmodule module) {
  ScopedActivateContext activated{context};
  auto status = cuda::ToStatus(cuModuleUnload(module));
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, context->device_ordinal())
        << "failed to unload module " << module << "; leaking: " << status;
  }
}

// Returns the integer output of cuDeviceGetAttribute.
absl::StatusOr<int> GetDeviceAttribute(CUdevice_attribute attribute,
                                       CUdevice device) {
  int val;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGetAttribute(&val, attribute, device)));
  return val;
}

// Returns the name of the device.
absl::StatusOr<std::string> GetDeviceName(CUdevice device) {
  static const size_t kCharLimit = 64;
  absl::InlinedVector<char, 4> chars(kCharLimit);
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGetName(chars.begin(), kCharLimit - 1, device),
                     "Failed to get device name"));
  chars[kCharLimit - 1] = '\0';
  return chars.begin();
}

// Returns the compute capability for the device; i.e (3, 5).
absl::StatusOr<CudaComputeCapability> GetComputeCapability(CUdevice device) {
  int cc_major = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDeviceGetAttribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)));

  int cc_minor = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDeviceGetAttribute(
      &cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)));

  bool has_accelerated_features = cc_major >= 9;
  return CudaComputeCapability(
      cc_major, cc_minor,
      has_accelerated_features
          ? CudaComputeCapability::FeatureExtension::kAcceleratedFeatures
          : CudaComputeCapability::FeatureExtension::kNone);
}

// Helper function that turns the integer output of cuDeviceGetAttribute to type
// T and wraps it in a absl::StatusOr.
template <typename T>
static absl::StatusOr<T> GetSimpleAttribute(CUdevice device,
                                            CUdevice_attribute attribute) {
  int value = -1;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&value, attribute, device),
      absl::StrCat("Could not retrieve CUDA device attribute (", attribute)));
  T converted = value;
  return converted;
}

// Returns the number of multiprocessors on the device (note that the device
// may be multi-GPU-per-board).
absl::StatusOr<int> GetMultiprocessorCount(CUdevice device) {
  return GetSimpleAttribute<int>(device,
                                 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
}

absl::StatusOr<int64_t> GetMaxSharedMemoryPerCore(CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
}

absl::StatusOr<int64_t> GetMaxSharedMemoryPerBlock(CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
}

absl::StatusOr<int64_t> GetMaxSharedMemoryPerBlockOptin(CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
}

absl::StatusOr<int64_t> GetMaxThreadsPerMultiprocessor(CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
}

absl::StatusOr<int64_t> GetMaxRegistersPerBlock(CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
}

absl::StatusOr<int64_t> GetThreadsPerWarp(CUdevice device) {
  return GetSimpleAttribute<int64_t>(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
}

absl::Status GetGridLimits(int* x, int* y, int* z, CUdevice device) {
  int value;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device),
      "Could not get device attribute"));
  *x = value;

  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device),
      "Could not get device attribute"));
  *y = value;

  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device),
      "Could not get device attribute"));
  *z = value;
  return absl::OkStatus();
}

// Returns the device associated with the given device_ordinal.
absl::StatusOr<CUdevice> GetDevice(int device_ordinal) {
  CUdevice device;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDeviceGet(&device, device_ordinal),
                                    "Failed call to cuDeviceGet"));
  return device;
}

// Returns the device associated with the given context.
absl::StatusOr<CUdevice> DeviceFromContext(Context* context) {
  ScopedActivateContext activated{context};
  CUdevice device = -1;
  auto status = cuda::ToStatus(cuCtxGetDevice(&device));
  if (status.ok()) {
    return device;
  }

  return status;
}

bool CanEnablePeerAccess(CUdevice from, CUdevice to) {
  int can_access_peer = -1;
  auto status =
      cuda::ToStatus(cuDeviceCanAccessPeer(&can_access_peer, from, to));
  if (!status.ok()) {
    LOG(ERROR) << "failed to detect peer access capability: " << status;
    return false;
  }

  return can_access_peer;
}

absl::Status EnablePeerAccess(Context* from, Context* to) {
  if (from == to) {
    return absl::OkStatus();  // A context can always access its own
                              // memory.
  }

  ScopedActivateContext activated{from};
  CUresult result = cuCtxEnablePeerAccess(
      tensorflow::down_cast<CudaContext*>(to)->context(), 0 /* = flags */);
  if (result != CUDA_SUCCESS &&
      result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return absl::InternalError(
        absl::StrFormat("failed to enable peer access from %p to %p: %s", from,
                        to, cuda::ToStatus(result).ToString()));
  }

  return absl::OkStatus();
}

// Returns the total amount of memory available on the device.
bool GetDeviceTotalMemory(CUdevice device, uint64_t* result) {
  size_t value{};
  auto status = cuda::ToStatus(cuDeviceTotalMem(&value, device));
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, device)
        << "failed to query total available memory: " << status;
    return false;
  }

  *result = value;
  return true;
}

bool IsEccEnabled(CUdevice device, bool* result) {
  int value = -1;
  auto status = cuda::ToStatus(
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device));
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, device) << "failed to query ECC status: " << status;
    return false;
  }

  *result = value;
  return true;
}

std::string GetPCIBusID(CUdevice device) {
  // PCI bus id is of the format [domain]:[bus]:[device].[function], and is 13
  // characters long in practice.
  constexpr int kBufferSize = 64;
  std::array<char, kBufferSize> raw_pci_bus_id{};
  absl::Status status = cuda::ToStatus(
      cuDeviceGetPCIBusId(raw_pci_bus_id.data(), kBufferSize, device));
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, device)
        << "failed to query PCI bus id for device: " << status;
    return "";
  }
  if (!absl::c_linear_search(raw_pci_bus_id, '\0')) {
    XLA_LOG_DEVICE(ERROR, device) << "PCI bus id is not null terminated.";
    return "";
  }
  // Lower the hex characters to match sysfs.
  return absl::AsciiStrToLower(absl::string_view(raw_pci_bus_id.data()));
}

bool HostRegister(Context* context, void* location, uint64_t size) {
  ScopedActivateContext activation(context);
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  auto status = cuda::ToStatus(
      cuMemHostRegister(location, size, CU_MEMHOSTREGISTER_PORTABLE));
  if (!status.ok()) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << status;
    return false;
  }
  return true;
}

bool HostUnregister(Context* context, void* location) {
  ScopedActivateContext activation(context);
  auto status = cuda::ToStatus(cuMemHostUnregister(location));
  if (!status.ok()) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << status;
    return false;
  }
  return true;
}

// Allocates memory on the GPU device.
void* DeviceAllocate(Context* context, uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  ScopedActivateContext activated{context};
  CUdeviceptr result = 0;
  auto status = cuda::ToStatus(cuMemAlloc(&result, bytes));
  if (!status.ok()) {
    // LOG(INFO) because this isn't always important to users (e.g. BFCAllocator
    // implements a retry if the first allocation fails).
    LOG(INFO) << "failed to allocate "
              << tsl::strings::HumanReadableNumBytes(bytes) << " (" << bytes
              << " bytes) from device: " << status;
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  XLA_VLOG_DEVICE(2, context->device_ordinal())
      << "allocated " << ptr << " for context " << context << " of " << bytes
      << " bytes";
  return ptr;
}

// Deallocates memory on the GPU device that was previously allocated via
// DeviceAllocate.
void DeviceDeallocate(Context* context, void* location) {
  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  auto status = cuda::ToStatus(cuMemFree(pointer));
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, context->device_ordinal())
        << "failed to free device memory at " << location
        << "; result: " << status;
  } else {
    XLA_VLOG_DEVICE(2, context->device_ordinal())
        << "deallocated " << location << " for context " << context;
  }
}

// Allocates memory on the host.
absl::StatusOr<void*> HostAllocate(Context* context, int numa_node,
                                   uint64_t size) {
  if (numa_node != tsl::port::kNUMANoAffinity) {
    // CUDA programming guide: "Any address of a variable ... returned by one
    // of the memory allocation routines from the driver ... API is always
    // aligned to at least 256 bytes."
    auto* buffer =
        tsl::port::NUMAMalloc(numa_node, size, /* minimum_alignment=*/256);
    if (buffer == nullptr && size > 0) {
      return absl::InternalError(absl::StrFormat(
          "%sFailed to allocate host memory of size %d "
          "pinned to NUMA node %d",
          xla::XlaFormatDevice(context->device_ordinal()), size, numa_node));
    }
    if (size > 0 && !HostRegister(context, buffer, size)) {
      tsl::port::NUMAFree(buffer, size);
      return absl::InternalError(absl::StrFormat(
          "%sFailed to register host memory of size %d pinned to "
          "NUMA node %d with the GPU driver",
          xla::XlaFormatDevice(context->device_ordinal()), size, numa_node));
    }
    return buffer;
  }

  ScopedActivateContext activation(context);
  void* buffer = nullptr;
  // "Portable" memory is visible to all CUDA contexts. Safe for our use
  // model.
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMemHostAlloc(&buffer, size, CU_MEMHOSTALLOC_PORTABLE)));
  if (!buffer && size > 0) {
    return absl::InternalError(
        absl::StrFormat("%sFailed to allocate pinned host memory of size %d",
                        xla::XlaFormatDevice(context->device_ordinal()), size));
  }
  return buffer;
}

// Deallocates memory allocated via HostAllocate.
void HostDeallocate(Context* context, int numa_node, void* location,
                    uint64_t size) {
  if (numa_node != tsl::port::kNUMANoAffinity) {
    if (size > 0) {
      HostUnregister(context, location);
    }
    tsl::port::NUMAFree(location, size);
  } else {
    ScopedActivateContext activation(context);
    auto status = cuda::ToStatus(cuMemFreeHost(location));
    if (!status.ok()) {
      XLA_LOG_DEVICE(ERROR, context->device_ordinal())
          << "error deallocating host memory at " << location << ": " << status;
    }
  }
}

// Creates a MemoryAllocation wrapping the given host buffer.
absl::StatusOr<std::unique_ptr<MemoryAllocation>> AllocateHostMemory(
    CudaContext* cuda_context, int numa_node, uint64_t size) {
  TF_ASSIGN_OR_RETURN(void* ptr, HostAllocate(cuda_context, numa_node, size));
  XLA_VLOG_DEVICE(2, cuda_context->device_ordinal())
      << "allocated " << ptr << " for context " << cuda_context << " of "
      << size << " bytes of host memory";
  return std::make_unique<GenericMemoryAllocation>(
      ptr, size, [cuda_context, numa_node](void* location, uint64_t size) {
        HostDeallocate(cuda_context, numa_node, location, size);
        XLA_VLOG_DEVICE(2, cuda_context->device_ordinal())
            << "deallocated collective memory at " << location
            << " for context " << cuda_context;
      });
}

absl::StatusOr<bool> IsVmmSupported(CUdevice device) {
  int deviceSupportsVmm = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDeviceGetAttribute(
      &deviceSupportsVmm,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device)));
  return deviceSupportsVmm;
}

absl::StatusOr<bool> IsRdmaSupported(CUdevice device) {
  int rdma_supported = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDeviceGetAttribute(
      &rdma_supported,
      CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device)));
  return rdma_supported;
}

absl::StatusOr<bool> IsMulticastSupported(CUdevice device) {
  int is_multicast_supported = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&is_multicast_supported,
                           CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device)));
  return is_multicast_supported;
}

CUmemAllocationProp GetVmmAllocationProperties(CUdevice device,
                                               bool is_rdma_supported) {
  CUmemAllocationProp properties = {};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.requestedHandleTypes =
      static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_NONE);
  properties.location.id = device;
  properties.allocFlags.gpuDirectRDMACapable = is_rdma_supported ? 1 : 0;
  return properties;
}

CUmemAccessDesc GetVmmAccessDescriptor(int device) {
  CUmemAccessDesc descriptor = {};
  descriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  descriptor.location.id = device;
  descriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return descriptor;
}

absl::StatusOr<CUmulticastObjectProp> CreateMulticastObjectProperties(
    int num_devices, size_t size) {
  CUmulticastObjectProp multicast_properties;
  memset(&multicast_properties, 0, sizeof(CUmulticastObjectProp));
  multicast_properties.numDevices = num_devices;

  multicast_properties.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  multicast_properties.flags = 0;

  size_t multicast_granularity = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMulticastGetGranularity(&multicast_granularity, &multicast_properties,
                                CU_MULTICAST_GRANULARITY_RECOMMENDED)));

  // Align up the size to the multicast granularity.
  multicast_properties.size =
      xla::RoundUpTo<size_t>(size, multicast_granularity);
  return multicast_properties;
}

absl::Status ToStatus(nvmlReturn_t result) {
  if (result == NVML_SUCCESS) {
    return absl::OkStatus();
  }
  // NVML library is not a part of the CUDA toolkit, so there might be a
  // situation when user is using newer CUDA, but the host NVML
  // version doen't have the required functions.
  if (result == NVML_ERROR_FUNCTION_NOT_FOUND) {
    return absl::InternalError("NVML library doesn't have required functions.");
  }
  return absl::InternalError(absl::StrFormat("Nvml call failed with %d(%s).",
                                             result, nvmlErrorString(result)));
}

// CUDA and Nvml can have different device ordering.
absl::StatusOr<nvmlDevice_t> GetNvmlDevice(const std::string& pci_bus_id) {
  nvmlDevice_t device;
  TF_RETURN_IF_ERROR(
      ToStatus(nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id.c_str(), &device)));
  return device;
}

absl::StatusOr<int64_t> GetDevicePcieBandwidth(nvmlDevice_t nvml_device) {
  // nvmlDeviceGetPcieSpeed returns wrong information. Verified with
  // nvbandwidth.
  unsigned int link_gen, link_width;
  nvmlReturn_t result =
      nvmlDeviceGetCurrPcieLinkGeneration(nvml_device, &link_gen);
  TF_RETURN_IF_ERROR(ToStatus(result));

  result = nvmlDeviceGetCurrPcieLinkWidth(nvml_device, &link_width);
  TF_RETURN_IF_ERROR(ToStatus(result));

  // PCIe v1 single lane speed. 0.25 GB/s
  int64_t lane_speed = 0.25 * 1024 * 1024 * 1024;
  for (int i = 1; i < link_gen; i++) {
    lane_speed *= 2;
  }

  return lane_speed * link_width;
}

absl::StatusOr<int> GetNumberOfActiveP2PNvlinks(nvmlDevice_t nvml_device) {
  int p2p_links = 0;

  constexpr int kBlackwellNvLinkCount = 18;
  for (unsigned int i = 0; i < kBlackwellNvLinkCount; i++) {
    nvmlEnableState_t is_active = NVML_FEATURE_DISABLED;
    nvmlReturn_t result = nvmlDeviceGetNvLinkState(nvml_device, i, &is_active);
    if (result == NVML_ERROR_NOT_SUPPORTED) {
      break;
    }
    TF_RETURN_IF_ERROR(ToStatus(result));
    if (is_active == NVML_FEATURE_DISABLED) {
      break;
    }

    uint32_t supported_p2p = 0;
    result = nvmlDeviceGetNvLinkCapability(
        nvml_device, i, NVML_NVLINK_CAP_P2P_SUPPORTED, &supported_p2p);
    if (result != NVML_ERROR_NOT_SUPPORTED) {
      TF_RETURN_IF_ERROR(ToStatus(result));
    }
    if (supported_p2p) {
      ++p2p_links;
    }
  }
  return p2p_links;
}

struct FabricInfo {
  std::string cluster_uuid;
  std::string clique_id;
};

absl::StatusOr<FabricInfo> GetDeviceFabricInfo(nvmlDevice_t device) {
#if CUDA_VERSION >= 12040
  nvmlGpuFabricInfoV_t fabricInfo{nvmlGpuFabricInfo_v2};
  fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;

  nvmlReturn_t result = nvmlDeviceGetGpuFabricInfoV(device, &fabricInfo);
  TF_RETURN_IF_ERROR(ToStatus(result));

  if (fabricInfo.state == NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
    std::string error_message =
        "[Ignore this message unless multi-node NVLink is used] "
        "CUDA driver version is too low for extracting fabric info (550+ "
        "required), or multi-node NVLink is not available.";
    VLOG(2) << error_message;
    return absl::InternalError(error_message);
  }

  static_assert(sizeof(fabricInfo.clusterUuid) == 16);
  std::string uuid_str = absl::StrFormat(
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      fabricInfo.clusterUuid[0], fabricInfo.clusterUuid[1],
      fabricInfo.clusterUuid[2], fabricInfo.clusterUuid[3],
      fabricInfo.clusterUuid[4], fabricInfo.clusterUuid[5],
      fabricInfo.clusterUuid[6], fabricInfo.clusterUuid[7],
      fabricInfo.clusterUuid[8], fabricInfo.clusterUuid[9],
      fabricInfo.clusterUuid[10], fabricInfo.clusterUuid[11],
      fabricInfo.clusterUuid[12], fabricInfo.clusterUuid[13],
      fabricInfo.clusterUuid[14], fabricInfo.clusterUuid[15]);

  return FabricInfo{uuid_str, absl::StrCat(fabricInfo.cliqueId)};
#else   // CUDA_VERSION >= 12040
  std::string error_message = "NVML usage is not supported";
  VLOG(2) << error_message;
  return absl::InternalError(error_message);
#endif  // CUDA_VERSION >= 12040
}

}  // namespace

bool CudaExecutor::MemoryTracker::Insert(CUdeviceptr ptr) {
  absl::MutexLock lock(mutex_);
  auto [it, inserted] = allocated_memory_.insert(ptr);
  return inserted;
}

bool CudaExecutor::MemoryTracker::Remove(CUdeviceptr ptr) {
  absl::MutexLock lock(mutex_);
  return allocated_memory_.erase(ptr) > 0;
}

// Given const GPU memory, returns a libcuda device pointer datatype, suitable
// for passing directly to libcuda APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// libcuda APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to libcuda functions which will honor constness.
static CUdeviceptr AsCudaDevicePtr(const DeviceAddressBase& gpu_mem) {
  return reinterpret_cast<CUdeviceptr>(gpu_mem.opaque());
}

// See description on const version above.
static CUdeviceptr AsCudaDevicePtr(DeviceAddressBase* gpu_mem) {
  return AsCudaDevicePtr(*gpu_mem);
}

absl::StatusOr<DeviceAddressBase> CudaExecutor::GetMemoryRange(
    const DeviceAddressBase& location) const {
  CUdeviceptr device_pointer;
  size_t size;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemGetAddressRange(&device_pointer, &size, AsCudaDevicePtr(location))));
  return DeviceAddressBase(reinterpret_cast<void*>(device_pointer), size);
}

std::unique_ptr<ActivateContext> CudaExecutor::Activate() {
  return std::make_unique<ScopedActivateContext>(cuda_context_);
}

CudaExecutor::~CudaExecutor() {
  CHECK(kernel_to_gpu_binary_.empty()) << "CudaExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "CudaExecutor has loaded modules.";
}

absl::StatusOr<xla::gpu::GpuCollectives*> GetGpuCollectives(
    StreamExecutor* executor) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_ASSIGN_OR_RETURN(xla::Collectives * collectives,
                      xla::CollectivesRegistry::Default("gpu"));
  return tsl::down_cast<xla::gpu::GpuCollectives*>(collectives);
}

CudaExecutor::VmmMemoryHandle::~VmmMemoryHandle() { CHECK_OK(Release()); }

absl::Status CudaExecutor::VmmMemoryHandle::Release() {
  if (handle_ != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle_))));
    handle_ = 0;
  }

  return absl::OkStatus();
}

CudaExecutor::VmmMemoryHandle::VmmMemoryHandle(VmmMemoryHandle&& other) {
  handle_ = other.handle_;
  other.handle_ = 0;
}

CudaExecutor::VmmMemoryHandle& CudaExecutor::VmmMemoryHandle::operator=(
    VmmMemoryHandle&& other) {
  if (this != &other) {
    CHECK_OK(Release());
    handle_ = other.handle_;
    other.handle_ = 0;
  }
  return *this;
}

absl::StatusOr<CudaExecutor::VmmMemoryHandle>
CudaExecutor::RetainVmmMemoryHandle(void* ptr) const {
  if (!is_vmm_supported_) {
    return absl::InternalError("VMM is not supported on this device.");
  }

  CUmemGenericAllocationHandle handle;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemRetainAllocationHandle(&handle, ptr)));

  return CudaExecutor::VmmMemoryHandle(static_cast<uint64_t>(handle));
}

absl::StatusOr<size_t> CudaExecutor::GetVmmGranularity() const {
  CUmemAllocationProp properties =
      GetVmmAllocationProperties(device_, is_rdma_supported_);
  size_t granularity = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemGetAllocationGranularity(
      &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)));
  return granularity;
}

absl::StatusOr<void*> CudaExecutor::VmmAllocateMemory(uint64_t bytes) {
  if (!is_vmm_supported_) {
    return absl::InternalError("VMM is not supported on this device.");
  }

  std::unique_ptr<ActivateContext> activation = Activate();

  CUmemAllocationProp properties =
      GetVmmAllocationProperties(device_, is_rdma_supported_);
  size_t granularity = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemGetAllocationGranularity(
      &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)));

  uint64_t padded_size = xla::RoundUpTo<uint64_t>(bytes, granularity);
  CUmemGenericAllocationHandle handle;

  // Create physical memory allocation.
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMemCreate(&handle, padded_size, &properties, 0)));

  // Reserve and map virtual address space.
  CUdeviceptr ptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemAddressReserve(&ptr, padded_size, granularity, 0, 0)));
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemMap(ptr, padded_size, 0, handle, 0)));

  XLA_VLOG_DEVICE(3, device_ordinal())
      << "VMM allocated " << ptr << " requested size: " << bytes
      << " padded size: " << padded_size << " granularity: " << granularity;

  int device_count = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cudaGetDeviceCount(&device_count)));
  for (int peer = 0; peer < device_count; peer++) {
    if (peer == device_ordinal() || CanEnablePeerAccessTo(peer)) {
      CUmemAccessDesc accessDesc = GetVmmAccessDescriptor(peer);
      TF_RETURN_IF_ERROR(
          cuda::ToStatus(cuMemSetAccess(ptr, padded_size, &accessDesc, 1)));
    }
  }

  if (!vmm_memory_tracker_.Insert(ptr)) {
    LOG(WARNING) << "[" << device_ordinal()
                 << "] VMM memory already tracked: " << ptr;
  }
  return reinterpret_cast<void*>(ptr);
}

absl::StatusOr<bool> CudaExecutor::VmmDeallocateMemory(void* ptr) {
  CUdeviceptr device_ptr = reinterpret_cast<CUdeviceptr>(ptr);
  if (!vmm_memory_tracker_.Remove(device_ptr)) {
    return false;
  }
  bool deletion_completed = false;
  absl::Cleanup cleanup = [&]() {
    if (!deletion_completed) {
      vmm_memory_tracker_.Insert(device_ptr);
    }
  };
  if (!is_vmm_supported_) {
    return absl::InternalError("VMM is not supported on this device.");
  }

  std::unique_ptr<ActivateContext> activation = Activate();

  CUmemGenericAllocationHandle handle = 0;
  {
    TF_ASSIGN_OR_RETURN(VmmMemoryHandle scoped_handle,
                        RetainVmmMemoryHandle(ptr));
    handle = static_cast<CUmemGenericAllocationHandle>(scoped_handle.handle());
  }
  size_t size = 0;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMemGetAddressRange(nullptr, &size, device_ptr)));
  VLOG(3) << "[" << device_ordinal() << "] VMM deallocated " << ptr
          << " size: " << size;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemUnmap(device_ptr, size)));
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemRelease(handle)));
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemAddressFree(device_ptr, size)));
  deletion_completed = true;
  return true;
}

absl::StatusOr<void*> CollectiveMemoryAllocate(StreamExecutor* executor,
                                               uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCollectives * gpu_collectives,
                      GetGpuCollectives(executor));
  return gpu_collectives->Allocate(bytes);
}

absl::Status CollectiveMemoryDeallocate(StreamExecutor* executor,
                                        void* location) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCollectives * gpu_collectives,
                      GetGpuCollectives(executor));
  return gpu_collectives->Deallocate(location);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
CudaExecutor::CreateMemoryAllocator(MemorySpace type) {
  if (type == MemorySpace::kUnified) {
    return std::make_unique<GenericMemoryAllocator>(
        [this](uint64_t size)
            -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          std::unique_ptr<ActivateContext> activation = Activate();
          CUdeviceptr result = 0;
          // "Portable" memory is visible to all CUDA contexts. Safe for our use
          // model.
          TF_RETURN_IF_ERROR(cuda::ToStatus(
              cuMemAllocManaged(&result, size, CU_MEM_ATTACH_GLOBAL)));
          void* ptr = reinterpret_cast<void*>(result);
          XLA_VLOG_DEVICE(2, device_ordinal())
              << "allocated " << ptr << " for context " << cuda_context_
              << " of " << size << " bytes in unified memory";
          return std::make_unique<GenericMemoryAllocation>(
              ptr, size, [this](void* location, uint64_t size) {
                std::unique_ptr<ActivateContext> activation = Activate();
                CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
                auto status = cuda::ToStatus(cuMemFree(pointer));
                if (!status.ok()) {
                  XLA_LOG_DEVICE(ERROR, device_ordinal())
                      << "failed to free unified memory at " << location
                      << "; result: " << status;
                } else {
                  XLA_VLOG_DEVICE(2, device_ordinal())
                      << "deallocated unified memory at " << location
                      << " for context " << cuda_context_;
                }
              });
        });
  }

  if (type == MemorySpace::kCollective) {
    // TODO(469289220): Use NCCL/NVSHMEM memory allocator here instead.
    return std::make_unique<GenericMemoryAllocator>(
        [this](uint64_t size)
            -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          TF_ASSIGN_OR_RETURN(void* ptr, CollectiveMemoryAllocate(this, size));
          XLA_VLOG_DEVICE(2, device_ordinal())
              << "allocated " << ptr << " for context " << cuda_context_
              << " of " << size << " bytes of collective memory";
          return std::make_unique<GenericMemoryAllocation>(
              ptr, size, [this](void* location, uint64_t size) {
                auto status = CollectiveMemoryDeallocate(this, location);
                if (!status.ok()) {
                  XLA_LOG_DEVICE(ERROR, device_ordinal())
                      << "failed to free collective memory at " << location
                      << "; result: " << status;
                } else {
                  XLA_VLOG_DEVICE(2, device_ordinal())
                      << "deallocated collective memory at " << location
                      << " for context " << cuda_context_;
                }
              });
        });
  }

  if (type == MemorySpace::kHost) {
    return std::make_unique<GenericMemoryAllocator>([this](uint64_t size) {
      return AllocateHostMemory(cuda_context_, numa_node_, size);
    });
  }

  return absl::UnimplementedError(
      absl::StrFormat("Unsupported memory type %d", type));
}

absl::Status CudaExecutor::Init() {
  TF_ASSIGN_OR_RETURN(device_, GetDevice(device_ordinal()));
  TF_ASSIGN_OR_RETURN(is_vmm_supported_, IsVmmSupported(device_));
  TF_ASSIGN_OR_RETURN(is_rdma_supported_, IsRdmaSupported(device_));
  TF_ASSIGN_OR_RETURN(is_multicast_supported_, IsMulticastSupported(device_));
  TF_ASSIGN_OR_RETURN(CudaContext * context,
                      CudaContext::Create(device_ordinal(), device_));
  cuda_context_ = context;
  TF_ASSIGN_OR_RETURN(delay_kernels_supported_, DelayKernelIsSupported());
  numa_node_ = ReadNumaNode(GetPCIBusID(device_), device_ordinal())
                   .value_or(tsl::port::kNUMANoAffinity);
  if (numa_node_ == tsl::port::kNUMANoAffinity) {
    XLA_VLOG_DEVICE(2, device_ordinal()) << "Could not determine NUMA node";
  }

  int cuda_device_count = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cudaGetDeviceCount(&cuda_device_count)));
  for (int i = 0; i < cuda_device_count; ++i) {
    if (i == device_ordinal()) {
      peer_access_cache_[i] = true;
      continue;
    }

    peer_access_cache_[i] = CanEnablePeerAccess(device_, i);
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> CudaExecutor::DelayKernelIsSupported() {
  // Check the assumption that this device supports unified addressing,
  // otherwise skip the delay kernel
  TF_ASSIGN_OR_RETURN(
      int status,
      GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device_));

  return static_cast<bool>(status);
}

absl::StatusOr<ModuleHandle> CudaExecutor::LoadModuleFromCuBin(
    const char* cubin) {
  ModuleHandle module_handle{cubin};
  uint64_t module_refcount;
  CUmodule module;
  std::tie(module, module_refcount) = gpu_binary_to_module_[module_handle];

  if (module == nullptr) {
    TF_ASSIGN_OR_RETURN(module, LoadCubin(cuda_context_, cubin));
    module_refcount = 1;
    XLA_VLOG_DEVICE(3, device_ordinal())
        << "Loaded CUBIN " << static_cast<const void*>(cubin) << " as module "
        << module;
  } else {
    ++module_refcount;
    XLA_VLOG_DEVICE(3, device_ordinal())
        << "CUBIN " << static_cast<const void*>(cubin)
        << " is already loaded as module " << module;
  }
  gpu_binary_to_module_[module_handle] = {module, module_refcount};
  return module_handle;
}

absl::StatusOr<ModuleHandle> CudaExecutor::LoadModuleFromPtx(const char* ptx) {
  ModuleHandle module_handle{ptx};
  uint64_t module_refcount;
  CUmodule module;
  std::tie(module, module_refcount) = gpu_binary_to_module_[module_handle];

  if (module == nullptr) {
    TF_ASSIGN_OR_RETURN(module, LoadPtx(cuda_context_, ptx));
    XLA_VLOG_DEVICE(3, device_ordinal())
        << "Loaded PTX " << static_cast<const void*>(ptx) << " as module "
        << module;
    module_refcount = 1;
  } else {
    ++module_refcount;
    XLA_VLOG_DEVICE(3, device_ordinal())
        << "PTX " << static_cast<const void*>(ptx)
        << " is already loaded as module " << module;
  }
  gpu_binary_to_module_[module_handle] = {module, module_refcount};
  return module_handle;
}

absl::StatusOr<std::unique_ptr<Kernel>> CudaExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  auto cuda_kernel = std::make_unique<CudaKernel>(this);
  const std::string& kernel_name = spec.kernel_name();

  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{in_memory_modules_mu_};
    const char* cubin = reinterpret_cast<const char*>(
        spec.cuda_cubin_in_memory()->cubin_bytes.data());
    TF_ASSIGN_OR_RETURN(ModuleHandle module_handle, LoadModuleFromCuBin(cubin));
    kernel_to_gpu_binary_[cuda_kernel.get()] = module_handle;

    CUmodule module = gpu_binary_to_module_.at(module_handle).first;
    XLA_VLOG_DEVICE(2, device_ordinal())
        << "getting function " << kernel_name << " from module " << module;
    TF_ASSIGN_OR_RETURN(
        CUfunction function,
        GetModuleFunction(cuda_context_, module, kernel_name.c_str()));
    cuda_kernel->set_gpu_function(function);

  } else if (spec.has_cuda_ptx_in_memory()) {
    const char* ptx = spec.cuda_ptx_in_memory()->ptx.data();
    if (ptx == nullptr) {
      XLA_LOG_DEVICE(FATAL, device_ordinal())
          << "Loader spec has no ptx for kernel " << kernel_name;
    }

    absl::MutexLock lock{in_memory_modules_mu_};
    TF_ASSIGN_OR_RETURN(ModuleHandle module_handle, LoadModuleFromPtx(ptx));
    kernel_to_gpu_binary_[cuda_kernel.get()] = module_handle;

    CUmodule module = gpu_binary_to_module_.at(module_handle).first;
    XLA_VLOG_DEVICE(2, device_ordinal())
        << "getting function " << kernel_name << " from module " << module;
    TF_ASSIGN_OR_RETURN(
        CUfunction function,
        GetModuleFunction(cuda_context_, module, kernel_name.c_str()));
    cuda_kernel->set_gpu_function(function);

  } else if (spec.has_in_process_symbol()) {
    void* symbol = spec.in_process_symbol()->symbol;

    XLA_VLOG_DEVICE(2, device_ordinal())
        << "Resolve CUDA kernel " << kernel_name
        << " from symbol pointer: " << symbol;
    cudaFunction_t func;
    std::unique_ptr<ActivateContext> scoped_activation = Activate();
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cudaGetFuncBySymbol(&func, symbol),
        absl::StrFormat("[%d] Failed call to cudaGetFuncBySymbol",
                        device_ordinal())));
    cuda_kernel->set_gpu_function(func);

  } else {
    return absl::InternalError("No method of loading CUDA kernel provided");
  }
  XLA_VLOG_DEVICE(3, device_ordinal())
      << "LoadKernel on kernel : " << kernel_name;

  {
    // Keep track of loaded kernels.
    absl::MutexLock lock{in_memory_modules_mu_};
    loaded_kernels_.insert(cuda_kernel.get());
  }

  // Update CUDA kernel properties after it was loaded in the CUDA context.
  cuda_kernel->set_name(kernel_name);

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the CUDA API.
  cuda_kernel->set_arity(spec.arity());

  TF_ASSIGN_OR_RETURN(KernelMetadata kernel_metadata,
                      cuda_kernel->GetKernelMetadata());
  cuda_kernel->set_metadata(kernel_metadata);
  if (std::holds_alternative<KernelLoaderSpec::KernelArgsPackingFunc>(
          spec.kernel_args_packing())) {
    cuda_kernel->set_args_packing(
        std::get<KernelLoaderSpec::KernelArgsPackingFunc>(
            spec.kernel_args_packing()));
  } else {
    const auto& packing_spec =
        std::get<KernelArgsPackingSpec>(spec.kernel_args_packing());
    cuda_kernel->set_args_packing(
        [packing_spec](const Kernel& kernel, const KernelArgs& args) {
          const auto& mem_args = Cast<KernelArgsDeviceAddressArray>(&args);
          return packing_spec.BuildArguments(mem_args->device_addr_args(),
                                             args.number_of_shared_bytes());
        });
  }
  return std::move(cuda_kernel);
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>>
CudaExecutor::CreateEventBasedTimer(Stream* stream, bool use_delay_kernel) {
  const CudaTimer::TimerType timer_type =
      (use_delay_kernel && ShouldLaunchDelayKernel() &&
       delay_kernels_supported_)
          ? CudaTimer::TimerType::kDelayKernel
          : CudaTimer::TimerType::kEventBased;

  TF_ASSIGN_OR_RETURN(CudaTimer timer,
                      CudaTimer::Create(this, stream, timer_type));
  return std::make_unique<CudaTimer>(std::move(timer));
}

bool CudaExecutor::UnloadGpuBinary(ModuleHandle gpu_binary) {
  auto module_it = gpu_binary_to_module_.find(gpu_binary);
  if (gpu_binary_to_module_.end() == module_it) {
    XLA_VLOG_DEVICE(3, device_ordinal())
        << "No loaded CUDA module for " << gpu_binary;
    return false;
  }
  auto& module = module_it->second.first;
  auto& refcount = module_it->second.second;
  XLA_VLOG_DEVICE(3, device_ordinal())
      << "Found CUDA module " << module << " with refcount " << refcount;
  if (--refcount == 0) {
    XLA_VLOG_DEVICE(3, device_ordinal()) << "Unloading CUDA module " << module;
    UnloadCudaModule(cuda_context_, module);
    gpu_binary_to_module_.erase(module_it);
  }
  return true;
}

void CudaExecutor::UnloadKernel(const Kernel* kernel) {
  XLA_VLOG_DEVICE(3, device_ordinal())
      << "Unloading kernel " << kernel << " : " << kernel->name();

  absl::MutexLock lock{in_memory_modules_mu_};
  loaded_kernels_.erase(kernel);

  auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
  if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
    // We might never see kernel being explicitly loaded if it was resolved from
    // in process symbol pointer (CUDA C++ device function pointer).
    XLA_VLOG_DEVICE(3, device_ordinal())
        << "Kernel " << kernel << " : " << kernel->name()
        << " has never been loaded.";
    return;
  }
  XLA_VLOG_DEVICE(3, device_ordinal())
      << "Kernel " << kernel << " : " << kernel->name()
      << " has loaded GPU code " << gpu_binary_it->second;
  UnloadGpuBinary(gpu_binary_it->second);
  kernel_to_gpu_binary_.erase(gpu_binary_it);
}

absl::StatusOr<ModuleHandle> CudaExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  // We store the pointer to the GPU binary (PTX or CUBIN) as
  // ModuleHandle::id().
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{in_memory_modules_mu_};
    return LoadModuleFromCuBin(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()));
  }
  if (spec.has_cuda_ptx_in_memory()) {
    if (!spec.cuda_ptx_in_memory()) {
      return absl::InternalError("PTX not found in spec");
    }

    absl::MutexLock lock{in_memory_modules_mu_};
    return LoadModuleFromPtx(spec.cuda_ptx_in_memory());
  }
  return absl::InternalError("No method of loading CUDA module provided");
}

bool CudaExecutor::UnloadModule(ModuleHandle module_handle) {
  absl::MutexLock lock{in_memory_modules_mu_};
  return UnloadGpuBinary(module_handle);
}

namespace {
absl::uint128 Fingerprint128(const absl::string_view s) {
  auto fp = tsl::Fingerprint128(s);
  return absl::MakeUint128(fp.high64, fp.low64);
}

}  // namespace

absl::StatusOr<std::shared_ptr<DeviceAddressBase>>
CudaExecutor::CreateOrShareConstant(Stream* stream,
                                    absl::Span<const uint8_t> content) {
  absl::MutexLock lock{shared_constants_mu_};
  // We assume all constants are uniquely identified by this hash. In the
  // (highly unlikely) event of a hash collision, the program will likely crash
  // (because the cached constant that will be returned by mistake is unlikely
  // to have the correct size).
  absl::uint128 fingerprint = Fingerprint128(absl::string_view(
      reinterpret_cast<const char*>(content.data()), content.size()));
  // Must insert nullptr first to get an iterator to the insertion point.
  auto insert_result = shared_constants_.insert(
      {fingerprint, std::weak_ptr<DeviceAddressBase>()});
  auto it = insert_result.first;
  bool was_already_in_cache = !insert_result.second;
  std::shared_ptr<DeviceAddressBase> shared_constant;

  if (was_already_in_cache) {
    shared_constant = it->second.lock();
  }

  if (shared_constant == nullptr) {
    // Either the constant wasn't found in the cache, or it was but its
    // weak_ptr had expired.
    auto new_constant = std::make_unique<DeviceAddressBase>(
        Allocate(content.size(), /*memory_space=*/0));
    if (new_constant->opaque() == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Failed to allocate %d bytes for new constant", content.size()));
    }

    TF_RETURN_IF_ERROR(
        stream->Memcpy(new_constant.get(), content.data(), content.size()));
    absl::Status status = stream->BlockHostUntilDone();
    if (!status.ok()) {
      Deallocate(new_constant.get());
      status.Update(absl::InternalError(absl::StrFormat(
          "Memcpy to device address %p failed", new_constant->opaque())));
      return status;
    }

    // Capturing 'this' in the custom deleter means this executor must
    // outlive all shared uses of this constant.
    shared_constant = std::shared_ptr<DeviceAddressBase>(
        new_constant.release(), [this](DeviceAddressBase* p) {
          Deallocate(p);
          delete p;
        });
    it->second = std::weak_ptr<DeviceAddressBase>(shared_constant);
  }

  return shared_constant;
}

DeviceAddressBase CudaExecutor::Allocate(uint64_t size, int64_t memory_space) {
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "CudaExecutor::Allocate size: " << size
      << " memory_space: " << memory_space;

  if (memory_space == static_cast<int64_t>(MemorySpace::kCollective)) {
    auto result = CollectiveMemoryAllocate(this, size);
    if (!result.ok()) {
      XLA_LOG_DEVICE(ERROR, device_ordinal())
          << "CudaExecutor::Allocate returns " << result.value();
    }
    XLA_VLOG_DEVICE(1, device_ordinal())
        << "CudaExecutor::Allocate returns " << result.value();
    return DeviceAddressBase(result.value(), size);
  }

  if (memory_space == static_cast<int64_t>(MemorySpace::kHost)) {
    auto result = HostAllocate(cuda_context_, numa_node_, size);
    if (!result.ok()) {
      XLA_LOG_DEVICE(ERROR, device_ordinal())
          << "Failed to allocate host memory: " << result.status();
      return DeviceAddressBase(nullptr, 0);
    }
    XLA_VLOG_DEVICE(1, device_ordinal())
        << "CudaExecutor::Allocate returns " << result.value();
    return DeviceAddressBase(result.value(), size);
  }

  if (memory_space == static_cast<int64_t>(MemorySpace::kP2P) &&
      is_vmm_supported_) {
    auto device_buf_base = VmmAllocateMemory(size);

    if (device_buf_base.ok()) {
      return DeviceAddressBase(device_buf_base.value(), size);
    }

    XLA_LOG_DEVICE(ERROR, device_ordinal())
        << "Failed to allocate memory with VMM: " << device_buf_base.status();

    return DeviceAddressBase(nullptr, 0);
  }

  CHECK(memory_space == static_cast<int64_t>(MemorySpace::kDevice) ||
        memory_space == static_cast<int64_t>(MemorySpace::kP2P));

  auto device_buf_base = DeviceAllocate(cuda_context_, size);
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "CudaExecutor::Allocate returns " << device_buf_base;
  return DeviceAddressBase(device_buf_base, size);
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
CudaExecutor::HostMemoryAllocate(uint64_t size) {
  return AllocateHostMemory(cuda_context_, numa_node_, size);
}

void CudaExecutor::Deallocate(DeviceAddressBase* mem) {
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "CudaExecutor::Deallocate mem: " << mem->opaque();

  auto status_or_memory_space = GetPointerMemorySpace(mem->opaque());
  if (!status_or_memory_space.ok()) {
    LOG(ERROR) << status_or_memory_space.status();
    return;
  }
  auto memory_space = status_or_memory_space.value();
  if (memory_space == MemorySpace::kHost) {
    HostDeallocate(cuda_context_, numa_node_, mem->opaque(), mem->size());
  } else {
    // Memory space is always kDevice here, so the only way to check if the
    // memory was allocated with VMM API is to try to retain the handle with VMM
    // API (which VmmDeallocateMemory does).
    auto result = VmmDeallocateMemory(mem->opaque());
    if (!result.ok()) {
      LOG(WARNING) << "Failed to deallocate VMM memory handle: "
                   << result.status();
    } else if (!result.value()) {  // If it was not allocated with VMM API.
      DeviceDeallocate(cuda_context_, mem->opaque());
    }
  }
}

bool CudaExecutor::SynchronizeAllActivity() {
  return cuda_context_->Synchronize().ok();
}

bool CudaExecutor::HostMemoryRegister(void* location, uint64_t size) {
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "Called StreamExecutor::HostMemoryRegister(data=" << location << ")";
  return HostRegister(cuda_context_, location, size);
}

bool CudaExecutor::HostMemoryUnregister(void* location) {
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "Called StreamExecutor::HostUnregister(data=" << location << ")";
  return HostUnregister(cuda_context_, location);
}

absl::Status CudaExecutor::SynchronousMemZero(DeviceAddressBase* location,
                                              uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  CUdeviceptr cuda_location = AsCudaDevicePtr(location);
  if (reinterpret_cast<uintptr_t>(location->opaque()) % sizeof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return cuda::ToStatus(
        cuMemsetD32(cuda_location, 0x0, size / sizeof(uint32_t)),
        "Failed to memset memory");
  }
  return cuda::ToStatus(cuMemsetD8(cuda_location, 0x0, size),
                        "Failed to memset memory");
}

absl::Status CudaExecutor::SynchronousMemcpy(DeviceAddressBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemcpyHtoD(AsCudaDevicePtr(gpu_dst), host_src, size),
      absl::StrFormat("%sfailed to synchronous memcpy from "
                      "host to device: GPU dst: %llx;"
                      " host src: %p; size: %u=0x%x",
                      xla::XlaFormatDevice(device_ordinal()),
                      AsCudaDevicePtr(gpu_dst), host_src, size, size)));
  XLA_VLOG_DEVICE(2, device_ordinal())
      << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return absl::OkStatus();
}

absl::Status CudaExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceAddressBase& gpu_src,
                                             uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemcpyDtoH(host_dst, AsCudaDevicePtr(gpu_src), size),
      absl::StrFormat("%sfailed to synchronous memcpy from device to host "
                      "host dst: %p; GPU src: %llx; size: %u=0x%x",
                      xla::XlaFormatDevice(device_ordinal()), host_dst,
                      AsCudaDevicePtr(gpu_src), size, size)));
  XLA_VLOG_DEVICE(2, device_ordinal()) << "successfully sync memcpy'd d2h of "
                                       << size << " bytes to " << host_dst;
  return absl::OkStatus();
}

void CudaExecutor::DeallocateStream(Stream* stream) {
  {
    absl::MutexLock lock(mu_);
    if (dnn_ != nullptr) {
      dnn_->NotifyStreamDestroyed(stream);
    }
  }
  absl::MutexLock l(alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(stream->platform_specific_handle().stream);
}

blas::BlasSupport* CudaExecutor::AsBlas() {
  absl::MutexLock lock(mu_);
  if (blas_ != nullptr) {
    return blas_.get();
  }

  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(cuda::kCudaPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().message();
    return nullptr;
  }

  auto blas = status.value()(this);
  blas_.reset(blas);
  return blas_.get();
}

dnn::DnnSupport* CudaExecutor::AsDnn() {
  absl::MutexLock lock(mu_);
  if (dnn_ != nullptr) {
    return dnn_.get();
  }
  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(cuda::kCudaPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().message();
    return nullptr;
  }

  auto dnn = status.value()(this);

  dnn_.reset(dnn);

  return dnn_.get();
}

fft::FftSupport* CudaExecutor::AsFft() {
  absl::MutexLock lock(mu_);
  if (fft_ != nullptr) {
    return fft_.get();
  }
  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(cuda::kCudaPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().message();
    return nullptr;
  }

  auto fft = status.value()(this);

  fft_.reset(fft);
  return fft_.get();
}

bool CudaExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  CudaExecutor* cuda_other = static_cast<CudaExecutor*>(other);
  absl::StatusOr<int> to_device = DeviceFromContext(cuda_other->cuda_context_);
  if (!to_device.ok()) {
    LOG(ERROR) << "failed to resolve 'to' peer access context to a device: "
               << to_device.status();
    return false;
  }
  return CanEnablePeerAccessTo(*to_device);
}

bool CudaExecutor::CanEnablePeerAccessTo(int other_device_ordinal) {
  auto it = peer_access_cache_.find(other_device_ordinal);
  if (it != peer_access_cache_.end()) {
    return it->second;
  }

  LOG(WARNING) << "Attemping to enable peer access from: " << device_ordinal()
               << " to: " << other_device_ordinal
               << " which was not available during initialization.";
  return false;
}

absl::Status CudaExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  CudaExecutor* cuda_other = static_cast<CudaExecutor*>(other);
  return EnablePeerAccess(cuda_context_, cuda_other->cuda_context_);
}

bool CudaExecutor::DeviceMemoryUsage(int64_t* free_out,
                                     int64_t* total_out) const {
  ScopedActivateContext activation(cuda_context_);
  size_t free = 0;
  size_t total = 0;
  auto status = cuda::ToStatus(cuMemGetInfo(&free, &total));
  if (!status.ok()) {
    LOG(ERROR) << "failed to query device memory info: " << status;
    return false;
  }

  *free_out = free;
  *total_out = total;
  return true;
}

absl::StatusOr<DeviceAddressBase> CudaExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  void* mem = nullptr;
  size_t bytes = 0;
  CHECK(static_cast<bool>(module_handle));

  {  // give limited scope to MutexLock
    absl::MutexLock lock{in_memory_modules_mu_};
    auto it = gpu_binary_to_module_.find(module_handle);
    CHECK(it != gpu_binary_to_module_.end());

    CUmodule gpu_module_handle = it->second.first;
    CHECK(gpu_module_handle != nullptr);
    TF_RETURN_IF_ERROR(
        GetModuleSymbol(cuda_context_, gpu_module_handle, symbol_name.c_str(),
                        reinterpret_cast<CUdeviceptr*>(&mem), &bytes));
    return DeviceAddressBase(mem, bytes);
  }

  return absl::NotFoundError(
      absl::StrCat("Check if module containing symbol ", symbol_name,
                   " is loaded (module_handle = ",
                   reinterpret_cast<uintptr_t>(module_handle.id()), ")"));
}

namespace {
absl::Status FillBlockDimLimit(CUdevice device, BlockDim* block_dim_limit) {
  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
  TF_RETURN_IF_ERROR(GetGridLimits(&x, &y, &z, device));
  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<std::unique_ptr<Event>> CudaExecutor::CreateEvent() {
  TF_ASSIGN_OR_RETURN(auto event, CudaEvent::Create(this, false));
  return std::make_unique<CudaEvent>(std::move(event));
}

absl::StatusOr<std::unique_ptr<Stream>> CudaExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(auto stream, CudaStream::Create(this, priority));
  absl::MutexLock l(alive_gpu_streams_mu_);
  alive_gpu_streams_[stream->stream_handle()] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>>
CudaExecutor::CreateCommandBuffer(CommandBuffer::Mode mode) {
  XLA_VLOG_DEVICE(2, device_ordinal())
      << "Create CUDA command buffer (CUDA graph)";
  return CudaCommandBuffer::Create(mode, this, cuda_context_);
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
CudaExecutor::CreateDeviceDescription(int device_ordinal) {
  TF_ASSIGN_OR_RETURN(CUdevice device, GetDevice(device_ordinal));
  TF_ASSIGN_OR_RETURN(CudaComputeCapability cc, GetComputeCapability(device));

  DeviceDescription desc;
  int32_t driver_version{};
  {
    // TODO(b/381052076): Return an error instead of silent failure once TF can
    // accommodate that.
    absl::Status result = cuda::ToStatus(cuDriverGetVersion(&driver_version),
                                         "Could not get driver version");
    if (!result.ok()) {
      LOG(ERROR) << result;
    }
  }
  desc.set_driver_version(
      ParseCudaVersion(driver_version).value_or(SemanticVersion{0, 0, 0}));

  int32_t runtime_version{};
  {
    // TODO(b/381052076): Return an error instead of silent failure once TF can
    // accommodate that.
    absl::Status result =
        cuda::ToStatus(cudaRuntimeGetVersion(&runtime_version),
                       "Failed call to cudaGetRuntimeVersion");
    if (!result.ok()) {
      LOG(ERROR) << result;
    }
  }
  desc.set_runtime_version(
      ParseCudaVersion(runtime_version).value_or(SemanticVersion{0, 0, 0}));
  desc.set_compile_time_toolkit_version(
      ParseCudaVersion(CUDA_VERSION).value_or(SemanticVersion{0, 0, 0}));

  // cudnnGetProperty (the function that backs GetLoadedCudnnVersion()) needs
  // 64KiB of stack, so we call it from a separate thread to avoid stack
  // overflows.
  absl::Notification cudnn_version_ready;
  GetDriverExecutor()->Schedule([&]() {
    absl::StatusOr<SemanticVersion> cudnn_version =
        cuda::GetLoadedCudnnVersion();
    if (cudnn_version.ok()) {
      desc.set_dnn_version(*cudnn_version);
    } else {
      LOG(WARNING)
          << "Failed to determine cuDNN version (Note that this is expected if "
             "the application doesn't link the cuDNN plugin): "
          << cudnn_version.status();
    }
    cudnn_version_ready.Notify();
  });
  cudnn_version_ready.WaitForNotification();

  std::string pci_bus_id = GetPCIBusID(device);
  desc.set_pci_bus_id(pci_bus_id);

  {
    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    std::optional<int> numa_node = ReadNumaNode(pci_bus_id, device_ordinal);
    // If the kernel reports -1, adjust to 0; leave as -1 if no value could be
    // obtained.
    desc.set_numa_node(numa_node.has_value() ? std::max(0, *numa_node)
                                             : tsl::port::kNUMANoAffinity);
  }

  {
    desc.set_threads_per_block_limit(
        GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device)
            .value());

    ThreadDim thread_dim_limit;
    thread_dim_limit.x =
        GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device).value();
    thread_dim_limit.y =
        GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device).value();
    thread_dim_limit.z =
        GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device).value();
    desc.set_thread_dim_limit(thread_dim_limit);
  }

  int sm_clock_khz =
      GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device).value();
  float device_clock_rate_ghz = static_cast<float>(sm_clock_khz) / 1e6;
  desc.set_clock_rate_ghz(device_clock_rate_ghz);

  {
    bool ecc_enabled = false;
    IsEccEnabled(device, &ecc_enabled);
    desc.set_ecc_enabled(ecc_enabled);
  }

  uint64_t device_memory_size = static_cast<uint64_t>(-1);
  GetDeviceTotalMemory(device, &device_memory_size);
  desc.set_device_memory_size(device_memory_size);

  int64_t l2_cache_bytes =
      GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device).value();
  desc.set_l2_cache_size(l2_cache_bytes);

  absl::StatusOr<int> mem_clock_khz =
      GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device_ordinal);
  absl::StatusOr<int> mem_bus_width_bits = GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device_ordinal);
  if (mem_clock_khz.ok() && mem_bus_width_bits.ok()) {
    // Temporary fix when driver reports 0.
    // Affected CUDA 13.1, driver r590, should be fixed in later releases.
    if (mem_clock_khz.value() == 0 || mem_bus_width_bits.value() == 0) {
      LOG(WARNING) << "Memory clock rate or bus width is 0";
      if (cc.major == 11 && cc.minor == 0) {  // Thor
        LOG(WARNING) << "Using hardcoded values for Thor";
        mem_clock_khz = 4266000;
        mem_bus_width_bits = 256;
      }
    }
    // Times 2 because HBM is DDR memory; it gets two data bits per each data
    // lane.
    desc.set_memory_bandwidth(2 * int64_t{mem_clock_khz.value()} * 1000 *
                              int64_t{mem_bus_width_bits.value()} / 8);
  }

  if (absl::StatusOr<nvmlDevice_t> device = GetNvmlDevice(pci_bus_id);
      device.ok()) {
    absl::StatusOr<int64_t> bandwidth = GetDevicePcieBandwidth(*device);
    if (bandwidth.ok()) {
      desc.set_pcie_bandwidth(*bandwidth);
    } else {
      LOG(ERROR) << bandwidth.status().message()
                 << " Assuming PCIe gen 3 x16 bandwidth.";
      bandwidth = 16LL * 1024 * 1024 * 1024;
    }

    absl::StatusOr<int64_t> p2p_link_count =
        GetNumberOfActiveP2PNvlinks(*device);
    DeviceInterconnectInfo info;
    if (p2p_link_count.ok()) {
      info.active_links = *p2p_link_count;
    } else {
      LOG(ERROR) << p2p_link_count;
    }
    absl::StatusOr<FabricInfo> fabric_info = GetDeviceFabricInfo(*device);
    if (fabric_info.ok()) {
      info.cluster_uuid = fabric_info->cluster_uuid;
      info.clique_id = fabric_info->clique_id;
    }
    desc.set_device_interconnect_info(info);
  }

  {
    BlockDim block_dim_limit;
    TF_RETURN_IF_ERROR(FillBlockDimLimit(device, &block_dim_limit));
    desc.set_block_dim_limit(block_dim_limit);
  }

  {
    TF_ASSIGN_OR_RETURN(std::string device_name, GetDeviceName(device));
    desc.set_name(device_name);
  }

  desc.set_platform_version(absl::StrCat("Compute Capability ", cc.ToString()));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  desc.set_device_address_bits(64);

  desc.set_device_vendor("NVIDIA Corporation");
  desc.set_cuda_compute_capability(cc);
  desc.set_shared_memory_per_core(GetMaxSharedMemoryPerCore(device).value());
  desc.set_shared_memory_per_block(GetMaxSharedMemoryPerBlock(device).value());
  desc.set_shared_memory_per_block_optin(
      GetMaxSharedMemoryPerBlockOptin(device).value());
  int core_count = GetMultiprocessorCount(device).value();
  desc.set_core_count(core_count);
  desc.set_fpus_per_core(GetFpusPerCore(cc));
  desc.set_threads_per_core_limit(
      GetMaxThreadsPerMultiprocessor(device).value());
  desc.set_registers_per_block_limit(GetMaxRegistersPerBlock(device).value());
  desc.set_threads_per_warp(GetThreadsPerWarp(device).value());
  desc.set_registers_per_core_limit(
      GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                         device)
          .value());

  FillExecutionUnitDesc(cc, device_clock_rate_ghz, desc);

  auto value_or = [](const auto& status_or, auto default_val) {
    if (status_or.ok()) {
      return *status_or;
    }
    return default_val;
  };

  // It would be better to use the PCI device ID or some other truly unique
  // identifier for the GPU model.  But getting this requires using NVML or
  // other hacks, which we don't have access to in OSS TensorFlow.
  //
  // Alternatively you might be tempted to use GetDeviceName as a
  // unique identifier, but this is not stable across GPU VBIOS versions.
  //
  // For now, this identifier is good enough.
  desc.set_model_str(absl::StrFormat(
      "sm_%s with %dB RAM, %d cores, %dKHz clock, %dKHz mem clock, %dB L2$",
      cc.ToString(), device_memory_size, core_count, sm_clock_khz,
      value_or(mem_clock_khz, 0), l2_cache_bytes));

  return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<MemorySpace> CudaExecutor::GetPointerMemorySpace(
    const void* ptr) {
  CUdeviceptr pointer = reinterpret_cast<CUdeviceptr>(const_cast<void*>(ptr));
  unsigned int is_managed;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuPointerGetAttribute(
      &is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, pointer)));

  if (is_managed) {
    return MemorySpace::kUnified;
  }

  unsigned int value;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuPointerGetAttribute(
      &value, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer)));
  switch (value) {
    case CU_MEMORYTYPE_DEVICE:
      return MemorySpace::kDevice;
    case CU_MEMORYTYPE_HOST:
      return MemorySpace::kHost;
    default:
      return absl::InternalError(
          absl::StrCat("unknown memory space provided by CUDA API: ", value));
  }
}

int CudaExecutor::GetGpuStreamPriority(StreamPriority priority) {
  if (priority == StreamPriority::Default) {
    return 0;
  }

  absl::call_once(stream_priority_once_, [this]() {
    std::unique_ptr<ActivateContext> activation = Activate();
    int lowest = 0;
    int highest = 0;
    absl::Status status =
        cuda::ToStatus(cuCtxGetStreamPriorityRange(&lowest, &highest));
    if (!status.ok()) {
      LOG(ERROR) << "Could not query stream priority range. Returning default "
                    "priority.";
      stream_priority_query_ok_ = false;
      return;
    }
    stream_priority_lowest_ = lowest;
    stream_priority_highest_ = highest;
    stream_priority_query_ok_ = true;
  });

  if (!stream_priority_query_ok_) {
    return 0;
  }
  return priority == StreamPriority::Highest ? stream_priority_highest_
                                             : stream_priority_lowest_;
}

absl::StatusOr<const CudaKernel*> CudaExecutor::GetCudaKernel(
    const Kernel* kernel) {
  absl::MutexLock lock{in_memory_modules_mu_};
  auto it = loaded_kernels_.find(kernel);
  if (it == loaded_kernels_.end()) {
    return absl::NotFoundError("Kernel not loaded in this executor.");
  }
  return static_cast<const CudaKernel*>(*it);
}

absl::StatusOr<TensorMap> CudaExecutor::CreateTensorMap(
    const TmaDescriptor& tma_desc, void* global_address) {
  TF_ASSIGN_OR_RETURN(CUtensorMapDataType data_type,
                      GetTensorMapDataType(tma_desc.element_size()));
  CUtensorMapSwizzle swizzle = GetTensorMapSwizzle(tma_desc.swizzle());
  CUtensorMapL2promotion l2_promotion =
      GetTensorMapL2Promotion(tma_desc.l2_promotion());
  CUtensorMapFloatOOBfill float_oob_fill =
      GetTensorMapFloatOOBFill(tma_desc.float_oob_fill());
  CUtensorMapInterleave interleave =
      GetTensorMapInterleave(tma_desc.interleave());

  CUtensorMap tensor_map;
  auto result = cuTensorMapEncodeTiled(
      &tensor_map, data_type, tma_desc.num_dimensions(), global_address,
      tma_desc.global_dims().data(), tma_desc.global_strides().data(),
      tma_desc.box_dims().data(), tma_desc.element_strides().data(), interleave,
      swizzle, l2_promotion, float_oob_fill);
  if (result != CUDA_SUCCESS) {
    const char* error_message;
    cuGetErrorString(result, &error_message);
    return absl::InternalError(absl::StrFormat(
        "Failed to create tensormap with cuTensorMapEncodeTiled: %s",
        error_message));
  }
  return absl::bit_cast<TensorMap>(tensor_map);
}

absl::StatusOr<std::unique_ptr<MulticastMemory>>
CudaExecutor::CreateMulticastMemory(uint64_t size, int num_devices) const {
  if (!is_multicast_supported_) {
    return absl::FailedPreconditionError(
        "Multicast memory is not supported on this platform.");
  }
  if (size == 0 || num_devices <= 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Multicast memory size must be > 0 and number of devices "
                     "must be greater than 1, but got size: ",
                     size, " and num_devices: ", num_devices, "."));
  }

  auto multicast_memory = std::make_unique<CudaMulticastMemory>();
  TF_RETURN_IF_ERROR(multicast_memory->Initialize(size, num_devices, this));
  return multicast_memory;
}

CudaExecutor::CudaMulticastMemory::~CudaMulticastMemory() {
  if (handle_ != 0) {
    for (auto const& [device_ordinal, mapped_memory_ptr] : mapped_devices_) {
      XLA_VLOG_DEVICE(3, device_ordinal) << "Unbind multicast: " << handle_;
      CHECK_OK(cuda::ToStatus(cuMulticastUnbind(handle_, device_ordinal,
                                                /*mcOffset=*/0, padded_size_)));

      XLA_VLOG_DEVICE(3, device_ordinal) << "Unmap ptr: " << mapped_memory_ptr;
      CHECK_OK(cuda::ToStatus(cuMemUnmap(mapped_memory_ptr, padded_size_)));
      XLA_VLOG_DEVICE(3, device_ordinal)
          << "Release address space: " << mapped_memory_ptr;
      CHECK_OK(
          cuda::ToStatus(cuMemAddressFree(mapped_memory_ptr, padded_size_)));
    }
    CHECK_OK(cuda::ToStatus(
        cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle_))));
  }
}

absl::Status CudaExecutor::CudaMulticastMemory::Initialize(
    uint64_t size, int num_devices, const GpuExecutor* gpu_executor) {
  const CudaExecutor* cuda_executor =
      dynamic_cast<const CudaExecutor*>(gpu_executor);
  if (cuda_executor == nullptr) {
    return absl::InvalidArgumentError("GpuExecutor is not a CudaExecutor.");
  }

  if (handle_ != 0) {
    return absl::FailedPreconditionError(
        "Multicast memory is already initialized.");
  }

  if (num_devices <= 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of devices must be greater than 1, but got ",
                     num_devices, "."));
  }

  CUmemAllocationProp properties = GetVmmAllocationProperties(
      cuda_executor->device_, cuda_executor->is_rdma_supported_);
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemGetAllocationGranularity(
      &granularity_, &properties, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)));

  padded_size_ = xla::RoundUpTo<size_t>(size, granularity_);
  num_devices_ = num_devices;
  TF_ASSIGN_OR_RETURN(CUmulticastObjectProp multicast_properties,
                      CreateMulticastObjectProperties(num_devices_, size));

  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMulticastCreate(&handle_, &multicast_properties)));
  XLA_VLOG_DEVICE(3, cuda_executor->device_ordinal())
      << "Created multicast memory: " << static_cast<uint64_t>(handle_)
      << " size: " << padded_size_ << " with granularity: " << granularity_
      << " for " << num_devices_ << " devices.";
  return absl::OkStatus();
}

absl::Status CudaExecutor::CudaMulticastMemory::SubscribeDevice(
    int device_number) {
  if (handle_ == 0) {
    return absl::FailedPreconditionError(
        "Multicast memory is not initialized.");
  }

  if (subscribed_devices_ >= num_devices_) {
    return absl::InvalidArgumentError("All devices are already subscribed.");
  }

  XLA_VLOG_DEVICE(3, device_number) << "Subscribe to multicast: " << handle_;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMulticastAddDevice(handle_, device_number)));
  subscribed_devices_++;
  return absl::OkStatus();
}

absl::StatusOr<void*> CudaExecutor::CudaMulticastMemory::MapMemory(
    const DeviceAddressBase& location, const GpuExecutor* gpu_executor) {
  const CudaExecutor* cuda_executor =
      dynamic_cast<const CudaExecutor*>(gpu_executor);
  if (cuda_executor == nullptr) {
    return absl::InvalidArgumentError("GpuExecutor is not a CudaExecutor.");
  }

  if (location.is_null()) {
    return absl::InvalidArgumentError("Device pointer is null.");
  }

  if (handle_ == 0) {
    return absl::FailedPreconditionError(
        "Multicast memory is not initialized.");
  }

  if (subscribed_devices_ != num_devices_) {
    return absl::FailedPreconditionError("All devices should be subscribed.");
  }

  TF_ASSIGN_OR_RETURN(CudaExecutor::VmmMemoryHandle memory_handle,
                      cuda_executor->RetainVmmMemoryHandle(location.opaque()));

  CUmemGenericAllocationHandle retained_memory_handle =
      static_cast<CUmemGenericAllocationHandle>(memory_handle.handle());

  TF_ASSIGN_OR_RETURN(auto base_address,
                      cuda_executor->GetMemoryRange(location));
  uint64_t offset = reinterpret_cast<uint64_t>(location.opaque()) -
                    reinterpret_cast<uint64_t>(base_address.opaque());

  // Bind the memory to the multicast object.
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMulticastBindMem(handle_, /*mcOffset=*/0, retained_memory_handle,
                         /*memOffset=*/offset, padded_size_, /*flags=*/0)));

  XLA_VLOG_DEVICE(3, cuda_executor->device_ordinal())
      << "Mapped multicast memory: " << static_cast<uint64_t>(handle_)
      << " size: " << padded_size_ << " with granularity: " << granularity_
      << " to address: " << location.opaque()
      << " offset from base range: " << offset;

  // Map a virtual address range for the multicast memory. Multicast
  // memory is used to reduce the data stored in the multicast object.
  CUdeviceptr multicast_device_ptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemAddressReserve(
      &multicast_device_ptr, padded_size_, granularity_, 0, 0)));

  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemMap(multicast_device_ptr, padded_size_, 0, handle_, 0)));

  CUmemAccessDesc accessDesc = GetVmmAccessDescriptor(cuda_executor->device_);
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemSetAccess(multicast_device_ptr, padded_size_, &accessDesc, 1)));

  absl::MutexLock subscription_lock(mapped_devices_mu_);
  mapped_devices_.emplace(cuda_executor->device_, multicast_device_ptr);
  void* multicast_address = reinterpret_cast<void*>(multicast_device_ptr);
  XLA_VLOG_DEVICE(3, cuda_executor->device_ordinal())
      << "Mapped address: " << location.opaque()
      << " to multimem address: " << multicast_address;
  return multicast_address;
}

}  // namespace gpu
}  // namespace stream_executor
