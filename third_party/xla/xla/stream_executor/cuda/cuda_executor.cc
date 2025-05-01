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
#include "absl/base/casts.h"
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
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_command_buffer.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/cuda/cuda_kernel.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/cuda_stream.h"
#include "xla/stream_executor/cuda/cuda_timer.h"
#include "xla/stream_executor/cuda/cuda_version_parser.h"
#include "xla/stream_executor/cuda/tma_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/read_numa_node.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
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
          LOG(ERROR) << "failed to load PTX text as a module: " << status;
          // As a precaution for null termination of the API-provided value,
          // ensure that at least the last byte is null.
          error_log_buffer[error_log_buffer_bytes ? error_log_buffer_bytes - 1
                                                  : 0] = '\0';
          LOG(ERROR) << "error log buffer (" << error_log_buffer_bytes
                     << " bytes): " << error_log_buffer.data();
          if (absl::StrContains(error_log_buffer.data(),
                                "Register allocation failed")) {
            returned_status = absl::ResourceExhaustedError(
                absl::StrFormat("Failed to load PTX text as a module (register "
                                "allocation failed): %s",
                                status.ToString()));
          } else {
            returned_status = status;
          }
          notification.Notify();
          return;
        }

        VLOG(3) << "PTX compilation info log (" << info_log_buffer_bytes
                << " bytes): " << info_log_buffer.data();
        VLOG(3) << "PTX compilation error log (" << error_log_buffer_bytes
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
      "Failed to load in-memory CUBIN (compiled for a different GPU?)."));
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
        absl::StrCat("There was an error before calling cuModuleGetFunction (",
                     cuda_error, "): ", cudaGetErrorName(cuda_error), " : ",
                     cudaGetErrorString(cuda_error)));
  }
  CUfunction function;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuModuleGetFunction(&function, module, kernel_name),
                     "Failed to get module function"));
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
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << status;
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
absl::Status GetComputeCapability(int* cc_major, int* cc_minor,
                                  CUdevice device) {
  *cc_major = 0;
  *cc_minor = 0;

  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDeviceGetAttribute(
      cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)));

  return cuda::ToStatus(cuDeviceGetAttribute(
      cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
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

bool CanEnablePeerAccess(Context* from, Context* to) {
  if (from == to) {
    return true;  // A context can always access its own memory.
  }

  auto from_device = DeviceFromContext(from);
  if (!from_device.ok()) {
    LOG(ERROR) << "failed to resolve 'from' peer access context to a device: "
               << from_device.status();
    return false;
  }
  auto to_device = DeviceFromContext(to);
  if (!to_device.ok()) {
    LOG(ERROR) << "failed to resolve 'to' peer access context to a device: "
               << to_device.status();
    return false;
  }
  return CanEnablePeerAccess(from_device.value(), to_device.value());
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
    LOG(ERROR) << "failed to query total available memory: " << status;
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
    LOG(ERROR) << "failed to query ECC status: " << status;
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
    LOG(ERROR) << "failed to query PCI bus id for device: " << status;
    return "";
  }
  if (!absl::c_linear_search(raw_pci_bus_id, '\0')) {
    LOG(ERROR) << "PCI bus id is not null terminated.";
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
  VLOG(2) << "allocated " << ptr << " for context " << context << " of "
          << bytes << " bytes";
  return ptr;
}

// Deallocates memory on the GPU device that was previously allocated via
// DeviceAllocate.
void DeviceDeallocate(Context* context, void* location) {
  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  auto status = cuda::ToStatus(cuMemFree(pointer));
  if (!status.ok()) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << status;
  } else {
    VLOG(2) << "deallocated " << location << " for context " << context;
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
          "Failed to allocate host memory of size %d pinned to NUMA node %d",
          size, numa_node));
    }
    if (size > 0 && !HostRegister(context, buffer, size)) {
      tsl::port::NUMAFree(buffer, size);
      return absl::InternalError(
          absl::StrFormat("Failed to register host memory of size %d pinned to "
                          "NUMA node %d with the GPU driver",
                          size, numa_node));
    }
    return buffer;
  } else {
    ScopedActivateContext activation(context);
    void* buffer = nullptr;
    // "Portable" memory is visible to all CUDA contexts. Safe for our use
    // model.
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuMemHostAlloc(&buffer, size, CU_MEMHOSTALLOC_PORTABLE)));
    if (!buffer && size > 0) {
      return absl::InternalError(absl::StrFormat(
          "Failed to allocate pinned host memory of size %d", size));
    }
    return buffer;
  }
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
      LOG(ERROR) << "error deallocating host memory at " << location << ": "
                 << status;
    }
  }
}

// Creates a MemoryAllocation wrapping the given host buffer.
absl::StatusOr<std::unique_ptr<MemoryAllocation>> AllocateHostMemory(
    CudaContext* cuda_context, int numa_node, uint64_t size) {
  TF_ASSIGN_OR_RETURN(void* ptr, HostAllocate(cuda_context, numa_node, size));
  VLOG(2) << "allocated " << ptr << " for context " << cuda_context << " of "
          << size << " bytes of host memory";
  return std::make_unique<GenericMemoryAllocation>(
      ptr, size, [cuda_context, numa_node](void* location, uint64_t size) {
        HostDeallocate(cuda_context, numa_node, location, size);
        VLOG(2) << "deallocated collective memory at " << location
                << " for context " << cuda_context;
      });
}

}  // namespace

// Given const GPU memory, returns a libcuda device pointer datatype, suitable
// for passing directly to libcuda APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// libcuda APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to libcuda functions which will honor constness.
static CUdeviceptr AsCudaDevicePtr(const DeviceMemoryBase& gpu_mem) {
  return reinterpret_cast<CUdeviceptr>(gpu_mem.opaque());
}

// See description on const version above.
static CUdeviceptr AsCudaDevicePtr(DeviceMemoryBase* gpu_mem) {
  return AsCudaDevicePtr(*gpu_mem);
}

absl::StatusOr<DeviceMemoryBase> CudaExecutor::GetMemoryRange(
    const DeviceMemoryBase& location) {
  CUdeviceptr device_pointer;
  size_t size;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemGetAddressRange(&device_pointer, &size, AsCudaDevicePtr(location))));
  return DeviceMemoryBase(reinterpret_cast<void*>(device_pointer), size);
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

absl::StatusOr<void*> CollectiveMemoryAllocate(StreamExecutor* executor,
                                               uint64_t bytes) {
  if (bytes == 0) return nullptr;

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
CudaExecutor::CreateMemoryAllocator(MemoryType type) {
  if (type == MemoryType::kUnified) {
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
          VLOG(2) << "allocated " << ptr << " for context " << cuda_context_
                  << " of " << size << " bytes in unified memory";
          return std::make_unique<GenericMemoryAllocation>(
              ptr, size, [this](void* location, uint64_t size) {
                std::unique_ptr<ActivateContext> activation = Activate();
                CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
                auto status = cuda::ToStatus(cuMemFree(pointer));
                if (!status.ok()) {
                  LOG(ERROR) << "failed to free unified memory at " << location
                             << "; result: " << status;
                } else {
                  VLOG(2) << "deallocated unified memory at " << location
                          << " for context " << cuda_context_;
                }
              });
        });
  } else if (type == MemoryType::kCollective) {
    return std::make_unique<GenericMemoryAllocator>(
        [this](uint64_t size)
            -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          TF_ASSIGN_OR_RETURN(void* ptr, CollectiveMemoryAllocate(this, size));
          VLOG(2) << "allocated " << ptr << " for context " << cuda_context_
                  << " of " << size << " bytes of collective memory";
          return std::make_unique<GenericMemoryAllocation>(
              ptr, size, [this](void* location, uint64_t size) {
                auto status = CollectiveMemoryDeallocate(this, location);
                if (!status.ok()) {
                  LOG(ERROR) << "failed to free collective memory at "
                             << location << "; result: " << status;
                } else {
                  VLOG(2) << "deallocated collective memory at " << location
                          << " for context " << cuda_context_;
                }
              });
        });
  } else if (type == MemoryType::kHost) {
    return std::make_unique<GenericMemoryAllocator>([this](uint64_t size) {
      return AllocateHostMemory(cuda_context_, numa_node_, size);
    });
  }
  return absl::UnimplementedError(
      absl::StrFormat("Unsupported memory type %d", type));
}

absl::Status CudaExecutor::Init() {
  TF_ASSIGN_OR_RETURN(device_, GetDevice(device_ordinal()));
  TF_ASSIGN_OR_RETURN(CudaContext * context,
                      CudaContext::Create(device_ordinal(), device_));
  cuda_context_ = context;
  TF_RETURN_IF_ERROR(GetComputeCapability(&cc_major_, &cc_minor_, device_));
  TF_ASSIGN_OR_RETURN(delay_kernels_supported_, DelayKernelIsSupported());
  numa_node_ = ReadNumaNode(GetPCIBusID(device_), device_ordinal())
                   .value_or(tsl::port::kNUMANoAffinity);
  if (numa_node_ == tsl::port::kNUMANoAffinity) {
    VLOG(2) << "Could not determine NUMA node of device ordinal "
            << device_ordinal();
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
    VLOG(3) << "Loaded CUBIN " << static_cast<const void*>(cubin)
            << " as module " << module;
  } else {
    ++module_refcount;
    VLOG(3) << "CUBIN " << static_cast<const void*>(cubin)
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
    VLOG(3) << "Loaded PTX " << static_cast<const void*>(ptx) << " as module "
            << module;
    module_refcount = 1;
  } else {
    ++module_refcount;
    VLOG(3) << "PTX " << static_cast<const void*>(ptx)
            << " is already loaded as module " << module;
  }
  gpu_binary_to_module_[module_handle] = {module, module_refcount};
  return module_handle;
}

absl::StatusOr<std::unique_ptr<Kernel>> CudaExecutor::LoadKernel(
    const MultiKernelLoaderSpec& spec) {
  auto cuda_kernel = std::make_unique<CudaKernel>(this);
  const std::string* kernel_name;

  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    kernel_name = &spec.cuda_cubin_in_memory().kernel_name();
    const char* cubin = reinterpret_cast<const char*>(
        spec.cuda_cubin_in_memory().cubin_bytes().data());
    TF_ASSIGN_OR_RETURN(ModuleHandle module_handle, LoadModuleFromCuBin(cubin));
    kernel_to_gpu_binary_[cuda_kernel.get()] = module_handle;

    CUmodule module = gpu_binary_to_module_.at(module_handle).first;
    VLOG(2) << "getting function " << *kernel_name << " from module " << module;
    TF_ASSIGN_OR_RETURN(
        CUfunction function,
        GetModuleFunction(cuda_context_, module, kernel_name->c_str()));
    cuda_kernel->set_gpu_function(function);

  } else if (spec.has_cuda_ptx_in_memory()) {
    kernel_name = &spec.cuda_ptx_in_memory().kernel_name();

    if (cc_major_ == 0 && cc_minor_ == 0) {
      return absl::InternalError("Compute capability not set");
    }

    const char* ptx = spec.cuda_ptx_in_memory().text(cc_major_, cc_minor_);
    if (ptx == nullptr) {
      ptx = spec.cuda_ptx_in_memory().default_text();
    }
    if (ptx == nullptr) {
      LOG(FATAL) << "Loader spec has no ptx for kernel " << *kernel_name;
    }

    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_ASSIGN_OR_RETURN(ModuleHandle module_handle, LoadModuleFromPtx(ptx));
    kernel_to_gpu_binary_[cuda_kernel.get()] = module_handle;

    CUmodule module = gpu_binary_to_module_.at(module_handle).first;
    VLOG(2) << "getting function " << *kernel_name << " from module " << module;
    TF_ASSIGN_OR_RETURN(
        CUfunction function,
        GetModuleFunction(cuda_context_, module, kernel_name->c_str()));
    cuda_kernel->set_gpu_function(function);

  } else if (spec.has_in_process_symbol()) {
    kernel_name = &spec.in_process_symbol().kernel_name();
    void* symbol = spec.in_process_symbol().symbol();

    VLOG(2) << "Resolve CUDA kernel " << *kernel_name
            << " from symbol pointer: " << symbol;
    cudaFunction_t func;
    TF_RETURN_IF_ERROR(cuda::ToStatus(cudaGetFuncBySymbol(&func, symbol),
                                      "Failed call to cudaGetFuncBySymbol"));
    cuda_kernel->set_gpu_function(func);

  } else {
    return absl::InternalError("No method of loading CUDA kernel provided");
  }
  VLOG(3) << "LoadKernel on kernel : " << *kernel_name;

  {
    // Keep track of loaded kernels.
    absl::MutexLock lock{&in_memory_modules_mu_};
    loaded_kernels_.insert(cuda_kernel.get());
  }

  // Update CUDA kernel properties after it was loaded in the CUDA context.
  cuda_kernel->set_name(*kernel_name);

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the CUDA API.
  cuda_kernel->set_arity(spec.arity());

  TF_ASSIGN_OR_RETURN(KernelMetadata kernel_metadata,
                      cuda_kernel->GetKernelMetadata());
  cuda_kernel->set_metadata(kernel_metadata);
  cuda_kernel->set_name(*kernel_name);
  cuda_kernel->set_args_packing(spec.kernel_args_packing());
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
    VLOG(3) << "No loaded CUDA module for " << gpu_binary;
    return false;
  }
  auto& module = module_it->second.first;
  auto& refcount = module_it->second.second;
  VLOG(3) << "Found CUDA module " << module << " with refcount " << refcount;
  if (--refcount == 0) {
    VLOG(3) << "Unloading CUDA module " << module;
    UnloadCudaModule(cuda_context_, module);
    gpu_binary_to_module_.erase(module_it);
  }
  return true;
}

void CudaExecutor::UnloadKernel(const Kernel* kernel) {
  VLOG(3) << "Unloading kernel " << kernel << " : " << kernel->name();

  absl::MutexLock lock{&in_memory_modules_mu_};
  loaded_kernels_.erase(kernel);

  auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
  if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
    // We might never see kernel being explicitly loaded if it was resolved from
    // in process symbol pointer (CUDA C++ device function pointer).
    VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
            << " has never been loaded.";
    return;
  }
  VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
          << " has loaded GPU code " << gpu_binary_it->second;
  UnloadGpuBinary(gpu_binary_it->second);
  kernel_to_gpu_binary_.erase(gpu_binary_it);
}

absl::StatusOr<ModuleHandle> CudaExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  // We store the pointer to the GPU binary (PTX or CUBIN) as
  // ModuleHandle::id().
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    return LoadModuleFromCuBin(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()));
  } else if (spec.has_cuda_ptx_in_memory()) {
    if (cc_major_ == 0 && cc_minor_ == 0) {
      return absl::InternalError("Compute capability not set");
    }

    if (!spec.cuda_ptx_in_memory()) {
      return absl::InternalError("PTX not found in spec");
    }

    absl::MutexLock lock{&in_memory_modules_mu_};
    return LoadModuleFromPtx(spec.cuda_ptx_in_memory());
  }
  return absl::InternalError("No method of loading CUDA module provided");
}

bool CudaExecutor::UnloadModule(ModuleHandle module_handle) {
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(module_handle);
}

namespace {
absl::uint128 Fingerprint128(const absl::string_view s) {
  auto fp = tsl::Fingerprint128(s);
  return absl::MakeUint128(fp.high64, fp.low64);
}

int fpus_per_core(int cc_major, int cc_minor) {
  // Source:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
  int n = 128;          // 5.x, 6.1, 6.2, 8.6, 9.0 -> 128.
  if (cc_major == 3) {  // 3.x -> 192.
    n = 192;
  } else if ((cc_major == 6 && cc_minor == 0) || (cc_major == 7) ||
             (cc_major == 8 && cc_minor == 0)) {
    n = 64;  // 6.0, 7.x, 8.0 -> 64.
  }
  return n;
}

}  // namespace

absl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
CudaExecutor::CreateOrShareConstant(Stream* stream,
                                    absl::Span<const uint8_t> content) {
  absl::MutexLock lock{&shared_constants_mu_};
  // We assume all constants are uniquely identified by this hash. In the
  // (highly unlikely) event of a hash collision, the program will likely crash
  // (because the cached constant that will be returned by mistake is unlikely
  // to have the correct size).
  absl::uint128 fingerprint = Fingerprint128(absl::string_view(
      reinterpret_cast<const char*>(content.data()), content.size()));
  // Must insert nullptr first to get an iterator to the insertion point.
  auto insert_result = shared_constants_.insert(
      {fingerprint, std::weak_ptr<DeviceMemoryBase>()});
  auto it = insert_result.first;
  bool was_already_in_cache = !insert_result.second;
  std::shared_ptr<DeviceMemoryBase> shared_constant;

  if (was_already_in_cache) {
    shared_constant = it->second.lock();
  }

  if (shared_constant == nullptr) {
    // Either the constant wasn't found in the cache, or it was but its
    // weak_ptr had expired.
    auto new_constant = std::make_unique<DeviceMemoryBase>(
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
    shared_constant = std::shared_ptr<DeviceMemoryBase>(
        new_constant.release(), [this](DeviceMemoryBase* p) {
          Deallocate(p);
          delete p;
        });
    it->second = std::weak_ptr<DeviceMemoryBase>(shared_constant);
  }

  return shared_constant;
}

DeviceMemoryBase CudaExecutor::Allocate(uint64_t size, int64_t memory_space) {
  VLOG(1) << "CudaExecutor::Allocate size: " << size
          << " memory_space: " << memory_space;

  if (memory_space == static_cast<int64_t>(MemoryType::kCollective)) {
    auto result = CollectiveMemoryAllocate(this, size);
    if (!result.ok()) {
      LOG(ERROR) << "Failed to allocate collective memory: " << result.status();
      return DeviceMemoryBase(nullptr, 0);
    }
    VLOG(1) << "CudaExecutor::Allocate returns " << result.value();
    return DeviceMemoryBase(result.value(), size);
  } else if (memory_space ==
             static_cast<int64_t>(stream_executor::MemoryType::kHost)) {
    auto result = HostAllocate(cuda_context_, numa_node_, size);
    if (!result.ok()) {
      LOG(ERROR) << "Failed to allocate host memory: " << result.status();
      return DeviceMemoryBase(nullptr, 0);
    }
    VLOG(1) << "CudaExecutor::Allocate returns " << result.value();
    return DeviceMemoryBase(result.value(), size);
  }
  CHECK_EQ(memory_space, 0);
  auto device_buf_base = DeviceAllocate(cuda_context_, size);
  VLOG(1) << "CudaExecutor::Allocate returns " << device_buf_base;
  return DeviceMemoryBase(device_buf_base, size);
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
CudaExecutor::HostMemoryAllocate(uint64_t size) {
  return AllocateHostMemory(cuda_context_, numa_node_, size);
}

void CudaExecutor::Deallocate(DeviceMemoryBase* mem) {
  VLOG(1) << "CudaExecutor::Deallocate mem: " << mem->opaque();

  auto status_or_memory_space = GetPointerMemorySpace(mem->opaque());
  if (!status_or_memory_space.ok()) {
    LOG(ERROR) << status_or_memory_space.status();
    return;
  }
  auto memory_space = status_or_memory_space.value();
  if (memory_space == MemoryType::kHost) {
    HostDeallocate(cuda_context_, numa_node_, mem->opaque(), mem->size());
  } else {
    DeviceDeallocate(cuda_context_, mem->opaque());
  }
}

bool CudaExecutor::SynchronizeAllActivity() {
  return cuda_context_->Synchronize().ok();
}

bool CudaExecutor::HostMemoryRegister(void* location, uint64_t size) {
  VLOG(1) << "Called StreamExecutor::HostMemoryRegister(data=" << location
          << ")";
  return HostRegister(cuda_context_, location, size);
}

bool CudaExecutor::HostMemoryUnregister(void* location) {
  VLOG(1) << "Called StreamExecutor::HostUnregister(data=" << location << ")";
  return HostUnregister(cuda_context_, location);
}

absl::Status CudaExecutor::SynchronousMemZero(DeviceMemoryBase* location,
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

absl::Status CudaExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemcpyHtoD(AsCudaDevicePtr(gpu_dst), host_src, size),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: GPU dst: %llx;"
          " host src: %p; size: %u=0x%x",
          AsCudaDevicePtr(gpu_dst), host_src, size, size)));
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return absl::OkStatus();
}

absl::Status CudaExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemcpyDtoH(host_dst, AsCudaDevicePtr(gpu_src), size),
      absl::StrFormat("failed to synchronous memcpy from device to host "
                      "host dst: %p; GPU src: %llx; size: %u=0x%x",
                      host_dst, AsCudaDevicePtr(gpu_src), size, size)));
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return absl::OkStatus();
}

void CudaExecutor::DeallocateStream(Stream* stream) {
  {
    absl::MutexLock lock(&mu_);
    if (dnn_ != nullptr) {
      dnn_->NotifyStreamDestroyed(stream);
    }
  }
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(stream->platform_specific_handle().stream);
}

blas::BlasSupport* CudaExecutor::AsBlas() {
  absl::MutexLock lock(&mu_);
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
  absl::MutexLock lock(&mu_);
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
  absl::MutexLock lock(&mu_);
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
  return CanEnablePeerAccess(cuda_context_, cuda_other->cuda_context_);
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

absl::StatusOr<DeviceMemoryBase> CudaExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  void* mem = nullptr;
  size_t bytes = 0;
  CHECK(static_cast<bool>(module_handle));

  {  // give limited scope to MutexLock
    absl::MutexLock lock{&in_memory_modules_mu_};
    auto it = gpu_binary_to_module_.find(module_handle);
    CHECK(it != gpu_binary_to_module_.end());

    CUmodule gpu_module_handle = it->second.first;
    CHECK(gpu_module_handle != nullptr);
    TF_RETURN_IF_ERROR(
        GetModuleSymbol(cuda_context_, gpu_module_handle, symbol_name.c_str(),
                        reinterpret_cast<CUdeviceptr*>(&mem), &bytes));
    return DeviceMemoryBase(mem, bytes);
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
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_[stream->stream_handle()] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>>
CudaExecutor::CreateCommandBuffer(CommandBuffer::Mode mode) {
  VLOG(2) << "Create CUDA command buffer (CUDA graph)";
  return CudaCommandBuffer::Create(mode, this, cuda_context_);
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
CudaExecutor::CreateDeviceDescription(int device_ordinal) {
  TF_ASSIGN_OR_RETURN(CUdevice device, GetDevice(device_ordinal));

  int cc_major;
  int cc_minor;
  TF_RETURN_IF_ERROR(GetComputeCapability(&cc_major, &cc_minor, device));

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

  {
    std::string pci_bus_id = GetPCIBusID(device);
    desc.set_pci_bus_id(pci_bus_id);

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
  desc.set_clock_rate_ghz(static_cast<float>(sm_clock_khz) / 1e6);

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
    // Times 2 because HBM is DDR memory; it gets two data bits per each data
    // lane.
    desc.set_memory_bandwidth(2 * int64_t{mem_clock_khz.value()} * 1000 *
                              int64_t{mem_bus_width_bits.value()} / 8);
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

  desc.set_platform_version(
      absl::StrCat("Compute Capability ", cc_major, ".", cc_minor));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  desc.set_device_address_bits(64);

  desc.set_device_vendor("NVIDIA Corporation");
  desc.set_cuda_compute_capability(cc_major, cc_minor);
  desc.set_shared_memory_per_core(GetMaxSharedMemoryPerCore(device).value());
  desc.set_shared_memory_per_block(GetMaxSharedMemoryPerBlock(device).value());
  desc.set_shared_memory_per_block_optin(
      GetMaxSharedMemoryPerBlockOptin(device).value());
  int core_count = GetMultiprocessorCount(device).value();
  desc.set_core_count(core_count);
  desc.set_fpus_per_core(fpus_per_core(cc_major, cc_minor));
  desc.set_threads_per_core_limit(
      GetMaxThreadsPerMultiprocessor(device).value());
  desc.set_registers_per_block_limit(GetMaxRegistersPerBlock(device).value());
  desc.set_threads_per_warp(GetThreadsPerWarp(device).value());
  desc.set_registers_per_core_limit(
      GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                         device)
          .value());

  auto value_or = [](const auto& status_or, auto default_val) {
    if (status_or.ok()) return *status_or;
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
      "sm_%d.%d with %dB RAM, %d cores, %dKHz clock, %dKHz mem clock, %dB L2$",
      cc_major, cc_minor, device_memory_size, core_count, sm_clock_khz,
      value_or(mem_clock_khz, 0), l2_cache_bytes));

  return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<MemoryType> CudaExecutor::GetPointerMemorySpace(
    const void* ptr) {
  CUdeviceptr pointer = reinterpret_cast<CUdeviceptr>(const_cast<void*>(ptr));
  unsigned int value;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuPointerGetAttribute(
      &value, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer)));
  switch (value) {
    case CU_MEMORYTYPE_DEVICE:
      return MemoryType::kDevice;
    case CU_MEMORYTYPE_HOST:
      return MemoryType::kHost;
    default:
      return absl::InternalError(
          absl::StrCat("unknown memory space provided by CUDA API: ", value));
  }
}

absl::StatusOr<const CudaKernel*> CudaExecutor::GetCudaKernel(
    const Kernel* kernel) {
  absl::MutexLock lock{&in_memory_modules_mu_};
  auto it = loaded_kernels_.find(kernel);
  if (it == loaded_kernels_.end()) {
    return absl::NotFoundError("Kernel not loaded in this executor.");
  }
  return static_cast<const CudaKernel*>(*it);
}

absl::StatusOr<TensorMap> CudaExecutor::CreateTensorMap(TmaDescriptor tma_desc,
                                                        void* global_address) {
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
      &tma_desc.global_dims()[0], &tma_desc.global_strides()[0],
      &tma_desc.box_dims()[0], &tma_desc.element_strides()[0], interleave,
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

}  // namespace gpu
}  // namespace stream_executor
