/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_executor.h"

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hip/hip_version.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
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
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/rocm/rocm_command_buffer.h"
#include "xla/stream_executor/rocm/rocm_context.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_kernel.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/rocm/rocm_stream.h"
#include "xla/stream_executor/rocm/rocm_timer.h"
#include "xla/stream_executor/rocm/rocm_version_parser.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/numbers.h"

namespace stream_executor {
namespace gpu {

namespace {
// Given const GPU memory, returns a librocm device pointer datatype, suitable
// for passing directly to librocm APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// librocm APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to librocm functions which will honor constness.
hipDeviceptr_t AsROCmDevicePtr(const DeviceMemoryBase& gpu_mem) {
  return const_cast<hipDeviceptr_t>(gpu_mem.opaque());
}

// See description on const version above.
hipDeviceptr_t AsROCmDevicePtr(DeviceMemoryBase* gpu_mem) {
  return AsROCmDevicePtr(*gpu_mem);
}

absl::uint128 Fingerprint128(const absl::string_view s) {
  auto fp = tsl::Fingerprint128(s);
  return absl::MakeUint128(fp.high64, fp.low64);
}

int fpus_per_core(std::string gcn_arch_name) {
  // Source:
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
  int n = 128;  // gfx90a and gfx908 -> 128
  if (gcn_arch_name.substr(0, 6) == "gfx906") {
    n = 64;
  }
  return n;
}

// ROCM driver routines may require a large amount of stack (particularly
// hipModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
tsl::thread::ThreadPool* GetDriverExecutor() {
  static tsl::thread::ThreadPool* thread_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), tsl::ThreadOptions(), "rocm_driver", 1);
  return thread_pool;
}

// Loads HSACO with the ROCM runtime and stores the resulting handle in
// "module". Any error logs that are produced are logged internally.
absl::StatusOr<hipModule_t> LoadHsaco(Context* context,
                                      const char* hsaco_contents) {
  absl::Notification notification;
  absl::Status returned_status = absl::OkStatus();
  hipModule_t module;
  GetDriverExecutor()->Schedule(
      [context, hsaco_contents, &module, &returned_status, &notification]() {
        ScopedActivateContext activation(context);
        hipError_t res = wrap::hipModuleLoadData(&module, hsaco_contents);

        if (res != hipSuccess) {
          returned_status = absl::InternalError(
              absl::StrCat("Failed to load HSACO: ", ToString(res)));
          notification.Notify();
        }

        CHECK(module != nullptr);
        notification.Notify();
      });
  notification.WaitForNotification();

  TF_RETURN_IF_ERROR(returned_status);
  return module;
}

// Retrieves a named kernel from a loaded module, and places the resulting
// handle into function (outparam) on success. Neither kernel_name nor
// function may be null. No ownership is taken of kernel_name.
absl::StatusOr<hipFunction_t> GetModuleFunction(Context* context,
                                                hipModule_t module,
                                                const char* kernel_name) {
  ScopedActivateContext activated(context);
  CHECK(module != nullptr && kernel_name != nullptr);
  hipFunction_t function;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipModuleGetFunction(&function, module, kernel_name),
               "Failed to get kernel"));
  return function;
}

// Retrieves a named global/constant symbol from a loaded module, and returns
// a device pointer and size of the symbol on success. symbol_name may not be
// null. At least one of dptr or bytes should not be null. No ownership is
// taken of symbol_name.
absl::Status GetModuleSymbol(Context* context, hipModule_t module,
                             const char* symbol_name, hipDeviceptr_t* dptr,
                             size_t* bytes) {
  ScopedActivateContext activated(context);
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  return ToStatus(wrap::hipModuleGetGlobal(dptr, bytes, module, symbol_name),
                  absl::StrCat("Failed to get symbol '", symbol_name, "'"));
}

// Unloads module from the current context via cuModuleUnload.
void UnloadRocmModule(Context* context, hipModule_t module) {
  ScopedActivateContext activated(context);
  hipError_t res = wrap::hipModuleUnload(module);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << ToString(res);
  }
}

// Returns the name of the device.
absl::StatusOr<std::string> GetDeviceName(hipDevice_t device) {
  static const size_t kCharLimit = 64;
  absl::InlinedVector<char, 4> chars(kCharLimit);
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGetName(chars.begin(), kCharLimit - 1, device),
               "Failed to get device name"));
  chars[kCharLimit - 1] = '\0';
  return chars.begin();
}

absl::StatusOr<int> GetGpuISAVersion(hipDevice_t device) {
  hipDeviceProp_t props;
  hipError_t result = wrap::hipGetDeviceProperties(&props, device);
  if (result == hipSuccess) {
    std::string gcnName = props.gcnArchName;
    std::vector<std::string> tokens = absl::StrSplit(gcnName, ':');
    std::string amdgpu_version = gcnName;
    if (!tokens.empty() && tokens[0].size() >= 3) {
      amdgpu_version = tokens[0].substr(3);
    }
    int version = std::stoi(amdgpu_version);
    return version;
  }
  return absl::InternalError(absl::StrFormat(
      "failed to determine AMDGpu ISA version for device %d", device));
}

// Return the full GCN Architecture Name for the device
// for eg: amdgcn-amd-amdhsa--gfx908:sramecc+:xnack-
absl::StatusOr<std::string> GetGpuGCNArchName(hipDevice_t device) {
  hipDeviceProp_t props;
  hipError_t result = wrap::hipGetDeviceProperties(&props, device);
  if (result == hipSuccess) {
    return props.gcnArchName;
  }
  return absl::InternalError(absl::StrFormat(
      "failed to determine AMDGpu GCN Arch Name for device %d", device));
}

// Helper function that turns the integer output of hipDeviceGetAttribute to
// type T and wraps it in a absl::StatusOr.
template <typename T>
static absl::StatusOr<T> GetSimpleAttribute(hipDevice_t device,
                                            hipDeviceAttribute_t attribute) {
  int value = -1;
  hipError_t result = wrap::hipDeviceGetAttribute(&value, attribute, device);
  if (result != hipSuccess) {
    return absl::NotFoundError(
        absl::StrCat("could not retrieve ROCM device attribute (", attribute,
                     "): ", ToString(result)));
  }
  T converted = value;
  return converted;
}

// Returns the number of multiprocessors on the device (note that the device
// may be multi-GPU-per-board).

absl::StatusOr<int> GetMultiprocessorCount(hipDevice_t device) {
  return GetSimpleAttribute<int>(device, hipDeviceAttributeMultiprocessorCount);
}

absl::StatusOr<int64_t> GetMaxSharedMemoryPerCore(hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(
      device, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor);
}

absl::StatusOr<int64_t> GetMaxSharedMemoryPerBlock(hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxSharedMemoryPerBlock);
}

absl::StatusOr<int64_t> GetMaxThreadsPerMultiprocessor(hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(
      device, hipDeviceAttributeMaxThreadsPerMultiProcessor);
}

absl::StatusOr<int64_t> GetMaxRegistersPerBlock(hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxRegistersPerBlock);
}

absl::StatusOr<int64_t> GetThreadsPerWarp(hipDevice_t device) {
  // TODO(ROCm): This is almost certainly wrong but tests seem to rely on it.
  return 32;
}

absl::Status GetGridLimits(int* x, int* y, int* z, hipDevice_t device) {
  int value;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGetAttribute(
                   &value, hipDeviceAttributeMaxGridDimX, device),
               "failed to query max grid dim x"));
  *x = value;

  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGetAttribute(
                   &value, hipDeviceAttributeMaxGridDimY, device),
               "failed to query max grid dim y"));
  *y = value;

  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGetAttribute(
                   &value, hipDeviceAttributeMaxGridDimZ, device),
               "failed to query max grid dim z"));
  *z = value;
  return absl::OkStatus();
}

// Returns the device associated with the given device_ordinal.
absl::StatusOr<hipDevice_t> GetDevice(int device_ordinal) {
  hipDevice_t device;
  hipError_t res = wrap::hipDeviceGet(&device, device_ordinal);
  if (res == hipSuccess) {
    return device;
  }

  return absl::InternalError(
      absl::StrCat("failed call to hipDeviceGet: ", ToString(res)));
}

// Returns the device associated with the given context.
absl::StatusOr<hipDevice_t> DeviceFromContext(Context* context) {
  ScopedActivateContext activated(context);
  hipDevice_t device = -1;
  hipError_t result = wrap::hipCtxGetDevice(&device);
  if (result == hipSuccess) return device;

  return absl::InternalError(
      absl::StrCat("failed to get device for context: ", ToString(result)));
}

bool CanEnablePeerAccess(hipDevice_t from, hipDevice_t to) {
  int can_access_peer = -1;
  hipError_t result = wrap::hipDeviceCanAccessPeer(&can_access_peer, from, to);
  if (result != hipSuccess) {
    LOG(ERROR) << "failed to detect peer access capability: "
               << ToString(result);
    return false;
  }
  return can_access_peer;
}

bool CanEnablePeerAccess(Context* from, Context* to) {
  // A context can always access its own memory.
  if (from == to) return true;

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
    return absl::OkStatus();  // A device can always access its own memory.
  }

  ScopedActivateContext activated(from);
  hipError_t result = wrap::hipCtxEnablePeerAccess(
      tensorflow::down_cast<RocmContext*>(to)->context(), 0 /* = flags */);
  if (result != hipSuccess && result != hipErrorPeerAccessAlreadyEnabled) {
    return absl::InternalError(
        absl::StrFormat("failed to enable peer access from %d to %d: %s",
                        from->device_ordinal(), to->device_ordinal(),
                        ToString(result).c_str()));
  }

  return absl::OkStatus();
}

std::string GetPCIBusID(hipDevice_t device) {
  std::string pci_bus_id;
  static const int kBufferSize = 64;
  absl::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  hipError_t res =
      wrap::hipDeviceGetPCIBusId(chars.begin(), kBufferSize - 1, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << ToString(res);
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

bool GetDeviceProperties(hipDeviceProp_t* device_properties,
                         int device_ordinal) {
  hipError_t res =
      wrap::hipGetDeviceProperties(device_properties, device_ordinal);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  return true;
}

// Allocates memory on the GPU device.
void* DeviceAllocate(Context* context, uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  ScopedActivateContext activated(context);
  hipDeviceptr_t result = 0;
  hipError_t res = wrap::hipMalloc(&result, bytes);
  if (res != hipSuccess) {
    // LOG(INFO) because this isn't always important to users (e.g. BFCAllocator
    // implements a retry if the first allocation fails).
    LOG(INFO) << "failed to allocate "
              << tsl::strings::HumanReadableNumBytes(bytes) << " (" << bytes
              << " bytes) from device: " << ToString(res);
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for device " << context->device_ordinal()
          << " of " << bytes << " bytes";
  return ptr;
}

// Deallocates memory on the GPU device that was previously allocated via
// DeviceAllocate.
void DeviceDeallocate(Context* context, void* location) {
  ScopedActivateContext activation(context);
  hipDeviceptr_t pointer = absl::bit_cast<hipDeviceptr_t>(location);
  hipError_t res = wrap::hipFree(pointer);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for device "
            << context->device_ordinal();
  }
}

// Allocates memory on the host.
absl::StatusOr<void*> HostAllocate(Context* context, uint64_t bytes) {
  ScopedActivateContext activation(context);
  void* host_mem = nullptr;
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipHostMalloc(&host_mem, bytes, hipHostMallocPortable),
               "failed to allocate host memory"));
  return host_mem;
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> AllocateHostMemory(
    RocmContext* rocm_context, uint64_t size) {
  TF_ASSIGN_OR_RETURN(void* ptr, HostAllocate(rocm_context, size));
  VLOG(2) << "allocated " << ptr << " for context " << rocm_context << " of "
          << size << " bytes of host memory";
  return std::make_unique<GenericMemoryAllocation>(
      ptr, size, [rocm_context](void* location, uint64_t size) {
        hipError_t res = wrap::hipHostFree(location);
        if (res != hipSuccess) {
          LOG(ERROR) << "error deallocating host memory at " << location << ": "
                     << ToString(res);
        }
        VLOG(2) << "deallocated host memory at " << location << " for context "
                << rocm_context;
      });
}

}  // namespace

RocmExecutor::~RocmExecutor() {
  for (auto& it : in_memory_modules_) {
    UnloadRocmModule(rocm_context_, it.second);
  }
  CHECK(kernel_to_gpu_binary_.empty()) << "RocmExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "RocmExecutor has loaded modules.";
}

std::unique_ptr<ActivateContext> RocmExecutor::Activate() {
  return std::make_unique<ScopedActivateContext>(rocm_context_);
}

bool RocmExecutor::UnloadModule(ModuleHandle module_handle) {
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(module_handle);
}

absl::StatusOr<DeviceMemoryBase> RocmExecutor::GetMemoryRange(
    const DeviceMemoryBase& location) {
  hipDeviceptr_t device_pointer;
  size_t size;
  hipError_t result = wrap::hipMemGetAddressRange(
      &device_pointer, &size, const_cast<void*>(location.opaque()));
  if (result == hipSuccess) {
    return DeviceMemoryBase(device_pointer, size);
  } else if (result == hipErrorNotFound) {
    // We differentiate between "this pointer is unknown" (return here) and
    // "there was an internal error while performing this operation" (return
    // below).
    return absl::NotFoundError(absl::StrFormat("not a device pointer %p; %s",
                                               location.opaque(),
                                               ToString(result).c_str()));
  }

  return absl::InternalError(
      absl::StrFormat("failed to get pointer into for device pointer %p; %s",
                      location.opaque(), ToString(result).c_str()));
}

absl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
RocmExecutor::CreateOrShareConstant(Stream* stream,
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
    DeviceMemoryBase* new_constant =
        new DeviceMemoryBase(Allocate(content.size(), /*memory_space=*/0));
    if (new_constant->opaque() == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Failed to allocate %d bytes for new constant", content.size()));
    }

    TF_RETURN_IF_ERROR(
        stream->Memcpy(new_constant, content.data(), content.size()));
    absl::Status status = stream->BlockHostUntilDone();
    if (!status.ok()) {
      Deallocate(new_constant);
      status.Update(absl::InternalError(absl::StrFormat(
          "Memcpy to device address %p failed", new_constant->opaque())));
      return status;
    }

    // Capturing 'this' in the custom deleter means this executor must
    // outlive all shared uses of this constant.
    shared_constant = std::shared_ptr<DeviceMemoryBase>(
        new_constant, [this](DeviceMemoryBase* p) {
          Deallocate(p);
          delete p;
        });
    it->second = std::weak_ptr<DeviceMemoryBase>(shared_constant);
  }

  return shared_constant;
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>>
RocmExecutor::CreateEventBasedTimer(Stream* stream, bool use_delay_kernel) {
  TF_ASSIGN_OR_RETURN(auto timer, RocmTimer::Create(this, stream));
  return std::make_unique<RocmTimer>(std::move(timer));
}

bool RocmExecutor::UnloadGpuBinary(ModuleHandle module_handle) {
  auto module_it = gpu_binary_to_module_.find(module_handle);
  if (gpu_binary_to_module_.end() == module_it) {
    VLOG(3) << "No loaded  HSACO module for " << module_handle;
    return false;
  }
  auto& module = module_it->second.first;
  auto& refcount = module_it->second.second;
  VLOG(3) << "Found HSACO module " << module << " with refcount " << refcount;
  if (--refcount == 0) {
    VLOG(3) << "Unloading  HSACO module " << module;
    UnloadRocmModule(rocm_context_, module);
    gpu_binary_to_module_.erase(module_it);
    ModuleHandle mem_it{};
    for (auto x : in_memory_modules_) {
      if (x.second == module) mem_it = x.first;
    }
    if (mem_it != ModuleHandle{}) in_memory_modules_.erase(mem_it);
  }
  return true;
}

void RocmExecutor::UnloadKernel(const Kernel* kernel) {
  VLOG(3) << "Unloading kernel " << kernel << " : " << kernel->name();

  absl::MutexLock lock{&in_memory_modules_mu_};
  loaded_kernels_.erase(kernel);
  auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
  if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
    VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
            << " has never been loaded.";
    return;  // We've never seen this kernel.
  }
  VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
          << " has loaded GPU code " << gpu_binary_it->second;
  UnloadGpuBinary(gpu_binary_it->second);
  kernel_to_gpu_binary_.erase(gpu_binary_it);
}

absl::Status RocmExecutor::Init() {
  TF_ASSIGN_OR_RETURN(device_, GetDevice(device_ordinal()));

  TF_ASSIGN_OR_RETURN(rocm_context_,
                      RocmContext::Create(device_ordinal(), device_));
  TF_ASSIGN_OR_RETURN(version_, GetGpuISAVersion(device_));
  // We initialize BLAS interfaces early here since otherwise it might create
  // us problems during hipBlasLt initialization under graph capture.
  // There is no real advantage of explicitly using 'lazy initialization' on
  // ROCM platform because rocBLAS/hipBlasLt already use 'lazy initialization'
  // internally
  return InitBlas();
}

absl::StatusOr<std::unique_ptr<Kernel>> RocmExecutor::LoadKernel(
    const MultiKernelLoaderSpec& spec) {
  auto rocm_kernel = std::make_unique<RocmKernel>(this);
  const std::string* kernel_name;

  if (spec.has_cuda_cubin_in_memory()) {
    kernel_name = &spec.cuda_cubin_in_memory().kernel_name();

    const char* hsaco = reinterpret_cast<const char*>(
        spec.cuda_cubin_in_memory().cubin_bytes().data());
    absl::MutexLock lock{&in_memory_modules_mu_};
    ModuleHandle module_handle{hsaco};
    hipModule_t& module = in_memory_modules_[module_handle];

    if (module == nullptr) {
      TF_ASSIGN_OR_RETURN(module, LoadHsaco(rocm_context_, hsaco));
    }
    kernel_to_gpu_binary_[rocm_kernel.get()] = module_handle;

    VLOG(2) << "getting function " << *kernel_name << " from module " << module;
    TF_ASSIGN_OR_RETURN(
        hipFunction_t function,
        GetModuleFunction(rocm_context_, module, kernel_name->c_str()));
    rocm_kernel->set_gpu_function(function);
  } else if (spec.has_in_process_symbol()) {
    kernel_name = &spec.in_process_symbol().kernel_name();
    void* symbol = spec.in_process_symbol().symbol();

    VLOG(1) << "Resolve ROCM kernel " << *kernel_name
            << " from symbol pointer: " << symbol;

#if TF_ROCM_VERSION >= 60200
    hipFunction_t func;
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipGetFuncBySymbol(&func, spec.in_process_symbol().symbol()),
        "Failed call to hipGetFuncBySymbol"));
    rocm_kernel->set_gpu_function(func);
#else
    rocm_kernel->set_gpu_function(
        static_cast<hipFunction_t>(spec.in_process_symbol().symbol()));
#endif  // TF_ROCM_VERSION >= 60200

  } else {
    return absl::InternalError("No method of loading ROCM kernel provided");
  }

  absl::MutexLock lock{&in_memory_modules_mu_};
  loaded_kernels_.insert(rocm_kernel.get());

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the ROCM API.
  rocm_kernel->set_arity(spec.arity());

  // unable to get kernel metadata for in-process kernel
  if (!spec.has_in_process_symbol()) {
    TF_ASSIGN_OR_RETURN(KernelMetadata kernel_metadata,
                        rocm_kernel->GetKernelMetadata());
    rocm_kernel->set_metadata(kernel_metadata);
  }
  rocm_kernel->set_name(*kernel_name);
  rocm_kernel->set_args_packing(spec.kernel_args_packing());
  return std::move(rocm_kernel);
}

absl::StatusOr<ModuleHandle> RocmExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  // We store the pointer to the HSACO binary as ModuleHandle::id().

  // TODO(ROCm): Need  generic term instead of cubin/cuda/ptx
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    return LoadModuleFromHsaco(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()));
  } else {
    return absl::InternalError("No HASCO binary found");
  }
}

absl::StatusOr<ModuleHandle> RocmExecutor::LoadModuleFromHsaco(
    const char* hsaco) {
  ModuleHandle module_handle{hsaco};
  uint64_t module_refcount;
  hipModule_t module;
  std::tie(module, module_refcount) = gpu_binary_to_module_[module_handle];

  if (module == nullptr) {
    TF_ASSIGN_OR_RETURN(module, LoadHsaco(rocm_context_, hsaco));
    module_refcount = 1;
    in_memory_modules_[module_handle] = module;
    VLOG(3) << "Loaded HSACO " << static_cast<const void*>(hsaco)
            << " as module " << module;
  } else {
    ++module_refcount;
    VLOG(3) << "HSACO " << static_cast<const void*>(hsaco)
            << " is already loaded as module " << module;
  }
  gpu_binary_to_module_[module_handle] = {module, module_refcount};
  return module_handle;
}

DeviceMemoryBase RocmExecutor::Allocate(uint64_t size, int64_t memory_space) {
  switch (static_cast<MemoryType>(memory_space)) {
    case MemoryType::kCollective:
    case MemoryType::kDevice:
      return DeviceMemoryBase(DeviceAllocate(rocm_context_, size), size);
    case MemoryType::kHost:
      if (auto result = HostAllocate(rocm_context_, size); result.ok()) {
        return DeviceMemoryBase(*result, size);
      }
      return DeviceMemoryBase(nullptr, 0);
    default:
      LOG(FATAL) << "Unsupported memory space: " << memory_space;
  }
}
absl::StatusOr<std::unique_ptr<MemoryAllocation>>
RocmExecutor::HostMemoryAllocate(uint64_t size) {
  return AllocateHostMemory(rocm_context_, size);
}

void RocmExecutor::Deallocate(DeviceMemoryBase* mem) {
  DeviceDeallocate(rocm_context_, mem->opaque());
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
RocmExecutor::CreateMemoryAllocator(MemoryType type) {
  switch (type) {
    case MemoryType::kUnified:
      return std::make_unique<GenericMemoryAllocator>(
          [this](uint64_t size)
              -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
            std::unique_ptr<ActivateContext> activation = Activate();
            hipDeviceptr_t result = nullptr;
            // "managed" memory is visible to both CPU and GPU.
            TF_RETURN_IF_ERROR(ToStatus(
                wrap::hipMallocManaged(&result, size, hipMemAttachGlobal),
                "Failed to allocate managed memory"));
            void* ptr = reinterpret_cast<void*>(result);
            VLOG(2) << "allocated " << ptr << " for context " << rocm_context_
                    << " of " << size << " bytes in unified memory";
            return std::make_unique<GenericMemoryAllocation>(
                ptr, size, [this](void* location, uint64_t size) {
                  std::unique_ptr<ActivateContext> activation = Activate();
                  hipDeviceptr_t pointer =
                      absl::bit_cast<hipDeviceptr_t>(location);
                  hipError_t res = wrap::hipFree(pointer);
                  if (res != hipSuccess) {
                    LOG(ERROR) << "failed to free unified memory at "
                               << location << "; result: " << ToString(res);
                  } else {
                    VLOG(2) << "deallocated unified memory at " << location
                            << " for context " << rocm_context_;
                  }
                });
          });
    case MemoryType::kCollective:
      return std::make_unique<GenericMemoryAllocator>(
          [](uint64_t size)
              -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
            void* ptr = nullptr;
            auto hipResult = wrap::hipMalloc(&ptr, size);
            if (hipResult != hipSuccess) {
              return absl::InternalError(absl::StrFormat(
                  "failed to allocate %s (%llu bytes) from device collective "
                  "memory: %s, "
                  "Last NCCL warning(error)",
                  tsl::strings::HumanReadableNumBytes(size), size,
                  hipGetErrorString(hipResult)));
            }
            VLOG(2) << "allocated " << ptr << " of " << size
                    << " bytes of collective memory";
            return std::make_unique<GenericMemoryAllocation>(
                ptr, size, [](void* location, uint64_t size) {
                  auto status = wrap::hipFree(location);
                  if (status != hipSuccess) {
                    LOG(ERROR) << "failed to free collective memory at "
                               << location << "; result: " << status;
                  } else {
                    VLOG(2) << "deallocated collective memory at " << location;
                  }
                });
          });
    case MemoryType::kHost:
      return std::make_unique<GenericMemoryAllocator>([this](uint64_t size) {
        return AllocateHostMemory(rocm_context_, size);
      });
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unsupported memory type %d", type));
  }
}

bool RocmExecutor::SynchronizeAllActivity() {
  return rocm_context_->Synchronize().ok();
}

bool RocmExecutor::HostMemoryRegister(void* location, uint64_t size) {
  VLOG(1) << "Called StreamExecutor::HostMemoryRegister(data=" << location
          << ")";

  std::unique_ptr<ActivateContext> activation = Activate();
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  auto status =
      ToStatus(hipHostRegister(location, size, hipHostRegisterPortable));
  if (!status.ok()) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << status;
    return false;
  }
  return true;
}

bool RocmExecutor::HostMemoryUnregister(void* location) {
  VLOG(1) << "Called StreamExecutor::HostUnregister(data=" << location << ")";

  std::unique_ptr<ActivateContext> activation = Activate();
  auto status = ToStatus(hipHostUnregister(location));
  if (!status.ok()) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << status;
    return false;
  }
  return true;
}

absl::Status RocmExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  hipDeviceptr_t rocm_location = AsROCmDevicePtr(location);
  if (reinterpret_cast<uintptr_t>(location->opaque()) % sizeof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return ToStatus(
        wrap::hipMemsetD32(rocm_location, 0x0, size / sizeof(uint32_t)),
        "Failed to memset memory");
  }
  return ToStatus(wrap::hipMemsetD8(rocm_location, 0x0, size),
                  "Failed to memset memory");
}

absl::Status RocmExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyHtoD(AsROCmDevicePtr(gpu_dst), const_cast<void*>(host_src),
                          size),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: Gpu dst: %p;"
          " host src: %p; size: %llu=0x%llx",
          AsROCmDevicePtr(gpu_dst), host_src, size, size)));
  VLOG(2) << "successfully sync memcpy'd h2d of " << size << " bytes";
  return absl::OkStatus();
}

absl::Status RocmExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyDtoH(host_dst, AsROCmDevicePtr(gpu_src), size),
      absl::StrFormat("failed to synchronous memcpy from device to host: "
                      "host dst: %p; Gpu src: %p; size: %llu=0x%llx",
                      host_dst, AsROCmDevicePtr(gpu_src), size, size)));
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return absl::OkStatus();
}

void RocmExecutor::DeallocateStream(Stream* stream) {
  {
    absl::MutexLock lock(&mu_);
    if (dnn_ != nullptr) {
      dnn_->NotifyStreamDestroyed(stream);
    }
  }
  RocmStream* rocm_stream = static_cast<RocmStream*>(stream);
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(rocm_stream->stream_handle());
}

absl::Status RocmExecutor::InitBlas() {
  absl::MutexLock lock(&mu_);
  PluginRegistry* registry = PluginRegistry::Instance();
  TF_ASSIGN_OR_RETURN(
      auto factory,
      registry->GetFactory<PluginRegistry::BlasFactory>(rocm::kROCmPlatformId));
  blas_.reset(factory(this));
  return absl::OkStatus();
}

blas::BlasSupport* RocmExecutor::AsBlas() {
  absl::MutexLock lock(&mu_);
  return blas_.get();
}

dnn::DnnSupport* RocmExecutor::AsDnn() {
  absl::MutexLock lock(&mu_);
  if (dnn_ != nullptr) {
    return dnn_.get();
  }
  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(rocm::kROCmPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().message();
    return nullptr;
  }

  auto dnn = status.value()(this);

  dnn_.reset(dnn);

  return dnn_.get();
}

fft::FftSupport* RocmExecutor::AsFft() {
  absl::MutexLock lock(&mu_);
  if (fft_ != nullptr) {
    return fft_.get();
  }
  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(rocm::kROCmPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().message();
    return nullptr;
  }

  auto fft = status.value()(this);

  fft_.reset(fft);
  return fft_.get();
}

bool RocmExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  RocmExecutor* rocm_other = static_cast<RocmExecutor*>(other);
  return CanEnablePeerAccess(rocm_context_, rocm_other->rocm_context_);
}

absl::Status RocmExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  RocmExecutor* rocm_other = static_cast<RocmExecutor*>(other);
  return EnablePeerAccess(rocm_context_, rocm_other->rocm_context_);
}

bool RocmExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  return rocm_context_->GetDeviceMemoryUsage(free, total);
}

absl::StatusOr<DeviceMemoryBase> RocmExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  void* mem = nullptr;
  size_t bytes = 0;

  absl::MutexLock lock{&in_memory_modules_mu_};
  if (static_cast<bool>(module_handle)) {
    auto it = gpu_binary_to_module_.find(module_handle);
    CHECK(it != gpu_binary_to_module_.end());
    TF_RETURN_IF_ERROR(
        GetModuleSymbol(rocm_context_, it->second.first, symbol_name.c_str(),
                        reinterpret_cast<hipDeviceptr_t*>(&mem), &bytes));
    return DeviceMemoryBase(mem, bytes);
  }

  for (auto& it : gpu_binary_to_module_) {
    TF_RETURN_IF_ERROR(
        GetModuleSymbol(rocm_context_, it.second.first, symbol_name.c_str(),
                        reinterpret_cast<hipDeviceptr_t*>(&mem), &bytes));
    return DeviceMemoryBase(mem, bytes);
  }

  LOG(INFO) << "Falied to find symbol in any modules: " << symbol_name;
  return absl::NotFoundError(
      absl::StrCat("Check if module containing symbol ", symbol_name,
                   " is loaded (module_handle = ",
                   reinterpret_cast<uintptr_t>(module_handle.id()), ")"));
}

absl::Status FillBlockDimLimit(hipDevice_t device, BlockDim* block_dim_limit) {
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

absl::StatusOr<std::unique_ptr<Event>> RocmExecutor::CreateEvent() {
  TF_ASSIGN_OR_RETURN(auto event,
                      RocmEvent::Create(this, /*allow_timing=*/false));
  return std::make_unique<RocmEvent>(std::move(event));
}

absl::StatusOr<std::unique_ptr<Stream>> RocmExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(auto stream, RocmStream::Create(this, priority));
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_[stream->stream_handle()] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>>
RocmExecutor::CreateCommandBuffer(CommandBuffer::Mode mode) {
  VLOG(2) << "Create ROCm command buffer (ROCm graph)";
  return RocmCommandBuffer::Create(mode, this);
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
RocmExecutor::CreateDeviceDescription(int device_ordinal) {
  TF_ASSIGN_OR_RETURN(hipDevice_t device, GetDevice(device_ordinal));

  TF_ASSIGN_OR_RETURN(std::string gcn_arch_name, GetGpuGCNArchName(device));

  DeviceDescription desc;

  {
    std::string pci_bus_id = GetPCIBusID(device);

    // Lower the hex characters to match sysfs.
    pci_bus_id = absl::AsciiStrToLower(pci_bus_id);
    desc.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    std::optional<int> numa_node = ReadNumaNode(pci_bus_id, device_ordinal);
    // If the kernel reports -1, adjust to 0; leave as -1 if no value could be
    // obtained.
    desc.set_numa_node(numa_node.has_value() ? std::max(0, *numa_node)
                                             : tsl::port::kNUMANoAffinity);
  }

  hipDeviceProp_t prop;
  if (GetDeviceProperties(&prop, device_ordinal)) {
    desc.set_threads_per_block_limit(prop.maxThreadsPerBlock);

    ThreadDim thread_dim_limit;
    thread_dim_limit.x = prop.maxThreadsDim[0];
    thread_dim_limit.y = prop.maxThreadsDim[1];
    thread_dim_limit.z = prop.maxThreadsDim[2];
    desc.set_thread_dim_limit(thread_dim_limit);

    float clock_rate_ghz = static_cast<float>(prop.clockRate) / 1e6;
    desc.set_clock_rate_ghz(clock_rate_ghz);

    // mem_bandwidth = 2 * mem_bus_width_in_bytes * mem_clock_rate_in_hz
    int64_t memory_bandwidth =
        2 * (static_cast<int64_t>(prop.memoryBusWidth) / 8) *
        (static_cast<int64_t>(prop.memoryClockRate) * 1000);
    desc.set_memory_bandwidth(memory_bandwidth);

    desc.set_l2_cache_size(prop.l2CacheSize);
  }

  // No way to query ECC status from the API.
  desc.set_ecc_enabled(false);

  uint64_t device_memory_size = -1;
  (void)RocmContext::GetDeviceTotalMemory(device, &device_memory_size);
  desc.set_device_memory_size(device_memory_size);

  {
    BlockDim block_dim_limit;
    TF_RETURN_IF_ERROR(FillBlockDimLimit(device, &block_dim_limit));
    desc.set_block_dim_limit(block_dim_limit);
  }

  {
    TF_ASSIGN_OR_RETURN(std::string device_name, GetDeviceName(device));
    desc.set_name(device_name.empty() ? gcn_arch_name : device_name);
  }

  desc.set_platform_version(
      absl::StrCat("AMDGPU ISA version: ", gcn_arch_name));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  desc.set_device_address_bits(64);

  desc.set_device_vendor("Advanced Micro Devices, Inc");
  desc.set_rocm_compute_capability(gcn_arch_name);

  desc.set_shared_memory_per_core(GetMaxSharedMemoryPerCore(device).value());
  desc.set_shared_memory_per_block(GetMaxSharedMemoryPerBlock(device).value());
  desc.set_shared_memory_per_block_optin(
      GetMaxSharedMemoryPerBlock(device).value());
  int core_count = GetMultiprocessorCount(device).value();
  desc.set_core_count(core_count);
  desc.set_fpus_per_core(fpus_per_core(gcn_arch_name));
  desc.set_threads_per_core_limit(
      GetMaxThreadsPerMultiprocessor(device).value());
  desc.set_registers_per_block_limit(GetMaxRegistersPerBlock(device).value());
  desc.set_threads_per_warp(GetThreadsPerWarp(device).value());
  desc.set_registers_per_core_limit(64 * 1024);
  desc.set_compile_time_toolkit_version(
      SemanticVersion{HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH});
  int32_t runtime_version;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipRuntimeGetVersion(&runtime_version),
                              "Failed call to hipRuntimeGetVersion"));
  desc.set_runtime_version(
      ParseRocmVersion(runtime_version).value_or(SemanticVersion{0, 0, 0}));
  int32_t driver_version;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipDriverGetVersion(&driver_version),
                              "Could not get driver version"));
  desc.set_driver_version(
      ParseRocmVersion(driver_version).value_or(SemanticVersion{0, 0, 0}));

  // It would be better to use the PCI device ID or some other truly unique
  // identifier for the GPU model.  But getting this requires using NVML or
  // other hacks, which we don't have access to in OSS TensorFlow.
  //
  // Alternatively you might be tempted to use GetDeviceName as a
  // unique identifier, but this is not stable across GPU VBIOS versions.
  //
  // TODO(jlebar): This really should be more unique.  In CUDA land, we mix in
  // the clock speed and L2 cache size.
  desc.set_model_str(
      absl::StrFormat("%dB RAM, %d cores", device_memory_size, core_count));

  return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<MemoryType> RocmExecutor::GetPointerMemorySpace(
    const void* ptr) {
  hipDeviceptr_t pointer =
      reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(ptr));
  unsigned int value;
  hipError_t result = wrap::hipPointerGetAttribute(
      &value, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer);
  if (result == hipSuccess) {
    switch (value) {
      case hipMemoryTypeDevice:
        return MemoryType::kDevice;
      case hipMemoryTypeHost:
        return MemoryType::kHost;
      default:
        return absl::InternalError(
            absl::StrCat("unknown memory space provided by ROCM API: ", value));
    }
  }

  return absl::InternalError(absl::StrCat(
      "failed to query device pointer for memory space: ", ToString(result)));
}

absl::StatusOr<const RocmKernel*> RocmExecutor::GetRocmKernel(
    const Kernel* kernel) {
  absl::MutexLock lock{&in_memory_modules_mu_};
  auto it = loaded_kernels_.find(kernel);
  if (it == loaded_kernels_.end()) {
    return absl::NotFoundError("Kernel not loaded in this executor.");
  }
  return static_cast<const RocmKernel*>(*it);
}
}  // namespace gpu

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(rocm_executor, {});
