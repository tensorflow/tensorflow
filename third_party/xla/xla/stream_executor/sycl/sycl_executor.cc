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

#include "xla/stream_executor/sycl/sycl_executor.h"

#include <unistd.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::sycl {

namespace dnn = stream_executor::dnn;
namespace sycl = ::sycl;
namespace DeviceInfo = sycl::info::device;

#define RETURN_IF_ZE_ERROR(expr, msg)                            \
  do {                                                           \
    ze_result_t result = (expr);                                 \
    if (result != ZE_RESULT_SUCCESS) {                           \
      return absl::InternalError(                                \
          absl::StrCat(msg, ", got Level Zero error ", result)); \
    }                                                            \
  } while (0)

namespace {

void* AsSyclDevicePtr(const DeviceMemoryBase& gpu_mem) {
  return const_cast<void*>(gpu_mem.opaque());
}

void* AsSyclDevicePtr(DeviceMemoryBase* gpu_mem) {
  return AsSyclDevicePtr(*gpu_mem);
}

// Returns the device name for the given SYCL device.
absl::StatusOr<std::string> GetDeviceName(const sycl::device& device) {
  try {
    return device.get_info<DeviceInfo::name>();
  } catch (const sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("GetDeviceName: SYCL exception: ", e.what()));
  }
}

// Destroy (unload) a Level Zero module from the given SYCL context,
// ensuring proper GPU resource cleanup.
void UnloadLevelZeroModule(SyclContext* context, ze_module_handle_t module) {
  if (module != nullptr) {
    ze_result_t destroy_status = zeModuleDestroy(module);
    if (destroy_status != ZE_RESULT_SUCCESS) {
      LOG(FATAL)
          << "UnloadLevelZeroModule: Failed to destroy module, got Level "
             "Zero error: "
          << destroy_status;
    }
    VLOG(2) << "UnloadLevelZeroModule: Successfully destroyed module " << module
            << " for device ordinal " << context->device_ordinal();
  }
}

// Loads a SPIR-V binary into a Level Zero module for the given SYCL context
// and device.
absl::StatusOr<ze_module_handle_t> LoadLevelZeroModule(
    SyclContext* context, const char* spirv_binary, const size_t spirv_size) {
  const sycl::context& sycl_context = context->context();
  TF_ASSIGN_OR_RETURN(sycl::device sycl_device,
                      SyclDevicePool::GetDevice(context->device_ordinal()));
  auto lz_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
  auto lz_context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);

  // Create module description for SPIR-V binary
  ze_module_desc_t module_desc = {
      ZE_STRUCTURE_TYPE_MODULE_DESC,
      /*pNext=*/nullptr,
      ZE_MODULE_FORMAT_IL_SPIRV,
      spirv_size,
      reinterpret_cast<const uint8_t*>(spirv_binary),
      /*pBuildFlags=*/nullptr,
      /*pConstants=*/nullptr};

  ze_module_build_log_handle_t log_handle;
  ze_module_handle_t module;
  ze_result_t result =
      zeModuleCreate(lz_context, lz_device, &module_desc, &module, &log_handle);
  if (result != ZE_RESULT_SUCCESS) {
    // If module creation fails, retrieve the build log and return it
    // as part of the error status.
    size_t log_size = 0;
    RETURN_IF_ZE_ERROR(zeModuleBuildLogGetString(log_handle, &log_size,
                                                 /*pBuildLog=*/nullptr),
                       "LoadLevelZeroModule: Failed to query build log size");

    // Allocate buffer to hold the build log.
    std::unique_ptr<char[]> log_buffer(new char[log_size]);
    RETURN_IF_ZE_ERROR(
        zeModuleBuildLogGetString(log_handle, &log_size, log_buffer.get()),
        "LoadLevelZeroModule: Failed to retrieve build log string");

    // Destroy the build log handle to free resources.
    RETURN_IF_ZE_ERROR(
        zeModuleBuildLogDestroy(log_handle),
        "LoadLevelZeroModule: Failed to destroy build log handle");

    return absl::InternalError(absl::StrCat(
        "LoadLevelZeroModule: Failed to create module, got Level Zero error ",
        result, ": ", log_buffer.get()));
  }
  // Module created successfully.
  return module;
}

// Retrieve a SYCL kernel by name from a loaded Level Zero module in the
// given SYCL context.
// If VLOG(2) is enabled, logs all kernel names available in the module for
// debugging.
absl::StatusOr<std::unique_ptr<sycl::kernel>> GetModuleFunction(
    SyclContext* context, ze_module_handle_t module_handle,
    const char* kernel_name) {
  const sycl::context& sycl_context = context->context();
  if (module_handle == nullptr) {
    return absl::InternalError("GetModuleFunction: module_handle is null");
  }

  if (kernel_name == nullptr) {
    return absl::InternalError("GetModuleFunction: kernel_name is null");
  }

  // If VLOG is enabled, log all kernel names in the Level Zero module for
  // debugging.
  if (VLOG_IS_ON(2)) {
    // Get the number of kernels available in the module.
    uint32_t kernel_count = 0;
    RETURN_IF_ZE_ERROR(
        zeModuleGetKernelNames(module_handle, &kernel_count, nullptr),
        "GetModuleFunction: Failed to get the number of kernels");

    if (kernel_count == 0) {
      VLOG(2) << "Level Zero module has no kernels.";
    } else {
      // Allocate space to hold the kernel name pointers.
      std::unique_ptr<const char*[]> kernel_name_ptrs(
          new const char*[kernel_count]);

      // Retrieve the kernel names.
      RETURN_IF_ZE_ERROR(
          zeModuleGetKernelNames(module_handle, &kernel_count,
                                 kernel_name_ptrs.get()),
          "GetModuleFunction: Failed to retrieve the kernel names");

      // Build a list of kernel names for logging.
      std::vector<absl::string_view> kernel_names_vec(kernel_count);
      for (uint32_t i = 0; i < kernel_count; ++i) {
        kernel_names_vec[i] = absl::string_view(kernel_name_ptrs[i]);
      }
      VLOG(2) << "Required kernel name: " << kernel_name;
      VLOG(2) << "Level Zero module has kernels: "
              << absl::StrJoin(kernel_names_vec, "; ");
    }
  }

  ze_kernel_desc_t kernel_desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC,
                                  /*pNext=*/nullptr, /*flags=*/0, kernel_name};
  ze_kernel_handle_t lz_kernel;
  ze_result_t kernel_create_status =
      zeKernelCreate(module_handle, &kernel_desc, &lz_kernel);
  if (kernel_create_status != ZE_RESULT_SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "GetModuleFunction: Failed to create kernel '", kernel_name,
        "' from module. Level Zero error: ", kernel_create_status));
  }

  // Ownership: The newly created kernel (lz_kernel) is managed by the module,
  // and will be destroyed when the module is unloaded via
  // UnloadLevelZeroModule.

  sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>({module_handle},
                                                               sycl_context);
  sycl::kernel kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {kernel_bundle, lz_kernel}, sycl_context);
  return std::make_unique<sycl::kernel>(kernel);
}

// Computes a 128-bit fingerprint for the given string.
absl::uint128 Fingerprint128(const absl::string_view s) {
  auto fp = tsl::Fingerprint128(s);
  return absl::MakeUint128(fp.high64, fp.low64);
}

// Retrieves the device address and size of a global symbol from a Level Zero
// module in the given SYCL context.
// On success, sets device_ptr to the symbol address and symbol_size to its
// size.
absl::Status GetModuleSymbol(SyclContext* context, ze_module_handle_t module,
                             const char* symbol_name, void** device_ptr,
                             size_t* symbol_size) {
  if (module == nullptr || symbol_name == nullptr ||
      (*device_ptr == nullptr && symbol_size == nullptr)) {
    return absl::InvalidArgumentError(
        "GetModuleSymbol: Null input argument(s) provided.");
  }
  ze_result_t status =
      zeModuleGetGlobalPointer(module, symbol_name, symbol_size, device_ptr);
  if (status != ZE_RESULT_SUCCESS) {
    // The symbol may not be present in this module.
    return absl::InternalError(
        absl::StrCat("Failed to get symbol '", symbol_name,
                     "\" from module. Level Zero error: ", status));
  }
  return absl::OkStatus();
}

absl::Status SynchronousMemsetUint32(SyclContext* context, void* location,
                                     uint32_t value, size_t uint32_count) {
  TF_RETURN_IF_ERROR(SyclMemfillDevice(context->device_ordinal(), location,
                                       value, uint32_count));
  VLOG(2) << absl::StrFormat(
      "Completed synchronous memset32: %u uint32s at %p with value 0x%08x on "
      "device %d",
      uint32_count, location, value, context->device_ordinal());
  return absl::OkStatus();
}

absl::Status SynchronousMemsetUint8(SyclContext* context, void* location,
                                    uint8_t value, size_t size) {
  TF_RETURN_IF_ERROR(
      SyclMemsetDevice(context->device_ordinal(), location, value, size));
  VLOG(2) << absl::StrFormat(
      "Completed synchronous memset8: %u bytes at %p with value 0x%02x on "
      "device %d",
      size, location, value, context->device_ordinal());
  return absl::OkStatus();
}

absl::Status SyclSynchronousMemcpyH2D(SyclContext* context, void* gpu_dst,
                                      const void* host_src, uint64_t size) {
  TF_RETURN_IF_ERROR(SyclMemcpyHostToDevice(context->device_ordinal(), gpu_dst,
                                            host_src, size));
  VLOG(2) << absl::StrFormat(
      "Completed synchronous host-to-device memcpy: %u bytes to %p", size,
      gpu_dst);
  return absl::OkStatus();
}

absl::Status SyclSynchronousMemcpyD2H(SyclContext* context, void* host_dst,
                                      void* gpu_src, uint64_t size) {
  TF_RETURN_IF_ERROR(SyclMemcpyDeviceToHost(context->device_ordinal(), host_dst,
                                            gpu_src, size));
  VLOG(2) << absl::StrFormat(
      "Completed synchronous device-to-host memcpy: %u bytes to %p", size,
      host_dst);
  return absl::OkStatus();
}

// Allocates memory on the GPU device.
void* DeviceAllocate(SyclContext* context, uint64_t bytes) {
  if (bytes == 0) {
    VLOG(2)
        << "DeviceAllocate: Trying to allocate 0 bytes, skipping allocation.";
    return nullptr;
  }
  auto malloc_status = SyclMallocDevice(context->device_ordinal(), bytes);
  if (!malloc_status.ok()) {
    LOG(ERROR) << "DeviceAllocate: Failed to allocate device memory of size "
               << bytes << ", got " << malloc_status.status();
    return nullptr;
  }
  void* ptr = malloc_status.value();
  VLOG(2) << absl::StrFormat(
      "Allocated %zu bytes of device memory at %p for device ordinal %d", bytes,
      ptr, context->device_ordinal());
  return ptr;
}

// Deallocates memory allocated on the host, device or shared memory.
void DeviceDeallocate(SyclContext* context, void* location) {
  auto free_status = SyclFree(context->device_ordinal(), location);
  if (!free_status.ok()) {
    LOG(ERROR) << absl::StrFormat(
        "DeviceDeallocate: Failed to free device memory at %p, got %s",
        location, free_status.ToString());
    return;
  }
  VLOG(2) << absl::StrFormat(
      "Successfully deallocated device memory at %p for device ordinal %d",
      location, context->device_ordinal());
}

absl::StatusOr<void*> HostAllocate(SyclContext* context, int device_ordinal,
                                   uint64_t bytes) {
  TF_ASSIGN_OR_RETURN(void* host_mem, SyclMallocHost(device_ordinal, bytes));
  if (host_mem == nullptr) {
    return absl::InternalError(
        absl::StrFormat("HostAllocate: failed to allocate %u bytes of host "
                        "memory for device ordinal %d.",
                        bytes, device_ordinal));
  }
  return host_mem;
}

// Allocate host memory accessible by the host and mappable for device access.
absl::StatusOr<std::unique_ptr<MemoryAllocation>> AllocateHostMemory(
    SyclContext* sycl_context, int device_ordinal, uint64_t size) {
  TF_ASSIGN_OR_RETURN(void* host_mem,
                      HostAllocate(sycl_context, device_ordinal, size));
  VLOG(2) << "Allocated host memory for ptr " << host_mem
          << " using device ordinal " << device_ordinal << " for " << size
          << " bytes";
  return std::make_unique<GenericMemoryAllocation>(
      host_mem, size,
      [sycl_context, device_ordinal](void* location, uint64_t size) {
        absl::Status free_status = SyclFree(device_ordinal, location);
        if (free_status != absl::OkStatus()) {
          LOG(ERROR) << absl::StrFormat(
              "AllocateHostMemory: failed to free host memory at %p, got %s",
              location, free_status.ToString());
        } else {
          VLOG(2) << "Successfully deallocated host memory for ptr " << location
                  << " using device ordinal " << device_ordinal;
        }
      });
}

}  // namespace

class OneDnnSupport : public dnn::DnnSupport {
 public:
  absl::Status Init() override { return absl::OkStatus(); }

  absl::StatusOr<dnn::VersionInfo> GetVersion() override {
    // Return a default version since it will be implemented later.
    return dnn::VersionInfo(0, 0, 0);
  }

  absl::Status DoPoolForward(dnn::DataType element_type, Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             DeviceMemoryBase input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceMemoryBase output_data,
                             ScratchAllocator* workspace_allocator) override {
    return absl::UnimplementedError(
        "OneDnnSupport::DoPoolForward is not implemented for SYCL");
  }

  absl::Status DoPoolBackward(dnn::DataType element_type, Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              DeviceMemoryBase input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              DeviceMemoryBase output_data,
                              DeviceMemoryBase input_diff_data,
                              DeviceMemoryBase output_diff_data,
                              ScratchAllocator* workspace_allocator) override {
    return absl::UnimplementedError(
        "OneDnnSupport::DoPoolBackward is not implemented for SYCL");
  }
};

SyclExecutor::~SyclExecutor() {
  for (auto& it : in_memory_modules_) {
    UnloadLevelZeroModule(sycl_context_.get(), it.second);
  }
  CHECK(kernel_to_gpu_binary_.empty()) << "SyclExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "SyclExecutor has loaded modules.";
  sycl_context_.reset();
}

absl::Status SyclExecutor::Init() {
  TF_ASSIGN_OR_RETURN(device_, SyclDevicePool::GetDevice(device_ordinal()));
  TF_ASSIGN_OR_RETURN(sycl_context_, SyclContext::Create(device_ordinal()));

  // Return OK status since StreamExecutor is usually initialized via
  // TF_ASSERT_OK_AND_ASSIGN in unit tests.
  return absl::OkStatus();
}

dnn::DnnSupport* SyclExecutor::AsDnn() {
  static std::unique_ptr<dnn::DnnSupport> dnn =
      std::make_unique<OneDnnSupport>();
  return dnn.get();
}

absl::StatusOr<std::unique_ptr<Kernel>> SyclExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  // Check that a SPIR-V binary is provided in the spec.
  if (!spec.has_cuda_cubin_in_memory()) {
    return absl::InternalError(
        "SyclExecutor::LoadKernel: No SPIR-V binary provided in spec.");
  }

  // Create a new SyclKernel instance for the loaded kernel.
  auto sycl_kernel = std::make_unique<SyclKernel>(this);
  const std::string& kernel_name = spec.kernel_name();
  const char* spirv_binary = reinterpret_cast<const char*>(
      spec.cuda_cubin_in_memory()->cubin_bytes.data());
  size_t spirv_size = spec.cuda_cubin_in_memory()->cubin_bytes.size();

  ModuleHandle module_handle{spirv_binary};
  ze_module_handle_t module = nullptr;

  // Check if the module is already loaded.
  {
    absl::MutexLock lock{&in_memory_modules_mu_};
    auto in_mem_it = in_memory_modules_.find(module_handle);
    if (in_mem_it != in_memory_modules_.end()) {
      module = in_mem_it->second;
    }
  }

  // If module is not loaded, load it outside the lock for efficiency.
  // Only the first thread to load the module inserts it into the cache.
  // Other threads reuse the cached module and unload their own redundant
  // module.
  if (module == nullptr) {
    TF_ASSIGN_OR_RETURN(module, LoadLevelZeroModule(sycl_context_.get(),
                                                    spirv_binary, spirv_size));
    {
      absl::MutexLock lock{&in_memory_modules_mu_};
      // Try to insert the newly loaded module into the cache.
      auto [in_mem_it, inserted] =
          in_memory_modules_.emplace(module_handle, module);
      if (!inserted) {
        // Another thread loaded the module first.
        // Unload the redundant module inside the lock since unloading is fast
        // and also to avoid resource leaks.
        UnloadLevelZeroModule(sycl_context_.get(), module);
        module = in_mem_it->second;

        // Increment reference count in gpu_binary_to_module_.
        auto gpu_bin_it = gpu_binary_to_module_.find(module_handle);
        if (gpu_bin_it != gpu_binary_to_module_.end()) {
          ++(gpu_bin_it->second.second);
        } else {
          // This should not happen since in_memory_modules_ and
          // gpu_binary_to_module_ should be consistent.
          return absl::InternalError(
              "SyclExecutor::LoadKernel: Inconsistent module cache state.");
        }
      } else {
        // Newly inserted module: Set reference count to 1 in
        // gpu_binary_to_module_.
        gpu_binary_to_module_[module_handle] = std::make_pair(module, 1);
      }
    }
  }

  // Retrieve the kernel function from the loaded module.
  VLOG(2) << "Getting function " << kernel_name << " from module " << module;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<sycl::kernel> function,
      GetModuleFunction(sycl_context_.get(), module, kernel_name.c_str()));
  {
    absl::MutexLock lock{&in_memory_modules_mu_};
    // Track which kernels are loaded and their associated modules.
    kernel_to_gpu_binary_[sycl_kernel.get()] = module_handle;
    loaded_kernels_.insert(sycl_kernel.get());
  }

  // Set kernel function and metadata.
  sycl_kernel->set_gpu_function(function.release());
  // We have to trust the kernel loader spec arity because there doesn't
  // appear to be a way to reflect on the number of expected arguments w/the
  // SPIR API.
  sycl_kernel->set_arity(spec.arity());
  // TODO (intel-tf): Once SyclKernel::GetKernelMetadata() is implemented,
  // we should use it here via set_metadata().
  sycl_kernel->set_name(kernel_name);
  // Set argument packing function if provided.
  if (std::holds_alternative<KernelLoaderSpec::KernelArgsPackingFunc>(
          spec.kernel_args_packing())) {
    sycl_kernel->set_args_packing(
        std::get<KernelLoaderSpec::KernelArgsPackingFunc>(
            spec.kernel_args_packing()));
  }
  return std::move(sycl_kernel);
}

bool SyclExecutor::UnloadModule(ModuleHandle module_handle) {
  VLOG(3) << "SyclExecutor::UnloadModule: Unloading module " << module_handle;
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(module_handle);
}

void SyclExecutor::UnloadKernel(const Kernel* kernel) {
  VLOG(3) << "SyclExecutor::UnloadKernel: Unloading kernel " << kernel << " : "
          << kernel->name();
  absl::MutexLock lock{&in_memory_modules_mu_};
  {
    loaded_kernels_.erase(kernel);
    auto it = kernel_to_gpu_binary_.find(kernel);
    if (it == kernel_to_gpu_binary_.end()) {
      VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
              << " has never been loaded.";
      return;  // We've never seen this kernel.
    }
    VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
            << " has loaded GPU code " << it->second;
    UnloadGpuBinary(it->second);
    kernel_to_gpu_binary_.erase(it);
  }
}

absl::StatusOr<ModuleHandle> SyclExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  if (spec.has_cuda_cubin_in_memory()) {
    return LoadModuleFromSpirv(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()),
        spec.cuda_cubin_in_memory().size());
  }
  return absl::InternalError(
      "SyclExecutor::LoadModule: No SPIR-V binary found, cannot load module.");
}

absl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
SyclExecutor::CreateOrShareConstant(Stream* stream,
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
      return absl::InternalError(
          absl::StrFormat("SyclExecutor::CreateOrShareConstant: Failed to "
                          "allocate %d bytes for new constant",
                          content.size()));
    }

    TF_RETURN_IF_ERROR(
        stream->Memcpy(new_constant, content.data(), content.size()));
    absl::Status status = stream->BlockHostUntilDone();
    if (!status.ok()) {
      Deallocate(new_constant);
      status.Update(absl::InternalError(
          absl::StrFormat("SyclExecutor::CreateOrShareConstant: Memcpy to "
                          "device address %p failed",
                          new_constant->opaque())));
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

DeviceMemoryBase SyclExecutor::Allocate(uint64_t size, int64_t memory_space) {
  switch (static_cast<MemoryType>(memory_space)) {
    case MemoryType::kCollective:
    case MemoryType::kDevice: {
      return DeviceMemoryBase(DeviceAllocate(sycl_context_.get(), size), size);
    }
    case MemoryType::kHost: {
      auto result = HostAllocate(sycl_context_.get(), device_ordinal(), size);
      return (result.ok() ? DeviceMemoryBase(*result, size)
                          : DeviceMemoryBase(nullptr, 0));
    }
    default: {
      LOG(FATAL) << "SyclExecutor::Allocate: unsupported memory space: "
                 << memory_space;
    }
  }
}

void SyclExecutor::Deallocate(DeviceMemoryBase* mem) {
  if (mem == nullptr || mem->opaque() == nullptr) {
    VLOG(2) << "SyclExecutor::Deallocate: Attempting to deallocate a null "
               "device pointer, skipping deallocation.";
    return;
  }
  DeviceDeallocate(sycl_context_.get(), mem->opaque());
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
SyclExecutor::CreateMemoryAllocator(MemoryType type) {
  switch (type) {
    case MemoryType::kUnified:
      return std::make_unique<GenericMemoryAllocator>(
          [this](uint64_t size)
              -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
            // Shared memory is visible to both CPU and GPU.
            TF_ASSIGN_OR_RETURN(void* ptr,
                                SyclMallocShared(device_ordinal(), size));
            VLOG(2) << "Allocated shared memory for ptr " << ptr
                    << " using device_ordinal " << device_ordinal() << " for "
                    << size << " bytes";
            return std::make_unique<GenericMemoryAllocation>(
                ptr, size, [this](void* location, uint64_t size) {
                  absl::Status res = SyclFree(device_ordinal(), location);
                  if (res != absl::OkStatus()) {
                    LOG(ERROR) << "Error deallocating shared memory at "
                               << location << ": " << res;
                  } else {
                    VLOG(2) << "Deallocated shared memory for ptr " << location
                            << " using device_ordinal " << device_ordinal();
                  }
                });
          });
    case MemoryType::kCollective:
      return std::make_unique<GenericMemoryAllocator>(
          [this](uint64_t size)
              -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
            // At the allocation level, collective memory is the same as device
            // memory.
            TF_ASSIGN_OR_RETURN(void* ptr,
                                SyclMallocDevice(device_ordinal(), size));
            VLOG(2) << "Allocated collective/device memory for ptr " << ptr
                    << " using device_ordinal " << device_ordinal() << " for "
                    << size << " bytes";
            return std::make_unique<GenericMemoryAllocation>(
                ptr, size, [this](void* location, uint64_t size) {
                  absl::Status res = SyclFree(device_ordinal(), location);
                  if (res != absl::OkStatus()) {
                    LOG(ERROR) << "Error deallocating collective/device memory "
                                  "at "
                               << location << ": " << res;
                  } else {
                    VLOG(2) << "Deallocated collective/device memory for ptr "
                            << location << " using device_ordinal "
                            << device_ordinal();
                  }
                });
          });
    case MemoryType::kHost:
      return std::make_unique<GenericMemoryAllocator>([this](uint64_t size) {
        return AllocateHostMemory(sycl_context_.get(), device_ordinal(), size);
      });
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "SyclExecutor::CreateMemoryAllocator: unsupported memory type %d",
          type));
  }
}

bool SyclExecutor::SynchronizeAllActivity() {
  return sycl_context_->Synchronize().ok();
}

absl::Status SyclExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % sizeof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return SynchronousMemsetUint32(sycl_context_.get(),
                                   AsSyclDevicePtr(location), 0x0,
                                   size / sizeof(uint32_t));
  }
  return SynchronousMemsetUint8(sycl_context_.get(), AsSyclDevicePtr(location),
                                0x0, size);
}

absl::Status SyclExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  return SyclSynchronousMemcpyH2D(sycl_context_.get(), AsSyclDevicePtr(gpu_dst),
                                  host_src, size);
}

absl::Status SyclExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
  return SyclSynchronousMemcpyD2H(sycl_context_.get(), host_dst,
                                  AsSyclDevicePtr(gpu_src), size);
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
SyclExecutor::CreateDeviceDescription(int device_ordinal) {
  // TODO(intel-tf): Properly populate SYCL device description.
  // Returns a default-constructed DeviceDescription to allow StreamExecutor
  // initialization for tests and code paths that do not require device info.
  DeviceDescription desc;
  return std::make_unique<DeviceDescription>(desc);
}

absl::StatusOr<std::unique_ptr<Stream>> SyclExecutor::CreateStream(
    bool enable_multiple_streams,
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<SyclStream> stream,
      SyclStream::Create(this, enable_multiple_streams, priority));
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_[stream->stream_handle()] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<Event>> SyclExecutor::CreateEvent() {
  TF_ASSIGN_OR_RETURN(auto event, SyclEvent::Create(this));
  return std::make_unique<SyclEvent>(std::move(event));
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
SyclExecutor::HostMemoryAllocate(uint64_t size) {
  return AllocateHostMemory(sycl_context_.get(), device_ordinal(), size);
}

void SyclExecutor::DeallocateStream(Stream* stream) {
  SyclStream* sycl_stream = static_cast<SyclStream*>(stream);
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(sycl_stream->stream_handle());
}

absl::Status SyclExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  return absl::UnimplementedError(
      "SyclExecutor::EnablePeerAccessTo is not implemented.");
}

bool SyclExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  // TODO (intel-tf): Implement this feature for SYCL.
  LOG(INFO) << "SyclExecutor::CanEnablePeerAccessTo is not implemented.";
  return false;
}

bool SyclExecutor::DeviceMemoryUsage(int64_t* free_bytes,
                                     int64_t* total_bytes) const {
  if (free_bytes == nullptr || total_bytes == nullptr) {
    LOG(ERROR) << "SyclExecutor::DeviceMemoryUsage: Output pointer is null.";
    return false;
  }

  try {
    // Query total global memory available on the device.
    uint64_t total_memory_bytes =
        device_.get_info<DeviceInfo::global_mem_size>();

    // SYCL does not provide a standard way to query free device memory.
    // We set free memory to -1 to indicate unknown.
    uint64_t free_memory_bytes = static_cast<uint64_t>(-1);

    *total_bytes = static_cast<int64_t>(total_memory_bytes);
    *free_bytes = static_cast<int64_t>(free_memory_bytes);
    return true;
  } catch (const sycl::exception& e) {
    LOG(ERROR) << "SYCL exception in SyclExecutor::DeviceMemoryUsage: "
               << e.what();
    return false;
  }
}

absl::StatusOr<const SyclKernel*> SyclExecutor::GetSyclKernel(
    const Kernel* kernel) {
  absl::MutexLock lock{&in_memory_modules_mu_};
  auto it = loaded_kernels_.find(kernel);
  if (it == loaded_kernels_.end()) {
    return absl::NotFoundError(
        "SyclExecutor::GetSyclKernel: Kernel not loaded in this executor.");
  }
  return static_cast<const SyclKernel*>(*it);
}

absl::StatusOr<ModuleHandle> SyclExecutor::LoadModuleFromSpirv(
    const char* spirv_binary, size_t spirv_size) {
  // TODO(intel-tf):
  // 1. Use absl::Span<const uint8_t> for SPIR-V binary input.
  // 2. Compute a fingerprint of the SPIR-V binary to use as the module handle
  //    instead of the raw pointer.
  ModuleHandle module_handle{spirv_binary};
  {
    absl::MutexLock lock(&in_memory_modules_mu_);
    auto it = gpu_binary_to_module_.find(module_handle);
    if (it != gpu_binary_to_module_.end()) {
      // Module already loaded: increment reference count.
      ++(it->second.second);
      VLOG(2) << "LoadModuleFromSpirv: SPIR-V "
              << static_cast<const void*>(spirv_binary)
              << " is already loaded as module " << it->second.first;
      return module_handle;
    }
  }

  // Load module outside the lock since it is a slow operation.
  // TODO(intel-tf): Remove redundant loads via absl::call_once.
  // NOTE: Only the first thread to load the module inserts it into the cache.
  // Concurrent threads reuse the cached module and discard their own redundant
  // module via UnloadLevelZeroModule to avoid resource leaks.
  TF_ASSIGN_OR_RETURN(
      ze_module_handle_t lz_module_handle,
      LoadLevelZeroModule(sycl_context_.get(), spirv_binary, spirv_size));

  {
    absl::MutexLock lock(&in_memory_modules_mu_);
    auto it = gpu_binary_to_module_.find(module_handle);
    if (it != gpu_binary_to_module_.end()) {
      // Another thread loaded the module first.
      // Unload the redundant module inside the lock since unloading is fast
      // and also to avoid resource leaks.
      UnloadLevelZeroModule(sycl_context_.get(), lz_module_handle);
      ++(it->second.second);
      VLOG(2) << "LoadModuleFromSpirv: SPIR-V "
              << static_cast<const void*>(spirv_binary)
              << " is already loaded as module " << it->second.first;
      return module_handle;
    }
    // Cache the newly loaded module and set its reference count to 1.
    in_memory_modules_[module_handle] = lz_module_handle;
    gpu_binary_to_module_[module_handle] =
        std::make_pair(lz_module_handle, /*reference count=*/1);
  }
  VLOG(2) << "LoadModuleFromSpirv: Loaded SPIR-V "
          << static_cast<const void*>(spirv_binary) << " as module "
          << lz_module_handle;
  return module_handle;
}

bool SyclExecutor::UnloadGpuBinary(ModuleHandle module_handle) {
  auto module_it = gpu_binary_to_module_.find(module_handle);
  if (module_it == gpu_binary_to_module_.end()) {
    VLOG(3) << "SyclExecutor::UnloadGpuBinary: SPIR-V module for "
            << module_handle << " not found. It may have never been loaded.";
    return false;
  }

  // SPIR-V module found: decrement reference count.
  ze_module_handle_t lz_module_handle = module_it->second.first;
  uint64_t& ref_count = module_it->second.second;
  VLOG(3) << "SyclExecutor::UnloadGpuBinary: Found SPIR-V module "
          << lz_module_handle << " with reference count " << ref_count;

  if (--ref_count == 0) {
    // Unload SPIR-V module and remove it from the caches.
    VLOG(3) << "SyclExecutor::UnloadGpuBinary: Unloading SPIR-V module "
            << lz_module_handle;
    UnloadLevelZeroModule(sycl_context_.get(), lz_module_handle);
    gpu_binary_to_module_.erase(module_it);
    ModuleHandle mem_it{};
    // TODO(intel-tf): Optimize lookup for larger number of modules.
    for (const auto& it : in_memory_modules_) {
      if (it.second == lz_module_handle) mem_it = it.first;
    }
    if (mem_it != ModuleHandle{}) in_memory_modules_.erase(mem_it);
  }
  return true;
}

}  // namespace stream_executor::sycl
