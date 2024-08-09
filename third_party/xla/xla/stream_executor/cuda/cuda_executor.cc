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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include "absl/numeric/int128.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/gpu_diagnostics.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#define PATH_MAX MAX_PATH
#else
#include <unistd.h>
#endif

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_diagnostics.h"
#include "xla/stream_executor/cuda/cuda_driver.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/delay_kernel.h"
#include "xla/stream_executor/gpu/gpu_collectives.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_event.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_runtime.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_timer.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

// LOG(ERROR) uses a const named ERROR, so a macro with the same name is
// always unwanted. This happens on Windows that defines such a macro.
#undef ERROR

#ifdef PLATFORMS_GPUS_CUDA_DYNAMIC_LIBCUDA_DYNAMIC_LIBCUDA_H_
#error \
    "No driver calls in this file, wrap driver functionality in cuda_driver.cc."
#endif

#ifdef __CUDA_RUNTIME_H__
#error \
    "CUDA runtime being included into CUDA GPU executor; should be driver only."
#endif

extern bool FLAGS_check_gpu_leaks;
bool FLAGS_prefer_cubin_to_ptx = true;

namespace stream_executor {
namespace gpu {

namespace {

bool ShouldLaunchDelayKernel() {
  // Only launch the delay kernel if CUDA_LAUNCH_BLOCKING is not set to 1.
  static bool value = [] {
    const char* blocking = std::getenv("CUDA_LAUNCH_BLOCKING");
    return !blocking || std::string_view{blocking} != "1";
  }();
  return value;
}

}  // namespace
static GpuEvent* AsGpuEvent(Event* event) {
  DCHECK(event != nullptr);
  return static_cast<GpuEvent*>(event);
}

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

GpuContext* ExtractGpuContext(GpuExecutor* cuda_exec) {
  CHECK(cuda_exec != nullptr);
  return cuda_exec->gpu_context();
}

GpuExecutor::~GpuExecutor() {
  CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
  if (context_ != nullptr) {
    GpuDriver::DestroyContext(context_);
  }
}

absl::Status GpuExecutor::Init() {
  TF_RETURN_IF_ERROR(GpuDriver::Init());
  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(device_ordinal_, &device_));
  TF_RETURN_IF_ERROR(
      GpuDriver::CreateContext(device_ordinal_, device_, &context_));
  TF_RETURN_IF_ERROR(
      GpuDriver::GetComputeCapability(&cc_major_, &cc_minor_, device_));
  return absl::OkStatus();
}

absl::StatusOr<bool> GpuExecutor::DelayKernelIsSupported(GpuStream* stream) {
  // Check the assumption that this device supports unified addressing,
  // otherwise skip the delay kernel
  TF_ASSIGN_OR_RETURN(int status,
                      GpuDriver::GetDeviceAttribute(
                          CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device_));

  return static_cast<bool>(status);
}

absl::Status GpuExecutor::LoadModuleFromCuBin(const char* cubin,
                                              CUmodule* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[cubin];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(GpuDriver::LoadCubin(context_, cubin, module));
    module_refcount = 1;
    VLOG(3) << "Loaded CUBIN " << static_cast<const void*>(cubin)
            << " as module " << *module;
  } else {
    ++module_refcount;
    VLOG(3) << "CUBIN " << static_cast<const void*>(cubin)
            << " is already loaded as module " << *module;
  }
  gpu_binary_to_module_[cubin] = {*module, module_refcount};
  return absl::OkStatus();
}

absl::Status GpuExecutor::LoadModuleFromPtx(const char* ptx, CUmodule* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[ptx];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(GpuDriver::LoadPtx(context_, ptx, module));
    VLOG(3) << "Loaded PTX " << static_cast<const void*>(ptx) << " as module "
            << *module;
    module_refcount = 1;
  } else {
    ++module_refcount;
    VLOG(3) << "PTX " << static_cast<const void*>(ptx)
            << " is already loaded as module " << module;
  }
  gpu_binary_to_module_[ptx] = {*module, module_refcount};
  return absl::OkStatus();
}

absl::Status GpuExecutor::LoadModuleFromHsaco(const char* hsaco,
                                              CUmodule* module) {
  return absl::InternalError(
      "Feature not supported on CUDA platform (LoadModuleFromHsaco)");
}

absl::StatusOr<std::unique_ptr<Kernel>> GpuExecutor::LoadKernel(
    const MultiKernelLoaderSpec& spec) {
  auto cuda_kernel = std::make_unique<GpuKernel>(this);
  CUmodule module;
  const std::string* kernel_name;

  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    kernel_name = &spec.cuda_cubin_in_memory().kernel_name();
    const char* cubin = reinterpret_cast<const char*>(
        spec.cuda_cubin_in_memory().cubin_bytes().data());
    TF_RETURN_IF_ERROR(LoadModuleFromCuBin(cubin, &module));
    kernel_to_gpu_binary_[cuda_kernel.get()] = cubin;

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
    TF_RETURN_IF_ERROR(LoadModuleFromPtx(ptx, &module));
    kernel_to_gpu_binary_[cuda_kernel.get()] = ptx;

  } else if (spec.has_in_process_symbol()) {
    kernel_name = &spec.in_process_symbol().kernel_name();
    void* symbol = spec.in_process_symbol().symbol();

    VLOG(2) << "Resolve CUDA kernel " << *kernel_name
            << " from symbol pointer: " << symbol;
    TF_ASSIGN_OR_RETURN(
        GpuFunctionHandle function,
        GpuRuntime::GetFuncBySymbol(spec.in_process_symbol().symbol()));
    cuda_kernel->set_gpu_function(function);

  } else {
    return absl::InternalError("No method of loading CUDA kernel provided");
  }
  VLOG(3) << "LoadKernel on kernel : " << *kernel_name;
  // If we resolved kernel from a symbol pointer, there is no need to load it
  // from a module, as CUDA runtime did that automatically for us.
  if (!spec.has_in_process_symbol()) {
    VLOG(2) << "getting function " << *kernel_name << " from module " << module;
    GpuFunctionHandle function;
    TF_RETURN_IF_ERROR(GpuDriver::GetModuleFunction(
        context_, module, kernel_name->c_str(), &function));
    cuda_kernel->set_gpu_function(function);
  }

  // Update CUDA kernel properties after it was loaded in the CUDA context.
  cuda_kernel->set_name(*kernel_name);

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the CUDA API.
  cuda_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
  TF_RETURN_IF_ERROR(GetKernelMetadata(cuda_kernel.get(), &kernel_metadata));
  cuda_kernel->set_metadata(kernel_metadata);
  cuda_kernel->set_name(*kernel_name);
  cuda_kernel->set_args_packing(spec.kernel_args_packing());
  return std::move(cuda_kernel);
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>>
GpuExecutor::CreateEventBasedTimer(GpuStream* stream, bool use_delay_kernel) {
  GpuSemaphore semaphore{};
  if (!use_delay_kernel) {
    LOG(WARNING)
        << "Skipping the delay kernel, measurement accuracy will be reduced";
  }

  if (use_delay_kernel && ShouldLaunchDelayKernel()) {
    TF_ASSIGN_OR_RETURN(bool is_supported, DelayKernelIsSupported(stream));

    if (is_supported) {
      TF_ASSIGN_OR_RETURN(semaphore, LaunchDelayKernel(stream));
    } else {
      LOG(WARNING) << "Skipping the delay kernel as it's not supported, "
                      "measurement accuracy will be reduced.";
    }
  }
  TF_ASSIGN_OR_RETURN(auto start_event, CreateGpuEvent(/*allow_timing=*/true));
  TF_ASSIGN_OR_RETURN(auto stop_event, CreateGpuEvent(/*allow_timing=*/true));
  TF_RETURN_IF_ERROR(start_event->Record(stream->gpu_stream()));
  return std::make_unique<GpuTimer>(gpu_context(), std::move(start_event),
                                    std::move(stop_event), stream,
                                    std::move(semaphore));
}

bool GpuExecutor::UnloadGpuBinary(const void* gpu_binary) {
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
    GpuDriver::UnloadModule(context_, module);
    gpu_binary_to_module_.erase(module_it);
  }
  return true;
}

void GpuExecutor::UnloadKernel(const Kernel* kernel) {
  VLOG(3) << "Unloading kernel " << kernel << " : " << kernel->name();

  absl::MutexLock lock{&in_memory_modules_mu_};
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

absl::Status GpuExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                     ModuleHandle* module_handle) {
  // In GpuExecutor we store the pointer to the GPU binary (PTX or CUBIN) as
  // ModuleHandle::id().
  CUmodule cu_module;
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(LoadModuleFromCuBin(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()),
        &cu_module));
    *module_handle = ModuleHandle(const_cast<void*>(
        static_cast<const void*>(spec.cuda_cubin_in_memory().data())));
    return absl::OkStatus();
  } else if (spec.has_cuda_ptx_in_memory()) {
    if (cc_major_ == 0 && cc_minor_ == 0) {
      return absl::InternalError("Compute capability not set");
    }

    if (!spec.cuda_ptx_in_memory()) {
      return absl::InternalError("PTX not found in spec");
    }

    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(
        LoadModuleFromPtx(spec.cuda_ptx_in_memory(), &cu_module));
    *module_handle = ModuleHandle(
        const_cast<void*>(static_cast<const void*>(spec.cuda_ptx_in_memory())));
    return absl::OkStatus();
  }
  return absl::InternalError("No method of loading CUDA module provided");
}

bool GpuExecutor::UnloadModule(ModuleHandle module_handle) {
  const char* gpu_binary = reinterpret_cast<const char*>(module_handle.id());
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(gpu_binary);
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
GpuExecutor::CreateOrShareConstant(Stream* stream,
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

absl::Status GpuExecutor::GetKernelMetadata(GpuKernel* cuda_kernel,
                                            KernelMetadata* kernel_metadata) {
  int value;
  TF_RETURN_IF_ERROR(GpuDriver::FuncGetAttribute(
      CU_FUNC_ATTRIBUTE_NUM_REGS, cuda_kernel->gpu_function(), &value));
  kernel_metadata->set_registers_per_thread(value);

  TF_RETURN_IF_ERROR(
      GpuDriver::FuncGetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                  cuda_kernel->gpu_function(), &value));
  kernel_metadata->set_shared_memory_bytes(value);
  return absl::OkStatus();
}

absl::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const Kernel& kernel, const KernelArgs& args) {
  return Launch(stream, thread_dims, block_dims, std::nullopt, kernel, args);
}

absl::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const ClusterDim& cluster_dims,
                                 const Kernel& kernel, const KernelArgs& args) {
  return Launch(stream, thread_dims, block_dims,
                std::make_optional(cluster_dims), kernel, args);
}

absl::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const std::optional<ClusterDim>& cluster_dims,
                                 const Kernel& kernel, const KernelArgs& args) {
  CUstream custream = AsGpuStreamValue(stream);
  const GpuKernel* cuda_kernel = AsGpuKernel(&kernel);
  CUfunction cufunc = cuda_kernel->gpu_function();

  if (cuda_kernel->cache_config() != KernelCacheConfig::kNoPreference) {
    TF_RETURN_IF_ERROR(GpuDriver::FuncSetCacheConfig(
        cufunc, cuda_kernel->GetGpuCacheConfig()));
  }

  // Launch CUDA kernels with packed arguments.
  auto launch = [&](const KernelArgsPackedArrayBase& packed) {
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
          context_, kernel.name(), cufunc, cluster_dims->x, cluster_dims->y,
          cluster_dims->z, block_dims.x, block_dims.y, block_dims.z,
          thread_dims.x, thread_dims.y, thread_dims.z,
          packed.number_of_shared_bytes(), custream, params,
          /*extra=*/nullptr);
    } else {
      return GpuDriver::LaunchKernel(
          context_, kernel.name(), cufunc, block_dims.x, block_dims.y,
          block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
          packed.number_of_shared_bytes(), custream, params,
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

DeviceMemoryBase GpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
  if (memory_space == 1) {
    auto result = GpuCollectives::CollectiveMemoryAllocate(context_, size);
    if (!result.ok()) {
      LOG(ERROR) << result.status();
    }
    return DeviceMemoryBase(*result, size);
  } else if (memory_space ==
             static_cast<int64_t>(stream_executor::MemoryType::kHost)) {
    return DeviceMemoryBase(GpuDriver::HostAllocate(context_, size), size);
  }
  CHECK_EQ(memory_space, 0);
  return DeviceMemoryBase(GpuDriver::DeviceAllocate(context_, size), size);
}

void GpuExecutor::Deallocate(DeviceMemoryBase* mem) {
  GpuDriver::DeviceDeallocate(context_, mem->opaque());
}

bool GpuExecutor::SynchronizeAllActivity() {
  return GpuDriver::SynchronizeContext(context_);
}

absl::Status GpuExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                             uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsCudaDevicePtr(location), 0x0, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                           0x0, size);
}

absl::Status GpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src,
                                            uint64_t size) {
  return GpuDriver::SynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                         host_src, size);
}

absl::Status GpuExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64_t size) {
  return GpuDriver::SynchronousMemcpyD2H(context_, host_dst,
                                         AsCudaDevicePtr(gpu_src), size);
}

void GpuExecutor::DeallocateStream(Stream* stream) {
  {
    absl::MutexLock lock(&mu_);
    if (dnn_ != nullptr) {
      dnn_->NotifyStreamDestroyed(stream);
    }
  }
  GpuStream* gpu_stream = AsGpuStream(stream);
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(gpu_stream->gpu_stream());
}

absl::Status GpuExecutor::BlockHostUntilDone(Stream* stream) {
  return GpuDriver::SynchronizeStream(context_, AsGpuStreamValue(stream));
}

blas::BlasSupport* GpuExecutor::AsBlas() {
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

dnn::DnnSupport* GpuExecutor::AsDnn() {
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

fft::FftSupport* GpuExecutor::AsFft() {
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

bool GpuExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  GpuExecutor* cuda_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::CanEnablePeerAccess(context_, cuda_other->context_);
}

absl::Status GpuExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  GpuExecutor* cuda_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::EnablePeerAccess(context_, cuda_other->context_);
}

bool GpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  return GpuDriver::GetDeviceMemoryInfo(context_, free, total);
}

absl::StatusOr<DeviceMemoryBase> GpuExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  void* mem = nullptr;
  size_t bytes = 0;
  CHECK(static_cast<bool>(module_handle));

  {  // give limited scope to mutex_lock
    absl::MutexLock lock{&in_memory_modules_mu_};
    auto it = gpu_binary_to_module_.find(module_handle.id());
    CHECK(it != gpu_binary_to_module_.end());

    GpuModuleHandle gpu_module_handle = it->second.first;
    CHECK(gpu_module_handle != nullptr);
    TF_RETURN_IF_ERROR(GpuDriver::GetModuleSymbol(
        context_, gpu_module_handle, symbol_name.c_str(),
        reinterpret_cast<CUdeviceptr*>(&mem), &bytes));
    return DeviceMemoryBase(mem, bytes);
  }

  return absl::NotFoundError(
      absl::StrCat("Check if module containing symbol ", symbol_name,
                   " is loaded (module_handle = ",
                   reinterpret_cast<uintptr_t>(module_handle.id()), ")"));
}

absl::Status FillBlockDimLimit(GpuDeviceHandle device,
                               BlockDim* block_dim_limit) {
  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
  TF_RETURN_IF_ERROR(GpuDriver::GetGridLimits(&x, &y, &z, device));
  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<GpuEvent>> GpuExecutor::CreateGpuEvent(
    bool allow_timing) {
  auto gpu_event = std::make_unique<CudaEvent>(gpu_context());
  TF_RETURN_IF_ERROR(gpu_event->Init(allow_timing));
  return std::move(gpu_event);
}

absl::StatusOr<std::unique_ptr<Event>> GpuExecutor::CreateEvent() {
  return CreateGpuEvent(/*allow_timing=*/false);
}

absl::StatusOr<std::unique_ptr<Stream>> GpuExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(auto event, CreateGpuEvent(/*allow_timing=*/false));
  auto stream = std::make_unique<GpuStream>(this, std::move(event), priority);
  absl::MutexLock l(&alive_gpu_streams_mu_);
  TF_RETURN_IF_ERROR(stream->Init());
  auto gpu_stream = stream->gpu_stream();
  alive_gpu_streams_[gpu_stream] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>> GpuExecutor::CreateCommandBuffer(
    CommandBuffer::Mode mode) {
  VLOG(2) << "Create CUDA command buffer (CUDA graph)";
  GpuGraphHandle graph = nullptr;
  TF_RETURN_IF_ERROR(GpuDriver::CreateGraph(&graph));
  return std::make_unique<GpuCommandBuffer>(mode, /*parent=*/this, graph);
}

std::unique_ptr<GpuCommandBuffer> GpuExecutor::CreateCommandBuffer(
    CommandBuffer::Mode mode, GpuGraphHandle graph, bool is_owned_graph) {
  VLOG(2) << "Create CUDA command buffer (CUDA graph) from existing graph "
          << graph << "; is_owned_graph=" << is_owned_graph;
  return std::make_unique<GpuCommandBuffer>(mode, /*parent=*/this, graph,
                                            is_owned_graph);
}

GpuContext* GpuExecutor::gpu_context() { return context_; }

// Attempts to read the NUMA node corresponding to the GPU device's PCI bus out
// of SysFS. Returns -1 if it cannot.
//
// For anything more complicated/prod-focused than this, you'll likely want to
// turn to gsys' topology modeling.
static int TryToReadNumaNode(const std::string& pci_bus_id,
                             int device_ordinal) {
#if defined(PLATFORM_WINDOWS)
  // Windows support for NUMA is not currently implemented. Return node 0.
  return 0;
#else
  VLOG(2) << "trying to read NUMA node for device ordinal: " << device_ordinal;
  static const int kUnknownNumaNode = -1;

  if (pci_bus_id.empty()) {
    LOG(INFO) << "no PCI bus ID for device ordinal: " << device_ordinal;
    return kUnknownNumaNode;
  }

  std::string filename =
      absl::StrFormat("/sys/bus/pci/devices/%s/numa_node", pci_bus_id);

  // We have to use fopen/fread here so that the device properties can be
  // populated before InitGoogle procedure has been completed (at which point we
  // could use the file::* utilities).
  FILE* file = fopen(filename.c_str(), "r");
  if (file == nullptr) {
    LOG(INFO) << "could not open file to read NUMA node: " << filename
              << "\nYour kernel may have been built without NUMA support.";
    return kUnknownNumaNode;
  }

  std::string content;
  char buf[32];
  size_t did_read = fread(buf, sizeof(buf[0]), sizeof(buf) - 1, file);
  buf[did_read] = '\0';
  content = buf;

  int32_t value;
  if (absl::SimpleAtoi(content, &value)) {
    if (value < 0) {  // See http://b/18228951 for details on this path.
      LOG(INFO) << "successful NUMA node read from SysFS had negative value ("
                << value
                << "), but there must be at least one NUMA node"
                   ", so returning NUMA node zero."
                   " See more at "
                   "https://github.com/torvalds/linux/blob/v6.0/Documentation/"
                   "ABI/testing/sysfs-bus-pci#L344-L355";
      fclose(file);
      return 0;
    }
    fclose(file);
    return value;
  }

  LOG(WARNING)
      << "could not convert SysFS file contents to integral NUMA node value: "
      << content;

  fclose(file);
  return kUnknownNumaNode;
#endif
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
GpuExecutor::CreateDeviceDescription(int device_ordinal) {
  GpuDeviceHandle device;
  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(device_ordinal, &device));

  int cc_major;
  int cc_minor;
  TF_RETURN_IF_ERROR(
      GpuDriver::GetComputeCapability(&cc_major, &cc_minor, device));

  internal::DeviceDescriptionBuilder builder;

  {
    int driver_version = GpuDriver::GetDriverVersion().value_or(0);
    std::string augmented_driver_version = absl::StrFormat(
        "%d (%s)", driver_version,
        cuda::DriverVersionStatusToString(Diagnostician::FindDsoVersion()));
    builder.set_driver_version(augmented_driver_version);
  }

  {
    std::string pci_bus_id = GpuDriver::GetPCIBusID(device);

    // Lower the hex characters to match sysfs.
    pci_bus_id = absl::AsciiStrToLower(pci_bus_id);
    builder.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = TryToReadNumaNode(pci_bus_id, device_ordinal);
    builder.set_numa_node(numa_node);
  }

  {
    builder.set_threads_per_block_limit(
        GpuDriver::GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                      device)
            .value());

    ThreadDim thread_dim_limit;
    thread_dim_limit.x = GpuDriver::GetDeviceAttribute(
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device)
                             .value();
    thread_dim_limit.y = GpuDriver::GetDeviceAttribute(
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device)
                             .value();
    thread_dim_limit.z = GpuDriver::GetDeviceAttribute(
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device)
                             .value();
    builder.set_thread_dim_limit(thread_dim_limit);
  }

  int sm_clock_khz =
      GpuDriver::GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device)
          .value();
  builder.set_clock_rate_ghz(static_cast<float>(sm_clock_khz) / 1e6);

  {
    bool ecc_enabled = false;
    (void)GpuDriver::IsEccEnabled(device, &ecc_enabled);
    builder.set_ecc_enabled(ecc_enabled);
  }

  uint64_t device_memory_size = static_cast<uint64_t>(-1);
  (void)GpuDriver::GetDeviceTotalMemory(device, &device_memory_size);
  builder.set_device_memory_size(device_memory_size);

  int64_t l2_cache_bytes =
      GpuDriver::GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device)
          .value();
  builder.set_l2_cache_size(l2_cache_bytes);

  absl::StatusOr<int> mem_clock_khz = GpuDriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device_ordinal);
  absl::StatusOr<int> mem_bus_width_bits = GpuDriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device_ordinal);
  if (mem_clock_khz.ok() && mem_bus_width_bits.ok()) {
    // Times 2 because HBM is DDR memory; it gets two data bits per each data
    // lane.
    builder.set_memory_bandwidth(2 * int64_t{mem_clock_khz.value()} * 1000 *
                                 int64_t{mem_bus_width_bits.value()} / 8);
  }

  {
    BlockDim block_dim_limit;
    TF_RETURN_IF_ERROR(FillBlockDimLimit(device, &block_dim_limit));
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    std::string device_name;
    TF_RETURN_IF_ERROR(GpuDriver::GetDeviceName(device, &device_name));
    builder.set_name(device_name);
  }

  builder.set_platform_version(
      absl::StrCat("Compute Capability ", cc_major, ".", cc_minor));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  builder.set_device_address_bits(64);

  builder.set_device_vendor("NVIDIA Corporation");
  builder.set_cuda_compute_capability(cc_major, cc_minor);
  builder.set_shared_memory_per_core(
      GpuDriver::GetMaxSharedMemoryPerCore(device).value());
  builder.set_shared_memory_per_block(
      GpuDriver::GetMaxSharedMemoryPerBlock(device).value());
  builder.set_shared_memory_per_block_optin(
      GpuDriver::GetMaxSharedMemoryPerBlockOptin(device).value());
  int core_count = GpuDriver::GetMultiprocessorCount(device).value();
  builder.set_core_count(core_count);
  builder.set_fpus_per_core(fpus_per_core(cc_major, cc_minor));
  builder.set_threads_per_core_limit(
      GpuDriver::GetMaxThreadsPerMultiprocessor(device).value());
  builder.set_registers_per_block_limit(
      GpuDriver::GetMaxRegistersPerBlock(device).value());
  builder.set_threads_per_warp(GpuDriver::GetThreadsPerWarp(device).value());
  builder.set_registers_per_core_limit(
      GpuDriver::GetDeviceAttribute(
          CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device)
          .value());

  auto value_or = [](const auto& status_or, auto default_val) {
    if (status_or.ok()) return *status_or;
    return default_val;
  };

  // It would be better to use the PCI device ID or some other truly unique
  // identifier for the GPU model.  But getting this requires using NVML or
  // other hacks, which we don't have access to in OSS TensorFlow.
  //
  // Alternatively you might be tempted to use GpuDriver::GetDeviceName as a
  // unique identifier, but this is not stable across GPU VBIOS versions.
  //
  // For now, this identifier is good enough.
  builder.set_model_str(absl::StrFormat(
      "sm_%d.%d with %dB RAM, %d cores, %dKHz clock, %dKHz mem clock, %dB L2$",
      cc_major, cc_minor, device_memory_size, core_count, sm_clock_khz,
      value_or(mem_clock_khz, 0), l2_cache_bytes));

  return builder.Build();
}

}  // namespace gpu
}  // namespace stream_executor
