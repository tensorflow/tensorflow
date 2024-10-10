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

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/debugging/leak_check.h"
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
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_collectives.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/cuda/cuda_kernel.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_runtime.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/cuda_stream.h"
#include "xla/stream_executor/cuda/cuda_timer.h"
#include "xla/stream_executor/cuda/cuda_version_parser.h"
#include "xla/stream_executor/cuda/delay_kernel.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/gpu/read_numa_node.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

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

absl::Status FuncGetAttribute(CUfunction_attribute attribute, CUfunction func,
                              int* attribute_value) {
  return cuda::ToStatus(
      cuFuncGetAttribute(attribute_value, attribute, func),
      absl::StrCat("Failed to query kernel attribute: ", attribute));
}

// CUDA driver routines may require a large amount of stack (particularly
// cuModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
tsl::thread::ThreadPool* GetDriverExecutor() {
  static tsl::thread::ThreadPool* thread_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), tsl::ThreadOptions(), "cuda_driver", 1);
  return thread_pool;
}

// Loads ptx_contents with the CUDA driver's PTX JIT and stores the resulting
// handle in "module". Any error logs that are produced are logged internally.
absl::Status LoadPtx(Context* context, const char* ptx_contents,
                     CUmodule* module) {
  absl::Notification notification;
  absl::Status returned_status = absl::OkStatus();
  GetDriverExecutor()->Schedule(
      [context, ptx_contents, module, &returned_status, &notification]() {
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
        {
          // TODO(leary) Need to see if NVIDIA can expunge the leakiness in
          // their module loading: see http://b/13248943
          absl::LeakCheckDisabler disabler;
          status = cuda::ToStatus(cuModuleLoadDataEx(
              module, ptx_data, TF_ARRAYSIZE(options), options, option_values));
        }

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

  return returned_status;
}

// Loads cubin_bytes with the CUDA driver's blob loading interface and stores
// the resulting handle in "module".
absl::Status LoadCubin(Context* context, const char* cubin_bytes,
                       CUmodule* module) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(
      cuModuleLoadFatBinary(module, cubin_bytes),
      "Failed to load in-memory CUBIN (compiled for a different GPU?).");
}

// Retrieves a named kernel from a loaded module, and places the resulting
// handle into function (outparam) on success. Neither kernel_name nor
// function may be null. No ownership is taken of kernel_name.
absl::Status GetModuleFunction(Context* context, CUmodule module,
                               const char* kernel_name, CUfunction* function) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  cudaError_t cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return absl::InternalError(
        absl::StrCat("There was an error before calling cuModuleGetFunction (",
                     cuda_error, "): ", cudaGetErrorName(cuda_error), " : ",
                     cudaGetErrorString(cuda_error)));
  }
  return cuda::ToStatus(cuModuleGetFunction(function, module, kernel_name),
                        "Failed to get module function");
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

CudaExecutor::~CudaExecutor() {
  CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
  set_context(nullptr);
}

absl::Status CudaExecutor::Init() {
  TF_RETURN_IF_ERROR(GpuDriver::Init());
  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(device_ordinal(), &device_));
  TF_ASSIGN_OR_RETURN(Context * context,
                      CudaContext::Create(device_ordinal(), device_));
  set_context(context);
  TF_RETURN_IF_ERROR(
      GpuDriver::GetComputeCapability(&cc_major_, &cc_minor_, device_));
  TF_ASSIGN_OR_RETURN(delay_kernels_supported_, DelayKernelIsSupported());
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

absl::Status CudaExecutor::LoadModuleFromCuBin(const char* cubin,
                                               CUmodule* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[cubin];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(LoadCubin(gpu_context(), cubin, module));
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

absl::Status CudaExecutor::LoadModuleFromPtx(const char* ptx,
                                             CUmodule* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[ptx];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(LoadPtx(gpu_context(), ptx, module));
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

absl::Status CudaExecutor::LoadModuleFromHsaco(const char* hsaco,
                                               CUmodule* module) {
  return absl::InternalError(
      "Feature not supported on CUDA platform (LoadModuleFromHsaco)");
}

absl::StatusOr<std::unique_ptr<Kernel>> CudaExecutor::LoadKernel(
    const MultiKernelLoaderSpec& spec) {
  auto cuda_kernel = std::make_unique<CudaKernel>(this);
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
        CudaRuntime::GetFuncBySymbol(spec.in_process_symbol().symbol()));
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
    TF_RETURN_IF_ERROR(GetModuleFunction(gpu_context(), module,
                                         kernel_name->c_str(), &function));
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
CudaExecutor::CreateEventBasedTimer(GpuStream* stream, bool use_delay_kernel) {
  GpuSemaphore semaphore{};

  if (use_delay_kernel && ShouldLaunchDelayKernel() &&
      delay_kernels_supported_) {
    TF_ASSIGN_OR_RETURN(semaphore, LaunchDelayKernel(stream));
  }
  TF_ASSIGN_OR_RETURN(auto start_event, CreateGpuEvent(/*allow_timing=*/true));
  TF_ASSIGN_OR_RETURN(auto stop_event, CreateGpuEvent(/*allow_timing=*/true));
  TF_RETURN_IF_ERROR(stream->RecordEvent(start_event.get()));
  return std::make_unique<CudaTimer>(gpu_context(), std::move(start_event),
                                     std::move(stop_event), stream,
                                     std::move(semaphore));
}

bool CudaExecutor::UnloadGpuBinary(const void* gpu_binary) {
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
    UnloadCudaModule(gpu_context(), module);
    gpu_binary_to_module_.erase(module_it);
  }
  return true;
}

void CudaExecutor::UnloadKernel(const Kernel* kernel) {
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

absl::Status CudaExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
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

bool CudaExecutor::UnloadModule(ModuleHandle module_handle) {
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

absl::Status CudaExecutor::GetKernelMetadata(GpuKernel* cuda_kernel,
                                             KernelMetadata* kernel_metadata) {
  int value;
  TF_RETURN_IF_ERROR(FuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS,
                                      cuda_kernel->gpu_function(), &value));
  kernel_metadata->set_registers_per_thread(value);

  TF_RETURN_IF_ERROR(FuncGetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                      cuda_kernel->gpu_function(), &value));
  kernel_metadata->set_shared_memory_bytes(value);
  return absl::OkStatus();
}

DeviceMemoryBase CudaExecutor::Allocate(uint64_t size, int64_t memory_space) {
  if (memory_space == 1) {
    auto result =
        CudaCollectives::CollectiveMemoryAllocate(gpu_context(), size);
    if (!result.ok()) {
      LOG(ERROR) << result.status();
    }
    return DeviceMemoryBase(nullptr, 0);
  } else if (memory_space ==
             static_cast<int64_t>(stream_executor::MemoryType::kHost)) {
    return DeviceMemoryBase(GpuDriver::HostAllocate(gpu_context(), size), size);
  }
  CHECK_EQ(memory_space, 0);
  return DeviceMemoryBase(GpuDriver::DeviceAllocate(gpu_context(), size), size);
}

void CudaExecutor::Deallocate(DeviceMemoryBase* mem) {
  auto status_or_memory_space = GetPointerMemorySpace(mem->opaque());
  if (!status_or_memory_space.ok()) {
    LOG(ERROR) << status_or_memory_space.status();
    return;
  }
  auto memory_space = status_or_memory_space.value();
  if (memory_space == MemoryType::kHost) {
    GpuDriver::HostDeallocate(gpu_context(), mem->opaque());
  } else {
    GpuDriver::DeviceDeallocate(gpu_context(), mem->opaque());
  }
}

bool CudaExecutor::SynchronizeAllActivity() {
  return gpu_context()->Synchronize().ok();
}

bool CudaExecutor::HostMemoryRegister(void* location, uint64_t size) {
  VLOG(1) << "Called StreamExecutor::HostMemoryRegister(data=" << location
          << ")";

  return GpuDriver::HostRegister(gpu_context(), location, size);
}

bool CudaExecutor::HostMemoryUnregister(void* location) {
  VLOG(1) << "Called StreamExecutor::HostUnregister(data=" << location << ")";

  return GpuDriver::HostUnregister(gpu_context(), location);
}

absl::Status CudaExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return GpuDriver::SynchronousMemsetUint32(
        gpu_context(), AsCudaDevicePtr(location), 0x0, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(
      gpu_context(), AsCudaDevicePtr(location), 0x0, size);
}

absl::Status CudaExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  return GpuDriver::SynchronousMemcpyH2D(
      gpu_context(), AsCudaDevicePtr(gpu_dst), host_src, size);
}

absl::Status CudaExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
  return GpuDriver::SynchronousMemcpyD2H(gpu_context(), host_dst,
                                         AsCudaDevicePtr(gpu_src), size);
}

void CudaExecutor::DeallocateStream(Stream* stream) {
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

absl::Status CudaExecutor::BlockHostUntilDone(Stream* stream) {
  return GpuDriver::SynchronizeStream(gpu_context(), AsGpuStreamValue(stream));
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
  GpuExecutor* cuda_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::CanEnablePeerAccess(gpu_context(),
                                        cuda_other->gpu_context());
}

absl::Status CudaExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  GpuExecutor* cuda_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::EnablePeerAccess(gpu_context(), cuda_other->gpu_context());
}

bool CudaExecutor::DeviceMemoryUsage(int64_t* free_out,
                                     int64_t* total_out) const {
  ScopedActivateContext activation(gpu_context());
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

  {  // give limited scope to mutex_lock
    absl::MutexLock lock{&in_memory_modules_mu_};
    auto it = gpu_binary_to_module_.find(module_handle.id());
    CHECK(it != gpu_binary_to_module_.end());

    GpuModuleHandle gpu_module_handle = it->second.first;
    CHECK(gpu_module_handle != nullptr);
    TF_RETURN_IF_ERROR(
        GetModuleSymbol(gpu_context(), gpu_module_handle, symbol_name.c_str(),
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

absl::StatusOr<std::unique_ptr<GpuEvent>> CudaExecutor::CreateGpuEvent(
    bool allow_timing) {
  auto gpu_event = std::make_unique<CudaEvent>(gpu_context());
  TF_RETURN_IF_ERROR(gpu_event->Init(allow_timing));
  return std::move(gpu_event);
}

absl::StatusOr<std::unique_ptr<Event>> CudaExecutor::CreateEvent() {
  return CreateGpuEvent(/*allow_timing=*/false);
}

absl::StatusOr<std::unique_ptr<Stream>> CudaExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(auto event, CreateGpuEvent(/*allow_timing=*/false));
  TF_ASSIGN_OR_RETURN(auto stream,
                      CudaStream::Create(this, std::move(event), priority));
  absl::MutexLock l(&alive_gpu_streams_mu_);
  auto gpu_stream = stream->gpu_stream();
  alive_gpu_streams_[gpu_stream] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>>
CudaExecutor::CreateCommandBuffer(CommandBuffer::Mode mode) {
  VLOG(2) << "Create CUDA command buffer (CUDA graph)";
  GpuGraphHandle graph = nullptr;
  TF_RETURN_IF_ERROR(GpuDriver::CreateGraph(&graph));
  return std::make_unique<GpuCommandBuffer>(mode, /*parent=*/this, graph);
}

absl::Status CudaExecutor::TrimGraphMemory() {
  return GpuDriver::DeviceGraphMemTrim(device_);
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
CudaExecutor::CreateDeviceDescription(int device_ordinal) {
  GpuDeviceHandle device;
  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(device_ordinal, &device));

  int cc_major;
  int cc_minor;
  TF_RETURN_IF_ERROR(
      GpuDriver::GetComputeCapability(&cc_major, &cc_minor, device));

  DeviceDescription desc;

  desc.set_driver_version(
      ParseCudaVersion(GpuDriver::GetDriverVersion().value_or(0))
          .value_or(SemanticVersion{0, 0, 0}));
  desc.set_runtime_version(
      ParseCudaVersion(CudaRuntime::GetRuntimeVersion().value_or(0))
          .value_or(SemanticVersion{0, 0, 0}));
  desc.set_compile_time_toolkit_version(
      ParseCudaVersion(CUDA_VERSION).value_or(SemanticVersion{0, 0, 0}));

  {
    std::string pci_bus_id = GpuDriver::GetPCIBusID(device);

    // Lower the hex characters to match sysfs.
    pci_bus_id = absl::AsciiStrToLower(pci_bus_id);
    desc.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = ReadNumaNode(pci_bus_id, device_ordinal);
    desc.set_numa_node(numa_node);
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
    (void)GpuDriver::IsEccEnabled(device, &ecc_enabled);
    desc.set_ecc_enabled(ecc_enabled);
  }

  uint64_t device_memory_size = static_cast<uint64_t>(-1);
  (void)GpuDriver::GetDeviceTotalMemory(device, &device_memory_size);
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
    std::string device_name;
    TF_RETURN_IF_ERROR(GpuDriver::GetDeviceName(device, &device_name));
    desc.set_name(device_name);
  }

  desc.set_platform_version(
      absl::StrCat("Compute Capability ", cc_major, ".", cc_minor));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  desc.set_device_address_bits(64);

  desc.set_device_vendor("NVIDIA Corporation");
  desc.set_cuda_compute_capability(cc_major, cc_minor);
  desc.set_shared_memory_per_core(
      GpuDriver::GetMaxSharedMemoryPerCore(device).value());
  desc.set_shared_memory_per_block(
      GpuDriver::GetMaxSharedMemoryPerBlock(device).value());
  desc.set_shared_memory_per_block_optin(
      GpuDriver::GetMaxSharedMemoryPerBlockOptin(device).value());
  int core_count = GpuDriver::GetMultiprocessorCount(device).value();
  desc.set_core_count(core_count);
  desc.set_fpus_per_core(fpus_per_core(cc_major, cc_minor));
  desc.set_threads_per_core_limit(
      GpuDriver::GetMaxThreadsPerMultiprocessor(device).value());
  desc.set_registers_per_block_limit(
      GpuDriver::GetMaxRegistersPerBlock(device).value());
  desc.set_threads_per_warp(GpuDriver::GetThreadsPerWarp(device).value());
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
  // Alternatively you might be tempted to use GpuDriver::GetDeviceName as a
  // unique identifier, but this is not stable across GPU VBIOS versions.
  //
  // For now, this identifier is good enough.
  desc.set_model_str(absl::StrFormat(
      "sm_%d.%d with %dB RAM, %d cores, %dKHz clock, %dKHz mem clock, %dB L2$",
      cc_major, cc_minor, device_memory_size, core_count, sm_clock_khz,
      value_or(mem_clock_khz, 0), l2_cache_bytes));

  return std::make_unique<DeviceDescription>(std::move(desc));
}

}  // namespace gpu
}  // namespace stream_executor
