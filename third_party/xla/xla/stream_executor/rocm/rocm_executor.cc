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
#include <utility>
#include <variant>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hip/hip_version.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/gpu/read_numa_node.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/rocm/rocm_context.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_kernel.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/rocm/rocm_runtime.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/rocm/rocm_stream.h"
#include "xla/stream_executor/rocm/rocm_timer.h"
#include "xla/stream_executor/rocm/rocm_version_parser.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

#define RETURN_IF_ROCM_ERROR(expr, ...)                                  \
  do {                                                                   \
    hipError_t _res = (expr);                                            \
    if (TF_PREDICT_FALSE(_res != hipSuccess)) {                          \
      if (_res == hipErrorOutOfMemory)                                   \
        return absl::ResourceExhaustedError(absl::StrCat(                \
            __VA_ARGS__, ":", ::stream_executor::gpu::ToString(_res)));  \
      else                                                               \
        return absl::InternalError(absl::StrCat(                         \
            __VA_ARGS__, ": ", ::stream_executor::gpu::ToString(_res))); \
    }                                                                    \
  } while (0)

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

absl::Status FuncGetAttribute(hipFunction_attribute attribute,
                              hipFunction_t func, int* attribute_value) {
  RETURN_IF_ROCM_ERROR(
      wrap::hipFuncGetAttribute(attribute_value, attribute, func),
      "Failed to query kernel attribute: ", attribute);
  return absl::OkStatus();
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
absl::Status LoadHsaco(Context* context, const char* hsaco_contents,
                       hipModule_t* module) {
  absl::Notification notification;
  absl::Status returned_status = absl::OkStatus();
  GetDriverExecutor()->Schedule(
      [context, hsaco_contents, module, &returned_status, &notification]() {
        ScopedActivateContext activation{context};
        void* hsaco_data = const_cast<char*>(hsaco_contents);

        hipError_t res = wrap::hipModuleLoadData(module, hsaco_data);

        if (res != hipSuccess) {
          returned_status = absl::InternalError(
              absl::StrCat("Failed to load HSACO: ", ToString(res)));
          notification.Notify();
        }

        CHECK(module != nullptr);
        notification.Notify();
      });
  notification.WaitForNotification();

  return returned_status;
}

// Retrieves a named kernel from a loaded module, and places the resulting
// handle into function (outparam) on success. Neither kernel_name nor
// function may be null. No ownership is taken of kernel_name.
absl::Status GetModuleFunction(Context* context, hipModule_t module,
                               const char* kernel_name,
                               hipFunction_t* function) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  RETURN_IF_ROCM_ERROR(
      wrap::hipModuleGetFunction(function, module, kernel_name),
      "Failed to get kernel");
  return absl::OkStatus();
}

// Retrieves a named global/constant symbol from a loaded module, and returns
// a device pointer and size of the symbol on success. symbol_name may not be
// null. At least one of dptr or bytes should not be null. No ownership is
// taken of symbol_name.
absl::Status GetModuleSymbol(Context* context, hipModule_t module,
                             const char* symbol_name, hipDeviceptr_t* dptr,
                             size_t* bytes) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  RETURN_IF_ROCM_ERROR(
      wrap::hipModuleGetGlobal(dptr, bytes, module, symbol_name),
      absl::StrCat("Failed to get symbol '", symbol_name, "'"));
  return absl::OkStatus();
}

// Unloads module from the current context via cuModuleUnload.
void UnloadRocmModule(Context* context, hipModule_t module) {
  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipModuleUnload(module);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << ToString(res);
  }
}
}  // namespace

RocmExecutor::~RocmExecutor() {
  for (auto& it : disk_modules_) {
    UnloadRocmModule(gpu_context(), it.second);
  }
  for (auto& it : in_memory_modules_) {
    UnloadRocmModule(gpu_context(), it.second);
  }
  set_context(nullptr);
  CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
}

bool RocmExecutor::UnloadModule(ModuleHandle module_handle) {
  const char* gpu_binary = reinterpret_cast<const char*>(module_handle.id());
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(gpu_binary);
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
RocmExecutor::CreateEventBasedTimer(GpuStream* stream, bool use_delay_kernel) {
  TF_ASSIGN_OR_RETURN(auto timer, RocmTimer::Create(gpu_context(), stream));
  return std::make_unique<RocmTimer>(std::move(timer));
}

bool RocmExecutor::UnloadGpuBinary(const void* gpu_binary) {
  auto module_it = gpu_binary_to_module_.find(gpu_binary);
  if (gpu_binary_to_module_.end() == module_it) {
    VLOG(3) << "No loaded  HSACO module for " << gpu_binary;
    return false;
  }
  auto& module = module_it->second.first;
  auto& refcount = module_it->second.second;
  VLOG(3) << "Found HSACO module " << module << " with refcount " << refcount;
  if (--refcount == 0) {
    VLOG(3) << "Unloading  HSACO module " << module;
    UnloadRocmModule(gpu_context(), module);
    gpu_binary_to_module_.erase(module_it);
    const char* mem_it = nullptr;
    for (auto x : in_memory_modules_) {
      if (x.second == module) mem_it = x.first;
    }
    if (mem_it != nullptr) in_memory_modules_.erase(mem_it);
  }
  return true;
}

void RocmExecutor::UnloadKernel(const Kernel* kernel) {
  VLOG(3) << "Unloading kernel " << kernel << " : " << kernel->name();

  absl::MutexLock lock{&in_memory_modules_mu_};
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
  TF_RETURN_IF_ERROR(GpuDriver::Init());

  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(device_ordinal(), &device_));

  TF_ASSIGN_OR_RETURN(rocm_context_,
                      RocmContext::Create(device_ordinal(), device_));
  set_context(rocm_context_);
  return GpuDriver::GetGpuISAVersion(&version_, device_);
}

absl::StatusOr<std::unique_ptr<Kernel>> RocmExecutor::LoadKernel(
    const MultiKernelLoaderSpec& spec) {
  auto rocm_kernel = std::make_unique<RocmKernel>(this);
  hipModule_t module = nullptr;
  const std::string* kernel_name;

  if (spec.has_cuda_cubin_in_memory()) {
    kernel_name = &spec.cuda_cubin_in_memory().kernel_name();

    const char* hsaco = reinterpret_cast<const char*>(
        spec.cuda_cubin_in_memory().cubin_bytes().data());
    absl::MutexLock lock{&in_memory_modules_mu_};
    module = in_memory_modules_[hsaco];

    if (module == nullptr) {
      TF_RETURN_IF_ERROR(LoadHsaco(gpu_context(), hsaco, &module));
    }
    kernel_to_gpu_binary_[rocm_kernel.get()] = hsaco;
  } else if (spec.has_in_process_symbol()) {
    kernel_name = &spec.in_process_symbol().kernel_name();
    void* symbol = spec.in_process_symbol().symbol();

    VLOG(1) << "Resolve ROCM kernel " << *kernel_name
            << " from symbol pointer: " << symbol;

#if TF_ROCM_VERSION >= 60200
    TF_ASSIGN_OR_RETURN(
        GpuFunctionHandle function,
        RocmRuntime::GetFuncBySymbol(spec.in_process_symbol().symbol()));
    rocm_kernel->set_gpu_function(function);
#else
    rocm_kernel->set_gpu_function(
        static_cast<hipFunction_t>(spec.in_process_symbol().symbol()));
#endif  // TF_ROCM_VERSION >= 60200

  } else {
    return absl::InternalError("No method of loading ROCM kernel provided");
  }

  // If we resolved kernel from a symbol pointer, there is no need to load it
  // from a module, as ROCm runtime did that automatically for us.
  if (!spec.has_in_process_symbol()) {
    VLOG(2) << "getting function " << *kernel_name << " from module " << module;
    GpuFunctionHandle function;
    TF_RETURN_IF_ERROR(GetModuleFunction(gpu_context(), module,
                                         kernel_name->c_str(), &function));
    rocm_kernel->set_gpu_function(function);
  }

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the ROCM API.
  rocm_kernel->set_arity(spec.arity());

  // unable to get kernel metadata for in-process kernel
  if (!spec.has_in_process_symbol()) {
    KernelMetadata kernel_metadata;
    TF_RETURN_IF_ERROR(GetKernelMetadata(rocm_kernel.get(), &kernel_metadata));
    rocm_kernel->set_metadata(kernel_metadata);
  }
  rocm_kernel->set_name(*kernel_name);
  rocm_kernel->set_args_packing(spec.kernel_args_packing());
  return std::move(rocm_kernel);
}

absl::Status RocmExecutor::GetKernelMetadata(GpuKernel* rocm_kernel,
                                             KernelMetadata* kernel_metadata) {
  int value = 0;
  TF_RETURN_IF_ERROR(FuncGetAttribute(HIP_FUNC_ATTRIBUTE_NUM_REGS,
                                      rocm_kernel->gpu_function(), &value));
  kernel_metadata->set_registers_per_thread(value);

  TF_RETURN_IF_ERROR(FuncGetAttribute(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                      rocm_kernel->gpu_function(), &value));
  kernel_metadata->set_shared_memory_bytes(value);
  return absl::OkStatus();
}

absl::Status RocmExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                      ModuleHandle* module_handle) {
  // In GpuExecutor we store the pointer to the  HSACO binary  as
  // ModuleHandle::id().
  hipModule_t hip_module = nullptr;
  // TODO(ROCm): Need  generic term instead of cubin/cuda/ptx
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(LoadModuleFromHsaco(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()),
        &hip_module));
    *module_handle = ModuleHandle(const_cast<void*>(
        static_cast<const void*>(spec.cuda_cubin_in_memory().data())));
    return absl::OkStatus();
  } else {
    return absl::InternalError("No HASCO binary found");
  }
}

absl::Status RocmExecutor::LoadModuleFromHsaco(const char* hsaco,
                                               hipModule_t* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[hsaco];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(LoadHsaco(gpu_context(), hsaco, module));
    module_refcount = 1;
    in_memory_modules_[hsaco] = *module;
    VLOG(3) << "Loaded HSACO " << static_cast<const void*>(hsaco)
            << " as module " << *module;
  } else {
    ++module_refcount;
    VLOG(3) << "HSACO " << static_cast<const void*>(hsaco)
            << " is already loaded as module " << *module;
  }
  gpu_binary_to_module_[hsaco] = {*module, module_refcount};
  return absl::OkStatus();
}

DeviceMemoryBase RocmExecutor::Allocate(uint64_t size, int64_t memory_space) {
  if (memory_space ==
      static_cast<int64_t>(stream_executor::MemoryType::kHost)) {
    return DeviceMemoryBase(GpuDriver::HostAllocate(gpu_context(), size), size);
  }
  CHECK_EQ(memory_space, 0);
  return DeviceMemoryBase(GpuDriver::DeviceAllocate(gpu_context(), size), size);
}

void RocmExecutor::Deallocate(DeviceMemoryBase* mem) {
  GpuDriver::DeviceDeallocate(gpu_context(), mem->opaque());
}

bool RocmExecutor::SynchronizeAllActivity() {
  return gpu_context()->Synchronize().ok();
}

absl::Status RocmExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return GpuDriver::SynchronousMemsetUint32(
        gpu_context(), AsROCmDevicePtr(location), 0x0, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(
      gpu_context(), AsROCmDevicePtr(location), 0x0, size);
}

absl::Status RocmExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  return GpuDriver::SynchronousMemcpyH2D(
      gpu_context(), AsROCmDevicePtr(gpu_dst), host_src, size);
}

absl::Status RocmExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
  return GpuDriver::SynchronousMemcpyD2H(gpu_context(), host_dst,
                                         AsROCmDevicePtr(gpu_src), size);
}

void RocmExecutor::DeallocateStream(Stream* stream) {
  {
    absl::MutexLock lock(&mu_);
    if (dnn_ != nullptr) {
      dnn_->NotifyStreamDestroyed(stream);
    }
  }
  GpuStream* rocm_stream = AsGpuStream(stream);
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(rocm_stream->gpu_stream());
}

absl::Status RocmExecutor::BlockHostUntilDone(Stream* stream) {
  return GpuDriver::SynchronizeStream(gpu_context(), AsGpuStreamValue(stream));
}

blas::BlasSupport* RocmExecutor::AsBlas() {
  absl::MutexLock lock(&mu_);
  if (blas_ != nullptr) {
    return blas_.get();
  }

  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(rocm::kROCmPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().message();
    return nullptr;
  }

  auto blas = status.value()(this);
  blas_.reset(blas);
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
  GpuExecutor* rocm_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::CanEnablePeerAccess(gpu_context(),
                                        rocm_other->gpu_context());
}

absl::Status RocmExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  GpuExecutor* rocm_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::EnablePeerAccess(gpu_context(), rocm_other->gpu_context());
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
    auto it = gpu_binary_to_module_.find(module_handle.id());
    CHECK(it != gpu_binary_to_module_.end());
    TF_RETURN_IF_ERROR(
        GetModuleSymbol(gpu_context(), it->second.first, symbol_name.c_str(),
                        reinterpret_cast<hipDeviceptr_t*>(&mem), &bytes));
    return DeviceMemoryBase(mem, bytes);
  }

  for (auto& it : gpu_binary_to_module_) {
    TF_RETURN_IF_ERROR(
        GetModuleSymbol(gpu_context(), it.second.first, symbol_name.c_str(),
                        reinterpret_cast<hipDeviceptr_t*>(&mem), &bytes));
    return DeviceMemoryBase(mem, bytes);
  }

  LOG(INFO) << "Falied to find symbol in any modules: " << symbol_name;
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

absl::StatusOr<std::unique_ptr<Event>> RocmExecutor::CreateEvent() {
  TF_ASSIGN_OR_RETURN(auto event,
                      RocmEvent::Create(gpu_context(), /*allow_timing=*/false));
  return std::make_unique<RocmEvent>(std::move(event));
}

absl::StatusOr<std::unique_ptr<Stream>> RocmExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(auto event,
                      RocmEvent::Create(gpu_context(), /*allow_timing=*/false));
  TF_ASSIGN_OR_RETURN(auto stream,
                      RocmStream::Create(this, std::move(event), priority));
  absl::MutexLock l(&alive_gpu_streams_mu_);
  auto gpu_stream = stream->gpu_stream();
  alive_gpu_streams_[gpu_stream] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>>
RocmExecutor::CreateCommandBuffer(CommandBuffer::Mode mode) {
  VLOG(2) << "Create ROCm command buffer (ROCm graph)";
  GpuGraphHandle graph = nullptr;
  TF_RETURN_IF_ERROR(GpuDriver::CreateGraph(&graph));
  return std::make_unique<GpuCommandBuffer>(mode, /*parent=*/this, graph);
}

absl::Status RocmExecutor::TrimGraphMemory() {
  return GpuDriver::DeviceGraphMemTrim(device_);
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
RocmExecutor::CreateDeviceDescription(int device_ordinal) {
  GpuDeviceHandle device;
  auto status = GpuDriver::GetDevice(device_ordinal, &device);
  if (!status.ok()) {
    return status;
  }

  int version;
  status = GpuDriver::GetGpuISAVersion(&version, device);
  if (!status.ok()) {
    return status;
  }

  std::string gcn_arch_name;
  status = GpuDriver::GetGpuGCNArchName(device, &gcn_arch_name);
  if (!status.ok()) {
    return status;
  }

  DeviceDescription desc;

  {
    std::string pci_bus_id = GpuDriver::GetPCIBusID(device);

    // Lower the hex characters to match sysfs.
    pci_bus_id = absl::AsciiStrToLower(pci_bus_id);
    desc.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = ReadNumaNode(pci_bus_id, device_ordinal);
    desc.set_numa_node(numa_node);
  }

  hipDeviceProp_t prop;
  if (GpuDriver::GetDeviceProperties(&prop, device_ordinal)) {
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

  {
    bool ecc_enabled = false;
    (void)GpuDriver::IsEccEnabled(device, &ecc_enabled);
    desc.set_ecc_enabled(ecc_enabled);
  }

  uint64_t device_memory_size = -1;
  (void)RocmContext::GetDeviceTotalMemory(device, &device_memory_size);
  desc.set_device_memory_size(device_memory_size);

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
      absl::StrCat("AMDGPU ISA version: ", gcn_arch_name));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  desc.set_device_address_bits(64);

  desc.set_device_vendor("Advanced Micro Devices, Inc");
  desc.set_rocm_compute_capability(gcn_arch_name);

  desc.set_shared_memory_per_core(
      GpuDriver::GetMaxSharedMemoryPerCore(device).value());
  desc.set_shared_memory_per_block(
      GpuDriver::GetMaxSharedMemoryPerBlock(device).value());
  int core_count = GpuDriver::GetMultiprocessorCount(device).value();
  desc.set_core_count(core_count);
  desc.set_fpus_per_core(fpus_per_core(gcn_arch_name));
  desc.set_threads_per_core_limit(
      GpuDriver::GetMaxThreadsPerMultiprocessor(device).value());
  desc.set_registers_per_block_limit(
      GpuDriver::GetMaxRegistersPerBlock(device).value());
  desc.set_threads_per_warp(GpuDriver::GetThreadsPerWarp(device).value());
  desc.set_registers_per_core_limit(64 * 1024);
  desc.set_compile_time_toolkit_version(
      SemanticVersion{HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH});
  desc.set_runtime_version(
      ParseRocmVersion(RocmRuntime::GetRuntimeVersion().value_or(0))
          .value_or(SemanticVersion{0, 0, 0}));
  desc.set_driver_version(
      ParseRocmVersion(GpuDriver::GetDriverVersion().value_or(0))
          .value_or(SemanticVersion{0, 0, 0}));

  int cc_major = 0;
  int cc_minor = 0;
  GpuDriver::GetComputeCapability(&cc_major, &cc_minor, device).IgnoreError();

  // It would be better to use the PCI device ID or some other truly unique
  // identifier for the GPU model.  But getting this requires using NVML or
  // other hacks, which we don't have access to in OSS TensorFlow.
  //
  // Alternatively you might be tempted to use GpuDriver::GetDeviceName as a
  // unique identifier, but this is not stable across GPU VBIOS versions.
  //
  // TODO(jlebar): This really should be more unique.  In CUDA land, we mix in
  // the clock speed and L2 cache size.
  desc.set_model_str(absl::StrFormat("cc_%d.%d with %dB RAM, %d cores",
                                     cc_major, cc_minor, device_memory_size,
                                     core_count));

  return std::make_unique<DeviceDescription>(std::move(desc));
}

}  // namespace gpu

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(rocm_executor, {});
