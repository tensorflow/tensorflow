/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <unistd.h>

#include "absl/base/casts.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/gpu/gpu_event.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/mathutil.h"
#include "tensorflow/stream_executor/lib/numbers.h"
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/rocm/rocm_diagnostics.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"

#ifdef PLATFORMS_GPUS_ROCM_DYNAMIC_LIBROCM_DYNAMIC_LIBROCM_H_
#error \
    "No driver calls in this file, wrap driver functionality in rocm_driver.cc."
#endif

#ifdef __ROCM_RUNTIME_H__
#error \
    "ROCM runtime being included into ROCM GPU executor; should be driver only."
#endif

namespace stream_executor {
namespace gpu {

static GpuEvent* AsGpuEvent(Event* event) {
  DCHECK(event != nullptr);
  return static_cast<GpuEvent*>(event->implementation());
}

// Given a platform-independent timer datatype, returns the internal ROCM
// platform implementation pointer.
static GpuTimer* AsGpuTimer(Timer* timer) {
  DCHECK(timer != nullptr);
  return static_cast<GpuTimer*>(timer->implementation());
}

// Given const GPU memory, returns a librocm device pointer datatype, suitable
// for passing directly to librocm APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// librocm APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to librocm functions which will honor constness.
static hipDeviceptr_t AsROCmDevicePtr(const DeviceMemoryBase& gpu_mem) {
  return const_cast<hipDeviceptr_t>(gpu_mem.opaque());
}

// See description on const version above.
static hipDeviceptr_t AsROCmDevicePtr(DeviceMemoryBase* gpu_mem) {
  return AsROCmDevicePtr(*gpu_mem);
}

static GpuContext* GetGpuContext(Stream* stream) {
  return static_cast<GpuExecutor*>(stream->parent()->implementation())
      ->gpu_context();
}

GpuContext* ExtractGpuContext(GpuExecutor* rocm_exec) {
  CHECK(rocm_exec != nullptr);
  return rocm_exec->gpu_context();
}

GpuExecutor* ExtractGpuExecutor(StreamExecutor* stream_exec) {
  return static_cast<GpuExecutor*>(stream_exec->implementation());
}

GpuExecutor::~GpuExecutor() {
  for (auto& it : disk_modules_) {
    GpuDriver::UnloadModule(context_, it.second);
  }
  for (auto& it : in_memory_modules_) {
    GpuDriver::UnloadModule(context_, it.second);
  }
  if (context_ != nullptr) {
    GpuDriver::DestroyContext(context_);
  }
  CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
}
bool GpuExecutor::UnloadModule(ModuleHandle module_handle) {
  const char* gpu_binary = reinterpret_cast<const char*>(module_handle.id());
  mutex_lock lock{in_memory_modules_mu_};
  return UnloadGpuBinary(gpu_binary);
}

bool GpuExecutor::UnloadGpuBinary(const void* gpu_binary) {
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
    GpuDriver::UnloadModule(context_, module);
    gpu_binary_to_module_.erase(module_it);
  }
  return true;
}

void GpuExecutor::UnloadKernel(const KernelBase* kernel) {
  LOG(FATAL) << "Feature not supported on ROCM platform (UnloadKernel)";
}

port::Status GpuExecutor::Init(int device_ordinal,
                               DeviceOptions device_options) {
  device_ordinal_ = device_ordinal;

  auto status = GpuDriver::Init();
  if (!status.ok()) {
    return status;
  }

  status = GpuDriver::GetDevice(device_ordinal_, &device_);
  if (!status.ok()) {
    return status;
  }

  status = GpuDriver::CreateContext(device_ordinal_, device_, device_options,
                                    &context_);
  if (!status.ok()) {
    return status;
  }

  return GpuDriver::GetGpuISAVersion(&version_, device_);
}

bool GpuExecutor::FindOnDiskForComputeCapability(
    absl::string_view filename, absl::string_view canonical_suffix,
    string* found_filename) const {
  LOG(FATAL) << "Feature not supported on ROCM platform "
                "(FindOnDiskForComputeCapability)";
  return false;
}

bool GpuExecutor::FindOnDiskForISAVersion(absl::string_view filename,
                                          absl::string_view canonical_suffix,
                                          string* found_filename) const {
  if (version_ == 0) {
    return false;
  }

  string cc_specific =
      absl::StrCat(filename, ".cc", version_, canonical_suffix);
  if (port::FileExists(cc_specific).ok()) {
    VLOG(2) << "found AMDGPU ISA version-specific file, using that: "
            << cc_specific;
    *found_filename = cc_specific;
    return true;
  }

  VLOG(2) << "could not find AMDGPU ISA version-specific file at: "
          << cc_specific;
  if (port::FileExists(string(filename)).ok()) {
    *found_filename = string(filename);
    return true;
  }

  return false;
}

// Returns the path to the running executable.
// N.B. Derived from //knowledge/smalltalk/background_kb.cc
// Arg: strip_exe: if true, remove the name of the executable itself from the
//                 returned string. Example: calling this from /usr/bin/foo
//                 would return /usr/bin.
static string GetBinaryDir(bool strip_exe) {
  char exe_path[PATH_MAX] = {0};
  CHECK_ERR(readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1));
  // Make sure it's null-terminated:
  exe_path[sizeof(exe_path) - 1] = 0;

  if (strip_exe) {
    // The exe is the last component of the path, so remove one component.
    string ret = exe_path;
    std::vector<string> components = port::Split(exe_path, '/');
    components.pop_back();
    return port::Join(components, "/");
  }
  return exe_path;
}

bool GpuExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                            KernelBase* kernel) {
  GpuKernel* rocm_kernel = AsGpuKernel(kernel);
  hipModule_t module = nullptr;
  const string* kernelname;

  const OnDiskKernelLoaderSpec* on_disk_spec = nullptr;
  bool has_cubin = spec.has_cuda_cubin_on_disk();
  if (has_cubin) {
    on_disk_spec = &spec.cuda_cubin_on_disk();
  }

  if (on_disk_spec != nullptr) {
    LOG(WARNING) << "loading ROCM kernel from disk is not supported";
    return false;
  } else if (spec.has_cuda_cubin_in_memory()) {
    kernelname = &spec.cuda_cubin_in_memory().kernelname();

    const char* hsaco = spec.cuda_cubin_in_memory().bytes();
    mutex_lock lock{in_memory_modules_mu_};
    module = in_memory_modules_[hsaco];

    if (module == nullptr) {
      if (!GpuDriver::LoadHsaco(context_, hsaco, &module)) {
        LOG(ERROR) << "failed to load HSACO\n";
        return false;
      }
      in_memory_modules_[hsaco] = module;
    }
  } else {
    LOG(WARNING) << "no method of loading ROCM kernel provided";
    return false;
  }

  VLOG(2) << "getting function " << *kernelname << " from module " << module;
  if (!GpuDriver::GetModuleFunction(context_, module, kernelname->c_str(),
                                    rocm_kernel->gpu_function_ptr())) {
    return false;
  }

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the ROCM API.
  rocm_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
  if (!GetKernelMetadata(rocm_kernel, &kernel_metadata)) {
    LOG(WARNING) << "Unable to get metadata for kernel " << kernelname;
  }
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(*kernelname);
  return true;
}

bool GpuExecutor::GetKernelMetadata(GpuKernel* rocm_kernel,
                                    KernelMetadata* kernel_metadata) {
  int value = 0;
  // TODO(ROCm) implement this feature in HIP
  kernel_metadata->set_registers_per_thread(value);

  // TODO(ROCm) implement this feature in HIP
  kernel_metadata->set_shared_memory_bytes(value);

  return true;
}

bool GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                         const BlockDim& block_dims, const KernelBase& kernel,
                         const KernelArgsArrayBase& args) {
  CHECK_EQ(kernel.Arity(), args.number_of_arguments());
  GpuStreamHandle hipstream = AsGpuStreamValue(stream);
  const GpuKernel* rocm_kernel = AsGpuKernel(&kernel);
  hipFunction_t hipfunc = rocm_kernel->AsGpuFunctionHandle();

  // Only perform/print the occupancy check once.  Even just checking to see
  // whether we've done an occupancy check on this kernel before isn't free
  // (because we have to synchronize), so we only do this at -v 2+.
  if (VLOG_IS_ON(2)) {
    mutex_lock lock(launched_kernels_mu_);
    if (!launched_kernels_.count(hipfunc)) {
      VlogOccupancyInfo(kernel, thread_dims, block_dims);
      // TODO(rspringer): Remove elements from launched_kernels_...if we ever
      // expose a kernel/module deallocation method.
      launched_kernels_.insert(hipfunc);
    }
  }

  if (rocm_kernel->GetPreferredCacheConfig() !=
      KernelCacheConfig::kNoPreference) {
    GpuDriver::FuncSetCacheConfig(hipfunc, rocm_kernel->GetGpuCacheConfig());
  }

  // prepare kernargs
  // KernelArgsArrayBase keeps the pointer of arguments
  // deference them here
  std::vector<void*> kernargs;
  KernelArgIterator iter = args.arg_iterator();
  while (iter.has_next()) {
    KernelArg arg = iter.next();
    VLOG(2) << "*(arg.address): "
            << reinterpret_cast<void*>(
                   *static_cast<const uint64_t*>(arg.address));
    kernargs.push_back(
        reinterpret_cast<void*>(*static_cast<const uint64_t*>(arg.address)));
  }

  size_t size = sizeof(void*) * kernargs.size();
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  if (!GpuDriver::LaunchKernel(
          GetGpuContext(stream), hipfunc, block_dims.x, block_dims.y,
          block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
          args.number_of_shared_bytes(), hipstream, nullptr, (void**)&config)) {
    LOG(ERROR) << "failed to launch ROCM kernel with args: "
               << args.number_of_arguments()
               << "; thread dim: " << thread_dims.ToString()
               << "; block dim: " << block_dims.ToString();
    return false;
  }

  return true;
}

int GpuExecutor::CalculateOccupancy(const DeviceDescription& device_description,
                                    uint64 registers_per_thread,
                                    uint64 shared_memory_per_block,
                                    const ThreadDim& thread_dims,
                                    GpuFunctionHandle func) {
  LOG(FATAL) << "Feature not supported on ROCM platform (CalculateOccupancy)";
  return 0;
}

int GpuExecutor::CompareOccupancy(int* initial_blocks,
                                  const DeviceDescription& device_description,
                                  uint64 registers_per_thread,
                                  uint64 shared_memory_per_block,
                                  const ThreadDim& thread_dims,
                                  GpuFunctionHandle func) {
  LOG(FATAL) << "Feature not supported on ROCM platform (CompareOccupancy)";
  return 0;
}

bool GpuExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                             ModuleHandle* module_handle) {
  // In GpuExecutor we store the pointer to the  HSACO binary  as
  // ModuleHandle::id().
  hipModule_t hip_module = nullptr;
  // TODO(ROCm): Need  generic term instead of cubin/cuda/ptx
  if (spec.has_cuda_cubin_in_memory()) {
    mutex_lock lock{in_memory_modules_mu_};
    if (!LoadModuleFromHsaco(
            reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()),
            &hip_module)) {
      return false;
    }
    *module_handle = ModuleHandle(const_cast<void*>(
        static_cast<const void*>(spec.cuda_cubin_in_memory().data())));
    return true;
  } else {
    LOG(ERROR) << "No HSACO binary found \n";
    return false;
  }
}

bool GpuExecutor::LoadModuleFromCuBin(const char* cubin, hipModule_t* module) {
  LOG(FATAL) << "Feature not supported on ROCM platform (LoadModuleFromCuBin)";
  return false;
}

bool GpuExecutor::LoadModuleFromPtx(const char* ptx, hipModule_t* module) {
  LOG(FATAL) << "Feature not supported on ROCM platform (LoadModuleFromPtx)";
  return false;
}

bool GpuExecutor::LoadModuleFromHsaco(const char* hsaco, hipModule_t* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[hsaco];

  if (*module == nullptr) {
    if (!GpuDriver::LoadHsaco(context_, hsaco, module)) {
      LOG(ERROR) << "failed to load : HSACO \n";
      return false;
    }
    module_refcount = 1;
    VLOG(3) << "Loaded HSACO " << static_cast<const void*>(hsaco)
            << " as module " << *module;
  } else {
    ++module_refcount;
    VLOG(3) << "HSACO " << static_cast<const void*>(hsaco)
            << " is already loaded as module " << *module;
  }
  gpu_binary_to_module_[hsaco] = {*module, module_refcount};
  return true;
}

// This is a non-essential operation; if there's a failure, proceed without
// logging an error. It's nearly certain that in case of failures, we'd never
// get here in the first place; these are very low-impact routines.
void GpuExecutor::VlogOccupancyInfo(const KernelBase& kernel,
                                    const ThreadDim& thread_dims,
                                    const BlockDim& block_dims) {
  // TODO(ROCm) implement this feature in HIP
}

void* GpuExecutor::Allocate(uint64 size) {
  return GpuDriver::DeviceAllocate(context_, size);
}

void* GpuExecutor::GetSubBuffer(DeviceMemoryBase* mem, uint64 offset_bytes,
                                uint64 size_bytes) {
  // offset and size are in bytes, so char* works as the pointer type.
  return reinterpret_cast<char*>(mem->opaque()) + offset_bytes;
}

void GpuExecutor::Deallocate(DeviceMemoryBase* mem) {
  // ROCM "sub-buffers" are just pointer + offset, so no dealloc is necessary.
  if (!mem->is_sub_buffer()) {
    GpuDriver::DeviceDeallocate(context_, mem->opaque());
  }
}

bool GpuExecutor::HostMemoryRegister(void* location, uint64 size) {
  if (location == nullptr || size == 0) {
    LOG(WARNING) << "attempting to register null or zero-sized memory: "
                 << location << "; size " << size;
  }
  VLOG(2) << "registering " << location << " size " << size;
  return GpuDriver::HostRegister(context_, location, size);
}

bool GpuExecutor::HostMemoryUnregister(void* location) {
  VLOG(2) << "unregistering " << location;
  return GpuDriver::HostUnregister(context_, location);
}

bool GpuExecutor::SynchronizeAllActivity() {
  return GpuDriver::SynchronizeContext(context_);
}

bool GpuExecutor::SynchronousMemZero(DeviceMemoryBase* location, uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsROCmDevicePtr(location), 0x0, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsROCmDevicePtr(location),
                                           0x0, size);
}

bool GpuExecutor::SynchronousMemSet(DeviceMemoryBase* location, int value,
                                    uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    // hipMemset reinterprets "value" as a uint8.
    uint8 byte_value = static_cast<uint8>(value);
    uint32 pattern = (byte_value << 24) | (byte_value << 16) |
                     (byte_value << 8) | byte_value;
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsROCmDevicePtr(location), pattern, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsROCmDevicePtr(location),
                                           value, size);
}

port::Status GpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src, uint64 size) {
  return GpuDriver::SynchronousMemcpyH2D(context_, AsROCmDevicePtr(gpu_dst),
                                         host_src, size);
}

port::Status GpuExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64 size) {
  return GpuDriver::SynchronousMemcpyD2H(context_, host_dst,
                                         AsROCmDevicePtr(gpu_src), size);
}

port::Status GpuExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64 size) {
  return GpuDriver::SynchronousMemcpyD2D(context_, AsROCmDevicePtr(gpu_dst),
                                         AsROCmDevicePtr(gpu_src), size);
}

bool GpuExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                          uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return Memset32(stream, location, 0x0, size);
  } else {
    return Memset(stream, location, 0x0, size);
  }
}

bool GpuExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                         uint8 pattern, uint64 size) {
  VLOG(2) << "enqueueing memset8 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  return GpuDriver::AsynchronousMemsetUint8(context_, AsROCmDevicePtr(location),
                                            pattern, size,
                                            AsGpuStreamValue(stream));
}

bool GpuExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                           uint32 pattern, uint64 size) {
  VLOG(2) << "enqueueing memset32 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
        size % 4 == 0);
  return GpuDriver::AsynchronousMemsetUint32(
      context_, AsROCmDevicePtr(location), pattern, size / 4,
      AsGpuStreamValue(stream));
}

bool GpuExecutor::Memcpy(Stream* stream, void* host_dst,
                         const DeviceMemoryBase& gpu_src, uint64 size) {
  return GpuDriver::AsynchronousMemcpyD2H(context_, host_dst,
                                          AsROCmDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                         const void* host_src, uint64 size) {
  return GpuDriver::AsynchronousMemcpyH2D(context_, AsROCmDevicePtr(gpu_dst),
                                          host_src, size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::MemcpyDeviceToDevice(Stream* stream,
                                       DeviceMemoryBase* gpu_dst,
                                       const DeviceMemoryBase& gpu_src,
                                       uint64 size) {
  return GpuDriver::AsynchronousMemcpyD2D(context_, AsROCmDevicePtr(gpu_dst),
                                          AsROCmDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::HostCallback(Stream* stream,
                               std::function<port::Status()> callback) {
  auto callback_ptr = new std::function<void()>([callback]() {
    port::Status s = callback();
    if (!s.ok()) {
      LOG(WARNING) << "Host callback failed: " << s;
    }
  });
  return GpuDriver::AddStreamCallback(context_, AsGpuStreamValue(stream),
                                      InternalHostCallback, callback_ptr);
}

/* static */ void GpuExecutor::InternalHostCallback(GpuStreamHandle stream,
                                                    hipError_t status,
                                                    void* data) {
  std::function<void()>* callback =
      reinterpret_cast<std::function<void()>*>(data);
  (*callback)();
  delete callback;
}

port::Status GpuExecutor::AllocateEvent(Event* event) {
  return AsGpuEvent(event)->Init();
}

port::Status GpuExecutor::DeallocateEvent(Event* event) {
  return AsGpuEvent(event)->Destroy();
}

port::Status GpuExecutor::RecordEvent(Stream* stream, Event* event) {
  return AsGpuEvent(event)->Record(AsGpuStream(stream));
}

port::Status GpuExecutor::WaitForEvent(Stream* stream, Event* event) {
  if (GpuDriver::WaitStreamOnEvent(context_, AsGpuStream(stream)->gpu_stream(),
                                   AsGpuEvent(event)->gpu_event())) {
    return port::Status::OK();
  } else {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat("error recording waiting for ROCM event on stream %p",
                        stream)};
  }
}

Event::Status GpuExecutor::PollForEventStatus(Event* event) {
  return AsGpuEvent(event)->PollForStatus();
}

bool GpuExecutor::AllocateStream(Stream* stream) {
  return AsGpuStream(stream)->Init();
}

void GpuExecutor::DeallocateStream(Stream* stream) {
  GpuStream* rocm_stream = AsGpuStream(stream);
  if (!rocm_stream->IsIdle()) {
    LOG(ERROR) << "Deallocating stream with pending work";
  }
  rocm_stream->Destroy();
}

bool GpuExecutor::AllocateTimer(Timer* timer) {
  return AsGpuTimer(timer)->Init();
}

void GpuExecutor::DeallocateTimer(Timer* timer) {
  AsGpuTimer(timer)->Destroy();
}

bool GpuExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  GpuEventHandle other_completed_event = *AsGpuStream(other)->completed_event();
  bool ok = GpuDriver::RecordEvent(context_, other_completed_event,
                                   AsGpuStreamValue(other))
                .ok();
  if (!ok) {
    LOG(ERROR) << "failed to record completion event; "
                  "therefore, failed to create inter-stream dependency";
    return false;
  }

  return GpuDriver::WaitStreamOnEvent(context_, AsGpuStreamValue(dependent),
                                      other_completed_event);
}

bool GpuExecutor::StartTimer(Stream* stream, Timer* timer) {
  return AsGpuTimer(timer)->Start(AsGpuStream(stream));
}

bool GpuExecutor::StopTimer(Stream* stream, Timer* timer) {
  return AsGpuTimer(timer)->Stop(AsGpuStream(stream));
}

port::Status GpuExecutor::BlockHostUntilDone(Stream* stream) {
  return GpuDriver::SynchronizeStream(context_, AsGpuStreamValue(stream));
}

blas::BlasSupport* GpuExecutor::CreateBlas() {
  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(rocm::kROCmPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

dnn::DnnSupport* GpuExecutor::CreateDnn() {
  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(rocm::kROCmPlatformId,
                                                       plugin_config_.dnn());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

fft::FftSupport* GpuExecutor::CreateFft() {
  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(rocm::kROCmPlatformId,
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

rng::RngSupport* GpuExecutor::CreateRng() {
  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(rocm::kROCmPlatformId,
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

// TODO(rspringer): Remove in b/18544742.
bool GpuExecutor::SupportsDnn() const { return true; }

bool GpuExecutor::CanEnablePeerAccessTo(StreamExecutorInterface* other) {
  GpuExecutor* rocm_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::CanEnablePeerAccess(context_, rocm_other->context_);
}

port::Status GpuExecutor::EnablePeerAccessTo(StreamExecutorInterface* other) {
  GpuExecutor* rocm_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::EnablePeerAccess(context_, rocm_other->context_);
}

SharedMemoryConfig GpuExecutor::GetDeviceSharedMemoryConfig() {
  port::StatusOr<hipSharedMemConfig> rocm_config =
      GpuDriver::ContextGetSharedMemConfig(context_);
  if (!rocm_config.ok()) {
    // Don't log; the failed call will log necessary output.
    return SharedMemoryConfig::kDefault;
  }

  switch (rocm_config.ValueOrDie()) {
    case hipSharedMemBankSizeDefault:
      return SharedMemoryConfig::kDefault;
    case hipSharedMemBankSizeFourByte:
      return SharedMemoryConfig::kFourByte;
    case hipSharedMemBankSizeEightByte:
      return SharedMemoryConfig::kEightByte;
    default:
      LOG(FATAL) << "Invalid shared memory configuration returned: "
                 << rocm_config.ValueOrDie();
  }
}

port::Status GpuExecutor::SetDeviceSharedMemoryConfig(
    SharedMemoryConfig config) {
  hipSharedMemConfig rocm_config;
  switch (config) {
    case SharedMemoryConfig::kDefault:
      rocm_config = hipSharedMemBankSizeDefault;
      break;
    case SharedMemoryConfig::kFourByte:
      rocm_config = hipSharedMemBankSizeFourByte;
      break;
    case SharedMemoryConfig::kEightByte:
      rocm_config = hipSharedMemBankSizeEightByte;
      break;
    default:
      LOG(FATAL) << "Invalid shared memory configuration specified: "
                 << static_cast<int>(config);
  }
  return GpuDriver::ContextSetSharedMemConfig(context_, rocm_config);
}

bool GpuExecutor::DeviceMemoryUsage(int64* free, int64* total) const {
  return GpuDriver::GetDeviceMemoryInfo(context_, free, total);
}

bool GpuExecutor::GetSymbol(const string& symbol_name,
                            ModuleHandle module_handle, void** mem,
                            size_t* bytes) {
  {  // give limited scope to mutex_lock
    mutex_lock lock{disk_modules_mu_};
    for (auto& it : disk_modules_) {
      if (GpuDriver::GetModuleSymbol(context_, it.second, symbol_name.c_str(),
                                     reinterpret_cast<hipDeviceptr_t*>(mem),
                                     bytes)) {
        return true;
      }
    }
  }

  {  // give limited scope to mutex_lock
    mutex_lock lock{in_memory_modules_mu_};
    for (auto& it : in_memory_modules_) {
      if (GpuDriver::GetModuleSymbol(context_, it.second, symbol_name.c_str(),
                                     reinterpret_cast<hipDeviceptr_t*>(mem),
                                     bytes)) {
        return true;
      }
    }
  }

  {  // give limited scope to mutex_lock
    mutex_lock lock{in_memory_modules_mu_};
    if (static_cast<bool>(module_handle)) {
      auto it = gpu_binary_to_module_.find(module_handle.id());
      CHECK(it != gpu_binary_to_module_.end());
      if (GpuDriver::GetModuleSymbol(
              context_, it->second.first, symbol_name.c_str(),
              reinterpret_cast<hipDeviceptr_t*>(mem), bytes)) {
        return true;
      }
    }

    for (auto& it : gpu_binary_to_module_) {
      if (GpuDriver::GetModuleSymbol(
              context_, it.second.first, symbol_name.c_str(),
              reinterpret_cast<hipDeviceptr_t*>(mem), bytes)) {
        return true;
      }
    }
  }

  LOG(INFO) << "Falied to find symbol in any modules: " << symbol_name;
  return false;
}

bool GpuExecutor::FillBlockDimLimit(BlockDim* block_dim_limit) const {
  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
  if (!GpuDriver::GetGridLimits(&x, &y, &z, device_)) {
    return false;
  }

  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return true;
}

bool GpuExecutor::SupportsBlas() const { return true; }

bool GpuExecutor::SupportsFft() const { return true; }

bool GpuExecutor::SupportsRng() const { return true; }

std::unique_ptr<internal::EventInterface>
GpuExecutor::CreateEventImplementation() {
  return std::unique_ptr<internal::EventInterface>(new GpuEvent(this));
}

std::unique_ptr<internal::KernelInterface>
GpuExecutor::CreateKernelImplementation() {
  return std::unique_ptr<internal::KernelInterface>(new GpuKernel());
}

std::unique_ptr<internal::StreamInterface>
GpuExecutor::GetStreamImplementation() {
  return std::unique_ptr<internal::StreamInterface>(new GpuStream(this));
}

std::unique_ptr<internal::TimerInterface>
GpuExecutor::GetTimerImplementation() {
  return std::unique_ptr<internal::TimerInterface>(new GpuTimer(this));
}

void* GpuExecutor::GpuContextHack() { return context_; }

GpuContext* GpuExecutor::gpu_context() { return context_; }

// Attempts to read the NUMA node corresponding to the GPU device's PCI bus out
// of SysFS. Returns -1 if it cannot.
//
// For anything more complicated/prod-focused than this, you'll likely want to
// turn to gsys' topology modeling.
static int TryToReadNumaNode(const string& pci_bus_id, int device_ordinal) {
  // TODO(ROCm) implement this feature in HIP
  return 1;
}

DeviceDescription* GpuExecutor::PopulateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  {
    int driver_version = 0;
    (void)GpuDriver::GetDriverVersion(&driver_version);
    string augmented_driver_version = absl::StrFormat(
        "%d (%s)", driver_version,
        rocm::DriverVersionStatusToString(Diagnostician::FindDsoVersion())
            .c_str());
    builder.set_driver_version(augmented_driver_version);
  }

  {
    string pci_bus_id = GpuDriver::GetPCIBusID(device_);

    // Lower the hex characters to match sysfs.
    pci_bus_id = port::Lowercase(pci_bus_id);
    builder.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = TryToReadNumaNode(pci_bus_id, device_ordinal_);
    builder.set_numa_node(numa_node);
  }

  hipDeviceProp_t prop;
  if (GpuDriver::GetDeviceProperties(&prop, device_ordinal_)) {
    builder.set_threads_per_block_limit(prop.maxThreadsPerBlock);

    ThreadDim thread_dim_limit;
    thread_dim_limit.x = prop.maxThreadsDim[0];
    thread_dim_limit.y = prop.maxThreadsDim[1];
    thread_dim_limit.z = prop.maxThreadsDim[2];
    builder.set_thread_dim_limit(thread_dim_limit);

    float clock_rate_ghz = static_cast<float>(prop.clockRate) / 1e6;
    builder.set_clock_rate_ghz(clock_rate_ghz);
  }

  {
    bool ecc_enabled = false;
    (void)GpuDriver::IsEccEnabled(device_, &ecc_enabled);
    builder.set_ecc_enabled(ecc_enabled);
  }

  {
    uint64 device_memory_size = -1;
    (void)GpuDriver::GetDeviceTotalMemory(device_, &device_memory_size);
    builder.set_device_memory_size(device_memory_size);
  }

  {
    BlockDim block_dim_limit;
    FillBlockDimLimit(&block_dim_limit);
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    string device_name;
    (void)GpuDriver::GetDeviceName(device_, &device_name);
    builder.set_name(device_name);
  }

  builder.set_platform_version(
      absl::StrCat("AMDGPU ISA version: gfx", version_));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  builder.set_device_address_bits(64);

  builder.set_device_vendor("Advanced Micro Devices, Inc");
  builder.set_rocm_amdgpu_isa_version(version_);
  builder.set_shared_memory_per_core(
      GpuDriver::GetMaxSharedMemoryPerCore(device_).ValueOrDie());
  builder.set_shared_memory_per_block(
      GpuDriver::GetMaxSharedMemoryPerBlock(device_).ValueOrDie());
  builder.set_core_count(
      GpuDriver::GetMultiprocessorCount(device_).ValueOrDie());
  builder.set_threads_per_core_limit(
      GpuDriver::GetMaxThreadsPerMultiprocessor(device_).ValueOrDie());
  builder.set_registers_per_block_limit(
      GpuDriver::GetMaxRegistersPerBlock(device_).ValueOrDie());
  builder.set_threads_per_warp(
      GpuDriver::GetThreadsPerWarp(device_).ValueOrDie());
  builder.set_registers_per_core_limit(64 * 1024);

  auto built = builder.Build();
  return built.release();
}

}  // namespace gpu

void initialize_rocm_gpu_executor() {
  *internal::MakeROCMExecutorImplementation() = [](const PluginConfig& config) {
    return new gpu::GpuExecutor{config};
  };
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(rocm_gpu_executor, {
  stream_executor::initialize_rocm_gpu_executor();
});
