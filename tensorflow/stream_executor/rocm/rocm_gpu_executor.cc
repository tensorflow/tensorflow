/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/rocm/rocm_gpu_executor.h"

#include <unistd.h>
#include "tensorflow/stream_executor/rocm/rocm_diagnostics.h"
#include "tensorflow/stream_executor/rocm/rocm_driver.h"
#include "tensorflow/stream_executor/rocm/rocm_event.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"
#include "tensorflow/stream_executor/rocm/rocm_timer.h"
#include "tensorflow/stream_executor/dso_loader.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/lib/casts.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/mathutil.h"
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"
#include "tensorflow/stream_executor/lib/numbers.h"

#ifdef PLATFORMS_GPUS_ROCM_DYNAMIC_LIBROCM_DYNAMIC_LIBROCM_H_
#error \
    "No driver calls in this file, wrap driver functionality in rocm_driver.cc."
#endif

#ifdef __ROCM_RUNTIME_H__
#error \
    "ROCM runtime being included into ROCM GPU executor; should be driver only."
#endif

namespace stream_executor {
namespace rocm {

static ROCMEvent *AsROCMEvent(Event *event) {
  DCHECK(event != nullptr);
  return static_cast<ROCMEvent *>(event->implementation());
}


// Given a platform-independent timer datatype, returns the internal ROCM
// platform implementation pointer.
static ROCMTimer *AsROCMTimer(Timer *timer) {
  DCHECK(timer != nullptr);
  return static_cast<ROCMTimer *>(timer->implementation());
}

// Given const GPU memory, returns a librocm device pointer datatype, suitable
// for passing directly to librocm APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// librocm APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to librocm functions which will honor constness.
static hipDeviceptr_t AsROCmDevicePtr(const DeviceMemoryBase &gpu_mem) {
  return const_cast<hipDeviceptr_t>(gpu_mem.opaque());
}

// See description on const version above.
static hipDeviceptr_t AsROCmDevicePtr(DeviceMemoryBase *gpu_mem) {
  return AsROCmDevicePtr(*gpu_mem);
}

static int GetROCmDeviceOrdinal(Stream *stream) {
  return static_cast<ROCMExecutor *>(stream->parent()->implementation())
    ->device_ordinal();
}

int ExtractROCmDeviceOrdinal(ROCMExecutor *rocm_exec) {
  CHECK(rocm_exec != nullptr);
  return rocm_exec->device_ordinal();
}

ROCMExecutor *ExtractROCmExecutor(StreamExecutor *stream_exec) {
  return static_cast<ROCMExecutor *>(stream_exec->implementation());
}

ROCMExecutor::~ROCMExecutor() {
  for (auto &it : disk_modules_) {
    ROCMDriver::UnloadModule(device_ordinal_, it.second);
  }
  for (auto &it : in_memory_modules_) {
    ROCMDriver::UnloadModule(device_ordinal_, it.second);
  }
}

port::Status ROCMExecutor::Init(int device_ordinal,
                                DeviceOptions device_options) {
  device_ordinal_ = device_ordinal;

  auto status = ROCMDriver::Init();
  if (!status.ok()) {
    return status;
  }

  status = ROCMDriver::GetDevice(device_ordinal_, &device_);
  if (!status.ok()) {
    return status;
  }

  return ROCMDriver::GetAMDGPUISAVersion(&version_, device_);
}

bool ROCMExecutor::FindOnDiskForISAVersion(
    port::StringPiece filename, port::StringPiece canonical_suffix,
    string *found_filename) const {
  if (version_ == 0) {
    return false;
  }

  string cc_specific =
      port::StrCat(filename, ".cc", version_, canonical_suffix);
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

bool ROCMExecutor::GetKernel(const MultiKernelLoaderSpec &spec,
                             KernelBase *kernel) {
  ROCMKernel *rocm_kernel = AsROCMKernel(kernel);
  hipModule_t module = nullptr;
  const string *kernelname;

  const OnDiskKernelLoaderSpec *on_disk_spec = nullptr;
  bool has_ptx = spec.has_cuda_ptx_on_disk();
  if (has_ptx) {
    on_disk_spec = &spec.cuda_ptx_on_disk();
  }

  if (on_disk_spec != nullptr) {
    LOG(WARNING) << "loading ROCM kernel from disk is not supported";
    return false;
  } else if (spec.has_cuda_ptx_in_memory()) {
    kernelname = &spec.cuda_ptx_in_memory().kernelname();

    const char *hsaco = spec.cuda_ptx_in_memory().original_default_text();
    mutex_lock lock{in_memory_modules_mu_};
    module = in_memory_modules_[hsaco];

    if (module == nullptr) {
      if (!ROCMDriver::LoadHsaco(device_ordinal_, hsaco, &module)) {
        LOG(ERROR) << "failed to load HSACO\n";
        return false;
      }
      in_memory_modules_[hsaco] = module;
    }
  } else {
    LOG(WARNING) << "no method of loading ROCM kernel provided";
    return false;
  }

  VLOG(2) << "getting function " << kernelname << " from module " << module;
  if (!ROCMDriver::GetModuleFunction(device_ordinal_, module, kernelname->c_str(),
                                     rocm_kernel->rocm_function_ptr())) {
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

bool ROCMExecutor::GetKernelMetadata(ROCMKernel *rocm_kernel,
                                     KernelMetadata *kernel_metadata) {
  int value = 0;
  // ROCM TODO implement this feature in HIP
  kernel_metadata->set_registers_per_thread(value);

  // ROCM TODO implement this feature in HIP
  kernel_metadata->set_shared_memory_bytes(value);

  return true;
}

bool ROCMExecutor::Launch(Stream *stream, const ThreadDim &thread_dims,
                          const BlockDim &block_dims, const KernelBase &kernel,
                          const KernelArgsArrayBase &args) {
  CHECK_EQ(kernel.Arity(), args.number_of_arguments());
  hipStream_t hipstream = AsROCMStreamValue(stream);
  const ROCMKernel *rocm_kernel = AsROCMKernel(&kernel);
  hipFunction_t hipfunc = rocm_kernel->AsROCMFunctionValue();

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
    ROCMDriver::FuncSetCacheConfig(hipfunc, rocm_kernel->GetROCMCacheConfig());
  }

  // prepare kernargs
  // KernelArgsArrayBase keeps the pointer of arguments
  // deference them here
  std::vector<void*> kernargs;
  KernelArgIterator iter = args.arg_iterator();
  while (iter.has_next()) {
    KernelArg arg = iter.next();
    VLOG(2) << "*(arg.address): " << reinterpret_cast<void*>(*static_cast<const uint64_t*>(arg.address));
    kernargs.push_back(reinterpret_cast<void*>(*static_cast<const uint64_t*>(arg.address)));
  }

  size_t size = sizeof(void*) * kernargs.size();
  void *config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs.data(),
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
    HIP_LAUNCH_PARAM_END
  };

  if (!ROCMDriver::LaunchKernel(GetROCmDeviceOrdinal(stream), hipfunc, block_dims.x,
                                block_dims.y, block_dims.z, thread_dims.x,
                                thread_dims.y, thread_dims.z,
                                args.number_of_shared_bytes(), hipstream,
                                nullptr, (void**)&config)) {
    LOG(ERROR) << "failed to launch ROCM kernel with args: "
               << args.number_of_arguments()
               << "; thread dim: " << thread_dims.ToString()
               << "; block dim: " << block_dims.ToString();
    return false;
  }

  return true;
}

// This is a non-essential operation; if there's a failure, proceed without
// logging an error. It's nearly certain that in case of failures, we'd never
// get here in the first place; these are very low-impact routines.
void ROCMExecutor::VlogOccupancyInfo(const KernelBase &kernel,
                                     const ThreadDim &thread_dims,
                                     const BlockDim &block_dims) {
  // ROCM TODO implement this feature in HIP
}

void *ROCMExecutor::Allocate(uint64 size) {
  return ROCMDriver::DeviceAllocate(device_ordinal_, size);
}

void *ROCMExecutor::AllocateSubBuffer(DeviceMemoryBase *mem,
                                      uint64 offset_bytes, uint64 size_bytes) {
  // offset and size are in bytes, so char* works as the pointer type.
  return reinterpret_cast<char *>(mem->opaque()) + offset_bytes;
}

void ROCMExecutor::Deallocate(DeviceMemoryBase *mem) {
  // ROCM "sub-buffers" are just pointer + offset, so no dealloc is necessary.
  if (!mem->is_sub_buffer()) {
    ROCMDriver::DeviceDeallocate(device_ordinal_, mem->opaque());
  }
}

bool ROCMExecutor::HostMemoryRegister(void *location, uint64 size) {
  if (location == nullptr || size == 0) {
    LOG(WARNING) << "attempting to register null or zero-sized memory: "
                 << location << "; size " << size;
  }
  VLOG(2) << "registering " << location << " size " << size;
  return ROCMDriver::HostRegister(device_ordinal_, location, size);
}

bool ROCMExecutor::HostMemoryUnregister(void *location) {
  VLOG(2) << "unregistering " << location;
  return ROCMDriver::HostUnregister(device_ordinal_, location);
}

bool ROCMExecutor::SynchronizeAllActivity() {
  return ROCMDriver::SynchronizeDevice(device_ordinal_);
}

bool ROCMExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return ROCMDriver::SynchronousMemsetUint32(
        device_ordinal_, AsROCmDevicePtr(location), 0x0, size / 4);
  }
  return ROCMDriver::SynchronousMemsetUint8(device_ordinal_, AsROCmDevicePtr(location),
                                            0x0, size);
}

bool ROCMExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                     uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    // hipMemset reinterprets "value" as a uint8.
    uint8 byte_value = static_cast<uint8>(value);
    uint32 pattern = (byte_value << 24) | (byte_value << 16) |
                     (byte_value << 8) | byte_value;
    return ROCMDriver::SynchronousMemsetUint32(
        device_ordinal_, AsROCmDevicePtr(location), pattern, size / 4);
  }
  return ROCMDriver::SynchronousMemsetUint8(device_ordinal_, AsROCmDevicePtr(location),
                                            value, size);
}

port::Status ROCMExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                             const void *host_src,
                                             uint64 size) {
  return ROCMDriver::SynchronousMemcpyH2D(device_ordinal_, AsROCmDevicePtr(gpu_dst),
                                          host_src, size);
}

port::Status ROCMExecutor::SynchronousMemcpy(void *host_dst,
                                             const DeviceMemoryBase &gpu_src,
                                             uint64 size) {
  return ROCMDriver::SynchronousMemcpyD2H(device_ordinal_, host_dst,
                                          AsROCmDevicePtr(gpu_src), size);
}

port::Status ROCMExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src, uint64 size) {
  return ROCMDriver::SynchronousMemcpyD2D(device_ordinal_, AsROCmDevicePtr(gpu_dst),
                                          AsROCmDevicePtr(gpu_src), size);
}

bool ROCMExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                           uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return Memset32(stream, location, 0x0, size);
  } else {
    return Memset(stream, location, 0x0, size);
  }
}

bool ROCMExecutor::Memset(Stream *stream, DeviceMemoryBase *location,
                           uint8 pattern, uint64 size) {
  VLOG(2) << "enqueueing memset8 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  return ROCMDriver::AsynchronousMemsetUint8(
      device_ordinal_, AsROCmDevicePtr(location), pattern, size,
      AsROCMStreamValue(stream));
}

bool ROCMExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                            uint32 pattern, uint64 size) {
  VLOG(2) << "enqueueing memset32 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
        size % 4 == 0);
  return ROCMDriver::AsynchronousMemsetUint32(
      device_ordinal_, AsROCmDevicePtr(location), pattern, size / 4,
      AsROCMStreamValue(stream));
}

bool ROCMExecutor::Memcpy(Stream *stream, void *host_dst,
                          const DeviceMemoryBase &gpu_src, uint64 size) {
  return ROCMDriver::AsynchronousMemcpyD2H(device_ordinal_, host_dst,
                                           AsROCmDevicePtr(gpu_src), size,
                                           AsROCMStreamValue(stream));
}

bool ROCMExecutor::Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                          const void *host_src, uint64 size) {
  return ROCMDriver::AsynchronousMemcpyH2D(device_ordinal_, AsROCmDevicePtr(gpu_dst),
                                           host_src, size,
                                           AsROCMStreamValue(stream));
}

bool ROCMExecutor::MemcpyDeviceToDevice(Stream *stream,
                                        DeviceMemoryBase *gpu_dst,
                                        const DeviceMemoryBase &gpu_src,
                                        uint64 size) {
  return ROCMDriver::AsynchronousMemcpyD2D(device_ordinal_, AsROCmDevicePtr(gpu_dst),
                                           AsROCmDevicePtr(gpu_src), size,
                                           AsROCMStreamValue(stream));
}

bool ROCMExecutor::HostCallback(Stream *stream,
                                std::function<void()> callback) {
  auto callback_ptr = new std::function<void()>(callback);
  return ROCMDriver::AddStreamCallback(device_ordinal_, AsROCMStreamValue(stream),
                                       InternalHostCallback, callback_ptr);
}

/* static */ void ROCMExecutor::InternalHostCallback(hipStream_t stream,
                                                     hipError_t status,
                                                     void *data) {
  std::function<void()> *callback =
      reinterpret_cast<std::function<void()> *>(data);
  (*callback)();
  delete callback;
}

port::Status ROCMExecutor::AllocateEvent(Event *event) {
  return AsROCMEvent(event)->Init();
}

port::Status ROCMExecutor::DeallocateEvent(Event *event) {
  return AsROCMEvent(event)->Destroy();
}

port::Status ROCMExecutor::RecordEvent(Stream *stream, Event *event) {
  return AsROCMEvent(event)->Record(AsROCMStream(stream));
}

port::Status ROCMExecutor::WaitForEvent(Stream *stream, Event *event) {
  if (ROCMDriver::WaitStreamOnEvent(device_ordinal_,
                                    AsROCMStream(stream)->rocm_stream(),
                                    AsROCMEvent(event)->rocm_event())) {
    return port::Status::OK();
  } else {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("error recording waiting for ROCM event on stream %p",
                     stream)};
  }
}

Event::Status ROCMExecutor::PollForEventStatus(Event *event) {
  return AsROCMEvent(event)->PollForStatus();
}

bool ROCMExecutor::AllocateStream(Stream *stream) {
  return AsROCMStream(stream)->Init();
}

void ROCMExecutor::DeallocateStream(Stream *stream) {
  ROCMStream *rocm_stream = AsROCMStream(stream);
  if (!rocm_stream->IsIdle()) {
    LOG(ERROR) << "Deallocating stream with pending work";
  }
  rocm_stream->Destroy();
}

bool ROCMExecutor::AllocateTimer(Timer *timer) {
  return AsROCMTimer(timer)->Init();
}

void ROCMExecutor::DeallocateTimer(Timer *timer) {
  AsROCMTimer(timer)->Destroy();
}

bool ROCMExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  hipEvent_t other_completed_event = *AsROCMStream(other)->completed_event();
  bool ok = ROCMDriver::RecordEvent(device_ordinal_, other_completed_event,
                                    AsROCMStreamValue(other))
      .ok();
  if (!ok) {
    LOG(ERROR) << "failed to record completion event; "
                  "therefore, failed to create inter-stream dependency";
    return false;
  }

  return ROCMDriver::WaitStreamOnEvent(device_ordinal_, AsROCMStreamValue(dependent),
                                       other_completed_event);
}

bool ROCMExecutor::StartTimer(Stream *stream, Timer *timer) {
  return AsROCMTimer(timer)->Start(AsROCMStream(stream));
}

bool ROCMExecutor::StopTimer(Stream *stream, Timer *timer) {
  return AsROCMTimer(timer)->Stop(AsROCMStream(stream));
}

port::Status ROCMExecutor::BlockHostUntilDone(Stream *stream) {
  return ROCMDriver::SynchronizeStream(device_ordinal_, AsROCMStreamValue(stream));
}

blas::BlasSupport *ROCMExecutor::CreateBlas() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(kROCmPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

dnn::DnnSupport *ROCMExecutor::CreateDnn() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(kROCmPlatformId,
                                                       plugin_config_.dnn());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

fft::FftSupport *ROCMExecutor::CreateFft() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(kROCmPlatformId,
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

rng::RngSupport *ROCMExecutor::CreateRng() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(kROCmPlatformId,
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

// TODO(rspringer): Remove in b/18544742.
bool ROCMExecutor::SupportsDnn() const {
  return true;
}

bool ROCMExecutor::CanEnablePeerAccessTo(StreamExecutorInterface *other) {
  ROCMExecutor *rocm_other = static_cast<ROCMExecutor *>(other);
  return ROCMDriver::CanEnablePeerAccess(device_ordinal_, rocm_other->device_ordinal());
}

port::Status ROCMExecutor::EnablePeerAccessTo(StreamExecutorInterface *other) {
  ROCMExecutor *rocm_other = static_cast<ROCMExecutor *>(other);
  return ROCMDriver::EnablePeerAccess(device_ordinal_, rocm_other->device_ordinal());
}

SharedMemoryConfig ROCMExecutor::GetDeviceSharedMemoryConfig() {
  port::StatusOr<hipSharedMemConfig> rocm_config =
      ROCMDriver::DeviceGetSharedMemConfig(device_ordinal_);
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

port::Status ROCMExecutor::SetDeviceSharedMemoryConfig(
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
  return ROCMDriver::DeviceSetSharedMemConfig(device_ordinal_, rocm_config);
}

bool ROCMExecutor::DeviceMemoryUsage(int64 *free, int64 *total) const {
  return ROCMDriver::GetDeviceMemoryInfo(device_ordinal_, free, total);
}

bool ROCMExecutor::GetSymbol(const string& symbol_name, ModuleHandle module_handle, void **mem,
                             size_t *bytes) {
  {  // give limited scope to mutex_lock
    mutex_lock lock{disk_modules_mu_};
    for (auto &it : disk_modules_) {
      if (ROCMDriver::GetModuleSymbol(device_ordinal_, it.second, symbol_name.c_str(),
                                      reinterpret_cast<hipDeviceptr_t *>(mem),
                                      bytes)) {
        return true;
      }
    }
  }

  {  // give limited scope to mutex_lock
    mutex_lock lock{in_memory_modules_mu_};
    for (auto &it : in_memory_modules_) {
      if (ROCMDriver::GetModuleSymbol(device_ordinal_, it.second, symbol_name.c_str(),
                                      reinterpret_cast<hipDeviceptr_t *>(mem),
                                      bytes)) {
        return true;
      }
    }
  }

  LOG(INFO) << "Falied to find symbol in any modules: " << symbol_name;
  return false;
}

bool ROCMExecutor::FillBlockDimLimit(BlockDim *block_dim_limit) const {
  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
  if (!ROCMDriver::GetGridLimits(&x, &y, &z, device_)) {
    return false;
  }

  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return true;
}

bool ROCMExecutor::SupportsBlas() const { return true; }

bool ROCMExecutor::SupportsFft() const { return true; }

bool ROCMExecutor::SupportsRng() const { return true; }

std::unique_ptr<internal::EventInterface>
ROCMExecutor::CreateEventImplementation() {
  return std::unique_ptr<internal::EventInterface>(new ROCMEvent(this));
}

std::unique_ptr<internal::KernelInterface>
ROCMExecutor::CreateKernelImplementation() {
  return std::unique_ptr<internal::KernelInterface>(new ROCMKernel());
}

std::unique_ptr<internal::StreamInterface>
ROCMExecutor::GetStreamImplementation() {
  return std::unique_ptr<internal::StreamInterface>(new ROCMStream(this));
}

std::unique_ptr<internal::TimerInterface>
ROCMExecutor::GetTimerImplementation() {
  return std::unique_ptr<internal::TimerInterface>(new ROCMTimer(this));
}

void *ROCMExecutor::GpuContextHack() { return nullptr; }

// Attempts to read the NUMA node corresponding to the GPU device's PCI bus out
// of SysFS. Returns -1 if it cannot.
//
// For anything more complicated/prod-focused than this, you'll likely want to
// turn to gsys' topology modeling.
static int TryToReadNumaNode(const string &pci_bus_id, int device_ordinal) {
  // ROCM TODO implement this feature in HIP
  return 1;
}

// Set of device-specific parameters that cannot be
// queried from the driver API.  These values instead are baked into a
// lookup table indexed by AMDGPU ISA version.
struct UnqueryableDeviceParams {
  int version;
  uint64 blocks_per_core_limit;
  uint64 registers_per_core_limit;
  uint64 registers_per_thread_limit;
  uint64 warp_alloc_granularity;
  uint64 register_alloc_granularity;
  uint64 shared_memory_alloc_granularity;
};

static const UnqueryableDeviceParams kAllUnqueryableDeviceParams[] = {
  {
    803,        // AMDGPU ISA version (803)
    16,         // blocks_per_core_limit
    64 * 1024,  // registers_per_core_limit
    255,        // registers_per_thread_limit
    4,          // warp_alloc_granularity
    256,        // register_alloc_granularity
    256         // shared_memory_alloc_granularity
  },
  {
    900,        // AMDGPU ISA version (900)
    16,         // blocks_per_core_limit
    64 * 1024,  // registers_per_core_limit
    255,        // registers_per_thread_limit
    4,          // warp_alloc_granularity
    256,        // register_alloc_granularity
    256         // shared_memory_alloc_granularity
  },
  {
    906,        // AMDGPU ISA version (900)
    16,         // blocks_per_core_limit
    64 * 1024,  // registers_per_core_limit
    255,        // registers_per_thread_limit
    4,          // warp_alloc_granularity
    256,        // register_alloc_granularity
    256         // shared_memory_alloc_granularity
  },
};

DeviceDescription *ROCMExecutor::PopulateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  {
    int driver_version = 0;
    (void)ROCMDriver::GetDriverVersion(&driver_version);
    string augmented_driver_version = port::Printf(
        "%d (%s)", driver_version,
        DriverVersionStatusToString(Diagnostician::FindDsoVersion()).c_str());
    builder.set_driver_version(augmented_driver_version);
  }

  {
    string pci_bus_id = ROCMDriver::GetPCIBusID(device_);

    // Lower the hex characters to match sysfs.
    pci_bus_id = port::Lowercase(pci_bus_id);
    builder.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = TryToReadNumaNode(pci_bus_id, device_ordinal_);
    builder.set_numa_node(numa_node);
  }

  hipDeviceProp_t prop;
  if (ROCMDriver::GetDeviceProperties(&prop, device_ordinal_)) {
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
    (void)ROCMDriver::IsEccEnabled(device_, &ecc_enabled);
    builder.set_ecc_enabled(ecc_enabled);
  }

  {
    uint64 device_memory_size = -1;
    (void)ROCMDriver::GetDeviceTotalMemory(device_, &device_memory_size);
    builder.set_device_memory_size(device_memory_size);
  }

  {
    BlockDim block_dim_limit;
    FillBlockDimLimit(&block_dim_limit);
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    string device_name;
    (void)ROCMDriver::GetDeviceName(device_, &device_name);
    builder.set_name(device_name);
  }

  for (size_t i = 0; i < TF_ARRAYSIZE(kAllUnqueryableDeviceParams); i++) {
    const auto &params = kAllUnqueryableDeviceParams[i];
    if (params.version == version_) {
      builder.set_blocks_per_core_limit(params.blocks_per_core_limit);
      builder.set_registers_per_core_limit(params.registers_per_core_limit);
      builder.set_registers_per_thread_limit(params.registers_per_thread_limit);
      builder.set_warp_alloc_granularity(params.warp_alloc_granularity);
      builder.set_register_alloc_granularity(params.register_alloc_granularity);
      builder.set_shared_memory_alloc_granularity(
          params.shared_memory_alloc_granularity);
    }
  }

  builder.set_platform_version(
      port::StrCat("AMDGPU ISA version: gfx", version_));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  builder.set_device_address_bits(64);

  builder.set_device_vendor("Advanced Micro Devices, Inc");
  builder.set_rocm_amdgpu_isa_version(version_);
  builder.set_shared_memory_per_core(
      ROCMDriver::GetMaxSharedMemoryPerCore(device_).ValueOrDie());
  builder.set_shared_memory_per_block(
      ROCMDriver::GetMaxSharedMemoryPerBlock(device_).ValueOrDie());
  builder.set_core_count(
      ROCMDriver::GetMultiprocessorCount(device_).ValueOrDie());
  builder.set_threads_per_core_limit(
      ROCMDriver::GetMaxThreadsPerMultiprocessor(device_).ValueOrDie());
  builder.set_registers_per_block_limit(
      ROCMDriver::GetMaxRegistersPerBlock(device_).ValueOrDie());
  builder.set_threads_per_warp(
      ROCMDriver::GetThreadsPerWarp(device_).ValueOrDie());

  auto built = builder.Build();
  return built.release();
}

}  // namespace rocm

void initialize_rocm_gpu_executor() {
  *internal::MakeROCMExecutorImplementation() = [](const PluginConfig &config) {
    return new rocm::ROCMExecutor{config};
  };
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(
    rocm_gpu_executor, {stream_executor::initialize_rocm_gpu_executor();});
