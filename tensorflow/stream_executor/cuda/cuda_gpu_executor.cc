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

#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#define PATH_MAX MAX_PATH
#else
#include <unistd.h>
#endif
#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_event.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_timer.h"
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

namespace perftools {
namespace gputools {
namespace cuda {

// Hook that can be used to CUBIN-ate PTX before it is loaded into the driver.
// It has been observed that loading both PTX and cubins into the driver library
// can cause it to crash, but loading only CUBINs avoids those crashes;
// therefore, it's useful to have this hook to hack in uniform CUBIN-ation of
// PTX code.
//
// As this is an implementation-detail workaround, the usage is to declare this
// variable with extern linkage and populate it from another translation unit.
std::function<string(const string &)> g_cubinate;

static CUDAEvent *AsCUDAEvent(Event *event) {
  DCHECK(event != nullptr);
  return static_cast<CUDAEvent *>(event->implementation());
}


// Given a platform-independent timer datatype, returns the internal CUDA
// platform implementation pointer.
static CUDATimer *AsCUDATimer(Timer *timer) {
  DCHECK(timer != nullptr);
  return static_cast<CUDATimer *>(timer->implementation());
}

// Given const GPU memory, returns a libcuda device pointer datatype, suitable
// for passing directly to libcuda APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// libcuda APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to libcuda functions which will honor constness.
static CUdeviceptr AsCudaDevicePtr(const DeviceMemoryBase &gpu_mem) {
  return reinterpret_cast<CUdeviceptr>(gpu_mem.opaque());
}

// See description on const version above.
static CUdeviceptr AsCudaDevicePtr(DeviceMemoryBase *gpu_mem) {
  return AsCudaDevicePtr(*gpu_mem);
}

CudaContext* ExtractCudaContext(CUDAExecutor *cuda_exec) {
  CHECK(cuda_exec != nullptr);
  return cuda_exec->cuda_context();
}

CUDAExecutor *ExtractCudaExecutor(StreamExecutor *stream_exec) {
  return static_cast<CUDAExecutor *>(stream_exec->implementation());
}

CUDAExecutor::~CUDAExecutor() {
  CHECK(kernel_to_gpu_binary_.empty()) << "CUDAExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "CUDAExecutor has loaded modules.";
  if (context_ != nullptr) {
    CUDADriver::DestroyContext(context_);
  }
}

port::Status CUDAExecutor::Init(int device_ordinal,
                                DeviceOptions device_options) {
  device_ordinal_ = device_ordinal;

  auto status = CUDADriver::Init();
  if (!status.ok()) {
    return status;
  }

  status = CUDADriver::GetDevice(device_ordinal_, &device_);
  if (!status.ok()) {
    return status;
  }

  status = CUDADriver::CreateContext(device_, device_options, &context_);
  if (!status.ok()) {
    return status;
  }

  return CUDADriver::GetComputeCapability(&cc_major_, &cc_minor_, device_);
}

bool CUDAExecutor::FindOnDiskForComputeCapability(
    port::StringPiece filename, port::StringPiece canonical_suffix,
    string *found_filename) const {
  if (cc_major_ == 0 && cc_minor_ == 0) {
    return false;
  }

  string cc_specific =
      port::StrCat(filename, ".cc", cc_major_, cc_minor_, canonical_suffix);
  if (port::FileExists(cc_specific).ok()) {
    VLOG(2) << "found compute-capability-specific file, using that: "
            << cc_specific;
    *found_filename = cc_specific;
    return true;
  }

  VLOG(2) << "could not find compute-capability specific file at: "
          << cc_specific;
  if (port::FileExists(filename.ToString()).ok()) {
    *found_filename = filename.ToString();
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
#if defined(__APPLE__)
    uint32_t buffer_size = 0U;
    _NSGetExecutablePath(nullptr, &buffer_size);
    char unresolved_path[buffer_size];
    _NSGetExecutablePath(unresolved_path, &buffer_size);
    CHECK_ERR(realpath(unresolved_path, exe_path) ? 1 : -1);
#else
#if defined(PLATFORM_WINDOWS)
  HMODULE hModule = GetModuleHandle(NULL);
  GetModuleFileName(hModule, exe_path, MAX_PATH);
#else
  CHECK_ERR(readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1));
#endif
#endif
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

bool CUDAExecutor::GetKernel(const MultiKernelLoaderSpec &spec,
                             KernelBase *kernel) {
  CUDAKernel *cuda_kernel = AsCUDAKernel(kernel);
  CUmodule module;
  const string *kernelname;

  VLOG(3) << "GetKernel on kernel " << kernel << " : " << kernel->name();

  if (spec.has_cuda_cubin_in_memory()) {
    kernelname = &spec.cuda_cubin_in_memory().kernelname();
    const char *cubin = spec.cuda_cubin_in_memory().bytes();
    mutex_lock lock{in_memory_modules_mu_};
    uint64_t module_refcount;
    std::tie(module, module_refcount) = gpu_binary_to_module_[cubin];

    if (module == nullptr) {
      auto load_status = CUDADriver::LoadCubin(context_, cubin, &module);
      if (!load_status.ok()) {
        LOG(ERROR) << "failed to load CUBIN: " << load_status;
        return false;
      }
      module_refcount = 1;
      VLOG(3) << "Loaded CUBIN " << static_cast<const void *>(cubin)
              << " as module " << module;
    } else {
      ++module_refcount;
      VLOG(3) << "CUBIN " << static_cast<const void *>(cubin)
              << " is already loaded as module " << module;
    }
    kernel_to_gpu_binary_[kernel] = cubin;
    gpu_binary_to_module_[cubin] = {module, module_refcount};
  } else if (spec.has_cuda_ptx_in_memory()) {
    kernelname = &spec.cuda_ptx_in_memory().kernelname();

    if (cc_major_ == 0 && cc_minor_ == 0) {
      return false;
    }

    const char *ptx = spec.cuda_ptx_in_memory().text(cc_major_, cc_minor_);
    if (ptx == nullptr) {
      ptx = spec.cuda_ptx_in_memory().default_text();
    }
    if (ptx == nullptr) {
      LOG(FATAL) << "loader spec has no ptx for kernel " << *kernelname;
      return false;
    }

    mutex_lock lock{in_memory_modules_mu_};
    uint64_t module_refcount;
    std::tie(module, module_refcount) = gpu_binary_to_module_[ptx];

    if (module == nullptr) {
      if (!CUDADriver::LoadPtx(context_, ptx, &module)) {
        LOG(ERROR) << "failed to load PTX for kernel " << *kernelname;
        return false;
      }
      VLOG(3) << "Loaded PTX " << static_cast<const void *>(ptx)
              << " as module " << module;
      module_refcount = 1;
    } else {
      ++module_refcount;
      VLOG(3) << "PTX " << static_cast<const void *>(ptx)
              << " is already loaded as module " << module;
    }
    kernel_to_gpu_binary_[kernel] = ptx;
    gpu_binary_to_module_[ptx] = {module, module_refcount};
  } else {
    LOG(WARNING) << "no method of loading CUDA kernel provided";
    return false;
  }
  VLOG(2) << "getting function " << *kernelname << " from module " << module;
  if (!CUDADriver::GetModuleFunction(context_, module, kernelname->c_str(),
                                     cuda_kernel->cuda_function_ptr())) {
    return false;
  }

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the CUDA API.
  cuda_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
  if (!GetKernelMetadata(cuda_kernel, &kernel_metadata)) {
    LOG(WARNING) << "unable to get metadata for kernel " << *kernelname;
  }
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(*kernelname);
  return true;
}

void CUDAExecutor::UnloadKernel(const KernelBase *kernel) {
  VLOG(3) << "Unloading kernel " << kernel << " : " << kernel->name();

  mutex_lock lock{in_memory_modules_mu_};
  auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
  if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
    VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
            << " has never been loaded.";
    return;  // We've never seen this kernel.
  }
  VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
          << " has loaded GPU code " << gpu_binary_it->second;
  auto module_it = gpu_binary_to_module_.find(gpu_binary_it->second);
  if (gpu_binary_to_module_.end() == module_it) {
    VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
            << " has no loaded CUDA module.";
    return;  // This kernel never loaded any modules
  }
  auto &module = module_it->second.first;
  auto &refcount = module_it->second.second;
  VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
          << " has loaded GPU code " << gpu_binary_it->second
          << " into CUDA module " << module << " with refcount " << refcount;
  if (--refcount == 0) {
    VLOG(3) << "Unloading CUDA module " << module;
    CUDADriver::UnloadModule(context_, module);
    gpu_binary_to_module_.erase(module_it);
  }
  kernel_to_gpu_binary_.erase(gpu_binary_it);
}

bool CUDAExecutor::GetKernelMetadata(CUDAKernel *cuda_kernel,
                                     KernelMetadata *kernel_metadata) {
  int value;
  if (!CUDADriver::FuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS,
                                    *cuda_kernel->cuda_function_ptr(),
                                    &value)) {
    return false;
  }
  kernel_metadata->set_registers_per_thread(value);

  if (!CUDADriver::FuncGetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                    *cuda_kernel->cuda_function_ptr(),
                                    &value)) {
    return false;
  }
  kernel_metadata->set_shared_memory_bytes(value);

  return true;
}

bool CUDAExecutor::Launch(Stream *stream, const ThreadDim &thread_dims,
                          const BlockDim &block_dims, const KernelBase &kernel,
                          const KernelArgsArrayBase &args) {
  CHECK_EQ(kernel.Arity(), args.number_of_arguments());
  CUstream custream = AsCUDAStreamValue(stream);
  const CUDAKernel *cuda_kernel = AsCUDAKernel(&kernel);
  CUfunction cufunc = cuda_kernel->AsCUDAFunctionValue();

  // Only perform/print the occupancy check once.  Even just checking to see
  // whether we've done an occupancy check on this kernel before isn't free
  // (because we have to synchronize), so we only do this at -v 2+.
  if (VLOG_IS_ON(2)) {
    mutex_lock lock(launched_kernels_mu_);
    if (!launched_kernels_.count(cufunc)) {
      VlogOccupancyInfo(kernel, thread_dims, block_dims);
      // TODO(rspringer): Remove elements from launched_kernels_...if we ever
      // expose a kernel/module deallocation method.
      launched_kernels_.insert(cufunc);
    }
  }

  if (cuda_kernel->GetPreferredCacheConfig() !=
      KernelCacheConfig::kNoPreference) {
    CUDADriver::FuncSetCacheConfig(cufunc, cuda_kernel->GetCUDACacheConfig());
  }

  void **kernel_params = const_cast<void **>(args.argument_addresses().data());

  if (!CUDADriver::LaunchKernel(context_, cufunc, block_dims.x, block_dims.y,
                                block_dims.z, thread_dims.x, thread_dims.y,
                                thread_dims.z, args.number_of_shared_bytes(),
                                custream, kernel_params,
                                nullptr /* = extra */)) {
    LOG(ERROR) << "failed to launch CUDA kernel " << kernel.name() << " with "
               << args.number_of_arguments()
               << " args; thread dim: " << thread_dims.ToString()
               << "; block dim: " << block_dims.ToString();
    return false;
  }

  return true;
}

// This is a non-essential operation; if there's a failure, proceed without
// logging an error. It's nearly certain that in case of failures, we'd never
// get here in the first place; these are very low-impact routines.
void CUDAExecutor::VlogOccupancyInfo(const KernelBase &kernel,
                                     const ThreadDim &thread_dims,
                                     const BlockDim &block_dims) {
  VLOG(2) << "Computing kernel occupancy for kernel "
          << kernel.demangled_name();
  VLOG(2) << "Thread dimensions (" << thread_dims.x << ", " << thread_dims.y
          << ", " << thread_dims.z << ")";

  int regs_per_thread;
  if (!kernel.metadata().registers_per_thread(&regs_per_thread)) {
    return;
  }

  int smem_per_block;
  if (!kernel.metadata().shared_memory_bytes(&smem_per_block)) {
    return;
  }

  const DeviceDescription &device_description =
      kernel.parent()->GetDeviceDescription();

  uint64 blocks_per_sm = CalculateOccupancy(
      device_description, regs_per_thread, smem_per_block, thread_dims);
  VLOG(2) << "Resident blocks per SM is " << blocks_per_sm;

  // To increase occupancy, there must be a sufficient number of blocks
  // available to spread across the sm's at this new improved occupancy level.
  int multiprocessor_count = device_description.core_count();
  int block_count = block_dims.x * block_dims.y * block_dims.z;
  int available_blocks_per_sm =
      port::MathUtil::CeilOfRatio(block_count, multiprocessor_count);
  if (available_blocks_per_sm <= static_cast<int64>(blocks_per_sm)) {
    VLOG(2) << "Occupancy is limited by number of blocks available per sm.";
    return;
  }

  uint64 improved_regs_per_thread = CalculateRegisterLimitForTargetOccupancy(
      device_description, smem_per_block, thread_dims, blocks_per_sm + 1);
  if (improved_regs_per_thread != 0) {
    VLOG(2) << "Reducing register usage from " << regs_per_thread
            << " to " << improved_regs_per_thread
            << " could increase resident blocks per SM by one.";
  } else {
    VLOG(2) << "Resident blocks per SM cannot be increased by reducing "
        "register usage.";
  }
}

void *CUDAExecutor::Allocate(uint64 size) {
  return CUDADriver::DeviceAllocate(context_, size);
}

void *CUDAExecutor::AllocateSubBuffer(DeviceMemoryBase *mem,
                                      uint64 offset_bytes, uint64 size_bytes) {
  // offset and size are in bytes, so char* works as the pointer type.
  return reinterpret_cast<char *>(mem->opaque()) + offset_bytes;
}

void CUDAExecutor::Deallocate(DeviceMemoryBase *mem) {
  // CUDA "sub-buffers" are just pointer + offset, so no dealloc is necessary.
  if (!mem->is_sub_buffer()) {
    CUDADriver::DeviceDeallocate(context_, mem->opaque());
  }
}

bool CUDAExecutor::HostMemoryRegister(void *location, uint64 size) {
  if (location == nullptr || size == 0) {
    LOG(WARNING) << "attempting to register null or zero-sized memory: "
                 << location << "; size " << size;
  }
  VLOG(2) << "registering " << location << " size " << size;
  return CUDADriver::HostRegister(context_, location, size);
}

bool CUDAExecutor::HostMemoryUnregister(void *location) {
  VLOG(2) << "unregistering " << location;
  return CUDADriver::HostUnregister(context_, location);
}

bool CUDAExecutor::SynchronizeAllActivity() {
  return CUDADriver::SynchronizeContext(context_);
}

bool CUDAExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return CUDADriver::SynchronousMemsetUint32(
        context_, AsCudaDevicePtr(location), 0x0, size / 4);
  }
  return CUDADriver::SynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                            0x0, size);
}

bool CUDAExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                     uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    // cudaMemset reinterprets "value" as a uint8.
    uint8 byte_value = static_cast<uint8>(value);
    uint32 pattern = (byte_value << 24) | (byte_value << 16) |
                     (byte_value << 8) | byte_value;
    return CUDADriver::SynchronousMemsetUint32(
        context_, AsCudaDevicePtr(location), pattern, size / 4);
  }
  return CUDADriver::SynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                            value, size);
}

port::Status CUDAExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                             const void *host_src,
                                             uint64 size) {
  return CUDADriver::SynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                          host_src, size);
}

port::Status CUDAExecutor::SynchronousMemcpy(void *host_dst,
                                             const DeviceMemoryBase &gpu_src,
                                             uint64 size) {
  return CUDADriver::SynchronousMemcpyD2H(context_, host_dst,
                                          AsCudaDevicePtr(gpu_src), size);
}

port::Status CUDAExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src, uint64 size) {
  return CUDADriver::SynchronousMemcpyD2D(context_, AsCudaDevicePtr(gpu_dst),
                                          AsCudaDevicePtr(gpu_src), size);
}

bool CUDAExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                           uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return Memset32(stream, location, 0x0, size);
  } else {
    return Memset(stream, location, 0x0, size);
  }
}

bool CUDAExecutor::Memset(Stream *stream, DeviceMemoryBase *location,
                           uint8 pattern, uint64 size) {
  VLOG(2) << "enqueueing memset8 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  return CUDADriver::AsynchronousMemsetUint8(
      context_, AsCudaDevicePtr(location), pattern, size,
      AsCUDAStreamValue(stream));
}

bool CUDAExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                            uint32 pattern, uint64 size) {
  VLOG(2) << "enqueueing memset32 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
        size % 4 == 0);
  return CUDADriver::AsynchronousMemsetUint32(
      context_, AsCudaDevicePtr(location), pattern, size / 4,
      AsCUDAStreamValue(stream));
}

bool CUDAExecutor::Memcpy(Stream *stream, void *host_dst,
                          const DeviceMemoryBase &gpu_src, uint64 size) {
  return CUDADriver::AsynchronousMemcpyD2H(context_, host_dst,
                                           AsCudaDevicePtr(gpu_src), size,
                                           AsCUDAStreamValue(stream));
}

bool CUDAExecutor::Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                          const void *host_src, uint64 size) {
  return CUDADriver::AsynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                           host_src, size,
                                           AsCUDAStreamValue(stream));
}

bool CUDAExecutor::MemcpyDeviceToDevice(Stream *stream,
                                        DeviceMemoryBase *gpu_dst,
                                        const DeviceMemoryBase &gpu_src,
                                        uint64 size) {
  return CUDADriver::AsynchronousMemcpyD2D(context_, AsCudaDevicePtr(gpu_dst),
                                           AsCudaDevicePtr(gpu_src), size,
                                           AsCUDAStreamValue(stream));
}

bool CUDAExecutor::HostCallback(Stream *stream,
                                std::function<void()> callback) {
  auto callback_ptr = new std::function<void()>(callback);
  return CUDADriver::AddStreamCallback(context_, AsCUDAStreamValue(stream),
                                       InternalHostCallback, callback_ptr);
}

/* static */ void CUDAExecutor::InternalHostCallback(CUstream stream,
                                                     CUresult status,
                                                     void *data) {
  std::function<void()> *callback =
      reinterpret_cast<std::function<void()> *>(data);
  (*callback)();
  delete callback;
}

port::Status CUDAExecutor::AllocateEvent(Event *event) {
  return AsCUDAEvent(event)->Init();
}

port::Status CUDAExecutor::DeallocateEvent(Event *event) {
  return AsCUDAEvent(event)->Destroy();
}

port::Status CUDAExecutor::RecordEvent(Stream *stream, Event *event) {
  return AsCUDAEvent(event)->Record(AsCUDAStream(stream));
}

port::Status CUDAExecutor::WaitForEvent(Stream *stream, Event *event) {
  if (CUDADriver::WaitStreamOnEvent(context_,
                                    AsCUDAStream(stream)->cuda_stream(),
                                    AsCUDAEvent(event)->cuda_event())) {
    return port::Status::OK();
  } else {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("error recording waiting for CUDA event on stream %p",
                     stream)};
  }
}

Event::Status CUDAExecutor::PollForEventStatus(Event *event) {
  return AsCUDAEvent(event)->PollForStatus();
}

bool CUDAExecutor::AllocateStream(Stream *stream) {
  return AsCUDAStream(stream)->Init();
}

void CUDAExecutor::DeallocateStream(Stream *stream) {
  CUDAStream *cuda_stream = AsCUDAStream(stream);
  if (!cuda_stream->IsIdle()) {
    LOG(ERROR) << "Deallocating stream with pending work";
  }
  cuda_stream->Destroy();
}

bool CUDAExecutor::AllocateTimer(Timer *timer) {
  return AsCUDATimer(timer)->Init();
}

void CUDAExecutor::DeallocateTimer(Timer *timer) {
  AsCUDATimer(timer)->Destroy();
}

bool CUDAExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  CUevent other_completed_event = *AsCUDAStream(other)->completed_event();
  bool ok = CUDADriver::RecordEvent(context_, other_completed_event,
                                    AsCUDAStreamValue(other))
      .ok();
  if (!ok) {
    LOG(ERROR) << "failed to record completion event; "
                  "therefore, failed to create inter-stream dependency";
    return false;
  }

  return CUDADriver::WaitStreamOnEvent(context_, AsCUDAStreamValue(dependent),
                                       other_completed_event);
}

bool CUDAExecutor::StartTimer(Stream *stream, Timer *timer) {
  return AsCUDATimer(timer)->Start(AsCUDAStream(stream));
}

bool CUDAExecutor::StopTimer(Stream *stream, Timer *timer) {
  return AsCUDATimer(timer)->Stop(AsCUDAStream(stream));
}

port::Status CUDAExecutor::BlockHostUntilDone(Stream *stream) {
  return CUDADriver::SynchronizeStream(context_, AsCUDAStreamValue(stream));
}

blas::BlasSupport *CUDAExecutor::CreateBlas() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(kCudaPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

dnn::DnnSupport *CUDAExecutor::CreateDnn() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(kCudaPlatformId,
                                                       plugin_config_.dnn());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

fft::FftSupport *CUDAExecutor::CreateFft() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(kCudaPlatformId,
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

rng::RngSupport *CUDAExecutor::CreateRng() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(kCudaPlatformId,
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

// TODO(rspringer): Remove in b/18544742.
bool CUDAExecutor::SupportsDnn() const {
  return true;
}

bool CUDAExecutor::CanEnablePeerAccessTo(StreamExecutorInterface *other) {
  CUDAExecutor *cuda_other = static_cast<CUDAExecutor *>(other);
  return CUDADriver::CanEnablePeerAccess(context_, cuda_other->context_);
}

port::Status CUDAExecutor::EnablePeerAccessTo(StreamExecutorInterface *other) {
  CUDAExecutor *cuda_other = static_cast<CUDAExecutor *>(other);
  return CUDADriver::EnablePeerAccess(context_, cuda_other->context_);
}

SharedMemoryConfig CUDAExecutor::GetDeviceSharedMemoryConfig() {
  port::StatusOr<CUsharedconfig> cuda_config =
      CUDADriver::ContextGetSharedMemConfig(context_);
  if (!cuda_config.ok()) {
    // Don't log; the failed call will log necessary output.
    return SharedMemoryConfig::kDefault;
  }

  switch (cuda_config.ValueOrDie()) {
    case CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE:
      return SharedMemoryConfig::kDefault;
    case CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE:
      return SharedMemoryConfig::kFourByte;
    case CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE:
      return SharedMemoryConfig::kEightByte;
    default:
      LOG(FATAL) << "Invalid shared memory configuration returned: "
                 << cuda_config.ValueOrDie();
  }
}

port::Status CUDAExecutor::SetDeviceSharedMemoryConfig(
    SharedMemoryConfig config) {
  CUsharedconfig cuda_config;
  switch (config) {
    case SharedMemoryConfig::kDefault:
      cuda_config = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
      break;
    case SharedMemoryConfig::kFourByte:
      cuda_config = CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE;
      break;
    case SharedMemoryConfig::kEightByte:
      cuda_config = CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE;
      break;
    default:
      LOG(FATAL) << "Invalid shared memory configuration specified: "
                 << static_cast<int>(config);
  }
  return CUDADriver::ContextSetSharedMemConfig(context_, cuda_config);
}

bool CUDAExecutor::DeviceMemoryUsage(int64 *free, int64 *total) const {
  return CUDADriver::GetDeviceMemoryInfo(context_, free, total);
}

bool CUDAExecutor::GetSymbol(const string& symbol_name, void **mem,
                             size_t *bytes) {
  {  // give limited scope to mutex_lock
    mutex_lock lock{in_memory_modules_mu_};
    for (auto &it : gpu_binary_to_module_) {
      CUmodule module = it.second.first;
      CHECK(module != nullptr);
      if (CUDADriver::GetModuleSymbol(context_, module, symbol_name.c_str(),
                                      reinterpret_cast<CUdeviceptr *>(mem),
                                      bytes)) {
        return true;
      }
    }
  }

  LOG(INFO) << "Falied to find symbol in any modules: " << symbol_name;
  return false;
}

bool CUDAExecutor::FillBlockDimLimit(BlockDim *block_dim_limit) const {
  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
  if (!CUDADriver::GetGridLimits(&x, &y, &z, device_)) {
    return false;
  }

  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return true;
}

bool CUDAExecutor::SupportsBlas() const { return true; }

bool CUDAExecutor::SupportsFft() const { return true; }

bool CUDAExecutor::SupportsRng() const { return true; }

std::unique_ptr<internal::EventInterface>
CUDAExecutor::CreateEventImplementation() {
  return std::unique_ptr<internal::EventInterface>(new CUDAEvent(this));
}

std::unique_ptr<internal::KernelInterface>
CUDAExecutor::CreateKernelImplementation() {
  return std::unique_ptr<internal::KernelInterface>(new CUDAKernel());
}

std::unique_ptr<internal::StreamInterface>
CUDAExecutor::GetStreamImplementation() {
  return std::unique_ptr<internal::StreamInterface>(new CUDAStream(this));
}

std::unique_ptr<internal::TimerInterface>
CUDAExecutor::GetTimerImplementation() {
  return std::unique_ptr<internal::TimerInterface>(new CUDATimer(this));
}

void *CUDAExecutor::CudaContextHack() { return context_; }

CudaContext* CUDAExecutor::cuda_context() { return context_; }

// Attempts to read the NUMA node corresponding to the GPU device's PCI bus out
// of SysFS. Returns -1 if it cannot.
//
// For anything more complicated/prod-focused than this, you'll likely want to
// turn to gsys' topology modeling.
static int TryToReadNumaNode(const string &pci_bus_id, int device_ordinal) {
#if defined(__APPLE__)
  LOG(INFO) << "OS X does not support NUMA - returning NUMA node zero";
  return 0;
#elif defined(PLATFORM_WINDOWS)
  // Windows support for NUMA is not currently implemented. Return node 0.
  return 0;
#elif defined(__aarch64__)
  LOG(INFO) << "ARM64 does not support NUMA - returning NUMA node zero";
  return 0;
#else
  VLOG(2) << "trying to read NUMA node for device ordinal: " << device_ordinal;
  static const int kUnknownNumaNode = -1;

  if (pci_bus_id.empty()) {
    LOG(INFO) << "no PCI bus ID for device ordinal: " << device_ordinal;
    return kUnknownNumaNode;
  }

  string filename =
      port::Printf("/sys/bus/pci/devices/%s/numa_node", pci_bus_id.c_str());

  // We have to use fopen/fread here so that the device properties can be
  // populated before InitGoogle procedure has been completed (at which point we
  // could use the file::* utilities).
  FILE *file = fopen(filename.c_str(), "r");
  if (file == nullptr) {
    LOG(ERROR) << "could not open file to read NUMA node: " << filename
               << "\nYour kernel may have been built without NUMA support.";
    return kUnknownNumaNode;
  }

  string content;
  char buf[32];
  size_t did_read = fread(buf, sizeof(buf[0]), sizeof(buf) - 1, file);
  buf[did_read] = '\0';
  content = buf;

  int32 value;
  if (port::safe_strto32(content, &value)) {
    if (value < 0) {  // See http://b/18228951 for details on this path.
      LOG(INFO) << "successful NUMA node read from SysFS had negative value ("
                << value << "), but there must be at least one NUMA node"
                            ", so returning NUMA node zero";
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

// Set of compute capability specific device parameters that cannot be
// queried from the driver API.  These values instead are baked into a
// lookup table indexed by compute capability version.
struct UnqueryableDeviceParams {
  int cc_major;
  int cc_minor;
  uint64 blocks_per_core_limit;
  uint64 registers_per_core_limit;
  uint64 registers_per_thread_limit;
  uint64 warp_alloc_granularity;
  uint64 register_alloc_granularity;
  uint64 shared_memory_alloc_granularity;
};

// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
// https://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls
static const UnqueryableDeviceParams kAllUnqueryableDeviceParams[] = {
    {
        2, 0,       // compute capability (2.0)
        8,          // blocks_per_core_limit
        32 * 1024,  // registers_per_core_limit
        63,         // registers_per_thread_limit
        2,          // warp_alloc_granularity
        64,         // register_alloc_granularity
        128,        // shared_memory_alloc_granularity
    },
    {
        2, 1,       // compute capability (2.1)
        8,          // blocks_per_core_limit
        32 * 1024,  // registers_per_core_limit
        63,         // registers_per_thread_limit
        2,          // warp_alloc_granularity
        64,         // register_alloc_granularity
        128,        // shared_memory_alloc_granularity
    },
    {
        3, 0,       // compute capability (3.0)
        16,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        63,         // registers_per_thread_limit
        4,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
    {
        3, 2,       // compute capability (3.2)
        16,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        255,        // registers_per_thread_limit
        4,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
    {
        3, 5,       // compute capability (3.5)
        16,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        255,        // registers_per_thread_limit
        4,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
    {
        3, 7,        // compute capability (3.7)
        16,          // blocks_per_core_limit
        128 * 1024,  // registers_per_core_limit
        255,         // registers_per_thread_limit
        4,           // warp_alloc_granularity
        256,         // register_alloc_granularity
        256,         // shared_memory_alloc_granularity
    },
    {
        5, 0,       // compute capability (5.0)
        32,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        255,        // registers_per_thread_limit
        4,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
    {
        5, 2,       // compute capability (5.2)
        32,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        255,        // registers_per_thread_limit
        4,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
    {
        5, 3,       // compute capability (5.3)
        32,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        255,        // registers_per_thread_limit
        4,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
    {
        6, 0,       // compute capability (6.0)
        32,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        255,        // registers_per_thread_limit
        2,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
    {
        6, 1,       // compute capability (6.1)
        32,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        255,        // registers_per_thread_limit
        4,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
    {
        6, 2,       // compute capability (6.2)
        32,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        255,        // registers_per_thread_limit
        4,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
    // TODO(jlebar): Confirm the alloc granularity values for sm_70.  These are
    // not published in the spreadsheet linked above.  Currently we guess that
    // they're the same as sm_60.
    {
        7, 0,       // compute capability (7.0)
        32,         // blocks_per_core_limit
        64 * 1024,  // registers_per_core_limit
        255,        // registers_per_thread_limit
        2,          // warp_alloc_granularity
        256,        // register_alloc_granularity
        256,        // shared_memory_alloc_granularity
    },
};

DeviceDescription *CUDAExecutor::PopulateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  {
    int driver_version = 0;
    (void)CUDADriver::GetDriverVersion(&driver_version);
    string augmented_driver_version = port::Printf(
        "%d (%s)", driver_version,
        DriverVersionStatusToString(Diagnostician::FindDsoVersion()).c_str());
    builder.set_driver_version(augmented_driver_version);
  }

  {
    string pci_bus_id = CUDADriver::GetPCIBusID(device_);

    // Lower the hex characters to match sysfs.
    pci_bus_id = port::Lowercase(pci_bus_id);
    builder.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = TryToReadNumaNode(pci_bus_id, device_ordinal_);
    builder.set_numa_node(numa_node);
  }

  CUdevprop prop;
  if (CUDADriver::GetDeviceProperties(&prop, device_ordinal_)) {
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
    (void)CUDADriver::IsEccEnabled(device_, &ecc_enabled);
    builder.set_ecc_enabled(ecc_enabled);
  }

  {
    uint64 device_memory_size = -1;
    (void)CUDADriver::GetDeviceTotalMemory(device_, &device_memory_size);
    builder.set_device_memory_size(device_memory_size);
  }

  port::StatusOr<int> mem_clock_khz = CUDADriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device_ordinal_);
  port::StatusOr<int> mem_bus_width_bits = CUDADriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device_ordinal_);
  if (mem_clock_khz.ok() && mem_bus_width_bits.ok()) {
    // Times 2 because HBM is DDR memory; it gets two data bits per each data
    // lane.
    builder.set_memory_bandwidth(2 * int64_t{mem_clock_khz.ValueOrDie()} *
                                 1000 *
                                 int64_t{mem_bus_width_bits.ValueOrDie()} / 8);
  }

  {
    BlockDim block_dim_limit;
    FillBlockDimLimit(&block_dim_limit);
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    string device_name;
    (void)CUDADriver::GetDeviceName(device_, &device_name);
    builder.set_name(device_name);
  }

  for (size_t i = 0; i < ARRAYSIZE(kAllUnqueryableDeviceParams); i++) {
    const auto &params = kAllUnqueryableDeviceParams[i];
    if (params.cc_major == cc_major_ && params.cc_minor == cc_minor_) {
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
      port::StrCat("Compute Capability ", cc_major_, ".", cc_minor_));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  builder.set_device_address_bits(64);

  builder.set_device_vendor("NVIDIA Corporation");
  builder.set_cuda_compute_capability(cc_major_, cc_minor_);
  builder.set_shared_memory_per_core(
      CUDADriver::GetMaxSharedMemoryPerCore(device_).ValueOrDie());
  builder.set_shared_memory_per_block(
      CUDADriver::GetMaxSharedMemoryPerBlock(device_).ValueOrDie());
  builder.set_core_count(
      CUDADriver::GetMultiprocessorCount(device_).ValueOrDie());
  builder.set_threads_per_core_limit(
      CUDADriver::GetMaxThreadsPerMultiprocessor(device_).ValueOrDie());
  builder.set_registers_per_block_limit(
      CUDADriver::GetMaxRegistersPerBlock(device_).ValueOrDie());
  builder.set_threads_per_warp(
      CUDADriver::GetThreadsPerWarp(device_).ValueOrDie());

  auto built = builder.Build();
  return built.release();
}

}  // namespace cuda

namespace gpu = ::perftools::gputools;

void initialize_cuda_gpu_executor() {
  *gpu::internal::MakeCUDAExecutorImplementation() = [](
      const gpu::PluginConfig &config) {
    return new gpu::cuda::CUDAExecutor{config};
  };
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(
    cuda_gpu_executor, {perftools::gputools::initialize_cuda_gpu_executor();});
