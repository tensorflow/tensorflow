/* Copyright 2015 Google Inc. All Rights Reserved.

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
#include <unistd.h>

#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_event.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_timer.h"
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

#ifdef PLATFORMS_GPUS_CUDA_DYNAMIC_LIBCUDA_DYNAMIC_LIBCUDA_H_
#error \
    "No driver calls in this file, wrap driver functionality in cuda_driver.cc."
#endif

#ifdef __CUDA_RUNTIME_H__
#error \
    "CUDA runtime being included into CUDA GPU executor; should be driver only."
#endif

extern bool FLAGS_check_gpu_leaks;
tensorflow::int32 FLAGS_register_occupancy_warning_threshold;
bool FLAGS_prefer_cubin_to_ptx = true;

namespace perftools {
namespace gputools {
namespace rng {
class RngSupport;
}  // namespace rng
}  // namespace gputools
}  // namespace perftools

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

static CudaContext* GetCudaContext(Stream *stream) {
  return static_cast<CUDAExecutor *>(stream->parent()->implementation())
      ->cuda_context();
}

CudaContext* ExtractCudaContext(CUDAExecutor *cuda_exec) {
  CHECK(cuda_exec != nullptr);
  return cuda_exec->cuda_context();
}

CUDAExecutor *ExtractCudaExecutor(StreamExecutor *stream_exec) {
  return static_cast<CUDAExecutor *>(stream_exec->implementation());
}

CUDAExecutor::~CUDAExecutor() {
  for (auto &it : disk_modules_) {
    CUDADriver::UnloadModule(context_, it.second);
  }
  for (auto &it : in_memory_modules_) {
    CUDADriver::UnloadModule(context_, it.second);
  }
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

  // TODO(22689637): Eliminate unnecessary ToString()s when all dependencies
  // have been migrated.
  string cc_specific = port::StrCat(filename.ToString(), ".cc", cc_major_,
                                    cc_minor_, canonical_suffix.ToString());
  if (port::FileExists(cc_specific)) {
    VLOG(2) << "found compute-capability-specific file, using that: "
            << cc_specific;
    *found_filename = cc_specific;
    return true;
  }

  VLOG(2) << "could not find compute-capability specific file at: "
          << cc_specific;
  if (port::FileExists(filename.ToString())) {
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
    CHECK_ERR(readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1));
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
  CUmodule module = nullptr;
  const string *kernelname;

  const OnDiskKernelLoaderSpec *on_disk_spec = nullptr;
  bool has_ptx = spec.has_cuda_ptx_on_disk();
  bool has_cubin = spec.has_cuda_cubin_on_disk();
  if (has_cubin && (!has_ptx || FLAGS_prefer_cubin_to_ptx)) {
    on_disk_spec = &spec.cuda_cubin_on_disk();
  } else if (has_ptx) {
    on_disk_spec = &spec.cuda_ptx_on_disk();
  }

  if (on_disk_spec != nullptr) {
  } else if (spec.has_cuda_ptx_in_memory()) {
    kernelname = &spec.cuda_ptx_in_memory().kernelname();

    if (cc_major_ == 0 && cc_minor_ == 0) {
      return false;
    }

    // Note that the orignal ptx may be compressed, and the ptx we get below is
    // the decompressed result. To cache the module we should use the original
    // ptx (compressed one) as the key. This is because for the same compressed
    // ptx, we may get different decompressed ptx wrt the pointer value.
    const char *ptx = spec.cuda_ptx_in_memory().text(cc_major_, cc_minor_);
    const char *orig_ptx =
        spec.cuda_ptx_in_memory().original_text(cc_major_, cc_minor_);
    if (ptx == nullptr || orig_ptx == nullptr) {
      ptx = spec.cuda_ptx_in_memory().default_text();
      orig_ptx = spec.cuda_ptx_in_memory().original_default_text();
    }
    if (ptx == nullptr || orig_ptx == nullptr) {
      LOG(FATAL) << "could not load ptx for kernel " << kernelname;
      return false;
    }

    mutex_lock lock{in_memory_modules_mu_};
    module = in_memory_modules_[orig_ptx];

    if (module == nullptr) {
      if (g_cubinate == nullptr) {
        if (!CUDADriver::LoadPtx(context_, ptx, &module)) {
          return false;
        }
      } else {
        string cubin = g_cubinate(ptx);
        auto load_status =
            CUDADriver::LoadCubin(context_, cubin.c_str(), &module);
        if (!load_status.ok()) {
          LOG(ERROR) << "failed to load cubin via hook: " << load_status;
          return false;
        }
      }
      in_memory_modules_[orig_ptx] = module;
    }
  } else if (spec.has_cuda_cubin_in_memory()) {
    kernelname = &spec.cuda_cubin_in_memory().kernelname();
    const char *cubin = spec.cuda_cubin_in_memory().bytes();
    mutex_lock lock{in_memory_modules_mu_};
    module = in_memory_modules_[cubin];

    if (module == nullptr) {
      auto load_status = CUDADriver::LoadCubin(context_, cubin, &module);
      if (!load_status.ok()) {
        LOG(ERROR) << "failed to load CUBIN: " << load_status;
        return false;
      }

      in_memory_modules_[cubin] = module;
    }
  } else {
    LOG(WARNING) << "no method of loading CUDA kernel provided";
    return false;
  }

  VLOG(2) << "getting function " << kernelname << " from module " << module;
  if (!CUDADriver::GetModuleFunction(context_, module, kernelname->c_str(),
                                     cuda_kernel->cuda_function_ptr())) {
    return false;
  }

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the CUDA API.
  cuda_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
  if (!GetKernelMetadata(cuda_kernel, &kernel_metadata)) {
    LOG(WARNING) << "Unable to get metadata for kernel " << kernelname;
  }
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(*kernelname);
  return true;
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
                          const std::vector<KernelArg> &args) {
  CHECK_EQ(kernel.Arity(), args.size());
  CUstream custream = AsCUDAStreamValue(stream);
  const CUDAKernel *cuda_kernel = AsCUDAKernel(&kernel);
  CUfunction cufunc = cuda_kernel->AsCUDAFunctionValue();

  std::vector<void *> addrs;
  addrs.reserve(args.size());
  int shmem_bytes = 0;
  for (size_t i = 0; i < args.size(); i++) {
    switch (args[i].type) {
      case KernelArg::kNormal:
        addrs.push_back(const_cast<void *>(
            static_cast<const void *>(args[i].data.begin())));
        break;
      case KernelArg::kSharedMemory:
        shmem_bytes += args[i].bytes;
        break;
      default:
        LOG(ERROR) << "Invalid kernel arg type passed (" << args[i].type
                   << ") for arg " << i;
        return false;
    }
  }

  // Only perform/print the occupancy check 1x.
  launched_kernels_mu_.lock();
  if (launched_kernels_.find(cufunc) == launched_kernels_.end()) {
    OccupancyCheck(kernel, thread_dims, block_dims);
    // TODO(rspringer): Remove elements from launched_kernels_...if we ever
    // expose a kernel/module deallocation method.
    launched_kernels_.insert(cufunc);
  }
  launched_kernels_mu_.unlock();

  if (cuda_kernel->GetPreferredCacheConfig() !=
      KernelCacheConfig::kNoPreference) {
    CUDADriver::FuncSetCacheConfig(cufunc, cuda_kernel->GetCUDACacheConfig());
  }

  if (!CUDADriver::LaunchKernel(
          GetCudaContext(stream), cufunc, block_dims.x, block_dims.y,
          block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
          shmem_bytes, custream, addrs.data(), nullptr /* = extra */)) {
    LOG(ERROR) << "failed to launch CUDA kernel with args: " << args.size()
               << "; thread dim: " << thread_dims.ToString()
               << "; block dim: " << block_dims.ToString();
    return false;
  }

  return true;
}

// This is a non-essential operation; if there's a failure, proceed without
// logging an error. It's nearly certain that in case of failures, we'd never
// get here in the first place; these are very low-impact routines.
void CUDAExecutor::OccupancyCheck(const KernelBase &kernel,
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

    uint64 reg_reduction = regs_per_thread - improved_regs_per_thread;
    if (reg_reduction <=
        static_cast<uint64>(FLAGS_register_occupancy_warning_threshold)) {
      LOG(INFO) << "Notice: occupancy would increase if register usage was"
                << " reduced from " << regs_per_thread
                << " to " << improved_regs_per_thread
                << " registers per thread for kernel: "
                << kernel.demangled_name();
    }
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

bool CUDAExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                     const void *host_src, uint64 size) {
  return CUDADriver::SynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                          host_src, size);
}

bool CUDAExecutor::SynchronousMemcpy(void *host_dst,
                                     const DeviceMemoryBase &gpu_src,
                                     uint64 size) {
  return CUDADriver::SynchronousMemcpyD2H(context_, host_dst,
                                          AsCudaDevicePtr(gpu_src), size);
}

bool CUDAExecutor::SynchronousMemcpyDeviceToDevice(
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

bool CUDAExecutor::BlockHostUntilDone(Stream *stream) {
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
    mutex_lock lock{disk_modules_mu_};
    for (auto &it : disk_modules_) {
      if (CUDADriver::GetModuleSymbol(context_, it.second, symbol_name.c_str(),
                                      reinterpret_cast<CUdeviceptr *>(mem),
                                      bytes)) {
        return true;
      }
    }
  }

  {  // give limited scope to mutex_lock
    mutex_lock lock{in_memory_modules_mu_};
    for (auto &it : in_memory_modules_) {
      if (CUDADriver::GetModuleSymbol(context_, it.second, symbol_name.c_str(),
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

KernelArg CUDAExecutor::DeviceMemoryToKernelArg(
    const DeviceMemoryBase &gpu_mem) const {
  const void* arg = gpu_mem.opaque();
  const uint8 *arg_ptr = reinterpret_cast<const uint8 *>(&arg);

  KernelArg kernel_arg;
  kernel_arg.type = KernelArg::kNormal;
  kernel_arg.data = port::InlinedVector<uint8, 4>(arg_ptr, arg_ptr + sizeof(arg));
  kernel_arg.bytes = sizeof(arg);
  return kernel_arg;
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

// Attemps to read the NUMA node corresponding to the GPU device's PCI bus out
// of SysFS. Returns -1 if it cannot.
//
// For anything more complicated/prod-focused than this, you'll likely want to
// turn to gsys' topology modeling.
static int TryToReadNumaNode(const string &pci_bus_id, int device_ordinal) {
#if defined(__APPLE__)
  LOG(INFO) << "OS X does not support NUMA - returning NUMA node zero";
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
    LOG(ERROR) << "could not open file to read NUMA node: " << filename;
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
      return 0;
    }
    return value;
  }

  LOG(WARNING)
      << "could not convert SysFS file contents to integral NUMA node value: "
      << content;

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

static const UnqueryableDeviceParams kAllUnqueryableDeviceParams[] = {
  {
    3, 5,       // compute capability (3.5)
    16,         // blocks_per_core_limit
    64 * 1024,  // registers_per_core_limit
    255,        // registers_per_thread_limit
    4,          // warp_alloc_granularity
    256,        // register_alloc_granularity
    256         // shared_memory_alloc_granularity
  }
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
  port::StatusOr<void *> status =
      gpu::internal::CachedDsoLoader::GetLibcudaDsoHandle();
  if (!status.ok()) {
    gpu::cuda::Diagnostician::LogDriverVersionInformation();
    LOG(INFO) << "LD_LIBRARY_PATH: " << getenv("LD_LIBRARY_PATH");
    LOG(INFO) << "failed to find libcuda.so on this system: "
              << status.status();
  }

  // TODO(b/22689637): Temporary until users are migrated off of PlatformKind.
  gpu::PluginRegistry::Instance()->MapPlatformKindToId(
      gpu::PlatformKind::kCuda, gpu::cuda::kCudaPlatformId);

  *gpu::internal::MakeCUDAExecutorImplementation() = [](
      const gpu::PluginConfig &config) {
    return new gpu::cuda::CUDAExecutor{config};
  };
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(
    cuda_gpu_executor, {perftools::gputools::initialize_cuda_gpu_executor();});
