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
#include <unistd.h>

#include <memory>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/gpu/gpu_collectives.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_runtime.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_timer.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/dso_loader.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/rocm/rocm_diagnostics.h"
#include "xla/stream_executor/rocm/rocm_driver.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_interface.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

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
  return static_cast<GpuEvent*>(event);
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
  return static_cast<GpuExecutor*>(stream->parent())->gpu_context();
}

GpuContext* ExtractGpuContext(GpuExecutor* rocm_exec) {
  CHECK(rocm_exec != nullptr);
  return rocm_exec->gpu_context();
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
  CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
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

int fpus_per_core(std::string gcn_arch_name) {
  // Source:
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
  int n = 128;  // gfx90a and gfx908 -> 128
  if (gcn_arch_name.substr(0, 6) == "gfx906") {
    n = 64;
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
    const char* mem_it = nullptr;
    for (auto x : in_memory_modules_) {
      if (x.second == module) mem_it = x.first;
    }
    if (mem_it != nullptr) in_memory_modules_.erase(mem_it);
  }
  return true;
}

void GpuExecutor::UnloadKernel(const Kernel* kernel) {
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

absl::Status GpuExecutor::Init() {
  auto status = GpuDriver::Init();
  if (!status.ok()) {
    return status;
  }

  status = GpuDriver::GetDevice(device_ordinal_, &device_);
  if (!status.ok()) {
    return status;
  }

  status = GpuDriver::CreateContext(device_ordinal_, device_, &context_);
  if (!status.ok()) {
    return status;
  }

  return GpuDriver::GetGpuISAVersion(&version_, device_);
}

// Returns the path to the running executable.
// N.B. Derived from //knowledge/smalltalk/background_kb.cc
// Arg: strip_exe: if true, remove the name of the executable itself from the
//                 returned string. Example: calling this from /usr/bin/foo
//                 would return /usr/bin.
static std::string GetBinaryDir(bool strip_exe) {
  char exe_path[PATH_MAX] = {0};
  CHECK_NE(readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1), -1);
  // Make sure it's null-terminated:
  exe_path[sizeof(exe_path) - 1] = 0;

  if (strip_exe) {
    // The exe is the last component of the path, so remove one component.
    std::string ret = exe_path;
    std::vector<std::string> components = absl::StrSplit(exe_path, '/');
    components.pop_back();
    return absl::StrJoin(components, "/");
  }
  return exe_path;
}

absl::Status GpuExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                    Kernel* kernel) {
  GpuKernel* rocm_kernel = AsGpuKernel(kernel);
  hipModule_t module = nullptr;
  const std::string* kernel_name;

  if (spec.has_cuda_cubin_in_memory()) {
    kernel_name = &spec.cuda_cubin_in_memory().kernel_name();

    const char* hsaco = reinterpret_cast<const char*>(
        spec.cuda_cubin_in_memory().cubin_bytes().data());
    absl::MutexLock lock{&in_memory_modules_mu_};
    module = in_memory_modules_[hsaco];

    if (module == nullptr) {
      TF_RETURN_IF_ERROR(GpuDriver::LoadHsaco(context_, hsaco, &module));
    }
    kernel_to_gpu_binary_[kernel] = hsaco;
  } else if (spec.has_in_process_symbol()) {
    kernel_name = &spec.in_process_symbol().kernel_name();
    void* symbol = spec.in_process_symbol().symbol();

    VLOG(1) << "Resolve ROCM kernel " << *kernel_name
            << " from symbol pointer: " << symbol;

    *rocm_kernel->gpu_function_ptr() =
        static_cast<hipFunction_t>(spec.in_process_symbol().symbol());
  } else {
    return absl::InternalError("No method of loading ROCM kernel provided");
  }

  // If we resolved kernel from a symbol pointer, there is no need to load it
  // from a module, as ROCm runtime did that automatically for us.
  if (!spec.has_in_process_symbol()) {
    VLOG(2) << "getting function " << *kernel_name << " from module " << module;
    TF_RETURN_IF_ERROR(
        GpuDriver::GetModuleFunction(context_, module, kernel_name->c_str(),
                                     rocm_kernel->gpu_function_ptr()));
  }

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the ROCM API.
  rocm_kernel->set_arity(spec.arity());

  // unable to get kernel metadata for in-process kernel
  if (!spec.has_in_process_symbol()) {
    KernelMetadata kernel_metadata;
    TF_RETURN_IF_ERROR(GetKernelMetadata(rocm_kernel, &kernel_metadata));
    kernel->set_metadata(kernel_metadata);
  }
  kernel->set_name(*kernel_name);
  kernel->set_args_packing(spec.kernel_args_packing());
  return absl::OkStatus();
}

absl::Status GpuExecutor::GetKernelMetadata(GpuKernel* rocm_kernel,
                                            KernelMetadata* kernel_metadata) {
  int value = 0;
  TF_RETURN_IF_ERROR(GpuDriver::FuncGetAttribute(
      HIP_FUNC_ATTRIBUTE_NUM_REGS, *rocm_kernel->gpu_function_ptr(), &value));
  kernel_metadata->set_registers_per_thread(value);

  TF_RETURN_IF_ERROR(
      GpuDriver::FuncGetAttribute(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                  *rocm_kernel->gpu_function_ptr(), &value));
  kernel_metadata->set_shared_memory_bytes(value);
  return absl::OkStatus();
}

absl::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const Kernel& kernel, const KernelArgs& args) {
  GpuStreamHandle hipstream = AsGpuStreamValue(stream);
  const GpuKernel* rocm_kernel = AsGpuKernel(&kernel);
  hipFunction_t hipfunc = rocm_kernel->AsGpuFunctionHandle();

  // Only perform/print the occupancy check once.  Even just checking to see
  // whether we've done an occupancy check on this kernel before isn't free
  // (because we have to synchronize), so we only do this at -v 2+.
  if (VLOG_IS_ON(2)) {
    absl::MutexLock lock(&launched_kernels_mu_);
    if (!launched_kernels_.count(hipfunc)) {
      VlogOccupancyInfo(stream->parent()->GetDeviceDescription(), kernel,
                        thread_dims, block_dims);
      // TODO(rspringer): Remove elements from launched_kernels_...if we ever
      // expose a kernel/module deallocation method.
      launched_kernels_.insert(hipfunc);
    }
  }

  if (rocm_kernel->cache_config() != KernelCacheConfig::kNoPreference) {
    TF_RETURN_IF_ERROR(GpuDriver::FuncSetCacheConfig(
        hipfunc, rocm_kernel->GetGpuCacheConfig()));
  }

  auto launch = [&](const KernelArgsPackedArrayBase& packed) {
    CHECK_EQ(kernel.Arity() + (args.number_of_shared_bytes() > 0),
             packed.number_of_arguments());

    void** kernel_params =
        const_cast<void**>(packed.argument_addresses().data());

    return GpuDriver::LaunchKernel(
        GetGpuContext(stream), kernel.name(), hipfunc, block_dims.x,
        block_dims.y, block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
        args.number_of_shared_bytes(), hipstream, kernel_params, nullptr);
  };

  auto* packed_args = DynCast<KernelArgsPackedArrayBase>(&args);
  if (packed_args) return launch(*packed_args);

  if (auto* device_mem = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
    auto& pack = kernel.args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    TF_ASSIGN_OR_RETURN(auto packed_args, pack(kernel, *device_mem));
    return launch(*packed_args);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

absl::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const ClusterDim& cluster_dims,
                                 const Kernel& kernel, const KernelArgs& args) {
  if (cluster_dims.x != 1 || cluster_dims.y != 1 || cluster_dims.z != 1)
    return absl::UnimplementedError("Not implemented for ROCm");
  return Launch(stream, thread_dims, block_dims, kernel, args);
}

absl::Status GpuExecutor::Submit(Stream* stream,
                                 const CommandBuffer& command_buffer) {
  if (command_buffer.mode() != CommandBuffer::Mode::kPrimary) {
    return absl::InvalidArgumentError(
        "Can't submit non-primary command buffer for execution");
  }

  auto exec = GpuCommandBuffer::Cast(&command_buffer)->executable();
  VLOG(3) << "Launch command buffer execuable graph " << exec
          << " on a stream: " << stream;
  return GpuDriver::GraphLaunch(exec, AsGpuStreamValue(stream));
}

absl::Status GpuExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
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

absl::Status GpuExecutor::LoadModuleFromCuBin(const char* cubin,
                                              hipModule_t* module) {
  LOG(FATAL) << "Feature not supported on ROCM platform (LoadModuleFromCuBin)";
}

absl::Status GpuExecutor::LoadModuleFromPtx(const char* ptx,
                                            hipModule_t* module) {
  LOG(FATAL) << "Feature not supported on ROCM platform (LoadModuleFromPtx)";
}

absl::Status GpuExecutor::LoadModuleFromHsaco(const char* hsaco,
                                              hipModule_t* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[hsaco];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(GpuDriver::LoadHsaco(context_, hsaco, module));
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

// This is a non-essential operation; if there's a failure, proceed without
// logging an error. It's nearly certain that in case of failures, we'd never
// get here in the first place; these are very low-impact routines.
void GpuExecutor::VlogOccupancyInfo(const DeviceDescription& device_description,
                                    const Kernel& kernel,
                                    const ThreadDim& thread_dims,
                                    const BlockDim& block_dims) {
  VLOG(2) << "Computing kernel occupancy for kernel "
          << kernel.demangled_name();
  VLOG(2) << "Thread dimensions (" << thread_dims.x << ", " << thread_dims.y
          << ", " << thread_dims.z << ")";

  auto regs_per_thread = kernel.metadata().registers_per_thread();
  auto smem_per_block = kernel.metadata().shared_memory_bytes();

  if (!regs_per_thread && !smem_per_block) {
    return;
  }

  const GpuKernel* rocm_kernel = AsGpuKernel(&kernel);
  auto hipfunc = rocm_kernel->AsGpuFunctionHandle();

  int blocks_per_sm = CalculateOccupancy(device_description, *regs_per_thread,
                                         *smem_per_block, thread_dims, hipfunc);
  VLOG(2) << "Resident blocks per SM is " << blocks_per_sm;

  int suggested_threads =
      CompareOccupancy(&blocks_per_sm, device_description, *regs_per_thread,
                       *smem_per_block, thread_dims, hipfunc);
  if (suggested_threads != 0) {
    VLOG(2) << "The rocm occupancy calculator recommends using "
            << suggested_threads
            << " threads per block to achieve an occupancy of " << blocks_per_sm
            << " blocks per SM.";
  }
}

// Compute and return maximum blocks per core (occupancy) based on the
// device description, some kernel characteristics and the number of threads per
// block.  If unable to compute occupancy, zero is returned.
int GpuExecutor::CalculateOccupancy(const DeviceDescription& device_description,
                                    uint64_t registers_per_thread,
                                    uint64_t shared_memory_per_block,
                                    const ThreadDim& thread_dims,
                                    GpuFunctionHandle func) {
  int suggested_blocks = 0;
  int suggested_threads = 0;
  (void)rocm::OccupancyGetMaxPotentialBlockSize(
      &suggested_blocks, &suggested_threads, func, shared_memory_per_block, 0);
  return suggested_blocks;
}

// Compute and return the suggested thread count to achieve ideal occupancy.
// If the provided thread dimensions match this number, zero is returned.
int GpuExecutor::CompareOccupancy(int* initial_blocks,
                                  const DeviceDescription& device_description,
                                  uint64_t registers_per_thread,
                                  uint64_t shared_memory_per_block,
                                  const ThreadDim& thread_dims,
                                  GpuFunctionHandle func) {
  int suggested_blocks = 0;
  int suggested_threads = 0;
  (void)rocm::OccupancyGetMaxPotentialBlockSize(
      &suggested_blocks, &suggested_threads, func, shared_memory_per_block, 0);
  if (suggested_blocks > *initial_blocks) {
    *initial_blocks = suggested_blocks;
    return suggested_threads;
  } else {
    return 0;
  }
}

DeviceMemoryBase GpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
  if (memory_space ==
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
        context_, AsROCmDevicePtr(location), 0x0, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsROCmDevicePtr(location),
                                           0x0, size);
}

absl::Status GpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src,
                                            uint64_t size) {
  return GpuDriver::SynchronousMemcpyH2D(context_, AsROCmDevicePtr(gpu_dst),
                                         host_src, size);
}

absl::Status GpuExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64_t size) {
  return GpuDriver::SynchronousMemcpyD2H(context_, host_dst,
                                         AsROCmDevicePtr(gpu_src), size);
}

absl::Status GpuExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                  uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return Memset32(stream, location, 0x0, size);
  } else {
    return Memset(stream, location, 0x0, size);
  }
}

absl::Status GpuExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                 uint8 pattern, uint64_t size) {
  VLOG(2) << "enqueueing memset8 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  return GpuDriver::AsynchronousMemsetUint8(context_, AsROCmDevicePtr(location),
                                            pattern, size,
                                            AsGpuStreamValue(stream));
}

absl::Status GpuExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                   uint32 pattern, uint64_t size) {
  VLOG(2) << "enqueueing memset32 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
        size % 4 == 0);
  return GpuDriver::AsynchronousMemsetUint32(
      context_, AsROCmDevicePtr(location), pattern, size / 4,
      AsGpuStreamValue(stream));
}

absl::Status GpuExecutor::Memcpy(Stream* stream, void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) {
  bool ok = GpuDriver::AsynchronousMemcpyD2H(context_, host_dst,
                                             AsROCmDevicePtr(gpu_src), size,
                                             AsGpuStreamValue(stream));

  // TODO(b/326130105): Change AsynchronousMemcpyD2H calls to return
  // absl::Status.
  if (!ok) {
    return absl::InternalError("Failed to memcpy from device to host.");
  }
  return absl::OkStatus();
}

absl::Status GpuExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) {
  bool ok = GpuDriver::AsynchronousMemcpyH2D(context_, AsROCmDevicePtr(gpu_dst),
                                             host_src, size,
                                             AsGpuStreamValue(stream));
  // TODO(b/326130105): Change AsynchronousMemcpyD2H calls to return
  // absl::Status.
  if (!ok) {
    return absl::InternalError("Failed to memcpy from device to host.");
  }
  return absl::OkStatus();
}

bool GpuExecutor::MemcpyDeviceToDevice(Stream* stream,
                                       DeviceMemoryBase* gpu_dst,
                                       const DeviceMemoryBase& gpu_src,
                                       uint64_t size) {
  return GpuDriver::AsynchronousMemcpyD2D(context_, AsROCmDevicePtr(gpu_dst),
                                          AsROCmDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::HostCallback(Stream* stream,
                               absl::AnyInvocable<absl::Status() &&> callback) {
  auto callback_ptr =
      new absl::AnyInvocable<void() &&>([cb = std::move(callback)]() mutable {
        absl::Status s = std::move(cb)();
        if (!s.ok()) {
          LOG(WARNING) << "Host callback failed: " << s;
        }
      });
  return GpuDriver::AddStreamCallback(context_, AsGpuStreamValue(stream),
                                      InternalHostCallback, callback_ptr);
}

/* static */ void GpuExecutor::InternalHostCallback(void* data) {
  auto* callback = reinterpret_cast<absl::AnyInvocable<void() &&>*>(data);
  std::move (*callback)();
  delete callback;
}

absl::Status GpuExecutor::RecordEvent(Stream* stream, Event* event) {
  return AsGpuEvent(event)->Record(AsGpuStream(stream));
}

absl::Status GpuExecutor::WaitForEvent(Stream* stream, Event* event) {
  if (GpuDriver::WaitStreamOnEvent(context_, AsGpuStream(stream)->gpu_stream(),
                                   AsGpuEvent(event)->gpu_event())) {
    return absl::OkStatus();
  } else {
    return absl::Status{
        absl::StatusCode::kInternal,
        absl::StrFormat("error recording waiting for ROCM event on stream %p",
                        stream)};
  }
}

void GpuExecutor::DeallocateStream(Stream* stream) {
  {
    absl::MutexLock lock(&mu_);
    if (dnn_ != nullptr) {
      dnn_->NotifyStreamDestroyed(stream);
    }
  }
  GpuStream* rocm_stream = AsGpuStream(stream);
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(rocm_stream->platform_specific_stream());
  if (!rocm_stream->IsIdle()) {
    LOG(ERROR) << "Deallocating stream with pending work";
  }
  rocm_stream->Destroy();
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

dnn::DnnSupport* GpuExecutor::AsDnn() {
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

fft::FftSupport* GpuExecutor::AsFft() {
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

bool GpuExecutor::CanEnablePeerAccessTo(StreamExecutorInterface* other) {
  GpuExecutor* rocm_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::CanEnablePeerAccess(context_, rocm_other->context_);
}

absl::Status GpuExecutor::EnablePeerAccessTo(StreamExecutorInterface* other) {
  GpuExecutor* rocm_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::EnablePeerAccess(context_, rocm_other->context_);
}

bool GpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  return GpuDriver::GetDeviceMemoryInfo(context_, free, total);
}

absl::StatusOr<DeviceMemoryBase> GpuExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  void* mem = nullptr;
  size_t bytes = 0;

  absl::MutexLock lock{&in_memory_modules_mu_};
  if (static_cast<bool>(module_handle)) {
    auto it = gpu_binary_to_module_.find(module_handle.id());
    CHECK(it != gpu_binary_to_module_.end());
    if (GpuDriver::GetModuleSymbol(
            context_, it->second.first, symbol_name.c_str(),
            reinterpret_cast<hipDeviceptr_t*>(&mem), &bytes)) {
      return DeviceMemoryBase(mem, bytes);
    }
  }

  for (auto& it : gpu_binary_to_module_) {
    if (GpuDriver::GetModuleSymbol(
            context_, it.second.first, symbol_name.c_str(),
            reinterpret_cast<hipDeviceptr_t*>(&mem), &bytes)) {
      return DeviceMemoryBase(mem, bytes);
    }
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

absl::StatusOr<std::unique_ptr<Event>> GpuExecutor::CreateEvent() {
  auto gpu_event = std::make_unique<RocmEvent>(this);
  TF_RETURN_IF_ERROR(gpu_event->Init());
  return std::move(gpu_event);
}

absl::StatusOr<std::unique_ptr<Stream>> GpuExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  auto gpu_stream = std::make_unique<GpuStream>(this);
  if (priority.has_value()) {
    if (std::holds_alternative<StreamPriority>(*priority)) {
      gpu_stream->SetPriority(std::get<StreamPriority>(*priority));
    } else {
      gpu_stream->SetPriority(std::get<int>(*priority));
    }
  }
  absl::MutexLock l(&alive_gpu_streams_mu_);
  bool init_worked = gpu_stream->Init();
  if (init_worked) {
    auto platform_specific_stream = gpu_stream->platform_specific_stream();
    alive_gpu_streams_[platform_specific_stream] = gpu_stream.get();
    return std::move(gpu_stream);
  } else {
    return absl::InvalidArgumentError("Failed to initialize GPU stream");
  }
}

absl::StatusOr<std::unique_ptr<Kernel>> GpuExecutor::CreateKernel() {
  return std::make_unique<GpuKernel>(this);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>> GpuExecutor::CreateCommandBuffer(
    CommandBuffer::Mode mode) {
  VLOG(2) << "Create ROCm command buffer (ROCm graph)";
  GpuGraphHandle graph = nullptr;
  TF_RETURN_IF_ERROR(GpuDriver::CreateGraph(&graph));
  return std::make_unique<GpuCommandBuffer>(mode, /*parent=*/this, graph);
}

std::unique_ptr<GpuCommandBuffer> GpuExecutor::CreateCommandBuffer(
    CommandBuffer::Mode mode, GpuGraphHandle graph, bool is_owned_graph) {
  VLOG(2) << "Create HIP command buffer (HIP graph) from existing graph "
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
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
GpuExecutor::CreateDeviceDescription(int device_ordinal) {
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

  internal::DeviceDescriptionBuilder builder;

  {
    int version = GpuDriver::GetDriverVersion().value_or(-1);
    std::string augmented_driver_version = absl::StrFormat(
        "%d (%s)", version,
        rocm::DriverVersionStatusToString(Diagnostician::FindDsoVersion())
            .c_str());
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

  hipDeviceProp_t prop;
  if (GpuDriver::GetDeviceProperties(&prop, device_ordinal)) {
    builder.set_threads_per_block_limit(prop.maxThreadsPerBlock);

    ThreadDim thread_dim_limit;
    thread_dim_limit.x = prop.maxThreadsDim[0];
    thread_dim_limit.y = prop.maxThreadsDim[1];
    thread_dim_limit.z = prop.maxThreadsDim[2];
    builder.set_thread_dim_limit(thread_dim_limit);

    float clock_rate_ghz = static_cast<float>(prop.clockRate) / 1e6;
    builder.set_clock_rate_ghz(clock_rate_ghz);

    // mem_bandwidth = 2 * mem_bus_width_in_bytes * mem_clock_rate_in_hz
    int64_t memory_bandwidth = 2 * (int64_t(prop.memoryBusWidth) / 8) *
                               (int64_t(prop.memoryClockRate) * 1000);
    builder.set_memory_bandwidth(memory_bandwidth);

    builder.set_l2_cache_size(prop.l2CacheSize);
  }

  {
    bool ecc_enabled = false;
    (void)GpuDriver::IsEccEnabled(device, &ecc_enabled);
    builder.set_ecc_enabled(ecc_enabled);
  }

  uint64_t device_memory_size = -1;
  (void)GpuDriver::GetDeviceTotalMemory(device, &device_memory_size);
  builder.set_device_memory_size(device_memory_size);

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
      absl::StrCat("AMDGPU ISA version: ", gcn_arch_name));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  builder.set_device_address_bits(64);

  builder.set_device_vendor("Advanced Micro Devices, Inc");
  builder.set_rocm_compute_capability(gcn_arch_name);

  builder.set_shared_memory_per_core(
      GpuDriver::GetMaxSharedMemoryPerCore(device).value());
  builder.set_shared_memory_per_block(
      GpuDriver::GetMaxSharedMemoryPerBlock(device).value());
  int core_count = GpuDriver::GetMultiprocessorCount(device).value();
  builder.set_core_count(core_count);
  builder.set_fpus_per_core(fpus_per_core(gcn_arch_name));
  builder.set_threads_per_core_limit(
      GpuDriver::GetMaxThreadsPerMultiprocessor(device).value());
  builder.set_registers_per_block_limit(
      GpuDriver::GetMaxRegistersPerBlock(device).value());
  builder.set_threads_per_warp(GpuDriver::GetThreadsPerWarp(device).value());
  builder.set_registers_per_core_limit(64 * 1024);

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
  builder.set_model_str(absl::StrFormat("cc_%d.%d with %dB RAM, %d cores",
                                        cc_major, cc_minor, device_memory_size,
                                        core_count));

  return builder.Build();
}

}  // namespace gpu

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(rocm_executor, {});
