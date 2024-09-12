/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_EXECUTOR_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/gpu_collectives.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/host_memory_allocation.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// This class implements GpuExecutor for NVIDIA GPUs that use CUDA libraries.
class CudaExecutor : public GpuExecutor {
 public:
  CudaExecutor(Platform* platform, int device_ordinal)
      : GpuExecutor(platform, device_ordinal) {}
  ~CudaExecutor() override;
  absl::Status Init() override;
  bool SynchronizeAllActivity() override;

  absl::StatusOr<void*> CollectiveMemoryAllocate(uint64_t size) override {
    return GpuCollectives::CollectiveMemoryAllocate(gpu_context(), size);
  }

  absl::Status CollectiveMemoryDeallocate(void* location) override {
    return GpuCollectives::CollectiveMemoryDeallocate(gpu_context(), location);
  }

  absl::StatusOr<std::unique_ptr<EventBasedTimer>> CreateEventBasedTimer(
      GpuStream* stream, bool use_delay_kernel) override;
  absl::StatusOr<DeviceMemoryBase> GetSymbol(
      const std::string& symbol_name, ModuleHandle module_handle) override;
  absl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override;
  absl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override;
  void DeallocateStream(Stream* stream) override;
  absl::Status BlockHostUntilDone(Stream* stream) override;
  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(StreamExecutor* other) override;
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;
  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const MultiKernelLoaderSpec& spec) override;
  void UnloadKernel(const Kernel* kernel) override;
  absl::Status LoadModule(const MultiModuleLoaderSpec& spec,
                          ModuleHandle* module_handle) override;
  bool UnloadModule(ModuleHandle module_handle) override;
  absl::StatusOr<std::shared_ptr<DeviceMemoryBase>> CreateOrShareConstant(
      Stream* stream, absl::Span<const uint8_t> content) override;
  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceMemoryBase* mem) override;
  blas::BlasSupport* AsBlas() override;
  fft::FftSupport* AsFft() override;
  dnn::DnnSupport* AsDnn() override;
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;
  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;
  absl::StatusOr<std::unique_ptr<CommandBuffer>> CreateCommandBuffer(
      CommandBuffer::Mode mode) override;
  absl::Status TrimGraphMemory() override;
  int cc_major() const { return cc_major_; }
  int cc_minor() const { return cc_minor_; }

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return GpuExecutor::CreateDeviceDescription(device_ordinal());
  }
  void* UnifiedMemoryAllocate(uint64_t size) override {
    return GpuDriver::UnifiedMemoryAllocate(gpu_context(), size);
  }

  void UnifiedMemoryDeallocate(void* location) override {
    return GpuDriver::UnifiedMemoryDeallocate(gpu_context(), location);
  }
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override {
    auto* buffer = GpuDriver::HostAllocate(gpu_context(), size);
    if (buffer == nullptr && size > 0) {
      return absl::InternalError(
          absl::StrFormat("Failed to allocate HostMemory of size %d", size));
    }
    return std::make_unique<HostMemoryAllocation>(buffer, size, this);
  }

  void HostMemoryDeallocate(void* location) override {
    return GpuDriver::HostDeallocate(gpu_context(), location);
  }

  absl::StatusOr<MemoryType> GetPointerMemorySpace(const void* ptr) override {
    return GpuDriver::GetPointerMemorySpace(
        reinterpret_cast<GpuDevicePtr>(const_cast<void*>(ptr)));
  }

  Stream* FindAllocatedStream(void* gpu_stream) override {
    absl::MutexLock lock(&alive_gpu_streams_mu_);
    auto it = alive_gpu_streams_.find(gpu_stream);
    if (it == alive_gpu_streams_.end()) {
      return nullptr;
    }
    return it->second;
  }

 private:
  // Collects metadata for the specified kernel.
  absl::Status GetKernelMetadata(GpuKernel* cuda_kernel,
                                 KernelMetadata* kernel_metadata);

  // (supported on CUDA only)
  absl::Status LoadModuleFromCuBin(const char* cubin, GpuModuleHandle* module)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // Loads the PTX text `ptx` as a CUDA module.  `ptx` must be null terminated.
  // (supported on CUDA only)
  absl::Status LoadModuleFromPtx(const char* ptx, GpuModuleHandle* module)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // (supported on ROCm only)
  absl::Status LoadModuleFromHsaco(const char* hsaco, GpuModuleHandle* module)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  bool UnloadGpuBinary(const void* gpu_binary)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // Creates a GpuEvent for the given stream.
  absl::StatusOr<std::unique_ptr<GpuEvent>> CreateGpuEvent(bool allow_timing);

  // Returns true if a delay kernel is supported.
  absl::StatusOr<bool> DelayKernelIsSupported();

  // Guards the on-disk-module mapping.
  absl::Mutex disk_modules_mu_;

  // Mapping from filename to GPUModuleHandle, if it was already retrieved.
  // Multiple GPUFunctionHandle are usually obtained from a single
  // GPUModuleHandle so we attempt to hit in this mapping first, before
  // retrieving it.
  std::map<std::string, GpuModuleHandle> disk_modules_
      ABSL_GUARDED_BY(disk_modules_mu_);

  // Guards the in-memory-module mapping.
  absl::Mutex in_memory_modules_mu_;

  std::map<const char*, GpuModuleHandle> in_memory_modules_
      ABSL_GUARDED_BY(in_memory_modules_mu_);

  absl::Mutex shared_constants_mu_;
  // On-device constants that can be shared between multiple executables. A
  // pointer for a given constant will expire when no executables require use
  // of that constant anymore.
  std::map<const absl::uint128, std::weak_ptr<DeviceMemoryBase>>
      shared_constants_ ABSL_GUARDED_BY(shared_constants_mu_);

  // Kernel -> loaded GPU binary. Many kernels may load the same binary.
  std::unordered_map<const Kernel*, const void*> kernel_to_gpu_binary_
      ABSL_GUARDED_BY(in_memory_modules_mu_);
  // GPU binary (PTX or CUBIN or HSACO) -> {CUDA module, reference count}.
  std::unordered_map<const void*, std::pair<GpuModuleHandle, uint64_t>>
      gpu_binary_to_module_ ABSL_GUARDED_BY(in_memory_modules_mu_);

  // Handle for the CUDA device being operated on. Immutable
  // post-initialization.
  GpuDeviceHandle device_;

  // True if delay kernels are supported.
  bool delay_kernels_supported_ = false;

  // The major version of the compute capability for device_.
  int cc_major_;

  // The minor version of the compute capability for device_.
  int cc_minor_;

  // Reader/writer lock for mutable data structures on this object.
  absl::Mutex mu_;

  // Memoized DNN support object -- we only want to create this once when asked
  // for a DNN interface.
  std::unique_ptr<dnn::DnnSupport> dnn_ ABSL_GUARDED_BY(mu_);

  // Memoized FFT support object -- we only want to create this once when asked
  // for a FFT interface.
  std::unique_ptr<fft::FftSupport> fft_ ABSL_GUARDED_BY(mu_);

  // Memoized BLAS support object -- we only want to create this once when asked
  // for a BLAS interface.
  std::unique_ptr<blas::BlasSupport> blas_ ABSL_GUARDED_BY(mu_);

  absl::Mutex alive_gpu_streams_mu_;

  // Lookup map for alive streams, from raw stream pointers.
  absl::flat_hash_map<void*, Stream*> alive_gpu_streams_
      ABSL_GUARDED_BY(alive_gpu_streams_mu_);
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_EXECUTOR_H_
