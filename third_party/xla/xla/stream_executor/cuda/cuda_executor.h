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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_device_allocator.h"
#include "xla/stream_executor/cuda/cuda_host_allocator.h"
#include "xla/stream_executor/cuda/cuda_kernel.h"
#include "xla/stream_executor/cuda/cuda_vmm_allocator.h"
#include "xla/stream_executor/cuda/host_callback_registry.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tensor_map.h"

namespace stream_executor::gpu {

// This class implements GpuExecutor for NVIDIA GPUs that use CUDA libraries.
class CudaExecutor : public GpuExecutor {
 public:
  CudaExecutor(Platform* platform, int device_ordinal,
               absl::Duration monitor_poll_interval = absl::Seconds(5))
      : GpuExecutor(platform, device_ordinal),
        host_callback_registry_(std::make_unique<HostCallbackRegistry>(
            device_ordinal, monitor_poll_interval)) {}

  ~CudaExecutor() override;
  std::unique_ptr<ActivateContext> Activate() override;
  absl::Status Init() override;
  int numa_node() const override { return numa_node_; }
  bool SynchronizeAllActivity() override;
  HostCallbackRegistry* GetHostCallbackRegistry() const {
    return host_callback_registry_.get();
  }
  absl::StatusOr<DeviceAddressBase> GetMemoryRange(
      const DeviceAddressBase& location) const override;
  absl::StatusOr<std::unique_ptr<EventBasedTimer>> CreateEventBasedTimer(
      Stream* stream, bool use_delay_kernel) override;
  absl::StatusOr<DeviceAddressBase> GetSymbol(
      const std::string& symbol_name, ModuleHandle module_handle) override;
  absl::Status SynchronousMemcpy(DeviceAddressBase* gpu_dst,
                                 const void* host_src, uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceAddressBase& gpu_src,
                                 uint64_t size) override;
  void DeallocateStream(Stream* stream) override;
  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(int other_device_ordinal) override;
  bool DeviceMemoryUsage(int64_t* free_out, int64_t* total_out) const override;
  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;
  void UnloadKernel(const Kernel* kernel) override;
  absl::StatusOr<ModuleHandle> LoadModule(
      const MultiModuleLoaderSpec& spec) override;
  bool UnloadModule(ModuleHandle module_handle) override;
  absl::StatusOr<std::shared_ptr<DeviceAddressBase>> CreateOrShareConstant(
      Stream* stream, absl::Span<const uint8_t> content) override;
  DeviceAddressBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceAddressBase* mem) override;
  blas::BlasSupport* AsBlas() override;
  fft::FftSupport* AsFft() override;
  dnn::DnnSupport* AsDnn() override;
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;
  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;
  absl::StatusOr<std::unique_ptr<CommandBuffer>> CreateCommandBuffer(
      CommandBuffer::Mode mode) override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CudaExecutor::CreateDeviceDescription(device_ordinal());
  }
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override;

  bool HostMemoryRegister(void* location, uint64_t size) override;
  bool HostMemoryUnregister(void* location) override;

  bool IsVmmMemory(const DeviceAddressBase& address) override;

  absl::StatusOr<MemorySpace> GetPointerMemorySpace(const void* ptr) override;

  Stream* FindAllocatedStream(void* gpu_stream) override {
    absl::MutexLock lock(alive_gpu_streams_mu_);
    auto it = alive_gpu_streams_.find(gpu_stream);
    if (it == alive_gpu_streams_.end()) {
      return nullptr;
    }
    return it->second;
  }

  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  // Returns a CudaKernel pointer for a given Kernel, if the kernel is
  // associated with this executor. Otherwise a NotFound error is returned.
  absl::StatusOr<const CudaKernel*> GetCudaKernel(const Kernel* kernel);

  // Creates, allocates, and copies a CUtensorMap object for the given TMA
  // descriptor. Returns a TensorMap, which is 128 bytes of storage, to be
  // passed by value to the kernel.
  absl::StatusOr<TensorMap> CreateTensorMap(const TmaDescriptor& tma_desc,
                                            void* global_address) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocator>> CreateMemoryAllocator(
      MemorySpace type) override;

  // Returns the granularity which is the minimum unit of memory that can be
  // allocated with VMM API. In order to map the memory slices to multicast
  // object, the offset of the slices should be aligned with this granularity.
  absl::StatusOr<size_t> GetVmmGranularity() const;

  int GetGpuStreamPriority(StreamPriority priority) override;

  // RAII wrapper for a VMM memory handle.
  class VmmMemoryHandle {
   public:
    explicit VmmMemoryHandle(uint64_t handle) : handle_(handle) {}
    ~VmmMemoryHandle();
    VmmMemoryHandle(const VmmMemoryHandle&) = delete;
    VmmMemoryHandle& operator=(const VmmMemoryHandle&) = delete;
    VmmMemoryHandle(VmmMemoryHandle&&);
    VmmMemoryHandle& operator=(VmmMemoryHandle&&);

    uint64_t handle() const { return handle_; }

   private:
    absl::Status Release();
    uint64_t handle_;
  };

  class CudaMulticastMemory : public MulticastMemory {
   public:
    CudaMulticastMemory()
        : handle_(0),
          padded_size_(0),
          granularity_(0),
          num_devices_(0),
          subscribed_devices_(0) {}
    ~CudaMulticastMemory() override;

    absl::Status SubscribeDevice(int device_number) override;

    absl::StatusOr<void*> MapMemory(const DeviceAddressBase& location,
                                    const GpuExecutor* gpu_executor) override;

   private:
    friend class CudaExecutor;
    absl::Status Initialize(uint64_t size, int num_devices,
                            const GpuExecutor* gpu_executor);
    CUmemGenericAllocationHandle handle_;
    uint64_t padded_size_;
    uint64_t granularity_;
    int num_devices_;
    std::atomic<int> subscribed_devices_;
    absl::flat_hash_map<int, CUdeviceptr> mapped_devices_
        ABSL_GUARDED_BY(mapped_devices_mu_);
    absl::Mutex mapped_devices_mu_;
  };

  absl::StatusOr<std::unique_ptr<MulticastMemory>> CreateMulticastMemory(
      uint64_t size, int num_devices) const override;

  // Returns a handle to the given memory if it was allocated with VMM API.
  absl::StatusOr<VmmMemoryHandle> RetainVmmMemoryHandle(void* ptr) const;

  bool is_multicast_supported() const override {
    return is_multicast_supported_;
  }

  bool is_fabric_supported() const { return vmm_options_.enable_fabric_handle; }

 private:
  // Allocates memory using the given allocator and tracks the resulting
  // allocation. Returns an empty DeviceAddressBase on failure.
  DeviceAddressBase AllocateAndTrack(MemoryAllocator& allocator, uint64_t size,
                                     absl::string_view kind);

  // Loads a module in cubin format.
  absl::StatusOr<ModuleHandle> LoadModuleFromCuBin(const char* cubin)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // Loads the PTX text `ptx` as a CUDA module. `ptx` must be null terminated.
  absl::StatusOr<ModuleHandle> LoadModuleFromPtx(const char* ptx)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  bool UnloadGpuBinary(ModuleHandle gpu_binary)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // Returns true if a delay kernel is supported.
  absl::StatusOr<bool> DelayKernelIsSupported();

  // Whether multicast objects are supported by this device.
  bool is_multicast_supported_ = false;

  // VMM allocator options, probed during Init() with fallback. The handle-type
  // flags may differ from device-reported caps (e.g. in MIG or containers).
  CudaVmmAllocator::Options vmm_options_;

  // Guards the in-memory-module mapping.
  absl::Mutex in_memory_modules_mu_;

  absl::Mutex shared_constants_mu_;
  // On-device constants that can be shared between multiple executables. A
  // pointer for a given constant will expire when no executables require use
  // of that constant anymore.
  std::map<const absl::uint128, std::weak_ptr<DeviceAddressBase>>
      shared_constants_ ABSL_GUARDED_BY(shared_constants_mu_);

  // Kernel -> loaded GPU module. Many kernels may load the same binary.
  absl::flat_hash_map<const Kernel*, ModuleHandle> kernel_to_gpu_binary_
      ABSL_GUARDED_BY(in_memory_modules_mu_);

  // Loaded GPU module handle -> {CUDA module, reference count}.
  absl::flat_hash_map<ModuleHandle, std::pair<CUmodule, uint64_t>>
      gpu_binary_to_module_ ABSL_GUARDED_BY(in_memory_modules_mu_);

  // Set of loaded kernels. This contains all kernels loaded by this executor,
  // including in-process kernels.
  absl::flat_hash_set<const Kernel*> loaded_kernels_
      ABSL_GUARDED_BY(in_memory_modules_mu_);

  // Handle for the CUDA device being operated on. Immutable
  // post-initialization.
  CUdevice device_;

  // True if delay kernels are supported.
  bool delay_kernels_supported_ = false;

  // The NUMA node of the CPU closest to device_
  int numa_node_;

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

  // Memory allocators for supported memory spaces.
  std::unique_ptr<CudaDeviceAllocator> device_allocator_;
  std::unique_ptr<CudaHostAllocator> host_allocator_;
  std::unique_ptr<CudaVmmAllocator> vmm_allocator_;  // null if VMM unsupported

  // Tracks allocations made through the memory allocators, bridging the RAII
  // MemoryAllocation API to the raw-pointer Allocate/Deallocate interface.
  MemoryAllocator::AllocationTracker allocation_tracker_;

  // CudaContext for this device.
  CudaContext* cuda_context_;

  // Cached CUDA stream priority range. Initialized once on first non-default
  // request and then reused for subsequent calls.
  absl::once_flag stream_priority_once_;
  int stream_priority_lowest_ = 0;
  int stream_priority_highest_ = 0;
  bool stream_priority_query_ok_ = false;
  absl::flat_hash_map<int, bool> peer_access_cache_;
  std::unique_ptr<HostCallbackRegistry> host_callback_registry_{nullptr};
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_EXECUTOR_H_
