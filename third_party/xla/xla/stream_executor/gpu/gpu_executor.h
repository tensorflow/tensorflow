/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// The CUDA implementation of the StreamExecutorInterface functionality.
// CUDA inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the CUDA streams
// programming model provided by the libcuda.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

class StreamExecutor;

namespace gpu {

// CUDA-platform implementation of the platform-agnostic
// StreamExecutorInterface.
class GpuExecutor : public internal::StreamExecutorInterface {
  // Helper classes to attach a type erased state to the GpuExecutor. Currently,
  // we just need to support some XLA specific state.
  class Object {
    struct Concept {
      virtual ~Concept() {}
    };
    template <typename T>
    struct Model : Concept {
      explicit Model(StreamExecutor* se) : object(se) {}
      T object;
    };

   public:
    template <typename T>
    T* getOrCreate(StreamExecutor* se) {
      absl::MutexLock l(&mu_);
      if (!object_) {
        object_ = std::make_unique<Model<T>>(se);
      }
      return &(dynamic_cast<Model<T>*>(object_.get())->object);
    }

   private:
    absl::Mutex mu_;
    std::unique_ptr<Concept> object_ ABSL_GUARDED_BY(mu_);
  };

 public:
  // sub_platform indicates the subplatform used in this executor; it must
  // be a CUDA type.
  GpuExecutor()
      : device_(0),
        context_(nullptr),
        device_ordinal_(0),
        cc_major_(0),
        cc_minor_(0),
        version_(0) {}

  // See the corresponding StreamExecutor methods for method comments on the
  // following overrides.

  ~GpuExecutor() override;

  tsl::Status Init(int device_ordinal, DeviceOptions device_options) override;

  tsl::Status GetKernel(const MultiKernelLoaderSpec& spec,
                        Kernel* kernel) override;

  // (supported on CUDA only)
  void UnloadKernel(const Kernel* kernel) override;
  tsl::Status LoadModule(const MultiModuleLoaderSpec& spec,
                         ModuleHandle* module_handle) override;
  bool UnloadModule(ModuleHandle module_handle) override;

  // Allocates and initializes a new constant on the device with the given
  // content. Or, if a device with identical content is already on-device,
  // returns a pointer to that buffer with shared ownership.
  tsl::StatusOr<std::shared_ptr<DeviceMemoryBase>> CreateOrShareConstant(
      Stream* stream, absl::Span<const uint8_t> content) override;

  tsl::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                     const BlockDim& block_dims, const Kernel& k,
                     const KernelArgs& args) override;

  tsl::Status Submit(Stream* stream,
                     const CommandBuffer& command_buffer) override;

  // (supported on CUDA only)
  int CalculateOccupancy(const DeviceDescription& device_description,
                         uint64_t registers_per_thread,
                         uint64_t shared_memory_per_block,
                         const ThreadDim& thread_dims, GpuFunctionHandle func);

  // (supported on CUDA only)
  int CompareOccupancy(int* initial_blocks,
                       const DeviceDescription& device_description,
                       uint64_t registers_per_thread,
                       uint64_t shared_memory_per_block,
                       const ThreadDim& thread_dims, GpuFunctionHandle func);

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;

  void* GetSubBuffer(DeviceMemoryBase* mem, uint64_t offset_bytes,
                     uint64_t size_bytes) override;

  void Deallocate(DeviceMemoryBase* mem) override;

  void* UnifiedMemoryAllocate(uint64_t size) override {
    return GpuDriver::UnifiedMemoryAllocate(context_, size);
  }

  void UnifiedMemoryDeallocate(void* location) override {
    return GpuDriver::UnifiedMemoryDeallocate(context_, location);
  }

  // CUDA allocation/registration functions are necessary because the driver
  // internally sets up buffers for DMA operations (and page locks them).
  // There's no external interface for us to otherwise control these DMA
  // settings.
  void* HostMemoryAllocate(uint64_t size) override {
    return GpuDriver::HostAllocate(context_, size);
  }

  void HostMemoryDeallocate(void* location) override {
    return GpuDriver::HostDeallocate(context_, location);
  }

  bool HostMemoryRegister(void* location, uint64_t size) override;

  bool HostMemoryUnregister(void* location) override;

  bool SynchronizeAllActivity() override;

  tsl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                 uint64_t size) override;

  tsl::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                uint64_t size) override;

  tsl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) override;

  tsl::Status SynchronousMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) override;

  tsl::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                              const DeviceMemoryBase& gpu_src,
                                              uint64_t size) override;

  tsl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                      uint64_t size) override;
  tsl::Status Memset(Stream* stream, DeviceMemoryBase* location,
                     uint8_t pattern, uint64_t size) override;
  tsl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                       uint32_t pattern, uint64_t size) override;

  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64_t size) override;

  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64_t size) override;

  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64_t size) override;

  bool HostCallback(Stream* stream,
                    absl::AnyInvocable<tsl::Status() &&> callback) override;

  bool AllocateStream(Stream* stream) override;

  void DeallocateStream(Stream* stream) override;

  bool CreateStreamDependency(Stream* dependent, Stream* other) override;

  tsl::Status AllocateEvent(Event* event) override;

  tsl::Status DeallocateEvent(Event* event) override;

  tsl::Status RecordEvent(Stream* stream, Event* event) override;

  tsl::Status WaitForEvent(Stream* stream, Event* event) override;

  tsl::Status WaitForEventOnExternalStream(std::intptr_t stream,
                                           Event* event) override;

  Event::Status PollForEventStatus(Event* event) override;

  tsl::Status BlockHostUntilDone(Stream* stream) override;

  tsl::Status EnablePeerAccessTo(StreamExecutorInterface* other) override;

  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  // Search for the symbol in the given module and returns a device pointer and
  // size. Returns false if symbol does not exist. 'module_handle' must not
  // be null.
  bool GetSymbol(const std::string& symbol_name, ModuleHandle module_handle,
                 void** mem, size_t* bytes) override;

  tsl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CreateDeviceDescription(device_ordinal_);
  }

  static tsl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  blas::BlasSupport* CreateBlas() override;

  fft::FftSupport* CreateFft() override;

  dnn::DnnSupport* CreateDnn() override;

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override;

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override;

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override;

  tsl::StatusOr<std::unique_ptr<internal::CommandBufferInterface>>
  GetCommandBufferImplementation(CommandBuffer::Mode mode) override;

  // Wraps existing Gpu graph handle into an instance of Gpu command buffer.
  // This is required for wrapping nested graphs constructed for conditional
  // nodes and owned by a parent graph executable.
  std::unique_ptr<internal::CommandBufferInterface>
  GetCommandBufferImplementation(CommandBuffer::Mode mode, GpuGraphHandle graph,
                                 bool is_owned_graph);

  void* platform_specific_context() override;

  GpuContext* gpu_context();

  // Provide a type-erased way of attaching arbitrary XLA specific state to the
  // GpuExecutor. XLA based execution will use this method to attach per-stream
  // executor XLA specific objects (like the Infeed and Outfeed managers) to the
  // stream executor, so that their lifetimes can be tied to the lifetime of the
  // stream executor for which that object is allocated for. This simplifies
  // memory management as compared to having these objects reside on the side
  // and then either leaking or having to implement callbacks that the SE
  // destructors call to deallocate any side state that is associated with that
  // SE object.
  template <typename T>
  T* getOrCreateXLAState(StreamExecutor* se) {
    return xla_state_.getOrCreate<T>(se);
  }

  Stream* FindAllocatedStream(void* gpu_stream) override {
    absl::MutexLock lock(&alive_gpu_streams_mu_);
    auto it = alive_gpu_streams_.find(gpu_stream);
    if (it == alive_gpu_streams_.end()) {
      return nullptr;
    }
    return it->second;
  }

  GpuDeviceHandle device() const { return device_; }
  int cc_major() const { return cc_major_; }
  int cc_minor() const { return cc_minor_; }

 private:
  // Host callback landing routine invoked by CUDA.
  // data: User-provided callback provided to HostCallback() above, captured
  //       as a std::function<void()>. Allocated/initialized inside
  //       HostCallback() and owned and deleted by this call.
  static void InternalHostCallback(void* data);

  // Collects metadata for the specified kernel.
  tsl::Status GetKernelMetadata(GpuKernel* cuda_kernel,
                                KernelMetadata* kernel_metadata);

  // Prints to VLOG(2) information about the kernel's occupancy and how it might
  // be improved.
  void VlogOccupancyInfo(const Kernel& kernel, const ThreadDim& thread_dims,
                         const BlockDim& block_dims);

  // (supported on CUDA only)
  tsl::Status LoadModuleFromCuBin(const char* cubin, GpuModuleHandle* module)
      TF_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // Loads the PTX text `ptx` as a CUDA module.  `ptx` must be null terminated.
  // (supported on CUDA only)
  tsl::Status LoadModuleFromPtx(const char* ptx, GpuModuleHandle* module)
      TF_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // (supported on ROCm only)
  tsl::Status LoadModuleFromHsaco(const char* hsaco, GpuModuleHandle* module)
      TF_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  bool UnloadGpuBinary(const void* gpu_binary)
      TF_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

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

  // Guards the launched kernel set.
  absl::Mutex launched_kernels_mu_;

  // Keeps track of the set of launched kernels. Currently used to suppress the
  // occupancy check on subsequent launches.
  std::set<GpuFunctionHandle> launched_kernels_
      ABSL_GUARDED_BY(launched_kernels_mu_);

  // Handle for the CUDA device being operated on. Immutable
  // post-initialization.
  GpuDeviceHandle device_;

  // Handle for session with the library/driver. Immutable post-initialization.
  GpuContext* context_;

  // The device ordinal value that this executor was initialized with; recorded
  // for use in getting device metadata. Immutable post-initialization.
  int device_ordinal_;

  // The major version of the compute capability for device_.
  int cc_major_;

  // The minor version of the compute capability for device_.
  int cc_minor_;

  // GPU ISA version for device_.
  int version_;

  // Type erased XLA specific state attached to GpuExecutor.
  Object xla_state_;

  absl::Mutex alive_gpu_streams_mu_;

  // Lookup map for alive streams, from raw stream pointers.
  absl::flat_hash_map<void*, Stream*> alive_gpu_streams_
      ABSL_GUARDED_BY(alive_gpu_streams_mu_);

  GpuExecutor(const GpuExecutor&) = delete;
  void operator=(const GpuExecutor&) = delete;
};

inline GpuExecutor* ExtractGpuExecutor(StreamExecutor* stream_exec) {
  return static_cast<GpuExecutor*>(stream_exec->implementation());
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
