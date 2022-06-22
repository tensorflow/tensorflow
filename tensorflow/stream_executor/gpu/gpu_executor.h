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
#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_

#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/gpu/gpu_kernel.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {

class StreamExecutor;

namespace gpu {

// Pointer-to-implementation object type with virtual destruction for any XLA
// specific data hanging off of the GpuExecutor.
class XLAInterface {
 public:
  // Default constructor for the abstract interface.
  explicit XLAInterface() {}

  // Default destructor for the abstract interface.
  virtual ~XLAInterface() {}
};

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
      tensorflow::mutex_lock l(mu_);
      if (!object_) {
        object_ = std::make_unique<Model<T>>(se);
      }
      return &(dynamic_cast<Model<T>*>(object_.get())->object);
    }

   private:
    tensorflow::mutex mu_;
    std::unique_ptr<Concept> object_ ABSL_GUARDED_BY(mu_);
  };

 public:
  // sub_platform indicates the subplatform used in this executor; it must
  // be a CUDA type.
  explicit GpuExecutor(const PluginConfig& plugin_config)
      : device_(0),
        context_(nullptr),
        device_ordinal_(0),
        cc_major_(0),
        cc_minor_(0),
        version_(0),
        plugin_config_(plugin_config) {}

  // See the corresponding StreamExecutor methods for method comments on the
  // following overrides.

  ~GpuExecutor() override;

  port::Status Init(int device_ordinal, DeviceOptions device_options) override;

  port::Status GetKernel(const MultiKernelLoaderSpec& spec,
                         KernelBase* kernel) override;
  // (supported on CUDA only)
  void UnloadKernel(const KernelBase* kernel) override;
  port::Status LoadModule(const MultiModuleLoaderSpec& spec,
                          ModuleHandle* module_handle) override;
  bool UnloadModule(ModuleHandle module_handle) override;

  // Allocates and initializes a new constant on the device with the given
  // content. Or, if a device with identical content is already on-device,
  // returns a pointer to that buffer with shared ownership.
  port::StatusOr<std::shared_ptr<DeviceMemoryBase>> CreateOrShareConstant(
      Stream* stream, const std::vector<uint8_t>& content) override;

  port::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                      const BlockDim& block_dims, const KernelBase& k,
                      const KernelArgsArrayBase& args) override;

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

  port::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override;

  port::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                 uint64_t size) override;

  port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) override;

  port::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override;

  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                               const DeviceMemoryBase& gpu_src,
                                               uint64_t size) override;

  port::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64_t size) override;
  port::Status Memset(Stream* stream, DeviceMemoryBase* location, uint8 pattern,
                      uint64_t size) override;
  port::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32 pattern, uint64_t size) override;

  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64_t size) override;

  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64_t size) override;

  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64_t size) override;

  bool HostCallback(Stream* stream,
                    std::function<port::Status()> callback) override;

  bool AllocateStream(Stream* stream) override;

  void DeallocateStream(Stream* stream) override;

  bool CreateStreamDependency(Stream* dependent, Stream* other) override;

  bool AllocateTimer(Timer* timer) override;

  void DeallocateTimer(Timer* timer) override;

  bool StartTimer(Stream* stream, Timer* timer) override;

  bool StopTimer(Stream* stream, Timer* timer) override;

  port::Status AllocateEvent(Event* event) override;

  port::Status DeallocateEvent(Event* event) override;

  port::Status RecordEvent(Stream* stream, Event* event) override;

  port::Status WaitForEvent(Stream* stream, Event* event) override;

  Event::Status PollForEventStatus(Event* event) override;

  port::Status BlockHostUntilDone(Stream* stream) override;

  int PlatformDeviceCount() override { return GpuDriver::GetDeviceCount(); }

  port::Status EnablePeerAccessTo(StreamExecutorInterface* other) override;

  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  // Search for the symbol in the given module and returns a device pointer and
  // size. Returns false if symbol does not exist. 'module_handle' must not
  // be null.
  bool GetSymbol(const std::string& symbol_name, ModuleHandle module_handle,
                 void** mem, size_t* bytes) override;

  port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CreateDeviceDescription(device_ordinal_);
  }

  static port::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  bool SupportsBlasPlans() const override;

  bool SupportsBlas() const override;

  blas::BlasSupport* CreateBlas() override;

  bool SupportsFft() const override;

  fft::FftSupport* CreateFft() override;

  bool SupportsRng() const override;

  rng::RngSupport* CreateRng() override;

  bool SupportsDnn() const override;

  dnn::DnnSupport* CreateDnn() override;

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override;

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override;

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override;

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override;

  void* GpuContextHack() override;

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

 private:
  // Attempts to find a more specific version of the file indicated by
  // filename by looking for compute-capability-specific suffixed versions; i.e.
  // looking for "foo.ptx" will check to see if "foo.ptx.cc30.ptx" is present if
  // we're on a compute capability 3.0 machine.
  // (supported on CUDA only)
  bool FindOnDiskForComputeCapability(absl::string_view filename,
                                      absl::string_view canonical_suffix,
                                      std::string* found_filename) const;

  // Attempts to find a more specific version of the file indicated by
  // filename by looking for AMDGPU ISA-specific suffixed versions.
  // (supported on ROCm only)

  bool FindOnDiskForISAVersion(absl::string_view filename,
                               absl::string_view canonical_suffix,
                               std::string* found_filename) const;

  // Host callback landing routine invoked by CUDA.
  // data: User-provided callback provided to HostCallback() above, captured
  //       as a std::function<void()>. Allocated/initialized inside
  //       HostCallback() and owned and deleted by this call.
  static void InternalHostCallback(GpuStreamHandle stream, GpuStatus status,
                                   void* data);

  // Collects metadata for the specified kernel.
  port::Status GetKernelMetadata(GpuKernel* cuda_kernel,
                                 KernelMetadata* kernel_metadata);

  // Prints to VLOG(2) information about the kernel's occupancy and how it might
  // be improved.
  void VlogOccupancyInfo(const KernelBase& kernel, const ThreadDim& thread_dims,
                         const BlockDim& block_dims);

  // (supported on CUDA only)
  port::Status LoadModuleFromCuBin(const char* cubin, GpuModuleHandle* module)
      TF_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // Loads the PTX text `ptx` as a CUDA module.  `ptx` must be null terminated.
  // (supported on CUDA only)
  port::Status LoadModuleFromPtx(const char* ptx, GpuModuleHandle* module)
      TF_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // (supported on ROCm only)
  port::Status LoadModuleFromHsaco(const char* hsaco, GpuModuleHandle* module)
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
      TF_GUARDED_BY(disk_modules_mu_);

  // Guards the in-memory-module mapping.
  absl::Mutex in_memory_modules_mu_;

  std::map<const char*, GpuModuleHandle> in_memory_modules_
      TF_GUARDED_BY(in_memory_modules_mu_);

  absl::Mutex shared_constants_mu_;
  // On-device constants that can be shared between multiple executables. A
  // pointer for a given constant will expire when no executables require use
  // of that constant anymore.
  std::map<const absl::uint128, std::weak_ptr<DeviceMemoryBase>>
      shared_constants_ ABSL_GUARDED_BY(shared_constants_mu_);

  // Kernel -> loaded GPU binary. Many kernels may load the same binary.
  std::unordered_map<const KernelBase*, const void*> kernel_to_gpu_binary_
      TF_GUARDED_BY(in_memory_modules_mu_);
  // GPU binary (PTX or CUBIN or HSACO) -> {CUDA module, reference count}.
  std::unordered_map<const void*, std::pair<GpuModuleHandle, uint64_t>>
      gpu_binary_to_module_ TF_GUARDED_BY(in_memory_modules_mu_);

  // Guards the launched kernel set.
  absl::Mutex launched_kernels_mu_;

  // Keeps track of the set of launched kernels. Currently used to suppress the
  // occupancy check on subsequent launches.
  std::set<GpuFunctionHandle> launched_kernels_
      TF_GUARDED_BY(launched_kernels_mu_);

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

  // The plugin configuration associated with this instance.
  PluginConfig plugin_config_;

  // Type erased XLA specific state attached to GpuExecutor.
  Object xla_state_;

  absl::Mutex alive_gpu_streams_mu_;

  // Lookup map for alive streams, from raw stream pointers.
  absl::flat_hash_map<void*, Stream*> alive_gpu_streams_
      TF_GUARDED_BY(alive_gpu_streams_mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(GpuExecutor);
};

inline GpuExecutor* ExtractGpuExecutor(StreamExecutor* stream_exec) {
  return static_cast<GpuExecutor*>(stream_exec->implementation());
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
