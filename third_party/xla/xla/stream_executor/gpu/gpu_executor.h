/* Copyright 2019 The OpenXLA Authors.

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

// The CUDA implementation of the StreamExecutor functionality.
// CUDA inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the CUDA streams
// programming model provided by the libcuda.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/gpu_collectives.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/host_memory_allocation.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_common.h"
#include "tsl/platform/thread_annotations.h"

namespace stream_executor {

class StreamExecutor;

namespace gpu {

class GpuKernel;
class GpuCommandBuffer;

// CUDA-platform implementation of the platform-agnostic
// StreamExecutor.
class GpuExecutor : public StreamExecutorCommon {
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
  GpuExecutor(Platform* platform, int device_ordinal)
      : StreamExecutorCommon(platform),
        device_(0),
        context_(nullptr),
        device_ordinal_(device_ordinal),
        cc_major_(0),
        cc_minor_(0),
        version_(0) {}

  // See the corresponding StreamExecutor methods for method comments on the
  // following overrides.

  ~GpuExecutor() override;

  absl::Status Init() override;

  int device_ordinal() const override { return device_ordinal_; };

  absl::Status GetKernel(const MultiKernelLoaderSpec& spec,
                         Kernel* kernel) override;

  // (supported on CUDA only)
  void UnloadKernel(const Kernel* kernel) override;
  absl::Status LoadModule(const MultiModuleLoaderSpec& spec,
                          ModuleHandle* module_handle) override;
  bool UnloadModule(ModuleHandle module_handle) override;

  // Allocates and initializes a new constant on the device with the given
  // content. Or, if a device with identical content is already on-device,
  // returns a pointer to that buffer with shared ownership.
  absl::StatusOr<std::shared_ptr<DeviceMemoryBase>> CreateOrShareConstant(
      Stream* stream, absl::Span<const uint8_t> content) override;

  absl::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                      const BlockDim& block_dims, const Kernel& kernel,
                      const KernelArgs& args) override;

  absl::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                      const BlockDim& block_dims,
                      const ClusterDim& cluster_dims, const Kernel& kernel,
                      const KernelArgs& args) override;

  absl::Status Submit(Stream* stream,
                      const CommandBuffer& command_buffer) override;

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;

  void Deallocate(DeviceMemoryBase* mem) override;

  void* UnifiedMemoryAllocate(uint64_t size) override {
    return GpuDriver::UnifiedMemoryAllocate(context_, size);
  }

  void UnifiedMemoryDeallocate(void* location) override {
    return GpuDriver::UnifiedMemoryDeallocate(context_, location);
  }

  absl::StatusOr<void*> CollectiveMemoryAllocate(uint64_t size) override {
    return GpuCollectives::CollectiveMemoryAllocate(context_, size);
  }

  absl::Status CollectiveMemoryDeallocate(void* location) override {
    return GpuCollectives::CollectiveMemoryDeallocate(context_, location);
  }

  // CUDA allocation/registration functions are necessary because the driver
  // internally sets up buffers for DMA operations (and page locks them).
  // There's no external interface for us to otherwise control these DMA
  // settings.
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override {
    auto* buffer = GpuDriver::HostAllocate(context_, size);
    if (buffer == nullptr && size > 0) {
      return absl::InternalError(
          absl::StrFormat("Failed to allocate HostMemory of size %d", size));
    }
    return std::make_unique<HostMemoryAllocation>(buffer, size, this);
  }

  void HostMemoryDeallocate(void* location) override {
    return GpuDriver::HostDeallocate(context_, location);
  }

  bool SynchronizeAllActivity() override;

  absl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override;

  absl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) override;

  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override;

  absl::Status Memset(Stream* stream, DeviceMemoryBase* location,
                      uint8_t pattern, uint64_t size) override;

  bool HostCallback(Stream* stream,
                    absl::AnyInvocable<absl::Status() &&> callback) override;

  void DeallocateStream(Stream* stream) override;

  absl::Status BlockHostUntilDone(Stream* stream) override;

  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;

  bool CanEnablePeerAccessTo(StreamExecutor* other) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  absl::StatusOr<DeviceMemoryBase> GetSymbol(
      const std::string& symbol_name, ModuleHandle module_handle) override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CreateDeviceDescription(device_ordinal_);
  }

  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  blas::BlasSupport* AsBlas() override;

  fft::FftSupport* AsFft() override;

  dnn::DnnSupport* AsDnn() override;

  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority =
          std::nullopt) override;

  absl::StatusOr<std::unique_ptr<Kernel>> CreateKernel() override;

  absl::StatusOr<std::unique_ptr<CommandBuffer>> CreateCommandBuffer(
      CommandBuffer::Mode mode) override;

  // Wraps existing Gpu graph handle into an instance of Gpu command buffer.
  // This is required for wrapping nested graphs constructed for conditional
  // nodes and owned by a parent graph executable.
  std::unique_ptr<GpuCommandBuffer> CreateCommandBuffer(
      CommandBuffer::Mode mode, GpuGraphHandle graph, bool is_owned_graph);

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

  absl::StatusOr<std::vector<ApiTrace>> ExtractApiTrace() override {
    absl::MutexLock lock(&logger_mu_);
    return std::move(argument_logs_);
  }

  absl::Status RecordApiTrace(ApiTrace call) override {
    absl::MutexLock lock(&logger_mu_);
    if (std::holds_alternative<GemmCallTrace>(call) &&
        (argument_logging_mode_ & kLogGemm)) {
      argument_logs_.push_back(call);
    }
    return absl::OkStatus();
  }

  bool SetArgumentLoggingMode(uint64_t mode) override {
    absl::MutexLock lock(&logger_mu_);
    argument_logging_mode_ = mode;
    return true;
  }

  uint64_t GetArgumentLoggingMode() const { return argument_logging_mode_; }

 private:
  // Host callback landing routine invoked by CUDA.
  // data: User-provided callback provided to HostCallback() above, captured
  //       as a std::function<void()>. Allocated/initialized inside
  //       HostCallback() and owned and deleted by this call.
  static void InternalHostCallback(void* data);

  // Collects metadata for the specified kernel.
  absl::Status GetKernelMetadata(GpuKernel* cuda_kernel,
                                 KernelMetadata* kernel_metadata);

  // (supported on CUDA only)
  absl::Status LoadModuleFromCuBin(const char* cubin, GpuModuleHandle* module)
      TF_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // Loads the PTX text `ptx` as a CUDA module.  `ptx` must be null terminated.
  // (supported on CUDA only)
  absl::Status LoadModuleFromPtx(const char* ptx, GpuModuleHandle* module)
      TF_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // (supported on ROCm only)
  absl::Status LoadModuleFromHsaco(const char* hsaco, GpuModuleHandle* module)
      TF_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  absl::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                      const BlockDim& block_dims,
                      const std::optional<ClusterDim>& cluster_dims,
                      const Kernel& kernel, const KernelArgs& args);

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

  absl::Mutex logger_mu_;

  mutable std::vector<ApiTrace> argument_logs_ ABSL_GUARDED_BY(logger_mu_);

  uint64_t argument_logging_mode_ = 0;

  GpuExecutor(const GpuExecutor&) = delete;
  void operator=(const GpuExecutor&) = delete;
};

inline GpuExecutor* ExtractGpuExecutor(StreamExecutor* stream_exec) {
  return static_cast<GpuExecutor*>(stream_exec);
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
