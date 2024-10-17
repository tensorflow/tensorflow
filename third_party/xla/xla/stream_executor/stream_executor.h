/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/allocator_stats.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {

// Identifies the memory space where an allocation resides.
enum class MemoryType { kDevice = 0, kUnified, kCollective, kHost = 5 };

inline std::string MemoryTypeString(MemoryType memory_type) {
  switch (memory_type) {
    case MemoryType::kDevice:
      return "device";
    case MemoryType::kUnified:
      return "unified";
    case MemoryType::kCollective:
      return "collective";
    case MemoryType::kHost:
      return "host";
  }
}

/// The StreamExecutor is a single-device abstraction for:
//
// * Loading/launching data-parallel-kernels
// * Invoking pre-canned high-performance library routines (like matrix
//   multiply)
//
// Interface which defines the method for interacting with an accelerator device
// (e.g. GPU, TPU).
class StreamExecutor {
 public:
  virtual ~StreamExecutor() = default;

  // Returns an ActivateContext that ensures the StreamExecutor's context is
  // activated for the duration of the returned ActivateContext's scope.
  virtual std::unique_ptr<ActivateContext> Activate() = 0;

  // Returns a reference to the platform that created this executor.
  virtual const Platform* GetPlatform() const = 0;

  // Initializes the device for use.
  virtual absl::Status Init() = 0;

  // Returns the device ordinal.
  virtual int device_ordinal() const { return -1; }

  // Creates and initializes a Stream.
  virtual absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) = 0;
  absl::StatusOr<std::unique_ptr<Stream>> CreateStream() {
    return CreateStream(std::nullopt);
  }
  // Creates an EventBasedTimer for the given stream.
  virtual absl::StatusOr<std::unique_ptr<EventBasedTimer>>
  CreateEventBasedTimer(Stream* stream, bool use_delay_kernel) {
    return absl::UnimplementedError("Not Implemented");
  }

  // Creates and initializes an Event.
  virtual absl::StatusOr<std::unique_ptr<Event>> CreateEvent() = 0;

  // Obtains metadata about the underlying device.
  // The value is cached on first use.
  virtual const DeviceDescription& GetDeviceDescription() const = 0;

  // Synchronously allocates an array on the device of type T with element_count
  // elements.
  template <typename T>
  DeviceMemory<T> AllocateArray(uint64_t element_count,
                                int64_t memory_space = 0);

  // Convenience wrapper that allocates space for a single element of type T in
  // device memory.
  template <typename T>
  DeviceMemory<T> AllocateScalar() {
    return AllocateArray<T>(1);
  }

  // Loads a kernel from a MultiKernelLoaderSpec.
  //
  // Parameters:
  //   spec: The MultiKernelLoaderSpec is usually generated as a compile-time
  //    constant into an appropriate namespace.
  virtual absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const MultiKernelLoaderSpec& spec) {
    return absl::UnimplementedError("Not Implemented");
  }

  // Releases any state associated with the previously loaded kernel.
  virtual void UnloadKernel(const Kernel* kernel) {}

  // Unloads the module with handle `module_handle`.
  virtual bool UnloadModule(ModuleHandle module_handle) { return false; }

  // Loads a module for the platform this StreamExecutor is acting upon.
  //
  // `spec` describes the module to be loaded.  On success returns the handle
  // for the loaded module. Otherwise, returns the error which has occurred.
  virtual absl::StatusOr<ModuleHandle> LoadModule(
      const MultiModuleLoaderSpec& spec) {
    return absl::UnimplementedError("Not Implemented");
  }

  // Creates a shared constant using the content provided.
  virtual absl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
  CreateOrShareConstant(Stream* stream, absl::Span<const uint8_t> content) {
    return absl::UnimplementedError("Not Implemented");
  }

  // Synchronously allocates size bytes on the underlying platform and returns
  // a DeviceMemoryBase representing that allocation. In the case of failure,
  // nullptr is returned.
  virtual DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) = 0;
  DeviceMemoryBase Allocate(uint64_t size) {
    return Allocate(size, /*memory_space=*/0);
  }
  // Deallocates the DeviceMemory previously allocated via this interface.
  // Deallocation of a nullptr-representative value is permitted.
  virtual void Deallocate(DeviceMemoryBase* mem) = 0;

  // Allocates unified memory space of the given size, if supported.
  // See
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
  // for more details on unified memory.
  virtual void* UnifiedMemoryAllocate(uint64_t size) { return nullptr; }

  // Deallocates unified memory space previously allocated with
  // UnifiedMemoryAllocate.
  virtual void UnifiedMemoryDeallocate(void* mem) {}

  // Allocates collective device memory using ncclMemAlloc.
  // See
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html
  // for more details on User Buffer Registration.
  virtual absl::StatusOr<void*> CollectiveMemoryAllocate(uint64_t size) {
    return absl::UnimplementedError("Not implemented");
  }

  // Deallocates collective device memory previously allocated with
  // CollectiveMemoryAllocate.
  virtual absl::Status CollectiveMemoryDeallocate(void* mem) {
    return absl::UnimplementedError("Not implemented");
  }

  // Allocates a region of host memory and registers it with the platform API.
  // Memory allocated in this manner is required for use in asynchronous memcpy
  // operations, such as Stream::Memcpy.
  virtual absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) = 0;

  // Deallocates a region of host memory allocated by HostMemoryAllocate().
  virtual void HostMemoryDeallocate(void* mem) = 0;

  // Returns the memory space of the given pointer.
  virtual absl::StatusOr<MemoryType> GetPointerMemorySpace(const void* ptr) {
    return absl::UnimplementedError("Not implemented");
  }

  // Synchronizes all activity occurring in the StreamExecutor's context.
  virtual bool SynchronizeAllActivity() = 0;

  // Blocks the caller while "size" bytes are zeroed out (in POD fashion) at the
  // given location in device memory.
  virtual absl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                          uint64_t size) = 0;

  // Returns a DeviceMemoryBase representing the range [base, base + size)
  // for the given DeviceMemoryBase, such that location is contained within the
  // returned range.
  virtual absl::StatusOr<DeviceMemoryBase> GetMemoryRange(
      const DeviceMemoryBase& location) {
    return absl::UnimplementedError("Not implemented for this executor.");
  }

  virtual bool HostMemoryUnregister(void* location) { return false; };
  virtual bool HostMemoryRegister(void* location, uint64_t size) {
    return false;
  };

  // Blocks the caller while "size" bytes are copied to the given location in
  // device memory.
  virtual absl::Status SynchronousMemcpy(DeviceMemoryBase* device_dst,
                                         const void* host_src,
                                         uint64_t size) = 0;
  absl::Status SynchronousMemcpyH2D(const void* host_src, int64_t size,
                                    DeviceMemoryBase* device_dst) {
    return SynchronousMemcpy(device_dst, host_src, size);
  }

  // Blocks the caller while "size" bytes are copied to the given location
  // in host memory.
  virtual absl::Status SynchronousMemcpy(void* host_dst,
                                         const DeviceMemoryBase& device_src,
                                         uint64_t size) = 0;
  absl::Status SynchronousMemcpyD2H(const DeviceMemoryBase& device_src,
                                    int64_t size, void* host_dst) {
    return SynchronousMemcpy(host_dst, device_src, size);
  }

  // Deallocates stream resources on the underlying platform.
  virtual void DeallocateStream(Stream* stream) = 0;

  // Enables peer access from this StreamExecutor to memory
  // allocated by other, such that launched device code, memcpies, etc may
  // access it directly.
  virtual absl::Status EnablePeerAccessTo(StreamExecutor* other) = 0;

  // Returns whether it's possible to enable peer access from this
  // StreamExecutor to memory allocated by another.
  virtual bool CanEnablePeerAccessTo(StreamExecutor* other) = 0;

  // Returns the underlying device memory usage information, if it is available.
  // If it is not available (false is returned), free/total may not be
  // initialized.
  virtual bool DeviceMemoryUsage(int64_t* free, int64_t* total) const {
    return false;
  }

  // Retrieves device pointer and size for a symbol. To use
  // constant memory in CUDA, GetSymbol has to be used. Returns DeviceMemoryBase
  // describing the symbol in memory if symbol is found.
  //
  // If ModuleHandle is set then we search for `symbol_name` only within the
  // module corresponding to `module_handle`.  Otherwise all loaded modules are
  // searched.
  virtual absl::StatusOr<DeviceMemoryBase> GetSymbol(
      const std::string& symbol_name, ModuleHandle module_handle) {
    return absl::UnimplementedError("Not implemented");
  }

  // Creates a new DeviceDescription object. Ownership is transferred to the
  // caller.
  virtual absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription() const = 0;

  // Gets-or-creates a BlasSupport datatype that can be used to execute BLAS
  // routines on the current platform.
  //
  // Returns null if there was an error initializing the BLAS support for the
  // underlying platform.
  virtual blas::BlasSupport* AsBlas() { return nullptr; }

  // Gets or creates a FftSupport datatype that can be used to execute FFT
  // routines on the current platform.
  //
  // Returns null if there was an error initializing the FFT support for the
  // underlying platform.
  virtual fft::FftSupport* AsFft() { return nullptr; }

  // Gets-or-creates  a DnnSupport datatype that can be used for neural network
  // routines on the current platform.
  //
  // Returns null if there was an error initializing the DNN support for the
  // underlying platform.
  virtual dnn::DnnSupport* AsDnn() { return nullptr; }

  // Creates a new CommandBuffer object.
  virtual absl::StatusOr<std::unique_ptr<CommandBuffer>> CreateCommandBuffer(
      CommandBuffer::Mode mode) {
    return absl::UnimplementedError("Command buffers are not implemented");
  }

  // Returns allocator statistics.
  virtual std::optional<AllocatorStats> GetAllocatorStats() {
    return std::nullopt;
  }

  // Clears the internal stats except for the `in_use` fields  and sets the
  // `peak_bytes_in_use` to be equal to the `bytes_in_use`. Returns true if
  // implemented.
  virtual bool ClearAllocatorStats() { return false; }

  // Clears the compilation cache from volatile memory. Returns OK if no
  // compilation cache exists or if clearing the compilation cache is
  // unsupported. Caches in non-volatile storage are unaffected.
  virtual absl::Status FlushCompilationCache() { return absl::OkStatus(); }

  // Returns a stream allocated by this executor, or nullptr if not found.
  virtual Stream* FindAllocatedStream(void* device_stream) { return nullptr; }

  // Returns the memory limit in bytes supported by this executor.
  virtual int64_t GetMemoryLimitBytes() const = 0;

  // The following methods access an internal log of some subset
  // of arguments passed to other class methods.
  // Used for testing/debugging purposes.

  struct GemmCallTrace {
    enum class GemmType {
      kPlain = 0,
      kStridedBatched = 1,
      kBatched = 2,
      kBlasLt = 3
    };
    GemmType op;
    int flags;
    uint64_t size1, size2;
  };
  // This may be expanded as necessary to trace other calls
  using ApiTrace = std::variant<GemmCallTrace>;

  // Retrieves and clears internal argument logs.
  virtual absl::StatusOr<std::vector<ApiTrace>> ExtractApiTrace() {
    return absl::UnimplementedError("Not implemented");
  }
  virtual absl::Status RecordApiTrace(ApiTrace call) {
    return absl::UnimplementedError("Not implemented");
  }

  static constexpr uint64_t kLogGemm = 1 << 0;

  // Sets the argument logging mode. Returns true if 'mode' is valid.
  // The mode is a bitmask of the kLog* constants.
  virtual bool SetArgumentLoggingMode(uint64_t mode) { return false; }
};

template <typename T>
inline DeviceMemory<T> StreamExecutor::AllocateArray(uint64_t element_count,
                                                     int64_t memory_space) {
  uint64_t bytes = sizeof(T) * element_count;
  auto memory_limit_bytes = GetMemoryLimitBytes();
  if (memory_limit_bytes > 0 &&
      static_cast<int64_t>(bytes) > memory_limit_bytes) {
    LOG(WARNING) << "Not enough memory to allocate " << bytes << " on device "
                 << device_ordinal()
                 << " within provided limit.  limit=" << memory_limit_bytes
                 << "]";
    return DeviceMemory<T>();
  }
  return DeviceMemory<T>(Allocate(bytes, memory_space));
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
