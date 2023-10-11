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

// Interfaces for platform-dependent implementations to satisfy. This are
// delegated to from the StreamExecutor in pointer-to-implementation style; i.e.
// the StreamExecutor is just a husk that delegates calls to the
// platform-specific objects which implement the interfaces defined here.

#ifndef XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
#define XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "xla/stream_executor/allocator_stats.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_options.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_cache_config.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/trace_listener.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

class Stream;

namespace internal {

//===----------------------------------------------------------------------===//
// EventInterface
//===----------------------------------------------------------------------===//

// Platform-dependent interface class for the generic Events interface, in
// the PIMPL style.
class EventInterface {
 public:
  EventInterface() = default;
  virtual ~EventInterface() = default;

 private:
  EventInterface(const EventInterface&) = delete;
  void operator=(const EventInterface&) = delete;
};

//===----------------------------------------------------------------------===//
// KernelInterface
//===----------------------------------------------------------------------===//

// Pointer-to-implementation object type (i.e. the KernelBase class delegates to
// this interface) with virtual destruction. This class exists for the
// platform-dependent code to hang any kernel data/resource info/functionality
// off of.
class KernelInterface {
 public:
  // Default constructor for the abstract interface.
  KernelInterface() = default;

  // Default destructor for the abstract interface.
  virtual ~KernelInterface() = default;

  // Returns the number of formal parameters that this kernel accepts.
  virtual unsigned Arity() const = 0;

  // Sets the preferred cache configuration.
  virtual void SetPreferredCacheConfig(KernelCacheConfig config) = 0;

  // Gets the preferred cache configuration.
  virtual KernelCacheConfig GetPreferredCacheConfig() const = 0;

 private:
  KernelInterface(const KernelInterface&) = delete;
  void operator=(const KernelInterface&) = delete;
};

//===----------------------------------------------------------------------===//
// CommandBufferInterface
//===----------------------------------------------------------------------===//

// Platform-dependent interface class for implementing generic CommandBuffer.
//
// TODO(ezhulenev): Currently we assume that all operations between barriers
// can execute concurrently, and it's up to the caller to insert barriers to
// guarantee correctness. Consider adding finer grained synchronization
// mechanism between different commands.
//
// TODO(ezhulenev): Currently command buffers do no support updates, and once
// finalized can be executed as recorded. We need to support cheap command
// buffer updates that in GPU backend will be mapped to CUDA/HIP graph node
// updates.
class CommandBufferInterface {
 public:
  CommandBufferInterface() = default;
  virtual ~CommandBufferInterface() = default;

  // Traces `function` invocation by recording all operations on the `stream`
  // into the command buffer. Command buffer must be empty.
  virtual tsl::Status Trace(Stream* stream,
                            absl::AnyInvocable<tsl::Status()> function) = 0;

  // Adds a kernel launch command to the command buffer.
  virtual tsl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                             const KernelBase& kernel,
                             const KernelArgsArrayBase& args) = 0;

  // Adds a device-to-device memory copy to the command buffer.
  virtual tsl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                           const DeviceMemoryBase& src,
                                           uint64_t size) = 0;

  // Finalizes command buffer and makes it executable. Once command buffer is
  // finalized no commands can be added to it.
  virtual tsl::Status Finalize() = 0;

  // Begins command buffer update. Command buffer update should be finalized
  // before it can be executed.
  virtual tsl::Status Update() = 0;

  // Returns command buffer execution mode.
  virtual CommandBuffer::Mode mode() const = 0;

  // Returns command buffer state.
  virtual CommandBuffer::State state() const = 0;

 private:
  CommandBufferInterface(const CommandBufferInterface&) = delete;
  void operator=(const CommandBufferInterface&) = delete;
};

//===----------------------------------------------------------------------===//
// StreamInterface
//===----------------------------------------------------------------------===//

// Pointer-to-implementation object type (i.e. the Stream class delegates to
// this interface) with virtual destruction. This class exists for the
// platform-dependent code to hang any kernel data/resource info/functionality
// off of.
class StreamInterface {
 public:
  // Default constructor for the abstract interface.
  StreamInterface() = default;

  // Default destructor for the abstract interface.
  virtual ~StreamInterface() = default;

  // Sets priority for a stream.
  virtual void SetPriority(StreamPriority priority) {
    LOG(ERROR) << "SetPriority unimplemented for this stream.";
  }

  virtual void SetPriority(int priority) {
    LOG(ERROR) << "SetPriority unimplemented for this stream.";
  }

  // Gets priority for a stream.
  virtual std::variant<StreamPriority, int> priority() const {
    return StreamPriority::Default;
  }

  // Returns the GPU stream associated with this platform's stream
  // implementation, or nullptr otherwise.
  virtual void* GpuStreamHack() { return nullptr; }

  // Returns a pointer to a GPU stream associated with this platform's stream,
  // or a nullptr.
  virtual void** GpuStreamMemberHack() { return nullptr; }

 private:
  StreamInterface(const StreamInterface&) = delete;
  void operator=(const StreamInterface&) = delete;
};

//===----------------------------------------------------------------------===//
// StreamExecutorInterface
//===----------------------------------------------------------------------===//

// Interface for the different StreamExecutor platforms (i.e. CUDA, OpenCL).
//
// Various platforms will provide an implementation that satisfy this interface.
class StreamExecutorInterface {
 public:
  // Default constructor for the abstract interface.
  StreamExecutorInterface() = default;

  // Default destructor for the abstract interface.
  virtual ~StreamExecutorInterface() = default;

  // Returns the (transitively) wrapped executor if this executor is
  // wrapping another executor; otherwise, returns this.
  virtual StreamExecutorInterface* GetUnderlyingExecutor() { return this; }

  // See the StreamExecutor interface for comments on the same-named methods.
  virtual tsl::Status Init(int device_ordinal,
                           DeviceOptions device_options) = 0;

  // This value is cached by the wrapping StreamExecutor instance, so it's OK if
  // this function is slow.
  //
  // The wrapping StreamExecutor will use the platform name if this is nullopt.
  virtual std::optional<std::string> MakeDeviceDescriptionStr() const {
    return std::nullopt;
  }

  virtual tsl::Status GetKernel(const MultiKernelLoaderSpec& spec,
                                KernelBase* kernel) {
    return absl::UnimplementedError("Not Implemented");
  }
  virtual bool UnloadModule(ModuleHandle module_handle) { return false; }
  virtual tsl::Status LoadModule(const MultiModuleLoaderSpec& spec,
                                 ModuleHandle* module_handle) {
    return absl::UnimplementedError("Not Implemented");
  }
  virtual tsl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
  CreateOrShareConstant(Stream* stream, const std::vector<uint8_t>& content) {
    return absl::UnimplementedError("Not Implemented");
  }
  virtual tsl::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                             const BlockDim& block_dims, const KernelBase& k,
                             const KernelArgsArrayBase& args) {
    return absl::UnimplementedError("Not Implemented");
  }

  virtual tsl::Status Submit(Stream* stream,
                             const CommandBuffer& command_buffer) {
    return absl::UnimplementedError("Not Implemented");
  }

  // Releases any state associated with the kernel.
  virtual void UnloadKernel(const KernelBase* kernel) {}
  virtual DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) = 0;
  DeviceMemoryBase Allocate(uint64_t size) {
    return Allocate(size, /*memory_space=*/0);
  }
  virtual void* GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset,
                             uint64_t size) = 0;
  virtual void Deallocate(DeviceMemoryBase* mem) = 0;
  // Allocates unified memory space of the given size, if supported.
  // See
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
  // for more details on unified memory.
  virtual void* UnifiedMemoryAllocate(uint64_t size) { return nullptr; }

  // Deallocates unified memory space previously allocated with
  // UnifiedMemoryAllocate.
  virtual void UnifiedMemoryDeallocate(void* mem) {}
  virtual void* HostMemoryAllocate(uint64_t size) = 0;
  virtual void HostMemoryDeallocate(void* mem) = 0;
  virtual bool HostMemoryRegister(void* mem, uint64_t size) = 0;
  virtual bool HostMemoryUnregister(void* mem) = 0;
  virtual bool SynchronizeAllActivity() = 0;
  virtual tsl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                         uint64_t size) = 0;
  virtual tsl::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                        uint64_t size) = 0;
  virtual tsl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                        const void* host_src,
                                        uint64_t size) = 0;
  virtual tsl::Status SynchronousMemcpy(void* host_dst,
                                        const DeviceMemoryBase& gpu_src,
                                        uint64_t size) = 0;
  virtual tsl::Status SynchronousMemcpyDeviceToDevice(
      DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src,
      uint64_t size) = 0;
  virtual tsl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                              uint64_t size) = 0;
  virtual tsl::Status Memset(Stream* stream, DeviceMemoryBase* location,
                             uint8 pattern, uint64_t size) {
    return tsl::errors::Internal("Not implemented");
  }
  virtual tsl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                               uint32_t pattern, uint64_t size) = 0;
  virtual bool Memcpy(Stream* stream, void* host_dst,
                      const DeviceMemoryBase& gpu_src, uint64_t size) = 0;
  virtual bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                      const void* host_src, uint64_t size) = 0;
  virtual bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                                    const DeviceMemoryBase& gpu_src,
                                    uint64_t size) = 0;
  virtual bool HostCallback(Stream* stream,
                            absl::AnyInvocable<tsl::Status() &&> callback) = 0;
  virtual tsl::Status AllocateEvent(Event* event) = 0;
  virtual tsl::Status DeallocateEvent(Event* event) = 0;
  virtual tsl::Status RecordEvent(Stream* stream, Event* event) = 0;
  virtual tsl::Status WaitForEvent(Stream* stream, Event* event) = 0;
  virtual tsl::Status WaitForEventOnExternalStream(std::intptr_t stream,
                                                   Event* event) {
    return absl::UnimplementedError(
        "WaitForEventOnExternalStream not supported on this executor.");
  }
  virtual Event::Status PollForEventStatus(Event* event) = 0;
  virtual bool AllocateStream(Stream* stream) = 0;
  virtual void DeallocateStream(Stream* stream) = 0;
  virtual bool CreateStreamDependency(Stream* dependent, Stream* other) = 0;
  virtual tsl::Status BlockHostUntilDone(Stream* stream) = 0;
  virtual tsl::Status GetStatus(Stream* stream) {
    return absl::UnimplementedError(
        "GetStatus is not supported on this executor.");
  }
  virtual tsl::Status EnablePeerAccessTo(StreamExecutorInterface* other) = 0;
  virtual bool CanEnablePeerAccessTo(StreamExecutorInterface* other) = 0;

  virtual int64_t GetDeviceLoad() { return -1; }

  virtual bool DeviceMemoryUsage(int64_t* free, int64_t* total) const {
    return false;
  }

  // Retrieves device pointer and size for a symbol. The device pointer is
  // stored at mem, and the size is stored at size. Either mem or bytes can be
  // null, however, both of them cannot be null at the same time. To use
  // constant memory in CUDA, GetSymbol has to be used. Returns true if symbol
  // is found.
  //
  // If ModuleHandle is set then we search for `symbol_name` only within the
  // module corresponding to `module_handle`.  Otherwise all loaded modules are
  // searched.
  virtual bool GetSymbol(const std::string& symbol_name,
                         ModuleHandle module_handle, void** mem,
                         size_t* bytes) {
    return false;
  }

  // Creates a new DeviceDescription object. Ownership is transferred to the
  // caller.
  virtual tsl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription() const = 0;

  // Attempts to register the provided TraceListener with the device-specific
  // Executor implementation. When this is called, the PIMPL interface has
  // already taken ownership of the object and is managing the generic tracing
  // events. The device-specific implementation must determine if the passed
  // listener is of a type appropriate for it to trace during registration (and
  // before dispatching events to it).
  // Returns true if the listener was successfully registered, false otherwise.
  // Does not take ownership of listener.
  virtual bool RegisterTraceListener(TraceListener* listener) { return false; }

  // Unregisters the specified listener from the device-specific Executor.
  // Returns true if the listener was successfully registered, false otherwise.
  virtual bool UnregisterTraceListener(TraceListener* listener) {
    return false;
  }

  // Creates a new BlasSupport object, ownership is transferred to the caller.
  //
  // This may return null if the BLAS initialization fails or this object does
  // not support BLAS.
  virtual blas::BlasSupport* CreateBlas() { return nullptr; }

  // Creates a new fft::FftSupport object, ownership is transferred to the
  // caller.
  // This may return null if the FFT initialization fails or this object does
  // not support FFT.
  virtual fft::FftSupport* CreateFft() { return nullptr; }

  // Creates a new DnnSupport object, ownership is transferred to the caller.
  // This may return null if the DNN initialization fails or this object does
  // not support Dnns.
  virtual dnn::DnnSupport* CreateDnn() { return nullptr; }

  // Each call creates a new instance of the platform-specific implementation of
  // the corresponding interface type.
  virtual std::unique_ptr<EventInterface> CreateEventImplementation() = 0;
  virtual std::unique_ptr<KernelInterface> CreateKernelImplementation() = 0;
  virtual std::unique_ptr<StreamInterface> GetStreamImplementation() = 0;

  virtual tsl::StatusOr<std::unique_ptr<CommandBufferInterface>>
  GetCommandBufferImplementation(CommandBuffer::Mode mode) {
    return absl::UnimplementedError("Command buffers are not implemented");
  }

  // Returns the CUDA or ROCm context associated with this StreamExecutor
  // platform implementation.
  //
  // WARNING: checks that the underlying platform is, in fact, CUDA or ROCm,
  // causing a fatal error if it is not. This hack is made available solely for
  // use from distbelief code, which temporarily has strong ties to CUDA or ROCm
  // as a platform.
  virtual void* GpuContextHack() { return nullptr; }

  // Return allocator statistics.
  virtual std::optional<AllocatorStats> GetAllocatorStats() {
    return std::nullopt;
  }

  // If implemented, clears the internal stats except for the `in_use` fields
  // and sets the `peak_bytes_in_use` to be equal to the `bytes_in_use`. Returns
  // true if implemented.
  //
  // REQUIRES: GetAllocatorStats is overridden.
  virtual bool ClearAllocatorStats() { return false; }

  // Clears the compilation cache from volatile memory. Returns OK if no
  // compilation cache exists or if clearing the compilation cache is
  // unsupported. Caches in non-volatile storage are unaffected.
  virtual tsl::Status FlushCompilationCache() { return ::tsl::OkStatus(); }

  // Returns a stream allocated by this executor, or nullptr if not found.
  // Performs linear search over alive GPU streams.
  virtual Stream* FindAllocatedStream(void* /*gpu_stream*/) { return nullptr; }

 private:
  StreamExecutorInterface(const StreamExecutorInterface&) = delete;
  void operator=(const StreamExecutorInterface&) = delete;
};

}  // namespace internal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
