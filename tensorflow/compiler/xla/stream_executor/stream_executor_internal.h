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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/stream_executor/allocator_stats.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/device_options.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/stream_executor/event.h"
#include "tensorflow/compiler/xla/stream_executor/kernel.h"
#include "tensorflow/compiler/xla/stream_executor/kernel_cache_config.h"
#include "tensorflow/compiler/xla/stream_executor/kernel_spec.h"
#include "tensorflow/compiler/xla/stream_executor/launch_dim.h"
#include "tensorflow/compiler/xla/stream_executor/lib/status.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/module_spec.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/plugin_registry.h"
#include "tensorflow/compiler/xla/stream_executor/trace_listener.h"

namespace stream_executor {

class Stream;
class Timer;

// An opaque handle to a loaded module.
//
// An instance of this is returned from StreamExecutor::GetModule.
class ModuleHandle {
 public:
  /*implicit*/ ModuleHandle(void* id = nullptr) : id_(id) {}

  // A ModuleHandle with id() == nullptr is an invalid module handle, akin to a
  // null pointer.
  void* id() const { return id_; }

  explicit operator bool() const { return id() != nullptr; }

 private:
  void* id_;
};

namespace internal {

// Platform-dependent interface class for the generic Events interface, in
// the PIMPL style.
class EventInterface {
 public:
  EventInterface() {}
  virtual ~EventInterface() {}

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(EventInterface);
};

// Pointer-to-implementation object type (i.e. the KernelBase class delegates to
// this interface) with virtual destruction. This class exists for the
// platform-dependent code to hang any kernel data/resource info/functionality
// off of.
class KernelInterface {
 public:
  // Default constructor for the abstract interface.
  KernelInterface() {}

  // Default destructor for the abstract interface.
  virtual ~KernelInterface() {}

  // Returns the number of formal parameters that this kernel accepts.
  virtual unsigned Arity() const = 0;

  // Sets the preferred cache configuration.
  virtual void SetPreferredCacheConfig(KernelCacheConfig config) = 0;

  // Gets the preferred cache configuration.
  virtual KernelCacheConfig GetPreferredCacheConfig() const = 0;

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(KernelInterface);
};

// Pointer-to-implementation object type (i.e. the Stream class delegates to
// this interface) with virtual destruction. This class exists for the
// platform-dependent code to hang any kernel data/resource info/functionality
// off of.
class StreamInterface {
 public:
  // Default constructor for the abstract interface.
  StreamInterface() {}

  // Default destructor for the abstract interface.
  virtual ~StreamInterface() {}

  // Returns the GPU stream associated with this platform's stream
  // implementation, or nullptr otherwise.
  virtual void* GpuStreamHack() { return nullptr; }

  // Returns a pointer to a GPU stream associated with this platform's stream,
  // or a nullptr.
  virtual void** GpuStreamMemberHack() { return nullptr; }

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(StreamInterface);
};

// Pointer-to-implementation object type (i.e. the Timer class delegates to
// this interface) with virtual destruction. This class exists for the
// platform-dependent code to hang any timer data/resource info/functionality
// off of.
class TimerInterface {
 public:
  // Default constructor for the abstract interface.
  TimerInterface() {}

  // Default destructor for the abstract interface.
  virtual ~TimerInterface() {}

  // Returns the number of microseconds elapsed in a completed timer.
  virtual uint64_t Microseconds() const = 0;

  // Returns the number of nanoseconds elapsed in a completed timer.
  virtual uint64_t Nanoseconds() const = 0;

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(TimerInterface);
};

// Interface for the different StreamExecutor platforms (i.e. CUDA, OpenCL).
//
// Various platforms will provide an implementation that satisfy this interface.
class StreamExecutorInterface {
 public:
  // Default constructor for the abstract interface.
  StreamExecutorInterface() {}

  // Default destructor for the abstract interface.
  virtual ~StreamExecutorInterface() {}

  // Returns the (transitively) wrapped executor if this executor is
  // wrapping another executor; otherwise, returns this.
  virtual StreamExecutorInterface* GetUnderlyingExecutor() { return this; }

  // See the StreamExecutor interface for comments on the same-named methods.
  virtual port::Status Init(int device_ordinal,
                            DeviceOptions device_options) = 0;

  // This value is cached by the wrapping StreamExecutor instance, so it's OK if
  // this function is slow.
  //
  // The wrapping StreamExecutor will use the platform name if this is nullopt.
  virtual std::optional<std::string> MakeDeviceDescriptionStr() const {
    return std::nullopt;
  }

  virtual port::Status GetKernel(const MultiKernelLoaderSpec& spec,
                                 KernelBase* kernel) {
    return port::UnimplementedError("Not Implemented");
  }
  virtual bool UnloadModule(ModuleHandle module_handle) { return false; }
  virtual port::Status LoadModule(const MultiModuleLoaderSpec& spec,
                                  ModuleHandle* module_handle) {
    return port::UnimplementedError("Not Implemented");
  }
  virtual port::StatusOr<std::shared_ptr<DeviceMemoryBase>>
  CreateOrShareConstant(Stream* stream, const std::vector<uint8_t>& content) {
    return port::UnimplementedError("Not Implemented");
  }
  virtual port::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                              const BlockDim& block_dims, const KernelBase& k,
                              const KernelArgsArrayBase& args) {
    return port::UnimplementedError("Not Implemented");
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
  virtual port::Status SynchronousMemZero(DeviceMemoryBase* location,
                                          uint64_t size) = 0;
  virtual port::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                         uint64_t size) = 0;
  virtual port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                         const void* host_src,
                                         uint64_t size) = 0;
  virtual port::Status SynchronousMemcpy(void* host_dst,
                                         const DeviceMemoryBase& gpu_src,
                                         uint64_t size) = 0;
  virtual port::Status SynchronousMemcpyDeviceToDevice(
      DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src,
      uint64_t size) = 0;
  virtual port::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                               uint64_t size) = 0;
  virtual port::Status Memset(Stream* stream, DeviceMemoryBase* location,
                              uint8 pattern, uint64_t size) {
    return port::InternalError("Not implemented");
  }
  virtual port::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                                uint32_t pattern, uint64_t size) = 0;
  virtual bool Memcpy(Stream* stream, void* host_dst,
                      const DeviceMemoryBase& gpu_src, uint64_t size) = 0;
  virtual bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                      const void* host_src, uint64_t size) = 0;
  virtual bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                                    const DeviceMemoryBase& gpu_src,
                                    uint64_t size) = 0;
  virtual bool HostCallback(Stream* stream, std::function<void()> callback);
  virtual bool HostCallback(Stream* stream,
                            std::function<port::Status()> callback) = 0;
  virtual port::Status AllocateEvent(Event* event) = 0;
  virtual port::Status DeallocateEvent(Event* event) = 0;
  virtual port::Status RecordEvent(Stream* stream, Event* event) = 0;
  virtual port::Status WaitForEvent(Stream* stream, Event* event) = 0;
  virtual Event::Status PollForEventStatus(Event* event) = 0;
  virtual bool AllocateStream(Stream* stream) = 0;
  virtual void DeallocateStream(Stream* stream) = 0;
  virtual bool CreateStreamDependency(Stream* dependent, Stream* other) = 0;
  virtual bool AllocateTimer(Timer* timer) = 0;
  virtual void DeallocateTimer(Timer* timer) = 0;
  virtual bool StartTimer(Stream* stream, Timer* timer) = 0;
  virtual bool StopTimer(Stream* stream, Timer* timer) = 0;
  virtual port::Status BlockHostUntilDone(Stream* stream) = 0;
  virtual port::Status GetStatus(Stream* stream) {
    return port::Status(port::error::UNIMPLEMENTED,
                        "GetStatus is not supported on this executor.");
  }
  virtual int PlatformDeviceCount() = 0;
  virtual port::Status EnablePeerAccessTo(StreamExecutorInterface* other) = 0;
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
  virtual port::StatusOr<std::unique_ptr<DeviceDescription>>
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

  // Returns whether this StreamExecutor has BLAS support for its underlying
  // platform.
  virtual bool SupportsBlas() const { return false; }

  // Creates a new BlasSupport object, ownership is transferred to the caller.
  // If SupportsBlas() is false, this will always return null.
  //
  // If SupportsBlas() is true, this may return null, for example, if the BLAS
  // initialization fails.
  virtual blas::BlasSupport* CreateBlas() { return nullptr; }

  // Returns whether this StreamExecutor has FFT support for its underlying
  // platform.
  virtual bool SupportsFft() const { return false; }

  // Creates a new fft::FftSupport object, ownership is transferred to the
  // caller.
  // If SupportsFft() is false, this will always return null.
  //
  // If SupportsFft() is true, this may return null, for example, if the FFT
  // initialization fails.
  virtual fft::FftSupport* CreateFft() { return nullptr; }

  // Returns whether this StreamExecutor has Random Number Generation support
  // for
  // its underlying platform.
  virtual bool SupportsRng() const { return false; }

  // Returns whether this StreamExecutor has neural net support for its
  // underlying
  // platform.
  virtual bool SupportsDnn() const { return false; }

  // Creates a new RngSupport object, ownership is transferred to the caller.
  // If SupportsRng() is false, this will always return null.
  //
  // If SupportsRng() is true, this may return null, for example, if the RNG
  // initialization fails.
  virtual rng::RngSupport* CreateRng() { return nullptr; }

  // Creates a new DnnSupport object, ownership is transferred to the caller.
  // If SupportsDnn() is false, this will always return null.
  //
  // If SupportsDnn() is true, this may return null, for example, if the DNN
  // initialization fails.
  virtual dnn::DnnSupport* CreateDnn() { return nullptr; }

  // Each call creates a new instance of the platform-specific implementation of
  // the corresponding interface type.
  virtual std::unique_ptr<EventInterface> CreateEventImplementation() = 0;
  virtual std::unique_ptr<KernelInterface> CreateKernelImplementation() = 0;
  virtual std::unique_ptr<StreamInterface> GetStreamImplementation() = 0;
  virtual std::unique_ptr<TimerInterface> GetTimerImplementation() = 0;

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
  virtual port::Status FlushCompilationCache() { return ::tsl::OkStatus(); }

  // Returns a stream allocated by this executor, or nullptr if not found.
  // Performs linear search over alive GPU streams.
  virtual Stream* FindAllocatedStream(void* /*gpu_stream*/) { return nullptr; }

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(StreamExecutorInterface);
};

}  // namespace internal
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
