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

#ifndef TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
#define TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_

#include <atomic>
#include <set>
#include <tuple>
#include <vector>

#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/shared_memory_config.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/trace_listener.h"

namespace perftools {
namespace gputools {

// Structure used for device memory leak checking.
struct AllocRecord {
  // The requested allocation size of the buffer.
  uint64 bytes;

  // Holds a representation of the stack at the time the associated buffer was
  // allocated. Produced in a form described in
  // //util/symbolize/symbolized_stacktrace.h.
  string stack_trace;
};

// Forward declaration of private friend class.
template <typename BeginCallT, typename CompleteCallT,
          typename ReturnT, typename... BeginArgsT>
class ScopedTracer;

// A StreamExecutor manages a single device, in terms of executing work (kernel
// launches) and memory management (allocation/deallocation, memory copies to
// and from the device). It is conceptually the "handle" for a device -- Stream
// objects, which are used to enqueue work to run on the
// coprocessor have a StreamExecutor instance as their "parent" object.
//
// StreamExecutor objects have an underlying platform that is specified up
// front;
// e.g. either it is a CUDA or OpenCL executor.
//
// Thread-safe after initialization.
// StreamExecutor interface should not be invoked from a signal handler.
class StreamExecutor {
 public:
  explicit StreamExecutor(PlatformKind kind,
                          const PluginConfig &plugin_config = PluginConfig());
  StreamExecutor(const Platform *platform,
                 internal::StreamExecutorInterface *implementation);

  ~StreamExecutor();

  port::Status Init();
  port::Status Init(int device_ordinal, DeviceOptions device_options);

  // DEPRECATED: Do not use; use platform() instead.
  // Returns the platform that this StreamExecutor is acting upon.
  PlatformKind platform_kind() const { return platform_kind_; }

  // Returns a reference to the platform that created this executor.
  const Platform *platform() const { return platform_; }

  // Retrieves (loads) a kernel for the platform this StreamExecutor is acting
  // upon, if one exists.
  //
  // Parameters:
  //   spec: The MultiKernelLoaderSpec is usually generated as a compile-time
  //    constant into an appropriate namespace. For example, see
  //    perftools::gputools::executor_sample::kKernelLoaderSpecs, from which a
  //    MultiKernelLoaderSpec is selected.
  //   kernel: Outparam that the kernel is loaded into. A given Kernel
  //    instantiation should not be loaded into more than once.
  //
  // If an error occurs, or there is no kernel available for the StreamExecutor
  // platform, false is returned.
  bool GetKernel(const MultiKernelLoaderSpec &spec, KernelBase *kernel);

  // Synchronously allocates an array on the GPU device of type T with
  // element_count elements.
  template <typename T>
  DeviceMemory<T> AllocateArray(uint64 element_count);

  // As AllocateArray(), but returns a ScopedDeviceMemory<T>.
  template <typename T>
  ScopedDeviceMemory<T> AllocateOwnedArray(uint64 element_count) {
    return ScopedDeviceMemory<T>(this, AllocateArray<T>(element_count));
  }

  // Convenience wrapper that allocates space for a single element of type T
  // in GPU memory.
  template <typename T>
  DeviceMemory<T> AllocateScalar() {
    return AllocateArray<T>(1);
  }

  // As AllocateScalar(), but returns a ScopedDeviceMemory<T>.
  template <typename T>
  ScopedDeviceMemory<T> AllocateOwnedScalar() {
    return AllocateOwnedArray<T>(1);
  }

  // Synchronously allocates a scalar of type T on the GPU device that is
  // (POD) zero-byte initialized.
  template <typename T>
  DeviceMemory<T> AllocateZeroed();

  // As AllocateZeroed(), but returns a ScopedDeviceMemory<T>.
  template <typename T>
  ScopedDeviceMemory<T> AllocateOwnedZeroed() {
    return ScopedDeviceMemory<T>(this, AllocateZeroed<T>());
  }

  // Allocate a memory region inside another allocated memory region.
  // Offset and size are specified in terms of T elements.
  // Warning: Do not free a parent buffer before its sub-buffers; this may cause
  // use-after-free issues (the specific behavior is not consistent across
  // platforms).
  //  - Note: OpenCL uses refcounting to manage buffer lifetimes, so use of a
  //    sub-buffer after parent deallocation is expected to be safe. This will
  //    render your code non-platform-portable, however.
  template <typename T>
  DeviceMemory<T> AllocateSubBuffer(DeviceMemory<T> *parent,
                                    uint64 element_offset,
                                    uint64 element_count);

  // As AllocateSubBuffer(), but returns a ScopedDeviceMemory<T>.
  template <typename T>
  ScopedDeviceMemory<T> AllocateOwnedSubBuffer(DeviceMemory<T> *parent,
                                               uint64 element_offset,
                                               uint64 element_count) {
    return ScopedDeviceMemory<T>(
        this, AllocateSubBuffer<T>(parent, element_offset, element_count));
  }

  // Finds a symbol and returns device memory allocated to the symbol. The
  // symbol is searched in any kernels that were previously loaded through
  // GetKernel() before the GetSymbol() call. The user has to make sure that the
  // type of symbol and T match.
  // - Note: symbol_name should include its namespace as well. For example,
  //         pass "nms0::symbol" if referring to nms0::symbol.
  template <typename T>
  port::StatusOr<DeviceMemory<T>> GetSymbol(const string &symbol_name);

  // Deallocate the DeviceMemory previously allocated via this interface.
  // Deallocation of a nullptr-representative value is permitted.
  //
  // Resets the internal contents of mem to be null-representative, but this
  // null-out effect should not be relied upon in client code.
  void Deallocate(DeviceMemoryBase *mem);

  // Retrieves a mapping of active opaque GPU memory pointer to a string
  // representation of the [allocating thread's] stack at the time the pointer
  // was allocated. Useful for tracking GPU memory leaks.
  //
  // Note: this will only be populated if --check_gpu_leaks flag is activated.
  void GetMemAllocs(std::map<void *, AllocRecord> *records_out);

  // Allocates a region of host memory and registers it with the platform API.
  // Memory allocated in this manner (or allocated and registered with
  // HostMemoryRegister() is required for use in asynchronous memcpy operations,
  // such as Stream::ThenMemcpy.
  void *HostMemoryAllocate(uint64 bytes);

  // Deallocates a region of host memory allocated by HostMemoryAllocate().
  void HostMemoryDeallocate(void *location);

  // Registers a region of host memory with the platform API. Registered memory
  // (or memory allocated with HostMemoryAllocate) is required for use with
  // asynchronous memcpy operations, such as Stream::ThenMemcpy. This method
  // is used to register memory allocated outside the StreamExecutor;
  // HostMemoryAllocate implicitly registers its allocations and
  // HostMemoryDeallocate implicitly deregisters on deallocation.
  bool HostMemoryRegister(void *location, uint64 size) SE_MUST_USE_RESULT;

  // Unregisters a region of host memory registered with HostMemoryRegister.
  // This should be done before deallocating the region with delete[]/free/etc.
  bool HostMemoryUnregister(void *location) SE_MUST_USE_RESULT;

  // Synchronizes all activity occuring in the StreamExecutor's context (most
  // likely a whole device).
  bool SynchronizeAllActivity() SE_MUST_USE_RESULT;

  // Blocks the caller while "size" bytes are zeroed out (in POD fashion) at the
  // given location in GPU memory.
  bool SynchronousMemZero(DeviceMemoryBase *location,
                          uint64 size) SE_MUST_USE_RESULT;

  // Blocks the caller while "size" bytes are initialized to "value" (in POD
  // fashion) at the given location in GPU memory.
  bool SynchronousMemSet(DeviceMemoryBase *location, int value,
                         uint64 size) SE_MUST_USE_RESULT;

  // [deprecated] Blocks the caller while a data segment of the given size is
  // copied from the host source to the GPU destination.
  //
  // Deprecation: prefer explicit H2D below, to avoid error-prone API usage.
  bool SynchronousMemcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                         uint64 size) SE_MUST_USE_RESULT;

  // [deprecated] Blocks the caller while a data segment of the given size is
  // copied from the GPU source to the host destination.
  //
  // Deprecation: prefer explicit D2H below, to avoid error-prone API usage.
  bool SynchronousMemcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                         uint64 size) SE_MUST_USE_RESULT;

  // Same as SynchronousMemcpy(DeviceMemoryBase*, ...) above.
  port::Status SynchronousMemcpyH2D(const void *host_src, int64 size,
                                    DeviceMemoryBase *gpu_dst);

  // Alternative interface for memcpying from host to device that takes an
  // array slice. Checks that the destination size can accomodate the host
  // slice size.
  template <class T>
  port::Status SynchronousMemcpyH2D(port::ArraySlice<T> host_src,
                                    DeviceMemoryBase *gpu_dst) {
    auto host_size = host_src.size() * sizeof(T);
    CHECK(gpu_dst->size() == 0 || gpu_dst->size() >= host_size);
    return SynchronousMemcpyH2D(host_src.begin(), host_size, gpu_dst);
  }

  // Same as SynchronousMemcpy(void*, ...) above.
  port::Status SynchronousMemcpyD2H(const DeviceMemoryBase &gpu_src, int64 size,
                                    void *host_dst);

  // Alternative interface for memcpying from device to host that takes an
  // array slice. Checks that the destination size can accomodate the host
  // slice size.
  template <typename T>
  port::Status SynchronousMemcpyD2H(const DeviceMemory<T> &gpu_src,
                                    port::MutableArraySlice<T> host_dst) {
    auto host_size = host_dst.size() * sizeof(T);
    CHECK(gpu_src.size() == 0 || host_size >= gpu_src.size());
    return SynchronousMemcpyD2H(gpu_src, host_size, host_dst.begin());
  }

  // Blocks the caller while a data segment of the given size is copied from the
  // GPU source to the GPU destination.
  bool SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                         const DeviceMemoryBase &gpu_src,
                         uint64 size) SE_MUST_USE_RESULT;

  // Enqueues an operation onto stream to zero out size bytes at the given GPU
  // memory location. Neither stream nor location may be null. Returns whether
  // the operation was successfully enqueued onto the stream.
  bool MemZero(Stream *stream, DeviceMemoryBase *location,
               uint64 size) SE_MUST_USE_RESULT;

  // Enqueues an operation onto stream to set 32-bit patterns starting at
  // location, for byte count given by size. size must be 32-bit quantified
  // (i.e. evently divisible by 4). Returns whether the operation was
  // successfully enqueued onto the stream.
  bool Memset32(Stream *stream, DeviceMemoryBase *location, uint32 pattern,
                uint64 size) SE_MUST_USE_RESULT;

  // Enables peer access from this StreamExecutor to memory
  // allocated by other, such that launched device code, memcpies, etc may
  // access it directly.
  //
  // Both this StreamExecutor and other must be backed by the same platform (as
  // in
  // CUDA vs OpenCL) implementation.
  port::Status EnablePeerAccessTo(StreamExecutor *other);

  // Returns whether it's possible to enable peer access from this
  // StreamExecutor
  // to memory allocated by another.
  //
  // Even when this returns true, EnablePeerAccessTo may fail for other reasons;
  // this is more an up-front test as to whether it's expressly forbidden.
  bool CanEnablePeerAccessTo(StreamExecutor *other);

  // Gets the preferred shared memory configuration for the device to which this
  // executor is bound.
  SharedMemoryConfig GetDeviceSharedMemoryConfig();

  // Sets the preferred shared memory configuration for the device to which this
  // executor is bound.
  port::Status SetDeviceSharedMemoryConfig(SharedMemoryConfig config);

  // Obtains metadata about the underlying device.
  // The value is cached on first use.
  const DeviceDescription &GetDeviceDescription() const;

  // Returns the underlying device memory usage information, if it is available.
  // If it is not available (false is returned), free/total may not be
  // initialized.
  //
  // Note: "Free" reflects the amount of free memory on the underlying device,
  // so allocations via other StreamExecutors that have the same underlying
  // device
  // will be reflected in "free".
  bool DeviceMemoryUsage(int64 *free, int64 *total) const;

  // The device count reported by this StreamExecutor's platform.
  // Note: on OpenCL we implicitly select platform zero at the moment.
  int PlatformDeviceCount() const;

  // Returns whether the StreamExecutor supports BLAS routines for the platform
  // that underlies this interface.
  bool SupportsBlas() const;

  // Returns whether the StreamExecutor supports FFT routines for the platform
  // that underlies this interface.
  bool SupportsFft() const;

  // Returns whether the StreamExecutor supports RNG routines for the platform
  // that underlies this interface.
  bool SupportsRng() const;

  // Returns whether the StreamExecutor support neural net routines for the
  // platform that underlies this interface.
  bool SupportsDnn() const;

  // Get the list of supported algorithms for the forward convolution opeartion.
  bool GetConvolveAlgorithms(std::vector<dnn::AlgorithmType> *out_algorithms);

  // Get the list of supported algorithms for the backward convolution on data.
  bool GetConvolveBackwardDataAlgorithms(
      std::vector<dnn::AlgorithmType> *out_algorithms);

  // Get the list of supported algorithms for the backward convolution on the
  // filter.
  bool GetConvolveBackwardFilterAlgorithms(
      std::vector<dnn::AlgorithmType> *out_algorithms);

  // Create an RNN descriptor based on model shapes and configurations.
  // The caller retains the ownership of the descriptor.
  port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>> createRnnDescriptor(
      int num_layers, int hidden_size, int input_size,
      dnn::RnnInputMode input_mode, dnn::RnnDirectionMode direction_mode,
      dnn::RnnMode rnn_mode, dnn::DataType data_type, float dropout,
      uint64 seed, ScratchAllocator *state_allocator);

  // Create a RNN sequence descriptor that specifies either the input or output
  // sequence. The caller retains the ownership of the returned descriptor.
  port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  createRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                    int data_size, dnn::DataType data_type);

  // Create an RNN state descriptor that specifies the input or hidden state.
  // The caller retains the ownership of the returned descriptor.
  port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
  createRnnStateTensorDescriptor(int num_layer, int batch_size, int data_size,
                                 dnn::DataType data_type);

  // Returns the device ordinal that this StreamExecutor was initialized with.
  // Meaningless before initialization.
  int device_ordinal() const { return device_ordinal_; }

  // Returns a borrowed pointer to the underlying StreamExecutor implementation.
  internal::StreamExecutorInterface *implementation();

  // Warning: use Stream::ThenLaunch instead, this method is not for general
  // consumption. However, this is the only way to launch a kernel for which
  // the type signature is only known at runtime; say, if an application
  // supports loading/launching kernels with arbitrary type signatures.
  // In this case, the application is expected to know how to do parameter
  // packing that obeys the contract of the underlying platform implementation.
  //
  // Launches a data parallel kernel with the given thread/block
  // dimensionality and already-packed args/sizes to pass to the underlying
  // platform driver.
  //
  // This is called by Stream::Launch() to delegate to the platform's launch
  // implementation in StreamExecutorInterface::Launch().
  bool Launch(Stream *stream, const ThreadDim &thread_dims,
              const BlockDim &block_dims, const KernelBase &kernel,
              const std::vector<KernelArg> &args);

  // Gets-or-creates (creates with memoization) a FftSupport datatype that can
  // be used to execute FFT routines on the current platform.
  //
  // Ownership and user-facing is the same as AsBlas() below.
  //
  // Returns null if there was an error initializing the FFT support for the
  // underlying platform.
  fft::FftSupport *AsFft();

  // Gets-or-creates (creates with memoization) a DnnSupport datatype that can
  // be used for neural network routines on the current platform.
  //
  // Ownership and user-facing is the same as AsBlas() below.
  //
  // Returns null if there was an error initializing the DNN support for the
  // underlying platform.
  dnn::DnnSupport *AsDnn();

  // Turns StreamExecutor operation tracing on or off.
  void EnableTracing(bool enable);

  // Registers a trace listener to receive callbacks for only a single
  // StreamExecutor instance.
  // To register a listener for all executors for a given platform, see
  // Platform::RegisterTraceListener().
  // Does not take ownership of listener.
  void RegisterTraceListener(TraceListener* listener);

  // Removes a TraceListener from this StreamExecutor instance.
  // Returns false (and logs) in cases where the argument listener was not
  // previously registered.
  bool UnregisterTraceListener(TraceListener* listener);

  // Converts a DeviceMemory object into a KernelArg object for passing to the
  // device driver for kernel launch.
  KernelArg DeviceMemoryToKernelArg(const DeviceMemoryBase &gpu_mem) const;

 private:
  template <typename BeginCallT, typename CompleteCallT,
            typename ReturnT, typename... BeginArgsT>
  friend class ScopedTracer;
  friend class Event;
  friend class Stream;
  friend class Timer;
  template <typename... Params>
  friend class TypedKernel;
  template <typename... Args>
  friend struct ThenBlasImpl;

  // Gets-or-creates (creates with memoization) a BlasSupport datatype that can
  // be used to execute BLAS routines on the current platform. This is typically
  // not user-facing, as users will use the Stream::ThenBlas* family of routines
  // to entrain BLAS operations. See blas.h for additional details.
  //
  // Ownership is not transferred to the caller -- ownership is retained by this
  // object for memoization. This BLAS interface is also only expected to be
  // used by a Stream for entraining calls to BLAS functionality.
  //
  // Returns null if there was an error initializing the BLAS support for the
  // underlying platform.
  blas::BlasSupport *AsBlas();

  // Gets-or-creates (creates with memoization) an RngSupport datatype that can
  // be used for random-number-generation routines on the current platform.
  //
  // Ownership and user-facing is the same as AsBlas() above.
  //
  // Returns null if there was an error initializing the RNG support for the
  // underlying platform.
  rng::RngSupport *AsRng();

  // Causes the host code to synchronously wait for operations entrained onto
  // stream to complete. Effectively a join on the asynchronous GPU operations
  // enqueued on the stream before this program point.
  bool BlockHostUntilDone(Stream *stream);

  // Synchronously allocates size bytes on the underlying platform and returns
  // an opaque void* representing that allocation. In the case of failure,
  // nullptr is returned.
  void *Allocate(uint64 size);

  // Finds and retrieves device memory for the symbol on the underlying
  // platform.
  bool GetSymbol(const string& symbol_name, void **mem, size_t *bytes);

  // Entrains a memcpy operation onto stream, with a host destination location
  // host_dst and a GPU memory source, with target size size.
  bool Memcpy(Stream *stream, void *host_dst, const DeviceMemoryBase &gpu_src,
              uint64 size);

  // Entrains a memcpy operation onto stream, with a GPU destination location
  // and a host memory source, with target size size.
  bool Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst, const void *host_src,
              uint64 size);

  // Entrains a memcpy operation onto stream, with a GPU destination location
  // and a GPU source location, with target size size. Peer access should have
  // been enabled between the StreamExecutors owning the GPU memory regions.
  bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *gpu_dst,
                            const DeviceMemoryBase &gpu_src, uint64 size);

  // Entrains on a stream a user-specified function to be run on the host.
  // See Stream::ThenDoHostCallback for full details.
  bool HostCallback(Stream *stream, std::function<void()> callback);

  // Performs platform-specific allocation and initialization of an event.
  port::Status AllocateEvent(Event *event);

  // Performs platform-specific deallocation and cleanup of an event.
  port::Status DeallocateEvent(Event *event);

  // Inserts the specified event at the end of the specified stream.
  port::Status RecordEvent(Stream *stream, Event *event);

  // Wait for the specified event at the end of the specified stream.
  port::Status WaitForEvent(Stream *stream, Event *event);

  // Requests the current status of the event from the underlying platform.
  Event::Status PollForEventStatus(Event *event);

  // Allocates stream resources on the underlying platform for subject and
  // initializes its internals.
  bool AllocateStream(Stream *subject);

  // Deallocates stream resources on the underlying platform.
  void DeallocateStream(Stream *subject);

  // Causes dependent to not begin execution until other has finished its
  // last-enqueued work.
  bool CreateStreamDependency(Stream *dependent, Stream *other);

  // Allocates timer resources on the underlying platform for subject and
  // initializes its internals.
  bool AllocateTimer(Timer *subject);

  // Deallocates timer resources on the underlying platform.
  void DeallocateTimer(Timer *subject);

  // Records a start event for an interval timer.
  bool StartTimer(Stream *stream, Timer *timer);

  // Records a stop event for an interval timer.
  bool StopTimer(Stream *stream, Timer *timer);

  // Allocates a new metadata object, appropriately populated, on the heap, with
  // ownership transfer to caller.
  DeviceDescription *PopulateDeviceDescription() const;

  // Adds a task to the port::ThreadPool work queue. These tasks must be
  // fire-and-forget and have no external data or timing dependencies; their
  // execution order and completion time have no guarantees.
  // For an example of an appropriate task, see HostBlas::DoBlasGemmInternal;
  // there, temporary internal buffers are freed using this method.
  void EnqueueOnBackgroundThread(std::function<void()> task);

  // Adds an AllocRecord for 'opaque' of size 'bytes' to the record map, for
  // leak checking. NULL buffer pointers and buffer sizes of 0 will not be
  // tracked.
  void CreateAllocRecord(void *opaque, uint64 size);

  // Removes the AllocRecord keyed by 'opaque' from the record map. NULL
  // pointers will not be erased (as they're not tracked, per above).
  void EraseAllocRecord(void *opaque);

  // Calls the relevant TraceListener routine to begin tracing for the specified
  // asynchronous method.
  template <typename TraceCallT, typename... ArgsT>
  void SubmitTrace(TraceCallT trace_call, ArgsT&&... args);

  // Reader/writer lock for class-static StreamExecutor members.
  static mutex static_mu_;

  // Reader/writer lock for mutable data structures on this StreamExecutor.
  //
  // Mutable so that caching functions (like DeviceDescription, AsBlas, etc.)
  // can acquire the lock on their first (mutating) call as well.
  mutable mutex mu_;

  // Reference to the platform that created this executor.
  const Platform *platform_;

  // Pointer to the platform-specific-interface implementation. This is
  // delegated to by the interface routines in pointer-to-implementation
  // fashion.
  std::unique_ptr<internal::StreamExecutorInterface> implementation_;

  // A mapping of pointer (to GPU memory) to string representation of the stack
  // (of the allocating thread) at the time at which the pointer was allocated.
  std::map<void *, AllocRecord> mem_allocs_ GUARDED_BY(mu_);

  // Memoized BLAS support object -- we only want to create this once when asked
  // for a BLAS interface.
  std::unique_ptr<blas::BlasSupport> blas_ GUARDED_BY(mu_);

  // Memoized DNN support object -- we only want to create this once when asked
  // for an DNN interface.
  std::unique_ptr<dnn::DnnSupport> dnn_ GUARDED_BY(mu_);

  // Memoized FFT support object -- we only want to create this once when asked
  // for a FFT interface.
  std::unique_ptr<fft::FftSupport> fft_;

  // Memoized RNG support object -- we only want to create this once when asked
  // for an RNG interface.
  std::unique_ptr<rng::RngSupport> rng_ GUARDED_BY(mu_);

  // Slot to cache the owned DeviceDescription for the underlying device
  // once it has been quieried from DeviceDescription().
  mutable std::unique_ptr<DeviceDescription> device_description_
      GUARDED_BY(mu_);

  // The kind of the underlying platform that is being targeted, as passed
  // during construction.
  //
  // Immutable post-initialization.
  PlatformKind platform_kind_;

  // The device ordinal that this object was initialized with.
  //
  // Immutable post-initialization.
  int device_ordinal_;

  // Executor for handling host callback work that cannot be performed
  // by a host callback thread - for example, cleanup after a host BLAS routine
  // (which may make device API calls). This work cannot block the host
  // callback thread, will be completed asynchronously, and should be treated
  // as fire-and-forget. Assume no ordering guarantees WRT the tasks enqueued
  // here.
  //
  // Immutable post-initialization. Object is thread-safe.
  std::unique_ptr<port::ThreadPool> background_threads_;

  // Counter for the current number of live streams. This is used to check
  // for accidentally-outstanding streams at StreamExecutor teardown time, as
  // well
  // as to indicate leaks (via a large outstanding count being logged) in the
  // case we can't allocate more streams.
  std::atomic_int_fast32_t live_stream_count_;

  // Only one worker thread is needed; little work will be done by the
  // executor.
  static const int kNumBackgroundThreads = 1;

  // Indicates if StreamExecutor operation tracing should be performed.
  bool tracing_enabled_;

  // The set of TraceListeners registered for this StreamExecutor.
  std::set<TraceListener*> listeners_ GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(StreamExecutor);
};

////////////
// Inlines

template <typename T>
inline DeviceMemory<T> StreamExecutor::AllocateArray(uint64 element_count) {
  uint64 bytes = sizeof(T) * element_count;
  void *opaque = Allocate(bytes);
  return DeviceMemory<T>::MakeFromByteSize(opaque, bytes);
}

template <typename T>
inline port::StatusOr<DeviceMemory<T>> StreamExecutor::GetSymbol(
    const string &symbol_name) {
  // If failed to get the symbol, opaque/bytes are unchanged. Initialize them to
  // be nullptr/0 for consistency with DeviceMemory semantics.
  void *opaque = nullptr;
  size_t bytes = 0;
  if (GetSymbol(symbol_name, &opaque, &bytes)) {
    CHECK_EQ(bytes % sizeof(T), 0);
    return DeviceMemory<T>::MakeFromByteSize(opaque, bytes);
  }
  return port::Status(
      port::error::NOT_FOUND,
      port::StrCat("Check if kernel using the symbol is loaded: ",
                   symbol_name));
}

template <typename ElemT>
ScopedDeviceMemory<ElemT>::ScopedDeviceMemory(StreamExecutor *parent,
                                              DeviceMemoryBase value)
    : wrapped_(value), parent_(parent) {}

template <typename ElemT>
ScopedDeviceMemory<ElemT>::ScopedDeviceMemory(
    StreamExecutor *parent, std::initializer_list<ElemT> values)
    : ScopedDeviceMemory(parent, parent->AllocateArray<ElemT>(values.size())) {
  if (ptr() != nullptr) {
    std::vector<ElemT> local(values);
    if (!parent->SynchronousMemcpy(ptr(), const_cast<const ElemT *>(&local[0]),
                                   ptr()->size())) {
      Reset(nullptr);
    }
  }
}

template <typename ElemT>
ScopedDeviceMemory<ElemT>::~ScopedDeviceMemory() {
  parent_->Deallocate(&wrapped_);
}

template <typename ElemT>
void ScopedDeviceMemory<ElemT>::Reset(DeviceMemory<ElemT> updated) {
  parent_->Deallocate(&wrapped_);
  wrapped_ = updated;
}

template <typename ElemT>
void ScopedDeviceMemory<ElemT>::Reset(std::nullptr_t) {
  parent_->Deallocate(&wrapped_);
  wrapped_ = DeviceMemory<ElemT>{};
}

template <typename T>
DeviceMemory<T> StreamExecutor::AllocateZeroed() {
  void *opaque = Allocate(sizeof(T));
  if (opaque == nullptr) {
    return DeviceMemory<T>{};
  }

  DeviceMemory<T> result = DeviceMemory<T>::MakeFromByteSize(opaque, sizeof(T));
  bool ok = SynchronousMemZero(&result, sizeof(T));
  if (!ok) {
    Deallocate(&result);
    return DeviceMemory<T>{};
  }

  return result;
}

template <typename T>
DeviceMemory<T> StreamExecutor::AllocateSubBuffer(DeviceMemory<T> *parent,
                                                  uint64 element_offset,
                                                  uint64 element_count) {
  if (element_offset + element_count > parent->ElementCount()) {
    LOG(ERROR) << "requested sub-buffer allocation (offset + size) is greater "
               << "than parent allocation size: (" << element_offset << " + "
               << element_count << ") vs. (" << parent->ElementCount() << ")";
    return DeviceMemory<T>{};
  }

  void *opaque = implementation_->AllocateSubBuffer(
      parent, sizeof(T) * element_offset, sizeof(T) * element_count);
  if (opaque == nullptr) {
    return DeviceMemory<T>{};
  }
  CreateAllocRecord(opaque, sizeof(T) * element_count);
  return DeviceMemory<T>(DeviceMemoryBase(opaque, sizeof(T) * element_count,
                                    true /* = is_sub_buffer */));
}

template <typename... Params, typename... Args>
inline Stream &Stream::ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                                  const TypedKernel<Params...> &kernel,
                                  Args... args) {
  KernelInvocationChecker<std::tuple<Params...>,
                          std::tuple<Args...>>::CheckAllStaticAssert();
  if (ok()) {
    // This is the core that allows type-safe kernel launching.
    // Since the platforms take kernel arguments as tuples of (void *, size),
    // we pack the variadic parameters passed as ...args into the desired
    // tuple form and pass that packed form to the StreamExecutor::Launch()
    // implementation.
    std::vector<KernelArg> kernel_args;
    kernel_args.reserve(kernel.Arity());
    kernel.PackParams(&kernel_args, args...);
    bool ok =
        parent_->Launch(this, thread_dims, block_dims, kernel, kernel_args);
    if (!ok) {
      SetError();
      LOG(WARNING) << "parent failed to launch kernel: " << &kernel;
    }
  }
  return *this;
}

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
