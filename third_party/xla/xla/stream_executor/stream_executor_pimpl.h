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

#ifndef XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
#define XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/allocator_stats.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/trace_listener.h"
#include "tsl/platform/status.h"
#include "tsl/platform/threadpool.h"
#include "tsl/protobuf/dnn.pb.h"

// TODO(ezhulenev): Remove include of internal header. Currently we have too
// many targets depending on transitive dependencies.
#include "xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {

class Stream;

namespace internal {
class StreamExecutorInterface;
}  // namespace internal

// Forward declaration of private friend class.
template <typename BeginCallT, typename CompleteCallT, typename ReturnT,
          typename... BeginArgsT>
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
  StreamExecutor(
      const Platform* platform,
      std::unique_ptr<internal::StreamExecutorInterface> implementation,
      int device_ordinal);

  ~StreamExecutor();

  tsl::Status Init();
  tsl::Status Init(DeviceOptions device_options);

  // Returns a reference to the platform that created this executor.
  const Platform* platform() const { return platform_; }

  // Retrieves (loads) a kernel for the platform this StreamExecutor is acting
  // upon, if one exists.
  //
  // Parameters:
  //   spec: The MultiKernelLoaderSpec is usually generated as a compile-time
  //    constant into an appropriate namespace. For example, see
  //    stream_executor::executor_sample::kKernelLoaderSpecs, from which a
  //    MultiKernelLoaderSpec is selected.
  //   kernel: Outparam that the kernel is loaded into. A given Kernel
  //    instantiation should not be loaded into more than once.
  //
  // If an error occurs, or there is no kernel available for the StreamExecutor
  // platform, error status is returned.
  tsl::Status GetKernel(const MultiKernelLoaderSpec& spec, KernelBase* kernel);

  // Releases any state associated with the previously loaded kernel.
  void UnloadKernel(const KernelBase* kernel);

  // Loads a module for the platform this StreamExecutor is acting upon.
  //
  // `spec` describes the module to be loaded.  On success writes the handle for
  // the loaded module to `module_handle` and returns OkStatus().  Otherwise,
  // returns the error which has occurred.
  tsl::Status LoadModule(const MultiModuleLoaderSpec& spec,
                         ModuleHandle* module_handle);

  // Unloads the module with handle `module_handle`.
  bool UnloadModule(ModuleHandle module_handle);

  tsl::StatusOr<std::shared_ptr<DeviceMemoryBase>> CreateOrShareConstant(
      Stream* stream, const std::vector<uint8_t>& content);

  // Synchronously allocates an array on the device of type T with element_count
  // elements.
  template <typename T>
  DeviceMemory<T> AllocateArray(uint64_t element_count,
                                int64_t memory_space = 0);

  // As AllocateArray(), but returns a ScopedDeviceMemory<T>.
  template <typename T>
  ScopedDeviceMemory<T> AllocateOwnedArray(uint64_t element_count) {
    return ScopedDeviceMemory<T>(this, AllocateArray<T>(element_count));
  }

  // Convenience wrapper that allocates space for a single element of type T in
  // device memory.
  template <typename T>
  DeviceMemory<T> AllocateScalar() {
    return AllocateArray<T>(1);
  }

  // As AllocateScalar(), but returns a ScopedDeviceMemory<T>.
  template <typename T>
  ScopedDeviceMemory<T> AllocateOwnedScalar() {
    return AllocateOwnedArray<T>(1);
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
  DeviceMemory<T> GetSubBuffer(DeviceMemory<T>* parent, uint64_t element_offset,
                               uint64_t element_count);

  // An untyped version of GetSymbol.
  tsl::StatusOr<DeviceMemoryBase> GetUntypedSymbol(
      const std::string& symbol_name, ModuleHandle module_handle);

  // Deallocate the DeviceMemory previously allocated via this interface.
  // Deallocation of a nullptr-representative value is permitted.
  //
  // Resets the internal contents of mem to be null-representative, but this
  // null-out effect should not be relied upon in client code.
  void Deallocate(DeviceMemoryBase* mem);

  // Allocates unified memory space of the given size, if supported.
  // See
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
  // for more details on unified memory.
  void* UnifiedMemoryAllocate(uint64_t bytes);

  // Deallocates unified memory space previously allocated with
  // UnifiedMemoryAllocate.
  void UnifiedMemoryDeallocate(void* location);

  // Allocates a region of host memory and registers it with the platform API.
  // Memory allocated in this manner (or allocated and registered with
  // HostMemoryRegister() is required for use in asynchronous memcpy operations,
  // such as Stream::ThenMemcpy.
  void* HostMemoryAllocate(uint64_t size);

  // Deallocates a region of host memory allocated by HostMemoryAllocate().
  void HostMemoryDeallocate(void* location);

  // Synchronizes all activity occurring in the StreamExecutor's context (most
  // likely a whole device).
  bool SynchronizeAllActivity() ABSL_MUST_USE_RESULT;

  // Blocks the caller while "size" bytes are zeroed out (in POD fashion) at the
  // given location in device memory.
  tsl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                 uint64_t size) ABSL_MUST_USE_RESULT;

  // Same as SynchronousMemcpy(DeviceMemoryBase*, ...) above.
  tsl::Status SynchronousMemcpyH2D(const void* host_src, int64_t size,
                                   DeviceMemoryBase* device_dst);

  // Alternative interface for memcpying from host to device that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <class T>
  tsl::Status SynchronousMemcpyH2D(absl::Span<const T> host_src,
                                   DeviceMemoryBase* device_dst) {
    auto host_size = host_src.size() * sizeof(T);
    CHECK(device_dst->size() == 0 || device_dst->size() >= host_size);
    return SynchronousMemcpyH2D(host_src.begin(), host_size, device_dst);
  }

  // Same as SynchronousMemcpy(void*, ...) above.
  tsl::Status SynchronousMemcpyD2H(const DeviceMemoryBase& device_src,
                                   int64_t size, void* host_dst);

  // Blocks the caller while a data segment of the given size is copied from the
  // device source to the device destination.
  bool SynchronousMemcpy(DeviceMemoryBase* device_dst,
                         const DeviceMemoryBase& device_src,
                         uint64_t size) ABSL_MUST_USE_RESULT;

  // Enqueues an operation onto stream to zero out size bytes at the given
  // device memory location. Neither stream nor location may be null. Returns
  // whether the operation was successfully enqueued onto the stream.
  tsl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                      uint64_t size) ABSL_MUST_USE_RESULT;

  // Enqueues an operation onto stream to set 32-bit patterns starting at
  // location, for byte count given by size. size must be 32-bit quantified
  // (i.e. evently divisible by 4). Returns whether the operation was
  // successfully enqueued onto the stream.
  tsl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                       uint32_t pattern, uint64_t size);

  // Enables peer access from this StreamExecutor to memory
  // allocated by other, such that launched device code, memcpies, etc may
  // access it directly.
  //
  // Both this StreamExecutor and other must be backed by the same platform (as
  // in
  // CUDA vs OpenCL) implementation.
  tsl::Status EnablePeerAccessTo(StreamExecutor* other);

  // Returns whether it's possible to enable peer access from this
  // StreamExecutor
  // to memory allocated by another.
  //
  // Even when this returns true, EnablePeerAccessTo may fail for other reasons;
  // this is more an up-front test as to whether it's expressly forbidden.
  bool CanEnablePeerAccessTo(StreamExecutor* other);

  // Obtains metadata about the underlying device.
  // The value is cached on first use.
  const DeviceDescription& GetDeviceDescription() const;

  // If implemented, returns device specific measurement of load
  // (e.g. pending requests).
  int64_t GetDeviceLoad() const;

  // Returns the underlying device memory usage information, if it is available.
  // If it is not available (false is returned), free/total may not be
  // initialized.
  //
  // Note: "Free" reflects the amount of free memory on the underlying device,
  // so allocations via other StreamExecutors that have the same underlying
  // device
  // will be reflected in "free".
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const;

  // Returns the supported algorithms / execution plans for a convolution.
  tsl::Status GetConvolveRunners(
      bool use_cudnn_frontend, dnn::ConvolutionKind kind,
      dnn::DataType input_type, dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, ScratchAllocator* scratch_allocator,
      const NumericOptions& numeric_options,
      std::vector<std::unique_ptr<const dnn::ConvRunner>>* out_exec_plans);

  tsl::Status GetGraphConvolveRunners(
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, const NumericOptions& numeric_options,
      std::vector<std::unique_ptr<const dnn::GraphConvRunner>>* out_exec_plans,
      std::string serialized_graph);

  tsl::Status GetFusedConvolveRunners(
      bool use_cudnn_frontend, dnn::ConvolutionKind kind,
      dnn::DataType input_type, dnn::DataType bias_type,
      dnn::DataType output_type, double conv_input_scale,
      double side_input_scale, double leakyrelu_alpha, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& bias_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, dnn::ActivationMode activation_mode,
      const NumericOptions& numeric_options,
      std::vector<std::unique_ptr<const dnn::FusedConvRunner>>* out_exec_plans);

  tsl::Status GetFusedMatmulRunners(
      bool use_cudnn_frontend, dnn::DataType input_type,
      dnn::DataType bias_type, dnn::DataType output_type, Stream* stream,
      bool trans_a, bool trans_b, uint64_t m, uint64_t n, uint64_t k,
      int64_t lda, int64_t ldb, int64_t ldc,
      dnn::ActivationMode activation_mode, bool use_fallback,
      const NumericOptions& numeric_options,
      std::vector<std::unique_ptr<const dnn::FusedMatmulRunner>>*
          out_exec_plans);

  // Returns the list of supported algorithms for the forward convolution
  // operation.
  bool GetMIOpenConvolveAlgorithms(
      dnn::ConvolutionKind kind, dnn::DataType element_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      ScratchAllocator* scratch_allocator,
      std::vector<dnn::ProfileResult>* out_algorithms);

  // Returns the list of supported algorithms for rnn operation.
  bool GetRnnAlgorithms(std::vector<dnn::AlgorithmDesc>* out_algorithms);

  // Get the list of supported algorithms for BLAS gemm.
  bool GetBlasGemmAlgorithms(Stream* stream,
                             std::vector<blas::AlgorithmType>* out_algorithms);

  // Create an RNN descriptor based on model shapes and configurations.
  // The caller retains the ownership of the descriptor.
  tsl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>> createRnnDescriptor(
      int num_layers, int hidden_size, int input_size, int cell_size,
      int batch_size, dnn::RnnInputMode input_mode,
      dnn::RnnDirectionMode direction_mode, dnn::RnnMode rnn_mode,
      dnn::DataType data_type, const dnn::AlgorithmConfig& algorithm_config,
      const NumericOptions& numeric_options, float dropout, uint64_t seed,
      ScratchAllocator* state_allocator, bool use_padded_io);

  // Create a RNN sequence descriptor that specifies either the input or output
  // sequence. The caller retains the ownership of the returned descriptor.
  tsl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  createRnnSequenceTensorDescriptor(int max_seq_length, int batch_size,
                                    int data_size, dnn::DataType data_type);

  tsl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  createRnnSequenceTensorDescriptor(int max_seq_length, int batch_size,
                                    int data_size,
                                    const absl::Span<const int>& seq_lengths,
                                    bool time_major, dnn::DataType data_type);

  // Create an RNN state descriptor that specifies the input or hidden state.
  // The caller retains the ownership of the returned descriptor.
  tsl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
  createRnnStateTensorDescriptor(int num_layer, int batch_size, int data_size,
                                 dnn::DataType data_type);

  // Returns the device ordinal that this StreamExecutor was initialized with.
  // Meaningless before initialization.
  int device_ordinal() const { return device_ordinal_; }

  // Returns a borrowed pointer to the underlying StreamExecutor implementation.
  internal::StreamExecutorInterface* implementation();

  // Creates a kernel which can be launched with stream.ThenLaunch, such that
  // the types of the arguments provided for launch would have to match
  // types of the arguments provided at creation time.
  //
  // The kernel has a name kernel_name, and is based from provided PTX in ptx,
  // and (optional) compiled PTX in cubin_data.
  // The canonical storage for both ptx and cubin_data should outlive the
  // lifetime of the kernel.
  template <typename... Args>
  tsl::StatusOr<std::unique_ptr<TypedKernel<Args...>>> CreateTypedKernel(
      absl::string_view kernel_name, absl::string_view ptx,
      absl::Span<const uint8_t> cubin_data);

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
  tsl::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                     const BlockDim& block_dims, const KernelBase& kernel,
                     const KernelArgsArrayBase& args);

  // Submits command buffer for execution to the underlying platform driver.
  tsl::Status Submit(Stream* stream, const CommandBuffer& command_buffer);

  // Gets-or-creates (creates with memoization) a FftSupport datatype that can
  // be used to execute FFT routines on the current platform.
  //
  // Ownership and user-facing is the same as AsBlas() below.
  //
  // Returns null if there was an error initializing the FFT support for the
  // underlying platform.
  fft::FftSupport* AsFft();

  // Gets-or-creates (creates with memoization) a DnnSupport datatype that can
  // be used for neural network routines on the current platform.
  //
  // Ownership and user-facing is the same as AsBlas() below.
  //
  // Returns null if there was an error initializing the DNN support for the
  // underlying platform.
  dnn::DnnSupport* AsDnn();

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
  blas::BlasSupport* AsBlas();

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

  // Return allocator statistics.
  std::optional<AllocatorStats> GetAllocatorStats();

  // Clears the internal stats except for the `in_use` fields
  // and sets the `peak_bytes_in_use` to be equal to the `bytes_in_use`.
  bool ClearAllocatorStats();

  // Return an allocator which delegates to this stream executor for memory
  // allocation.
  StreamExecutorMemoryAllocator* GetAllocator() { return &allocator_; }

  // Returns a stream allocated by this executor, or nullptr if not found.
  // Performs linear search over alive GPU streams.
  Stream* FindAllocatedStream(void* gpu_stream);

 private:
  template <typename BeginCallT, typename CompleteCallT, typename ReturnT,
            typename... BeginArgsT>
  friend class ScopedTracer;
  friend class Event;
  friend class Stream;
  template <typename... Params>
  friend class TypedKernel;
  template <typename... Args>
  friend struct ThenBlasImpl;

  // Synchronously allocates size bytes on the underlying platform and returns
  // a DeviceMemoryBase representing that allocation. In the case of failure,
  // nullptr is returned.
  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space);

  void* GetUntypedSubBuffer(DeviceMemoryBase* parent, uint64_t offset,
                            uint64_t size);

  // Causes the host code to synchronously wait for operations entrained
  // onto stream to complete. Effectively a join on the asynchronous device
  // operations enqueued on the stream before this program point.
  tsl::Status BlockHostUntilDone(Stream* stream);

  // Without blocking the device, retrieve the current stream status.
  tsl::Status GetStatus(Stream* stream);

  // Finds and retrieves device memory for the symbol on the underlying
  // platform.
  bool GetSymbol(const std::string& symbol_name, ModuleHandle module_handle,
                 void** mem, size_t* bytes);

  // Entrains a memcpy operation onto stream, with a host destination location
  // host_dst and a device memory source, with target size size.
  bool Memcpy(Stream* stream, void* host_dst,
              const DeviceMemoryBase& device_src, uint64_t size);

  // Entrains a memcpy operation onto stream, with a device destination location
  // and a host memory source, with target size size.
  bool Memcpy(Stream* stream, DeviceMemoryBase* device_dst,
              const void* host_src, uint64_t size);

  // Entrains a memcpy operation onto stream, with a device destination location
  // and a device source location, with target size size. Peer access should
  // have been enabled between the StreamExecutors owning the device memory
  // regions.
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* device_dst,
                            const DeviceMemoryBase& device_src, uint64_t size);

  // Entrains on a stream a user-specified function to be run on the host.
  // See Stream::ThenDoHostCallback for full details.
  // This is the preferred form for a callback that may return an error.
  bool HostCallback(Stream* stream,
                    absl::AnyInvocable<tsl::Status() &&> callback);

  // Performs platform-specific allocation and initialization of an event.
  tsl::Status AllocateEvent(Event* event);

  // Performs platform-specific deallocation and cleanup of an event.
  tsl::Status DeallocateEvent(Event* event);

  // Inserts the specified event at the end of the specified stream.
  tsl::Status RecordEvent(Stream* stream, Event* event);

  // Wait for the specified event at the end of the specified stream.
  tsl::Status WaitForEvent(Stream* stream, Event* event);

  // Wait for the specified event at the end of the raw platform-specific
  // stream. Currently only implemented for GPU, where stream is a
  // GpuStreamHandle (e.g. cudaStream_t).
  tsl::Status WaitForEventOnExternalStream(std::intptr_t stream, Event* event);

  // Requests the current status of the event from the underlying platform.
  Event::Status PollForEventStatus(Event* event);

  // Allocates stream resources on the underlying platform and initializes its
  // internals.
  bool AllocateStream(Stream* stream);

  // Deallocates stream resources on the underlying platform.
  void DeallocateStream(Stream* stream);

  // Causes dependent to not begin execution until other has finished its
  // last-enqueued work.
  bool CreateStreamDependency(Stream* dependent, Stream* other);

  // Allocates a new metadata object, appropriately populated, on the heap, with
  // ownership transfer to caller.
  std::unique_ptr<DeviceDescription> CreateDeviceDescription() const;

  // Adds a task to the tsl::thread::ThreadPool work queue. These tasks must be
  // fire-and-forget and have no external data or timing dependencies; their
  // execution order and completion time have no guarantees.
  // For an example of an appropriate task, see HostBlas::DoBlasGemmInternal;
  // there, temporary internal buffers are freed using this method.
  void EnqueueOnBackgroundThread(std::function<void()> task);

  // Calls the relevant TraceListener routine to begin tracing for the specified
  // asynchronous method.
  template <typename TraceCallT, typename... ArgsT>
  void SubmitTrace(TraceCallT trace_call, ArgsT&&... args);

  // Reader/writer lock for class-static StreamExecutor members.
  static absl::Mutex static_mu_;

  // Reader/writer lock for mutable data structures on this StreamExecutor.
  //
  // Mutable so that caching functions (like DeviceDescription, AsBlas, etc.)
  // can acquire the lock on their first (mutating) call as well.
  mutable absl::Mutex mu_;

  // Reference to the platform that created this executor.
  const Platform* platform_;

  // Pointer to the platform-specific-interface implementation. This is
  // delegated to by the interface routines in pointer-to-implementation
  // fashion.
  std::unique_ptr<internal::StreamExecutorInterface> implementation_;

  // Memoized BLAS support object -- we only want to create this once when asked
  // for a BLAS interface.
  std::unique_ptr<blas::BlasSupport> blas_ ABSL_GUARDED_BY(mu_);

  // Memoized DNN support object -- we only want to create this once when asked
  // for an DNN interface.
  std::unique_ptr<dnn::DnnSupport> dnn_ ABSL_GUARDED_BY(mu_);

  // Memoized FFT support object -- we only want to create this once when asked
  // for a FFT interface.
  std::unique_ptr<fft::FftSupport> fft_;

  // Slot to cache the owned DeviceDescription for the underlying device
  // once it has been queried from DeviceDescription().
  mutable std::unique_ptr<DeviceDescription> device_description_
      ABSL_GUARDED_BY(mu_);

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
  std::unique_ptr<tsl::thread::ThreadPool> background_threads_;

  // Counter for the current number of live streams. This is used to check
  // for accidentally-outstanding streams at StreamExecutor teardown time, as
  // well
  // as to indicate leaks (via a large outstanding count being logged) in the
  // case we can't allocate more streams.
  std::atomic_int_fast32_t live_stream_count_;

  // Only one worker thread is needed; little work will be done by the
  // executor.
  static constexpr int kNumBackgroundThreads = 1;

  // Indicates if StreamExecutor operation tracing should be performed.
  bool tracing_enabled_;

  // The set of TraceListeners registered for this StreamExecutor.
  std::set<TraceListener*> listeners_ ABSL_GUARDED_BY(mu_);

  // Memory limit in bytes. Value less or equal to 0 indicates there is no
  // limit.
  int64_t memory_limit_bytes_;

  StreamExecutorMemoryAllocator allocator_;

  StreamExecutor(const StreamExecutor&) = delete;
  void operator=(const StreamExecutor&) = delete;
};

// A wrapper around ModuleHandle that uses RAII to manage its lifetime.
class ScopedModuleHandle {
 public:
  explicit ScopedModuleHandle(StreamExecutor* executor,
                              ModuleHandle module_handle)
      : executor_(executor), module_handle_(module_handle) {}

  ScopedModuleHandle(ScopedModuleHandle&& other) {
    executor_ = other.executor_;
    module_handle_ = other.module_handle_;
    other.executor_ = nullptr;
    other.module_handle_ = ModuleHandle();
  }

  ScopedModuleHandle& operator=(ScopedModuleHandle&& other) {
    executor_ = other.executor_;
    module_handle_ = other.module_handle_;
    other.executor_ = nullptr;
    other.module_handle_ = ModuleHandle();
    return *this;
  }

  ~ScopedModuleHandle() {
    if (static_cast<bool>(module_handle_)) {
      CHECK(executor_->UnloadModule(module_handle_));
    }
  }

 private:
  StreamExecutor* executor_;
  ModuleHandle module_handle_;

  ScopedModuleHandle(const ScopedModuleHandle&) = delete;
  void operator=(const ScopedModuleHandle&) = delete;
};

////////////
// Inlines

template <typename... Args>
inline tsl::StatusOr<std::unique_ptr<TypedKernel<Args...>>>
StreamExecutor::CreateTypedKernel(absl::string_view kernel_name,
                                  absl::string_view ptx,
                                  absl::Span<const uint8_t> cubin_data) {
  auto kernel_base = std::make_unique<TypedKernel<Args...>>(this);
  MultiKernelLoaderSpec loader_spec(kernel_base->kNumberOfParameters);
  loader_spec.AddCudaPtxInMemory(ptx, kernel_name);

  if (!cubin_data.empty()) {
    loader_spec.AddCudaCubinInMemory(
        reinterpret_cast<const char*>(cubin_data.data()), kernel_name);
  }

  TF_RETURN_IF_ERROR(GetKernel(loader_spec, kernel_base.get()));
  return std::move(kernel_base);
}

template <typename T>
inline DeviceMemory<T> StreamExecutor::AllocateArray(uint64_t element_count,
                                                     int64_t memory_space) {
  uint64_t bytes = sizeof(T) * element_count;
  return DeviceMemory<T>(Allocate(bytes, memory_space));
}

template <typename ElemT>
ScopedDeviceMemory<ElemT>::ScopedDeviceMemory(StreamExecutor* parent,
                                              DeviceMemoryBase value)
    : wrapped_(value),
      device_ordinal_(parent->device_ordinal()),
      allocator_(parent->GetAllocator()) {}

template <typename ElemT>
ScopedDeviceMemory<ElemT>::ScopedDeviceMemory(
    StreamExecutor* parent, std::initializer_list<ElemT> values)
    : ScopedDeviceMemory(parent, parent->AllocateArray<ElemT>(values.size())) {
  if (ptr() != nullptr) {
    std::vector<ElemT> local(values);
    if (!parent->SynchronousMemcpy(ptr(), const_cast<const ElemT*>(&local[0]),
                                   ptr()->size())) {
      TF_CHECK_OK(Free());
    }
  }
}

template <typename T>
DeviceMemory<T> StreamExecutor::GetSubBuffer(DeviceMemory<T>* parent,
                                             uint64_t element_offset,
                                             uint64_t element_count) {
  if (element_offset + element_count > parent->ElementCount()) {
    LOG(ERROR) << "requested sub-buffer allocation (offset + size) is greater "
               << "than parent allocation size: (" << element_offset << " + "
               << element_count << ") vs. (" << parent->ElementCount() << ")";
    return DeviceMemory<T>{};
  }

  void* opaque = GetUntypedSubBuffer(parent, sizeof(T) * element_offset,
                                     sizeof(T) * element_count);
  if (opaque == nullptr) {
    return DeviceMemory<T>{};
  }
  return DeviceMemory<T>(DeviceMemoryBase(opaque, sizeof(T) * element_count));
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
