// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_DEVICE_TYPE_H
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_TYPE_H

namespace Eigen {

// Default device for the machine (typically a single cpu core)
struct DefaultDevice {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return internal::aligned_malloc(num_bytes);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    internal::aligned_free(buffer);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
    ::memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
    ::memset(buffer, c, n);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t numThreads() const {
#ifndef __CUDA_ARCH__
    // Running on the host CPU
    return 1;
#else
    // Running on a CUDA device
    return 32;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t memcpyThreshold() const {
    return 2 * numThreads();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
#ifndef __CUDA_ARCH__
    // Running on the host CPU
    return l1CacheSize();
#else
    // Running on a CUDA device, return the amount of shared memory available.
    return 48*1024;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
#ifndef __CUDA_ARCH__
    // Running single threaded on the host CPU
    return l3CacheSize();
#else
    // Running on a CUDA device
    return firstLevelCacheSize();
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int majorDeviceVersion() const {
#ifndef __CUDA_ARCH__
    // Running single threaded on the host CPU
    // Should return an enum that encodes the ISA supported by the CPU
    return 1;
#else
    // Running on a CUDA device
    return __CUDA_ARCH__ / 100;
#endif
  }
};

// Multiple cpu cores
#ifdef EIGEN_USE_THREADS

#if __cplusplus > 199711
// This defines an interface that ThreadPoolDevice can take to use
// custom thread pools underneath.
class ThreadPoolInterface {
 public:
  virtual void Schedule(std::function<void()> fn) = 0;

  virtual ~ThreadPoolInterface() {}
};
#endif

// The implementation of the ThreadPool type ensures that the Schedule method
// runs the functions it is provided in FIFO order when the scheduling is done
// by a single thread.
#ifdef EIGEN_USE_CUSTOM_THREAD_POOL
class ThreadPool : public ThreadPoolInterface {
 public:
  // Construct a pool that contains "num_threads" threads.
  explicit ThreadPool(int num_threads) : threads_(num_threads), waiters_(num_threads) {
    for (int i = 0; i < num_threads; i++) {
      threads_.push_back(new std::thread([this]() { WorkerLoop(); }));
    }
  }

  // Wait until all scheduled work has finished and then destroy the
  // set of threads.
  ~ThreadPool() {
    {
      // Wait for all work to get done.
      std::unique_lock<std::mutex> l(mu_);
      while (!pending_.empty()) {
        empty_.wait(l);
      }
      exiting_ = true;

      // Wakeup all waiters.
      for (auto w : waiters_) {
        w->ready = true;
        w->work = nullptr;
        w->cv.notify_one();
      }
    }

    // Wait for threads to finish.
    for (auto t : threads_) {
      t->join();
      delete t;
    }
  }

  // Schedule fn() for execution in the pool of threads. The functions are
  // executed in the order in which they are scheduled.
  void Schedule(std::function<void()> fn) final {
    std::unique_lock<std::mutex> l(mu_);
    if (waiters_.empty()) {
      pending_.push_back(fn);
    } else {
      Waiter* w = waiters_.back();
      waiters_.pop_back();
      w->ready = true;
      w->work = fn;
      w->cv.notify_one();
    }
  }

 protected:
  void WorkerLoop() {
    std::unique_lock<std::mutex> l(mu_);
    Waiter w;
    while (!exiting_) {
      std::function<void()> fn;
      if (pending_.empty()) {
        // Wait for work to be assigned to me
        w.ready = false;
        waiters_.push_back(&w);
        while (!w.ready) {
          w.cv.wait(l);
        }
        fn = w.work;
        w.work = nullptr;
      } else {
        // Pick up pending work
        fn = pending_.front();
        pending_.pop_front();
        if (pending_.empty()) {
          empty_.notify_all();
        }
      }
      if (fn) {
        mu_.unlock();
        fn();
        mu_.lock();
      }
    }
  }

 private:
  struct Waiter {
    std::condition_variable cv;
    std::function<void()> work;
    bool ready;
  };

  std::mutex mu_;
  FixedSizeVector<std::thread*> threads_;               // All threads
  FixedSizeVector<Waiter*> waiters_;                    // Stack of waiting threads.
  std::deque<std::function<void()>> pending_;       // Queue of pending work
  std::condition_variable empty_;                   // Signaled on pending_.empty()
  bool exiting_ = false;
};


// Notification is an object that allows a user to to wait for another
// thread to signal a notification that an event has occurred.
//
// Multiple threads can wait on the same Notification object.
// but only one caller must call Notify() on the object.
class Notification {
 public:
  Notification() : notified_(false) {}
  ~Notification() {}

  void Notify() {
    std::unique_lock<std::mutex> l(mu_);
    eigen_assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  void WaitForNotification() {
    std::unique_lock<std::mutex> l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool notified_;
};

#else

// Notification is an object that allows a user to to wait for another
// thread to signal a notification that an event has occurred.
//
// Multiple threads can wait on the same Notification object.
// but only one caller must call Notify() on the object.
class Notification {
 public:
  Notification() : notified_(false) {}
  ~Notification() {}

  void Notify() {
    tensorflow::mutex_lock l(mu_);
    eigen_assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  void WaitForNotification() {
    tensorflow::mutex_lock l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }

 private:
  tensorflow::mutex mu_;
  tensorflow::condition_variable cv_;
  bool notified_;
};
#endif

// Runs an arbitrary function and then calls Notify() on the passed in
// Notification.
template <typename Function, typename... Args> struct FunctionWrapper
{
  static void run(Notification* n, Function f, Args... args) {
    f(args...);
    n->Notify();
  }
};

static EIGEN_STRONG_INLINE void wait_until_ready(Notification* n) {
  if (n) {
    n->WaitForNotification();
  }
}


struct MemcpyExecutor {
  typedef MemcpyExecutor Self;

  MemcpyExecutor(void *dst, const void *src) :
      m_dst(static_cast<char *>(dst)), m_src(static_cast<const char *>(src)) { }

  static EIGEN_STRONG_INLINE void run(const MemcpyExecutor* exec, size_t idx, size_t block_size) {
    ::memcpy(&(exec->m_dst[idx]), &(exec->m_src[idx]), block_size);
  }

 private:
  char* m_dst;
  const char* m_src;
};

struct MemsetExecutor {
  typedef MemsetExecutor Self;

  MemsetExecutor(void *buffer, int val) :
      m_buffer(static_cast<char *>(buffer)), m_val(val) { }

  static EIGEN_STRONG_INLINE void run(const MemsetExecutor* exec, size_t idx, size_t block_size) {
    ::memset(&(exec->m_buffer[idx]), exec->m_val, block_size);
  }

 private:
  char* m_buffer;
  const int m_val;
};


struct ThreadPoolDevice {
  // The ownership of the thread pool remains with the caller.
  ThreadPoolDevice(ThreadPoolInterface* pool, size_t num_cores)
      : pool_(pool), num_threads_(num_cores) {}

  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return internal::aligned_malloc(num_bytes);
  }

  EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    internal::aligned_free(buffer);
  }

  EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
#ifdef __ANDROID__
    ::memcpy(dst, src, n);
#else
    if (n <= 32768) {
      ::memcpy(dst, src, n);
    } else {
      MemcpyExecutor memcpy_executor(dst, src);
      execute(memcpy_executor, n);
    }
#endif
  }

  EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }

  EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }

  EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
#ifdef __ANDROID__
    ::memset(buffer, c, n);
#else
    if (n <= 32768) {
      ::memset(buffer, c, n);
    } else {
      MemsetExecutor memset_executor(buffer, c);
      execute(memset_executor, n);
    }
#endif
  }

  EIGEN_STRONG_INLINE size_t numThreads() const {
    return num_threads_;
  }

  EIGEN_STRONG_INLINE size_t memcpyThreshold() const {
    return 2 * numThreads();
  }

  EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
    return l1CacheSize();
  }

  EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // The l3 cache size is shared between all the cores.
    return l3CacheSize() / num_threads_;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int majorDeviceVersion() const {
    // Should return an enum that encodes the ISA supported by the CPU
    return 1;
  }

  template <class Function, class... Args>
  EIGEN_STRONG_INLINE Notification* enqueue(Function&& f, Args&&... args) const {
    Notification* n = new Notification();
    std::function<void()> func =
        std::bind(&FunctionWrapper<Function, Args...>::run, n, f, args...);
    pool_->Schedule(func);
    return n;
  }

  template <class Function, class... Args>
  EIGEN_STRONG_INLINE void enqueue_and_forget(Function&& f, Args&&... args) const {
    std::function<void()> func = std::bind(f, args...);
    pool_->Schedule(func);
  }

 private:
  template<typename Executor>
  EIGEN_STRONG_INLINE void execute(const Executor& exec, size_t n) const {
    // don't spawn a thread to process fewer than 1024 bytes (chosen by small amount of
    // experimentation)
    // TODO: make block_size a multiple of packet_size and align everything
    const size_t block_size = numext::maxi(static_cast<size_t>(1024), n / numThreads());
    const size_t block_count = n / block_size;
    eigen_assert(block_count <= numThreads());

    FixedSizeVector<Notification*> results(block_count);
    for (size_t block_idx = 0; block_idx < block_count; block_idx++) {
      results.push_back(enqueue(&Executor::run, &exec, block_idx * block_size, block_size));
    }

    if (block_count * block_size < n) {
      Executor::run(&exec, block_count * block_size, n - block_count * block_size);
    }

    // wait for threads to finish
    for (size_t block_idx = 0; block_idx < block_count; block_idx++) {
      results[block_idx]->WaitForNotification();
      delete results[block_idx];
    }
  }

  // todo: NUMA, ...
  size_t num_threads_;
  ThreadPoolInterface* pool_;
};
#endif


// GPU offloading
#ifdef EIGEN_USE_GPU

// An interface abstracting away device specific memory allocator.
class Allocator {
 public:
  virtual ~Allocator() {}
  EIGEN_DEVICE_FUNC virtual void* allocate(size_t num_bytes) const = 0;
  EIGEN_DEVICE_FUNC virtual void deallocate(void* buffer) const = 0;
};

#if !defined(__GCUDACC__) && !defined(__GCUDACC_HOST__)

// This defines an interface that GPUDevice can take to use
// CUDA streams underneath.
class StreamInterface {
 public:
  virtual ~StreamInterface() {}

  virtual const cudaStream_t& stream() const = 0;
  virtual const cudaDeviceProp& deviceProperties() const = 0;

  // Allocate memory on the actual device where the computation will run
  virtual void* allocate(size_t num_bytes) const = 0;
  virtual void deallocate(void* buffer) const = 0;
};

static cudaDeviceProp* m_deviceProperties;
static bool m_devicePropInitialized = false;
static tensorflow::mutex m_devicePropInitMutex(tensorflow::LINKER_INITIALIZED);

static void initializeDeviceProp() {
  if (!m_devicePropInitialized) {
    tensorflow::mutex_lock l(m_devicePropInitMutex);
    if (!m_devicePropInitialized) {
      int num_devices;
      cudaError_t status = cudaGetDeviceCount(&num_devices);
      eigen_check(status == cudaSuccess);
      m_deviceProperties = new cudaDeviceProp[num_devices];
      for (int i = 0; i < num_devices; ++i) {
        status = cudaGetDeviceProperties(&m_deviceProperties[i], i);
        eigen_check(status == cudaSuccess);
      }
      m_devicePropInitialized = true;
    }
  }
}

static const cudaStream_t default_stream = cudaStreamDefault;

class CudaStreamDevice : public StreamInterface {
 public:
  // Use the default stream on the current device
  CudaStreamDevice() : stream_(&default_stream) {
    cudaGetDevice(&device_);
    initializeDeviceProp();
  }
  // Use the default stream on the specified device
  CudaStreamDevice(int device) : stream_(&default_stream), device_(device) {
    initializeDeviceProp();
  }
  // Use the specified stream. Note that it's the
  // caller responsibility to ensure that the stream can run on
  // the specified device. If no device is specified the code
  // assumes that the stream is associated to the current gpu device.
  CudaStreamDevice(const cudaStream_t* stream, int device = -1)
      : stream_(stream), device_(device) {
    if (device < 0) {
      cudaGetDevice(&device_);
    } else {
      int num_devices;
      cudaError_t err = cudaGetDeviceCount(&num_devices);
      eigen_check(err == cudaSuccess);
      eigen_check(device < num_devices);
      device_ = device;
    }
    initializeDeviceProp();
  }

  const cudaStream_t& stream() const { return *stream_; }
  const cudaDeviceProp& deviceProperties() const {
    return m_deviceProperties[device_];
  }
  virtual void* allocate(size_t num_bytes) const {
    cudaError_t err = cudaSetDevice(device_);
    eigen_check(err == cudaSuccess);
    void* result;
    err = cudaMalloc(&result, num_bytes);
    eigen_check(err == cudaSuccess);
    eigen_check(result != NULL);
    return result;
  }
  virtual void deallocate(void* buffer) const {
    cudaError_t err = cudaSetDevice(device_);
    eigen_check(err == cudaSuccess);
    assert(buffer != NULL);
    err = cudaFree(buffer);
    assert(err == cudaSuccess);
  }

 private:
  const cudaStream_t* stream_;
  int device_;
};

static inline void setCudaSharedMemConfig(cudaSharedMemConfig config) {
  cudaError_t status = cudaDeviceSetSharedMemConfig(config);
  eigen_check(status == cudaSuccess);
}

struct GpuDevice {
  // Neither the cudastream nor the allocator is not owned: the caller is
  // responsible for their initialization and eventual destruction.
  explicit GpuDevice(const StreamInterface* stream) : stream_(stream) {
    eigen_assert(stream);
  }

  // TODO(bsteiner): This is an internal API, we should not expose it.
  EIGEN_STRONG_INLINE const cudaStream_t& stream() const {
    return stream_->stream();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
#ifndef __CUDA_ARCH__
    return stream_->allocate(num_bytes);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
    return NULL;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
#ifndef __CUDA_ARCH__
    stream_->deallocate(buffer);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice,
                                      stream_->stream());
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err =
        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream_->stream());
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err =
        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream_->stream());
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err = cudaMemsetAsync(buffer, c, n, stream_->stream());
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t numThreads() const {
    // FIXME
    return 32;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t memcpyThreshold() const {
    return 4 * 1024 * 1024;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
    // FIXME
    return 48*1024;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // We won't try to take advantage of the l2 cache for the time being, and
    // there is no l3 cache on cuda devices.
    return firstLevelCacheSize();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void synchronize() const {
#ifndef __CUDA_ARCH__
    cudaError_t err = cudaStreamSynchronize(stream_->stream());
    assert(err == cudaSuccess);
#else
    assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  inline int getNumCudaMultiProcessors() const {
    return stream_->deviceProperties().multiProcessorCount;
  }
  inline int maxCudaThreadsPerBlock() const {
    return stream_->deviceProperties().maxThreadsPerBlock;
  }
  inline int maxCudaThreadsPerMultiProcessor() const {
    return stream_->deviceProperties().maxThreadsPerMultiProcessor;
  }
  inline int sharedMemPerBlock() const {
    return stream_->deviceProperties().sharedMemPerBlock;
  }
  inline int majorDeviceVersion() const {
    return stream_->deviceProperties().major;
  }

  // This function checks if the CUDA runtime recorded an error for the
  // underlying stream device.
  inline bool ok() const {
    cudaError_t error = cudaStreamQuery(stream_->stream());
    return (error == cudaSuccess) || (error == cudaErrorNotReady);
  }

 private:
  const StreamInterface* stream_;
};

inline void assertCudaOk() {
  cudaError_t err = cudaGetLastError();

  assert(err != cudaErrorMissingConfiguration);
  assert(err != cudaErrorMemoryAllocation);
  assert(err != cudaErrorInitializationError);
  assert(err != cudaErrorLaunchFailure);
  assert(err != cudaErrorPriorLaunchFailure);
  assert(err != cudaErrorLaunchTimeout);
  assert(err != cudaErrorLaunchOutOfResources);
  assert(err != cudaErrorInvalidDeviceFunction);
  assert(err != cudaErrorInvalidConfiguration);
  assert(err != cudaErrorInvalidDevice);
  assert(err != cudaErrorInvalidValue);
  assert(err != cudaErrorInvalidPitchValue);
  assert(err != cudaErrorInvalidSymbol);
  assert(err != cudaErrorMapBufferObjectFailed);
  assert(err != cudaErrorUnmapBufferObjectFailed);
  assert(err != cudaErrorInvalidHostPointer);
  assert(err != cudaErrorInvalidDevicePointer);
  assert(err != cudaErrorInvalidTexture);
  assert(err != cudaErrorInvalidTextureBinding);
  assert(err != cudaErrorInvalidChannelDescriptor);
  assert(err != cudaErrorInvalidMemcpyDirection);
  assert(err != cudaErrorAddressOfConstant);
  assert(err != cudaErrorTextureFetchFailed);
  assert(err != cudaErrorTextureNotBound);
  assert(err != cudaErrorSynchronizationError);
  assert(err != cudaErrorInvalidFilterSetting);
  assert(err != cudaErrorInvalidNormSetting);
  assert(err != cudaErrorMixedDeviceExecution);
  assert(err != cudaErrorCudartUnloading);
  assert(err != cudaErrorUnknown);
  assert(err != cudaErrorNotYetImplemented);
  assert(err != cudaErrorMemoryValueTooLarge);
  assert(err != cudaErrorInvalidResourceHandle);
  assert(err != cudaErrorNotReady);
  assert(err != cudaErrorInsufficientDriver);
  assert(err != cudaErrorSetOnActiveProcess);
  assert(err != cudaErrorInvalidSurface);
  assert(err != cudaErrorNoDevice);
  assert(err != cudaErrorECCUncorrectable);
  assert(err != cudaErrorSharedObjectSymbolNotFound);
  assert(err != cudaErrorSharedObjectInitFailed);
  assert(err != cudaErrorUnsupportedLimit);
  assert(err != cudaErrorDuplicateVariableName);
  assert(err != cudaErrorDuplicateTextureName);
  assert(err != cudaErrorDuplicateSurfaceName);
  assert(err != cudaErrorDevicesUnavailable);
  assert(err != cudaErrorInvalidKernelImage);
  assert(err != cudaErrorNoKernelImageForDevice);
  assert(err != cudaErrorIncompatibleDriverContext);
  assert(err != cudaErrorPeerAccessAlreadyEnabled);
  assert(err != cudaErrorPeerAccessNotEnabled);
  assert(err != cudaErrorDeviceAlreadyInUse);
  assert(err != cudaErrorProfilerDisabled);
  assert(err != cudaErrorProfilerNotInitialized);
  assert(err != cudaErrorProfilerAlreadyStarted);
  assert(err != cudaErrorProfilerAlreadyStopped);
  assert(err != cudaErrorAssert);
  assert(err != cudaErrorTooManyPeers);
  assert(err != cudaErrorHostMemoryAlreadyRegistered);
  assert(err != cudaErrorHostMemoryNotRegistered);
  assert(err != cudaErrorOperatingSystem);
  assert(err != cudaErrorStartupFailure);
  assert(err != cudaErrorApiFailureBase);

  // catch errors types introduced after this function was written
  assert(err == cudaSuccess);
}

#define LAUNCH_CUDA_KERNEL(kernel, gridsize, blocksize, sharedmem, device, \
                           ...)                                            \
  do {                                                                     \
    (kernel)<<<(gridsize), (blocksize), (sharedmem), (device).stream()>>>( \
        __VA_ARGS__);                                                      \
    assertCudaOk();                                                        \
  } while (false)

#else  // __GCUDACC__

// The following is the version of GpuDevice for StreamExecutor
// (go/gpuexecutor) a GPU runtime that supports both CUDA and OpenCL.
// StreamExecutor is being developed as an open-source replacement for the CUDA
// runtime and is the runtime used when compiling with gcudacc. Differences
// between the CUDA runtime and StreamExecutor are abstracted away behind
// GpuDevice.

// TODO(jpienaar): Temporary workaround until b/18409724 is addressed.
enum cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};

static inline void setCudaSharedMemConfig(cudaSharedMemConfig cache_config) {
  // TODO(jpienaar): fix when implemented (b/18409724)
}

struct GpuDevice {
  GpuDevice()
      : stream_(perftools::gputools::MachineManager::singleton()->stream_for_device(0)),
        allocator_(nullptr),
        stream_exec_(stream_->parent()) {}

  GpuDevice(perftools::gputools::Stream* stream,
            const Allocator* alloc = nullptr)
      : stream_(stream), allocator_(alloc), stream_exec_(stream_->parent()) { }

  EIGEN_STRONG_INLINE perftools::gputools::Stream* stream() const {
    return stream_;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    if (allocator_ != nullptr) return allocator_->allocate(num_bytes);
#ifndef __CUDA_ARCH__
    perftools::gputools::DeviceMemory<char> mem =
        stream_exec_->AllocateArray<char>(num_bytes);
    return mem.opaque();
#else
    assert(false &&
           "The default device should be used instead to generate kernel code");
    return nullptr;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    if (allocator_ != nullptr) {
      allocator_->deallocate(buffer);
      return;
    }
#ifndef __CUDA_ARCH__
    perftools::gputools::DeviceMemoryBase gpu_mem(buffer);
    stream_exec_->Deallocate(&gpu_mem);
#else
    assert(false &&
           "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src,
                                                    size_t n) const {
#ifndef __CUDA_ARCH__
    perftools::gputools::DeviceMemoryBase gpu_to(dst);
    if (!stream_->ThenMemcpy(&gpu_to, perftools::gputools::DeviceMemoryBase(
                                          const_cast<void*>(src)),
                             n).ok()) {
      assert(false &&
             "failed during enqueue of 'copy perftools::gputools to "
             "perftools::gputools'");
    }
#else
    assert(false &&
           "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    perftools::gputools::DeviceMemoryBase gpu_to(dst);
    if (!stream_->ThenMemcpy(&gpu_to, src, n).ok()) {
      assert(false && "failed while enqueuing memcpy from host to device");
    }
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    if (!stream_->ThenMemcpy(dst, perftools::gputools::DeviceMemoryBase(
                                      const_cast<void*>(src)),
                             n).ok()) {
      assert(false && "failed while enqueuing memcpy from device to host");
    }
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
#ifndef __CUDA_ARCH__
    perftools::gputools::DeviceMemoryBase gpu_buffer{buffer};
    if (!stream_exec_->Memset32(stream_, &gpu_buffer, c, n)) {
      assert(false && "GPU memset failed.");
    }
#else
    assert(false &&
           "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t numThreads() const {
    // FIXME
    return 32;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t memcpyThreshold() const {
    return 4 * 1024 * 1024;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
    // FIXME
    return 48*1024;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // We won't try to take advantage of the l2 cache for the time being, and
    // there is no l3 cache on cuda devices.
    return firstLevelCacheSize();
  }

  EIGEN_STRONG_INLINE void synchronize() const {
    stream_->BlockHostUntilDone();
  }

  // A gpu::DeviceDescription is cached inside a StreamExecutor, so these calls
  // aren't expensive/wasteful.
  EIGEN_DEVICE_FUNC inline int getNumCudaMultiProcessors() const {
    return stream_exec_->GetDeviceDescription().core_count();
  }

  EIGEN_DEVICE_FUNC inline int maxCudaThreadsPerBlock() const {
    return stream_exec_->GetDeviceDescription().threads_per_block_limit();
  }

  EIGEN_DEVICE_FUNC inline int maxCudaThreadsPerMultiProcessor() const {
    return stream_exec_->GetDeviceDescription().threads_per_core_limit();
  }

  EIGEN_DEVICE_FUNC inline int sharedMemPerBlock() const {
    return stream_exec_->GetDeviceDescription().shared_memory_per_block();
  }

  EIGEN_DEVICE_FUNC inline int majorDeviceVersion() const {
    int major, minor;
    if (stream_exec_->GetDeviceDescription().cuda_compute_capability(&major,
                                                                  &minor)) {
      return major;
    } else {
      return 0;
    }
  }

  inline bool ok() const { return stream_->ok(); }

 private:
  perftools::gputools::Stream* stream_;
  perftools::gputools::StreamExecutor* stream_exec_;
  const Allocator* allocator_;
};

#define LAUNCH_CUDA_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)\
    (kernel) <<< (gridsize), (blocksize), (sharedmem), (device).stream() >>> (__VA_ARGS__);  \
  CHECK((device).stream()->ok());
#endif  // __GCUDACC__

#endif  // EIGEN_USE_GPU
}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_TYPE_H
