// Implements the StreamExecutor interface by passing through to its
// implementation_ value (in pointer-to-implementation style), which
// implements StreamExecutorInterface.

#include "tensorflow/stream_executor/stream_executor_pimpl.h"

#include <atomic>

#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/fft.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/notification.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace {
bool FLAGS_check_gpu_leaks = false;
}  // namespace

namespace perftools {
namespace gputools {
namespace {

// Maximum stack depth to report when generating backtrace on mem allocation
// (for GPU memory leak checker)
static const int kMaxStackDepth = 256;

// Make sure the executor is done with its work; we know (because this isn't
// publicly visible) that all enqueued work is quick.
void BlockOnThreadExecutor(port::ThreadPool *executor) {
  port::Notification n;
  executor->Schedule([&n]() { n.Notify(); });
  n.WaitForNotification();
}

internal::StreamExecutorInterface *StreamExecutorImplementationFromPlatformKind(
    PlatformKind platform_kind, const PluginConfig &plugin_config) {
  // Note: we use this factory-assignment-in-switch pattern instead of just
  // invoking the callable in case linkage is messed up -- instead of invoking a
  // nullptr std::function (due to failed registration) we give a nice
  // LOG(FATAL) message.
  internal::StreamExecutorFactory factory;
  switch (platform_kind) {
    case PlatformKind::kCuda:
      factory = *internal::MakeCUDAExecutorImplementation();
      break;
    case PlatformKind::kOpenCL:
      factory = *internal::MakeOpenCLExecutorImplementation();
      break;
    case PlatformKind::kOpenCLAltera:
      factory = *internal::MakeOpenCLAlteraExecutorImplementation();
      break;
    case PlatformKind::kHost:
      factory = internal::MakeHostExecutorImplementation;
      break;
    default:
      factory = nullptr;
  }
  if (factory == nullptr) {
    LOG(FATAL)
        << "cannot create GPU executor implementation for platform kind: "
        << PlatformKindString(platform_kind);
  }
  return factory(plugin_config);
}

std::atomic_int_fast64_t correlation_id_generator(0);

}  // namespace

template <typename BeginCallT, typename CompleteCallT,
          typename ReturnT, typename... BeginArgsT>
class ScopedTracer {
 public:
  ScopedTracer(StreamExecutor *stream_exec, BeginCallT begin_call,
               CompleteCallT complete_call, const ReturnT *result,
               BeginArgsT... begin_args)
      : stream_exec_(stream_exec),
        complete_call_(complete_call),
        result_(result) {
    if (stream_exec_->tracing_enabled_) {
      correlation_id_ =
          correlation_id_generator.fetch_add(1, std::memory_order_relaxed) - 1;
      Trace(begin_call, begin_args...);
    }
  }

  ~ScopedTracer() {
    if (stream_exec_->tracing_enabled_) {
      Trace(complete_call_, result_);
    }
  }

 private:
  template <typename CallbackT, typename... TraceArgsT>
  void Trace(CallbackT callback, TraceArgsT... args) {
    {
      // Instance tracers held in a block to limit the lock lifetime.
      shared_lock lock{stream_exec_->mu_};
      for (TraceListener *listener : stream_exec_->listeners_) {
        (listener->*callback)(correlation_id_,
                              std::forward<TraceArgsT>(args)...);
      }
    }
  }

  StreamExecutor *stream_exec_;
  CompleteCallT complete_call_;
  const ReturnT* result_;
  int64 correlation_id_;
};

template <typename BeginCallT, typename CompleteCallT, typename ReturnT,
          typename... BeginArgsT>
ScopedTracer<BeginCallT, CompleteCallT, ReturnT, BeginArgsT...>
MakeScopedTracer(StreamExecutor *stream_exec, BeginCallT begin_call,
                 CompleteCallT complete_call, ReturnT *result,
                 BeginArgsT... begin_args) {
  return ScopedTracer<BeginCallT, CompleteCallT, ReturnT, BeginArgsT...>(
      stream_exec, begin_call, complete_call, result,
      std::forward<BeginArgsT>(begin_args)...);
}

#define SCOPED_TRACE(LOC, ...)                                      \
  auto tracer = MakeScopedTracer(this, &LOC ## Begin,               \
                                 &LOC ## Complete, ## __VA_ARGS__);

/* static */ mutex StreamExecutor::static_mu_{LINKER_INITIALIZED};

StreamExecutor::StreamExecutor(PlatformKind platform_kind,
                               const PluginConfig &plugin_config)
    : implementation_(StreamExecutorImplementationFromPlatformKind(
          platform_kind, plugin_config)),
      platform_kind_(platform_kind),
      device_ordinal_(-1),
      background_threads_(new port::ThreadPool(
          port::Env::Default(), "stream_executor", kNumBackgroundThreads)),
      live_stream_count_(0),
      tracing_enabled_(false) {
  CheckPlatformKindIsValid(platform_kind);
}

StreamExecutor::StreamExecutor(
    PlatformKind platform_kind,
    internal::StreamExecutorInterface *implementation)
    : implementation_(implementation),
      platform_kind_(platform_kind),
      device_ordinal_(-1),
      background_threads_(new port::ThreadPool(
          port::Env::Default(), "stream_executor", kNumBackgroundThreads)),
      live_stream_count_(0),
      tracing_enabled_(false) {
  CheckPlatformKindIsValid(platform_kind);
}

StreamExecutor::~StreamExecutor() {
  BlockOnThreadExecutor(background_threads_.get());

  if (live_stream_count_.load() != 0) {
    LOG(WARNING) << "Not all streams were deallocated at executor destruction "
                 << "time. This may lead to unexpected/bad behavior - "
                 << "especially if any stream is still active!";
  }

  if (FLAGS_check_gpu_leaks) {
    for (auto it : mem_allocs_) {
      LOG(INFO) << "Memory alloced at executor exit: addr: "
                << port::Printf("%p", it.first)
                << ", bytes: " << it.second.bytes << ", trace: \n"
                << it.second.stack_trace;
    }
  }
}

port::Status StreamExecutor::Init(int device_ordinal,
                                  DeviceOptions device_options) {
  device_ordinal_ = device_ordinal;
  return implementation_->Init(device_ordinal, device_options);
}

port::Status StreamExecutor::Init() {
  return Init(0, DeviceOptions::Default());
}

bool StreamExecutor::GetKernel(const MultiKernelLoaderSpec &spec,
                               KernelBase *kernel) {
  return implementation_->GetKernel(spec, kernel);
}

void StreamExecutor::Deallocate(DeviceMemoryBase *mem) {
  VLOG(1) << "Called StreamExecutor::Deallocate(mem=" << mem->opaque()
          << ") mem->size()=" << mem->size();

  if (mem->opaque() != nullptr) {
    EraseAllocRecord(mem->opaque());
  }
  implementation_->Deallocate(mem);
  mem->Reset(nullptr, 0);
}

void StreamExecutor::GetMemAllocs(std::map<void *, AllocRecord> *records_out) {
  shared_lock lock{mu_};
  *records_out = mem_allocs_;
}

bool StreamExecutor::CanEnablePeerAccessTo(StreamExecutor *other) {
  return implementation_->CanEnablePeerAccessTo(other->implementation_.get());
}

port::Status StreamExecutor::EnablePeerAccessTo(StreamExecutor *other) {
  return implementation_->EnablePeerAccessTo(other->implementation_.get());
}

SharedMemoryConfig StreamExecutor::GetDeviceSharedMemoryConfig() {
  return implementation_->GetDeviceSharedMemoryConfig();
}

port::Status StreamExecutor::SetDeviceSharedMemoryConfig(
    SharedMemoryConfig config) {
  if (config != SharedMemoryConfig::kDefault &&
      config != SharedMemoryConfig::kFourByte &&
      config != SharedMemoryConfig::kEightByte) {
    string error_msg = port::Printf(
        "Invalid shared memory config specified: %d", static_cast<int>(config));
    LOG(ERROR) << error_msg;
    return port::Status{port::error::INVALID_ARGUMENT, error_msg};
  }
  return implementation_->SetDeviceSharedMemoryConfig(config);
}

const DeviceDescription &StreamExecutor::GetDeviceDescription() const {
  mutex_lock lock{mu_};
  if (device_description_ != nullptr) {
    return *device_description_;
  }

  device_description_.reset(PopulateDeviceDescription());
  return *device_description_;
}

int StreamExecutor::PlatformDeviceCount() const {
  return implementation_->PlatformDeviceCount();
}

bool StreamExecutor::SupportsBlas() const {
  return implementation_->SupportsBlas();
}

bool StreamExecutor::SupportsRng() const {
  return implementation_->SupportsRng();
}

bool StreamExecutor::SupportsDnn() const {
  return implementation_->SupportsDnn();
}

dnn::DnnSupport *StreamExecutor::AsDnn() {
  mutex_lock lock{mu_};
  if (dnn_ != nullptr) {
    return dnn_.get();
  }

  dnn_.reset(implementation_->CreateDnn());
  return dnn_.get();
}

blas::BlasSupport *StreamExecutor::AsBlas() {
  mutex_lock lock{mu_};
  if (blas_ != nullptr) {
    return blas_.get();
  }

  blas_.reset(implementation_->CreateBlas());
  return blas_.get();
}

fft::FftSupport *StreamExecutor::AsFft() {
  mutex_lock lock{mu_};
  if (fft_ != nullptr) {
    return fft_.get();
  }

  fft_.reset(implementation_->CreateFft());
  return fft_.get();
}

rng::RngSupport *StreamExecutor::AsRng() {
  mutex_lock lock{mu_};
  if (rng_ != nullptr) {
    return rng_.get();
  }

  rng_.reset(implementation_->CreateRng());
  return rng_.get();
}

bool StreamExecutor::Launch(Stream *stream, const ThreadDim &thread_dims,
                            const BlockDim &block_dims,
                            const KernelBase &kernel,
                            const std::vector<KernelArg> &args) {
  SubmitTrace(&TraceListener::LaunchSubmit, stream, thread_dims, block_dims,
              kernel, args);

  return implementation_->Launch(stream, thread_dims, block_dims, kernel, args);
}

bool StreamExecutor::BlockHostUntilDone(Stream *stream) {
  bool result;
  SCOPED_TRACE(TraceListener::BlockHostUntilDone, &result, stream);

  result = implementation_->BlockHostUntilDone(stream);
  return result;
}

void *StreamExecutor::Allocate(uint64 size) {
  void *buf = implementation_->Allocate(size);
  VLOG(1) << "Called StreamExecutor::Allocate(size=" << size
          << ") returns " << buf;
  CreateAllocRecord(buf, size);

  return buf;
}

bool StreamExecutor::GetSymbol(const string &symbol_name, void **mem,
                               size_t *bytes) {
  return implementation_->GetSymbol(symbol_name, mem, bytes);
}

void *StreamExecutor::HostMemoryAllocate(uint64 size) {
  void *buffer = implementation_->HostMemoryAllocate(size);
  VLOG(1) << "Called StreamExecutor::HostMemoryAllocate(size=" << size
          << ") returns " << buffer;
  return buffer;
}

void StreamExecutor::HostMemoryDeallocate(void *location) {
  VLOG(1) << "Called StreamExecutor::HostMemoryDeallocate(location="
          << location << ")";

  return implementation_->HostMemoryDeallocate(location);
}

bool StreamExecutor::HostMemoryRegister(void *location, uint64 size) {
  VLOG(1) << "Called StreamExecutor::HostMemoryRegister(location=" << location
          << ", size=" << size << ")";
  if (location == nullptr || size == 0) {
    LOG(WARNING) << "attempting to register null or zero-sized memory: "
                 << location << "; size " << size;
  }
  return implementation_->HostMemoryRegister(location, size);
}

bool StreamExecutor::HostMemoryUnregister(void *location) {
  VLOG(1) << "Called StreamExecutor::HostMemoryUnregister(location=" << location
          << ")";
  return implementation_->HostMemoryUnregister(location);
}

bool StreamExecutor::SynchronizeAllActivity() {
  VLOG(1) << "Called StreamExecutor::SynchronizeAllActivity()";
  bool ok = implementation_->SynchronizeAllActivity();

  // This should all be quick and infallible work, so we can perform the
  // synchronization even in the case of failure.
  BlockOnThreadExecutor(background_threads_.get());

  return ok;
}

bool StreamExecutor::SynchronousMemZero(DeviceMemoryBase *location,
                                        uint64 size) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemZero(location="
          << location << ", size=" << size << ")";

  return implementation_->SynchronousMemZero(location, size);
}

bool StreamExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                       uint64 size) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemSet(location="
          << location << ", value=" << value << ", size=" << size << ")";

  return implementation_->SynchronousMemSet(location, value, size);
}

bool StreamExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                       const void *host_src, uint64 size) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemcpy(gpu_dst="
          << gpu_dst->opaque() << ", host_src=" << host_src << ", size=" << size
          << ") H2D";

  // Tracing overloaded methods is very difficult due to issues with type
  // inference on template args. Since use of these overloaded methods is
  // discouraged anyway, this isn't a huge deal.
  return implementation_->SynchronousMemcpy(gpu_dst, host_src, size);
}

bool StreamExecutor::SynchronousMemcpy(void *host_dst,
                                       const DeviceMemoryBase &gpu_src,
                                       uint64 size) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemcpy(host_dst="
          << host_dst << ", gpu_src=" << gpu_src.opaque() << ", size=" << size
          << ") D2H";

  return implementation_->SynchronousMemcpy(host_dst, gpu_src, size);
}

bool StreamExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                       const DeviceMemoryBase &gpu_src,
                                       uint64 size) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemcpy(gpu_dst="
          << gpu_dst->opaque() << ", gpu_src=" << gpu_src.opaque() << ", size=" << size
          << ") D2D";

  return implementation_->SynchronousMemcpyDeviceToDevice(gpu_dst, gpu_src,
                                                          size);
}

port::Status StreamExecutor::SynchronousMemcpyD2H(
    const DeviceMemoryBase &gpu_src, int64 size, void *host_dst) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemcpyD2H(gpu_src="
          << gpu_src.opaque() << ", size=" << size << ", host_dst=" << host_dst << ")";

  port::Status result{port::Status::OK()};
  SCOPED_TRACE(TraceListener::SynchronousMemcpyD2H,
               &result, gpu_src, size, host_dst);

  if (!implementation_->SynchronousMemcpy(host_dst, gpu_src, size)) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf(
            "failed to synchronously memcpy device-to-host: GPU %p to host %p "
            "size %lld",
            gpu_src.opaque(), host_dst, size)};
  }

  return result;
}

port::Status StreamExecutor::SynchronousMemcpyH2D(const void *host_src,
                                                  int64 size,
                                                  DeviceMemoryBase *gpu_dst) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemcpyH2D(host_src="
          << host_src << ", size=" << size << ", gpu_dst" << gpu_dst->opaque() << ")";

  port::Status result{port::Status::OK()};
  SCOPED_TRACE(TraceListener::SynchronousMemcpyH2D,
               &result, host_src, size, gpu_dst);

  if (!implementation_->SynchronousMemcpy(gpu_dst, host_src, size)) {
    result = port::Status{
        port::error::INTERNAL,
        port::Printf("failed to synchronously memcpy host-to-device: host "
                     "%p to GPU %p size %lld",
                     host_src, gpu_dst->opaque(), size)};
  }

  return result;
}

bool StreamExecutor::Memcpy(Stream *stream, void *host_dst,
                            const DeviceMemoryBase &gpu_src, uint64 size) {
  return implementation_->Memcpy(stream, host_dst, gpu_src, size);
}

bool StreamExecutor::Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                            const void *host_src, uint64 size) {
  return implementation_->Memcpy(stream, gpu_dst, host_src, size);
}

bool StreamExecutor::MemcpyDeviceToDevice(Stream *stream,
                                          DeviceMemoryBase *gpu_dst,
                                          const DeviceMemoryBase &gpu_src,
                                          uint64 size) {
  return implementation_->MemcpyDeviceToDevice(stream, gpu_dst, gpu_src, size);
}

bool StreamExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                             uint64 size) {
  return implementation_->MemZero(stream, location, size);
}

bool StreamExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                              uint32 pattern, uint64 size) {
  CHECK_EQ(0, size % 4)
      << "need 32-bit multiple size to fill with 32-bit pattern";
  return implementation_->Memset32(stream, location, pattern, size);
}

bool StreamExecutor::HostCallback(Stream *stream,
                                  std::function<void()> callback) {
  return implementation_->HostCallback(stream, callback);
}

port::Status StreamExecutor::AllocateEvent(Event *event) {
  return implementation_->AllocateEvent(event);
}

port::Status StreamExecutor::DeallocateEvent(Event *event) {
  return implementation_->DeallocateEvent(event);
}

port::Status StreamExecutor::RecordEvent(Stream *stream, Event *event) {
  return implementation_->RecordEvent(stream, event);
}

port::Status StreamExecutor::WaitForEvent(Stream *stream, Event *event) {
  return implementation_->WaitForEvent(stream, event);
}

Event::Status StreamExecutor::PollForEventStatus(Event *event) {
  return implementation_->PollForEventStatus(event);
}

bool StreamExecutor::AllocateStream(Stream *stream) {
  live_stream_count_.fetch_add(1, std::memory_order_relaxed);
  if (!implementation_->AllocateStream(stream)) {
    auto count = live_stream_count_.fetch_sub(1);
    CHECK_GE(count, 0) << "live stream count should not dip below zero";
    LOG(INFO) << "failed to allocate stream; live stream count: " << count;
    return false;
  }

  return true;
}

void StreamExecutor::DeallocateStream(Stream *stream) {
  implementation_->DeallocateStream(stream);
  CHECK_GE(live_stream_count_.fetch_sub(1), 0)
      << "live stream count should not dip below zero";
}

bool StreamExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  return implementation_->CreateStreamDependency(dependent, other);
}

bool StreamExecutor::AllocateTimer(Timer *timer) {
  return implementation_->AllocateTimer(timer);
}

void StreamExecutor::DeallocateTimer(Timer *timer) {
  return implementation_->DeallocateTimer(timer);
}

bool StreamExecutor::StartTimer(Stream *stream, Timer *timer) {
  return implementation_->StartTimer(stream, timer);
}

bool StreamExecutor::StopTimer(Stream *stream, Timer *timer) {
  return implementation_->StopTimer(stream, timer);
}

DeviceDescription *StreamExecutor::PopulateDeviceDescription() const {
  return implementation_->PopulateDeviceDescription();
}

bool StreamExecutor::DeviceMemoryUsage(int64 *free, int64 *total) const {
  return implementation_->DeviceMemoryUsage(free, total);
}

KernelArg StreamExecutor::DeviceMemoryToKernelArg(
    const DeviceMemoryBase &gpu_mem) const {
  return implementation_->DeviceMemoryToKernelArg(gpu_mem);
}

void StreamExecutor::EnqueueOnBackgroundThread(std::function<void()> task) {
  background_threads_->Schedule(task);
}

void StreamExecutor::CreateAllocRecord(void *opaque, uint64 bytes) {
  if (FLAGS_check_gpu_leaks && opaque != nullptr && bytes != 0) {
    mutex_lock lock{mu_};
    mem_allocs_[opaque] = AllocRecord{
        bytes, ""};
  }
}

void StreamExecutor::EraseAllocRecord(void *opaque) {
  if (FLAGS_check_gpu_leaks && opaque != nullptr) {
    mutex_lock lock{mu_};
    if (mem_allocs_.find(opaque) == mem_allocs_.end()) {
      LOG(ERROR) << "Deallocating unknown pointer: "
                 << port::Printf("0x%p", opaque);
    } else {
      mem_allocs_.erase(opaque);
    }
  }
}

void StreamExecutor::EnableTracing(bool enabled) { tracing_enabled_ = enabled; }

void StreamExecutor::RegisterTraceListener(TraceListener *listener) {
  {
    mutex_lock lock{mu_};
    if (listeners_.find(listener) != listeners_.end()) {
      LOG(INFO) << "Attempt to register already-registered listener, "
                << listener;
    } else {
      listeners_.insert(listener);
    }
  }

  implementation_->RegisterTraceListener(listener);
}

bool StreamExecutor::UnregisterTraceListener(TraceListener *listener) {
  {
    mutex_lock lock{mu_};
    if (listeners_.find(listener) == listeners_.end()) {
      LOG(INFO) << "Attempt to unregister unknown listener, " << listener;
      return false;
    }
    listeners_.erase(listener);
  }

  implementation_->UnregisterTraceListener(listener);
  return true;
}

template <typename TraceCallT, typename... ArgsT>
void StreamExecutor::SubmitTrace(TraceCallT trace_call, ArgsT &&... args) {
  if (tracing_enabled_) {
    {
      // instance tracers held in a block to limit the lock lifetime.
      shared_lock lock{mu_};
      for (TraceListener *listener : listeners_) {
        (listener->*trace_call)(std::forward<ArgsT>(args)...);
      }
    }
  }
}

internal::StreamExecutorInterface *StreamExecutor::implementation() {
  return implementation_->GetUnderlyingExecutor();
}

}  // namespace gputools
}  // namespace perftools
