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

#include "tensorflow/core/platform/device_tracer.h"

#if GOOGLE_CUDA

#include <stdlib.h>
#include <memory>
#include <thread>

#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/cupti_wrapper.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"

namespace {

// Maps a MemcpyKind enum to a const string.
const char *getMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

// Maps a MemoryKind enum to a const string.
const char *getMemoryKindString(CUpti_ActivityMemoryKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
      return "Unknown";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "Pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "Pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "Device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "Array";
    default:
      break;
  }
  return "<unknown>";
}

// Maps an OverheadKind enum to a const string.
const char *getActivityOverheadKindString(CUpti_ActivityOverheadKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return "COMPILER";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return "BUFFER_FLUSH";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return "INSTRUMENTATION";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return "RESOURCE";
    default:
      break;
  }
  return "<unknown>";
}

}  // namespace

namespace tensorflow {
namespace devicetracer {

// Forward declaration.
class CUPTIManager;

// Returns a pointer to the CUPTIManager singleton.
CUPTIManager *GetCUPTIManager();

// Callback interface for consumers of CUPTI tracing.
class CUPTIClient {
 public:
  virtual ~CUPTIClient() {}

  // Records the mapping between correlation ID and kernel name.
  virtual void AddCorrelationId(uint32 correlation_id, const string &name,
                                const string &src_dev, const string &dst_dev) = 0;

  virtual void AddMemcpyRecord(const CUpti_ActivityMemcpy *memcpy) = 0;
  
  virtual void AddMemcpyRecord(const CUpti_ActivityMemcpy2 *memcpy) = 0;

  virtual void AddKernelRecord(const CUpti_ActivityKernel3 *kernel) = 0;

};

#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = cupti_wrapper_->call;                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      LOG(ERROR) << "cuda call " << #call << " failed " << _status; \
    }                                                               \
  } while (0)

// Singleton class to manage registration of CUPTI callbacks.
class CUPTIManager {
 public:
  CUPTIManager() {
    cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());
    CUPTI_CALL(ActivityRegisterCallbacks(BufferRequested, BufferCompleted));
  }

  // Enables tracing and delivers event callbacks to 'client'.
  // Does not take ownership of client.  Client's lifetime must persist
  // until tracing is disabled.
  Status EnableTrace(CUPTIClient *client);

  // Disable tracing.  No further events will be delivered to 'client'.
  Status DisableTrace(CUPTIClient *client);

  void MapThreadToClient(std::thread::id tid, CUPTIClient* client) {
    mutex_lock l(mu_);

    // Mapping tid -> client more than twice 
    if (thread_to_client_.find(tid) != thread_to_client_.end()) {
      CHECK(thread_to_client_[tid] == client);
      return;
    }

    thread_to_client_[tid] = client;
    if (client_to_threads_.find(client) == client_to_threads_.end()) {
      client_to_threads_[client];
    }
    client_to_threads_[client].insert(tid);
  }
  
  void AddCorrelationId(uint32 correlation_id, const string &name,
                        const string &src_dev, const string &dst_dev) {
    auto tid = std::this_thread::get_id();
    mutex_lock l(mu_);
    if (thread_to_client_.find(tid) == thread_to_client_.end()) {
      return;
    }

    auto client = thread_to_client_[tid];
    if (correlation_to_client_.find(correlation_id) != correlation_to_client_.end()) {
      CHECK(correlation_to_client_.find(correlation_id)->second == client);
      return;
    }

    client->AddCorrelationId(correlation_id, name, src_dev, dst_dev);
    correlation_to_client_[correlation_id] = client;
    if (client_to_correlations_.find(client) == client_to_correlations_.end()) {
      client_to_correlations_[client];
    }
    client_to_correlations_[client].insert(correlation_id);
  }

 private:
  // Static functions which we can use as CUPTI callbacks.
  static void BufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
    GetCUPTIManager()->InternalBufferRequested(buffer, size, maxNumRecords);
  }
  static void BufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) {
    GetCUPTIManager()->InternalBufferCompleted(ctx, streamId, buffer, size,
                                               validSize);
  }
  // These methods are called by the static stubs above.
  void InternalBufferRequested(uint8_t **buffer, size_t *size,
                               size_t *maxNumRecords);
  void InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                               uint8_t *buffer, size_t size, size_t validSize);

  // Invoked for each CUPTI activity reported.
  void ActivityCallback(const CUpti_Activity &activity);

  // Size of buffers used for CUPTI tracing.
  static constexpr size_t kBufferSize = 32 * 1024;
  // Required alignment of CUPTI buffers.
  static constexpr size_t kBufferAlignment = 8;

  mutex mu_;
  std::vector<CUPTIClient*>clients_ GUARDED_BY(mu_);
  std::map<std::thread::id, CUPTIClient*> thread_to_client_ GUARDED_BY(mu_);
  std::map<CUPTIClient*, std::set<std::thread::id>> client_to_threads_ GUARDED_BY(mu_);
  std::map<uint32, CUPTIClient*> correlation_to_client_ GUARDED_BY(mu_); 
  std::map<CUPTIClient*, std::set<uint32>> client_to_correlations_ GUARDED_BY(mu_);
  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;

  TF_DISALLOW_COPY_AND_ASSIGN(CUPTIManager);
};

Status CUPTIManager::EnableTrace(CUPTIClient *client) {
  mutex_lock l(mu_);
  // TODO(pbar) Work out the minimal set to trace.
  // We can currently manage without driver/runtime tracing.
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // These might be useful for annotations but require NVTX API.
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));

  if (clients_.size() == 0) {
    CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
    CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
    CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  }
  clients_.push_back(client);
  return Status::OK();
}

Status CUPTIManager::DisableTrace(CUPTIClient *client) {
  CUPTI_CALL(ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

  {
    // Don't acquire this lock until Flush returns, since Flush
    // will potentially cause callbacks into BufferCompleted.
    mutex_lock l(mu_);
    auto position = std::find(clients_.begin(), clients_.end(), client);
    if (position != clients_.end()) {
      clients_.erase(position);
      if (clients_.size() == 0) {
        // We turn off all tracing regardless.
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_DEVICE));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY2));
        CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
      }
      for (auto thread: client_to_threads_[client]) {
        CHECK(thread_to_client_.find(thread) != thread_to_client_.end());
        thread_to_client_.erase(thread_to_client_.find(thread));
      }
      client_to_threads_.erase(client_to_threads_.find(client));
      for (auto correlation: client_to_correlations_[client]) {
        CHECK(correlation_to_client_.find(correlation) != correlation_to_client_.end());
        correlation_to_client_.erase(correlation_to_client_.find(correlation));
      }
      client_to_correlations_.erase(client_to_correlations_.find(client));
    }
  }
  return Status::OK();
}

void CUPTIManager::InternalBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords) {
  VLOG(2) << "BufferRequested";
  void *p = port::AlignedMalloc(kBufferSize, kBufferAlignment);
  *size = kBufferSize;
  *buffer = reinterpret_cast<uint8_t *>(p);
  *maxNumRecords = 0;
}

void CUPTIManager::InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize) {
  VLOG(2) << "BufferCompleted";
  CUptiResult status;
  CUpti_Activity *record = nullptr;
 
  mutex_lock l(mu_);
  if (clients_.size() > 0  && validSize > 0) {
    do {
      status =
          cupti_wrapper_->ActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        ActivityCallback(*record);
      } else {
        break;
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(ActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      LOG(WARNING) << "Dropped " << dropped << " activity records";
    }
  }
  port::AlignedFree(buffer);
}

void CUPTIManager::ActivityCallback(const CUpti_Activity &record) {
  VLOG(2) << "ActivityCallback " << record.kind;

  switch (record.kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy *>(&record);
      CUPTIClient* client = correlation_to_client_[memcpy->correlationId];
      if (!client) {
        break;
      }
      client->AddMemcpyRecord(memcpy);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy2 *>(&record);
      CUPTIClient* client = correlation_to_client_[memcpy->correlationId];
      if (!client) {
        break;
      }
      client->AddMemcpyRecord(memcpy);
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      auto *kernel = reinterpret_cast<const CUpti_ActivityKernel3 *>(&record);
      CUPTIClient* client = correlation_to_client_[kernel->correlationId];
      if (!client) {
        break;
      }
      client->AddKernelRecord(kernel);
      break;
    }
    default:
      VLOG(1) << "ActivityCallback unhandled kind";
      break;
  }
}

CUPTIManager *GetCUPTIManager() {
  static CUPTIManager *manager = new CUPTIManager();
  return manager;
}

#ifdef _MSC_VER
#define __thread __declspec(thread)
#endif

// TODO(pbar) Move this to platform specific header file?
// Static thread local variable for POD types.
#define TF_STATIC_THREAD_LOCAL_POD(_Type_, _var_)                  \
  static __thread _Type_ s_obj_##_var_;                            \
  namespace {                                                      \
  class ThreadLocal_##_var_ {                                      \
   public:                                                         \
    ThreadLocal_##_var_() {}                                       \
    void Init() {}                                                 \
    inline _Type_ *pointer() const { return &s_obj_##_var_; }      \
    inline _Type_ *safe_pointer() const { return &s_obj_##_var_; } \
    _Type_ &get() const { return s_obj_##_var_; }                  \
    bool is_native_tls() const { return true; }                    \
                                                                   \
   private:                                                        \
    TF_DISALLOW_COPY_AND_ASSIGN(ThreadLocal_##_var_);              \
  } _var_;                                                         \
  }  // namespace

// Thread-local state recording the most recent annotation (if any).
// When non-null, this points to a string in the active annotation
// of the current thread.  The annotation is guaranteed to remain live
// for the duration of the CUPTI API callback.
TF_STATIC_THREAD_LOCAL_POD(const char *, tls_current_annotation);
TF_STATIC_THREAD_LOCAL_POD(const char *, tls_current_src_dev);
TF_STATIC_THREAD_LOCAL_POD(const char *, tls_current_dst_dev);

class DeviceTracerImpl : public DeviceTracer,
                         public CUPTIClient,
                         public tracing::TraceCollector {
 public:
  DeviceTracerImpl();
  ~DeviceTracerImpl() override;

  // DeviceTracer interface:
  Status Start() override;
  Status Stop() override;
  Status Collect(StepStatsCollector *collector) override;

  // tracing::TraceCollector interface:
  virtual std::unique_ptr<Handle> CreateAnnotationHandle(
      StringPiece name_part1, StringPiece name_part2,
      StringPiece src_dev, StringPiece dst_dev) const {
    cupti_manager_->MapThreadToClient(std::this_thread::get_id(), const_cast<DeviceTracerImpl*>(this));
    struct Impl : public tracing::TraceCollector::Handle {
      string annotation;
      string src_device;
      string dst_device;
      explicit Impl(string &&name_scope,
                    string src_dev,
                    string dst_dev) : annotation(name_scope),
                                      src_device(src_dev),
                                      dst_device(dst_dev) {
        VLOG(2) << "CreateAnnotationHandle " << annotation;
        // Remember the most recent ScopedAnnotation for each thread.
        tls_current_annotation.get() = annotation.c_str();
        tls_current_src_dev.get() = src_device.c_str();
        tls_current_dst_dev.get() = dst_device.c_str();

      }
      ~Impl() override {
        tls_current_annotation.get() = nullptr;
        tls_current_src_dev.get() = nullptr;
        tls_current_dst_dev.get() = nullptr;
      }
    };
    const string src_dev_str = src_dev.empty() ? "" : src_dev.data();
    const string dst_dev_str = dst_dev.empty() ? "" : dst_dev.data();
    return std::unique_ptr<Handle>(
        new Impl{ConcatenateNames(name_part1, name_part2),
                 src_dev_str, dst_dev_str});
  }

  virtual std::unique_ptr<Handle> CreateActivityHandle(StringPiece, StringPiece,
                                                       bool) const {
    // We don't do anything with 'Activities' yet.
    return nullptr;
  }

 protected:
  // This callback is used exclusively by CUPTIManager.
  friend class CUPTIManager;
  void AddCorrelationId(uint32 correlation_id, const string &name,
                        const string &src_dev, const string &dst_dev) override;

  void AddMemcpyRecord(const CUpti_ActivityMemcpy *memcpy) override {
    mutex_lock l(trace_mu_);
    if (memcpy_records_.size() >= kMaxRecords) return;
    memcpy_records_.push_back(MemcpyRecord{
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
          memcpy->correlationId, memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes});
  }
  
  void AddMemcpyRecord(const CUpti_ActivityMemcpy2 *memcpy) override {
    mutex_lock l(trace_mu_);
    if (memcpy_records_.size() >= kMaxRecords) return;
    memcpy_records_.push_back(MemcpyRecord{
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
          memcpy->correlationId, memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes});
  }

  void AddKernelRecord(const CUpti_ActivityKernel3 *kernel) override {
    mutex_lock l(trace_mu_);
    if (kernel_records_.size() >= kMaxRecords) return;
    kernel_records_.push_back(KernelRecord{kernel->start, kernel->end,
        kernel->deviceId, kernel->streamId,
        kernel->correlationId});
  }

 private:
  // Internal struct to record kernel launches.
  struct KernelRecord {
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint32 device_id;
    uint32 stream_id;
    uint32 correlation_id;
  };
  // Internal struct to record memcpy operations.
  struct MemcpyRecord {
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint32 device_id;
    uint32 stream_id;
    uint32 correlation_id;
    uint8 copyKind;
    uint8 srcKind;
    uint8 dstKind;
    uint64 bytes;
  };

  // This is the subscriber callback which is invoked directly by CUPTI.
  // The 'userdata' argument will be a pointer to the active 'DeviceTracerImpl'.
  static void CUPTIAPI ApiCallback(void *userdata, CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid, const void *cbdata);

  // Returns the current system time in microseconds.
  inline int64 NowInUsec() { return Env::Default()->NowMicros(); }

  void CollectKernelRecord(StepStatsCollector *collector, const KernelRecord &rec);
  void CollectKernelRecord(StepStatsCollector *collector, const MemcpyRecord &rec);
  void CollectMemcpyRecord(StepStatsCollector *collector, const MemcpyRecord &rec);

  CUPTIManager *cupti_manager_;
  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;
  CUpti_SubscriberHandle subscriber_;

  mutex trace_mu_;
  static constexpr size_t kMaxRecords = 1024 * 1024;
  std::map<uint32, string> correlations_ GUARDED_BY(trace_mu_);
  // If not memcpy record, src_dev == requested_dev.
  std::map<uint32, string> src_dev_correlations_ GUARDED_BY(trace_mu_);
  // Only memcpy records have dst_dev.
  std::map<uint32, string> dst_dev_correlations_ GUARDED_BY(trace_mu_);
  std::vector<KernelRecord> kernel_records_ GUARDED_BY(trace_mu_);
  std::vector<MemcpyRecord> memcpy_records_ GUARDED_BY(trace_mu_);

  mutex mu_;
  bool enabled_ GUARDED_BY(mu_);
  bool subscribed_ GUARDED_BY(mu_);
  int64 start_walltime_us_ GUARDED_BY(mu_);
  int64 end_walltime_us_ GUARDED_BY(mu_);
  uint64_t start_timestamp_ GUARDED_BY(mu_);
  uint64_t end_timestamp_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceTracerImpl);
};

DeviceTracerImpl::DeviceTracerImpl() {
  VLOG(1) << "DeviceTracer created.";
  cupti_manager_ = GetCUPTIManager();
  CHECK(cupti_manager_);
  cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());
  enabled_ = false;
  subscribed_ = false;
}

DeviceTracerImpl::~DeviceTracerImpl() {
  // Unregister the CUPTI callbacks if needed to prevent them from accessing
  // freed memory.
  Stop().IgnoreError();
}

Status DeviceTracerImpl::Start() {
  VLOG(1) << "DeviceTracer::Start";
  mutex_lock l(mu_);
  if (enabled_) {
    return errors::FailedPrecondition("DeviceTracer is already enabled.");
  }
  // There can only be one CUPTI subscriber.  If we can't create one then
  // there is another trace in progress (possibly by external code).
  CUptiResult ret;
  ret = cupti_wrapper_->Subscribe(
      &subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), cupti_manager_);
  if (ret != CUPTI_SUCCESS && ret != CUPTI_ERROR_MAX_LIMIT_REACHED) {
    return errors::Internal("Failed to create CUPTI subcriber.");
  }

  // Register as a TraceEngine to receive ScopedAnnotations.
  tracing::SetTraceCollector(this);
  
  if (ret != CUPTI_ERROR_MAX_LIMIT_REACHED) {
    subscribed_ = true;
    // Intercept launch and memcpy calls to capture the Op name annotation.
    // TODO(pbar) Add callbacks for memcpy variants.
    CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                              CUPTI_CB_DOMAIN_DRIVER_API,
                              CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                              CUPTI_CB_DOMAIN_RUNTIME_API,
                              CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020));
    CUPTI_CALL(EnableCallback(
        /*enable=*/1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020));

    CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                              CUPTI_CB_DOMAIN_DRIVER_API,
                              CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2));
    CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                              CUPTI_CB_DOMAIN_DRIVER_API,
                              CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2));
    CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                              CUPTI_CB_DOMAIN_DRIVER_API,
                              CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2));
    CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                              CUPTI_CB_DOMAIN_DRIVER_API,
                              CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2));
    CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                              CUPTI_CB_DOMAIN_DRIVER_API,
                              CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2));
    CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                              CUPTI_CB_DOMAIN_DRIVER_API,
                              CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2));
  }
  TF_RETURN_IF_ERROR(cupti_manager_->EnableTrace(this));

  CUPTI_CALL(GetTimestamp(&start_timestamp_));
  start_walltime_us_ = NowInUsec();
  enabled_ = true;
  return Status::OK();
}

Status DeviceTracerImpl::Stop() {
  VLOG(1) << "DeviceTracer::Stop";
  mutex_lock l(mu_);
  if (!enabled_) {
    return Status::OK();
  }
  if (subscribed_) {
    CUPTI_CALL(Unsubscribe(subscriber_));
  }
  tracing::SetTraceCollector(nullptr);
  TF_RETURN_IF_ERROR(cupti_manager_->DisableTrace(this));
  end_walltime_us_ = NowInUsec();
  CUPTI_CALL(GetTimestamp(&end_timestamp_));
  enabled_ = false;
  return Status::OK();
}

void DeviceTracerImpl::AddCorrelationId(uint32 correlation_id,
                                        const string &name,
                                        const string &src_dev,
                                        const string &dst_dev) {
  VLOG(2) << correlation_id << " : " << name;
  mutex_lock l(trace_mu_);
  if (correlations_.size() >= kMaxRecords) return;
  correlations_.emplace(correlation_id, name);
  src_dev_correlations_.emplace(correlation_id, src_dev);
  if (!dst_dev.empty()) {
    dst_dev_correlations_.emplace(correlation_id, dst_dev);
  }
}

/*static*/ void DeviceTracerImpl::ApiCallback(void *userdata,
                                              CUpti_CallbackDomain domain,
                                              CUpti_CallbackId cbid,
                                              const void *cbdata) {
  auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
  CUPTIManager *manager = reinterpret_cast<CUPTIManager *>(userdata);
  VLOG(2) << "ApiCallback " << domain << ":" << cbid
          << " func: " << cbInfo->functionName;

  // API callbacks are invoked synchronously on the thread making the
  // CUDA API call.  If this pointer is non-null then the ScopedAnnotation
  // must be valid.
  const char *tls_annotation = tls_current_annotation.get();
  const char *tls_src_device = tls_current_src_dev.get();
  const char *tls_dst_device = tls_current_dst_dev.get();
  
  const string src_device =
      tls_src_device ? tls_src_device : "";
  const string dst_device =
      tls_dst_device ? tls_dst_device : "";

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
      (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      auto *params = reinterpret_cast<const cuLaunchKernel_params *>(
          cbInfo->functionParams);
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "LAUNCH stream " << params->hStream << " correllation "
                << cbInfo->correlationId << " kernel " << cbInfo->symbolName;
      }
      const string annotation =
          tls_annotation ? tls_annotation : cbInfo->symbolName;
      manager->AddCorrelationId(cbInfo->correlationId, annotation, src_device, dst_device);
    }
  } else if ((domain == CUPTI_CB_DOMAIN_RUNTIME_API) &&
             (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
              cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      if (VLOG_IS_ON(2)) {
        auto *funcParams = reinterpret_cast<const cudaMemcpy_v3020_params *>(
            cbInfo->functionParams);
        size_t count = funcParams->count;
        enum cudaMemcpyKind kind = funcParams->kind;
        VLOG(2) << "MEMCPY count " << count << " kind " << kind;
      }
      if (tls_annotation) {
        const string annotation = tls_annotation;
        manager->AddCorrelationId(cbInfo->correlationId, annotation, src_device, dst_device);
      }
    }
  } else if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
             (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2)) {
    if (cbInfo->callbackSite == CUPTI_API_EXIT && tls_annotation) {
      const string annotation = tls_annotation;
      manager->AddCorrelationId(cbInfo->correlationId, annotation, src_device, dst_device);
    }
  } else {
    VLOG(1) << "Unhandled API Callback for " << domain << " " << cbid;
  }
}

void DeviceTracerImpl::CollectKernelRecord(StepStatsCollector *collector, 
                                           const KernelRecord &rec) {
  auto it = correlations_.find(rec.correlation_id);
  const string name = (it != correlations_.cend()) ? it->second : "unknown";
  const string stream_device = src_dev_correlations_.find(rec.correlation_id)->second;
  NodeExecStats *ns = new NodeExecStats;
  ns->set_all_start_micros(start_walltime_us_ +
                           ((rec.start_timestamp - start_timestamp_) / 1000));
  ns->set_op_start_rel_micros(0);
  auto elapsed_us =
      std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
  ns->set_op_end_rel_micros(elapsed_us);
  ns->set_all_end_rel_micros(elapsed_us);
  ns->set_node_name(name);
  // TODO(pbar) Generate details based on the kernel activity record.
  // ns->set_timeline_label(details);
  auto nscopy = new NodeExecStats;
  *nscopy = *ns;
  collector->Save(strings::StrCat(stream_device, "/stream:all"), ns);
  collector->Save(strings::StrCat(stream_device, "/stream:", rec.stream_id), nscopy);
}

void DeviceTracerImpl::CollectKernelRecord(StepStatsCollector *collector, 
                                           const MemcpyRecord &rec) {
  const auto kernel_rec = KernelRecord{rec.start_timestamp,
                                       rec.end_timestamp,
                                       rec.device_id,
                                       rec.stream_id,
                                       rec.correlation_id};
  CollectKernelRecord(collector, kernel_rec);
}

void DeviceTracerImpl::CollectMemcpyRecord(StepStatsCollector *collector,
                                           const MemcpyRecord &rec) {
  auto it = correlations_.find(rec.correlation_id);
  const string name = (it != correlations_.cend()) ? it->second : "unknown";
  const string src_device = src_dev_correlations_.find(rec.correlation_id)->second;
  const string dst_device = dst_dev_correlations_.find(rec.correlation_id)->second;
  NodeExecStats *ns = new NodeExecStats;
  ns->set_all_start_micros(start_walltime_us_ +
                           ((rec.start_timestamp - start_timestamp_) / 1000));
  ns->set_op_start_rel_micros(0);
  auto elapsed_us =
      std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
  ns->set_op_end_rel_micros(elapsed_us);
  ns->set_all_end_rel_micros(elapsed_us);
  auto copyKind = static_cast<CUpti_ActivityMemcpyKind>(rec.copyKind);
  auto srcKind = static_cast<CUpti_ActivityMemoryKind>(rec.srcKind);
  auto dstKind = static_cast<CUpti_ActivityMemoryKind>(rec.dstKind);
  const string details = strings::Printf(
      "[%llu] %s from %s to %s", rec.bytes, name.c_str(),
      src_device.c_str(), dst_device.c_str());
  ns->set_node_name(
      strings::StrCat(name, ":MEMCPY", getMemcpyKindString(copyKind)));
  ns->set_timeline_label(details);
  auto nscopy = new NodeExecStats;
  *nscopy = *ns;
  collector->Save(strings::StrCat(dst_device, "/memcpy:all"), ns);
}

Status DeviceTracerImpl::Collect(StepStatsCollector *collector) {
  mutex_lock l(mu_);
  if (enabled_) {
    return errors::FailedPrecondition("DeviceTracer is still enabled.");
  }

  mutex_lock l2(trace_mu_);
  for (const auto &rec : kernel_records_) {
    CollectKernelRecord(collector, rec);
  }
  for (const auto &rec : memcpy_records_) {
    // Some memcpy_records are actually kernel_records.
    // (those that don't have dst_dev)
    if (dst_dev_correlations_.find(rec.correlation_id) == dst_dev_correlations_.cend()) {
      CollectKernelRecord(collector, rec);
    } else {
      CollectMemcpyRecord(collector, rec);
    }
  }
  return Status::OK();
}

}  // namespace devicetracer

std::unique_ptr<DeviceTracer> CreateDeviceTracer() {
  std::unique_ptr<DeviceTracer> tracer(new devicetracer::DeviceTracerImpl());
  return tracer;
}

}  // namespace tensorflow

#else  // GOOGLE_CUDA

namespace tensorflow {

std::unique_ptr<DeviceTracer> CreateDeviceTracer() { return nullptr; }

DeviceTracer* CreateDeviceTracerRaw() { return nullptr; }
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
