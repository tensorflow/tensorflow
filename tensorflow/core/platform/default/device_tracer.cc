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

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "cuda/extras/CUPTI/include/cupti.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/internal/cpu/host_tracer.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace {
Status ToStatus(CUptiResult result) {
  if (result == CUPTI_SUCCESS) {
    return Status::OK();
  }
  const char* str = nullptr;
  cuptiGetResultString(result, &str);
  return errors::Unavailable("CUPTI error: ", str ? str : "<unknown>");
}

Status ToStatus(CUresult result) {
  if (result == CUDA_SUCCESS) {
    return Status::OK();
  }
  const char* str = nullptr;
  cuGetErrorName(result, &str);
  return errors::Unavailable("CUDA error: ", str ? str : "<unknown>");
}

void LogIfError(const Status& status) {
  if (status.ok()) {
    return;
  }
  LOG(ERROR) << status.error_message();
}

bool IsAscii(string& str) {
  for (auto& ch : str) {
    if (!absl::ascii_isascii(ch)) {
      return false;
    }
  }
  return true;
}

struct KernelRecord {
  const char* kernel_name;
  // TODO(csigg): cuStreamGetCtx introduced in CUDA 9.2 would allow us to only
  // record the stream and infer the context during collection.
  CUcontext context;
  CUstream stream;
  CUevent start_event;
  CUevent stop_event;
  const std::string* annotation;
};

struct MemcpyRecord {
  CUmemorytype src_type;
  CUmemorytype dst_type;
  size_t size_bytes;
  CUcontext context;
  CUstream stream;
  CUevent start_event;
  CUevent stop_event;
  const std::string* annotation;
};

Status CreateAndRecordEvent(CUevent* event, CUstream stream) {
  TF_RETURN_IF_ERROR(ToStatus(cuEventCreate(event, CU_EVENT_DEFAULT)));
  return ToStatus(cuEventRecord(*event, stream));
}

// Thread-local state recording the most recent annotation (if any).
// When non-null, this points to a string in the active annotation
// of the current thread.  The annotation is guaranteed to remain live
// for the duration of the CUPTI API callback.
static thread_local const char* tls_current_annotation;

// Stores a series of kernel and memcpy records.
class CudaEventRecorder {
 public:
  // Registers the start of a kernel launch. The returned index should be passed
  // to StopKernel() after the kernel launch has completed.
  size_t StartKernel(const char* kernel_name, CUcontext context,
                     CUstream stream) {
    KernelRecord record = {kernel_name, context, stream};
    LogIfError(CreateAndRecordEvent(&record.start_event, stream));
    mutex_lock lock(mutex_);
    if (tls_current_annotation) {
      record.annotation = &*annotations_.emplace(tls_current_annotation).first;
    }
    kernel_records_.push_back(record);
    return kernel_records_.size() - 1;
  }
  void StopKernel(size_t index) {
    mutex_lock lock(mutex_);
    auto& record = kernel_records_[index];
    LogIfError(CreateAndRecordEvent(&record.stop_event, record.stream));
  }

  // Registers the start of a copy operation. The returned index should be
  // passed to StopMemcpy() after the kernel launch has completed.
  size_t StartMemcpy(CUmemorytype src_type, CUmemorytype dst_type,
                     size_t size_bytes, CUcontext context, CUstream stream) {
    MemcpyRecord record = {src_type, dst_type, size_bytes, context, stream};
    LogIfError(CreateAndRecordEvent(&record.start_event, stream));
    mutex_lock lock(mutex_);
    if (tls_current_annotation) {
      record.annotation = &*annotations_.emplace(tls_current_annotation).first;
    }
    memcpy_records_.push_back(record);
    return memcpy_records_.size() - 1;
  }
  void StopMemcpy(size_t index) {
    mutex_lock lock(mutex_);
    auto& record = memcpy_records_[index];
    LogIfError(CreateAndRecordEvent(&record.stop_event, record.stream));
  }

  std::vector<KernelRecord> ConsumeKernelRecords() {
    mutex_lock lock(mutex_);
    return std::move(kernel_records_);
  }
  std::vector<MemcpyRecord> ConsumeMemcpyRecords() {
    mutex_lock lock(mutex_);
    return std::move(memcpy_records_);
  }

 private:
  mutex mutex_;
  std::unordered_set<std::string> annotations_ GUARDED_BY(mutex_);
  std::vector<KernelRecord> kernel_records_ GUARDED_BY(mutex_);
  std::vector<MemcpyRecord> memcpy_records_ GUARDED_BY(mutex_);
};

// Instances register callbacks with CUPTI to notify the event recorder before
// and after kernel launches and memory copies.
class CuptiCallbackHook {
 public:
  CuptiCallbackHook() : subscriber_(nullptr) {}

  Status Enable(CudaEventRecorder* recorder) {
    TF_RETURN_IF_ERROR(
        ToStatus(cuptiSubscribe(&subscriber_, &CuptiCallback, recorder)));
    for (auto cbid : {CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
                      CUPTI_DRIVER_TRACE_CBID_cuMemcpy,
                      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync,
                      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2,
                      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,
                      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2,
                      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2,
                      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2,
                      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2}) {
      TF_RETURN_IF_ERROR(ToStatus(cuptiEnableCallback(
          /*enable=*/1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid)));
    }
    return Status::OK();
  }

  ~CuptiCallbackHook() { LogIfError(ToStatus(cuptiUnsubscribe(subscriber_))); }

 private:
  static void CUPTIAPI CuptiCallback(void* userdata,
                                     CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid,
                                     const void* cbdata) {
    auto recorder = static_cast<CudaEventRecorder*>(userdata);
    auto data = static_cast<const CUpti_CallbackData*>(cbdata);
    DCHECK_EQ(domain, CUPTI_CB_DOMAIN_DRIVER_API);

    if (data->callbackSite == CUPTI_API_ENTER) {
      DriverApiEnterCallback(cbid, *data, recorder);
    } else {
      DriverApiExitCallback(cbid, *data, recorder);
    }
  }

  static CUmemorytype GetMemoryType(CUdeviceptr ptr) {
    CUmemorytype mem_type;
    auto status =
        cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {
      // Pointer not registered with CUDA, must be host memory.
      return CU_MEMORYTYPE_HOST;
    }
    LogIfError(ToStatus(status));
    return mem_type;
  }

  template <typename T>
  static void StartMemcpy(CUmemorytype src_type, CUmemorytype dst_type,
                          const CUpti_CallbackData& cbdata,
                          CudaEventRecorder* recorder) {
    auto params = static_cast<const T*>(cbdata.functionParams);
    *cbdata.correlationData = recorder->StartMemcpy(
        src_type, dst_type, params->ByteCount, cbdata.context, nullptr);
  }
  template <typename T>
  static void StartMemcpyAsync(CUmemorytype src_type, CUmemorytype dst_type,
                               const CUpti_CallbackData& cbdata,
                               CudaEventRecorder* recorder) {
    auto params = static_cast<const T*>(cbdata.functionParams);
    *cbdata.correlationData = recorder->StartMemcpy(
        src_type, dst_type, params->ByteCount, cbdata.context, params->hStream);
  }

  static void DriverApiEnterCallback(CUpti_CallbackId cbid,
                                     const CUpti_CallbackData& cbdata,
                                     CudaEventRecorder* recorder) {
    switch (cbid) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: {
        DCHECK_NE(cbdata.symbolName, nullptr);
        auto params =
            static_cast<const cuLaunchKernel_params*>(cbdata.functionParams);
        *cbdata.correlationData = recorder->StartKernel(
            cbdata.symbolName, cbdata.context, params->hStream);
        return;
      }

      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy: {
        auto params =
            static_cast<const cuMemcpy_params*>(cbdata.functionParams);
        return StartMemcpy<cuMemcpy_params>(GetMemoryType(params->src),
                                            GetMemoryType(params->dst), cbdata,
                                            recorder);
      }
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync: {
        auto params =
            static_cast<const cuMemcpyAsync_params*>(cbdata.functionParams);
        return StartMemcpyAsync<cuMemcpyAsync_params>(
            GetMemoryType(params->src), GetMemoryType(params->dst), cbdata,
            recorder);
      }

      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
        return StartMemcpy<cuMemcpyHtoD_v2_params>(
            CU_MEMORYTYPE_HOST, CU_MEMORYTYPE_DEVICE, cbdata, recorder);

      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
        return StartMemcpyAsync<cuMemcpyHtoDAsync_v2_params>(
            CU_MEMORYTYPE_HOST, CU_MEMORYTYPE_DEVICE, cbdata, recorder);

      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
        return StartMemcpy<cuMemcpyDtoH_v2_params>(
            CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, cbdata, recorder);
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
        return StartMemcpyAsync<cuMemcpyDtoHAsync_v2_params>(
            CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, cbdata, recorder);

      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
        return StartMemcpy<cuMemcpyDtoD_v2_params>(
            CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_DEVICE, cbdata, recorder);
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
        return StartMemcpyAsync<cuMemcpyDtoDAsync_v2_params>(
            CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_DEVICE, cbdata, recorder);

      default:
        LOG(ERROR) << "Unexpected callback id: " << cbid;
    }
  }

  static void DriverApiExitCallback(CUpti_CallbackId cbid,
                                    const CUpti_CallbackData& cbdata,
                                    CudaEventRecorder* recorder) {
    switch (cbid) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
        recorder->StopKernel(*cbdata.correlationData);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
        recorder->StopMemcpy(*cbdata.correlationData);
        break;
      default:
        LOG(ERROR) << "Unexpected callback id: " << cbid;
    }
  }

  CUpti_SubscriberHandle subscriber_;
};
}  // namespace

class TraceCollectorImpl : public tracing::TraceCollector {
 public:
  class ActivityHandle : public Handle {
   public:
    ActivityHandle(std::string&& name, int level)
        : trace_me_(std::move(name), level) {}

   private:
    profiler::TraceMe trace_me_;
  };
  TraceCollectorImpl() { tracing::SetTraceCollector(this); }

  ~TraceCollectorImpl() override {
    DCHECK(!active_trace_session_)
        << "Unexpected active trace session detected. ";
  }

  // Note the method can be called after a call to Stop().
  virtual std::unique_ptr<Handle> CreateAnnotationHandle(
      StringPiece name_part1, StringPiece name_part2) const {
    struct Impl : public tracing::TraceCollector::Handle {
      std::string annotation;
      explicit Impl(std::string&& name_scope) : annotation(name_scope) {
        VLOG(2) << "CreateAnnotationHandle " << annotation;
        // Remember the most recent ScopedAnnotation for each thread.
        tls_current_annotation = annotation.c_str();
      }
      ~Impl() override { tls_current_annotation = nullptr; }
    };
    return absl::make_unique<Impl>(ConcatenateNames(name_part1, name_part2));
  }

  virtual std::unique_ptr<Handle> CreateActivityHandle(
      StringPiece name_part1, StringPiece name_part2, bool is_expensive) const {
    if (!IsEnabledForActivities(is_expensive)) {
      return nullptr;
    }
    return absl::make_unique<ActivityHandle>(
        ConcatenateNames(name_part1, name_part2), GetLevel(is_expensive));
  }

  bool IsEnabledForAnnotations() const override {
    return active_trace_session_.load(std::memory_order_relaxed);
  }

  bool IsEnabledForActivities(bool is_expensive) const override {
    return profiler::TraceMeRecorder::Active(GetLevel(is_expensive));
  }

  void Start() {
    DCHECK(!active_trace_session_)
        << "Unexpected active trace session detected. ";
    active_trace_session_ = true;
  }

  void Stop() {
    DCHECK(active_trace_session_) << "No active trace session detected. ";
    active_trace_session_ = false;
  }

 private:
  static int GetLevel(bool is_expensive) {
    return profiler::GetTFTraceMeLevel(is_expensive);
  }

  std::atomic<bool> active_trace_session_;
};

TraceCollectorImpl* GlobalDefaultTraceCollector() {
  static auto* instance = new TraceCollectorImpl();
  return instance;
}

class DeviceTracerImpl : public DeviceTracer {
 public:
  DeviceTracerImpl();
  ~DeviceTracerImpl() override;

  // DeviceTracer interface:
  Status Start() override;
  Status Stop() override;
  Status Collect(StepStatsCollector* collector) override;

 private:
  std::unique_ptr<CudaEventRecorder> recorder_;
  std::unique_ptr<CuptiCallbackHook> cupti_hook_;

  mutex mu_;
  bool enabled_ GUARDED_BY(mu_);
  std::unique_ptr<profiler::cpu::HostTracer> host_tracer_ GUARDED_BY(mu_);
};

DeviceTracerImpl::DeviceTracerImpl() : recorder_(new CudaEventRecorder()) {
  VLOG(1) << "DeviceTracer created.";
  host_tracer_ = profiler::cpu::HostTracer::Create(2);
  enabled_ = false;
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
  cupti_hook_.reset(new CuptiCallbackHook());
  TF_RETURN_IF_ERROR(cupti_hook_->Enable(recorder_.get()));

  // Register as a TraceEngine to receive ScopedAnnotations.
  GlobalDefaultTraceCollector()->Start();

  host_tracer_->Start().IgnoreError();
  enabled_ = true;
  return Status::OK();
}

Status DeviceTracerImpl::Stop() {
  VLOG(1) << "DeviceTracer::Stop";
  mutex_lock l(mu_);
  if (!enabled_) {
    return Status::OK();
  }
  cupti_hook_.reset();
  GlobalDefaultTraceCollector()->Stop();

  enabled_ = false;
  host_tracer_->Stop().IgnoreError();
  return Status::OK();
}

namespace {
class CudaEventCollector {
  struct DeviceInfo {
    int ordinal;
    std::string name;
    int num_contexts;
  };

  struct ContextInfo {
    int index;
    const DeviceInfo* dev_info;
    int num_streams;
    CUevent end_event;
  };

  struct StreamInfo {
    std::string name;
    int index;  // 0 is reserved for null stream.
    const ContextInfo* ctx_info;
  };

  // Include context in key to distinguish null streams.
  using StreamKey = std::pair<CUcontext, CUstream>;

  CudaEventCollector(CudaEventRecorder* recorder, StepStatsCollector* collector)
      : recorder_(recorder), collector_(collector) {
    DCHECK(recorder != nullptr);
    DCHECK(collector != nullptr);
  }

  // Populates device_infos_ from all devices.
  Status InitializeDeviceInfos() {
    int count;
    TF_RETURN_IF_ERROR(ToStatus(cuDeviceGetCount(&count)));
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      CUdevice device;
      TF_RETURN_IF_ERROR(ToStatus(cuDeviceGet(&device, ordinal)));
      char name[100];
      TF_RETURN_IF_ERROR(ToStatus(cuDeviceGetName(name, sizeof(name), device)));
      device_infos_[device] = {ordinal, name};
    }
    return Status::OK();
  }

  // Returns element from context_infos_, adding it if not yet present.
  Status GetContextInfo(CUcontext context, ContextInfo** ctx_info_ptr) {
    auto it = context_infos_.find(context);

    if (it == context_infos_.end()) {
      TF_RETURN_IF_ERROR(ToStatus(cuCtxSetCurrent(context)));
      CUdevice device;
      TF_RETURN_IF_ERROR(ToStatus(cuCtxGetDevice(&device)));

      auto& dev_info = device_infos_[device];
      ContextInfo ctx_info = {dev_info.num_contexts++, &dev_info};
      it = context_infos_.emplace(context, ctx_info).first;
    }

    *ctx_info_ptr = &it->second;
    return Status::OK();
  }

  // Adds element to stream_infos_ if not yet present. If present, clear name
  // if it doesn't match parameter.
  Status AddStreamInfo(CUcontext context, CUstream stream,
                       absl::string_view name) {
    StreamKey key(context, stream);
    auto it = stream_infos_.find(key);
    if (it != stream_infos_.end()) {
      if (it->second.name != name) {
        it->second.name.clear();  // Stream with inconsistent names, clear it.
      }
      return Status::OK();
    }

    ContextInfo* ctx_info;
    TF_RETURN_IF_ERROR(GetContextInfo(context, &ctx_info));
    int index = stream ? ++ctx_info->num_streams : 0;
    StreamInfo stream_info = {static_cast<std::string>(name), index, ctx_info};
    stream_infos_.emplace(key, stream_info);
    return Status::OK();
  }

  // Returns string describing source and destination memory types.
  static std::string GetMemcpyName(const MemcpyRecord& record) {
    auto get_memory_type = [](CUmemorytype mem_type) {
      switch (mem_type) {
        case CU_MEMORYTYPE_HOST:
          return 'H';
        case CU_MEMORYTYPE_DEVICE:
          return 'D';
        case CU_MEMORYTYPE_ARRAY:
          return 'A';
        case CU_MEMORYTYPE_UNIFIED:
          return 'U';
        default:
          LOG(ERROR) << "Unknown memory type: " << mem_type;
          return '?';
      }
    };
    return absl::StrFormat("Memcpy%cto%c", get_memory_type(record.src_type),
                           get_memory_type(record.dst_type));
  }

  // Returns time in microseconds between events recorded on the GPU.
  static uint64_t GetElasedTimeUs(CUevent start, CUevent stop) {
    float elapsed_ms = 0.0f;
    LogIfError(ToStatus(cuEventElapsedTime(&elapsed_ms, start, stop)));
    return static_cast<uint64>(
        std::llroundf(1000 * std::max(elapsed_ms, 0.0f)));
  }

  // Synchronizes all contexts.
  Status Synchronize() const {
    for (const auto& pair : context_infos_) {
      TF_RETURN_IF_ERROR(ToStatus(cuCtxSetCurrent(pair.first)));
      TF_RETURN_IF_ERROR(ToStatus(cuCtxSynchronize()));
    }
    return Status::OK();
  }

  // Save stats to collector;
  Status SaveStats(std::unique_ptr<NodeExecStats> stats,
                   const StreamInfo& stream_info) const {
    auto ctx_info = stream_info.ctx_info;
    auto dev_info = ctx_info->dev_info;
    // TODO(csigg): tfprof_node.cc, run_metadata_test.py, and timeline_test.py
    // currently require this particular formatting.
    collector_->Save(
        absl::StrFormat("/device:GPU:%d/stream:all", dev_info->ordinal),
        new NodeExecStats(*stats));
    auto name = absl::StrFormat("/gpu:%d (%s)/context#%d/", dev_info->ordinal,
                                dev_info->name, ctx_info->index);
    if (stream_info.index) {
      absl::StrAppend(&name, "stream#", std::to_string(stream_info.index));
    } else {
      absl::StrAppend(&name, "null stream");
    }
    if (!stream_info.name.empty()) {
      absl::StrAppend(&name, ":", stream_info.name);
    }
    collector_->Save(name, stats.release());
    return Status::OK();
  }

  Status SaveRecord(const KernelRecord& record) const {
    if (!record.start_event || !record.stop_event) {
      return Status::OK();
    }
    const auto& stream_info =
        stream_infos_.at(StreamKey(record.context, record.stream));
    auto start_us =
        GetElasedTimeUs(record.start_event, stream_info.ctx_info->end_event);
    auto elapsed_us = GetElasedTimeUs(record.start_event, record.stop_event);

    auto stats = absl::make_unique<NodeExecStats>();
    std::string node_name = record.kernel_name;
    // Sometimes CUPTI returns invalid characters. See b/129892466.
    if (!IsAscii(node_name)) {
      node_name = "<invalid_name>";
    }
    if (record.annotation) {
      node_name = absl::StrCat(*record.annotation, "::", node_name);
    }
    stats->set_node_name(node_name);
    // TODO(csigg): Report grid size?
    std::string node_label;
    stats->set_timeline_label(node_label);
    stats->set_all_start_micros(end_walltime_us_ - start_us);
    stats->set_op_end_rel_micros(elapsed_us);
    stats->set_all_end_rel_micros(elapsed_us);
    return SaveStats(std::move(stats), stream_info);
  }

  Status SaveRecord(const MemcpyRecord& record) const {
    if (!record.start_event || !record.stop_event) {
      return Status::OK();
    }
    const auto& stream_info =
        stream_infos_.at(StreamKey(record.context, record.stream));
    auto start_us =
        GetElasedTimeUs(record.start_event, stream_info.ctx_info->end_event);
    auto elapsed_us = GetElasedTimeUs(record.start_event, record.stop_event);

    auto stats = absl::make_unique<NodeExecStats>();
    std::string node_name = GetMemcpyName(record);
    // Sometimes CUPTI returns invalid characters. See b/129892466.
    if (!IsAscii(node_name)) {
      node_name = "<invalid_name>";
    }
    if (record.annotation) {
      node_name = absl::StrCat(*record.annotation, "::", node_name);
    }
    stats->set_node_name(node_name);
    // TODO(csigg): Show label in Chrome trace viewer.
    std::string node_label = absl::StrFormat("%d bytes", record.size_bytes);
    stats->set_timeline_label(node_label);
    stats->set_all_start_micros(end_walltime_us_ - start_us);
    stats->set_op_end_rel_micros(elapsed_us);
    stats->set_all_end_rel_micros(elapsed_us);
    return SaveStats(std::move(stats), stream_info);
  }

  Status Collect() {
    TF_RETURN_IF_ERROR(InitializeDeviceInfos());

    auto kernel_records = recorder_->ConsumeKernelRecords();
    auto memcpy_records = recorder_->ConsumeMemcpyRecords();
    LOG(INFO) << "Collecting " << kernel_records.size() << " kernel records, "
              << memcpy_records.size() << " memcpy records.";

    // Gather all profiled streams and contexts.
    for (const auto& record : kernel_records) {
      TF_RETURN_IF_ERROR(
          AddStreamInfo(record.context, record.stream, "Kernel"));
    }
    for (const auto& record : memcpy_records) {
      TF_RETURN_IF_ERROR(
          AddStreamInfo(record.context, record.stream, GetMemcpyName(record)));
    }

    // Synchronize all contexts, record end events, synchronize again.
    TF_RETURN_IF_ERROR(Synchronize());
    for (auto& pair : context_infos_) {
      TF_RETURN_IF_ERROR(ToStatus(cuCtxSetCurrent(pair.first)));
      TF_RETURN_IF_ERROR(CreateAndRecordEvent(&pair.second.end_event, nullptr));
    }
    TF_RETURN_IF_ERROR(Synchronize());
    end_walltime_us_ = Env::Default()->NowMicros();

    for (const auto& record : kernel_records) {
      TF_RETURN_IF_ERROR(SaveRecord(record));
    }
    for (const auto& record : memcpy_records) {
      TF_RETURN_IF_ERROR(SaveRecord(record));
    }

    return Status::OK();
  }

 public:
  // Consumes the records in recorder and saves them to the collector.
  static Status Collect(CudaEventRecorder* recorder,
                        StepStatsCollector* collector) {
    CUcontext context;
    TF_RETURN_IF_ERROR(ToStatus(cuCtxGetCurrent(&context)));
    auto status = CudaEventCollector(recorder, collector).Collect();
    TF_RETURN_IF_ERROR(ToStatus(cuCtxSetCurrent(context)));
    return status;
  }

 private:
  CudaEventRecorder* recorder_;
  StepStatsCollector* collector_;

  absl::node_hash_map<CUdevice, DeviceInfo> device_infos_;
  absl::node_hash_map<CUcontext, ContextInfo> context_infos_;
  absl::flat_hash_map<StreamKey, StreamInfo, hash<StreamKey>> stream_infos_;
  int64 end_walltime_us_;
};
}  // namespace

Status DeviceTracerImpl::Collect(StepStatsCollector* collector) {
  mutex_lock l(mu_);
  if (enabled_) {
    return errors::FailedPrecondition("DeviceTracer is still enabled.");
  }

  TF_RETURN_IF_ERROR(CudaEventCollector::Collect(recorder_.get(), collector));
  host_tracer_->CollectDataToCollector(collector).IgnoreError();
  return Status::OK();
}

std::unique_ptr<DeviceTracer> CreateDeviceTracer() {
  auto status = cuInit(0);
  if (status != CUDA_SUCCESS) {
    LogIfError(ToStatus(status));
    return nullptr;
  }
  return absl::make_unique<DeviceTracerImpl>();
}
}  // namespace tensorflow
#else  // GOOGLE_CUDA

namespace tensorflow {

std::unique_ptr<DeviceTracer> CreateDeviceTracer() { return nullptr; }

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
