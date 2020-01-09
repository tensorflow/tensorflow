/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#include <stdlib.h>

#include <memory>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/internal/annotation_stack.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_tracer.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_wrapper.h"
#include "tensorflow/core/profiler/internal/parse_annotation.h"
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {

namespace {

bool IsHostEvent(const CuptiTracerEvent& event) {
  // DriverCallback(i.e. kernel launching) events are host events.
  if (event.source == CuptiTracerEventSource::DriverCallback) return true;
  // Non-overhead activity events are device events.
  if (event.type != CuptiTracerEventType::Overhead) return false;
  // Overhead events can be associated with a thread or a stream, etc.
  // If a valid thread id is specified, we consider it as a host event.
  return event.thread_id != CuptiTracerEvent::kInvalidThreadId;
}

void CreateXEvent(const CuptiTracerEvent& event, uint64 offset_ns,
                  XPlaneBuilder* plane, XLineBuilder* line) {
  std::string kernel_name = port::MaybeAbiDemangle(event.name.c_str());
  XEventMetadata* event_metadata = plane->GetOrCreateEventMetadata(kernel_name);
  XEventBuilder xevent = line->AddEvent(*event_metadata);
  xevent.SetTimestampNs(event.start_time_ns + offset_ns);
  xevent.SetEndTimestampNs(event.end_time_ns + offset_ns);
  if (event.correlation_id != CuptiTracerEvent::kInvalidCorrelationId) {
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kCorrelationId)),
                        event.correlation_id);
  }
  if (event.context_id != CuptiTracerEvent::kInvalidContextId) {
    xevent.AddStatValue(
        *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kContextId)),
        absl::StrCat("$$", static_cast<uint64>(event.context_id)));
  }
  if (event.type == CuptiTracerEventType::Kernel) {
    const std::string kernel_details =
        absl::StrFormat("regs:%u shm:%u grid:%u,%u,%u block:%u,%u,%u",
                        event.kernel_info.registers_per_thread,
                        event.kernel_info.static_shared_memory_usage,
                        event.kernel_info.grid_x, event.kernel_info.grid_y,
                        event.kernel_info.grid_z, event.kernel_info.block_x,
                        event.kernel_info.block_y, event.kernel_info.block_z);
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kKernelDetails)),
                        kernel_details);
  }
  if (event.type == CuptiTracerEventType::MemcpyH2D ||
      event.type == CuptiTracerEventType::MemcpyD2H ||
      event.type == CuptiTracerEventType::MemcpyD2D ||
      event.type == CuptiTracerEventType::MemcpyP2P ||
      event.type == CuptiTracerEventType::MemcpyOther) {
    const auto& memcpy_info = event.memcpy_info;
    std::string memcpy_details =
        absl::StrFormat("size:%u dest:%u async:%u", memcpy_info.num_bytes,
                        memcpy_info.destination, memcpy_info.async);
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kMemcpyDetails)),
                        memcpy_details);
  }
  if (event.type == CuptiTracerEventType::MemoryAlloc) {
    std::string memalloc_details =
        absl::StrFormat("num_bytes:%u", event.memalloc_info.num_bytes);
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kMemallocDetails)),
                        memalloc_details);
  }

  std::vector<Annotation> annotation_stack =
      ParseAnnotationStack(event.annotation);
  for (int i = 0; i < annotation_stack.size(); ++i) {
    xevent.AddStatValue(
        *plane->GetOrCreateStatMetadata(absl::StrCat("level ", i)),
        annotation_stack[i].name);
  }
  // If multiple metadata have the same key name, show the values from the top
  // of the stack (innermost annotation). Concatenate the values from "hlo_op".
  absl::flat_hash_set<absl::string_view> key_set;
  std::vector<absl::string_view> hlo_op_names;
  for (auto annotation = annotation_stack.rbegin();
       annotation != annotation_stack.rend(); ++annotation) {
    for (const Annotation::Metadata& metadata : annotation->metadata) {
      if (metadata.key == "tf_op") {
        continue;  // ignored, obtained from HLO proto via DebugInfoMap
      } else if (key_set.insert(metadata.key).second) {
        xevent.ParseAndAddStatValue(
            *plane->GetOrCreateStatMetadata(metadata.key), metadata.value);
      }
    }
  }
}
}  // namespace

// CuptiTraceCollectorImpl store the CuptiTracerEvents from CuptiTracer and
// eventually convert and filter them to StepStats or XSpace.
class CuptiTraceCollectorImpl : public CuptiTraceCollector {
 public:
  CuptiTraceCollectorImpl(const CuptiTracerCollectorOptions& option,
                          uint64 start_walltime_ns, uint64 start_gpu_ns)
      : CuptiTraceCollector(option),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gpu_ns_(start_gpu_ns),
        num_gpus_(option.num_gpus),
        per_device_collector_(option.num_gpus) {}

  void AddEvent(CuptiTracerEvent&& event) override {
    if (event.device_id >= num_gpus_) return;
    if (event.source == CuptiTracerEventSource::DriverCallback) {
      if (num_callback_events_ > options_.max_callback_api_events) {
        OnEventsDropped("trace collector", 1);
        return;
      }
      num_callback_events_++;
    } else {
      if (num_activity_events_ > options_.max_activity_api_events) {
        OnEventsDropped("trace collector", 1);
        return;
      }
      num_activity_events_++;
    }
    per_device_collector_[event.device_id].AddEvent(std::move(event));
  }
  void OnEventsDropped(const std::string& reason, uint32 num_events) override {}
  void Flush() override {}
  void Export(StepStatsCollector* trace_collector) {
    LOG(INFO) << " GpuTracer has collected " << num_callback_events_
              << " callback api events and " << num_activity_events_
              << " activity events.";
    for (int i = 0; i < num_gpus_; ++i) {
      per_device_collector_[i].Flush(i, start_walltime_ns_, start_gpu_ns_,
                                     trace_collector);
    }
  }
  void Export(XSpace* space) {
    LOG(INFO) << " GpuTracer has collected " << num_callback_events_
              << " callback api events and " << num_activity_events_
              << " activity events.";
    XPlaneBuilder host_plane(GetOrCreatePlane(space, kHostThreads));
    for (int device_ordinal = 0; device_ordinal < num_gpus_; ++device_ordinal) {
      std::string name = absl::StrCat(kGpuPlanePrefix, device_ordinal);
      XPlaneBuilder device_plane(GetOrCreatePlane(space, name));
      per_device_collector_[device_ordinal].Flush(
          start_walltime_ns_, start_gpu_ns_, &device_plane, &host_plane);
    }
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64 start_walltime_ns_;
  uint64 start_gpu_ns_;
  int num_gpus_;

  struct CorrelationInfo {
    CorrelationInfo(uint32 t, uint32 e) : thread_id(t), enqueue_time_ns(e) {}
    uint32 thread_id;
    uint64 enqueue_time_ns;
  };
  struct PerDeviceCollector {
    void AddEvent(CuptiTracerEvent&& event) {
      absl::MutexLock lock(&mutex);
      if (event.source == CuptiTracerEventSource::DriverCallback) {
        // Cupti api callback events were used to populate launch times etc.
        if (event.correlation_id != CuptiTracerEvent::kInvalidCorrelationId) {
          correlation_info.insert(
              {event.correlation_id,
               CorrelationInfo(event.thread_id, event.start_time_ns)});
        }
        events.emplace_back(std::move(event));
      } else {
        // Cupti activity events measure device times etc.
        events.emplace_back(std::move(event));
      }
    }

    void Flush(int32 device_ordinal, uint64 start_walltime_ns,
               uint64 start_gpu_ns, StepStatsCollector* collector) {
      absl::MutexLock lock(&mutex);
      stream_device = absl::StrCat("/device:GPU:", device_ordinal, "/stream:");
      memcpy_device = absl::StrCat("/device:GPU:", device_ordinal, "/memcpy");
      sync_device = absl::StrCat("/device:GPU:", device_ordinal, "/sync");
      for (auto& event : events) {
        NodeExecStats* ns = new NodeExecStats;
        ns->set_all_start_micros(
            (start_walltime_ns + (event.start_time_ns - start_gpu_ns)) / 1000);
        ns->set_op_start_rel_micros(0);
        auto elapsed_ns = event.end_time_ns - event.start_time_ns;
        ns->set_op_end_rel_micros(elapsed_ns / 1000);
        ns->set_all_end_rel_micros(elapsed_ns / 1000);

        if (event.source == CuptiTracerEventSource::DriverCallback) {
          // Legacy code ignore all other launch events except
          // cuStreamSynchronize.
          if (event.name == "cuStreamSynchronize") {
            ns->set_node_name(event.name);
            ns->set_timeline_label(absl::StrCat("ThreadId ", event.thread_id));
            ns->set_thread_id(event.thread_id);
            collector->Save(sync_device, ns);
          }
        } else {  // CuptiTracerEventSource::Activity
          // Get launch information if available.
          if (event.correlation_id != CuptiTracerEvent::kInvalidCorrelationId) {
            auto it = correlation_info.find(event.correlation_id);
            if (it != correlation_info.end()) {
              ns->set_scheduled_micros(it->second.enqueue_time_ns / 1000);
              ns->set_thread_id(it->second.thread_id);
            }
          }

          auto annotation_stack = ParseAnnotationStack(event.annotation);
          std::string kernel_name = port::MaybeAbiDemangle(event.name.c_str());
          std::string activity_name =
              !annotation_stack.empty()
                  ? std::string(annotation_stack.back().name)
                  : kernel_name;
          ns->set_node_name(activity_name);
          switch (event.type) {
            case CuptiTracerEventType::Kernel: {
              const std::string details = absl::StrFormat(
                  "regs:%u shm:%u grid:%u,%u,%u block:%u,%u,%u",
                  event.kernel_info.registers_per_thread,
                  event.kernel_info.static_shared_memory_usage,
                  event.kernel_info.grid_x, event.kernel_info.grid_y,
                  event.kernel_info.grid_z, event.kernel_info.block_x,
                  event.kernel_info.block_y, event.kernel_info.block_z);
              ns->set_timeline_label(absl::StrCat(kernel_name, " ", details,
                                                  "@@", event.annotation));
              auto nscopy = new NodeExecStats(*ns);
              collector->Save(absl::StrCat(stream_device, "all"), ns);
              collector->Save(absl::StrCat(stream_device, event.stream_id),
                              nscopy);
              break;
            }
            case CuptiTracerEventType::MemcpyH2D:
            case CuptiTracerEventType::MemcpyD2H:
            case CuptiTracerEventType::MemcpyD2D:
            case CuptiTracerEventType::MemcpyP2P: {
              std::string details = absl::StrCat(
                  activity_name, " bytes:", event.memcpy_info.num_bytes);
              if (event.memcpy_info.async) {
                absl::StrAppend(&details, " aync");
              }
              if (event.memcpy_info.destination != event.device_id) {
                absl::StrAppend(&details,
                                " to device:", event.memcpy_info.destination);
              }
              ns->set_timeline_label(std::move(details));
              auto nscopy = new NodeExecStats(*ns);
              collector->Save(memcpy_device, ns);
              collector->Save(
                  absl::StrCat(stream_device, event.stream_id, "<",
                               GetTraceEventTypeName(event.type), ">"),
                  nscopy);
              break;
            }
            default:
              ns->set_timeline_label(activity_name);
              collector->Save(stream_device, ns);
          }
        }
      }
    }

    void Flush(uint64 start_walltime_ns, uint64 start_gpu_ns,
               XPlaneBuilder* device_plane, XPlaneBuilder* host_plane) {
      absl::MutexLock lock(&mutex);

      const uint64 offset_ns = start_walltime_ns - start_gpu_ns;
      for (auto& event : events) {
        bool is_host_event = IsHostEvent(event);
        int64 line_id = is_host_event ? static_cast<int64>(event.thread_id)
                                      : event.stream_id;
        if (line_id == CuptiTracerEvent::kInvalidThreadId ||
            line_id == CuptiTracerEvent::kInvalidStreamId)
          continue;
        auto* plane = is_host_event ? host_plane : device_plane;
        XLineBuilder line = plane->GetOrCreateLine(line_id);
        CreateXEvent(event, offset_ns, plane, &line);
      }
    }

    absl::Mutex mutex;
    std::string stream_device GUARDED_BY(mutex);
    std::string memcpy_device GUARDED_BY(mutex);
    std::string sync_device GUARDED_BY(mutex);
    std::vector<CuptiTracerEvent> events GUARDED_BY(mutex);
    absl::flat_hash_map<uint32, CorrelationInfo> correlation_info
        GUARDED_BY(mutex);
  };
  absl::FixedArray<PerDeviceCollector> per_device_collector_;

  TF_DISALLOW_COPY_AND_ASSIGN(CuptiTraceCollectorImpl);
};

// GpuTracer for GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(CuptiTracer* cupti_tracer, CuptiInterface* cupti_interface)
      : cupti_tracer_(cupti_tracer) {
    VLOG(1) << "GpuTracer created.";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  Status Start() override;
  Status Stop() override;
  Status CollectData(RunMetadata* run_metadata) override;
  Status CollectData(XSpace* space) override;
  profiler::DeviceType GetDeviceType() override {
    return profiler::DeviceType::kGpu;
  }

 private:
  Status DoStart();
  Status DoStop();

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  CuptiTracer* cupti_tracer_;
  CuptiTracerOptions options_;
  StepStats step_stats_;
  std::unique_ptr<CuptiTraceCollectorImpl> cupti_collector_;
};

Status GpuTracer::DoStart() {
  if (!cupti_tracer_->IsAvailable()) {
    return errors::Unavailable("Another profile session running.");
  }

  options_.cbids_selected = {
      // KERNEL
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
      // MEMCPY
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2,
      // GENERIC
      CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize,
  };

  bool use_cupti_activity_api = true;
  ReadBoolFromEnvVar("TF_GPU_CUPTI_USE_ACTIVITY_API", true,
                     &use_cupti_activity_api)
      .IgnoreError();
  options_.enable_event_based_activity = !use_cupti_activity_api;

  bool trace_concurrent_kernels = false;
  ReadBoolFromEnvVar("TF_GPU_CUPTI_FORCE_CONCURRENT_KERNEL", false,
                     &trace_concurrent_kernels)
      .IgnoreError();
  options_.activities_selected.push_back(
      trace_concurrent_kernels ? CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
                               : CUPTI_ACTIVITY_KIND_KERNEL);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY2);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_OVERHEAD);

#if CUDA_VERSION < 10000
  if (!trace_concurrent_kernels) options_.cupti_finalize = true;
#endif

  CuptiTracerCollectorOptions collector_options;
  collector_options.num_gpus = cupti_tracer_->NumGpus();
  uint64 start_gputime_ns = CuptiTracer::GetTimestamp();
  uint64 start_walltime_ns = tensorflow::EnvTime::NowNanos();
  cupti_collector_ = absl::make_unique<CuptiTraceCollectorImpl>(
      collector_options, start_walltime_ns, start_gputime_ns);

  AnnotationStack::Enable(true);
  cupti_tracer_->Enable(options_, cupti_collector_.get());
  return Status::OK();
}

Status GpuTracer::Start() {
  Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return Status::OK();
  } else {
    profiling_state_ = State::kStartedError;
    return status;
  }
}

Status GpuTracer::DoStop() {
  cupti_tracer_->Disable();
  AnnotationStack::Enable(false);
  return Status::OK();
}

Status GpuTracer::Stop() {
  if (profiling_state_ == State::kStartedOk) {
    Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return Status::OK();
}

Status GpuTracer::CollectData(RunMetadata* run_metadata) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(1) << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, xprof failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(1) << "No trace data collected";
      return Status::OK();
    case State::kStoppedOk: {
      // Input run_metadata is shared by profiler interfaces, we need append.
      StepStatsCollector step_stats_collector(&step_stats_);
      if (cupti_collector_) {
        cupti_collector_->Export(&step_stats_collector);
      }
      step_stats_collector.Finalize();
      for (auto& dev_stats : *step_stats_.mutable_dev_stats()) {
        run_metadata->mutable_step_stats()->add_dev_stats()->Swap(&dev_stats);
      }
      return Status::OK();
    }
  }
  return errors::Internal("Invalid profiling state: ", profiling_state_);
}

Status GpuTracer::CollectData(XSpace* space) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(1) << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, xprof failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(1) << "No trace data collected";
      return Status::OK();
    case State::kStoppedOk: {
      if (cupti_collector_) {
        cupti_collector_->Export(space);
      }
      return Status::OK();
    }
  }
  return errors::Internal("Invalid profiling state: ", profiling_state_);
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const profiler::ProfilerOptions& options) {
  if (options.device_type != profiler::DeviceType::kGpu &&
      options.device_type != profiler::DeviceType::kUnspecified)
    return nullptr;
  profiler::CuptiTracer* cupti_tracer =
      profiler::CuptiTracer::GetCuptiTracerSingleton();
  if (!cupti_tracer->IsAvailable()) {
    return nullptr;
  }
  profiler::CuptiInterface* cupti_interface = profiler::GetCuptiInterface();
  return absl::make_unique<profiler::GpuTracer>(cupti_tracer, cupti_interface);
}

auto register_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
