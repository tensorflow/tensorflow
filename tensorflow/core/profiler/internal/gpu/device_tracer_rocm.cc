/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_ROCM

#include <stdlib.h>

#include <memory>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/internal/annotation_stack.h"
#include "tensorflow/core/profiler/internal/gpu/rocm_tracer.h"
#include "tensorflow/core/profiler/internal/parse_annotation.h"
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {

class RocmTraceCollectorImpl : public profiler::RocmTraceCollector {
 public:
  RocmTraceCollectorImpl(const RocmTraceCollectorOptions& options,
                         uint64 start_walltime_ns, uint64 start_gputime_ns)
      : RocmTraceCollector(options),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gputime_ns_(start_gputime_ns),
        per_device_collector_(options.num_gpus) {}

  void AddEvent(RocmTracerEvent&& event) override {
    mutex_lock lock(aggregated_events_mutex_);

    // TODO(rocm):
    // hard-code the device_id to 0 for now
    if (event.device_id > options_.num_gpus) event.device_id = 0;

    if (event.source == RocmTracerEventSource::ApiCallback) {
      if (num_callback_events_ > options_.max_callback_api_events) {
        OnEventsDropped("max callback event capacity reached", 1);
        return;
      }
      num_callback_events_++;
    }
    if (event.source == RocmTracerEventSource::Activity) {
      if (num_activity_events_ > options_.max_activity_api_events) {
        OnEventsDropped("max activity event capacity reached", 1);
        return;
      }
      num_activity_events_++;
    }

    auto iter = aggregated_events_.find(event.correlation_id);
    if (iter != aggregated_events_.end()) {
      // event with this correlation id already present
      // agrregate this event with the existing one
      switch (event.domain) {
        case RocmTracerEventDomain::HIP_API:
          switch (event.source) {
            case RocmTracerEventSource::ApiCallback:
              break;
            case RocmTracerEventSource::Activity:
              iter->second.start_time_ns = event.start_time_ns;
              iter->second.end_time_ns = event.end_time_ns;
              iter->second.annotation = event.annotation;
              break;
          }
          break;
        case RocmTracerEventDomain::HCC_OPS:
          switch (event.source) {
            case RocmTracerEventSource::ApiCallback:
              break;
            case RocmTracerEventSource::Activity:
              iter->second.device_id = event.device_id;
              iter->second.stream_id = event.stream_id;
              iter->second.start_time_ns = event.start_time_ns;
              iter->second.end_time_ns = event.end_time_ns;
              iter->second.annotation = event.annotation;
              break;
          }
          break;
      }
    } else {
      aggregated_events_.emplace(event.correlation_id, std::move(event));
    }
  }

  void OnEventsDropped(const std::string& reason, uint32 num_events) override {
    VLOG(kRocmTracerVlog) << "RocmTracerEvent(s) dropped (" << num_events
                          << ") : " << reason << ".";
  }

  void Flush() override {
    mutex_lock lock(aggregated_events_mutex_);

    VLOG(kRocmTracerVlog) << "RocmTraceCollector collected "
                          << num_callback_events_ << " callback events, "
                          << num_activity_events_
                          << " activity events, and aggregated them into "
                          << aggregated_events_.size() << " events.";

    for (auto& iter : aggregated_events_) {
      auto& event = iter.second;
      if (event.device_id > options_.num_gpus) {
        OnEventsDropped("failed to determine device id", 1);
      } else {
        per_device_collector_[event.device_id].AddEvent(event);
      }
    }
    aggregated_events_.clear();

    for (int i = 0; i < options_.num_gpus; ++i) {
      per_device_collector_[i].SortByStartTime();
    }
  }

  void Export(StepStats* step_stats) {
    for (int i = 0; i < options_.num_gpus; ++i) {
      per_device_collector_[i].Export(i, start_walltime_ns_, start_gputime_ns_,
                                      step_stats);
    }
  }
  void Export(XSpace* space) {}

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64 start_walltime_ns_;
  uint64 start_gputime_ns_;

  mutex aggregated_events_mutex_;
  absl::flat_hash_map<uint32, RocmTracerEvent> aggregated_events_
      GUARDED_BY(aggregated_events_mutex_);

  struct PerDeviceCollector {
    void AddEvent(const RocmTracerEvent& event) {
      mutex_lock lock(events_mutex);
      events.emplace_back(event);
    }

    void SortByStartTime() {
      mutex_lock lock(events_mutex);
      std::sort(
          events.begin(), events.end(),
          [](const RocmTracerEvent& event1, const RocmTracerEvent& event2) {
            return event1.start_time_ns < event2.start_time_ns;
          });
    }

    void Export(int32 device_ordinal, uint64 start_walltime_ns,
                uint64 start_gputime_ns, StepStats* step_stats) {
      mutex_lock lock(events_mutex);
      absl::flat_hash_map<std::pair<uint64 /*stream_id*/, RocmTracerEventType>,
                          DeviceStepStats*>
          per_stream_dev_stats;

      DeviceStepStats* generic_stream_dev_stats = nullptr;
      DeviceStepStats* all_streams_dev_stats = nullptr;
      DeviceStepStats* memcpy_dev_stats = nullptr;
      DeviceStepStats* sync_dev_stats = nullptr;

      for (const RocmTracerEvent& event : events) {
        DumpRocmTracerEvent(event, start_walltime_ns, start_gputime_ns);

        NodeExecStats* ns = new NodeExecStats;

        ns->set_all_start_micros(
            (start_walltime_ns + (event.start_time_ns - start_gputime_ns)) /
            1000);
        ns->set_op_start_rel_micros(0);
        auto elapsed_ns = event.end_time_ns - event.start_time_ns;
        ns->set_op_end_rel_micros(elapsed_ns / 1000);
        ns->set_all_end_rel_micros(elapsed_ns / 1000);

        auto annotation_stack = ParseAnnotationStack(event.annotation);
        std::string kernel_name = port::MaybeAbiDemangle(event.name.c_str());
        std::string activity_name =
            !annotation_stack.empty()
                ? std::string(annotation_stack.back().name)
                : kernel_name;
        ns->set_node_name(activity_name);

        ns->set_thread_id(event.thread_id);

        switch (event.type) {
          case RocmTracerEventType::Kernel: {
            ns->set_timeline_label(absl::StrFormat(
                "%s regs:%u shm:%u grid:%u,%u,%u block:%u,%u,%u@@%s",
                kernel_name, event.kernel_info.registers_per_thread,
                event.kernel_info.static_shared_memory_usage,
                event.kernel_info.grid_x, event.kernel_info.grid_y,
                event.kernel_info.grid_z, event.kernel_info.block_x,
                event.kernel_info.block_y, event.kernel_info.block_z,
                event.annotation));
            DeviceStepStats*& stream_dev_stats =
                per_stream_dev_stats[std::make_pair(event.stream_id,
                                                    event.type)];
            if (stream_dev_stats == nullptr) {
              stream_dev_stats = step_stats->add_dev_stats();
              stream_dev_stats->set_device(absl::StrCat(
                  "/device:GPU:", device_ordinal, "/stream:", event.stream_id,
                  "<", GetRocmTracerEventTypeName(event.type), ">"));
            }
            *stream_dev_stats->add_node_stats() = *ns;
            if (all_streams_dev_stats == nullptr) {
              all_streams_dev_stats = step_stats->add_dev_stats();
              all_streams_dev_stats->set_device(
                  absl::StrCat("/device:GPU:", device_ordinal, "/stream:all"));
            }
            all_streams_dev_stats->add_node_stats()->Swap(ns);
          } break;
          case RocmTracerEventType::MemcpyD2H:
          case RocmTracerEventType::MemcpyH2D:
          case RocmTracerEventType::MemcpyD2D:
          case RocmTracerEventType::MemcpyP2P: {
            std::string details = absl::StrCat(
                event.name, " bytes:", event.memcpy_info.num_bytes);
            if (event.memcpy_info.async) {
              absl::StrAppend(&details, " async");
            }
            if (event.memcpy_info.destination != event.device_id) {
              absl::StrAppend(&details,
                              " to device:", event.memcpy_info.destination);
            }
            ns->set_timeline_label(std::move(details));

            DeviceStepStats*& stream_dev_stats =
                per_stream_dev_stats[std::make_pair(event.stream_id,
                                                    event.type)];
            if (stream_dev_stats == nullptr) {
              stream_dev_stats = step_stats->add_dev_stats();
              stream_dev_stats->set_device(absl::StrCat(
                  "/device:GPU:", device_ordinal, "/stream:", event.stream_id,
                  "<", GetRocmTracerEventTypeName(event.type), ">"));
            }
            *stream_dev_stats->add_node_stats() = *ns;
            if (memcpy_dev_stats == nullptr) {
              memcpy_dev_stats = step_stats->add_dev_stats();
              memcpy_dev_stats->set_device(
                  absl::StrCat("/device:GPU:", device_ordinal, "/memcpy"));
            }
            memcpy_dev_stats->add_node_stats()->Swap(ns);

          } break;
          case RocmTracerEventType::MemoryAlloc: {
            std::string details = absl::StrCat(
                event.name, " bytes:", event.memalloc_info.num_bytes);
            ns->set_timeline_label(std::move(details));

            DeviceStepStats*& stream_dev_stats =
                per_stream_dev_stats[std::make_pair(event.stream_id,
                                                    event.type)];
            if (stream_dev_stats == nullptr) {
              stream_dev_stats = step_stats->add_dev_stats();
              stream_dev_stats->set_device(absl::StrCat(
                  "/device:GPU:", device_ordinal, "/stream:", event.stream_id,
                  "<", GetRocmTracerEventTypeName(event.type), ">"));
            }
            *stream_dev_stats->add_node_stats() = *ns;
          } break;
          case RocmTracerEventType::StreamSynchronize: {
            std::string details = event.name;
            ns->set_timeline_label(std::move(details));

            if (sync_dev_stats == nullptr) {
              sync_dev_stats = step_stats->add_dev_stats();
              sync_dev_stats->set_device(
                  absl::StrCat("/device:GPU:", device_ordinal, "/sync"));
            }
            sync_dev_stats->add_node_stats()->Swap(ns);
          } break;
          case RocmTracerEventType::Generic: {
            std::string details = event.name;
            ns->set_timeline_label(std::move(details));

            if (generic_stream_dev_stats == nullptr) {
              generic_stream_dev_stats = step_stats->add_dev_stats();
              generic_stream_dev_stats->set_device(
                  absl::StrCat("/device:GPU:", device_ordinal, "/stream:"));
            }
            generic_stream_dev_stats->add_node_stats()->Swap(ns);
          } break;
          default:
            DCHECK(false);
            break;
        }
      }
      events.clear();
    }

    mutex events_mutex;
    std::vector<RocmTracerEvent> events GUARDED_BY(events_mutex);
  };

  absl::FixedArray<PerDeviceCollector> per_device_collector_;
};

// GpuTracer for ROCm GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(RocmTracer* rocm_tracer) : rocm_tracer_(rocm_tracer) {
    VLOG(kRocmTracerVlog) << "GpuTracer created.";
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
  Status DoCollectData(StepStats* step_stats);
  Status DoCollectData(XSpace* space);

  RocmTracerOptions GetRocmTracerOptions();

  RocmTraceCollectorOptions GetRocmTraceCollectorOptions(uint32 num_gpus);

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  RocmTracer* rocm_tracer_;
  std::unique_ptr<RocmTraceCollectorImpl> rocm_trace_collector_;
};

RocmTracerOptions GpuTracer::GetRocmTracerOptions() {
  RocmTracerOptions options;

  std::vector<uint32_t> hip_api_domain_ops{
      HIP_API_ID_hipFree,
      HIP_API_ID_hipMalloc,
      HIP_API_ID_hipMemcpyDtoD,
      HIP_API_ID_hipMemcpyDtoDAsync,
      HIP_API_ID_hipMemcpyDtoH,
      HIP_API_ID_hipMemcpyDtoHAsync,
      HIP_API_ID_hipMemcpyHtoD,
      HIP_API_ID_hipMemcpyHtoDAsync,
      HIP_API_ID_hipModuleLaunchKernel,
      HIP_API_ID_hipStreamSynchronize,
  };
  options.api_callbacks.emplace(ACTIVITY_DOMAIN_HIP_API, hip_api_domain_ops);

  std::vector<uint32_t> empty_vec;
  // options.activity_tracing.emplace(ACTIVITY_DOMAIN_HIP_API,
  // hip_api_domain_ops);
  options.activity_tracing.emplace(ACTIVITY_DOMAIN_HIP_API, empty_vec);
  options.activity_tracing.emplace(ACTIVITY_DOMAIN_HCC_OPS, empty_vec);

  return options;
}

RocmTraceCollectorOptions GpuTracer::GetRocmTraceCollectorOptions(
    uint32 num_gpus) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 2 * 1024 * 1024;
  options.max_activity_api_events = 2 * 1024 * 1024;
  options.max_annotation_strings = 1024 * 1024;
  options.num_gpus = num_gpus;
  return options;
}

Status GpuTracer::DoStart() {
  if (!rocm_tracer_->IsAvailable()) {
    return errors::Unavailable("Another profile session running.");
  }

  AnnotationStack::Enable(true);

  RocmTraceCollectorOptions trace_collector_options =
      GetRocmTraceCollectorOptions(rocm_tracer_->NumGpus());
  uint64 start_gputime_ns = RocmTracer::GetTimestamp();
  uint64 start_walltime_ns = tensorflow::EnvTime::NowNanos();
  // VLOG(kRocmTracerVlog) << "CPU Start Time : " << start_walltime_ns / 1000
  // 			 << " , GPU Start Time : " << start_gputime_ns / 1000;
  rocm_trace_collector_ = std::make_unique<RocmTraceCollectorImpl>(
      trace_collector_options, start_walltime_ns, start_gputime_ns);

  RocmTracerOptions tracer_options = GetRocmTracerOptions();
  rocm_tracer_->Enable(tracer_options, rocm_trace_collector_.get());

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
  rocm_tracer_->Disable();
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

Status GpuTracer::DoCollectData(StepStats* step_stats) {
  if (rocm_trace_collector_) rocm_trace_collector_->Export(step_stats);
  return Status::OK();
}

Status GpuTracer::CollectData(RunMetadata* run_metadata) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(kRocmTracerVlog)
          << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(kRocmTracerVlog) << "No trace data collected";
      return Status::OK();
    case State::kStoppedOk: {
      // Input run_metadata is shared by profiler interfaces, we need append.
      StepStats step_stats;
      DoCollectData(&step_stats);
      for (auto& dev_stats : *step_stats.mutable_dev_stats()) {
        run_metadata->mutable_step_stats()->add_dev_stats()->Swap(&dev_stats);
      }
      return Status::OK();
    }
  }
  return errors::Internal("Invalid profiling state: ", profiling_state_);
}

Status GpuTracer::DoCollectData(XSpace* space) {
  if (rocm_trace_collector_) rocm_trace_collector_->Export(space);
  return Status::OK();
}

Status GpuTracer::CollectData(XSpace* space) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(kRocmTracerVlog)
          << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(kRocmTracerVlog) << "No trace data collected";
      return Status::OK();
    case State::kStoppedOk: {
      DoCollectData(space);
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

  profiler::RocmTracer* rocm_tracer =
      profiler::RocmTracer::GetRocmTracerSingleton();
  if (!rocm_tracer->IsAvailable()) return nullptr;

  return absl::make_unique<profiler::GpuTracer>(rocm_tracer);
}

auto register_rocm_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
