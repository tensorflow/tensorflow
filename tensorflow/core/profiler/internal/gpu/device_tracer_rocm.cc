/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/internal/cpu/annotation_stack.h"
#include "tensorflow/core/profiler/internal/gpu/rocm_tracer.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/utils/parse_annotation.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {

namespace {
// Set the all XLines of specified XPlane to starting walltime.
// Events time in both host and device planes are CUTPI timestamps.
// We set initial RocmTracer timestamp as start time for all lines to reflect
// this fact. Eventually we change line start time to corresponding
// start_walltime_ns to normalize with CPU wall time.
static void NormalizeTimeStamps(XPlaneBuilder* plane,
                                uint64_t start_walltime_ns) {
  plane->ForEachLine([&](tensorflow::profiler::XLineBuilder line) {
    line.SetTimestampNs(start_walltime_ns);
  });
}

void GetDeviceCapabilities(int32_t device_ordinal,
                           XPlaneBuilder* device_plane) {
  // TODO(rocm)
}

bool IsHostEvent(const RocmTracerEvent& event) {
  // TODO(rocm)
  // Classify all events as GPU events for now
  return false;
}

std::string GetDeviceXLineName(
    int64_t stream_id, absl::flat_hash_set<RocmTracerEventType>& event_types) {
  std::string line_name = absl::StrCat("Stream #", stream_id);
  event_types.erase(RocmTracerEventType::Unsupported);
  if (event_types.empty()) return line_name;
  std::vector<const char*> type_names;
  for (const auto event_type : event_types) {
    type_names.emplace_back(GetRocmTracerEventTypeName(event_type));
  }
  return absl::StrCat(line_name, "(", absl::StrJoin(type_names, ","), ")");
}

}  // namespace

class RocmTraceCollectorImpl : public profiler::RocmTraceCollector {
 public:
  RocmTraceCollectorImpl(const RocmTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : RocmTraceCollector(options),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gputime_ns_(start_gputime_ns),
        next_logical_device_id_(0),
        per_device_collector_(options.num_gpus) {
    // in the physical -> logical device_id map, add an explicit entry for
    // RocmTracerEvent::kInvalidDeviceId -> RocmTracerEvent::kInvalidDeviceId
    // event with this device_id are events for which we were not able to
    // determine the correct device_id via the API+Activity callbacks
    // we will special case such events in the Flush routine
    device_id_map_[RocmTracerEvent::kInvalidDeviceId] =
        RocmTracerEvent::kInvalidDeviceId;
  }

  void AddEvent(RocmTracerEvent&& event) override {
    mutex_lock lock(aggregated_events_mutex_);

    if (event.source == RocmTracerEventSource::ApiCallback) {
      if (num_callback_events_ > options_.max_callback_api_events) {
        OnEventsDropped("max callback event capacity reached",
                        event.correlation_id);
        DumpRocmTracerEvent(event, 0, 0);
        return;
      }
      num_callback_events_++;
    }
    if (event.source == RocmTracerEventSource::Activity) {
      if (num_activity_events_ > options_.max_activity_api_events) {
        OnEventsDropped("max activity event capacity reached",
                        event.correlation_id);
        DumpRocmTracerEvent(event, 0, 0);
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
              // Use the start/stop time from the HCC_OPS domain
              // unless this is one of those events for which we do not
              // receive any HCC activity record callback
              if (IsEventTypeWithoutHCCActivityRecordCallback(event.type)) {
                iter->second.start_time_ns = event.start_time_ns;
                iter->second.end_time_ns = event.end_time_ns;
              }
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
              // Use the annotation from the HIP_API domain
              // iter->second.annotation = event.annotation;
              break;
          }
          break;
      }
    } else {
      switch (event.source) {
        case RocmTracerEventSource::ApiCallback:
          aggregated_events_.emplace(event.correlation_id, std::move(event));
          break;
        case RocmTracerEventSource::Activity:
          // you would think that this cannot happen, but it does
          // This is primarily because the call "roctracer_flush_activity" does
          // not work as it should. Imagine a sequence where we enable/disable
          // tracing more than once in a single TF session.
          // If the "flush" that happens during disable, does not flush out all
          // the activity records, then they will show up during the subsequent
          // call to enable, and we will end up here!
          OnEventsDropped(
              "Activity event encountered before a corresponding API event",
              event.correlation_id);
          DumpRocmTracerEvent(event, 0, 0);
          break;
      }
    }
  }

  void OnEventsDropped(const std::string& reason,
                       uint32_t correlation_id) override {
    LOG(INFO) << "RocmTracerEvent dropped (correlation_id=" << correlation_id
              << ",) : " << reason << ".";
  }

  void Flush() override {
    mutex_lock lock(aggregated_events_mutex_);

    VLOG(3) << "RocmTraceCollector collected " << num_callback_events_
            << " callback events, " << num_activity_events_
            << " activity events, and aggregated them into "
            << aggregated_events_.size() << " events.";

    for (auto& iter : aggregated_events_) {
      auto& event = iter.second;

      // For some hip API events, we never get a corresponding HCC
      // activity record callback and hence we currently do not have a way
      // of associating a valid device_id and stream_id with those events.
      // For such events, explcitly set those id sto 0 for now
      if (IsEventTypeWithoutHCCActivityRecordCallback(event.type)) {
        DumpRocmTracerEvent(event, 0, 0);
        if (event.device_id == RocmTracerEvent::kInvalidDeviceId) {
          VLOG(3) << "Explicitly setting device_id to 0 for "
                     "event with correlation_id="
                  << event.correlation_id << ",";
          event.device_id = 0;
        } else {
          VLOG(3) << "Unexpectedly found a non-default "
                     "device_id for event with correlation_id="
                  << event.correlation_id << ",";
        }
        if (event.stream_id == RocmTracerEvent::kInvalidStreamId) {
          VLOG(3) << "Explicitly setting stream_id to 0 for "
                     "event with correlation_id="
                  << event.correlation_id << ",";
          event.stream_id = 0;
        } else {
          VLOG(3) << "Unexpectedly found a non-default "
                     "stream_id for event with correlation_id="
                  << event.correlation_id << ",";
        }
      }

      // determine the logical device id
      uint32_t physical_id = event.device_id;
      uint32_t logical_id = options_.num_gpus;
      auto kv_pair = device_id_map_.find(physical_id);
      if (kv_pair == device_id_map_.end()) {
        logical_id = next_logical_device_id_++;
        VLOG(3) << "Mapping physical device id " << physical_id
                << " to logical device id " << logical_id;
        device_id_map_[physical_id] = logical_id;
      } else {
        logical_id = kv_pair->second;
      }
      event.device_id = logical_id;

      if (event.device_id >= options_.num_gpus) {
        OnEventsDropped("logical device id >= num gpus", event.correlation_id);
        DumpRocmTracerEvent(event, 0, 0);
        continue;
      }

      if (event.stream_id == RocmTracerEvent::kInvalidStreamId) {
        OnEventsDropped("invalid stream id", event.correlation_id);
        DumpRocmTracerEvent(event, 0, 0);
        continue;
      }

      per_device_collector_[logical_id].AddEvent(event);
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

  void Export(XSpace* space) {
    uint64_t end_gputime_ns = RocmTracer::GetTimestamp();
    XPlaneBuilder host_plane(
        FindOrAddMutablePlaneWithName(space, kRoctracerApiPlaneName));
    for (int i = 0; i < options_.num_gpus; ++i) {
      std::string name = GpuPlaneName(i);
      XPlaneBuilder device_plane(FindOrAddMutablePlaneWithName(space, name));
      device_plane.SetId(i);
      per_device_collector_[i].Export(start_walltime_ns_, start_gputime_ns_,
                                      end_gputime_ns, &device_plane,
                                      &host_plane);
      GetDeviceCapabilities(i, &device_plane);
      NormalizeTimeStamps(&device_plane, start_walltime_ns_);
    }
    NormalizeTimeStamps(&host_plane, start_walltime_ns_);
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64_t start_walltime_ns_;
  uint64_t start_gputime_ns_;

  mutex aggregated_events_mutex_;
  absl::flat_hash_map<uint32_t, RocmTracerEvent> aggregated_events_
      TF_GUARDED_BY(aggregated_events_mutex_);

  // We need to create a map of
  //  event.device_id -> index into per_device_collector_ array
  // The event.device_id returned by the RocmTracer is the physical
  // device_id and not the logical device_id. Say for example we are
  // running on a node with 8 GPUs. The expected physical device_id(s)
  // for those 8 GPUs would be 0,1,2,3,4,5,6,7. On such a node, if we
  // run a test with HIP_VISIBLE_DEVICES=5, then "options.num_gpus_ == 1",
  // but the event.device_id field will have 5 in it!
  // So the event.device_id can be thought of as the physical device id
  // and the index can be thought of as the logical device id.
  // We cannot determine the actual phsyical device id logical device id
  // mapping here, so we determine it empirically
  std::map<uint32_t, uint32_t> device_id_map_;
  uint32_t next_logical_device_id_;

  bool IsEventTypeWithoutHCCActivityRecordCallback(RocmTracerEventType type) {
    switch (type) {
      case RocmTracerEventType::MemoryAlloc:
        return true;
        break;
      default:
        break;
    }
    return false;
  }

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

    void Export(int32_t device_ordinal, uint64_t start_walltime_ns,
                uint64_t start_gputime_ns, StepStats* step_stats) {
      mutex_lock lock(events_mutex);
      absl::flat_hash_map<
          std::pair<uint64_t /*stream_id*/, RocmTracerEventType>,
          DeviceStepStats*>
          per_stream_dev_stats;

      DeviceStepStats* generic_stream_dev_stats = nullptr;
      DeviceStepStats* all_streams_dev_stats = nullptr;
      DeviceStepStats* memcpy_dev_stats = nullptr;
      DeviceStepStats* sync_dev_stats = nullptr;

      for (const RocmTracerEvent& event : events) {
        DumpRocmTracerEvent(event, start_walltime_ns, start_gputime_ns);

        std::unique_ptr<NodeExecStats> ns(new NodeExecStats);

        ns->set_all_start_micros(
            (start_walltime_ns + (event.start_time_ns - start_gputime_ns)) /
            1000);
        ns->set_op_start_rel_micros(0);
        uint64_t elapsed_ns = event.end_time_ns - event.start_time_ns;
        ns->set_op_end_rel_micros(
            tensorflow::profiler::NanosToMicros(elapsed_ns));
        ns->set_all_end_rel_micros(
            tensorflow::profiler::NanosToMicros(elapsed_ns));

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
            all_streams_dev_stats->add_node_stats()->Swap(ns.release());
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
            memcpy_dev_stats->add_node_stats()->Swap(ns.release());
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
            sync_dev_stats->add_node_stats()->Swap(ns.release());
          } break;
          case RocmTracerEventType::Generic: {
            std::string details = event.name;
            ns->set_timeline_label(std::move(details));

            if (generic_stream_dev_stats == nullptr) {
              generic_stream_dev_stats = step_stats->add_dev_stats();
              generic_stream_dev_stats->set_device(
                  absl::StrCat("/device:GPU:", device_ordinal, "/stream:"));
            }
            generic_stream_dev_stats->add_node_stats()->Swap(ns.release());
          } break;
          default:
            DCHECK(false);
            break;
        }
      }
      events.clear();
    }

    void CreateXEvent(const RocmTracerEvent& event, XPlaneBuilder* plane,
                      uint64_t start_gpu_ns, uint64_t end_gpu_ns,
                      XLineBuilder* line) {
      if (event.start_time_ns < start_gpu_ns ||
          event.end_time_ns > end_gpu_ns ||
          event.start_time_ns > event.end_time_ns) {
        VLOG(2) << "events have abnormal timestamps:" << event.name
                << " start time(ns): " << event.start_time_ns
                << " end time(ns): " << event.end_time_ns;
        return;
      }
      std::string kernel_name = port::MaybeAbiDemangle(event.name.c_str());
      if (kernel_name.empty()) {
        kernel_name = GetRocmTracerEventTypeName(event.type);
      }
      XEventMetadata* event_metadata =
          plane->GetOrCreateEventMetadata(std::move(kernel_name));
      XEventBuilder xevent = line->AddEvent(*event_metadata);
      xevent.SetTimestampNs(event.start_time_ns);
      xevent.SetEndTimestampNs(event.end_time_ns);
      if (event.correlation_id != RocmTracerEvent::kInvalidCorrelationId) {
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kCorrelationId)),
                            event.correlation_id);
      }
      if (!event.annotation.empty()) {
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kKernelAnnotation)),
                            event.annotation);
      }
      switch (event.type) {
        case RocmTracerEventType::Kernel: {
          const std::string kernel_details = absl::StrFormat(
              "regs:%u shm:%u grid:%u,%u,%u block:%u,%u,%u",
              event.kernel_info.registers_per_thread,
              event.kernel_info.static_shared_memory_usage,
              event.kernel_info.grid_x, event.kernel_info.grid_y,
              event.kernel_info.grid_z, event.kernel_info.block_x,
              event.kernel_info.block_y, event.kernel_info.block_z);
          xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kKernelDetails)),
                              kernel_details);
        } break;

        case RocmTracerEventType::MemcpyD2H:
        case RocmTracerEventType::MemcpyH2D:
        case RocmTracerEventType::MemcpyD2D:
        case RocmTracerEventType::MemcpyP2P:
        case RocmTracerEventType::MemcpyOther: {
          const auto& memcpy_info = event.memcpy_info;
          std::string memcpy_details =
              absl::StrFormat("size:%u dest:%u async:%u", memcpy_info.num_bytes,
                              memcpy_info.destination, memcpy_info.async);
          xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kMemcpyDetails)),
                              memcpy_details);
        } break;
        case RocmTracerEventType::MemoryAlloc: {
          std::string memalloc_details =
              absl::StrFormat("num_bytes:%u", event.memalloc_info.num_bytes);
          xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kMemallocDetails)),
                              memalloc_details);
        } break;
        case RocmTracerEventType::StreamSynchronize: {
          // TODO(rocm)
          // Don't yet know what to do here
        } break;
        case RocmTracerEventType::Generic: {
          // TODO(rocm)
          // Don't yet know what to do here
        } break;
        default:
          DCHECK(false);
          break;
      }

      std::vector<Annotation> annotation_stack =
          ParseAnnotationStack(event.annotation);
      // If multiple metadata have the same key name, show the values from the
      // top of the stack (innermost annotation). Concatenate the values from
      // "hlo_op".
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
      // TODO(profiler): we should get rid of kLevel0, it is based on the
      // assumption that those op-related ScopedAnnotation are at the very TOP
      // level.
      if (!annotation_stack.empty()) {
        xevent.AddStatValue(
            *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kLevel0)),
            annotation_stack.begin()->name);
      }
    }

    void Export(uint64_t start_walltime_ns, uint64_t start_gputime_ns,
                uint64_t end_gputime_ns, XPlaneBuilder* device_plane,
                XPlaneBuilder* host_plane) {
      mutex_lock lock(events_mutex);
      // Tracking event types per line.
      absl::flat_hash_map<int64, absl::flat_hash_set<RocmTracerEventType>>
          events_types_per_line;
      for (const RocmTracerEvent& event : events) {
        DumpRocmTracerEvent(event, start_walltime_ns, start_gputime_ns);
        bool is_host_event = IsHostEvent(event);
        int64_t line_id = is_host_event ? static_cast<int64>(event.thread_id)
                                        : event.stream_id;
        if (line_id == RocmTracerEvent::kInvalidThreadId ||
            line_id == RocmTracerEvent::kInvalidStreamId)
          continue;
        auto* plane = is_host_event ? host_plane : device_plane;
        XLineBuilder line = plane->GetOrCreateLine(line_id);
        line.SetTimestampNs(start_gputime_ns);
        CreateXEvent(event, plane, start_gputime_ns, end_gputime_ns, &line);
        events_types_per_line[line_id].emplace(event.type);
      }
      device_plane->ForEachLine([&](tensorflow::profiler::XLineBuilder line) {
        line.SetName(
            GetDeviceXLineName(line.Id(), events_types_per_line[line.Id()]));
      });
      events.clear();
    }

    mutex events_mutex;
    std::vector<RocmTracerEvent> events TF_GUARDED_BY(events_mutex);
  };

  absl::FixedArray<PerDeviceCollector> per_device_collector_;
};

// GpuTracer for ROCm GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(RocmTracer* rocm_tracer) : rocm_tracer_(rocm_tracer) {
    LOG(INFO) << "GpuTracer created.";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  Status Start() override;
  Status Stop() override;
  Status CollectData(RunMetadata* run_metadata) override;
  Status CollectData(XSpace* space) override;

 private:
  Status DoStart();
  Status DoStop();
  Status DoCollectData(StepStats* step_stats);
  Status DoCollectData(XSpace* space);

  RocmTracerOptions GetRocmTracerOptions();

  RocmTraceCollectorOptions GetRocmTraceCollectorOptions(uint32_t num_gpus);

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
  std::vector<uint32_t> empty_vec;

  // clang formatting does not preserve one entry per line
  // clang-format off
  std::vector<uint32_t> hip_api_domain_ops{
      HIP_API_ID_hipExtModuleLaunchKernel,
      HIP_API_ID_hipFree,
      HIP_API_ID_hipHccModuleLaunchKernel,
      HIP_API_ID_hipLaunchKernel,
      HIP_API_ID_hipMalloc,
      HIP_API_ID_hipMemcpyAsync,
      HIP_API_ID_hipMemcpyDtoD,
      HIP_API_ID_hipMemcpyDtoDAsync,
      HIP_API_ID_hipMemcpyDtoH,
      HIP_API_ID_hipMemcpyDtoHAsync,
      HIP_API_ID_hipMemcpyHtoD,
      HIP_API_ID_hipMemcpyHtoDAsync,
      HIP_API_ID_hipMemsetD32,
      HIP_API_ID_hipMemsetD32Async,
      HIP_API_ID_hipMemsetD8,
      HIP_API_ID_hipMemsetD8Async,
      HIP_API_ID_hipModuleLaunchKernel,
      HIP_API_ID_hipStreamSynchronize,
  };
  // clang-format on

  options.api_callbacks.emplace(ACTIVITY_DOMAIN_HIP_API, hip_api_domain_ops);
  // options.api_callbacks.emplace(ACTIVITY_DOMAIN_HIP_API, empty_vec);

  // options.activity_tracing.emplace(ACTIVITY_DOMAIN_HIP_API,
  // hip_api_domain_ops);
  options.activity_tracing.emplace(ACTIVITY_DOMAIN_HIP_API, empty_vec);
  options.activity_tracing.emplace(ACTIVITY_DOMAIN_HCC_OPS, empty_vec);

  return options;
}

RocmTraceCollectorOptions GpuTracer::GetRocmTraceCollectorOptions(
    uint32_t num_gpus) {
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
  uint64_t start_gputime_ns = RocmTracer::GetTimestamp();
  uint64_t start_walltime_ns = tensorflow::EnvTime::NowNanos();
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
      VLOG(3) << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(3) << "No trace data collected";
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
      VLOG(3) << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(3) << "No trace data collected";
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
    const ProfileOptions& options) {
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED)
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
