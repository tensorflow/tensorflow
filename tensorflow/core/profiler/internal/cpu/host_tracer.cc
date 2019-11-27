/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/profiler/internal/parse_annotation.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/internal/traceme_recorder.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {
namespace cpu {
namespace {

// Controls TraceMeRecorder and converts TraceMeRecorder::Events into
// RunMetadata messages.
//
// Thread-safety: This class is go/thread-compatible.
class HostTracer : public ProfilerInterface {
 public:
  explicit HostTracer(int host_trace_level);
  ~HostTracer() override;

  // Starts recording TraceMes.
  Status Start() override;

  // Stops recording TraceMes.
  Status Stop() override;

  // Populates user traces and thread names in response.
  // The user traces and thread names are in no particular order.
  Status CollectData(RunMetadata* run_metadata) override;

  Status CollectData(XSpace* space) override;

  DeviceType GetDeviceType() override { return DeviceType::kCpu; }

 private:
  // Combine events created by TraceMe::ActivityStart and TraceMe::ActivityEnd,
  // which can be paired up by their activity_id.
  void MakeCompleteEvents();

  // Level of host tracing.
  const int host_trace_level_;

  // True if currently recording.
  bool recording_ = false;

  // Timestamp at the start of tracing.
  uint64 start_timestamp_ns_ = 0;

  // Container of all traced events.
  TraceMeRecorder::Events events_;
};

HostTracer::HostTracer(int host_trace_level)
    : host_trace_level_(host_trace_level) {}

HostTracer::~HostTracer() { Stop().IgnoreError(); }

Status HostTracer::Start() {
  if (recording_) {
    return Status(error::INTERNAL, "TraceMeRecorder already started");
  }
  recording_ = TraceMeRecorder::Start(host_trace_level_);
  if (!recording_) {
    return Status(error::INTERNAL, "Failed to start TraceMeRecorder");
  }
  start_timestamp_ns_ = EnvTime::NowNanos();
  return Status::OK();
}

Status HostTracer::Stop() {
  if (!recording_) {
    return Status(error::INTERNAL, "TraceMeRecorder not started");
  }
  events_ = TraceMeRecorder::Stop();
  recording_ = false;
  return Status::OK();
}

void HostTracer::MakeCompleteEvents() {
  // Track events create by ActivityStart and copy their data to events created
  // by ActivityEnd. TraceME records events in its destructor, so this results
  // in complete events sorted by their end_time in the thread they ended.
  // Within the same thread, the record created by ActivityStart must appear
  // before the record created by ActivityEnd. Cross-thread events must be
  // processed in a separate pass. A single map can be used because the
  // activity_id is globally unique.
  absl::flat_hash_map<uint64, TraceMeRecorder::Event*> start_events;
  std::vector<TraceMeRecorder::Event*> end_events;
  for (auto& thread : events_) {
    for (auto& event : thread.events) {
      if (event.start_time && !event.end_time) {  // ActivityStart
        start_events.emplace(event.activity_id, &event);
      } else if (!event.start_time && event.end_time) {  // ActivityEnd
        auto iter = start_events.find(event.activity_id);
        if (iter != start_events.end()) {  // same thread
          auto* start_event = iter->second;
          event.name = std::move(start_event->name);
          event.start_time = start_event->start_time;
          start_events.erase(iter);
        } else {  // cross-thread
          end_events.push_back(&event);
        }
      }
    }
  }
  for (auto* event : end_events) {  // cross-thread
    auto iter = start_events.find(event->activity_id);
    if (iter != start_events.end()) {
      auto* start_event = iter->second;
      event->name = std::move(start_event->name);
      event->start_time = start_event->start_time;
      start_events.erase(iter);
    }
  }
}

Status HostTracer::CollectData(RunMetadata* run_metadata) {
  if (recording_) {
    return errors::Internal("TraceMeRecorder not stopped");
  }
  MakeCompleteEvents();
  StepStatsCollector step_stats_collector(run_metadata->mutable_step_stats());

  constexpr char kUserMetadataMarker = '#';
  const string cpu_name = "/host:CPU";
  for (auto& thread : events_) {
    step_stats_collector.SaveThreadName(cpu_name, thread.thread.tid,
                                        thread.thread.name);
    for (auto& event : thread.events) {
      if (event.start_time && event.end_time) {
        NodeExecStats* ns = new NodeExecStats;
        if (event.name.back() != kUserMetadataMarker) {
          ns->set_node_name(std::move(event.name));
        } else {
          // Expect the format will be "<name>#<metadata>#"
          std::vector<absl::string_view> parts =
              absl::StrSplit(event.name, kUserMetadataMarker);
          if (parts.size() >= 2) {
            ns->set_node_name(string(parts[0]));
            ns->set_timeline_label(string(parts[1]));
          } else {
            ns->set_node_name(std::move(event.name));
          }
        }
        ns->set_all_start_micros(event.start_time / EnvTime::kMicrosToNanos);
        ns->set_all_end_rel_micros((event.end_time - event.start_time) /
                                   EnvTime::kMicrosToNanos);
        ns->set_thread_id(thread.thread.tid);
        step_stats_collector.Save(cpu_name, ns);
      }
    }
  }
  events_.clear();
  step_stats_collector.Finalize();
  return Status::OK();
}

Status HostTracer::CollectData(XSpace* space) {
  if (recording_) {
    return errors::Internal("TraceMeRecorder not stopped");
  }
  MakeCompleteEvents();
  XPlaneBuilder xplane(space->add_planes());
  xplane.SetName("Host Threads");
  absl::flat_hash_map<string, XEventMetadata*> xevent_metadata_by_name;
  absl::flat_hash_map<string, XStatMetadata*> xstat_metadata_by_name;
  for (const auto& thread : events_) {
    XLineBuilder xline = xplane.AddLine();
    xline.SetId(thread.thread.tid);
    xline.SetName(thread.thread.name);
    xline.SetTimestampNs(start_timestamp_ns_);
    xline.ReserveEvents(thread.events.size());
    for (const auto& event : thread.events) {
      if (event.start_time && event.end_time) {
        Annotation annotation = ParseAnnotation(event.name);
        XEventMetadata*& xevent_metadata =
            xevent_metadata_by_name[annotation.name];
        if (xevent_metadata == nullptr) {
          xevent_metadata =
              xplane.GetOrCreateEventMetadata(xevent_metadata_by_name.size());
          xevent_metadata->set_name(string(annotation.name));
        }
        XEventBuilder xevent = xline.AddEvent(*xevent_metadata);
        xevent.SetTimestampNs(event.start_time);
        xevent.SetEndTimestampNs(event.end_time);
        xevent.ReserveStats(annotation.metadata.size());
        for (const auto& metadata : annotation.metadata) {
          XStatMetadata*& xstat_metadata = xstat_metadata_by_name[metadata.key];
          if (xstat_metadata == nullptr) {
            xstat_metadata =
                xplane.GetOrCreateStatMetadata(xstat_metadata_by_name.size());
            xstat_metadata->set_name(string(metadata.key));
          }
          xevent.ParseAndAddStatValue(*xstat_metadata, metadata.value);
        }
      }
    }
  }
  events_.clear();
  return Status::OK();
}

}  // namespace

// Not in anonymous namespace for testing purposes.
std::unique_ptr<ProfilerInterface> CreateHostTracer(
    const profiler::ProfilerOptions& options) {
  if (options.host_tracer_level == 0) return nullptr;
  return absl::make_unique<HostTracer>(options.host_tracer_level);
}

auto register_host_tracer_factory = [] {
  bool enable;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_OSS_CPU_PROFILER", true, &enable));
  if (enable) {
    RegisterProfilerFactory(&CreateHostTracer);
  }
  return 0;
}();

}  // namespace cpu
}  // namespace profiler
}  // namespace tensorflow
