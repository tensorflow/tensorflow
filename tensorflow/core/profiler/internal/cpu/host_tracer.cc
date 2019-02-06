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
#include "tensorflow/core/profiler/internal/cpu/host_tracer.h"

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/platform/env_time.h"

namespace tensorflow {
namespace profiler {
namespace cpu {

/* static */ std::unique_ptr<HostTracer> HostTracer::Create(
    int host_trace_level) {
  return absl::WrapUnique(new HostTracer(host_trace_level));
}
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

constexpr char kUserMetadataMarker = '#';

Status HostTracer::CollectData(RunMetadata* run_metadata) {
  auto step_stats_collector =
      absl::make_unique<StepStatsCollector>(run_metadata->mutable_step_stats());
  return CollectDataToCollector(step_stats_collector.get());
}

Status HostTracer::CollectDataToCollector(
    StepStatsCollector* step_stats_collector) {
  if (events_.empty() && recording_) {
    events_ = TraceMeRecorder::Collect();
  }
  // Pair up start and end events, and add complete events to trace_entries.
  absl::flat_hash_map<uint64, uint64> end_times;
  for (const auto& thread : events_) {
    for (const auto& event : thread.events) {
      if (event.end_time && !event.start_time) {
        end_times.emplace(event.activity_id, event.end_time);
      }
    }
  }

  const string cpu_name = "/host:CPU";
  for (auto& thread : events_) {
    for (auto& event : thread.events) {
      if (!event.end_time) {
        auto it = end_times.find(event.activity_id);
        if (it != end_times.end()) event.end_time = it->second;
      }
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
        // TODO(fishx): Add thread name to RunMetadata
        step_stats_collector->Save(cpu_name, ns);
      }
    }
  }
  events_.clear();
  step_stats_collector->Finalize();
  return Status::OK();
}

}  // namespace cpu
}  // namespace profiler
}  // namespace tensorflow
