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

#include "absl/strings/str_split.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/profiler/internal/cpu/host_tracer_utils.h"
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/internal/traceme_recorder.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {
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

Status HostTracer::CollectData(RunMetadata* run_metadata) {
  if (recording_) {
    return errors::Internal("TraceMeRecorder not stopped");
  }
  MakeCompleteEvents(&events_);
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
  MakeCompleteEvents(&events_);
  XPlane* plane = GetOrCreatePlane(space, kHostThreads);
  plane->set_id(kHostPlaneId);
  ConvertCompleteEventsToXPlane(start_timestamp_ns_, events_, plane);
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

}  // namespace profiler
}  // namespace tensorflow
