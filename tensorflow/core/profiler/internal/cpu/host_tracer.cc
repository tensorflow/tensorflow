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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/cpu/host_tracer_utils.h"
#include "tensorflow/core/profiler/internal/cpu/traceme_recorder.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/protobuf/config.pb.h"

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
    return errors::Internal("TraceMeRecorder already started");
  }

  // All TraceMe captured should have a timestamp greater or equal to
  // start_timestamp_ns_ to prevent timestamp underflow in XPlane.
  // Therefore this have to be done before TraceMeRecorder::Start.
  start_timestamp_ns_ = GetCurrentTimeNanos();
  recording_ = TraceMeRecorder::Start(host_trace_level_);
  if (!recording_) {
    return errors::Internal("Failed to start TraceMeRecorder");
  }
  return Status::OK();
}

Status HostTracer::Stop() {
  if (!recording_) {
    return errors::Internal("TraceMeRecorder not started");
  }
  events_ = TraceMeRecorder::Stop();
  recording_ = false;
  return Status::OK();
}

Status HostTracer::CollectData(RunMetadata* run_metadata) {
  if (recording_) {
    return errors::Internal("TraceMeRecorder not stopped");
  }

  StepStats* step_stats = run_metadata->mutable_step_stats();
  DeviceStepStats* dev_stats = step_stats->add_dev_stats();
  dev_stats->set_device("/host:CPU");
  auto* thread_names = dev_stats->mutable_thread_names();

  constexpr char kUserMetadataMarker = '#';
  for (TraceMeRecorder::ThreadEvents& thread : events_) {
    thread_names->insert({thread.thread.tid, thread.thread.name});
    while (!thread.events.empty()) {
      auto event = std::move(thread.events.front());
      thread.events.pop_front();
      if (event.start_time && event.end_time) {
        NodeExecStats* ns = dev_stats->add_node_stats();
        if (event.name.back() != kUserMetadataMarker) {
          ns->set_node_name(std::move(event.name));
        } else {
          // Expect the format will be "<name>#<metadata>#"
          std::vector<absl::string_view> parts =
              absl::StrSplit(event.name, kUserMetadataMarker);
          if (parts.size() >= 2) {
            ns->set_node_name(std::string(parts[0]));
            ns->set_timeline_label(std::string(parts[1]));
          } else {
            ns->set_node_name(std::move(event.name));
          }
        }
        ns->set_all_start_micros(NanosToMicros(event.start_time));
        ns->set_all_end_rel_micros(
            NanosToMicros(event.end_time - event.start_time));
        ns->set_thread_id(thread.thread.tid);
      }
    }
  }
  events_.clear();
  return Status::OK();
}

Status HostTracer::CollectData(XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from HostTracer.";
  if (recording_) {
    return errors::Internal("TraceMeRecorder not stopped");
  }
  XPlane* plane = FindOrAddMutablePlaneWithName(space, kHostThreadsPlaneName);
  ConvertCompleteEventsToXPlane(start_timestamp_ns_, std::exchange(events_, {}),
                                plane);
  return Status::OK();
}

}  // namespace

// Not in anonymous namespace for testing purposes.
std::unique_ptr<ProfilerInterface> CreateHostTracer(
    const ProfileOptions& options) {
  if (options.host_tracer_level() == 0) return nullptr;
  return absl::make_unique<HostTracer>(options.host_tracer_level());
}

auto register_host_tracer_factory = [] {
  RegisterProfilerFactory(&CreateHostTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow
