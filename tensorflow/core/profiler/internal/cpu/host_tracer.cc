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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/cpu/host_tracer_utils.h"
#include "tensorflow/core/profiler/internal/cpu/traceme_recorder.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// Controls TraceMeRecorder and converts TraceMeRecorder::Events into XEvents.
//
// Thread-safety: This class is go/thread-compatible.
class HostTracer : public ProfilerInterface {
 public:
  explicit HostTracer(int host_trace_level);
  ~HostTracer() override;

  Status Start() override;

  Status Stop() override;

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

Status HostTracer::CollectData(XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from HostTracer.";
  if (recording_) {
    return errors::Internal("TraceMeRecorder not stopped");
  }
  if (events_.empty()) {
    return Status::OK();
  }
  XPlane* plane = FindOrAddMutablePlaneWithName(space, kHostThreadsPlaneName);
  ConvertCompleteEventsToXPlane(start_timestamp_ns_, std::exchange(events_, {}),
                                plane);
  return Status::OK();
}

}  // namespace

std::unique_ptr<ProfilerInterface> CreateHostTracer(
    const HostTracerOptions& options) {
  if (options.trace_level == 0) return nullptr;
  return absl::make_unique<HostTracer>(options.trace_level);
}

}  // namespace profiler
}  // namespace tensorflow
