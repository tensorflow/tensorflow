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
#include "tensorflow/compiler/xla/backends/profiler/cpu/host_tracer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/profiler/backends/cpu/host_tracer_utils.h"
#include "tensorflow/tsl/profiler/backends/cpu/traceme_recorder.h"
#include "tensorflow/tsl/profiler/lib/profiler_interface.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/utils/time_utils.h"
#include "tensorflow/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/tsl/profiler/utils/xplane_utils.h"

namespace xla {
namespace profiler {
namespace {

// Controls TraceMeRecorder and converts TraceMeRecorder::Events into XEvents.
//
// Thread-safety: This class is go/thread-compatible.
class HostTracer : public tsl::profiler::ProfilerInterface {
 public:
  explicit HostTracer(int host_trace_level);
  ~HostTracer() override;

  tsl::Status Start() override;  // TENSORFLOW_STATUS_OK

  tsl::Status Stop() override;  // TENSORFLOW_STATUS_OK

  tsl::Status CollectData(  // TENSORFLOW_STATUS_OK
      tensorflow::profiler::XSpace* space) override;

 private:
  // Level of host tracing.
  const int host_trace_level_;

  // True if currently recording.
  bool recording_ = false;

  // Timestamp at the start of tracing.
  uint64_t start_timestamp_ns_ = 0;

  // Container of all traced events.
  tsl::profiler::TraceMeRecorder::Events events_;
};

HostTracer::HostTracer(int host_trace_level)
    : host_trace_level_(host_trace_level) {}

HostTracer::~HostTracer() { Stop().IgnoreError(); }  // NOLINT

tsl::Status HostTracer::Start() {  // TENSORFLOW_STATUS_OK
  if (recording_) {
    return tsl::errors::Internal("TraceMeRecorder already started");
  }

  // All TraceMe captured should have a timestamp greater or equal to
  // start_timestamp_ns_ to prevent timestamp underflow in XPlane.
  // Therefore this have to be done before TraceMeRecorder::Start.
  start_timestamp_ns_ = tsl::profiler::GetCurrentTimeNanos();
  recording_ = tsl::profiler::TraceMeRecorder::Start(host_trace_level_);
  if (!recording_) {
    return tsl::errors::Internal("Failed to start TraceMeRecorder");
  }
  return tsl::OkStatus();
}

tsl::Status HostTracer::Stop() {  // TENSORFLOW_STATUS_OK
  if (!recording_) {
    return tsl::errors::Internal("TraceMeRecorder not started");
  }
  events_ = tsl::profiler::TraceMeRecorder::Stop();
  recording_ = false;
  return tsl::OkStatus();
}

tsl::Status HostTracer::CollectData(  // TENSORFLOW_STATUS_OK
    tensorflow::profiler::XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from HostTracer.";
  if (recording_) {
    return tsl::errors::Internal("TraceMeRecorder not stopped");
  }
  if (events_.empty()) {
    return tsl::OkStatus();
  }
  tensorflow::profiler::XPlane* plane =
      tsl::profiler::FindOrAddMutablePlaneWithName(
          space, tsl::profiler::kHostThreadsPlaneName);
  ConvertCompleteEventsToXPlane(start_timestamp_ns_, std::exchange(events_, {}),
                                plane);
  return tsl::OkStatus();
}

}  // namespace

std::unique_ptr<tsl::profiler::ProfilerInterface> CreateHostTracer(
    const HostTracerOptions& options) {
  if (options.trace_level == 0) return nullptr;
  return std::make_unique<HostTracer>(options.trace_level);
}

}  // namespace profiler
}  // namespace xla
