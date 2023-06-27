/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_VIEWER_COLOR_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_VIEWER_COLOR_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

// Pre-defined color names (excluding "black" and "white") from:
// https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html.
// Possible value of TraceEvent.color_id
enum TraceViewerColor {
  kThreadStateUninterruptible,
  kThreadStateIowait,
  kThreadStateRunning,
  kThreadStateRunnable,
  kThreadStateUnknown,
  kBackgroundMemoryDump,
  kLightMemoryDump,
  kDetailedMemoryDump,
  kVsyncHighlightColor,
  kGenericWork,
  kGood,
  kBad,
  kTerrible,
  kGrey,
  kYellow,
  kOlive,
  kRailResponse,
  kRailAnimation,
  kRailIdle,
  kRailLoad,
  kStartup,
  kHeapDumpStackFrame,
  kHeapDumpObjectType,
  kHeapDumpChildNodeArrow,
  kCqBuildRunning,
  kCqBuildPassed,
  kCqBuildFailed,
  kCqBuildAbandoned,
  kCqBuildAttemptRunnig,
  kCqBuildAttemptPassed,
  kCqBuildAttemptFailed,
};

// Number of named colors in TraceViewer.
constexpr uint32_t kNumTraceViewerColors =
    TraceViewerColor::kCqBuildAttemptFailed + 1;

// Returns the color name for a given color id.
// Used to decode the value in TraceEvent.color_id.
absl::string_view TraceViewerColorName(uint32_t color_id);

// Trace event colorer interface.
class TraceEventsColorerInterface {
 public:
  virtual ~TraceEventsColorerInterface() = default;

  // Allow sub-classes to set up coloring by processing the trace, e.g., by
  // capturing the names of devices and resources that need to be colored.
  virtual void SetUp(const Trace& trace) = 0;

  // Returns the color for a trace event.
  virtual std::optional<uint32_t> GetColor(const TraceEvent& event) const = 0;
};

class DefaultTraceEventsColorer : public TraceEventsColorerInterface {
 public:
  void SetUp(const Trace& trace) override {}

  std::optional<uint32_t> GetColor(const TraceEvent& event) const override {
    return std::nullopt;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_VIEWER_COLOR_H_
