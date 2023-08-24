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
#include "tensorflow/core/profiler/convert/trace_viewer/trace_viewer_color.h"

#include <cstdint>

namespace tensorflow {
namespace profiler {
namespace {

// Pre-defined color names (excluding "black" and "white") from:
// https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html.
// Use raw string to add double-quote around the color name.
const absl::string_view kTraceViewerColors[kNumTraceViewerColors] = {
    R"("thread_state_uninterruptible")",
    R"("thread_state_iowait")",
    R"("thread_state_running")",
    R"("thread_state_runnable")",
    R"("thread_state_unknown")",
    R"("background_memory_dump")",
    R"("light_memory_dump")",
    R"("detailed_memory_dump")",
    R"("vsync_highlight_color")",
    R"("generic_work")",
    R"("good")",
    R"("bad")",
    R"("terrible")",
    R"("grey")",
    R"("yellow")",
    R"("olive")",
    R"("rail_response")",
    R"("rail_animation")",
    R"("rail_idle")",
    R"("rail_load")",
    R"("startup")",
    R"("heap_dump_stack_frame")",
    R"("heap_dump_object_type")",
    R"("heap_dump_child_node_arrow")",
    R"("cq_build_running")",
    R"("cq_build_passed")",
    R"("cq_build_failed")",
    R"("cq_build_abandoned")",
    R"("cq_build_attempt_runnig")",
    R"("cq_build_attempt_passed")",
    R"("cq_build_attempt_failed")"};

}  // namespace

absl::string_view TraceViewerColorName(uint32_t color_id) {
  return kTraceViewerColors[color_id % kNumTraceViewerColors];
}

}  // namespace profiler
}  // namespace tensorflow
