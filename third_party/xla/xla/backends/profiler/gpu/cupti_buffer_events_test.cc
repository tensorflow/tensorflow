/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cupti_buffer_events.h"

#include <gtest/gtest.h>
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "xla/backends/profiler/gpu/cuda_version_variants.h"

namespace xla {
namespace profiler {
namespace test {

namespace {

// When some change in the CuptiTracerEvent struct is made, this test will
// fail. This is to ensure that the CuptiTracerEvent struct is not changed
// without updating the CuptiBufferEvents code.
TEST(CuptiBufferEventsTest, EventInitialization) {
  CuptiTracerEvent event{
      /* .type = */ CuptiTracerEventType::CudaGraph,
      /* .source = */ CuptiTracerEventSource::Activity,
      /* .name = */ "CudaGraphExec:2",
      /* .annotation = */ "annotation",
      /* .nvtx_range = */ "nvtx_range",
      /* .start_time_ns = */ 100,
      /* .end_time_ns = */ 200,
      /* .device_id = */ 6,
      /* .correlation_id = */ 8,
      /* .thread_id = */ 12345,
      /* .context_id = */ 9,
      /* .stream_id = */ 2,
      /* .graph_id = */ 5,
      /* .scope_range_id = */ 10,
      /* .graph_node_id = */ 11,
  };

  EXPECT_EQ(event.type, CuptiTracerEventType::CudaGraph);
  EXPECT_EQ(event.source, CuptiTracerEventSource::Activity);
  EXPECT_EQ(event.name, "CudaGraphExec:2");
  EXPECT_EQ(event.annotation, "annotation");
  EXPECT_EQ(event.nvtx_range, "nvtx_range");
  EXPECT_EQ(event.start_time_ns, 100);
  EXPECT_EQ(event.end_time_ns, 200);
  EXPECT_EQ(event.device_id, 6);
  EXPECT_EQ(event.correlation_id, 8);
  EXPECT_EQ(event.thread_id, 12345);
  EXPECT_EQ(event.context_id, 9);
  EXPECT_EQ(event.stream_id, 2);
  EXPECT_EQ(event.graph_id, 5);
  EXPECT_EQ(event.scope_range_id, 10);
  EXPECT_EQ(event.graph_node_id, 11);
}

TEST(CuptiBufferEventsTest, GetActivityOverheadKindString) {
  EXPECT_EQ(GetActivityOverheadKindString(CUPTI_ACTIVITY_OVERHEAD_UNKNOWN),
            "Overhead::UNKNOWN:0");
  EXPECT_EQ(
      GetActivityOverheadKindString(CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER),
      "COMPILER");
  EXPECT_EQ(
      GetActivityOverheadKindString(CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH),
      "BUFFER_FLUSH");
  EXPECT_EQ(GetActivityOverheadKindString(
                CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION),
            "INSTRUMENTATION");
  EXPECT_EQ(
      GetActivityOverheadKindString(CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE),
      "RESOURCE");
  if (!cuda_versions::GetExtraActivityOverheadKindString12080(
           static_cast<CUpti_ActivityOverheadKind>(4 << 16))
           .empty()) {
    EXPECT_EQ(GetActivityOverheadKindString(
                  static_cast<CUpti_ActivityOverheadKind>(4 << 16)),
              "RUNTIME_TRIGGERED_MODULE_LOADING");
    EXPECT_EQ(GetActivityOverheadKindString(
                  static_cast<CUpti_ActivityOverheadKind>(5 << 16)),
              "LAZY_FUNCTION_LOADING");
    EXPECT_EQ(GetActivityOverheadKindString(
                  static_cast<CUpti_ActivityOverheadKind>(6 << 16)),
              "COMMAND_BUFFER_FULL");
    EXPECT_EQ(GetActivityOverheadKindString(
                  static_cast<CUpti_ActivityOverheadKind>(7 << 16)),
              "ACTIVITY_BUFFER_REQUEST");
    EXPECT_EQ(GetActivityOverheadKindString(
                  static_cast<CUpti_ActivityOverheadKind>(8 << 16)),
              "UVM_ACTIVITY_INIT");
  }
  EXPECT_EQ(GetActivityOverheadKindString(
                static_cast<CUpti_ActivityOverheadKind>(999)),
            "Overhead::UNKNOWN:999");
}

}  // namespace
}  // namespace test
}  // namespace profiler
}  // namespace xla
