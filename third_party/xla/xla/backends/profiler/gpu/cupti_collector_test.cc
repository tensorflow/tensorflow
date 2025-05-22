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

#include <gtest/gtest.h>

#if GOOGLE_CUDA

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/platform/test.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(CuptiCollectorTest, TestPmSamplingDataToCounterLine) {
  PmSamples pm_samples({"metric1", "metric2"}, {{.range_index = 0,
                                                 .start_timestamp_ns = 100,
                                                 .end_timestamp_ns = 200,
                                                 .metric_values = {1.0, 2.0}},
                                                {.range_index = 1,
                                                 .start_timestamp_ns = 200,
                                                 .end_timestamp_ns = 300,
                                                 .metric_values = {3.0, 4.0}}});
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  pm_samples.PopulateCounterLine(&plane_builder);

  EXPECT_EQ(plane.lines_size(), 1);
  EXPECT_EQ(plane.lines(0).events_size(), 4);
  EXPECT_EQ(plane.event_metadata_size(), 2);
  EXPECT_EQ(plane.stat_metadata_size(), 2);
  absl::flat_hash_map<std::string, absl::flat_hash_map<uint64_t, double>>
      counter_events_values;
  for (const auto& event : plane.lines(0).events()) {
    counter_events_values[plane.event_metadata().at(event.metadata_id()).name()]
                         [event.offset_ps()] = event.stats(0).double_value();
  }
  EXPECT_THAT(counter_events_values,
              UnorderedElementsAre(
                  Pair("metric1", UnorderedElementsAre(Pair(100000, 1.0),
                                                       Pair(200000, 3.0))),
                  Pair("metric2", UnorderedElementsAre(Pair(100000, 2.0),
                                                       Pair(200000, 4.0)))));
}

TEST(CuptiCollectorTest, ExportCallbackActivityAndNvtxEvents) {
  CuptiTracerCollectorOptions options;
  options.max_activity_api_events = 100;
  options.max_callback_api_events = 100;
  options.num_gpus = 1;
  std::unique_ptr<CuptiTraceCollector> collector =
      CreateCuptiCollector(options, 0, 0);

  collector->AddEvent(CuptiTracerEvent{
      .type = CuptiTracerEventType::CudaGraph,
      .source = CuptiTracerEventSource::Activity,
      .name = "CudaGraphExec:2",
      .annotation = "annotation",
      .nvtx_range = "",
      .start_time_ns = 100,
      .end_time_ns = 200,
      .device_id = 0,
      .correlation_id = 8,
      .thread_id = 100,
      .context_id = 1,
      .stream_id = 2,
      .graph_id = 5,
  });

  collector->AddEvent(CuptiTracerEvent{
      .type = CuptiTracerEventType::Generic,
      .source = CuptiTracerEventSource::DriverCallback,
      .name = "cudaGraphLaunch",
      .annotation = "annotation",
      .nvtx_range = "",
      .start_time_ns = 90,
      .end_time_ns = 120,
      .device_id = 0,
      .correlation_id = 8,
      .thread_id = 100,
      .context_id = 1,
      .stream_id = 2,
      .graph_id = 5,
  });

  collector->AddEvent(CuptiTracerEvent{
      .type = CuptiTracerEventType::ThreadMarkerRange,
      .source = CuptiTracerEventSource::Activity,
      .name = "NVTX::MarkCudaGraphLaunch",
      .annotation = "annotation",
      .nvtx_range = "",
      .start_time_ns = 85,
      .end_time_ns = 125,
      .device_id = 0,
      .correlation_id = 0,
      .thread_id = 100,
      .context_id = 1,
      .stream_id = 2,
      .graph_id = 5,
  });

  ::tensorflow::profiler::XSpace space;
  collector->Export(&space, 210);

  // All the three planes must exist in the space:
  // Cupti-Driver-API, Cupti-NVTX, GpuDevice.
  const std::string gpu_device_plane_name = ::tsl::profiler::GpuPlaneName(0);
  const absl::flat_hash_set<absl::string_view> plane_names = {
      ::tsl::profiler::kCuptiDriverApiPlaneName,
      ::tsl::profiler::kCuptiActivityNvtxPlaneName, gpu_device_plane_name};
  int num_planes_to_check = 0;
  for (const auto& plane : space.planes()) {
    if (plane_names.contains(plane.name())) {
      ++num_planes_to_check;
    }
  }
  EXPECT_EQ(num_planes_to_check, static_cast<int>(plane_names.size()));

  // In each above plane, only one line is created, and it has one event.
  for (const auto& plane : space.planes()) {
    if (plane_names.contains(plane.name())) {
      ASSERT_EQ(plane.lines_size(), 1);
      ASSERT_EQ(plane.lines(0).events_size(), 1);
    }
  }
}

}  // namespace
}  // namespace profiler
}  // namespace xla

#endif  // GOOGLE_CUDA
