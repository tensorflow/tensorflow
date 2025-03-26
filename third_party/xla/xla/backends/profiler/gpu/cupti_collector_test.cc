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

#if GOOGLE_CUDA

#include "xla/backends/profiler/gpu/cupti_collector.h"

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
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

}  // namespace
}  // namespace profiler
}  // namespace xla

#endif  // GOOGLE_CUDA
