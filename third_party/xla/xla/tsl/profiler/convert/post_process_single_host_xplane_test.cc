/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tsl/profiler/convert/post_process_single_host_xplane.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::tensorflow::profiler::XSpace;

void CreateXSpace(XSpace* space) {
  // Create a python plane with one line (id 10) with two events, and another
  // line (id 15) with one event.
  XPlaneBuilder python_plane(space->add_planes());
  python_plane.SetName(kPythonTracerPlaneName);
  XLineBuilder py_thread1 = python_plane.GetOrCreateLine(10);
  py_thread1.SetName("thread/10");
  XEventBuilder py_event1 =
      py_thread1.AddEvent(*python_plane.GetOrCreateEventMetadata("py-event1"));
  py_event1.SetTimestampNs(1000);
  py_event1.SetDurationNs(900);
  XEventBuilder py_event2 =
      py_thread1.AddEvent(*python_plane.GetOrCreateEventMetadata("py-event2"));
  py_event2.SetTimestampNs(2000);
  py_event2.SetDurationNs(600);
  XLineBuilder py_thread2 = python_plane.GetOrCreateLine(15);
  py_thread2.SetName("thread/15");
  XEventBuilder py_event3 =
      py_thread2.AddEvent(*python_plane.GetOrCreateEventMetadata("py-event3"));
  py_event3.SetTimestampNs(1200);
  py_event3.SetDurationNs(800);

  // Create a cupti plane with two lines (id 10 and 20), each with one event.
  XPlaneBuilder cupti_plane(space->add_planes());
  cupti_plane.SetName(kCuptiDriverApiPlaneName);
  XLineBuilder cupti_thread1 = cupti_plane.GetOrCreateLine(10);
  cupti_thread1.SetName("thread/10");
  XEventBuilder cupti_event1 = cupti_thread1.AddEvent(
      *cupti_plane.GetOrCreateEventMetadata("cupti-event1"));
  cupti_event1.SetTimestampNs(1100);
  cupti_event1.SetDurationNs(800);
  XLineBuilder cupti_thread2 = cupti_plane.GetOrCreateLine(20);
  cupti_thread2.SetName("thread/20");
  XEventBuilder cupti_event2 = cupti_thread2.AddEvent(
      *cupti_plane.GetOrCreateEventMetadata("cupti-event2"));
  cupti_event2.SetTimestampNs(1100);
  cupti_event2.SetDurationNs(800);

  // Create a nvtx plane with two lines (id 10 and 50), each with one event.
  XPlaneBuilder nvtx_plane(space->add_planes());
  nvtx_plane.SetName(kCuptiActivityNvtxPlaneName);
  XLineBuilder nvtx_thread1 = nvtx_plane.GetOrCreateLine(10);
  nvtx_thread1.SetName("thread/10/NVTX");
  XEventBuilder nvtx_event1 = nvtx_thread1.AddEvent(
      *nvtx_plane.GetOrCreateEventMetadata("nvtx-event1"));
  nvtx_event1.SetTimestampNs(1200);
  nvtx_event1.SetDurationNs(600);

  XLineBuilder nvtx_thread2 = nvtx_plane.GetOrCreateLine(50);
  nvtx_thread2.SetName("thread/50/NVTX");
  XEventBuilder nvtx_event2 = nvtx_thread2.AddEvent(
      *nvtx_plane.GetOrCreateEventMetadata("nvtx-event2"));
  nvtx_event2.SetTimestampNs(1200);
  nvtx_event2.SetDurationNs(600);
}

TEST(ConvertXPlaneToTraceEvents, Convert) {
  XSpace xspace;
  CreateXSpace(&xspace);

  PostProcessSingleHostXSpace(&xspace, 0, 5000);

  // After merge, only one host plane exist, its name is kHostThreadsPlaneName.
  ASSERT_EQ(xspace.planes_size(), 1);
  EXPECT_EQ(xspace.planes(0).name(), kHostThreadsPlaneName);

  // In the host plane, there should be lines with ids {10, 15, 20, 50, new_id}
  // where new_id is reassigned from the line id 10 from the nvtx plane, as 10
  // is occupied by the cupti plane line with id 10.
  const XPlane& host_plane = xspace.planes(0);
  ASSERT_EQ(host_plane.lines_size(), 5);
  absl::flat_hash_set<int64_t> known_line_ids = {10, 15, 20, 50};
  absl::flat_hash_set<int64_t> line_ids, extra_new_line_ids;
  for (const XLine& line : host_plane.lines()) {
    auto line_id = line.id();
    if (known_line_ids.contains(line_id)) {
      line_ids.insert(line_id);
    } else {
      extra_new_line_ids.insert(line_id);
    }
  }
  EXPECT_THAT(line_ids, ::testing::ContainerEq(known_line_ids));
  EXPECT_THAT(extra_new_line_ids, ::testing::SizeIs(1));

  // The lines in the merged host plane should be sorted by name.
  EXPECT_THAT(host_plane.lines(0).name(), "thread/10");
  EXPECT_THAT(host_plane.lines(1).name(), "thread/10/NVTX");
  EXPECT_THAT(host_plane.lines(2).name(), "thread/15");
  EXPECT_THAT(host_plane.lines(3).name(), "thread/20");
  EXPECT_THAT(host_plane.lines(4).name(), "thread/50/NVTX");
  // The line with id 10 is merged from the python plane line with id 10 and the
  // cupti plane line with id 10, so it should have 3 (= 2 + 1) events.
  EXPECT_EQ(host_plane.lines(0).events_size(), 3);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
