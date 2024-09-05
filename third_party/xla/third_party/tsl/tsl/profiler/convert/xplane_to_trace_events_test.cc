/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/profiler/convert/xplane_to_trace_events.h"

#include <limits>
#include <utility>

#include "tsl/platform/test.h"
#include "tsl/profiler/protobuf/trace_events.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/utils/trace_utils.h"
#include "tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/utils/xplane_schema.h"

namespace tsl {
namespace profiler {
namespace {

using tensorflow::profiler::XSpace;

void CreateXSpace(XSpace* space) {
  XPlaneBuilder host_plane(space->add_planes());
  host_plane.SetName(kHostThreadsPlaneName);
  XLineBuilder thread1 = host_plane.GetOrCreateLine(10);
  thread1.SetName("thread1");
  XEventBuilder event1 =
      thread1.AddEvent(*host_plane.GetOrCreateEventMetadata("event1"));
  event1.SetTimestampNs(150000);
  event1.SetDurationNs(10000);
  event1.AddStatValue(*host_plane.GetOrCreateStatMetadata("tf_op"),
                      *host_plane.GetOrCreateStatMetadata("Relu"));
  XLineBuilder thread2 = host_plane.GetOrCreateLine(20);
  thread2.SetName("thread2");
  XEventBuilder event2 =
      thread2.AddEvent(*host_plane.GetOrCreateEventMetadata("event2"));
  event2.SetTimestampNs(160000);
  event2.SetDurationNs(10000);
  event2.AddStatValue(*host_plane.GetOrCreateStatMetadata("tf_op"),
                      *host_plane.GetOrCreateStatMetadata("Conv2D"));

  XPlaneBuilder device_plane(space->add_planes());
  device_plane.SetName(GpuPlaneName(0));
  device_plane.SetId(0);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(30);
  stream1.SetName("gpu stream 1");
  XEventBuilder event3 =
      stream1.AddEvent(*device_plane.GetOrCreateEventMetadata("kernel1"));
  event3.SetTimestampNs(180000);
  event3.SetDurationNs(10000);
  event3.AddStatValue(*device_plane.GetOrCreateStatMetadata("correlation id"),
                      55);
}

TEST(ConvertXPlaneToTraceEvents, Convert) {
  XSpace xspace;
  CreateXSpace(&xspace);

  TraceContainer container = ConvertXSpaceToTraceContainer(xspace);

  ASSERT_EQ(container.trace().devices_size(), 2);
  EXPECT_EQ(
      container.trace().devices().at(kHostThreadsDeviceId).resources_size(), 2);
  EXPECT_EQ(container.trace().devices().at(kFirstDeviceId).resources_size(), 1);
  EXPECT_EQ(container.UnsortedEvents().size(), 3);
}

TEST(ConvertXPlaneToTraceEvents, SkipAsyncOps) {
  XSpace xspace;
  XPlaneBuilder device_plane(xspace.add_planes());
  device_plane.SetName(GpuPlaneName(0));

  XLineBuilder async_ops = device_plane.GetOrCreateLine(10);
  async_ops.SetName(kXlaAsyncOpLineName);

  XEventBuilder event1 =
      async_ops.AddEvent(*device_plane.GetOrCreateEventMetadata("event1"));
  event1.SetTimestampNs(100);
  event1.SetDurationNs(1);

  TraceContainer container = ConvertXSpaceToTraceContainer(xspace);

  ASSERT_THAT(container.UnsortedEvents(), ::testing::IsEmpty());
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
