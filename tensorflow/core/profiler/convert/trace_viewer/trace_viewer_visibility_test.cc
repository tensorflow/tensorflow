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
#include "tensorflow/core/profiler/convert/trace_viewer/trace_viewer_visibility.h"

#include <cstdint>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tsl/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::Timespan;

constexpr uint32_t kDeviceId = 10;
constexpr uint32_t kResourceId = 1;
constexpr uint32_t kSrcResourceId = 2;
constexpr uint32_t kDstResourceId = 4;

TraceEvent Complete(Timespan span, uint32_t resource_id = kResourceId) {
  TraceEvent event;
  event.set_device_id(kDeviceId);
  event.set_resource_id(resource_id);
  event.set_timestamp_ps(span.begin_ps());
  event.set_duration_ps(span.duration_ps());
  return event;
}

TraceEvent Counter(uint64_t time_ps) {
  TraceEvent event;
  event.set_device_id(kDeviceId);
  event.set_timestamp_ps(time_ps);
  return event;
}

TraceEvent Flow(Timespan span, uint64_t flow_id, uint32_t resource_id) {
  TraceEvent event;
  event.set_flow_id(flow_id);
  event.set_device_id(kDeviceId);
  event.set_resource_id(resource_id);
  event.set_timestamp_ps(span.begin_ps());
  event.set_duration_ps(span.duration_ps());
  return event;
}

TEST(TraceViewerVisibilityTest, VisibilityNoDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000));

  // Instant events.
  EXPECT_FALSE(v.Visible(Complete(Timespan(999))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1000))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1500))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(2000))));
  EXPECT_FALSE(v.Visible(Complete(Timespan(2001))));

  // Complete events.
  EXPECT_FALSE(v.Visible(Complete(Timespan(900, 99))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(900, 100))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1450, 100))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(2000, 50))));
  EXPECT_FALSE(v.Visible(Complete(Timespan(2001, 50))));
}

// TODO(b/218368708): Counter events are currently always visible.
TEST(TraceViewerVisibilityTest, DISABLED_CounterEventsDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000), 100);

  // A counter event within the visible span is visible if its distance from the
  // previous event is >= resolution_ps.
  EXPECT_FALSE(v.Visible(Counter(999)));
  EXPECT_TRUE(v.Visible(Counter(1000)));
  EXPECT_FALSE(v.Visible(Counter(1099)));
  EXPECT_TRUE(v.Visible(Counter(1100)));
  EXPECT_TRUE(v.Visible(Counter(2000)));
  EXPECT_FALSE(v.Visible(Counter(2001)));
}

TEST(TraceViewerVisibilityTest, CompleteEventsDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000), 100);

  // First event is always visible.
  EXPECT_TRUE(v.Visible(Complete(Timespan(950, 50))));
  // Next visible event must have duration_ps >= resolution_ps or its distance
  // from the previous event must be >= resolution_ps.
  EXPECT_FALSE(v.Visible(Complete(Timespan(1050, 50))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1055, 200))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1355, 50))));
}

TEST(TraceViewerVisibilityTest, CompleteNestedEventsDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000), 100);

  // First event is always visible.
  EXPECT_TRUE(v.Visible(Complete(Timespan(1000, 200))));
  // Nested events are visible when increasing depth.
  EXPECT_TRUE(v.Visible(Complete(Timespan(1200, 190))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1250, 20))));
  // Next visible event must have duration_ps >= resolution_ps or its distance
  // from the previous event must be >= resolution_ps.
  EXPECT_FALSE(v.Visible(Complete(Timespan(1270, 20))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1290, 100))));
}

TEST(TraceViewerVisibilityTest, FlowEventsDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000), 100);

  // First event in the flow determines the visibility of the full flow.
  // First flow event in a row is always visible.
  EXPECT_TRUE(v.Visible(Flow(Timespan(1000, 50), 1, kSrcResourceId)));
  // Distance between arrow points must be >= resolution_ps.
  EXPECT_FALSE(v.Visible(Flow(Timespan(1050, 50), 2, kSrcResourceId)));
  EXPECT_TRUE(v.Visible(Flow(Timespan(1100, 50), 3, kSrcResourceId)));

  // Other events in the flow have the same visibility as the first event in
  // the flow.
  EXPECT_TRUE(v.Visible(Flow(Timespan(1100, 50), 1, kDstResourceId)));
  EXPECT_FALSE(v.Visible(Flow(Timespan(1200, 52), 2, kDstResourceId)));
  EXPECT_TRUE(v.Visible(Flow(Timespan(1252, 10), 3, kDstResourceId)));

  // Sanity check for first events: complete events with same distance between
  // events, the third event is not visible because the distance between
  // rectangles is not enough, unlike the distance between arrows.
  EXPECT_TRUE(v.Visible(Complete(Timespan(1300, 50))));
  EXPECT_FALSE(v.Visible(Complete(Timespan(1350, 50))));
  EXPECT_FALSE(v.Visible(Complete(Timespan(1400, 50))));

  // Sanity check for other events: if only the distance between arrows is
  // considered, the second flow would be visible and the third would be
  // invisible.
  EXPECT_TRUE(v.Visible(Flow(Timespan(1600, 50), 4, kResourceId)));
  EXPECT_TRUE(v.Visible(Flow(Timespan(1700, 52), 5, kResourceId)));
  EXPECT_FALSE(v.Visible(Flow(Timespan(1752, 10), 6, kResourceId)));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
