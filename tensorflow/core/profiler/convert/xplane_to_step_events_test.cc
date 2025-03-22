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

#include "tensorflow/core/profiler/convert/xplane_to_step_events.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/event_span.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {
namespace {

// Tests with a sample profile with two steps captured on the host but only one
// step on the device. On the host, each step consists of TraceContext ->
// FunctionRun -> ExecutorState::Process -> matmul. On the host, each step
// consists of matmul. The host's step db should be created only for the step
// observed on the host.
TEST(ConvertXPlaneToOpStats, CpuOnlyStepDbTest) {
  constexpr int64_t kFirstStepNum = 123;
  constexpr int64_t kSecondStepNum = 456;
  constexpr int64_t kFirstStepId = 0;
  constexpr int64_t kSecondStepId = 1;
  constexpr int64_t kFirstCorrelationId = 100;
  constexpr int64_t kSecondCorrelationId = 200;

  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, kFirstStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kFirstStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kFirstStepId}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               300, 100, {{StatType::kStepNum, kSecondStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               310, 90,
               {{StatType::kStepId, kSecondStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kSecondStepId}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 20,
               {{StatType::kStepId, kFirstStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kFirstStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 10,
               {{StatType::kCorrelationId, kFirstCorrelationId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 320, 20,
               {{StatType::kStepId, kSecondStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kSecondStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 330, 10,
               {{StatType::kCorrelationId, kSecondCorrelationId}});

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 50, 40,
               {{StatType::kCorrelationId, kFirstCorrelationId}});

  tsl::profiler::GroupTfEvents(&space);
  StepEvents device_step_events =
      ConvertDeviceTraceXPlaneToStepEvents(*device_plane);
  EXPECT_EQ(device_step_events.size(), 1);
  EXPECT_EQ(device_step_events[0].Events().size(), 1);
  StepEvents host_step_events =
      ConvertHostThreadsXPlaneToStepEvents(*host_plane, &device_step_events);
  // Should contain only the step which is also present on the device.
  EXPECT_EQ(host_step_events.size(), 1);
  // TraceContext should be added as a step marker.
  EXPECT_EQ(host_step_events[0].Markers().size(), 1);
  // FunctionRun shouldn't be added.
  EXPECT_EQ(host_step_events[0].Events().size(), 2);
}

TEST(ConvertXPlaneToStepEvents, TpuDevicePlaneToStepEvents) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  int64_t device_id = 1;
  plane.SetId(device_id);
  plane.SetName("/device:TPU:0");
  XLineBuilder op_line = plane.GetOrCreateLine(0);
  op_line.SetName(tsl::profiler::kXlaOpLineName);
  const XStatMetadata& program_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId));
  const XStatMetadata& symbol_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId));
  const XStatMetadata& group_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  {
    XEventMetadata* event_metadata =
        plane.GetOrCreateEventMetadata("op_long_name");
    event_metadata->set_display_name("op_name");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(program_id_stat, 1);
    stats.AddStatValue(symbol_id_stat, 1);
    {
      XEventBuilder event = op_line.AddEvent(*event_metadata);
      event.SetOffsetPs(0);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 1);
    }
    {
      XEventBuilder event = op_line.AddEvent(*event_metadata);
      event.SetOffsetPs(100);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 2);
    }
  }
  {
    XEventMetadata* event_metadata =
        plane.GetOrCreateEventMetadata("op_long_name2");
    event_metadata->set_display_name("op_name2");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(program_id_stat, 1);
    stats.AddStatValue(symbol_id_stat, 2);
    XEventBuilder event = op_line.AddEvent(*event_metadata);
    event.SetOffsetPs(50);
    event.SetDurationPs(50);
    event.AddStatValue(group_id_stat, 1);
  }
  XLineBuilder step_line = plane.GetOrCreateLine(1);
  step_line.SetName(tsl::profiler::kStepLineName);
  {
    XEventMetadata* event_metadata = plane.CreateEventMetadata();
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    {
      XEventBuilder event = step_line.AddEvent(*event_metadata);
      event.SetOffsetPs(0);
      event.SetDurationPs(100);
      event.AddStatValue(group_id_stat, 1);
    }
    {
      XEventBuilder event = step_line.AddEvent(*event_metadata);
      event.SetOffsetPs(100);
      event.SetDurationPs(100);
      event.AddStatValue(group_id_stat, 2);
    }
  }

  StepEvents step_events = ConvertDeviceTraceXPlaneToStepEvents(raw_plane);
  EXPECT_EQ(step_events.size(), 2);
  EXPECT_TRUE(step_events.contains(1));
  StepDetails step_1 = step_events[/*group_id=*/1];
  ASSERT_TRUE(step_1.PerCoreOpMetricsDb().contains(device_id));
  EXPECT_EQ(step_1.PerCoreOpMetricsDb().at(device_id).metrics_db_size(), 2);
  EXPECT_EQ(step_1.Markers().size(), 1);
  EXPECT_TRUE(step_events.contains(2));
  StepDetails step_2 = step_events[/*group_id=*/2];
  ASSERT_TRUE(step_2.PerCoreOpMetricsDb().contains(device_id));
  EXPECT_EQ(step_2.PerCoreOpMetricsDb().at(device_id).metrics_db_size(), 1);
  EXPECT_EQ(step_2.Markers().size(), 1);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
