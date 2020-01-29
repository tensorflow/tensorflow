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

#include "tensorflow/core/profiler/utils/group_events.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::CreateTfXPlaneVisitor;
using ::tensorflow::profiler::GetHostEventTypeStr;
using ::tensorflow::profiler::GetStatTypeStr;
using ::tensorflow::profiler::HostEventType;
using ::tensorflow::profiler::StatType;
using ::tensorflow::profiler::XLineBuilder;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XPlaneBuilder;
using ::tensorflow::profiler::XPlaneVisitor;

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    HostEventType event_type, int64 offset_ps, int64 duration_ps,
    const absl::flat_hash_map<StatType, int64 /*stat_value*/>& stats) {
  auto event_builder =
      line_builder->AddEvent(*plane_builder->GetOrCreateEventMetadata(
          GetHostEventTypeStr(event_type)));
  event_builder.SetOffsetPs(offset_ps);
  event_builder.SetDurationPs(duration_ps);
  for (const auto& stat_type_and_value : stats) {
    event_builder.AddStatValue(*plane_builder->GetOrCreateStatMetadata(
                                   GetStatTypeStr(stat_type_and_value.first)),
                               stat_type_and_value.second);
  }
}

// Test if events on the same thread are connected correctly according to the
// nesting relationship.
TEST(GroupEventsTest, ConnectIntraThreadTest) {
  constexpr int64 kLineId = 0;
  XPlane plane;
  XPlaneBuilder plane_builder(&plane);
  plane_builder.ReserveLines(1);
  auto line_builder = plane_builder.GetOrCreateLine(kLineId);
  CreateXEvent(&plane_builder, &line_builder, HostEventType::kTraceContext, 0,
               100, {});
  CreateXEvent(&plane_builder, &line_builder, HostEventType::kFunctionRun, 10,
               90, {});
  CreateXEvent(&plane_builder, &line_builder, HostEventType::kFunctionRun, 110,
               190, {});

  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(&plane);
  EventNodeMap event_node_map(
      {{HostEventType::kTraceContext, {}}, {HostEventType::kFunctionRun, {}}});
  ConnectIntraThread(plane_visitor, &plane, &event_node_map);
  EXPECT_EQ(event_node_map[HostEventType::kTraceContext].size(), 1);
  EXPECT_EQ(event_node_map[HostEventType::kFunctionRun].size(), 2);
  EXPECT_EQ(
      plane_visitor.GetEventType(event_node_map[HostEventType::kFunctionRun][0]
                                     ->GetParent()
                                     ->GetEvent()),
      HostEventType::kTraceContext);
  EXPECT_EQ(event_node_map[HostEventType::kFunctionRun][1]->GetParent(),
            nullptr);
}

// Test (1) if FunctionRun and ExecutorState::Process are connected correctly
// through id and (2) group_id is set correctly.
TEST(GroupEventsTest, ConnectInterThreadTest) {
  XPlane plane;
  XPlaneBuilder plane_builder(&plane);
  plane_builder.ReserveLines(2);

  auto main_thread = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &main_thread, HostEventType::kFunctionRun, 0,
               100, {{StatType::kStepId, 0}});

  auto tf_executor_thread = plane_builder.GetOrCreateLine(1);
  CreateXEvent(&plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 0, 100,
               {{StatType::kStepId, 0}});
  CreateXEvent(&plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 200, 300,
               {{StatType::kStepId, 1}});

  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(&plane);
  EventNodeMap event_node_map({{HostEventType::kFunctionRun, {}},
                               {HostEventType::kExecutorStateProcess, {}}});
  ConnectIntraThread(plane_visitor, &plane, &event_node_map);
  std::vector<InterThreadConnectInfo> connect_info_list(
      {{HostEventType::kFunctionRun,
        HostEventType::kExecutorStateProcess,
        {StatType::kStepId}}});
  ConnectInterThread(event_node_map, connect_info_list);
  EXPECT_EQ(event_node_map[HostEventType::kFunctionRun].size(), 1);
  EXPECT_EQ(event_node_map[HostEventType::kExecutorStateProcess].size(), 2);
  EXPECT_EQ(plane_visitor.GetEventType(
                event_node_map[HostEventType::kExecutorStateProcess][0]
                    ->GetParent()
                    ->GetEvent()),
            HostEventType::kFunctionRun);
  EXPECT_EQ(
      event_node_map[HostEventType::kExecutorStateProcess][1]->GetParent(),
      nullptr);
  EventGroupNameMap event_group_name_map;
  CreateEventGroup({HostEventType::kFunctionRun}, event_node_map,
                   &event_group_name_map);
  EXPECT_EQ(*event_node_map[HostEventType::kFunctionRun][0]->GetGroupId(), 0);
  EXPECT_EQ(
      *event_node_map[HostEventType::kExecutorStateProcess][0]->GetGroupId(),
      0);
  EXPECT_EQ(event_node_map[HostEventType::kExecutorStateProcess][1]
                ->GetGroupId()
                .has_value(),
            false);
  EXPECT_EQ(event_group_name_map.size(), 1);
  EXPECT_EQ(event_group_name_map[0], "0");
}

TEST(GroupEventsTest, GroupGpuTraceTest) {
  XPlane host_plane;
  XPlaneBuilder host_plane_builder(&host_plane);
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, 123}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90, {{StatType::kStepId, 0}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, 0}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kUnknownHostEventType, 30, 70,
               {{StatType::kCorrelationId, 100}, {StatType::kDeviceId, 1}});

  XPlane device_plane;
  XPlaneBuilder device_plane_builder(&device_plane);
  device_plane_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kGroupId));
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream,
               HostEventType::kUnknownHostEventType, 200, 300,
               {{StatType::kCorrelationId, 100}});

  std::vector<InterThreadConnectInfo> connect_info_list(
      {{HostEventType::kFunctionRun,
        HostEventType::kExecutorStateProcess,
        {StatType::kStepId}},
       {HostEventType::kKernelLaunch,
        HostEventType::kKernelExecute,
        {StatType::kCorrelationId}}});
  EventGroupNameMap event_group_name_map;
  GroupEvents(connect_info_list,
              {HostEventType::kTraceContext, HostEventType::kFunctionRun},
              &host_plane, {&device_plane}, &event_group_name_map);
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(&device_plane);
  EXPECT_EQ(device_plane.lines(0).events(0).stats_size(), 2);
  EXPECT_EQ(device_plane_visitor.GetStatType(
                device_plane.lines(0).events(0).stats(1)),
            StatType::kGroupId);
  EXPECT_EQ(event_group_name_map.size(), 1);
  EXPECT_EQ(event_group_name_map[0], "123");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
