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
#include "absl/types/optional.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(GroupEventsTest, GroupGpuTraceTest) {
  XSpace space;
  XPlaneBuilder host_plane_builder(space.add_planes());
  host_plane_builder.SetName(kHostThreads);
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
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 70,
               {{StatType::kCorrelationId, 100}});

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 200, 300,
               {{StatType::kCorrelationId, 100}});

  EventGroupNameMap event_group_name_map;
  GroupTfEvents(&space, &event_group_name_map);
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(device_plane);
  EXPECT_EQ(device_plane->lines(0).events(0).stats_size(), 3);
  EXPECT_EQ(device_plane_visitor.GetStatType(
                device_plane->lines(0).events(0).stats(1)),
            StatType::kGroupId);
  EXPECT_EQ(event_group_name_map.size(), 1);
  EXPECT_EQ(event_group_name_map[0], "123");
}

TEST(GroupEventsTest, GroupTensorFlowLoopTest) {
  XSpace space;
  XPlaneBuilder host_plane_builder(space.add_planes());
  host_plane_builder.SetName(kHostThreads);
  host_plane_builder.ReserveLines(1);

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 5, 10,
               {{StatType::kStepId, 0}, {StatType::kIterNum, 10}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, 0}, {StatType::kIterNum, 10}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 70,
               {{StatType::kCorrelationId, 100}});

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 200, 300,
               {{StatType::kCorrelationId, 100}});

  EventGroupNameMap event_group_name_map;
  GroupTfEvents(&space, &event_group_name_map);
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(device_plane);
  EXPECT_EQ(device_plane->lines(0).events(0).stats_size(), 3);
  EXPECT_EQ(device_plane_visitor.GetStatType(
                device_plane->lines(0).events(0).stats(1)),
            StatType::kGroupId);
  EXPECT_EQ(device_plane->lines(0).events(0).stats(1).int64_value(), 10);
  EXPECT_EQ(event_group_name_map.size(), 1);
  EXPECT_EQ(event_group_name_map[10], "10");
}

// When there are multiple TF loops, group_id is assigned in the order of TF
// loops' start times and iter_num. In this test case, the profile captures the
// last two iterations (iter_num=10,11) of the first TF loop (step_id=0) and the
// first two iterations (iter_num=0,1) of the second TF loop (step_id=1).
// group_id is initialized to the first TF loop's first iter_num (10) and then
// monotonically increased.
TEST(GroupEventsTest, GroupMultipleTensorFlowLoopsTest) {
  XSpace space;
  XPlaneBuilder host_plane_builder(space.add_planes());
  host_plane_builder.SetName(kHostThreads);
  host_plane_builder.ReserveLines(2);

  auto first_tf_executor_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &first_tf_executor_thread,
               HostEventType::kExecutorStateProcess, 220, 80,
               {{StatType::kStepId, 1}, {StatType::kIterNum, 0}});
  CreateXEvent(&host_plane_builder, &first_tf_executor_thread,
               HostEventType::kExecutorStateProcess, 320, 80,
               {{StatType::kStepId, 1}, {StatType::kIterNum, 1}});
  auto second_tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &second_tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, 0}, {StatType::kIterNum, 10}});
  CreateXEvent(&host_plane_builder, &second_tf_executor_thread,
               HostEventType::kExecutorStateProcess, 120, 80,
               {{StatType::kStepId, 0}, {StatType::kIterNum, 11}});

  EventGroupNameMap event_group_name_map;
  GroupTfEvents(&space, &event_group_name_map);
  EXPECT_EQ(event_group_name_map.size(), 4);
  EXPECT_TRUE(event_group_name_map.count(10));
  EXPECT_TRUE(event_group_name_map.count(11));
  EXPECT_TRUE(event_group_name_map.count(12));
  EXPECT_TRUE(event_group_name_map.count(13));
}

TEST(GroupEventsTest, GroupFunctionalOp) {
  XSpace space;
  XPlane* host_plane = space.add_planes();
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.SetName(kHostThreads);
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 200, {{StatType::kStepNum, 123}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 190, {{StatType::kStepId, 0}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, 0}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kRemoteCallOp, 30, 70,
               {{StatType::kFunctionStepId, 1}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 100, 150,
               {{StatType::kStepId, 1}});

  EventGroupNameMap event_group_name_map;
  GroupTfEvents(&space, &event_group_name_map);
  XPlaneVisitor host_plane_visitor = CreateTfXPlaneVisitor(host_plane);
  // Check that RemoteCallOp is grouped correctly so that all events belong
  // to the same group.
  host_plane_visitor.ForEachLine(
      [&](const tensorflow::profiler::XLineVisitor& line) {
        line.ForEachEvent(
            [&](const tensorflow::profiler::XEventVisitor& event) {
              absl::optional<int64> group_id;
              event.ForEachStat(
                  [&](const tensorflow::profiler::XStatVisitor& stat) {
                    if (stat.Type() == StatType::kGroupId) {
                      group_id = stat.IntValue();
                    }
                  });
              EXPECT_TRUE(group_id.has_value());
              EXPECT_EQ(*group_id, 0);
            });
      });
}

TEST(GroupEventsTest, EagerOpTest) {
  XSpace space;
  XPlane* host_plane = space.add_planes();
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.SetName(kHostThreads);
  host_plane_builder.ReserveLines(1);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  // Eagerly scheduled GPU kernel.
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kEagerKernelExecute, 10, 100, {});
  CreateXEvent(&host_plane_builder, &main_thread, "matmul", 10, 100,
               {{StatType::kCorrelationId, 100}});
  // Eagerly executed CPU TF op.
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kEagerKernelExecute, 120, 80, {});
  CreateXEvent(&host_plane_builder, &main_thread, "add:Add", 120, 80);

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  // Eagerly executed GPU kernel.
  CreateXEvent(&device_plane_builder, &stream, "matmul", 200, 300,
               {{StatType::kCorrelationId, 100}});

  GroupTfEvents(&space, /*event_group_name_map=*/nullptr);
  XPlaneVisitor host_plane_visitor = CreateTfXPlaneVisitor(host_plane);
  const XEvent& eager_cpu_tf_op = host_plane->lines(0).events(3);
  EXPECT_EQ(eager_cpu_tf_op.stats_size(), 1);
  EXPECT_EQ(host_plane_visitor.GetStatType(eager_cpu_tf_op.stats(0)),
            StatType::kIsEager);
  EXPECT_EQ(eager_cpu_tf_op.stats(0).int64_value(), 1);
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(device_plane);
  const XEvent& eager_gpu_kernel = device_plane->lines(0).events(0);
  EXPECT_EQ(eager_gpu_kernel.stats_size(), 2);
  EXPECT_EQ(device_plane_visitor.GetStatType(eager_gpu_kernel.stats(1)),
            StatType::kIsEager);
  EXPECT_EQ(eager_gpu_kernel.stats(1).int64_value(), 1);
}

TEST(GroupEventsTest, FunctionOpTest) {
  XSpace space;
  XPlane* host_plane = space.add_planes();
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.SetName(kHostThreads);
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, 123}});
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kEagerKernelExecute, 10, 90, {});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90, {{StatType::kStepId, 0}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, 0}});
  // GPU kernel scheduled inside tf.function.
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 30,
               {{StatType::kCorrelationId, 100}});
  // CPU TF op executed inside tf.function.
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "add:Add", 70, 20);

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  // GPU kernel executed as part of tf.function.
  CreateXEvent(&device_plane_builder, &stream, "matmul", 200, 300,
               {{StatType::kCorrelationId, 100}});

  GroupTfEvents(&space, /*event_group_name_map=*/nullptr);
  XPlaneVisitor host_plane_visitor = CreateTfXPlaneVisitor(host_plane);
  const XEvent& cpu_tf_op = host_plane->lines(1).events(2);
  EXPECT_EQ(cpu_tf_op.stats_size(), 2);
  EXPECT_EQ(host_plane_visitor.GetStatType(cpu_tf_op.stats(1)),
            StatType::kIsEager);
  EXPECT_EQ(cpu_tf_op.stats(1).int64_value(), 0);
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(device_plane);
  const XEvent& gpu_kernel = device_plane->lines(0).events(0);
  EXPECT_EQ(gpu_kernel.stats_size(), 3);
  EXPECT_EQ(device_plane_visitor.GetStatType(gpu_kernel.stats(2)),
            StatType::kIsEager);
  EXPECT_EQ(gpu_kernel.stats(2).int64_value(), 0);
}

TEST(GroupEventsTest, SemanticArgTest) {
  constexpr int64 kStepNum = 100;
  constexpr int kContextType = 123;
  constexpr uint64 kContextId = 456;

  XSpace raw_space;
  XPlane* raw_plane = raw_space.add_planes();
  XPlaneBuilder plane(raw_plane);
  plane.ReserveLines(2);
  auto root_producer = plane.GetOrCreateLine(0);
  CreateXEvent(&plane, &root_producer, HostEventType::kTraceContext, 0, 100,
               {{StatType::kIsRoot, 1}, {StatType::kStepNum, kStepNum}});
  CreateXEvent(&plane, &root_producer, HostEventType::kFunctionRun, 10, 90,
               {{StatType::kProducerType, kContextType},
                {StatType::kProducerId, kContextId}});
  auto consumer = plane.GetOrCreateLine(1);
  CreateXEvent(&plane, &consumer, HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kConsumerType, kContextType},
                {StatType::kConsumerId, kContextId}});

  GroupTfEvents(&raw_space, /*event_group_name_map=*/nullptr);
  int num_events = 0;
  CreateTfXPlaneVisitor(raw_plane).ForEachLine(
      [&](const tensorflow::profiler::XLineVisitor& line) {
        num_events += line.NumEvents();
        line.ForEachEvent(
            [&](const tensorflow::profiler::XEventVisitor& event) {
              absl::optional<int64> group_id;
              event.ForEachStat(
                  [&](const tensorflow::profiler::XStatVisitor& stat) {
                    if (stat.Type() == StatType::kGroupId) {
                      group_id = stat.IntValue();
                    }
                  });
              EXPECT_TRUE(group_id.has_value());
              EXPECT_EQ(*group_id, 0);
            });
      });
  EXPECT_EQ(num_events, 3);
}

TEST(GroupEventsTest, AsyncEventTest) {
  constexpr absl::string_view kParent = "parent";
  constexpr absl::string_view kAsync = "async";
  constexpr absl::string_view kChild = "child";

  XSpace raw_space;
  XPlane* raw_plane = raw_space.add_planes();
  XPlaneBuilder plane(raw_plane);
  plane.ReserveLines(1);
  auto line = plane.GetOrCreateLine(0);
  CreateXEvent(&plane, &line, kParent, 0, 100, {{StatType::kIsRoot, 1}});
  CreateXEvent(&plane, &line, kAsync, 10, 200, {{StatType::kIsAsync, 1}});
  CreateXEvent(&plane, &line, kChild, 20, 80);

  GroupTfEvents(&raw_space, /*event_group_name_map=*/nullptr);
  CreateTfXPlaneVisitor(raw_plane).ForEachLine(
      [&](const tensorflow::profiler::XLineVisitor& line) {
        EXPECT_EQ(line.NumEvents(), 3);
        line.ForEachEvent(
            [&](const tensorflow::profiler::XEventVisitor& event) {
              absl::optional<int64> group_id;
              event.ForEachStat(
                  [&](const tensorflow::profiler::XStatVisitor& stat) {
                    if (stat.Type() == StatType::kGroupId) {
                      group_id = stat.IntValue();
                    }
                  });
              if (event.Name() == kAsync) {
                EXPECT_FALSE(group_id.has_value());
              } else {
                EXPECT_TRUE(group_id.has_value());
                EXPECT_EQ(*group_id, 0);
              }
            });
      });
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
