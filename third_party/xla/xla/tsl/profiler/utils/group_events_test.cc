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

#include "xla/tsl/profiler/utils/group_events.h"

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

constexpr int64_t kTfExecutor = static_cast<int64_t>(ContextType::kTfExecutor);

TEST(GroupEventsTest, GroupGpuTraceLegacyRootTest) {
  constexpr int64_t kStepNum = 123;
  constexpr int64_t kStepId = 0;
  constexpr int64_t kCorrelationId = 100;

  XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(
      &host_plane_builder, &main_thread, HostEventType::kTraceContext, 0, 100,
      {{StatType::kGraphType, "train"}, {StatType::kStepNum, kStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kStepId},
                {StatType::kProducerType, kTfExecutor},
                {StatType::kProducerId, kStepId}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, kStepId},
                {StatType::kConsumerType, kTfExecutor},
                {StatType::kConsumerId, kStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 70,
               {{StatType::kCorrelationId, kCorrelationId}});

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 200, 300,
               {{StatType::kCorrelationId, kCorrelationId}});

  EventForest event_forest;
  GroupTfEvents(&space, &event_forest);
  const GroupMetadataMap& group_metadata_map =
      event_forest.GetGroupMetadataMap();
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(device_plane);
  EXPECT_EQ(device_plane->lines(0).events(0).stats_size(), 3);
  EXPECT_EQ(device_plane_visitor.GetStatType(
                device_plane->lines(0).events(0).stats(1).metadata_id()),
            StatType::kGroupId);
  EXPECT_EQ(group_metadata_map.size(), 1);
  EXPECT_EQ(group_metadata_map.at(0).name, "train 123");
}

TEST(GroupEventsTest, GroupGpuTraceTest) {
  constexpr int64_t kStepNum = 123;
  constexpr int64_t kStepId = 0;
  constexpr int64_t kCorrelationId = 100;

  XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(
      &host_plane_builder, &main_thread, "train", 0, 100,
      {{StatType::kStepNum, kStepNum}, {StatType::kIsRoot, int64_t{1}}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kStepId},
                {StatType::kProducerType, kTfExecutor},
                {StatType::kProducerId, kStepId}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, kStepId},
                {StatType::kConsumerType, kTfExecutor},
                {StatType::kConsumerId, kStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 70,
               {{StatType::kCorrelationId, kCorrelationId}});

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 200, 300,
               {{StatType::kCorrelationId, kCorrelationId}});

  EventForest event_forest;
  GroupTfEvents(&space, &event_forest);
  const GroupMetadataMap& group_metadata_map =
      event_forest.GetGroupMetadataMap();
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(device_plane);
  EXPECT_EQ(device_plane->lines(0).events(0).stats_size(), 3);
  EXPECT_EQ(device_plane_visitor.GetStatType(
                device_plane->lines(0).events(0).stats(1).metadata_id()),
            StatType::kGroupId);
  EXPECT_EQ(group_metadata_map.size(), 1);
  EXPECT_EQ(group_metadata_map.at(0).name, "train 123");
}

TEST(GroupEventsTest, GroupTensorFlowLoopTest) {
  constexpr int64_t kStepId = 0;
  constexpr int64_t kIterNum = 10;
  constexpr int64_t kCorrelationId = 100;
  constexpr int64_t kRawValue = 10;

  XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(1);

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 5, 10,
               {{StatType::kStepId, kStepId},
                {StatType::kIterNum, kIterNum},
                {StatType::kConsumerType, kTfExecutor},
                {StatType::kConsumerId, kStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, kStepId},
                {StatType::kIterNum, kIterNum},
                {StatType::kConsumerType, kTfExecutor},
                {StatType::kConsumerId, kStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 70,
               {{StatType::kCorrelationId, kCorrelationId}});

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 200, 300,
               {{StatType::kCorrelationId, kCorrelationId}});

  auto sync_flag_line = device_plane_builder.GetOrCreateLine(1);
  sync_flag_line.SetName(kTensorCoreSyncFlagLineName);
  CreateXEvent(&device_plane_builder, &sync_flag_line, "SyncWait", 200, 300,
               {{StatType::kRawValue, kRawValue}});

  EventForest event_forest;
  GroupTfEvents(&space, &event_forest);
  const GroupMetadataMap& group_metadata_map =
      event_forest.GetGroupMetadataMap();
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(device_plane);
  EXPECT_EQ(device_plane->lines(1).events(0).stats_size(), 1);
  EXPECT_EQ(device_plane_visitor.GetStatType(
                device_plane->lines(1).events(0).stats(0).metadata_id()),
            StatType::kRawValue);
  EXPECT_EQ(device_plane->lines(0).events(0).stats_size(), 3);
  EXPECT_EQ(device_plane_visitor.GetStatType(
                device_plane->lines(0).events(0).stats(1).metadata_id()),
            StatType::kGroupId);
  // group_id is assigned using a list of consecutive number starting from 0.
  EXPECT_EQ(device_plane->lines(0).events(0).stats(1).int64_value(), 0);
  EXPECT_EQ(group_metadata_map.size(), 1);
  // group name of ExecutorState::Process event is assigned using iter_num.
  ASSERT_TRUE(group_metadata_map.contains(0));
  EXPECT_EQ(group_metadata_map.at(0).name, "10");
}

// When there are multiple TF loops, group_id is assigned in the order of TF
// loops' start times and iter_num. In this test case, the profile captures the
// last two iterations (iter_num=10,11) of the first TF loop (step_id=0) and the
// first two iterations (iter_num=0,1) of the second TF loop (step_id=1).
// group_id is initialized to the first TF loop's first iter_num (10) and then
// monotonically increased.
TEST(GroupEventsTest, GroupMultipleTensorFlowLoopsTest) {
  constexpr int64_t kFirstStepId = 0;
  constexpr int64_t kSecondStepId = 1;
  constexpr int64_t kFirstIterNumStart = 10;
  constexpr int64_t kSecondIterNumStart = 0;

  XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(2);

  auto first_tf_executor_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &first_tf_executor_thread,
               HostEventType::kExecutorStateProcess, 220, 80,
               {{StatType::kStepId, kSecondStepId},
                {StatType::kIterNum, kSecondIterNumStart},
                {StatType::kConsumerType, kTfExecutor},
                {StatType::kConsumerId, kSecondStepId}});
  CreateXEvent(&host_plane_builder, &first_tf_executor_thread,
               HostEventType::kExecutorStateProcess, 320, 80,
               {{StatType::kStepId, kSecondStepId},
                {StatType::kIterNum, kSecondIterNumStart + 1},
                {StatType::kConsumerType, kTfExecutor},
                {StatType::kConsumerId, kSecondStepId}});
  auto second_tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &second_tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, kFirstStepId},
                {StatType::kIterNum, kFirstIterNumStart},
                {StatType::kConsumerType, kTfExecutor},
                {StatType::kConsumerId, kFirstStepId}});
  CreateXEvent(&host_plane_builder, &second_tf_executor_thread,
               HostEventType::kExecutorStateProcess, 120, 80,
               {{StatType::kStepId, kFirstStepId},
                {StatType::kIterNum, kFirstIterNumStart + 1},
                {StatType::kConsumerType, kTfExecutor},
                {StatType::kConsumerId, kFirstStepId}});

  EventForest event_forest;
  GroupTfEvents(&space, &event_forest);
  const GroupMetadataMap& group_metadata_map =
      event_forest.GetGroupMetadataMap();
  EXPECT_EQ(group_metadata_map.size(), 4);
  // group_id is assigned using a list of consecutive number starting from 0,
  // event with an earlier start time will get a smaller group_id.
  // group name of ExecutorState::Process event is assigned using iter_num.
  ASSERT_TRUE(group_metadata_map.contains(0));
  // iter_num 10 starts at timestamp 20, so it has the smallest group_id.
  EXPECT_EQ(group_metadata_map.at(0).name, "10");
  ASSERT_TRUE(group_metadata_map.contains(1));
  EXPECT_EQ(group_metadata_map.at(1).name, "11");
  ASSERT_TRUE(group_metadata_map.contains(2));
  EXPECT_EQ(group_metadata_map.at(2).name, "0");
  ASSERT_TRUE(group_metadata_map.contains(3));
  // iter_num 1 starts at timestamp 320, so it has the largest group_id.
  EXPECT_EQ(group_metadata_map.at(3).name, "1");
}

TEST(GroupEventsTest, EagerOpTest) {
  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.ReserveLines(1);
  auto main_thread = host_plane_builder.GetOrCreateLine(0);

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);
  auto gpu_stream = device_plane_builder.GetOrCreateLine(0);

  int64_t correlation_id = 100;
  // TF1 ops are NOT scheduled under kEagerKernelExecute events, they should be
  // considered NOT eager.
  const char* kTF1GpuLaunchEvent = "tf1 matmul";
  const char* kTF1GpuEvent = "tf1_kernel_matmul";
  CreateXEvent(&host_plane_builder, &main_thread, kTF1GpuLaunchEvent, 10, 90,
               {{StatType::kCorrelationId, correlation_id}});
  CreateXEvent(&device_plane_builder, &gpu_stream, kTF1GpuEvent, 200, 300,
               {{StatType::kCorrelationId, correlation_id}});
  ++correlation_id;

  // Eagerly scheduled GPU operator w/o is_func Xstat (legacy). The legacy trace
  // will also fall into this case, due to the fact we changed the EagerExecute
  // TraceMe format. We treat them as NOT eager
  const char* kLegacyGpuLaunchEvent = "legacy matmul";
  const char* kLegacyGpuEvent = "legacy_kernel_matmul";
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kEagerKernelExecute, 100, 200);
  CreateXEvent(&host_plane_builder, &main_thread, kLegacyGpuLaunchEvent, 110,
               190, {{StatType::kCorrelationId, correlation_id}});
  CreateXEvent(&device_plane_builder, &gpu_stream, kLegacyGpuEvent, 300, 400,
               {{StatType::kCorrelationId, correlation_id}});
  ++correlation_id;

  // Eagerly scheduled GPU op with is_func Xstat.
  const char* kEagerOpGpuLaunchEvent = "eager op matmul";
  const char* kEagerOpGpuEvent = "eager_op_kernel_matmul";
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kEagerKernelExecute, 200, 300,
               {{StatType::kIsFunc, static_cast<int64_t>(0)}});
  CreateXEvent(&host_plane_builder, &main_thread, kEagerOpGpuLaunchEvent, 210,
               290, {{StatType::kCorrelationId, correlation_id}});
  CreateXEvent(&device_plane_builder, &gpu_stream, kEagerOpGpuEvent, 400, 500,
               {{StatType::kCorrelationId, correlation_id}});
  ++correlation_id;

  // Eagerly scheduled GPU func with is_func Xstat.
  const char* kEagerFuncGpuLaunchEvent = "eager func matmul";
  const char* kEagerFuncGpuEvent = "eager_func_kernel_matmul";
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kEagerKernelExecute, 300, 400,
               {{StatType::kIsFunc, static_cast<int64_t>(1)}});
  CreateXEvent(&host_plane_builder, &main_thread, kEagerFuncGpuLaunchEvent, 310,
               390, {{StatType::kCorrelationId, correlation_id}});
  CreateXEvent(&device_plane_builder, &gpu_stream, kEagerFuncGpuEvent, 500, 600,
               {{StatType::kCorrelationId, correlation_id}});
  ++correlation_id;

  // Eagerly executed CPU TF op.
  const char* kEagerOpCpuEvent = "eager_op_cpu_kernel:Matmul";
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kEagerKernelExecute, 400, 500,
               {{StatType::kIsFunc, static_cast<int64_t>(0)}});
  CreateXEvent(&host_plane_builder, &main_thread, kEagerOpCpuEvent, 410, 490);

  // Eagerly executed CPU TF function.
  const char* kEagerFuncCpuEvent = "eager_func_cpu_kernel:Matmul";
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kEagerKernelExecute, 500, 600,
               {{StatType::kIsFunc, static_cast<int64_t>(1)}});
  CreateXEvent(&host_plane_builder, &main_thread, kEagerFuncCpuEvent, 510, 590);

  GroupTfEvents(&space);

  auto is_eager = [](const XEventVisitor& event) {
    auto eager_stats = event.GetStat(StatType::kIsEager);
    return eager_stats && eager_stats->IntValue();
  };
  // verify host ops.
  XPlaneVisitor host_plane_visitor = CreateTfXPlaneVisitor(host_plane);
  int interested_events_encountered = 0;
  host_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Name() == kEagerOpCpuEvent) {
        interested_events_encountered++;
        EXPECT_TRUE(is_eager(event));
      } else if (event.Name() == kEagerFuncCpuEvent) {
        interested_events_encountered++;
        EXPECT_FALSE(is_eager(event));
      }
    });
  });
  EXPECT_EQ(interested_events_encountered, 2);

  // verify device ops.
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(device_plane);
  interested_events_encountered = 0;
  device_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Name() == kTF1GpuEvent) {
        interested_events_encountered++;
        EXPECT_FALSE(is_eager(event));
      } else if (event.Name() == kLegacyGpuEvent) {
        interested_events_encountered++;
        EXPECT_FALSE(is_eager(event));
      } else if (event.Name() == kEagerOpGpuEvent) {
        interested_events_encountered++;
        EXPECT_TRUE(is_eager(event));
      } else if (event.Name() == kEagerFuncGpuEvent) {
        interested_events_encountered++;
        EXPECT_FALSE(is_eager(event));
      }
    });
  });
  EXPECT_EQ(interested_events_encountered, 4);
}

TEST(GroupEventsTest, FunctionOpTest) {
  constexpr int64_t kStepNum = 123;
  constexpr int64_t kStepId = 0;
  constexpr int64_t kCorrelationId = 100;

  XSpace space;
  XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, kStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kEagerKernelExecute, 10, 90);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kStepId},
                {StatType::kProducerType, kTfExecutor},
                {StatType::kProducerId, kStepId}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, kStepId},
                {StatType::kConsumerType, kTfExecutor},
                {StatType::kConsumerId, kStepId}});

  // GPU kernel scheduled inside tf.function.
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 30,
               {{StatType::kCorrelationId, kCorrelationId}});
  // CPU TF op executed inside tf.function.
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "add:Add", 70, 20);

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  // GPU kernel executed as part of tf.function.
  CreateXEvent(&device_plane_builder, &stream, "matmul", 200, 300,
               {{StatType::kCorrelationId, kCorrelationId}});

  GroupTfEvents(&space);
  XPlaneVisitor host_plane_visitor = CreateTfXPlaneVisitor(host_plane);
  const XEvent& cpu_tf_op = host_plane->lines(1).events(2);
  EXPECT_EQ(cpu_tf_op.stats_size(), 2);
  EXPECT_EQ(host_plane_visitor.GetStatType(cpu_tf_op.stats(1).metadata_id()),
            StatType::kIsEager);
  EXPECT_EQ(cpu_tf_op.stats(1).int64_value(), 0);
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(device_plane);
  const XEvent& gpu_kernel = device_plane->lines(0).events(0);
  EXPECT_EQ(gpu_kernel.stats_size(), 3);
  EXPECT_EQ(device_plane_visitor.GetStatType(gpu_kernel.stats(2).metadata_id()),
            StatType::kIsEager);
  EXPECT_EQ(gpu_kernel.stats(2).int64_value(), 0);
}

TEST(GroupEventsTest, SemanticArgTest) {
  constexpr int64_t kIsRoot = 1;
  constexpr int64_t kStepNum = 100;
  constexpr int64_t kContextType = 123;
  constexpr uint64 kContextId = 456;

  XSpace raw_space;
  XPlane* raw_plane = raw_space.add_planes();
  XPlaneBuilder plane(raw_plane);
  plane.ReserveLines(2);
  auto root_producer = plane.GetOrCreateLine(0);
  CreateXEvent(&plane, &root_producer, HostEventType::kTraceContext, 0, 100,
               {{StatType::kIsRoot, kIsRoot}, {StatType::kStepNum, kStepNum}});
  CreateXEvent(&plane, &root_producer, HostEventType::kFunctionRun, 10, 90,
               {{StatType::kProducerType, kContextType},
                {StatType::kProducerId, kContextId}});
  auto consumer = plane.GetOrCreateLine(1);
  CreateXEvent(&plane, &consumer, HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kConsumerType, kContextType},
                {StatType::kConsumerId, kContextId}});

  GroupTfEvents(&raw_space);
  int num_events = 0;
  CreateTfXPlaneVisitor(raw_plane).ForEachLine([&](const XLineVisitor& line) {
    num_events += line.NumEvents();
    line.ForEachEvent([&](const XEventVisitor& event) {
      std::optional<int64_t> group_id;
      if (std::optional<XStatVisitor> stat =
              event.GetStat(StatType::kGroupId)) {
        group_id = stat->IntValue();
      }
      EXPECT_TRUE(group_id.has_value());
      EXPECT_EQ(*group_id, 0);
    });
  });
  EXPECT_EQ(num_events, 3);
}

TEST(GroupEventsTest, SemanticIntArgNoMatchTest) {
  constexpr int64_t kIsRoot = 1;
  constexpr int64_t kStepNum = 100;
  constexpr int64_t kContextType = 123;
  constexpr uint64 kProducerId = 456;
  constexpr uint64 kConsumerId = 789;

  XSpace raw_space;
  XPlane* raw_plane = raw_space.add_planes();
  XPlaneBuilder plane(raw_plane);
  plane.ReserveLines(2);
  auto root_producer = plane.GetOrCreateLine(0);
  CreateXEvent(&plane, &root_producer, HostEventType::kTraceContext, 0, 100,
               {{StatType::kIsRoot, kIsRoot}, {StatType::kStepNum, kStepNum}});
  CreateXEvent(&plane, &root_producer, HostEventType::kFunctionRun, 10, 90,
               {{StatType::kProducerType, kContextType},
                {StatType::kProducerId, kProducerId}});
  auto consumer = plane.GetOrCreateLine(1);
  CreateXEvent(&plane, &consumer, HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kConsumerType, kContextType},
                {StatType::kConsumerId, kConsumerId}});

  GroupTfEvents(&raw_space);
  int num_events = 0;
  CreateTfXPlaneVisitor(raw_plane).ForEachLine([&](const XLineVisitor& line) {
    num_events += line.NumEvents();
    line.ForEachEvent([&](const XEventVisitor& event) {
      std::optional<int64_t> group_id;
      if (std::optional<XStatVisitor> stat =
              event.GetStat(StatType::kGroupId)) {
        group_id = stat->IntValue();
      }
      if (event.Type() == HostEventType::kExecutorStateProcess) {
        EXPECT_FALSE(group_id.has_value());
      } else {
        EXPECT_TRUE(group_id.has_value());
        EXPECT_EQ(*group_id, 0);
      }
    });
  });
  EXPECT_EQ(num_events, 3);
}

TEST(GroupEventsTest, SemanticUintArgNoMatchTest) {
  constexpr int64_t kIsRoot = 1;
  constexpr int64_t kStepNum = 100;
  constexpr int64_t kContextType = 123;
  constexpr uint64 kProducerId = UINT64_MAX;
  constexpr uint64 kConsumerId = UINT64_MAX - 1;

  XSpace raw_space;
  XPlane* raw_plane = raw_space.add_planes();
  XPlaneBuilder plane(raw_plane);
  plane.ReserveLines(2);
  auto root_producer = plane.GetOrCreateLine(0);
  CreateXEvent(&plane, &root_producer, HostEventType::kTraceContext, 0, 100,
               {{StatType::kIsRoot, kIsRoot}, {StatType::kStepNum, kStepNum}});
  CreateXEvent(&plane, &root_producer, HostEventType::kFunctionRun, 10, 90,
               {{StatType::kProducerType, kContextType},
                {StatType::kProducerId, kProducerId}});
  auto consumer = plane.GetOrCreateLine(1);
  CreateXEvent(&plane, &consumer, HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kConsumerType, kContextType},
                {StatType::kConsumerId, kConsumerId}});

  GroupTfEvents(&raw_space);
  int num_events = 0;
  CreateTfXPlaneVisitor(raw_plane).ForEachLine([&](const XLineVisitor& line) {
    num_events += line.NumEvents();
    line.ForEachEvent([&](const XEventVisitor& event) {
      std::optional<int64_t> group_id;
      if (std::optional<XStatVisitor> stat =
              event.GetStat(StatType::kGroupId)) {
        group_id = stat->IntValue();
      }
      if (event.Type() == HostEventType::kExecutorStateProcess) {
        EXPECT_FALSE(group_id.has_value());
      } else {
        EXPECT_TRUE(group_id.has_value());
        EXPECT_EQ(*group_id, 0);
      }
    });
  });
  EXPECT_EQ(num_events, 3);
}

TEST(GroupEventsTest, AsyncEventTest) {
  constexpr int64_t kIsRoot = 1;
  constexpr int64_t kIsAsync = 1;
  constexpr absl::string_view kParent = "parent";
  constexpr absl::string_view kAsync = "async";
  constexpr absl::string_view kChild = "child";

  XSpace raw_space;
  XPlane* raw_plane = raw_space.add_planes();
  XPlaneBuilder plane(raw_plane);
  plane.ReserveLines(1);
  auto line = plane.GetOrCreateLine(0);
  CreateXEvent(&plane, &line, kParent, 0, 100, {{StatType::kIsRoot, kIsRoot}});
  CreateXEvent(&plane, &line, kAsync, 10, 200,
               {{StatType::kIsAsync, kIsAsync}});
  CreateXEvent(&plane, &line, kChild, 20, 80);

  GroupTfEvents(&raw_space);
  CreateTfXPlaneVisitor(raw_plane).ForEachLine([&](const XLineVisitor& line) {
    EXPECT_EQ(line.NumEvents(), 3);
    line.ForEachEvent([&](const XEventVisitor& event) {
      std::optional<int64_t> group_id;
      if (std::optional<XStatVisitor> stat =
              event.GetStat(StatType::kGroupId)) {
        group_id = stat->IntValue();
      }
      if (event.Name() == kAsync) {
        EXPECT_FALSE(group_id.has_value());
      } else {
        EXPECT_TRUE(group_id.has_value());
        EXPECT_EQ(*group_id, 0);
      }
    });
  });
}

TEST(GroupEventsTest, BatchingSessionTest) {
  constexpr absl::string_view kSchedule = "Schedule";
  constexpr int64_t kBatchContextType =
      static_cast<int64_t>(ContextType::kSharedBatchScheduler);
  constexpr int64_t kBatchContextId = 123;
  constexpr int64_t kBatchingSessionRunRootLevel = 1;
  constexpr int64_t kProcessBatchRootLevel = 2;

  XSpace raw_space;
  XPlane* raw_plane = raw_space.add_planes();
  XPlaneBuilder plane(raw_plane);
  plane.ReserveLines(2);
  auto request_thread = plane.GetOrCreateLine(0);
  // First request.
  CreateXEvent(&plane, &request_thread, HostEventType::kBatchingSessionRun, 0,
               100, {{StatType::kIsRoot, kBatchingSessionRunRootLevel}});
  CreateXEvent(&plane, &request_thread, kSchedule, 0, 100,
               {{StatType::kProducerType, kBatchContextType},
                {StatType::kProducerId, kBatchContextId}});
  // Second request.
  CreateXEvent(&plane, &request_thread, HostEventType::kBatchingSessionRun, 200,
               100, {{StatType::kIsRoot, kBatchingSessionRunRootLevel}});
  CreateXEvent(&plane, &request_thread, kSchedule, 200, 100,
               {{StatType::kProducerType, kBatchContextType},
                {StatType::kProducerId, kBatchContextId}});
  auto batch_thread = plane.GetOrCreateLine(1);
  CreateXEvent(&plane, &batch_thread, HostEventType::kProcessBatch, 200, 100,
               {{StatType::kConsumerType, kBatchContextType},
                {StatType::kConsumerId, kBatchContextId},
                {StatType::kIsRoot, kProcessBatchRootLevel}});

  EventForest event_forest;
  GroupTfEvents(&raw_space, &event_forest);
  const GroupMetadataMap& group_metadata_map =
      event_forest.GetGroupMetadataMap();
  EXPECT_EQ(group_metadata_map.size(), 3);
  // Check that the ProcessBatch group has two BatchingSessionRun groups as
  // parents.
  EXPECT_EQ(group_metadata_map.at(0).parents.size(), 2);
  // Check that the BatchingSessionRun groups have one ProcessBatch group as a
  // child.
  EXPECT_EQ(group_metadata_map.at(1).children.size(), 1);
  EXPECT_EQ(group_metadata_map.at(2).children.size(), 1);
  // Check that the events have the selected_group_ids stat set.
  uint64 num_checked = 0;
  CreateTfXPlaneVisitor(raw_plane).ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      std::optional<int64_t> group_id;
      if (std::optional<XStatVisitor> stat =
              event.GetStat(StatType::kGroupId)) {
        group_id = stat->IntValue();
      }
      EXPECT_TRUE(group_id.has_value());
      if (line.Id() == 0 &&
          event.Type() == HostEventType::kBatchingSessionRun) {
        ++num_checked;
      } else if (line.Id() == 1 &&
                 event.Type() == HostEventType::kProcessBatch) {
        ++num_checked;
      }
    });
  });
  EXPECT_EQ(num_checked, 3);
}

TEST(GroupTPUEventsTest, TpuExecuteOpTest) {
  tensorflow::profiler::XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(1);
  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  // When there is a TF loop, events are grouped per TF loop iteration.
  CreateXEvent(
      &host_plane_builder, &main_thread, HostEventType::kExecutorStateProcess,
      20, 50,
      {{StatType::kStepId, int64_t{123}}, {StatType::kIterNum, int64_t{456}}});
  EventForest event_forest;
  GroupTpuEventsOSS(&space, {}, &event_forest);
  EXPECT_EQ(event_forest.GetGroupMetadataMap().size(), 1);
  XPlaneVisitor host_plane_visitor = CreateTfXPlaneVisitor(&space.planes(0));
  host_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      // All events should be grouped and have `group_id` set.
      EXPECT_TRUE(event.GetStat(StatType::kGroupId).has_value());
    });
  });
}

TEST(GroupTPUEventsTest, TpuRequestTest) {
  tensorflow::profiler::XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(1);
  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kSessionRun, 0,
               100, {{StatType::kIsRoot, int64_t{1}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               GetHostEventTypeStr(HostEventType::kEnqueueRequestLocked), 20,
               50,
               {{StatType::kQueueAddr, int64_t{123}},
                {StatType::kRequestId, int64_t{456}}});
  EventForest event_forest;
  GroupTpuEventsOSS(&space, {}, &event_forest);
  EXPECT_EQ(event_forest.GetGroupMetadataMap().size(), 1);
  XPlaneVisitor host_plane_visitor = CreateTfXPlaneVisitor(&space.planes(0));
  host_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      // All events should be grouped and have `group_id` set.
      EXPECT_TRUE(event.GetStat(StatType::kGroupId).has_value());
    });
  });
}

TEST(GroupTPUEventsTest, TpuProgramCallbackTest) {
  tensorflow::profiler::XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(1);
  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kSessionRun, 0,
               100, {{StatType::kIsRoot, int64_t{1}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               GetHostEventTypeStr(HostEventType::kDoEnqueueProgram), 20, 50,
               {{StatType::kRunId, int64_t{123}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{1}}});
  EventForest event_forest;
  GroupTpuEventsOSS(&space, {}, &event_forest);
  EXPECT_EQ(event_forest.GetGroupMetadataMap().size(), 1);
  XPlaneVisitor host_plane_visitor = CreateTfXPlaneVisitor(&space.planes(0));
  host_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      // All events should be grouped and have `group_id` set.
      EXPECT_TRUE(event.GetStat(StatType::kGroupId).has_value());
    });
  });
}

TEST(GroupTPUEventsTest, ModuleRootEventTest) {
  tensorflow::profiler::XSpace space;
  tensorflow::profiler::XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);
  auto step_line = device_plane_builder.GetOrCreateLine(0);
  step_line.SetName("Steps");
  CreateXEvent(&device_plane_builder, &step_line, "1", 100, 200,
               {{StatType::kStepNum, int64_t{1}}});
  auto module_line = device_plane_builder.GetOrCreateLine(1);
  module_line.SetName("XLA Modules");
  CreateXEvent(&device_plane_builder, &module_line, "module", 105, 199,
               {{StatType::kRunId, int64_t{123}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{1}}});
  auto hlo_line = device_plane_builder.GetOrCreateLine(2);
  hlo_line.SetName("XLA Ops");
  CreateXEvent(&device_plane_builder, &hlo_line, "matmul", 110, 190, {});
  EventForest event_forest;
  GroupTpuEventsOSS(&space, {device_plane}, &event_forest);
  XPlaneVisitor device_plane_visitor = CreateTfXPlaneVisitor(&space.planes(0));
  device_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(line.Name(), " ", event.Name()));
      // All events should be grouped and have `group_id` set.
      EXPECT_TRUE(event.GetStat(StatType::kGroupId).has_value());
    });
  });
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
