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
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/preprocess_xplane.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/lib/context_types.h"
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
  constexpr uint64_t kContextId = 456;

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
  constexpr uint64_t kProducerId = 456;
  constexpr uint64_t kConsumerId = 789;

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
  constexpr uint64_t kProducerId = UINT64_MAX;
  constexpr uint64_t kConsumerId = UINT64_MAX - 1;

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
  uint64_t num_checked = 0;
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
  tensorflow::profiler::XPlane* device_plane =
      GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0);
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(3);
  auto step_line = device_plane_builder.GetOrCreateLine(0);
  step_line.SetName("Steps");
  CreateXEvent(&device_plane_builder, &step_line, "1", 100, 200,
               {{StatType::kStepNum, int64_t{1}}});
  auto module_line = device_plane_builder.GetOrCreateLine(1);
  module_line.SetName("XLA Modules");
  CreateXEvent(&device_plane_builder, &module_line, "module", 105, 194,
               {{StatType::kRunId, int64_t{123}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{0}}});
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

TEST(GroupTPUEventsTest, MergeHostStepsTest) {
  XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(1);
  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  main_thread.SetName("main");
  CreateXEvent(
      &host_plane_builder, &main_thread, "train", 100, 10,
      {{StatType::kStepNum, int64_t{1}}, {StatType::kIsRoot, int64_t{1}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kDoEnqueueProgram, 100, 1,
               {{StatType::kRunId, int64_t{2}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{0}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kDoEnqueueProgram, 101, 2,
               {{StatType::kRunId, int64_t{3}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{0}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kDoEnqueueProgram, 103, 2,
               {{StatType::kRunId, int64_t{4}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{0}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kDoEnqueueProgram, 105, 4,
               {{StatType::kRunId, int64_t{5}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{0}}});
  XPlane* device_plane = GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0);
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);
  auto module_line = device_plane_builder.GetOrCreateLine(0);
  module_line.SetName(tsl::profiler::kXlaModuleLineName);
  CreateXEvent(
      &device_plane_builder, &module_line, "jit_something(1)", 1000, 10,
      {{StatType::kRunId, int64_t{2}}, {StatType::kQueueId, int64_t{0}}});
  CreateXEvent(
      &device_plane_builder, &module_line, "jit_something(1)", 1015, 100,
      {{StatType::kRunId, int64_t{3}}, {StatType::kQueueId, int64_t{0}}});
  CreateXEvent(
      &device_plane_builder, &module_line, "jit_something(1)", 1125, 50,
      {{StatType::kRunId, int64_t{4}}, {StatType::kQueueId, int64_t{0}}});
  CreateXEvent(
      &device_plane_builder, &module_line, "jit_something(1)", 1180, 25,
      {{StatType::kRunId, int64_t{5}}, {StatType::kQueueId, int64_t{0}}});
  auto step_line = device_plane_builder.GetOrCreateLine(1);
  step_line.SetName(kStepLineName);
  CreateXEvent(&device_plane_builder, &step_line, "0", 1000, 10,
               {{StatType::kDeviceOffsetPs, int64_t{1000}},
                {StatType::kDeviceDurationPs, int64_t{10}}});
  CreateXEvent(&device_plane_builder, &step_line, "1", 1015, 100,
               {{StatType::kDeviceOffsetPs, int64_t{1015}},
                {StatType::kDeviceDurationPs, int64_t{100}}});
  CreateXEvent(&device_plane_builder, &step_line, "2", 1125, 50,
               {{StatType::kDeviceOffsetPs, int64_t{1125}},
                {StatType::kDeviceDurationPs, int64_t{50}}});
  CreateXEvent(&device_plane_builder, &step_line, "3", 1180, 25,
               {{StatType::kDeviceOffsetPs, int64_t{1180}},
                {StatType::kDeviceDurationPs, int64_t{25}}});
  auto op_line = device_plane_builder.GetOrCreateLine(2);
  op_line.SetName(kXlaOpLineName);
  CreateXEvent(&device_plane_builder, &op_line, "offload.start.1", 1000, 5,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{0}},
                {StatType::kProducerId, int64_t{1}},
                {StatType::kProducerType,
                 static_cast<int64_t>(ContextType::kScOffload)}});
  CreateXEvent(&device_plane_builder, &op_line, "offload.done.1", 1005, 5, {});
  CreateXEvent(&device_plane_builder, &op_line, "offload.start.1", 1015, 5,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{1}},
                {StatType::kProducerId, int64_t{2}},
                {StatType::kProducerType,
                 static_cast<int64_t>(ContextType::kScOffload)}});
  CreateXEvent(&device_plane_builder, &op_line, "offload.done.1", 1020, 95, {});
  CreateXEvent(&device_plane_builder, &op_line, "offload.start.1", 1125, 5,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{2}},
                {StatType::kProducerId, int64_t{3}},
                {StatType::kProducerType,
                 static_cast<int64_t>(ContextType::kScOffload)}});
  CreateXEvent(&device_plane_builder, &op_line, "offload.done.1", 1130, 45, {});
  CreateXEvent(&device_plane_builder, &op_line, "offload.start.1", 1180, 5,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{3}},
                {StatType::kProducerId, int64_t{4}},
                {StatType::kProducerType,
                 static_cast<int64_t>(ContextType::kScOffload)}});
  CreateXEvent(&device_plane_builder, &op_line, "offload.done.1", 1185, 20, {});

  // TPU SparseCore Plane (device_id 0, core_type 1)
  XPlane* sparsecore_plane = GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0, 0);
  XPlaneBuilder sc_plane_builder(sparsecore_plane);
  sc_plane_builder.ReserveLines(3);

  auto sc_module_line = sc_plane_builder.GetOrCreateLine(0);
  sc_module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &sc_module_line, "offloaded(1)", 1001, 8,
               {
                   {StatType::kTcOffloadStartId, int64_t{1}},
               });
  CreateXEvent(&sc_plane_builder, &sc_module_line, "offloaded(1)", 1016, 98,
               {
                   {StatType::kTcOffloadStartId, int64_t{1}},
               });
  CreateXEvent(&sc_plane_builder, &sc_module_line, "offloaded(1)", 1126, 48,
               {
                   {StatType::kTcOffloadStartId, int64_t{1}},
               });
  CreateXEvent(&sc_plane_builder, &sc_module_line, "offloaded(1)", 1181, 23,
               {
                   {StatType::kTcOffloadStartId, int64_t{1}},
               });

  auto sc_step_line = sc_plane_builder.GetOrCreateLine(1);
  sc_step_line.SetName(kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "sc step 0", 1000, 10, {});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "sc step 1", 1015, 100, {});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "sc step 2", 1125, 50, {});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "sc step 3", 1180, 25, {});

  auto sc_op_line = sc_plane_builder.GetOrCreateLine(2);
  sc_op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(
      &sc_plane_builder, &sc_op_line, "sc_op_1", 1001, 8,
      {{StatType::kConsumerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kConsumerId, int64_t{1}}});
  CreateXEvent(
      &sc_plane_builder, &sc_op_line, "sc_op_1", 1016, 98,
      {{StatType::kConsumerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kConsumerId, int64_t{2}}});
  CreateXEvent(
      &sc_plane_builder, &sc_op_line, "sc_op_1", 1126, 48,
      {{StatType::kConsumerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kConsumerId, int64_t{3}}});
  CreateXEvent(
      &sc_plane_builder, &sc_op_line, "sc_op_1", 1181, 23,
      {{StatType::kConsumerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kConsumerId, int64_t{4}}});

  // Make sure to preprocess so that the Runtime events have a Producer/Consumer
  // event set created.
  PreprocessXSpace(&space);
  EventForest event_forest;
  GroupTpuEventsOSS(&space, {device_plane, sparsecore_plane}, &event_forest);
  EXPECT_EQ(event_forest.GetGroupMetadataMap().size(), 1);
  auto visitor = CreateTfXPlaneVisitor(device_plane);
  bool step_line_found = false;
  visitor.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() != kStepLineName) {
      return;
    }
    step_line_found = true;
    EXPECT_EQ(line.NumEvents(), 1);
    auto step_event = line.GetFirstEvent();
    EXPECT_EQ(step_event.GetTimespan().begin_ps(), 1000);
    EXPECT_EQ(step_event.GetTimespan().end_ps(), 1205);
    EXPECT_EQ(GetDeviceEventTimespan(step_event).begin_ps(), 1000);
    EXPECT_EQ(GetDeviceEventTimespan(step_event).end_ps(), 1205);
  });
  EXPECT_TRUE(step_line_found);

  auto sc_visitor = CreateTfXPlaneVisitor(sparsecore_plane);
  bool sc_step_line_found = false;
  sc_visitor.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() != kSparseCoreStepLineName) {
      return;
    }
    sc_step_line_found = true;
    EXPECT_EQ(line.NumEvents(), 1);
    auto step_event = line.GetFirstEvent();
    EXPECT_EQ(step_event.GetTimespan().begin_ps(), 1000);
    EXPECT_EQ(step_event.GetTimespan().end_ps(), 1205);
    EXPECT_EQ(GetDeviceEventTimespan(step_event).begin_ps(), 1000);
    EXPECT_EQ(GetDeviceEventTimespan(step_event).end_ps(), 1205);
  });
  EXPECT_TRUE(sc_step_line_found);
}

TEST(GroupTPUEventsTest, MergeOffloadedScSteps) {
  tensorflow::profiler::XSpace space;
  // No host plane in this test.

  // TPU TensorCore Plane (device_id 0)
  XPlane* tensorcore_plane = GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0);
  XPlaneBuilder tc_plane_builder(tensorcore_plane);
  tc_plane_builder.ReserveLines(3);

  auto tc_module_line = tc_plane_builder.GetOrCreateLine(0);
  tc_module_line.SetName(kXlaModuleLineName);
  // The module event is strictly within the step event (1000-2000).
  CreateXEvent(&tc_plane_builder, &tc_module_line, "jit_tc_module", 1010, 980,
               {{StatType::kRunId, int64_t{1}}});

  auto tc_step_line = tc_plane_builder.GetOrCreateLine(1);
  tc_step_line.SetName(kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "tc step 0", 1000, 1000,
               {{StatType::kDeviceOffsetPs, int64_t{1000}},
                {StatType::kDeviceDurationPs, int64_t{1000}}});

  auto tc_op_line = tc_plane_builder.GetOrCreateLine(2);
  tc_op_line.SetName(kXlaOpLineName);
  // First offload
  CreateXEvent(&tc_plane_builder, &tc_op_line, "offload.start.1", 1050, 50,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{0}},
                {StatType::kProducerId, int64_t{1}},
                {StatType::kProducerType,
                 static_cast<int64_t>(ContextType::kScOffload)}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "offload.done.1", 1100, 400, {});
  // Second offload
  CreateXEvent(&tc_plane_builder, &tc_op_line, "offload.start.1", 1550, 50,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{1}},
                {StatType::kProducerId, int64_t{2}},
                {StatType::kProducerType,
                 static_cast<int64_t>(ContextType::kScOffload)}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "offload.done.1", 1600, 400, {});

  // TPU SparseCore Plane (device_id 0, core_type 1)
  XPlane* sparsecore_plane = GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0, 1);
  XPlaneBuilder sc_plane_builder(sparsecore_plane);
  sc_plane_builder.ReserveLines(3);

  auto sc_module_line = sc_plane_builder.GetOrCreateLine(0);
  sc_module_line.SetName(kSparseCoreModuleLineName);
  // These module events are strictly within their respective step events.
  CreateXEvent(&sc_plane_builder, &sc_module_line, "offloaded(1)", 1101, 398,
               {{StatType::kTcOffloadStartId, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_module_line, "offloaded(1)", 1601, 398,
               {{StatType::kTcOffloadStartId, int64_t{1}}});

  auto sc_step_line = sc_plane_builder.GetOrCreateLine(1);
  sc_step_line.SetName(kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "sc step 0", 1100, 400, {});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "sc step 1", 1600, 400, {});

  auto sc_op_line = sc_plane_builder.GetOrCreateLine(2);
  sc_op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(
      &sc_plane_builder, &sc_op_line, "sc_op_1a", 1110, 100,
      {{StatType::kConsumerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kConsumerId, int64_t{1}}});
  CreateXEvent(
      &sc_plane_builder, &sc_op_line, "sc_op_2a", 1610, 100,
      {{StatType::kConsumerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kConsumerId, int64_t{2}}});

  // Make sure to preprocess so that the Runtime events have a Producer/Consumer
  // event set created.
  PreprocessXSpace(&space);
  EventForest event_forest;
  GroupTpuEventsOSS(&space, {tensorcore_plane, sparsecore_plane},
                    &event_forest);

  // We expect only one group as all events are linked.
  const GroupMetadataMap& group_metadata_map =
      event_forest.GetGroupMetadataMap();
  EXPECT_EQ(group_metadata_map.size(), 1);
  const int64_t expected_group_id = group_metadata_map.begin()->first;

  // Check the merged TensorCore step event.
  auto tc_visitor = CreateTfXPlaneVisitor(tensorcore_plane);
  tc_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(tensorcore_plane->name(), ": ", line.Name(),
                                " ", event.Name()));
      std::optional<XStatVisitor> group_id_stat =
          event.GetStat(StatType::kGroupId);
      ASSERT_TRUE(group_id_stat.has_value());
      EXPECT_EQ(group_id_stat->IntOrUintValue(), expected_group_id);
    });
  });

  // Check the merged SparseCore step event.
  auto sc_visitor = CreateTfXPlaneVisitor(sparsecore_plane);
  sc_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(sparsecore_plane->name(), ": ", line.Name(),
                                " ", event.Name()));
      std::optional<XStatVisitor> group_id_stat =
          event.GetStat(StatType::kGroupId);
      ASSERT_TRUE(group_id_stat.has_value());
      EXPECT_EQ(group_id_stat->IntOrUintValue(), expected_group_id);
    });
    if (line.Name() == kSparseCoreStepLineName) {
      EXPECT_EQ(line.NumEvents(), 1);
      auto step_event = line.GetFirstEvent();
      EXPECT_EQ(step_event.GetTimespan().begin_ps(), 1100);
      EXPECT_EQ(step_event.GetTimespan().end_ps(), 2000);
    }
  });
}

TEST(GroupTPUEventsTest, GroupOffloadedSparseCoreModulesHostLoopTest) {
  tensorflow::profiler::XSpace space;
  tensorflow::profiler::XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.ReserveLines(1);
  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  main_thread.SetName("main");

  CreateXEvent(&host_plane_builder, &main_thread, "host step 0", 0, 200,
               {{StatType::kIsRoot, int64_t{1}}});
  // Host event for TensorCore.
  CreateXEvent(&host_plane_builder, &main_thread, "DoEnqueueProgram", 100, 10,
               {{StatType::kRunId, int64_t{1}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kReplicaId, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{0}},
                {StatType::kCoreType, int64_t{0}}});  // kTpuTensorCore

  // TPU TensorCore Plane (device_id 0)
  XPlane* tensorcore_plane = GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0);
  XPlaneBuilder tc_plane_builder(tensorcore_plane);
  tc_plane_builder.ReserveLines(3);

  auto tc_module_line = tc_plane_builder.GetOrCreateLine(0);
  tc_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&tc_plane_builder, &tc_module_line, "jit(123)", 1000, 1000,
               {{StatType::kRunId, int64_t{1}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kReplicaId, int64_t{0}},
                {StatType::kCoreType, int64_t{0}}});

  auto tc_step_line = tc_plane_builder.GetOrCreateLine(1);
  tc_step_line.SetName(kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "tc step 0", 1000, 1000, {});

  auto tc_op_line = tc_plane_builder.GetOrCreateLine(2);
  tc_op_line.SetName(kXlaOpLineName);
  CreateXEvent(
      &tc_plane_builder, &tc_op_line, "offload_start", 1050, 100,
      {{StatType::kTcOffloadStartId, int64_t{123}},
       {StatType::kOffloadCoreId, int64_t{0}},
       {StatType::kOffloadExecutionIndex, int64_t{0}},
       {StatType::kProducerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kProducerId, int64_t{1}}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "offload_done", 1200, 750, {});

  // TPU SparseCore Plane (device_id 1)
  XPlane* sparsecore_plane = GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0, 0);
  XPlaneBuilder sc_plane_builder(sparsecore_plane);
  sc_plane_builder.ReserveLines(3);

  auto sc_module_line = sc_plane_builder.GetOrCreateLine(0);
  sc_module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &sc_module_line, "offloaded(123)", 1100, 800,
               {{StatType::kTcOffloadStartId, int64_t{123}}});

  auto sc_step_line = sc_plane_builder.GetOrCreateLine(1);
  sc_step_line.SetName(kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "sc step 0", 1100, 800, {});

  auto sc_op_line = sc_plane_builder.GetOrCreateLine(2);
  sc_op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(
      &sc_plane_builder, &sc_op_line, "offloaded_start.copy", 1100, 10,
      {{StatType::kConsumerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kConsumerId, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "offloaded_done.copy", 1120, 180,
               {});

  // Preprocess to create Producer/Consumer events.
  PreprocessXSpace(&space);
  EventForest event_forest;
  GroupTpuEventsOSS(&space, {tensorcore_plane, sparsecore_plane},
                    &event_forest);

  // We expect one group, where all events are grouped under the same group.
  EXPECT_EQ(event_forest.GetGroupMetadataMap().size(), 1);
  const int64_t expected_group_id =
      event_forest.GetGroupMetadataMap().begin()->first;

  // Check Host events.
  XPlaneVisitor host_visitor = CreateTfXPlaneVisitor(host_plane);
  int host_event_idx = 0;
  host_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(host_plane->name(), ": ", line.Name(), " ",
                                event.Name()));
      std::optional<XStatVisitor> group_id_stat =
          event.GetStat(StatType::kGroupId);
      ASSERT_TRUE(group_id_stat.has_value());
      EXPECT_EQ(group_id_stat->IntValue(), expected_group_id);
      host_event_idx++;
    });
  });
  EXPECT_EQ(host_event_idx, 2);

  // Check TensorCore events.
  XPlaneVisitor tc_visitor = CreateTfXPlaneVisitor(tensorcore_plane);
  tc_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(tensorcore_plane->name(), ": ",

                                line.Name(), " ", event.Name()));
      std::optional<XStatVisitor> group_id_stat =
          event.GetStat(StatType::kGroupId);
      ASSERT_TRUE(group_id_stat.has_value());
      // TensorCore events are associated with run_id 1, likely getting group_id
      // 0.
      EXPECT_EQ(group_id_stat->IntValue(), expected_group_id);
    });
  });

  // Check SparseCore events.
  XPlaneVisitor sc_visitor = CreateTfXPlaneVisitor(sparsecore_plane);
  sc_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(
          absl::StrCat(sparsecore_plane->name(), ": ",
                       ParseDeviceOrdinal(sparsecore_plane->name()).value(),
                       " ", line.Name(), " ", event.Name()));
      std::optional<XStatVisitor> group_id_stat =
          event.GetStat(StatType::kGroupId);
      ASSERT_TRUE(group_id_stat.has_value());
      // SparseCore events are associated with run_id 2, likely getting
      // group_id 1.
      EXPECT_EQ(group_id_stat->IntValue(), expected_group_id);
    });
  });
}

TEST(GroupTPUEventsTest, GroupOffloadedSparseCoreModulesDeviceLoopTest) {
  tensorflow::profiler::XSpace space;
  tensorflow::profiler::XPlane* host_plane = GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.ReserveLines(2);
  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  main_thread.SetName("main");

  // Tf Loop event
  CreateXEvent(
      &host_plane_builder, &main_thread, HostEventType::kExecutorStateProcess,
      100, 10,
      {{StatType::kStepId, int64_t{1}}, {StatType::kIterNum, int64_t{99}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               HostEventType::kTpuSystemExecute, 100, 9,
               {{StatType::kProducerType,
                 static_cast<int64_t>(ContextType::kTfrtTpuRuntime)},
                {StatType::kProducerId, int64_t{1}}});

  auto enqueue_thread = host_plane_builder.GetOrCreateLine(1);
  enqueue_thread.SetName("tf_enqueue");
  CreateXEvent(&host_plane_builder, &enqueue_thread,
               "tpu::System::Execute=>IssueSequencedEvent", 102, 10,
               {{StatType::kConsumerType,
                 static_cast<int64_t>(ContextType::kTfrtTpuRuntime)},
                {StatType::kConsumerId, int64_t{1}}});
  CreateXEvent(&host_plane_builder, &enqueue_thread,
               HostEventType::kDoEnqueueProgram, 103, 8,
               {{StatType::kRunId, int64_t{1}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kCoreType, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{0}}});

  // TPU TensorCore Plane (device_id 0)
  XPlane* tensorcore_plane = GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0);
  XPlaneBuilder tc_plane_builder(tensorcore_plane);
  tc_plane_builder.ReserveLines(3);

  auto tc_module_line = tc_plane_builder.GetOrCreateLine(0);
  tc_module_line.SetName(kXlaModuleLineName);
  // The module event encompasses the step event's time range (1000-2000).
  CreateXEvent(&tc_plane_builder, &tc_module_line, "jit(123)", 900, 1200,
               {{StatType::kRunId, int64_t{1}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kReplicaId, int64_t{0}},
                {StatType::kCoreType, int64_t{0}}});

  auto tc_step_line = tc_plane_builder.GetOrCreateLine(1);
  tc_step_line.SetName(kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "tc step 0", 1000, 1000, {});

  auto tc_op_line = tc_plane_builder.GetOrCreateLine(2);
  tc_op_line.SetName(kXlaOpLineName);
  CreateXEvent(
      &tc_plane_builder, &tc_op_line, "offload_start", 1050, 100,
      {{StatType::kTcOffloadStartId, int64_t{123}},
       {StatType::kOffloadCoreId, int64_t{0}},
       {StatType::kOffloadExecutionIndex, int64_t{0}},
       {StatType::kProducerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kProducerId, int64_t{1}}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "offload_done", 1200, 750, {});

  // TPU SparseCore Plane (device_id 1)
  XPlane* sparsecore_plane = GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0, 0);
  XPlaneBuilder sc_plane_builder(sparsecore_plane);
  sc_plane_builder.ReserveLines(3);

  auto sc_module_line = sc_plane_builder.GetOrCreateLine(0);
  sc_module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &sc_module_line, "offloaded(123)", 1100, 800,
               {{StatType::kTcOffloadStartId, int64_t{123}}});

  auto sc_step_line = sc_plane_builder.GetOrCreateLine(1);
  sc_step_line.SetName(kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "sc step 0", 1100, 800, {});

  auto sc_op_line = sc_plane_builder.GetOrCreateLine(2);
  sc_op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(
      &sc_plane_builder, &sc_op_line, "offloaded_start.copy", 1100, 100,
      {{StatType::kConsumerType, static_cast<int64_t>(ContextType::kScOffload)},
       {StatType::kConsumerId, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "offloaded_done.copy", 1300, 100,
               {});

  // Preprocess to create Producer/Consumer events.
  PreprocessXSpace(&space);
  EventForest event_forest;
  GroupTpuEventsOSS(&space, {tensorcore_plane, sparsecore_plane},
                    &event_forest);

  // We expect two groups, one for the host events and one for the device
  // events.
  EXPECT_EQ(event_forest.GetGroupMetadataMap().size(), 2);

  // Check Host events.
  XPlaneVisitor host_visitor = CreateTfXPlaneVisitor(host_plane);
  int host_event_idx = 0;
  host_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(host_plane->name(), ": ", line.Name(), " ",
                                event.Name()));
      std::optional<XStatVisitor> group_id_stat =
          event.GetStat(StatType::kGroupId);
      ASSERT_TRUE(group_id_stat.has_value());
      EXPECT_EQ(group_id_stat->IntValue(), 0);
      host_event_idx++;
    });
  });
  EXPECT_EQ(host_event_idx, 4);

  // Check TensorCore events.
  XPlaneVisitor tc_visitor = CreateTfXPlaneVisitor(tensorcore_plane);
  tc_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (line.Name() == kXlaModuleLineName) {
        // The module event encompasses multiple steps, so it cannot be grouped.
        return;
      }
      SCOPED_TRACE(absl::StrCat(tensorcore_plane->name(), ": ",

                                line.Name(), " ", event.Name()));
      std::optional<XStatVisitor> group_id_stat =
          event.GetStat(StatType::kGroupId);
      ASSERT_TRUE(group_id_stat.has_value());
      EXPECT_EQ(group_id_stat->IntValue(), 1);
    });
  });

  // Check SparseCore events.
  XPlaneVisitor sc_visitor = CreateTfXPlaneVisitor(sparsecore_plane);
  sc_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(
          absl::StrCat(sparsecore_plane->name(), ": ",
                       ParseDeviceOrdinal(sparsecore_plane->name()).value(),
                       " ", line.Name(), " ", event.Name()));
      std::optional<XStatVisitor> group_id_stat =
          event.GetStat(StatType::kGroupId);
      ASSERT_TRUE(group_id_stat.has_value());
      EXPECT_EQ(group_id_stat->IntValue(), 1);
    });
  });
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
