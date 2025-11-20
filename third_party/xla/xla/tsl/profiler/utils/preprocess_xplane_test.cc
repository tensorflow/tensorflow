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

#include "xla/tsl/profiler/utils/preprocess_xplane.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::tsl::profiler::CreateTfXPlaneVisitor;
using ::tsl::profiler::CreateXEvent;
using ::tsl::profiler::GetHostEventTypeStr;
using ::tsl::profiler::HostEventType;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XSpace;

TEST(PreprocessXPlane, IsRootStatsTest) {
  XSpace space;
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  plane_builder.ReserveLines(1);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder,
               GetHostEventTypeStr(HostEventType::kProcessBatch), 100, 100);
  CreateXEvent(&plane_builder, &line_builder,
               GetHostEventTypeStr(HostEventType::kBatchingSessionRun), 200,
               100);
  PreprocessXSpace(&space);
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      ASSERT_TRUE(event.GetStat(StatType::kIsRoot).has_value());
      int64_t is_root = event.GetStat(StatType::kIsRoot)->IntValue();
      if (event.Type() == HostEventType::kBatchingSessionRun) {
        EXPECT_EQ(is_root, 1);
      } else if (event.Type() == HostEventType::kProcessBatch) {
        EXPECT_EQ(is_root, 2);
      } else {
        CHECK(false);
      }
    });
  });
}

TEST(PreprocessXPlane, ProducerConsumerTest) {
  XSpace space;
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  plane_builder.ReserveLines(2);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(
      &plane_builder, &line_builder,
      GetHostEventTypeStr(HostEventType::kExecutorStateProcess), 100, 100,
      {{StatType::kStepId, int64_t{123}}, {StatType::kIterNum, int64_t{456}}});
  line_builder = plane_builder.GetOrCreateLine(1);
  CreateXEvent(
      &plane_builder, &line_builder,
      GetHostEventTypeStr(HostEventType::kTpuExecuteOp), 200, 100,
      {{StatType::kStepId, int64_t{123}}, {StatType::kIterNum, int64_t{456}}});
  PreprocessXSpace(&space);
  std::optional<uint64_t> producer_context_id, consumer_context_id;
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Type() == HostEventType::kExecutorStateProcess) {
        auto producer_type = event.GetStat(StatType::kProducerType);
        ASSERT_TRUE(producer_type.has_value());
        EXPECT_EQ(producer_type->IntValue(),
                  static_cast<int64_t>(ContextType::kLegacy));
        auto producer_id = event.GetStat(StatType::kProducerId);
        ASSERT_TRUE(producer_id.has_value());
        producer_context_id = producer_id->IntOrUintValue();
      } else if (event.Type() == HostEventType::kTpuExecuteOp) {
        auto consumer_type = event.GetStat(StatType::kConsumerType);
        ASSERT_TRUE(consumer_type.has_value());
        EXPECT_EQ(consumer_type->IntValue(),
                  static_cast<int64_t>(ContextType::kLegacy));
        auto consumer_id = event.GetStat(StatType::kConsumerId);
        ASSERT_TRUE(consumer_id.has_value());
        consumer_context_id = consumer_id->IntOrUintValue();
      } else {
        CHECK(false);
      }
    });
  });
  ASSERT_TRUE(producer_context_id && consumer_context_id);
  ASSERT_EQ(*producer_context_id, *consumer_context_id);
}

// Producer and consumer events are assigned different context ids if their
// context stats do not match, and will not be connected by the grouping code
// later.
TEST(PreprocessXPlane, ProducerConsumerNotMatchedTest) {
  XSpace space;
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  plane_builder.ReserveLines(2);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder,
               GetHostEventTypeStr(HostEventType::kExecutorStateProcess), 100,
               100,
               {{StatType::kStepId, int64_t{123}},
                {StatType::kIterNum, int64_t{456}},
                {StatType::kDeviceOrdinal, int64_t{789}}});
  line_builder = plane_builder.GetOrCreateLine(1);
  CreateXEvent(
      &plane_builder, &line_builder,
      GetHostEventTypeStr(HostEventType::kTpuExecuteOp), 200, 100,
      {{StatType::kStepId, int64_t{123}}, {StatType::kIterNum, int64_t{789}}});
  PreprocessXSpace(&space);
  std::optional<uint64_t> producer_context_id, consumer_context_id;
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Type() == HostEventType::kExecutorStateProcess) {
        auto producer_type = event.GetStat(StatType::kProducerType);
        ASSERT_TRUE(producer_type.has_value());
        EXPECT_EQ(producer_type->IntValue(),
                  static_cast<int64_t>(ContextType::kLegacy));
        auto producer_id = event.GetStat(StatType::kProducerId);
        ASSERT_TRUE(producer_id.has_value());
        producer_context_id = producer_id->IntOrUintValue();
      } else if (event.Type() == HostEventType::kTpuExecuteOp) {
        auto consumer_type = event.GetStat(StatType::kConsumerType);
        ASSERT_TRUE(consumer_type.has_value());
        EXPECT_EQ(consumer_type->IntValue(),
                  static_cast<int64_t>(ContextType::kLegacy));
        auto consumer_id = event.GetStat(StatType::kConsumerId);
        ASSERT_TRUE(consumer_id.has_value());
        consumer_context_id = consumer_id->IntOrUintValue();
      } else {
        CHECK(false);
      }
    });
  });
  ASSERT_TRUE(producer_context_id && consumer_context_id);
  ASSERT_NE(*producer_context_id, *consumer_context_id);
}

TEST(PreprocessXPlane, MissingLegacyStatTest) {
  XSpace space;
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  plane_builder.ReserveLines(2);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder,
               GetHostEventTypeStr(HostEventType::kExecutorStateProcess), 100,
               100, {{StatType::kStepId, int64_t{123}}});
  line_builder = plane_builder.GetOrCreateLine(1);
  CreateXEvent(&plane_builder, &line_builder,
               GetHostEventTypeStr(HostEventType::kTpuExecuteOp), 200, 100,
               {{StatType::kStepId, int64_t{123}}});
  PreprocessXSpace(&space);
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Type() == HostEventType::kExecutorStateProcess) {
        // Context stats should not be set if not all legacy context stats
        // exist.
        auto producer_type = event.GetStat(StatType::kProducerType);
        ASSERT_FALSE(producer_type.has_value());
        auto producer_id = event.GetStat(StatType::kProducerId);
        ASSERT_FALSE(producer_id.has_value());
      } else if (event.Type() == HostEventType::kTpuExecuteOp) {
        auto consumer_type = event.GetStat(StatType::kConsumerType);
        ASSERT_FALSE(consumer_type.has_value());
        auto consumer_id = event.GetStat(StatType::kConsumerId);
        ASSERT_FALSE(consumer_id.has_value());
      } else {
        CHECK(false);
      }
    });
  });
}

TEST(PreprocessXPlane, HostRunIdPreprocessorTest) {
  XSpace space;
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  plane_builder.ReserveLines(2);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  int64_t host_run_id = int64_t{582974244};
  int64_t device_run_id = int64_t{46103332};
  CreateXEvent(
      &plane_builder, &line_builder,
      GetHostEventTypeStr(HostEventType::kDoEnqueueContinuationProgram), 100,
      100, {});
  CreateXEvent(&plane_builder, &line_builder,
               GetHostEventTypeStr(HostEventType::kDoEnqueueProgram), 100, 100,
               {{StatType::kRunId, int64_t{host_run_id}}});
  CreateXEvent(&plane_builder, &line_builder,
               GetHostEventTypeStr(HostEventType::kTpuExecuteOp), 200, 100,
               {{StatType::kRunId, int64_t{device_run_id}}});
  CreateXEvent(&plane_builder, &line_builder,
               GetHostEventTypeStr(HostEventType::kCompleteCallbacks), 300, 100,
               {{StatType::kRunId, int64_t{host_run_id}}});
  line_builder = plane_builder.GetOrCreateLine(1);
  PreprocessXSpace(&space);
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Type() == HostEventType::kDoEnqueueContinuationProgram) {
        auto run_id = event.GetStat(StatType::kRunId);
        ASSERT_FALSE(run_id.has_value());
      } else if (event.Type() == HostEventType::kDoEnqueueProgram) {
        auto run_id = event.GetStat(StatType::kRunId);
        ASSERT_TRUE(run_id.has_value());
        ASSERT_EQ(run_id->IntValue(), device_run_id);
      } else if (event.Type() == HostEventType::kTpuExecuteOp) {
        auto run_id = event.GetStat(StatType::kRunId);
        ASSERT_TRUE(run_id.has_value());
        ASSERT_EQ(run_id->IntValue(), device_run_id);
      } else if (event.Type() == HostEventType::kCompleteCallbacks) {
        auto run_id = event.GetStat(StatType::kRunId);
        ASSERT_TRUE(run_id.has_value());
        ASSERT_EQ(run_id->IntValue(), device_run_id);
      } else {
        CHECK(false);
      }
    });
  });
}

TEST(PreprocessXPlane, ThreadPoolPreprocessorTest) {
  XSpace space;
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  auto main_line = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &main_line, kThreadpoolListenerRecord, 100, 100,
               {{StatType::kProducerType,
                 static_cast<int64_t>(ContextType::kThreadpoolEvent)},
                {StatType::kProducerId, int64_t{123}}});
  auto thread_pool_line = plane_builder.GetOrCreateLine(1);
  CreateXEvent(&plane_builder, &thread_pool_line,
               kThreadpoolListenerStartRegion, 200, 0,
               {{StatType::kConsumerType,
                 static_cast<int64_t>(ContextType::kThreadpoolEvent)},
                {StatType::kConsumerId, int64_t{123}}});
  CreateXEvent(&plane_builder, &thread_pool_line, kThreadpoolListenerStopRegion,
               300, 0,
               {{StatType::kConsumerType,
                 static_cast<int64_t>(ContextType::kThreadpoolEvent)},
                {StatType::kConsumerId, int64_t{123}}});

  bool new_event_added = false;
  PreprocessXSpace(&space);
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Name() == kThreadpoolListenerRegion) {
        new_event_added = true;
        EXPECT_EQ(event.DurationPs(), 100);
        EXPECT_EQ(event.TimestampPs(), 200);
        auto stat = event.GetStat(StatType::kConsumerId);
        EXPECT_TRUE(stat.has_value());
        EXPECT_EQ(stat->IntOrUintValue(), 123);
      }
    });
  });
  EXPECT_TRUE(new_event_added);
}

TEST(PreprocessXPlane, XContextStatsAccessorNPETest) {
  auto xplane = std::make_unique<XPlane>();
  XPlaneBuilder xplane_builder(xplane.get());
  XLine xline;
  XLineBuilder xline_builder(&xline, &xplane_builder);
  XEvent xevent;
  XEventBuilder xevent_builder(&xline, &xplane_builder, &xevent);
  XContextStatsAccessor<int64_t, StatType::kRunId> run_id_accessor;

  ASSERT_FALSE(run_id_accessor.Initialize(xplane_builder));
  EXPECT_EQ(run_id_accessor.GetStat(xevent_builder), std::nullopt);
}

TEST(PreprocessXPlane, SparseCoreOffloadingTest) {
  XSpace space;

  // --- TensorCore Plane ---
  XPlane* tensorcore_plane = GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0);
  XPlaneBuilder tc_plane_builder(tensorcore_plane);

  // kStepLineName
  auto tc_step_line = tc_plane_builder.GetOrCreateLine(0);
  tc_step_line.SetName(kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "Step 1", 100, 100, {});

  // kXlaModuleLineName
  auto tc_module_line = tc_plane_builder.GetOrCreateLine(1);
  tc_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&tc_plane_builder, &tc_module_line, "Module 1", 100, 100, {});

  // kXlaOpLineName
  auto tc_op_line = tc_plane_builder.GetOrCreateLine(2);
  tc_op_line.SetName(kXlaOpLineName);
  CreateXEvent(&tc_plane_builder, &tc_op_line, "TC Offload 0", 100, 49,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{1}}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "TC Offload 1", 150, 49,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{2}}});

  // --- SparseCore Plane ---
  XPlane* sparsecore_plane = space.add_planes();
  XPlaneBuilder sc_plane_builder(sparsecore_plane);
  sc_plane_builder.SetName(
      absl::StrCat(tensorcore_plane->name(), " SparseCore 0"));

  // kSparseCoreStepLineName
  auto sc_step_line = sc_plane_builder.GetOrCreateLine(0);
  sc_step_line.SetName(kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "SC Step 1", 100, 100, {});

  // kSparseCoreModuleLineName
  auto sc_module_line = sc_plane_builder.GetOrCreateLine(1);
  sc_module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &sc_module_line, "sc_offload(0)", 110, 38,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_module_line, "sc_offload(0)", 160, 38,
               {{StatType::kTcOffloadStartId, int64_t{1}},
                {StatType::kOffloadCoreId, int64_t{0}},
                {StatType::kOffloadExecutionIndex, int64_t{2}}});

  // kSparseCoreOpLineName
  auto sc_op_line = sc_plane_builder.GetOrCreateLine(2);
  sc_op_line.SetName(kSparseCoreOpLineName);
  // These should match the TC offload events
  CreateXEvent(&sc_plane_builder, &sc_op_line, "sc_offload_op.1", 111, 10, {});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "sc_offload_op.1", 160, 10, {});

  PreprocessXSpace(&space);

  // Verify that the cross-plane offload stats are added by
  // SparseCoreOffloadPreprocessor. TC events should gain kScOffloadStartId.
  XPlaneVisitor tc_plane_visitor = CreateTfXPlaneVisitor(tensorcore_plane);
  int64_t sc_offload_context_type =
      static_cast<int64_t>(ContextType::kScOffload);
  absl::flat_hash_set<int64_t> producer_ids;
  tc_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() != kXlaOpLineName) {
      return;
    }
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(line.Name(), " ", event.Name()));
      ASSERT_TRUE(event.GetStat(StatType::kProducerType).has_value());
      ASSERT_TRUE(event.GetStat(StatType::kProducerId).has_value());
      EXPECT_EQ(event.GetStat(StatType::kProducerType)->IntValue(),
                sc_offload_context_type);
      producer_ids.insert(
          event.GetStat(StatType::kProducerId)->IntOrUintValue());
    });
  });
  EXPECT_EQ(producer_ids.size(), 2);

  // SC events should gain kTcOffloadStartId.
  XPlaneVisitor sc_plane_visitor = CreateTfXPlaneVisitor(sparsecore_plane);
  absl::flat_hash_set<int64_t> consumer_ids;
  sc_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() != kSparseCoreModuleLineName) {
      return;
    }
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(line.Name(), " ", event.Name()));
      ASSERT_TRUE(event.GetStat(StatType::kConsumerType).has_value());
      ASSERT_TRUE(event.GetStat(StatType::kConsumerId).has_value());
      EXPECT_EQ(event.GetStat(StatType::kConsumerType)->IntValue(),
                sc_offload_context_type);
      consumer_ids.insert(
          event.GetStat(StatType::kConsumerId)->IntOrUintValue());
    });
  });
  EXPECT_EQ(consumer_ids.size(), 2);
  EXPECT_EQ(producer_ids, consumer_ids);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
