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
#include <string>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
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

TEST(PreprocessXPlane, NormalizeGpuHloModuleIdsTest) {
  XSpace space;
  XPlane* plane1 = space.add_planes();
  plane1->set_id(0);
  plane1->set_name("plane0");
  XPlaneBuilder plane_builder1(plane1);

  // Add stat metadata for kHloProto
  XStatMetadata& hlo_proto_stat_metadata =
      (*plane1->mutable_stat_metadata())[1];
  hlo_proto_stat_metadata.set_id(1);
  hlo_proto_stat_metadata.set_name(GetStatTypeStr(StatType::kHloProto));
  // Add stat metadata for kFingerprint
  XStatMetadata& fingerprint_stat_metadata =
      (*plane1->mutable_stat_metadata())[2];
  fingerprint_stat_metadata.set_id(2);
  fingerprint_stat_metadata.set_name(GetStatTypeStr(StatType::kFingerprint));

  // Add event metadata with an HLO proto stat
  int64_t old_event_metadata_id1 = 123;
  std::string hlo_proto_str = "hlo_module";
  const uint64_t fingerprint = 999;
  XEventMetadata* event_metadata1 =
      plane_builder1.GetOrCreateEventMetadata(old_event_metadata_id1);
  event_metadata1->set_name("my_hlo_event");
  {
    XStat* stat = event_metadata1->add_stats();
    stat->set_metadata_id(1);
    stat->set_bytes_value(hlo_proto_str);
  }
  {
    XStat* stat = event_metadata1->add_stats();
    stat->set_metadata_id(2);
    stat->set_uint64_value(fingerprint);
  }

  // Add an event that uses this metadata
  auto line_builder1 = plane_builder1.GetOrCreateLine(0);
  XEventBuilder event_builder1 = line_builder1.AddEvent(*event_metadata1);
  event_builder1.SetOffsetPs(1000);
  event_builder1.SetDurationPs(2000);

  // Also add an event that doesn't have an HLO proto, it should be unchanged.
  int64_t other_event_metadata_id = 789;
  XEventMetadata* other_event_metadata =
      plane_builder1.GetOrCreateEventMetadata(other_event_metadata_id);
  other_event_metadata->set_name("other_event");
  XEventBuilder other_event_builder =
      line_builder1.AddEvent(*other_event_metadata);
  other_event_builder.SetOffsetPs(5000);
  other_event_builder.SetDurationPs(1000);

  PreprocessXSpace(&space);

  // Verification
  XPlaneVisitor plane_visitor1 = CreateTfXPlaneVisitor(&space.planes(0));
  uint64_t new_program_id1 =
      absl::HashOf(absl::StrCat(hlo_proto_str, 0, fingerprint));
  EXPECT_NE(new_program_id1, old_event_metadata_id1);
  // Check event metadata is remapped
  EXPECT_EQ(plane_visitor1.GetEventMetadata(old_event_metadata_id1),
            &XEventMetadata::default_instance());
  const XEventMetadata* new_event_metadata1 =
      plane_visitor1.GetEventMetadata(new_program_id1);
  ASSERT_NE(new_event_metadata1, &XEventMetadata::default_instance());
  EXPECT_EQ(new_event_metadata1->id(), new_program_id1);
  EXPECT_EQ(new_event_metadata1->name(), "my_hlo_event");

  // Check event is remapped
  bool hlo_event_found1 = false;
  bool other_event_found1 = false;
  plane_visitor1.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.OffsetPs() == 1000) {
        hlo_event_found1 = true;
        EXPECT_EQ(event.metadata()->id(), new_program_id1);
      } else if (event.OffsetPs() == 5000) {
        other_event_found1 = true;
        EXPECT_EQ(event.metadata()->id(), other_event_metadata_id);
      }
    });
  });
  EXPECT_TRUE(hlo_event_found1);
  EXPECT_TRUE(other_event_found1);
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

}  // namespace
}  // namespace profiler
}  // namespace tsl
