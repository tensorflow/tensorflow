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

#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {
#if defined(PLATFORM_GOOGLE)
using ::testing::EqualsProto;
using ::testing::proto::IgnoringRepeatedFieldOrdering;
#endif

constexpr double kMaxError = 1E-10;

TEST(OpMetricsDbTest, IdleTimeRatio) {
  OpMetricsDb metrics_db_0;
  metrics_db_0.set_total_time_ps(100000000);
  metrics_db_0.set_total_op_time_ps(60000000);
  EXPECT_NEAR(0.4, IdleTimeRatio(metrics_db_0), kMaxError);

  OpMetricsDb metrics_db_1;
  metrics_db_1.set_total_time_ps(200000000);
  metrics_db_1.set_total_op_time_ps(150000000);
  EXPECT_NEAR(0.25, IdleTimeRatio(metrics_db_1), kMaxError);

  OpMetricsDb metrics_db_2;
  metrics_db_1.set_total_time_ps(0);
  metrics_db_1.set_total_op_time_ps(0);
  EXPECT_NEAR(1.0, IdleTimeRatio(metrics_db_2), kMaxError);
}

TEST(OpMetricsDbTest, FromXEventHandlesMissingOccurrences) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XLineBuilder line = plane.GetOrCreateLine(0);
  XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("metadata");
  event_metadata->set_display_name("display_name");
  XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)), 1);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 2);
  stats.AddStatValue(*plane.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kDeduplicatedName)),
                     "deduplicated_name");
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)), "tf_op");
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kHloCategory)),
      "tf_op_category");
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kFlops)), 3);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kModelFlops)), 4);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kBytesAccessed)),
      5);
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetPs(0);
  event.SetDurationPs(100);
  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  tsl::profiler::XEventVisitor event_visitor(
      &plane_visitor, &raw_plane.lines(0), &raw_plane.lines(0).events(0));
  OpMetrics op_metrics = FromXEvent(event_visitor);

#if defined(PLATFORM_GOOGLE)
  EXPECT_THAT(op_metrics, EqualsProto(R"pb(
                occurrences: 1
                time_ps: 100
                self_time_ps: 100
                dma_stall_ps: 0
                hlo_module_id: 1
                flops: 3
                model_flops: 4
                bytes_accessed: 5
                name: "display_name"
                long_name: "metadata"
                deduplicated_name: "deduplicated_name"
                category: "tf_op_category"
                provenance: "tf_op"
                min_time_ps: 100
                num_cores: 1
              )pb"));
#endif
}

TEST(OpMetricsDbTest, GetOpKeyFromXEvent) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("metadata");
  event_metadata->set_display_name("display_name");
  XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)), 1);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 2);
  XLineBuilder line = plane.GetOrCreateLine(0);
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetPs(0);
  event.SetDurationPs(100);
  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  tsl::profiler::XEventVisitor event_visitor(
      &plane_visitor, &raw_plane.lines(0), &raw_plane.lines(0).events(0));
  XEventsOpMetricsDbBuilder::OpKey op_key = GetOpKeyFromXEvent(event_visitor);
  EXPECT_EQ(op_key.program_id, 1);
  EXPECT_EQ(op_key.symbol_id, 2);
}

TEST(OpMetricsDbTest, XEventsOpMetricsDbBuilder) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XLineBuilder line = plane.GetOrCreateLine(0);
  {
    XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("m1");
    event_metadata->set_display_name("display_name1");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)),
        1);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 1);
    XEventBuilder event = line.AddEvent(*event_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(100);
    XEventBuilder event2 = line.AddEvent(*event_metadata);
    event2.SetOffsetPs(100);
    event2.SetDurationPs(100);
  }
  {
    XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("m2");
    event_metadata->set_display_name("display_name2");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)),
        1);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 2);
    XEventBuilder event = line.AddEvent(*event_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(100);
  }
  {
    XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("m3");
    event_metadata->set_display_name("display_name3");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 1);
    XEventBuilder event = line.AddEvent(*event_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(100);
  }

  XEventsOpMetricsDbBuilder builder;
  XEventsOpMetricsDbBuilder legacy_builder;
  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      builder.AddOpMetric(FromXEvent(event), GetOpKeyFromXEvent(event));
      legacy_builder.AddOpMetric(event);
    });
  });
#if defined(PLATFORM_GOOGLE)
  OpMetricsDb legacy_db = legacy_builder.Finalize();
  OpMetricsDb db = builder.Finalize();
  EXPECT_THAT(db, IgnoringRepeatedFieldOrdering(EqualsProto(legacy_db)));
  EXPECT_THAT(db, IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                metrics_db {
                  hlo_module_id: 1
                  self_time_ps: 200
                  occurrences: 2
                  name: "display_name1"
                  long_name: "m1"
                  time_ps: 200
                  min_time_ps: 100
                  num_cores: 1
                }
                metrics_db {
                  hlo_module_id: 1
                  self_time_ps: 100
                  occurrences: 1
                  name: "display_name2"
                  long_name: "m2"
                  time_ps: 100
                  min_time_ps: 100
                  num_cores: 1
                }
                total_op_time_ps: 300
              )pb")));
#endif
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
