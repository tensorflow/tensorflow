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

#include "tensorflow/tsl/profiler/utils/xplane_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/utils/math_utils.h"
#include "tensorflow/tsl/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/tsl/profiler/utils/xplane_builder.h"
#include "tensorflow/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/tsl/profiler/utils/xplane_visitor.h"

namespace tsl {
namespace profiler {
namespace {

using ::testing::Property;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

#if defined(PLATFORM_GOOGLE)
using ::testing::EqualsProto;
using ::testing::proto::IgnoringRepeatedFieldOrdering;
#endif

XEvent CreateEvent(int64_t offset_ps, int64_t duration_ps) {
  XEvent event;
  event.set_offset_ps(offset_ps);
  event.set_duration_ps(duration_ps);
  return event;
}

TEST(XPlaneUtilsTest, AddAndRemovePlanes) {
  XSpace space;

  auto* p1 = FindOrAddMutablePlaneWithName(&space, "p1");
  EXPECT_EQ(p1, FindPlaneWithName(space, "p1"));
  auto* p2 = FindOrAddMutablePlaneWithName(&space, "p2");
  EXPECT_EQ(p2, FindPlaneWithName(space, "p2"));
  auto* p3 = FindOrAddMutablePlaneWithName(&space, "p3");
  EXPECT_EQ(p3, FindPlaneWithName(space, "p3"));

  // Removing a plane does not invalidate pointers to other planes.

  RemovePlane(&space, p2);
  EXPECT_EQ(space.planes_size(), 2);
  EXPECT_EQ(p1, FindPlaneWithName(space, "p1"));
  EXPECT_EQ(p3, FindPlaneWithName(space, "p3"));

  RemovePlane(&space, p1);
  EXPECT_EQ(space.planes_size(), 1);
  EXPECT_EQ(p3, FindPlaneWithName(space, "p3"));

  RemovePlane(&space, p3);
  EXPECT_EQ(space.planes_size(), 0);
}

TEST(XPlaneUtilsTest, RemoveEmptyPlanes) {
  XSpace space;
  RemoveEmptyPlanes(&space);
  EXPECT_EQ(space.planes_size(), 0);

  auto* plane1 = space.add_planes();
  plane1->set_name("p1");
  plane1->add_lines()->set_name("p1l1");
  plane1->add_lines()->set_name("p1l2");

  auto* plane2 = space.add_planes();
  plane2->set_name("p2");

  auto* plane3 = space.add_planes();
  plane3->set_name("p3");
  plane3->add_lines()->set_name("p3l1");

  auto* plane4 = space.add_planes();
  plane4->set_name("p4");

  RemoveEmptyPlanes(&space);
  ASSERT_EQ(space.planes_size(), 2);
  EXPECT_EQ(space.planes(0).name(), "p1");
  EXPECT_EQ(space.planes(1).name(), "p3");
}

TEST(XPlaneUtilsTest, RemoveEmptyLines) {
  XPlane plane;
  RemoveEmptyLines(&plane);
  EXPECT_EQ(plane.lines_size(), 0);

  auto* line1 = plane.add_lines();
  line1->set_name("l1");
  line1->add_events();
  line1->add_events();

  auto* line2 = plane.add_lines();
  line2->set_name("l2");

  auto* line3 = plane.add_lines();
  line3->set_name("l3");
  line3->add_events();

  auto* line4 = plane.add_lines();
  line4->set_name("l4");

  RemoveEmptyLines(&plane);
  ASSERT_EQ(plane.lines_size(), 2);
  EXPECT_EQ(plane.lines(0).name(), "l1");
  EXPECT_EQ(plane.lines(1).name(), "l3");
}

TEST(XPlaneUtilsTest, RemoveLine) {
  XPlane plane;
  const XLine* line1 = plane.add_lines();
  const XLine* line2 = plane.add_lines();
  const XLine* line3 = plane.add_lines();
  RemoveLine(&plane, line2);
  ASSERT_EQ(plane.lines_size(), 2);
  EXPECT_EQ(&plane.lines(0), line1);
  EXPECT_EQ(&plane.lines(1), line3);
}

TEST(XPlaneUtilsTest, RemoveEvents) {
  XLine line;
  const XEvent* event1 = line.add_events();
  const XEvent* event2 = line.add_events();
  const XEvent* event3 = line.add_events();
  const XEvent* event4 = line.add_events();
  RemoveEvents(&line, {event1, event3});
  ASSERT_EQ(line.events_size(), 2);
  EXPECT_EQ(&line.events(0), event2);
  EXPECT_EQ(&line.events(1), event4);
}

TEST(XPlaneUtilsTest, SortXPlaneTest) {
  XPlane plane;
  XLine* line = plane.add_lines();
  *line->add_events() = CreateEvent(200, 100);
  *line->add_events() = CreateEvent(100, 100);
  *line->add_events() = CreateEvent(120, 50);
  *line->add_events() = CreateEvent(120, 30);
  SortXPlane(&plane);
  ASSERT_EQ(plane.lines_size(), 1);
  ASSERT_EQ(plane.lines(0).events_size(), 4);
  EXPECT_EQ(plane.lines(0).events(0).offset_ps(), 100);
  EXPECT_EQ(plane.lines(0).events(0).duration_ps(), 100);
  EXPECT_EQ(plane.lines(0).events(1).offset_ps(), 120);
  EXPECT_EQ(plane.lines(0).events(1).duration_ps(), 50);
  EXPECT_EQ(plane.lines(0).events(2).offset_ps(), 120);
  EXPECT_EQ(plane.lines(0).events(2).duration_ps(), 30);
  EXPECT_EQ(plane.lines(0).events(3).offset_ps(), 200);
  EXPECT_EQ(plane.lines(0).events(3).duration_ps(), 100);
}

namespace {

XLineBuilder CreateXLine(XPlaneBuilder* plane, absl::string_view name,
                         absl::string_view display, int64_t id,
                         int64_t timestamp_ns) {
  XLineBuilder line = plane->GetOrCreateLine(id);
  line.SetName(name);
  line.SetTimestampNs(timestamp_ns);
  line.SetDisplayNameIfEmpty(display);
  return line;
}

XEventBuilder CreateXEvent(XPlaneBuilder* plane, XLineBuilder line,
                           absl::string_view event_name,
                           absl::optional<absl::string_view> display,
                           int64_t offset_ns, int64_t duration_ns) {
  XEventMetadata* event_metadata = plane->GetOrCreateEventMetadata(event_name);
  if (display) event_metadata->set_display_name(std::string(*display));
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetNs(offset_ns);
  event.SetDurationNs(duration_ns);
  return event;
}

template <typename T, typename V>
void CreateXStats(XPlaneBuilder* plane, T* stats_owner,
                  absl::string_view stats_name, V stats_value) {
  stats_owner->AddStatValue(*plane->GetOrCreateStatMetadata(stats_name),
                            stats_value);
}

void CheckXLine(const XLine& line, absl::string_view name,
                absl::string_view display, int64_t start_time_ns,
                int64_t events_size) {
  EXPECT_EQ(line.name(), name);
  EXPECT_EQ(line.display_name(), display);
  EXPECT_EQ(line.timestamp_ns(), start_time_ns);
  EXPECT_EQ(line.events_size(), events_size);
}

void CheckXEvent(const XEvent& event, const XPlane& plane,
                 absl::string_view name, absl::string_view display,
                 int64_t offset_ns, int64_t duration_ns, int64_t stats_size) {
  const XEventMetadata& event_metadata =
      plane.event_metadata().at(event.metadata_id());
  EXPECT_EQ(event_metadata.name(), name);
  EXPECT_EQ(event_metadata.display_name(), display);
  EXPECT_EQ(event.offset_ps(), NanoToPico(offset_ns));
  EXPECT_EQ(event.duration_ps(), NanoToPico(duration_ns));
  EXPECT_EQ(event.stats_size(), stats_size);
}
}  // namespace

TEST(XPlaneUtilsTest, MergeXPlaneTest) {
  XPlane src_plane, dst_plane;
  constexpr int64_t kLineIdOnlyInSrcPlane = 1LL;
  constexpr int64_t kLineIdOnlyInDstPlane = 2LL;
  constexpr int64_t kLineIdInBothPlanes = 3LL;   // src start ts < dst start ts
  constexpr int64_t kLineIdInBothPlanes2 = 4LL;  // src start ts > dst start ts

  {  // Populate the source plane.
    XPlaneBuilder src(&src_plane);
    CreateXStats(&src, &src, "plane_stat1", 1);    // only in source.
    CreateXStats(&src, &src, "plane_stat3", 3.0);  // shared by source/dest.

    auto l1 = CreateXLine(&src, "l1", "d1", kLineIdOnlyInSrcPlane, 100);
    auto e1 = CreateXEvent(&src, l1, "event1", "display1", 1, 2);
    CreateXStats(&src, &e1, "event_stat1", 2.0);
    auto e2 = CreateXEvent(&src, l1, "event2", absl::nullopt, 3, 4);
    CreateXStats(&src, &e2, "event_stat2", 3);

    auto l2 = CreateXLine(&src, "l2", "d2", kLineIdInBothPlanes, 200);
    auto e3 = CreateXEvent(&src, l2, "event3", absl::nullopt, 5, 7);
    CreateXStats(&src, &e3, "event_stat3", 2.0);
    auto e4 = CreateXEvent(&src, l2, "event4", absl::nullopt, 6, 8);
    CreateXStats(&src, &e4, "event_stat4", 3);
    CreateXStats(&src, &e4, "event_stat5", 3);

    auto l5 = CreateXLine(&src, "l5", "d5", kLineIdInBothPlanes2, 700);
    CreateXEvent(&src, l5, "event51", absl::nullopt, 9, 10);
    CreateXEvent(&src, l5, "event52", absl::nullopt, 11, 12);
  }

  {  // Populate the destination plane.
    XPlaneBuilder dst(&dst_plane);
    CreateXStats(&dst, &dst, "plane_stat2", 2);  // only in dest
    CreateXStats(&dst, &dst, "plane_stat3", 4);  // shared but different.

    auto l3 = CreateXLine(&dst, "l3", "d3", kLineIdOnlyInDstPlane, 300);
    auto e5 = CreateXEvent(&dst, l3, "event5", absl::nullopt, 11, 2);
    CreateXStats(&dst, &e5, "event_stat6", 2.0);
    auto e6 = CreateXEvent(&dst, l3, "event6", absl::nullopt, 13, 4);
    CreateXStats(&dst, &e6, "event_stat7", 3);

    auto l2 = CreateXLine(&dst, "l4", "d4", kLineIdInBothPlanes, 400);
    auto e7 = CreateXEvent(&dst, l2, "event7", absl::nullopt, 15, 7);
    CreateXStats(&dst, &e7, "event_stat8", 2.0);
    auto e8 = CreateXEvent(&dst, l2, "event8", "display8", 16, 8);
    CreateXStats(&dst, &e8, "event_stat9", 3);

    auto l6 = CreateXLine(&dst, "l6", "d6", kLineIdInBothPlanes2, 300);
    CreateXEvent(&dst, l6, "event61", absl::nullopt, 21, 10);
    CreateXEvent(&dst, l6, "event62", absl::nullopt, 22, 12);
  }

  MergePlanes(src_plane, &dst_plane);

  XPlaneVisitor plane(&dst_plane);
  EXPECT_EQ(dst_plane.lines_size(), 4);

  // Check plane level stats,
  EXPECT_EQ(dst_plane.stats_size(), 3);
  absl::flat_hash_map<absl::string_view, absl::string_view> plane_stats;
  plane.ForEachStat([&](const XStatVisitor& stat) {
    if (stat.Name() == "plane_stat1") {
      EXPECT_EQ(stat.IntValue(), 1);
    } else if (stat.Name() == "plane_stat2") {
      EXPECT_EQ(stat.IntValue(), 2);
    } else if (stat.Name() == "plane_stat3") {
      // XStat in src_plane overrides the counter-part in dst_plane.
      EXPECT_EQ(stat.DoubleValue(), 3.0);
    } else {
      EXPECT_TRUE(false);
    }
  });

  // 3 plane level stats, 9 event level stats.
  EXPECT_EQ(dst_plane.stat_metadata_size(), 12);

  {  // Old lines are untouched.
    const XLine& line = dst_plane.lines(0);
    CheckXLine(line, "l3", "d3", 300, 2);
    CheckXEvent(line.events(0), dst_plane, "event5", "", 11, 2, 1);
    CheckXEvent(line.events(1), dst_plane, "event6", "", 13, 4, 1);
  }
  {  // Lines with the same id are merged.
    // src plane start timestamp > dst plane start timestamp
    const XLine& line = dst_plane.lines(1);
    // NOTE: use minimum start time of src/dst.
    CheckXLine(line, "l4", "d4", 200, 4);
    CheckXEvent(line.events(0), dst_plane, "event7", "", 215, 7, 1);
    CheckXEvent(line.events(1), dst_plane, "event8", "display8", 216, 8, 1);
    CheckXEvent(line.events(2), dst_plane, "event3", "", 5, 7, 1);
    CheckXEvent(line.events(3), dst_plane, "event4", "", 6, 8, 2);
  }
  {  // Lines with the same id are merged.
    // src plane start timestamp < dst plane start timestamp
    const XLine& line = dst_plane.lines(2);
    CheckXLine(line, "l6", "d6", 300, 4);
    CheckXEvent(line.events(0), dst_plane, "event61", "", 21, 10, 0);
    CheckXEvent(line.events(1), dst_plane, "event62", "", 22, 12, 0);
    CheckXEvent(line.events(2), dst_plane, "event51", "", 409, 10, 0);
    CheckXEvent(line.events(3), dst_plane, "event52", "", 411, 12, 0);
  }
  {  // Lines only in source are "copied".
    const XLine& line = dst_plane.lines(3);
    CheckXLine(line, "l1", "d1", 100, 2);
    CheckXEvent(line.events(0), dst_plane, "event1", "display1", 1, 2, 1);
    CheckXEvent(line.events(1), dst_plane, "event2", "", 3, 4, 1);
  }
}

TEST(XPlaneUtilsTest, FindPlanesWithPrefix) {
  XSpace xspace;
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:0");
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:1");
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:2");
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:3");
  XPlane* p4 = FindOrAddMutablePlaneWithName(&xspace, "test-do-not-include:0");

  std::vector<const XPlane*> xplanes =
      FindPlanesWithPrefix(xspace, "test-prefix");
  ASSERT_EQ(4, xplanes.size());
  for (const XPlane* plane : xplanes) {
    ASSERT_NE(p4, plane);
  }
}

TEST(XplaneUtilsTest, FindMutablePlanesWithPrefix) {
  XSpace xspace;
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:0");
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:1");
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:2");
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:3");
  XPlane* p4 = FindOrAddMutablePlaneWithName(&xspace, "test-do-not-include:0");

  std::vector<XPlane*> xplanes =
      FindMutablePlanesWithPrefix(&xspace, "test-prefix");
  ASSERT_EQ(4, xplanes.size());
  for (XPlane* plane : xplanes) {
    ASSERT_NE(p4, plane);
  }
}

TEST(XplaneUtilsTest, FindPlanesWithPredicate) {
  XSpace xspace;
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:0");
  XPlane* p1 = FindOrAddMutablePlaneWithName(&xspace, "test-prefix:1");

  std::vector<const XPlane*> xplanes = FindPlanes(
      xspace,
      [](const XPlane& xplane) { return xplane.name() == "test-prefix:1"; });
  ASSERT_EQ(1, xplanes.size());
  ASSERT_EQ(p1, xplanes[0]);
}

TEST(XplaneUtilsTest, FindMutablePlanesWithPredicate) {
  XSpace xspace;
  FindOrAddMutablePlaneWithName(&xspace, "test-prefix:0");
  XPlane* p1 = FindOrAddMutablePlaneWithName(&xspace, "test-prefix:1");

  std::vector<XPlane*> xplanes = FindMutablePlanes(
      &xspace, [](XPlane& xplane) { return xplane.name() == "test-prefix:1"; });
  ASSERT_EQ(1, xplanes.size());
  ASSERT_EQ(p1, xplanes[0]);
}

TEST(XplaneUtilsTest, TestAggregateXPlanes) {
  XPlane xplane;
  XPlaneBuilder builder(&xplane);
  XEventMetadata* event_metadata1 = builder.GetOrCreateEventMetadata(1);
  event_metadata1->set_name("EventMetadata1");
  XEventMetadata* event_metadata2 = builder.GetOrCreateEventMetadata(2);
  event_metadata2->set_name("EventMetadata2");
  XEventMetadata* event_metadata3 = builder.GetOrCreateEventMetadata(3);
  event_metadata3->set_name("EventMetadata3");
  XEventMetadata* event_metadata4 = builder.GetOrCreateEventMetadata(4);
  event_metadata4->set_name("EventMetadata4");

  XLineBuilder line = builder.GetOrCreateLine(1);
  line.SetName(kTensorFlowOpLineName);
  XEventBuilder event1 = line.AddEvent(*event_metadata1);
  event1.SetOffsetNs(0);
  event1.SetDurationNs(5);
  XEventBuilder event3 = line.AddEvent(*event_metadata3);
  event3.SetOffsetNs(0);
  event3.SetDurationNs(2);
  XEventBuilder event2 = line.AddEvent(*event_metadata2);
  event2.SetOffsetNs(5);
  event2.SetDurationNs(5);
  XEventBuilder event4 = line.AddEvent(*event_metadata2);
  event4.SetOffsetNs(10);
  event4.SetDurationNs(5);
  XEventBuilder event5 = line.AddEvent(*event_metadata4);
  event5.SetOffsetNs(15);
  event5.SetDurationNs(6);
  XEventBuilder event6 = line.AddEvent(*event_metadata1);
  event6.SetOffsetNs(15);
  event6.SetDurationNs(4);
  XEventBuilder event7 = line.AddEvent(*event_metadata3);
  event7.SetOffsetNs(15);
  event7.SetDurationNs(3);

  XPlane aggregated_xplane;
  AggregateXPlane(xplane, aggregated_xplane);

// Protobuf matchers are unavailable in OSS (b/169705709)
#if defined(PLATFORM_GOOGLE)
  ASSERT_THAT(aggregated_xplane,
              IgnoringRepeatedFieldOrdering(EqualsProto(
                  R"pb(lines {
                         id: 1
                         name: "TensorFlow Ops"
                         events {
                           metadata_id: 1
                           duration_ps: 9000
                           stats { metadata_id: 2 int64_value: 4000 }
                           stats { metadata_id: 3 int64_value: 4000 }
                           num_occurrences: 2
                         }
                         events {
                           metadata_id: 3
                           duration_ps: 5000
                           stats { metadata_id: 2 int64_value: 2000 }
                           num_occurrences: 2
                         }
                         events {
                           metadata_id: 2
                           duration_ps: 10000
                           stats { metadata_id: 2 int64_value: 5000 }
                           num_occurrences: 2
                         }
                         events {
                           metadata_id: 4
                           duration_ps: 6000
                           stats { metadata_id: 3 int64_value: 2000 }
                           num_occurrences: 1
                         }
                       }
                       event_metadata {
                         key: 1
                         value { id: 1 name: "EventMetadata1" }
                       }
                       event_metadata {
                         key: 2
                         value { id: 2 name: "EventMetadata2" }
                       }
                       event_metadata {
                         key: 3
                         value { id: 3 name: "EventMetadata3" }
                       }
                       event_metadata {
                         key: 4
                         value { id: 4 name: "EventMetadata4" }
                       }
                       stat_metadata {
                         key: 1
                         value { id: 1 name: "total_profile_duration_ps" }
                       }
                       stat_metadata {
                         key: 2
                         value { id: 2 name: "min_duration_ps" }
                       }
                       stat_metadata {
                         key: 3
                         value { id: 3 name: "self_duration_ps" }
                       }
                       stats { metadata_id: 1 uint64_value: 21000 }
                  )pb")));
#endif
}

TEST(XPlanuUtilsTest, TestInstantEventDoesNotFail) {
  XPlane xplane;
  XPlaneBuilder xplane_builder(&xplane);
  XEventMetadata* event_metadata1 = xplane_builder.GetOrCreateEventMetadata(1);
  XEventMetadata* event_metadata2 = xplane_builder.GetOrCreateEventMetadata(2);

  XLineBuilder line = xplane_builder.GetOrCreateLine(1);
  line.SetName(kTensorFlowOpLineName);
  XEventBuilder event1 = line.AddEvent(*event_metadata1);
  XEventBuilder event2 = line.AddEvent(*event_metadata2);

  event1.SetOffsetNs(1);
  event1.SetDurationNs(0);
  event2.SetOffsetNs(1);
  event2.SetDurationNs(0);

  XPlane aggregated_xplane;
  AggregateXPlane(xplane, aggregated_xplane);

  EXPECT_THAT(aggregated_xplane.lines(),
              UnorderedElementsAre(Property(&XLine::events, SizeIs(2))));
}

TEST(XplaneutilsTest, TestEventMetadataStatsAreCopied) {
  XPlane xplane;
  XPlaneBuilder xplane_builder(&xplane);
  XEventMetadata* event_metadata = xplane_builder.GetOrCreateEventMetadata(1);

  XStatsBuilder<XEventMetadata> stats(event_metadata, &xplane_builder);
  stats.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      "TestFunction");
  XLineBuilder line = xplane_builder.GetOrCreateLine(1);
  line.SetName(kTensorFlowOpLineName);
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetDurationNs(0);
  event.SetOffsetNs(0);

  XPlane aggregated_xplane;
  AggregateXPlane(xplane, aggregated_xplane);

  XPlaneVisitor visitor = CreateTfXPlaneVisitor(&aggregated_xplane);

  XEventMetadataVisitor metadata_visitor(&visitor, visitor.GetEventMetadata(1));
  std::optional<XStatVisitor> stat = metadata_visitor.GetStat(StatType::kTfOp);

  ASSERT_TRUE(stat.has_value());
  EXPECT_EQ(stat->Name(), "tf_op");
  EXPECT_EQ(stat->StrOrRefValue(), "TestFunction");
}

TEST(XplaneutilsTest, TestEventMetadataStatsAreCopiedForRefValue) {
  XPlane xplane;
  XPlaneBuilder xplane_builder(&xplane);
  XEventMetadata* event_metadata = xplane_builder.GetOrCreateEventMetadata(1);

  XStatsBuilder<XEventMetadata> stats(event_metadata, &xplane_builder);
  stats.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      *xplane_builder.GetOrCreateStatMetadata("TestFunction"));
  XLineBuilder line = xplane_builder.GetOrCreateLine(1);
  line.SetName(kTensorFlowOpLineName);
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetDurationNs(0);
  event.SetOffsetNs(0);

  XPlane aggregated_xplane;
  AggregateXPlane(xplane, aggregated_xplane);

  XPlaneVisitor visitor = CreateTfXPlaneVisitor(&aggregated_xplane);

  XEventMetadataVisitor metadata_visitor(&visitor, visitor.GetEventMetadata(1));
  std::optional<XStatVisitor> stat = metadata_visitor.GetStat(StatType::kTfOp);

  ASSERT_TRUE(stat.has_value());
  EXPECT_EQ(stat->Name(), "tf_op");
  EXPECT_EQ(stat->StrOrRefValue(), "TestFunction");
}

TEST(XplaneutilsTest, TestIsXSpaceGrouped) {
  XSpace space;
  {
    XPlaneBuilder p1(space.add_planes());
    auto l1 = CreateXLine(&p1, "l1", "d1", 1, 100);
    auto e1 = CreateXEvent(&p1, l1, "event1", "display1", 1, 2);
    CreateXStats(&p1, &e1, "event_stat1", 2.0);
  }
  EXPECT_FALSE(IsXSpaceGrouped(space));

  {
    XPlaneBuilder p2(space.add_planes());
    auto l2 = CreateXLine(&p2, "l2", "d2", 1, 100);
    auto e2 = CreateXEvent(&p2, l2, "event2", "display2", 1, 2);
    CreateXStats(&p2, &e2, "group_id", 1);
  }
  LOG(ERROR) << space.DebugString();
  EXPECT_TRUE(IsXSpaceGrouped(space));
}

TEST(XplaneutilsTest, TestIsHostPlane) {
  XSpace xspace;
  auto xplane_host_thread = FindOrAddMutablePlaneWithName(&xspace, "/host:CPU");
  auto xplane_host_cpu = FindOrAddMutablePlaneWithName(&xspace, "Host CPUs");
  auto xplane_tfstreamz =
      FindOrAddMutablePlaneWithName(&xspace, "/host:tfstreamz");
  auto xplane_metadata =
      FindOrAddMutablePlaneWithName(&xspace, "/host:metadata");
  auto xplane_syscalls = FindOrAddMutablePlaneWithName(&xspace, "Syscalls");
  auto xplane_python_tracer =
      FindOrAddMutablePlaneWithName(&xspace, "/host:python-tracer");
  auto xplane_custom_prefix =
      FindOrAddMutablePlaneWithName(&xspace, "/device:CUSTOM:123");
  auto xplane_legacy_custom =
      FindOrAddMutablePlaneWithName(&xspace, "/custom:456");
  auto xplane_cupti = FindOrAddMutablePlaneWithName(&xspace, "/host:CUPTI");
  EXPECT_TRUE(IsHostPlane(*xplane_host_thread));
  EXPECT_TRUE(IsHostPlane(*xplane_host_cpu));
  EXPECT_TRUE(IsHostPlane(*xplane_tfstreamz));
  EXPECT_TRUE(IsHostPlane(*xplane_metadata));
  EXPECT_TRUE(IsHostPlane(*xplane_syscalls));
  EXPECT_TRUE(IsHostPlane(*xplane_python_tracer));
  EXPECT_FALSE(IsHostPlane(*xplane_custom_prefix));
  EXPECT_FALSE(IsHostPlane(*xplane_legacy_custom));
  EXPECT_TRUE(IsHostPlane(*xplane_cupti));
}

TEST(XplaneutilsTest, TestIsDevicePlane) {
  XSpace xspace;
  auto xplane_host_thread = FindOrAddMutablePlaneWithName(&xspace, "/host:CPU");
  auto xplane_device_thread =
      FindOrAddMutablePlaneWithName(&xspace, "/device:TPU");
  auto xplane_task_env_thread =
      FindOrAddMutablePlaneWithName(&xspace, "Task Environment");
  auto xplane_custom_prefix =
      FindOrAddMutablePlaneWithName(&xspace, "/device:CUSTOM:123");
  auto xplane_legacy_custom =
      FindOrAddMutablePlaneWithName(&xspace, "/custom:456");
  EXPECT_FALSE(IsDevicePlane(*xplane_host_thread));
  EXPECT_FALSE(IsDevicePlane(*xplane_task_env_thread));
  EXPECT_TRUE(IsDevicePlane(*xplane_device_thread));
  EXPECT_TRUE(IsDevicePlane(*xplane_custom_prefix));
  EXPECT_TRUE(IsDevicePlane(*xplane_legacy_custom));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
