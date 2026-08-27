/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/tsl/profiler/utils/xplane_visitor.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

TEST(XPlaneVisitorTest, GetStatTest) {
  XPlane plane;
  XPlaneBuilder xplane_builder(&plane);
  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventBuilder event_builder = xline_builder.AddEvent(
      *xplane_builder.GetOrCreateEventMetadata("test_event"));

  const XStatMetadata* stat_meta =
      xplane_builder.GetOrCreateStatMetadata("test_stat");
  event_builder.AddStatValue(*stat_meta, int64_t{42});

  XPlaneVisitor xplane_visitor(&plane);
  bool event_found = false;
  xplane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      event_found = true;
      std::optional<XStatVisitor> stat = event_visitor.GetStat(*stat_meta);
      ASSERT_TRUE(stat.has_value());
      EXPECT_EQ(stat->IntValue(), 42);

      const XStatMetadata* missing_meta =
          xplane_builder.GetOrCreateStatMetadata("missing_stat");
      std::optional<XStatVisitor> missing_stat =
          event_visitor.GetStat(*missing_meta);
      EXPECT_FALSE(missing_stat.has_value());
    });
  });
  EXPECT_TRUE(event_found);
}

TEST(XPlaneVisitorTest, GetEventOrMetadataStatTest) {
  XPlane plane;
  XPlaneBuilder xplane_builder(&plane);
  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventMetadata* event_meta =
      xplane_builder.GetOrCreateEventMetadata("test_event");
  const XStatMetadata* stat_meta =
      xplane_builder.GetOrCreateStatMetadata("test_stat");
  const XStatMetadata* meta_only_stat =
      xplane_builder.GetOrCreateStatMetadata("meta_only_stat");
  const XStatMetadata* string_stat =
      xplane_builder.GetOrCreateStatMetadata("string_stat");
  const XStatMetadata* double_stat =
      xplane_builder.GetOrCreateStatMetadata("double_stat");
  const XStatMetadata* bytes_stat =
      xplane_builder.GetOrCreateStatMetadata("bytes_stat");
  const XStatMetadata* uint64_bool_stat =
      xplane_builder.GetOrCreateStatMetadata("uint64_bool_stat");

  XStatsBuilder<XEventMetadata> event_meta_builder(event_meta, &xplane_builder);
  event_meta_builder.AddStatValue(*stat_meta, int64_t{10});
  event_meta_builder.AddStatValue(*meta_only_stat, int64_t{30});
  event_meta_builder.AddStatValue(*string_stat, "hello");
  event_meta_builder.AddStatValue(*double_stat, 3.14);
  event_meta_builder.AddStatValue(*uint64_bool_stat, uint64_t{1});
  XStat* bytes_stat_pb = event_meta->add_stats();
  bytes_stat_pb->set_metadata_id(bytes_stat->id());
  bytes_stat_pb->set_bytes_value("test_bytes");

  XEventBuilder event_builder = xline_builder.AddEvent(*event_meta);
  event_builder.AddStatValue(*stat_meta, int64_t{20});

  TypeGetter stat_type_getter =
      [](absl::string_view name) -> std::optional<int64_t> {
    if (name == "test_stat") {
      return 100;
    }
    if (name == "meta_only_stat") {
      return 200;
    }
    if (name == "string_stat") {
      return 300;
    }
    if (name == "double_stat") {
      return 400;
    }
    if (name == "bytes_stat") {
      return 500;
    }
    if (name == "uint64_bool_stat") {
      return 600;
    }
    return std::nullopt;
  };
  XPlaneVisitor xplane_visitor(&plane, {}, {stat_type_getter});
  bool event_found = false;
  xplane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      event_found = true;
      std::optional<XStatVisitor> stat =
          event_visitor.GetEventOrMetadataStat(100);
      ASSERT_TRUE(stat.has_value());
      // Event stat (20) should take precedence over metadata stat (10).
      EXPECT_EQ(stat->IntValue(), 20);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<int64_t>(100), 20);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat(100, -1), 20);

      // Metadata-only stat should fall back to metadata value (30).
      std::optional<XStatVisitor> meta_stat =
          event_visitor.GetEventOrMetadataStat(200);
      ASSERT_TRUE(meta_stat.has_value());
      EXPECT_EQ(meta_stat->IntValue(), 30);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<int64_t>(200), 30);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat(200, -1), 30);

      // String stat from metadata should work with both typed optional and
      // default overloads.
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<absl::string_view>(300),
                "hello");
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<absl::string_view>(
                    300, "default"),
                "hello");

      // Double stat specialization should return double value:
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<double>(400), 3.14);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<double>(400, -1.0), 3.14);
      // Calling <double> on an int stat should return nullopt or default_val
      // (-1.0):
      EXPECT_FALSE(
          event_visitor.GetEventOrMetadataStat<double>(100).has_value());
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<double>(100, -1.0), -1.0);

      // String_view specialization should handle bytes (kBytesValue):
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<absl::string_view>(500),
                "test_bytes");
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<absl::string_view>(
                    500, "default"),
                "test_bytes");

      // Bool specialization should handle uint64_t stat values correctly:
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<bool>(600), true);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<bool>(600, false), true);

      // Absent stat should return nullopt or default value.
      EXPECT_FALSE(event_visitor.GetEventOrMetadataStat(700).has_value());
      EXPECT_FALSE(
          event_visitor.GetEventOrMetadataStat<int64_t>(700).has_value());
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat(700, -1), -1);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<absl::string_view>(
                    700, "default"),
                "default");

      // Type mismatch should return nullopt or the default value.
      EXPECT_FALSE(
          event_visitor.GetEventOrMetadataStat<int64_t>(300).has_value());
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat(300, -1), -1);
      EXPECT_FALSE(event_visitor.GetEventOrMetadataStat<absl::string_view>(100)
                       .has_value());
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<absl::string_view>(
                    100, "default"),
                "default");
    });
  });
  EXPECT_TRUE(event_found);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
