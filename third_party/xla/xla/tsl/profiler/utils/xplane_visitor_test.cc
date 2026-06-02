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

}  // namespace
}  // namespace profiler
}  // namespace tsl
