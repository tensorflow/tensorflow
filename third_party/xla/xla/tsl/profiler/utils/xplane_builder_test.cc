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
#include "xla/tsl/profiler/utils/xplane_builder.h"

#include <string>

#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/platform/test.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

TEST(TimespanTests, NonInstantSpanIncludesSingleTimeTests) {
  XPlane plane;
  XPlaneBuilder xplane_builder(&plane);
  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventBuilder event_builder = xline_builder.AddEvent(
      *xplane_builder.GetOrCreateEventMetadata("1st event"));
  constexpr auto kBoolStat = true;
  constexpr auto kInt32Stat = int32_t{1234};
  constexpr auto kInt64Stat = int64_t{1234} << 32;
  constexpr auto kUint32Stat = uint32_t{5678};
  constexpr auto kUint64Stat = uint64_t{5678} << 32;
  constexpr auto kFloatStat = 0.5f;
  constexpr auto kDoubleStat = 1.0;
  constexpr auto kStringStat = "abc";
  constexpr auto kRefStat = "referenced abc";
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("bool stat"), kBoolStat);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("int32 stat"), kInt32Stat);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("int64 stat"), kInt64Stat);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("uint32 stat"), kUint32Stat);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("uint64 stat"), kUint64Stat);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("string stat"), kStringStat);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("float stat"), kFloatStat);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("double stat"), kDoubleStat);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("ref stat"),
      *xplane_builder.GetOrCreateStatMetadata(kRefStat));

  XPlaneVisitor xplane_visitor(&plane);
  EXPECT_EQ(xplane_visitor.NumLines(), 1);
  int num_stats = 0;
  xplane_visitor.ForEachLine([&](const XLineVisitor& xline) {
    xline.ForEachEvent([&](const XEventVisitor& xevent) {
      EXPECT_EQ(xevent.Name(), "1st event");
      xevent.ForEachStat([&](const XStatVisitor& stat) {
        if (stat.Name() == "bool stat") {
          EXPECT_EQ(stat.BoolValue(), kBoolStat);
          num_stats++;
        } else if (stat.Name() == "int32 stat") {
          EXPECT_EQ(stat.IntValue(), kInt32Stat);
          EXPECT_EQ(stat.IntOrUintValue(), kInt32Stat);
          num_stats++;
        } else if (stat.Name() == "int64 stat") {
          EXPECT_EQ(stat.IntValue(), kInt64Stat);
          EXPECT_EQ(stat.IntOrUintValue(), kInt64Stat);
          num_stats++;
        } else if (stat.Name() == "uint32 stat") {
          EXPECT_EQ(stat.UintValue(), kUint32Stat);
          EXPECT_EQ(stat.IntOrUintValue(), kUint32Stat);
          num_stats++;
        } else if (stat.Name() == "uint64 stat") {
          EXPECT_EQ(stat.UintValue(), kUint64Stat);
          EXPECT_EQ(stat.IntOrUintValue(), kUint64Stat);
          num_stats++;
        } else if (stat.Name() == "string stat") {
          EXPECT_EQ(stat.StrOrRefValue(), kStringStat);
          num_stats++;
        } else if (stat.Name() == "float stat") {
          EXPECT_EQ(stat.DoubleValue(), kFloatStat);
          num_stats++;
        } else if (stat.Name() == "double stat") {
          EXPECT_EQ(stat.DoubleValue(), kDoubleStat);
          num_stats++;
        } else if (stat.Name() == "ref stat") {
          EXPECT_EQ(stat.StrOrRefValue(), kRefStat);
          num_stats++;
        }
      });
    });
  });
  EXPECT_EQ(num_stats, 9);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
