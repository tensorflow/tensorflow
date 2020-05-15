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
#include "tensorflow/core/profiler/utils/xplane_builder.h"

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(TimespanTests, NonInstantSpanIncludesSingleTimeTests) {
  XPlane plane;
  XPlaneBuilder xplane_builder(&plane);
  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventBuilder event_builder = xline_builder.AddEvent(
      *xplane_builder.GetOrCreateEventMetadata("1st event"));
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("int stat"), 1234LL);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("string stat"),
      std::string("abc"));
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("double stat"), 1.0);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("ref stat"),
      *xplane_builder.GetOrCreateStatMetadata("referenced abc"));

  XPlaneVisitor xplane_visitor(&plane);
  EXPECT_EQ(xplane_visitor.NumLines(), 1);
  int num_stats = 0;
  xplane_visitor.ForEachLine([&](const XLineVisitor& xline) {
    xline.ForEachEvent([&](const XEventVisitor& xevent) {
      EXPECT_EQ(xevent.Name(), "1st event");
      xevent.ForEachStat([&](const XStatVisitor& stat) {
        if (stat.Name() == "int stat") {
          EXPECT_EQ(stat.IntValue(), 1234LL);
          num_stats++;
        } else if (stat.Name() == "string stat") {
          EXPECT_EQ(stat.StrOrRefValue(), "abc");
          num_stats++;
        } else if (stat.Name() == "double stat") {
          EXPECT_EQ(stat.DoubleValue(), 1.0);
          num_stats++;
        } else if (stat.Name() == "ref stat") {
          EXPECT_EQ(stat.StrOrRefValue(), "referenced abc");
          num_stats++;
        }
      });
    });
  });
  EXPECT_EQ(num_stats, 4);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
