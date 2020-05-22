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

#include "tensorflow/core/profiler/utils/derived_timeline.h"

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(DerivedTimelineTest, EmptySpaceTest) {
  XSpace space;
  EventGroupNameMap event_group_name_map;
  GenerateDerivedTimeLines(event_group_name_map, &space);
  EXPECT_EQ(space.planes_size(), 0);
}

// Checks that HLO module events are expanded.
TEST(DerivedTimelineTest, HloModuleNameTest) {
  const absl::string_view kHloModuleName = "hlo_module";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  EventGroupNameMap event_group_name_map;
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  auto first_event = CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100);
  first_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kHloModule)),
                           kHloModuleName);
  first_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kKernelDetails)),
                           kKernelDetails);
  auto second_event =
      CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300);
  second_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kHloModule)),
                            kHloModuleName);
  second_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kKernelDetails)),
                            kKernelDetails);
  GenerateDerivedTimeLines(event_group_name_map, &space);
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  // Only the hlo module line is added and other empty lines are removed at the
  // end.
  EXPECT_EQ(plane_visitor.NumLines(), 2);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    EXPECT_EQ(line_visitor.Id(), kThreadIdHloModule);
    EXPECT_EQ(line_visitor.NumEvents(), 1);
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      EXPECT_EQ(event_visitor.Name(), kHloModuleName);
    });
  });
}

// Checks that the TF op events are expanded.
TEST(DerivedTimelineTest, TfOpLineTest) {
  const absl::string_view kTfOpName = "mul:Mul";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  EventGroupNameMap event_group_name_map;
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  auto first_event = CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100);
  first_event.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kLevel0)),
      kTfOpName);
  first_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kKernelDetails)),
                           kKernelDetails);
  auto second_event =
      CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300);
  second_event.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kLevel0)),
      kTfOpName);
  second_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kKernelDetails)),
                            kKernelDetails);
  GenerateDerivedTimeLines(event_group_name_map, &space);
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  // Only the tf op line is added and other empty lines are removed at the end.
  EXPECT_EQ(plane_visitor.NumLines(), 2);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    EXPECT_EQ(line_visitor.Id(), kThreadIdTfOp);
    EXPECT_EQ(line_visitor.NumEvents(), 1);
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      EXPECT_EQ(event_visitor.Name(), kTfOpName);
      EXPECT_EQ(event_visitor.OffsetPs(), 0);
      EXPECT_EQ(event_visitor.DurationPs(), 500);
    });
  });
}

// Checks that the dependency between the step line and the TF op line prevents
// TF op events from being expanded.
TEST(DerivedTimelineTest, DependencyTest) {
  const absl::string_view kTfOpName = "mul:Mul";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  EventGroupNameMap event_group_name_map({{0, "train 0"}, {1, "train 1"}});
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  auto first_event = CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
                                  {{StatType::kGroupId, 0}});
  first_event.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kLevel0)),
      kTfOpName);
  first_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kKernelDetails)),
                           kKernelDetails);
  auto second_event = CreateXEvent(&plane_builder, &line_builder, "op2", 200,
                                   300, {{StatType::kGroupId, 1}});
  second_event.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kLevel0)),
      kTfOpName);
  second_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kKernelDetails)),
                            kKernelDetails);
  GenerateDerivedTimeLines(event_group_name_map, &space);
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  // The step line and the TF op line are added.
  EXPECT_EQ(plane_visitor.NumLines(), 3);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Id() == 0) return;
    EXPECT_TRUE(line_visitor.Id() == kThreadIdStepInfo ||
                line_visitor.Id() == kThreadIdTfOp);
    EXPECT_EQ(line_visitor.NumEvents(), 2);
  });
}

// Checks that the TF op events are expanded.
TEST(DerivedTimelineTest, TfOpNameScopeTest) {
  const absl::string_view kTfOpName = "scope1/scope2/mul:Mul";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  EventGroupNameMap event_group_name_map;
  XPlane* plane = space.add_planes();
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  auto first_event = CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100);
  first_event.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kLevel0)),
      kTfOpName);
  first_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kKernelDetails)),
                           kKernelDetails);
  auto second_event =
      CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300);
  second_event.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kLevel0)),
      kTfOpName);
  second_event.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kKernelDetails)),
                            kKernelDetails);
  GenerateDerivedTimeLines(event_group_name_map, &space);
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  // The TF name scope line and the TF op line are added.
  EXPECT_EQ(plane_visitor.NumLines(), 3);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    int64 line_id = line_visitor.Id();
    if (line_id == 0) {
      return;
    } else if (line_id == kThreadIdTfNameScope) {
      EXPECT_EQ(line_visitor.NumEvents(), 2);
      line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
        EXPECT_EQ(event_visitor.OffsetPs(), 0);
        EXPECT_EQ(event_visitor.DurationPs(), 500);
      });
    } else if (line_id == kThreadIdTfOp) {
      EXPECT_EQ(line_visitor.NumEvents(), 1);
      line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
        EXPECT_EQ(event_visitor.Name(), kTfOpName);
        EXPECT_EQ(event_visitor.OffsetPs(), 0);
        EXPECT_EQ(event_visitor.DurationPs(), 500);
      });
    }
  });
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
