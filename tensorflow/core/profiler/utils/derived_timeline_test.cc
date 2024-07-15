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

#include <map>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/utils/group_events.h"
#include "tsl/profiler/utils/tf_xplane_visitor.h"
#include "tsl/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(DerivedTimelineTest, EmptySpaceTest) {
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  GenerateDerivedTimeLines(group_metadata_map, &space);
  EXPECT_EQ(space.planes_size(), 0);
}

// Checks that HLO module events are expanded.
TEST(DerivedTimelineTest, HloModuleNameTest) {
  const absl::string_view kHloModuleName = "hlo_module";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kHloModule, kHloModuleName},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kHloModule, kHloModuleName},
                {StatType::kKernelDetails, kKernelDetails}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
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
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
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
  constexpr int64_t kFirstGroupId = 0;
  constexpr int64_t kSecondGroupId = 1;

  const absl::string_view kTfOpName = "mul:Mul";
  const absl::string_view kKernelDetails = "kernel_details";
  XSpace space;
  tsl::profiler::GroupMetadataMap group_metadata_map(
      {{0, {"train 0"}}, {1, {"train 1"}}});
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kGroupId, kFirstGroupId},
                {StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kGroupId, kSecondGroupId},
                {StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
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
  tsl::profiler::GroupMetadataMap group_metadata_map;
  XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder plane_builder(plane);
  auto line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 300,
               {{StatType::kTfOp, kTfOpName},
                {StatType::kKernelDetails, kKernelDetails}});
  GenerateDerivedTimeLines(group_metadata_map, &space);
  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
  // The TF name scope line and the TF op line are added.
  EXPECT_EQ(plane_visitor.NumLines(), 3);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    int64_t line_id = line_visitor.Id();
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

// Checks that the TF op events are expanded.
TEST(DerivedTimelineTest, TfOpNameScopeShrinkTest) {
  {
    // Case 1: shirnk is possible.
    XSpace space;
    tsl::profiler::GroupMetadataMap group_metadata_map;
    XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
    XPlaneBuilder plane_builder(plane);
    auto line_builder = plane_builder.GetOrCreateLine(0);
    CreateXEvent(&plane_builder, &line_builder, "op1", 0, 10000,
                 {{StatType::kTfOp, "a/b/c/Add:Add"},
                  {StatType::kKernelDetails, "blah"}});
    CreateXEvent(
        &plane_builder, &line_builder, "op2", 20000, 30000,
        {{StatType::kTfOp, "a/d/Mul:Mul"}, {StatType::kKernelDetails, "blah"}});
    GenerateDerivedTimeLines(group_metadata_map, &space);
    XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
    // The TF name scope line and the TF op line are added.
    EXPECT_EQ(plane_visitor.NumLines(), 3);
    plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
      int64_t line_id = line_visitor.Id();
      if (line_id == 0) {
        return;
      } else if (line_id == kThreadIdTfNameScope) {
        EXPECT_EQ(line_visitor.NumEvents(), 4);
        std::map<absl::string_view, uint64_t> durations;
        line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
          durations[event_visitor.Name()] = event_visitor.DurationPs();
        });
        EXPECT_EQ(durations["a"], 50000);
        EXPECT_EQ(durations["b"], 10000);
        EXPECT_EQ(durations["c"], 9000);  // shrinked to be distinguish with b.
        EXPECT_EQ(durations["d"], 30000);
      }
    });
  }
  {
    // Case 2: shirnk is impossible due to top event is too small.
    XSpace space;
    tsl::profiler::GroupMetadataMap group_metadata_map;
    XPlane* plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
    XPlaneBuilder plane_builder(plane);
    auto line_builder = plane_builder.GetOrCreateLine(0);
    CreateXEvent(&plane_builder, &line_builder, "op1", 0, 10000,
                 {{StatType::kTfOp, "a/b/c/d/e/Add:Add"},
                  {StatType::kKernelDetails, "blah"}});
    CreateXEvent(&plane_builder, &line_builder, "op2", 10000, 2000,
                 {{StatType::kTfOp, "a/b/c/d/f/Sub:Sub"},
                  {StatType::kKernelDetails, "blah"}});
    CreateXEvent(
        &plane_builder, &line_builder, "op3", 20000, 30000,
        {{StatType::kTfOp, "a/g/Mul:Mul"}, {StatType::kKernelDetails, "blah"}});
    GenerateDerivedTimeLines(group_metadata_map, &space);
    XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(plane);
    // The TF name scope line and the TF op line are added.
    EXPECT_EQ(plane_visitor.NumLines(), 3);
    plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
      int64_t line_id = line_visitor.Id();
      if (line_id == 0) {
        return;
      } else if (line_id == kThreadIdTfNameScope) {
        EXPECT_EQ(line_visitor.NumEvents(), 7);
        std::map<absl::string_view, uint64_t> durations;
        line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
          durations[event_visitor.Name()] = event_visitor.DurationPs();
        });
        for (const auto& [name, duration] : durations) {
          LOG(ERROR) << name << ": " << duration;
        }
        EXPECT_EQ(durations["a"], 50000);
        EXPECT_EQ(durations["b"], 12000);
        EXPECT_EQ(durations["c"], 11000);  // shrinked to be distinguish with b.
        EXPECT_EQ(durations["d"], 11000);  // not shrinked because of f.
        EXPECT_EQ(durations["e"], 10000);
        EXPECT_EQ(durations["f"], 1000);
        EXPECT_EQ(durations["g"], 30000);
      }
    });
  }
}

TEST(DerivedTimelineTest, DeriveLinesForXlaCpuOps) {
  XPlane xplane;
  XPlaneBuilder plane_builder(&xplane);
  plane_builder.SetName(tsl::profiler::kHostThreadsPlaneName);

  absl::string_view main_line_name = "main";
  auto line_builder = plane_builder.GetOrCreateLine(0);
  line_builder.SetName(main_line_name);
  CreateXEvent(&plane_builder, &line_builder, "op1", 0, 100,
               {{StatType::kHloModule, "Module1"}});
  CreateXEvent(&plane_builder, &line_builder, "op2", 200, 400,
               {{StatType::kHloModule, "Module2"}});

  DeriveLinesForXlaCpuOps(&xplane);

  XPlaneVisitor plane_visitor = tsl::profiler::CreateTfXPlaneVisitor(&xplane);
  EXPECT_EQ(plane_visitor.NumLines(), 2);
  plane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    if (line_visitor.Name() == main_line_name) return;
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      if (event_visitor.Name() == "Module1") {
        EXPECT_EQ(event_visitor.DurationPs(), 100);
        EXPECT_EQ(event_visitor.OffsetPs(), 0);
      } else if (event_visitor.Name() == "Module2") {
        EXPECT_EQ(event_visitor.DurationPs(), 400);
        EXPECT_EQ(event_visitor.OffsetPs(), 200);
      } else {
        FAIL() << "Found Event " << event_visitor.Name();
      }
    });
  });
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
