/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"

#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/platform/test.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::testing::Optional;
using ::testing::UnorderedElementsAre;

TEST(TpuXPlaneUtilsTest, GetTensorCoreXPlanesFromXSpace) {
  XSpace xspace;

  XPlane* p1 = FindOrAddMutablePlaneWithName(&xspace, TpuPlaneName(0));
  XPlane* p2 = FindOrAddMutablePlaneWithName(&xspace, TpuPlaneName(1));
  FindOrAddMutablePlaneWithName(&xspace, TpuPlaneName(2) + "Postfix");

  std::vector<const XPlane*> xplanes = FindTensorCorePlanes(xspace);

  EXPECT_THAT(xplanes, UnorderedElementsAre(p1, p2));
}

TEST(TpuXPlaneUtilsTest, GetMutableTensorCoreXPlanesFromXSpace) {
  XSpace xspace;
  XPlane* p1 = FindOrAddMutablePlaneWithName(&xspace, TpuPlaneName(0));
  XPlane* p2 = FindOrAddMutablePlaneWithName(&xspace, TpuPlaneName(1));

  FindOrAddMutablePlaneWithName(&xspace, TpuPlaneName(2) + "Postfix");

  std::vector<XPlane*> xplanes = FindMutableTensorCorePlanes(&xspace);

  EXPECT_THAT(xplanes, UnorderedElementsAre(p1, p2));
}

TEST(TpuXPlaneUtilsTest, GetTensorCoreIdFromPlaneName) {
  EXPECT_EQ(GetTensorCoreId(TpuPlaneName(0)), 0);
}

TEST(TpuXPlaneUtilsTest, IsNotTensorCorePlaneName) {
  EXPECT_FALSE(GetTensorCoreId("/metadata:0").has_value());
}

TEST(TpuXPlaneUtilsTest, IsNotTensorCorePlaneNameWithPrefix) {
  EXPECT_FALSE(
      GetTensorCoreId(absl::StrCat("/prefix", TpuPlaneName(0))).has_value());
}

TEST(TpuXplaneUtilsTest, GetSparseCorePlanesFromXSpace) {
  XSpace space;
  XPlane* p1 = FindOrAddMutablePlaneWithName(&space, TpuPlaneName(0));
  XPlane* p2 = FindOrAddMutablePlaneWithName(&space, TpuPlaneName(1));
  XPlane* p3 = FindOrAddMutablePlaneWithName(
      &space, absl::StrCat(TpuPlaneName(0), " SparseCore 0"));
  XPlane* p4 = FindOrAddMutablePlaneWithName(
      &space, absl::StrCat(TpuPlaneName(0), " SparseCore 1"));

  EXPECT_THAT(FindTensorCorePlanes(space), UnorderedElementsAre(p1, p2));
  EXPECT_THAT(FindPlanesWithPrefix(space, kTpuPlanePrefix),
              UnorderedElementsAre(p1, p2, p3, p4));
  EXPECT_THAT(GetSparseCoreId(p3->name()), Optional(0));
  EXPECT_THAT(GetSparseCoreId(p4->name()), Optional(1));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
