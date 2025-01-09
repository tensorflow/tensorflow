/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/utils/timestamp_utils.h"

#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace profiler {
using ::testing::Eq;

TEST(TimestampUtilsTest, StartAndStopTimestampAreAdded) {
  XSpace xspace;

  SetSessionTimestamps(1000, 2000, xspace);

  const XPlane* xplane = FindPlaneWithName(xspace, kTaskEnvPlaneName);

  XPlaneVisitor visitor(xplane, {}, {FindTaskEnvStatType});

  auto start_time = visitor.GetStat(TaskEnvStatType::kEnvProfileStartTime);
  auto stop_time = visitor.GetStat(TaskEnvStatType::kEnvProfileStopTime);

  EXPECT_THAT(start_time->IntOrUintValue(), Eq(1000));
  EXPECT_THAT(stop_time->IntOrUintValue(), Eq(2000));
}

}  // namespace profiler

}  // namespace tsl
