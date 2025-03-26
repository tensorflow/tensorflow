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
#include "tensorflow/core/profiler/convert/duty_cycle_combiner.h"

#include <gtest/gtest.h>
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/convert/duty_cycle_tracker.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::Timespan;

TEST(DutyCycleAnalysisTest, CombineMultiCoreChipTest) {
  DutyCycleTracker core0_tracker;
  core0_tracker.AddInterval(Timespan::FromEndPoints(10, 20), true);
  core0_tracker.AddInterval(Timespan::FromEndPoints(20, 30), false);
  DutyCycleTracker core1_tracker;
  core1_tracker.AddInterval(Timespan::FromEndPoints(10, 20), false);
  core1_tracker.AddInterval(Timespan::FromEndPoints(20, 30), true);

  DutyCycleCombiner combiner;
  combiner.CombineCore(core0_tracker, 0);
  combiner.CombineCore(core1_tracker, 0);

  EXPECT_EQ(combiner.GetTotalActiveTimePs(), 20);
  EXPECT_EQ(combiner.GetTotalIdleTimePs(), 0);
}

TEST(DutyCycleAnalysisTest, CombineMultiChipTest) {
  DutyCycleTracker chip0_tracker;
  chip0_tracker.AddInterval(Timespan::FromEndPoints(10, 20), true);
  chip0_tracker.AddInterval(Timespan::FromEndPoints(20, 30), false);
  DutyCycleTracker chip1_tracker;
  chip1_tracker.AddInterval(Timespan::FromEndPoints(10, 20), true);
  chip1_tracker.AddInterval(Timespan::FromEndPoints(20, 30), false);

  DutyCycleCombiner combiner;
  combiner.CombineChip(chip0_tracker);
  combiner.CombineChip(chip1_tracker);

  EXPECT_EQ(combiner.GetTotalActiveTimePs(), 20);
  EXPECT_EQ(combiner.GetTotalIdleTimePs(), 20);
}

TEST(DutyCycleAnalysisTest, CombineMultiChipAndCoreTest) {
  DutyCycleTracker chip0_core0_tracker;
  chip0_core0_tracker.AddInterval(Timespan::FromEndPoints(10, 20), false);
  chip0_core0_tracker.AddInterval(Timespan::FromEndPoints(20, 30), true);
  DutyCycleTracker chip0_core1_tracker;
  chip0_core1_tracker.AddInterval(Timespan::FromEndPoints(10, 20), true);
  chip0_core1_tracker.AddInterval(Timespan::FromEndPoints(20, 30), false);
  DutyCycleTracker chip1_tracker;
  chip1_tracker.AddInterval(Timespan::FromEndPoints(15, 25), true);
  chip1_tracker.AddInterval(Timespan::FromEndPoints(10, 30), false);

  DutyCycleCombiner combiner;
  combiner.CombineCore(chip0_core0_tracker, 0);
  combiner.CombineCore(chip0_core1_tracker, 0);
  combiner.CombineChip(chip1_tracker);

  EXPECT_EQ(combiner.GetTotalActiveTimePs(), 30);
  EXPECT_EQ(combiner.GetTotalIdleTimePs(), 10);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
