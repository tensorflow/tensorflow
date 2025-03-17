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
#include "tensorflow/core/profiler/convert/duty_cycle_tracker.h"

#include <sys/types.h>

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::Timespan;

TEST(DutyCycleTrackerTest, NonOverlappingIntervalsTest) {
  DutyCycleTracker tracker;
  tracker.AddInterval(Timespan::FromEndPoints(10, 20), true);
  tracker.AddInterval(Timespan::FromEndPoints(30, 40), true);
  EXPECT_EQ(tracker.GetActiveTimePs(), 20);
  EXPECT_EQ(tracker.GetIdleTimePs(), 10);
  EXPECT_EQ(tracker.GetDurationPs(), 30);
  EXPECT_NEAR(tracker.DutyCycle(), 0.6666, 0.0001);
}

TEST(DutyCycleTrackerTest, OverlappingIntervalsTest) {
  DutyCycleTracker tracker;
  tracker.AddInterval(Timespan::FromEndPoints(10, 20), true);
  tracker.AddInterval(Timespan::FromEndPoints(30, 40), true);
  tracker.AddInterval(Timespan::FromEndPoints(20, 35), true);
  EXPECT_EQ(tracker.GetActiveTimePs(), 30);
  EXPECT_EQ(tracker.GetIdleTimePs(), 0);
  EXPECT_EQ(tracker.GetDurationPs(), 30);
  EXPECT_EQ(tracker.DutyCycle(), 1.0);
}

TEST(DutyCycleTrackerTest, DutyCycleTestWithIncludedIntervals) {
  DutyCycleTracker tracker;
  tracker.AddInterval(Timespan::FromEndPoints(10, 40), true);
  tracker.AddInterval(Timespan::FromEndPoints(20, 30), true);
  EXPECT_EQ(tracker.GetActiveTimePs(), 30);
  EXPECT_EQ(tracker.GetIdleTimePs(), 0);
  EXPECT_EQ(tracker.GetDurationPs(), 30);
  EXPECT_EQ(tracker.DutyCycle(), 1.0);
}

TEST(DutyCycleTrackerTest, UnionTest) {
  DutyCycleTracker tracker;
  tracker.AddInterval(Timespan::FromEndPoints(0, 10), true);
  tracker.AddInterval(Timespan::FromEndPoints(20, 30), true);

  DutyCycleTracker other_tracker;
  other_tracker.AddInterval(Timespan::FromEndPoints(10, 20), true);
  other_tracker.AddInterval(Timespan::FromEndPoints(30, 40), true);

  tracker.Union(other_tracker);
  EXPECT_EQ(tracker.GetActiveTimePs(), 40);
  EXPECT_EQ(tracker.GetIdleTimePs(), 0);
  EXPECT_EQ(tracker.GetDurationPs(), 40);
}

TEST(DutyCycleTrackerTest, OverlappingMixedIntervalsTest) {
  DutyCycleTracker tracker;
  EXPECT_EQ(tracker.GetActiveTimePs(), 0);
  tracker.AddInterval(Timespan::FromEndPoints(10, 20), true);
  tracker.AddInterval(Timespan::FromEndPoints(20, 30), false);
  EXPECT_EQ(tracker.GetActiveTimePs(), 10);
  EXPECT_EQ(tracker.GetIdleTimePs(), 10);
}

void BM_DutyCycleTracker_AddInterval(::testing::benchmark::State& state) {
  std::vector<Timespan> timespans;
  timespans.reserve(state.range(0));
  for (uint64_t i = 0; i < state.range(0); ++i) {
    timespans.push_back(Timespan::FromEndPoints(i * 2, i * 2 + 1));
  }
  for (auto s : state) {
    DutyCycleTracker tracker;
    for (const auto& timespan : timespans) {
      tracker.AddInterval(timespan, true);
    }
  }
  state.SetItemsProcessed(state.iterations() * timespans.size());
}

BENCHMARK(BM_DutyCycleTracker_AddInterval)->Range(1 << 15, 1 << 21);

void BM_DutyCycleTracker_AddInterval_Merge(::testing::benchmark::State& state) {
  std::vector<Timespan> timespans;
  timespans.reserve(state.range(0));
  for (uint64_t i = 0; i < state.range(0); ++i) {
    timespans.push_back(Timespan::FromEndPoints(i, i + 1));
  }
  for (auto s : state) {
    DutyCycleTracker tracker;
    for (const auto& timespan : timespans) {
      tracker.AddInterval(timespan, true);
    }
  }
  state.SetItemsProcessed(state.iterations() * timespans.size());
}

BENCHMARK(BM_DutyCycleTracker_AddInterval_Merge)->Range(1 << 15, 1 << 21);

void BM_DutyCycleTracker_Union(::testing::benchmark::State& state) {
  DCHECK_GT(state.range(1), 1);
  DCHECK_LT(state.range(1), state.range(0));
  DutyCycleTracker tracker_a;
  DutyCycleTracker tracker_b;
  uint64_t merge_rate = state.range(1);
  for (uint64_t i = 0; i < state.range(0); ++i) {
    tracker_a.AddInterval(Timespan(i * 2, 1), true);
    if (i % merge_rate == 0) {
      tracker_b.AddInterval(Timespan(i * 2 + 1, merge_rate * 2 - 1), true);
    }
  }
  for (auto s : state) {
    DutyCycleTracker unioned_tracker;
    unioned_tracker.Union(tracker_a);
    unioned_tracker.Union(tracker_b);
  }
  state.SetItemsProcessed(state.iterations() *
                          (state.range(0) + state.range(0) / merge_rate));
}

BENCHMARK(BM_DutyCycleTracker_Union)->RangePair(1 << 10, 1 << 16, 2, 10);

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
