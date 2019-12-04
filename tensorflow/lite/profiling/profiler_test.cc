/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <unistd.h>

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <thread>  // NOLINT(build/c++11)

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace profiling {
namespace {

double GetDurationOfEventMs(const ProfileEvent* event) {
  return (event->end_timestamp_us - event->begin_timestamp_us) / 1e3;
}

void SleepForQuarterSecond(tflite::Profiler* profiler) {
  ScopedProfile profile(profiler, "SleepForQuarter");
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

void ChildFunction(tflite::Profiler* profiler) {
  ScopedProfile profile(profiler, "Child");
  SleepForQuarterSecond(profiler);
}

void ParentFunction(tflite::Profiler* profiler) {
  ScopedProfile profile(profiler, "Parent");
  for (int i = 0; i < 2; i++) {
    ChildFunction(profiler);
  }
}

TEST(ProfilerTest, NoProfilesAreCollectedWhenDisabled) {
  BufferedProfiler profiler(1024);
  ParentFunction(&profiler);
  auto profile_events = profiler.GetProfileEvents();
  EXPECT_EQ(0, profile_events.size());
}

TEST(ProfilingTest, ProfilesAreCollected) {
  BufferedProfiler profiler(1024);
  profiler.StartProfiling();
  ParentFunction(&profiler);
  profiler.StopProfiling();
  auto profile_events = profiler.GetProfileEvents();
  // ParentFunction calls the ChildFunction 2 times.
  // Each ChildFunction calls SleepForQuarterSecond once.
  // We expect 1 entry for ParentFunction, 2 for ChildFunction and 2 for
  // SleepForQuarterSecond: Total: 1+ 2 + 2 = 5
  //  Profiles should look like:
  //  Parent ~ 500 ms (due to 2 Child calls)
  //   - Child ~ 250 ms (due to SleepForQuarter calls)
  //       - SleepForQuarter ~ 250ms
  //   - Child ~ 250 ms (due to SleepForQuarter calls)
  //      - SleepForQuarter ~ 250ms
  //
  ASSERT_EQ(5, profile_events.size());
  EXPECT_EQ("Parent", profile_events[0]->tag);
  EXPECT_EQ("Child", profile_events[1]->tag);
  EXPECT_EQ("SleepForQuarter", profile_events[2]->tag);
  EXPECT_EQ("Child", profile_events[3]->tag);
  EXPECT_EQ("SleepForQuarter", profile_events[4]->tag);

#ifndef ADDRESS_SANITIZER
  // ASAN build is sometimes very slow. Set a large epsilon to avoid flakiness.
  // Due to flakiness, just verify relative values match.
  const int eps_ms = 50;
  auto parent_ms = GetDurationOfEventMs(profile_events[0]);
  double child_ms[2], sleep_for_quarter_ms[2];
  child_ms[0] = GetDurationOfEventMs(profile_events[1]);
  child_ms[1] = GetDurationOfEventMs(profile_events[3]);
  sleep_for_quarter_ms[0] = GetDurationOfEventMs(profile_events[2]);
  sleep_for_quarter_ms[1] = GetDurationOfEventMs(profile_events[4]);
  EXPECT_NEAR(parent_ms, child_ms[0] + child_ms[1], eps_ms);
  EXPECT_NEAR(child_ms[0], sleep_for_quarter_ms[0], eps_ms);
  EXPECT_NEAR(child_ms[1], sleep_for_quarter_ms[1], eps_ms);
#endif
}

TEST(ProfilingTest, NullProfiler) {
  Profiler* profiler = nullptr;
  { SCOPED_TAGGED_OPERATOR_PROFILE(profiler, "noop", 1); }
}

TEST(ProfilingTest, ScopedProfile) {
  BufferedProfiler profiler(1024);
  profiler.StartProfiling();
  { SCOPED_TAGGED_OPERATOR_PROFILE(&profiler, "noop", 1); }
  profiler.StopProfiling();
  auto profile_events = profiler.GetProfileEvents();
  EXPECT_EQ(1, profile_events.size());
}

TEST(ProfilingTest, NoopProfiler) {
  NoopProfiler profiler;
  profiler.StartProfiling();
  { SCOPED_TAGGED_OPERATOR_PROFILE(&profiler, "noop", 1); }
  profiler.StopProfiling();
  auto profile_events = profiler.GetProfileEvents();
  EXPECT_EQ(0, profile_events.size());
}

}  // namespace
}  // namespace profiling
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
