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

#include "tensorflow/core/profiler/utils/step_intersection.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profiler {
namespace {

using PerHostStepDb =
    absl::flat_hash_map<uint32 /*=host_id*/, StepDatabaseResult>;

constexpr uint64 kStepDurationPs = 2000000000;
constexpr uint32 kNumStepsPerHost = 10;
constexpr uint64 kStepGapPs = 0;
constexpr uint32 kNumCoresPerHost = 8;

PerCoreStepInfo CreateOneTestStep(uint32 host_id, uint32 num_steps,
                                  uint32 step_idx, uint64 step_begin_ps) {
  PerCoreStepInfo result;
  uint32 step_num =
      step_idx * host_id;  // creates the situation where each host has a
                           // different step number for the same step.
  result.set_step_num(step_num);
  StepInfoResult info;
  info.set_step_num(step_num);
  if (host_id == 0 && step_idx == (num_steps - 1)) {
    // Makes the last step on host_id is little bit shorter so that host-0 will
    // be chosen as the chief.
    info.set_duration_ps(kStepDurationPs - 1);
  } else {
    info.set_duration_ps(kStepDurationPs);
  }
  info.set_begin_ps(step_begin_ps);
  // Don't care about the rest of the fields in StepInfoResult.
  for (uint32 core_id = 0; core_id < kNumCoresPerHost; core_id++) {
    (*result.mutable_step_info_per_core())[core_id] = info;
    // Don't care about the rest of the fields in PerCoreStepInfo.
  }
  return result;
}

PerHostStepDb CreateTestSteps(uint32 num_hosts, uint64 shift_ps) {
  PerHostStepDb result;
  uint64 first_step_begin_ps = 0;
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    StepDatabaseResult step_db;
    uint64 step_begin_ps = first_step_begin_ps;
    for (uint32 step_idx = 0; step_idx < kNumStepsPerHost; step_idx++) {
      *step_db.add_step_sequence() =
          CreateOneTestStep(host_id, kNumStepsPerHost, step_idx, step_begin_ps);
      step_begin_ps += (kStepDurationPs + kStepGapPs);
    }
    result[host_id] = step_db;
    first_step_begin_ps += shift_ps;
  }
  return result;
}

PerHostStepDb CreateEmptyIntersectTestSteps() {
  PerHostStepDb result;

  uint64 step_begin_ps;
  uint32 host_id;

  // Host-0
  host_id = 0;
  step_begin_ps = 0;
  uint64 host_0_num_steps = 10;
  StepDatabaseResult step_db_0;
  for (uint32 step_idx = 0; step_idx < host_0_num_steps; step_idx++) {
    *step_db_0.add_step_sequence() =
        CreateOneTestStep(host_id, host_0_num_steps, step_idx, step_begin_ps);
    step_begin_ps += (kStepDurationPs + kStepGapPs);
  }
  result[host_id] = step_db_0;

  // Host-1
  host_id = 1;
  step_begin_ps = (host_0_num_steps - 2) * (kStepDurationPs + kStepGapPs);
  uint64 host_1_num_steps = 5;
  StepDatabaseResult step_db_1;
  for (uint32 step_idx = 0; step_idx < host_1_num_steps; step_idx++) {
    *step_db_1.add_step_sequence() =
        CreateOneTestStep(host_id, host_1_num_steps, step_idx, step_begin_ps);
    step_begin_ps += (kStepDurationPs + kStepGapPs);
  }
  result[host_id] = step_db_1;

  // Host-2
  host_id = 2;
  step_begin_ps = (host_0_num_steps + host_1_num_steps - 4) *
                  (kStepDurationPs + kStepGapPs);
  uint64 host_2_num_steps = 10;
  StepDatabaseResult step_db_2;
  for (uint32 step_idx = 0; step_idx < host_2_num_steps; step_idx++) {
    *step_db_2.add_step_sequence() =
        CreateOneTestStep(host_id, host_2_num_steps, step_idx, step_begin_ps);
    step_begin_ps += (kStepDurationPs + kStepGapPs);
  }
  result[host_id] = step_db_2;

  return result;
}

PerHostStepDb CreateNoStep(uint32 num_hosts) {
  PerHostStepDb result;
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    StepDatabaseResult step_db;
    result[host_id] = step_db;
  }
  return result;
}

absl::flat_hash_map<uint32 /*=host_id*/, const StepDatabaseResult*> Convert(
    const PerHostStepDb& perhost_stepdb) {
  absl::flat_hash_map<uint32 /*=host_id*/, const StepDatabaseResult*> result;
  for (const auto& hostid_stepdb : perhost_stepdb) {
    auto host_id = hostid_stepdb.first;
    const auto& step_db = hostid_stepdb.second;
    result[host_id] = &step_db;
  }
  return result;
}

TEST(StepIntersectionTest, EachHostShiftedBy1StepDuration) {
  uint32 num_hosts = 4;
  uint64 shift_ps = kStepDurationPs;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32 dst_num_steps = kNumStepsPerHost - num_hosts + 1;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  uint32 src_first_step_index = intersection.FirstStepIndex(0);
  EXPECT_EQ(src_first_step_index, num_hosts - 1);
  std::vector<uint32> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32 i = 0; i < dst_num_steps; i++) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
}

TEST(StepIntersectionTest, ExactlyNoShift) {
  uint32 num_hosts = 4;
  uint64 shift_ps = 0;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32 dst_num_steps = kNumStepsPerHost;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  std::vector<uint32> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32 i = 0; i < dst_num_steps; i++) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    uint32 src_first_step_index = intersection.FirstStepIndex(host_id);
    EXPECT_EQ(src_first_step_index, 0);
  }
}

TEST(StepIntersectionTest, EachHostShiftedByJustABit) {
  uint32 num_hosts = 4;
  uint64 shift_ps = 100;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32 dst_num_steps = kNumStepsPerHost;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  std::vector<uint32> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32 i = 0; i < dst_num_steps; i++) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    uint32 src_first_step_index = intersection.FirstStepIndex(host_id);
    EXPECT_EQ(src_first_step_index, 0);
  }
}

TEST(StepIntersectionTest, SingleHost) {
  uint32 num_hosts = 1;
  uint64 shift_ps = 0;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32 dst_num_steps = kNumStepsPerHost;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  std::vector<uint32> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32 i = 0; i < dst_num_steps; i++) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    uint32 src_first_step_index = intersection.FirstStepIndex(host_id);
    EXPECT_EQ(src_first_step_index, 0);
  }
}

TEST(StepIntersectionTest, WithMaxSteps) {
  uint32 num_hosts = 4;
  uint64 shift_ps = 0;
  uint32 max_steps = 3;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), kNumStepsPerHost - max_steps);
  EXPECT_EQ(intersection.NumSteps(), max_steps);
}

TEST(StepIntersectionTest, NoStep) {
  uint32 num_hosts = 4;
  uint32 max_steps = 100;
  PerHostStepDb perhost_stepdb = CreateNoStep(num_hosts);
  StepIntersection intersection =
      StepIntersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.NumSteps(), 0);
  EXPECT_FALSE(intersection.EmptyIntersect());
}

TEST(StepIntersectionTest, EmptyIntersection) {
  uint32 max_steps = 100;
  PerHostStepDb perhost_stepdb = CreateEmptyIntersectTestSteps();
  StepIntersection intersection =
      StepIntersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  EXPECT_EQ(intersection.NumSteps(), 0);
  EXPECT_TRUE(intersection.EmptyIntersect());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
