/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
// This class is designed to get accurate profiles for programs

#include "xla/tsl/platform/profile_utils/cpu_utils.h"

#include "xla/tsl/platform/profile_utils/clock_cycle_profiler.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace profile_utils {

static constexpr bool DBG = false;

class CpuUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override { CpuUtils::EnableClockCycleProfiling(); }
};

TEST_F(CpuUtilsTest, SetUpTestCase) {}

TEST_F(CpuUtilsTest, TearDownTestCase) {}

TEST_F(CpuUtilsTest, CheckGetCurrentClockCycle) {
  static constexpr int LOOP_COUNT = 10;
  const uint64 start_clock_count = CpuUtils::GetCurrentClockCycle();
  CHECK_GT(start_clock_count, 0);
  uint64 prev_clock_count = start_clock_count;
  for (int i = 0; i < LOOP_COUNT; ++i) {
    const uint64 clock_count = CpuUtils::GetCurrentClockCycle();
    CHECK_GE(clock_count, prev_clock_count);
    prev_clock_count = clock_count;
  }
  const uint64 end_clock_count = CpuUtils::GetCurrentClockCycle();
  if (DBG) {
    LOG(INFO) << "start clock = " << start_clock_count;
    LOG(INFO) << "end clock = " << end_clock_count;
    LOG(INFO) << "average clock = "
              << ((end_clock_count - start_clock_count) / LOOP_COUNT);
  }
}

TEST_F(CpuUtilsTest, CheckCycleCounterFrequency) {
#if (defined(__powerpc__) ||                                             \
     defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)) || \
    (defined(__s390x__))
  const uint64 cpu_frequency = CpuUtils::GetCycleCounterFrequency();
  CHECK_GT(cpu_frequency, 0);
  CHECK_NE(cpu_frequency, unsigned(CpuUtils::INVALID_FREQUENCY));
#else
  const int64_t cpu_frequency = CpuUtils::GetCycleCounterFrequency();
  CHECK_GT(cpu_frequency, 0);
  CHECK_NE(cpu_frequency, CpuUtils::INVALID_FREQUENCY);
#endif
  if (DBG) {
    LOG(INFO) << "Cpu frequency = " << cpu_frequency;
  }
}

TEST_F(CpuUtilsTest, CheckMicroSecPerClock) {
  const double micro_sec_per_clock = CpuUtils::GetMicroSecPerClock();
  CHECK_GT(micro_sec_per_clock, 0.0);
  if (DBG) {
    LOG(INFO) << "Micro sec per clock = " << micro_sec_per_clock;
  }
}

TEST_F(CpuUtilsTest, SimpleUsageOfClockCycleProfiler) {
  static constexpr int LOOP_COUNT = 10;
  ClockCycleProfiler prof;
  for (int i = 0; i < LOOP_COUNT; ++i) {
    prof.Start();
    prof.Stop();
  }
  EXPECT_EQ(LOOP_COUNT, static_cast<int>(prof.GetCount() + 0.5));
  if (DBG) {
    prof.DumpStatistics("CpuUtilsTest");
  }
}

}  // namespace profile_utils
}  // namespace tsl
