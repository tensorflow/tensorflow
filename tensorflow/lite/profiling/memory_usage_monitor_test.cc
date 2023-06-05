/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/profiling/memory_usage_monitor.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/lite/profiling/memory_info.h"

namespace tflite {
namespace profiling {
namespace memory {

class MemoryUsageNotSupportedSampler : public MemoryUsageMonitor::Sampler {
 public:
  bool IsSupported() override { return false; }
};

TEST(MemoryUsageMonitor, NotSupported) {
  MemoryUsageMonitor monitor1(50, std::unique_ptr<MemoryUsageMonitor::Sampler>(
                                      new MemoryUsageNotSupportedSampler()));
  EXPECT_FLOAT_EQ(MemoryUsageMonitor::kInvalidMemUsageMB,
                  monitor1.GetPeakMemUsageInMB());

  MemoryUsageMonitor monitor2(50, nullptr);
  EXPECT_FLOAT_EQ(MemoryUsageMonitor::kInvalidMemUsageMB,
                  monitor2.GetPeakMemUsageInMB());
}

// Just smoke tests for different call combinations.
class MemoryUsageMonitorTest : public ::testing::Test {
 protected:
  class FakeMemoryUsageSampler : public MemoryUsageMonitor::Sampler {
   public:
    explicit FakeMemoryUsageSampler(int64_t* num_sleeps)
        : sleep_cnt_(num_sleeps) {}
    bool IsSupported() override { return true; }
    MemoryUsage GetMemoryUsage() override {
      MemoryUsage result;
      result.mem_footprint_kb = 5 * ((*sleep_cnt_) + 1) * 1024;
      return result;
    }
    void SleepFor(const absl::Duration& duration) override {
      (*sleep_cnt_)++;
      absl::SleepFor(duration);
    }

   private:
    int64_t* const sleep_cnt_ = nullptr;
  };

  void SetUp() override {
    monitor_ = std::make_unique<MemoryUsageMonitor>(
        50, std::unique_ptr<MemoryUsageMonitor::Sampler>(
                new FakeMemoryUsageSampler(&num_sleeps_)));
  }

  int64_t num_sleeps_ = 0;
  std::unique_ptr<MemoryUsageMonitor> monitor_ = nullptr;
};

TEST_F(MemoryUsageMonitorTest, StartAndStop) {
  monitor_->Start();
  monitor_->Stop();
  EXPECT_FLOAT_EQ(5.0 * (num_sleeps_ + 1), monitor_->GetPeakMemUsageInMB());
}

TEST_F(MemoryUsageMonitorTest, NoStartAndStop) {
  monitor_->Stop();
  EXPECT_FLOAT_EQ(MemoryUsageMonitor::kInvalidMemUsageMB,
                  monitor_->GetPeakMemUsageInMB());
}

TEST_F(MemoryUsageMonitorTest, StartAndNoStop) {
  monitor_->Start();
  EXPECT_FLOAT_EQ(MemoryUsageMonitor::kInvalidMemUsageMB,
                  monitor_->GetPeakMemUsageInMB());
}

TEST_F(MemoryUsageMonitorTest, StopFirst) {
  monitor_->Stop();
  EXPECT_FLOAT_EQ(MemoryUsageMonitor::kInvalidMemUsageMB,
                  monitor_->GetPeakMemUsageInMB());
  monitor_->Start();
  EXPECT_FLOAT_EQ(MemoryUsageMonitor::kInvalidMemUsageMB,
                  monitor_->GetPeakMemUsageInMB());
}

TEST_F(MemoryUsageMonitorTest, MultiStartAndStops) {
  monitor_->Start();
  monitor_->Start();
  monitor_->Stop();
  monitor_->Stop();
  EXPECT_FLOAT_EQ(5.0 * (num_sleeps_ + 1), monitor_->GetPeakMemUsageInMB());
}

TEST_F(MemoryUsageMonitorTest, StartStopPairs) {
  monitor_->Start();
  monitor_->Stop();
  EXPECT_FLOAT_EQ(5.0 * (num_sleeps_ + 1), monitor_->GetPeakMemUsageInMB());

  monitor_->Start();
  // Sleep for at least for a duration that's longer than the sampling interval
  // passed to 'monitor_' (i.e. 50 ms) to simulate the memory usage increase.
  absl::SleepFor(absl::Milliseconds(100));
  monitor_->Stop();
  EXPECT_GE(num_sleeps_, 1);
  EXPECT_FLOAT_EQ(5.0 * (num_sleeps_ + 1), monitor_->GetPeakMemUsageInMB());
}

}  // namespace memory
}  // namespace profiling
}  // namespace tflite
