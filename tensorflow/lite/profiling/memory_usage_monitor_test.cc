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

#include <atomic>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
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
    explicit FakeMemoryUsageSampler(
        std::atomic<int64_t>* num_sleeps,
        absl::Notification* first_sample_notification)
        : sleep_cnt_(num_sleeps),
          first_sample_notification_(first_sample_notification) {
      notification_called_.clear();
    }
    bool IsSupported() override { return true; }
    MemoryUsage GetMemoryUsage() override {
      MemoryUsage result;
      result.mem_footprint_kb = 5 * (sleep_cnt_->load() + 1) * 1024;
      if (!notification_called_.test_and_set()) {
        first_sample_notification_->Notify();
      }
      return result;
    }
    void SleepFor(const absl::Duration& duration) override {
      absl::SleepFor(duration);
      sleep_cnt_->fetch_add(1);
    }

   private:
    std::atomic<int64_t>* const sleep_cnt_ = nullptr;
    absl::Notification* first_sample_notification_ = nullptr;
    std::atomic_flag notification_called_;
  };

  void SetUp() override {
    first_sample_notification_ = std::make_unique<absl::Notification>();
    monitor_ = std::make_unique<MemoryUsageMonitor>(
        /*sampling_interval_ms=*/50,
        std::unique_ptr<MemoryUsageMonitor::Sampler>(new FakeMemoryUsageSampler(
            &num_sleeps_, first_sample_notification_.get())));
  }

  void WaitForPeakSync(float* sync_peak = nullptr,
                       int64_t* sync_sleeps = nullptr) {
    float peak;
    int64_t sleeps;
    auto condition = [&peak, &sleeps, this]() {
      peak = monitor_->GetPeakMemUsageInMB();
      sleeps = num_sleeps_.load();
      return peak == 5.0 * (sleeps + 1);
    };
    // Use a timeout to avoid infinite loops.
    absl::Time deadline = absl::Now() + absl::Seconds(3);
    while (!condition() && absl::Now() < deadline) {
      absl::SleepFor(absl::Milliseconds(10));
    }
    if (sync_peak) *sync_peak = peak;
    if (sync_sleeps) *sync_sleeps = sleeps;
  }

  std::atomic<int64_t> num_sleeps_{0};
  std::unique_ptr<absl::Notification> first_sample_notification_;
  std::unique_ptr<MemoryUsageMonitor> monitor_ = nullptr;
};

TEST_F(MemoryUsageMonitorTest, StartAndStop) {
  monitor_->Start();
  first_sample_notification_->WaitForNotificationWithTimeout(absl::Seconds(1));
  float peak;
  int64_t sleeps;
  WaitForPeakSync(&peak, &sleeps);
  monitor_->Stop();
  EXPECT_FLOAT_EQ(5.0 * (sleeps + 1), peak);
}

TEST_F(MemoryUsageMonitorTest, NoStartAndStop) {
  monitor_->Stop();
  EXPECT_FLOAT_EQ(MemoryUsageMonitor::kInvalidMemUsageMB,
                  monitor_->GetPeakMemUsageInMB());
}

TEST_F(MemoryUsageMonitorTest, StartAndNoStop) {
  monitor_->Start();
  first_sample_notification_->WaitForNotificationWithTimeout(absl::Seconds(1));
  float peak;
  int64_t sleeps;
  WaitForPeakSync(&peak, &sleeps);
  EXPECT_FLOAT_EQ(5.0 * (sleeps + 1), peak);
}

TEST_F(MemoryUsageMonitorTest, StopFirst) {
  monitor_->Stop();
  EXPECT_FLOAT_EQ(MemoryUsageMonitor::kInvalidMemUsageMB,
                  monitor_->GetPeakMemUsageInMB());
  monitor_->Start();
  monitor_->Stop();
  EXPECT_GT(monitor_->GetPeakMemUsageInMB(), 0);
}

TEST_F(MemoryUsageMonitorTest, MultiStartAndStops) {
  monitor_->Start();
  monitor_->Start();
  monitor_->Stop();
  monitor_->Stop();
  float peak;
  int64_t sleeps;
  WaitForPeakSync(&peak, &sleeps);
  EXPECT_FLOAT_EQ(5.0 * (sleeps + 1), peak);
}

TEST_F(MemoryUsageMonitorTest, StartStopPairs) {
  monitor_->Start();
  first_sample_notification_->WaitForNotificationWithTimeout(absl::Seconds(1));
  float peak;
  int64_t sleeps;
  WaitForPeakSync(&peak, &sleeps);
  monitor_->Stop();
  EXPECT_FLOAT_EQ(5.0 * (sleeps + 1), peak);

  monitor_->Start();
  // Sleep for at least for a duration that's longer than the sampling interval
  // passed to 'monitor_' (i.e. 50 ms) to simulate the memory usage increase.
  absl::SleepFor(absl::Milliseconds(100));
  WaitForPeakSync(&peak, &sleeps);
  monitor_->Stop();
  EXPECT_GE(sleeps, 1);
  EXPECT_FLOAT_EQ(5.0 * (sleeps + 1), peak);
}

TEST_F(MemoryUsageMonitorTest, StartReadStop) {
  monitor_->Start();
  // Sleep to allow the monitor to make the first sample.
  first_sample_notification_->WaitForNotificationWithTimeout(absl::Seconds(1));
  float peak;
  int64_t sleeps;
  WaitForPeakSync(&peak, &sleeps);
  EXPECT_FLOAT_EQ(5.0 * (sleeps + 1), peak);
  // Sleep for at least for a duration that's longer than the sampling interval
  // passed to 'monitor_' (i.e. 50 ms) to simulate the memory usage increase.
  absl::SleepFor(absl::Milliseconds(100));
  WaitForPeakSync(&peak, &sleeps);
  EXPECT_FLOAT_EQ(5.0 * (sleeps + 1), peak);
  monitor_->Stop();
}

}  // namespace memory
}  // namespace profiling
}  // namespace tflite
