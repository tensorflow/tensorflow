/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/benchmark/gpu_memory_profiling_listener.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/lite/tools/benchmark/benchmark_listener.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/benchmark/gpu_memory_info.h"

namespace tflite {
namespace benchmark {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::HasSubstr;

constexpr int64_t kMb = 1024 * 1024;

// Deterministic stub sampler that serves scripted boundary and background
// snapshots without sleeping.
class ScriptedGpuMemorySampler : public GpuMemoryProfilingListener::Sampler {
 public:
  explicit ScriptedGpuMemorySampler(
      std::vector<GpuMemorySnapshot> boundary_samples,
      std::vector<GpuMemorySnapshot> background_samples = {},
      bool supported = true)
      : boundary_samples_(std::move(boundary_samples)),
        background_samples_(std::move(background_samples)),
        supported_(supported) {}

  bool IsSupported() const override { return supported_; }

  GpuMemorySnapshot Sample(bool include_external_tools) override {
    absl::MutexLock lock(mutex_);
    call_external_flags_.push_back(include_external_tools);
    if (include_external_tools) {
      if (boundary_index_ < boundary_samples_.size()) {
        return boundary_samples_[boundary_index_++];
      }
      return boundary_samples_.empty() ? GpuMemorySnapshot{}
                                       : boundary_samples_.back();
    }
    if (background_index_ < background_samples_.size()) {
      const GpuMemorySnapshot sample = background_samples_[background_index_++];
      if (background_index_ == background_samples_.size() &&
          !all_background_consumed_.HasBeenNotified()) {
        all_background_consumed_.Notify();
      }
      return sample;
    }
    return background_samples_.empty() ? GpuMemorySnapshot{}
                                       : background_samples_.back();
  }

  void WaitBetweenSamples(absl::Duration interval,
                          const absl::Notification& stop_signal) override {
    (void)interval;
    bool has_pending_background = false;
    {
      absl::MutexLock lock(mutex_);
      has_pending_background = background_index_ < background_samples_.size();
    }
    if (has_pending_background) {
      // Deliver scripted background samples deterministically without sleeping,
      // then block on `stop_signal` once all scripted background samples have
      // been consumed.
      return;
    }
    stop_signal.WaitForNotification();
  }

  bool WaitForAllBackgroundSamples(absl::Duration timeout) {
    return all_background_consumed_.WaitForNotificationWithTimeout(timeout);
  }

  std::vector<bool> call_external_flags() const {
    absl::MutexLock lock(mutex_);
    return call_external_flags_;
  }

 private:
  mutable absl::Mutex mutex_;
  std::vector<GpuMemorySnapshot> boundary_samples_;
  std::vector<GpuMemorySnapshot> background_samples_;
  size_t boundary_index_ = 0;
  size_t background_index_ = 0;
  std::vector<bool> call_external_flags_;
  absl::Notification all_background_consumed_;
  bool supported_ = true;
};

TEST(GpuMemoryProfilingListenerTest,
     TracksDeltasAndPeaksWithoutExternalToolsDuringPeriodicPolling) {
  const GpuMemorySnapshot start_snap{
      .gpu_bytes = 0,
  };
  const GpuMemorySnapshot init_snap{
      .gpu_bytes = 200 * kMb,
      .gpu_breakdown = {{"gpu_driver_mali", 200 * kMb}},
  };
  const GpuMemorySnapshot mid_run_peak{
      .gpu_bytes = 450 * kMb,
      .gpu_breakdown = {{"gpu_driver_mali", 450 * kMb}},
  };
  const GpuMemorySnapshot end_snap{
      .gpu_bytes = 300 * kMb,
      .gpu_breakdown = {{"gpu_driver_mali", 300 * kMb}},
  };

  auto sampler = std::make_unique<ScriptedGpuMemorySampler>(
      /*boundary_samples=*/std::vector<GpuMemorySnapshot>{start_snap, init_snap,
                                                          end_snap},
      /*background_samples=*/std::vector<GpuMemorySnapshot>{mid_run_peak});
  ScriptedGpuMemorySampler* sampler_ptr = sampler.get();

  GpuMemoryProfilingListener listener(/*sampling_interval_ms=*/250,
                                      std::move(sampler));
  listener.Start();
  BenchmarkParams params;
  listener.OnBenchmarkStart(params);
  ASSERT_TRUE(sampler_ptr->WaitForAllBackgroundSamples(absl::Seconds(5)));
  listener.OnBenchmarkEnd(BenchmarkResults());

  // Periodic background samples must pass include_external_tools=false, while
  // the 3 untimed boundary snapshots pass include_external_tools=true.
  EXPECT_THAT(sampler_ptr->call_external_flags(),
              ElementsAre(true, true, false, true));

  EXPECT_EQ(listener.num_samples(), 4);
  EXPECT_THAT(listener.peak_gpu_mb(), FloatEq(450.0f));
  ASSERT_TRUE(listener.peak_gpu_snapshot().has_value());
  EXPECT_EQ(listener.peak_gpu_snapshot()->gpu_bytes, 450 * kMb);

  const std::string summary = listener.FormatSummary();
  EXPECT_THAT(summary, HasSubstr("Init GPU memory delta:    200.00 MB"));
  EXPECT_THAT(summary, HasSubstr("Overall GPU memory delta: 300.00 MB"));
  EXPECT_THAT(summary,
              HasSubstr("Peak GPU memory:          450.00 MB (samples=4)"));
  EXPECT_THAT(summary, HasSubstr("Peak GPU detail:          450.00 MB "
                                 "(gpu_driver_mali=450.00 MB)"));

  listener.Stop();
  listener.Reset();
  EXPECT_EQ(listener.num_samples(), 0);
  EXPECT_FALSE(listener.peak_gpu_snapshot().has_value());
}

TEST(GpuMemoryProfilingListenerTest,
     ReportsUnavailableWhenUnsupportedOrRestricted) {
  auto sampler = std::make_unique<ScriptedGpuMemorySampler>(
      /*boundary_samples=*/std::vector<GpuMemorySnapshot>{},
      /*background_samples=*/std::vector<GpuMemorySnapshot>{},
      /*supported=*/false);
  GpuMemoryProfilingListener listener(/*sampling_interval_ms=*/250,
                                      std::move(sampler));
  EXPECT_FALSE(listener.IsSupported());
  listener.Start();
  BenchmarkParams params;
  listener.OnBenchmarkStart(params);
  listener.OnBenchmarkEnd(BenchmarkResults());
  EXPECT_EQ(listener.num_samples(), 0);
  EXPECT_THAT(listener.peak_gpu_mb(),
              FloatEq(GpuMemoryProfilingListener::kInvalidMemUsageMB));
  EXPECT_THAT(listener.FormatSummary(),
              HasSubstr("GPU memory metrics unavailable"));
}

TEST(GpuMemoryProfilingListenerTest,
     AutoStartsInOnBenchmarkStartAndResetsAcrossRunsWhenAttachedExternally) {
  const GpuMemorySnapshot run1_init{
      .gpu_bytes = 500 * kMb,
      .gpu_breakdown = {{"gpu_driver_drm", 500 * kMb}},
  };
  const GpuMemorySnapshot run1_end{
      .gpu_bytes = 510 * kMb,
      .gpu_breakdown = {{"gpu_driver_drm", 510 * kMb}},
  };
  const GpuMemorySnapshot run2_init{
      .gpu_bytes = 120 * kMb,
      .gpu_breakdown = {{"gpu_driver_drm", 120 * kMb}},
  };
  const GpuMemorySnapshot run2_end{
      .gpu_bytes = 130 * kMb,
      .gpu_breakdown = {{"gpu_driver_drm", 130 * kMb}},
  };

  auto sampler = std::make_unique<ScriptedGpuMemorySampler>(
      /*boundary_samples=*/std::vector<GpuMemorySnapshot>{run1_init, run1_end,
                                                          run2_init, run2_end});
  GpuMemoryProfilingListener listener(/*sampling_interval_ms=*/250,
                                      std::move(sampler));

  BenchmarkParams params;
  listener.OnBenchmarkStart(params);
  listener.OnBenchmarkEnd(BenchmarkResults());
  EXPECT_FALSE(listener.start_snapshot().has_value());
  EXPECT_THAT(listener.peak_gpu_mb(), FloatEq(510.0f));
  EXPECT_THAT(listener.FormatSummary(), HasSubstr("Post-init GPU memory:"));

  // Second run on the same external listener resets peaks from run 1.
  listener.OnBenchmarkStart(params);
  listener.OnBenchmarkEnd(BenchmarkResults());
  EXPECT_THAT(listener.peak_gpu_mb(), FloatEq(130.0f));
}

TEST(GpuMemoryProfilingListenerTest,
     PreInitZeroSampleDoesNotMaskPostInitUnavailableOnUnsupportedDrmDriver) {
  const GpuMemorySnapshot pre_init_zero{.gpu_bytes = 0};
  const GpuMemorySnapshot post_init_unavailable{
      .gpu_bytes = GpuMemorySnapshot::kUnavailable,
  };

  auto sampler = std::make_unique<ScriptedGpuMemorySampler>(
      /*boundary_samples=*/std::vector<GpuMemorySnapshot>{
          pre_init_zero, post_init_unavailable, post_init_unavailable});
  GpuMemoryProfilingListener listener(/*sampling_interval_ms=*/250,
                                      std::move(sampler));

  listener.Start();
  BenchmarkParams params;
  listener.OnBenchmarkStart(params);
  listener.OnBenchmarkEnd(BenchmarkResults());

  EXPECT_FALSE(listener.peak_gpu_snapshot().has_value());
  EXPECT_THAT(listener.peak_gpu_mb(),
              FloatEq(GpuMemoryProfilingListener::kInvalidMemUsageMB));
  EXPECT_THAT(listener.FormatSummary(),
              HasSubstr("GPU memory metrics unavailable"));
}

}  // namespace
}  // namespace benchmark
}  // namespace tflite
