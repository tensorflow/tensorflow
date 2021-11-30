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

#ifndef TENSORFLOW_LITE_PROFILING_MEMORY_USAGE_MONITOR_H_
#define TENSORFLOW_LITE_PROFILING_MEMORY_USAGE_MONITOR_H_

#include <memory>
#include <thread>  // NOLINT(build/c++11)

#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/lite/profiling/memory_info.h"

namespace tflite {
namespace profiling {
namespace memory {

// This class could help to tell the peak memory footprint of a running program.
// It achieves this by spawning a thread to check the memory usage periodically
// at a pre-defined frequency.
class MemoryUsageMonitor {
 public:
  // A helper class that does memory usage sampling. This allows injecting an
  // external dependency for the sake of testing or providing platform-specific
  // implementations.
  class Sampler {
   public:
    virtual ~Sampler() {}
    virtual bool IsSupported() { return MemoryUsage::IsSupported(); }
    virtual MemoryUsage GetMemoryUsage() {
      return tflite::profiling::memory::GetMemoryUsage();
    }
    virtual void SleepFor(const absl::Duration& duration) {
      absl::SleepFor(duration);
    }
  };

  static constexpr float kInvalidMemUsageMB = -1.0f;

  explicit MemoryUsageMonitor(int sampling_interval_ms = 50)
      : MemoryUsageMonitor(sampling_interval_ms,
                           std::unique_ptr<Sampler>(new Sampler())) {}
  MemoryUsageMonitor(int sampling_interval_ms,
                     std::unique_ptr<Sampler> sampler);
  ~MemoryUsageMonitor() { StopInternal(); }

  void Start();
  void Stop();

  // For simplicity, we will return kInvalidMemUsageMB for the either following
  // conditions:
  // 1. getting memory usage isn't supported on the platform.
  // 2. the memory usage is being monitored (i.e. we've created the
  // 'check_memory_thd_'.
  float GetPeakMemUsageInMB() const {
    if (!is_supported_ || check_memory_thd_ != nullptr) {
      return kInvalidMemUsageMB;
    }
    return peak_max_rss_kb_ / 1024.0;
  }

  MemoryUsageMonitor(MemoryUsageMonitor&) = delete;
  MemoryUsageMonitor& operator=(const MemoryUsageMonitor&) = delete;
  MemoryUsageMonitor(MemoryUsageMonitor&&) = delete;
  MemoryUsageMonitor& operator=(const MemoryUsageMonitor&&) = delete;

 private:
  void StopInternal();

  std::unique_ptr<Sampler> sampler_ = nullptr;
  bool is_supported_ = false;
  std::unique_ptr<absl::Notification> stop_signal_ = nullptr;
  absl::Duration sampling_interval_;
  std::unique_ptr<std::thread> check_memory_thd_ = nullptr;
  int64_t peak_max_rss_kb_ = static_cast<int64_t>(kInvalidMemUsageMB * 1024);
};

}  // namespace memory
}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_MEMORY_USAGE_MONITOR_H_
