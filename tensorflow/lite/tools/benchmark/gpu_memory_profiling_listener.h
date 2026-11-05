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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_GPU_MEMORY_PROFILING_LISTENER_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_GPU_MEMORY_PROFILING_LISTENER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/lite/tools/benchmark/benchmark_listener.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/benchmark/gpu_memory_info.h"

namespace tflite {
namespace benchmark {

// Opt-in BenchmarkListener that samples and logs GPU memory usage and
// per-source breakdowns independently of the default MemoryUsageMonitor and
// BenchmarkResults metrics.
//
// On Linux and Android, periodic background samples read only in-process
// `/proc/self/*` files (avoiding subprocess fork overhead during timed
// inference runs and degrading gracefully in sandboxed environments). On
// Android CLI runs outside app sandboxes, untimed phase boundaries also query
// `dumpsys gpu --gpumem` once per boundary to capture unmapped driver pools
// (such as device-only OpenCL allocations on PowerVR).
class GpuMemoryProfilingListener : public BenchmarkListener {
 public:
  static constexpr int kDefaultSamplingIntervalMs = 250;
  static constexpr float kInvalidMemUsageMB = -1.0f;

  // A helper class that performs GPU memory sampling. Allows injecting a fake
  // sampler in unit tests without sleeping.
  class Sampler {
   public:
    virtual ~Sampler() = default;
    virtual bool IsSupported() const {
      return GpuMemorySnapshot::IsSupported();
    }
    // `include_external_tools` is false on periodic background ticks and true
    // only at untimed benchmark phase boundaries.
    virtual GpuMemorySnapshot Sample(bool include_external_tools) {
      return GetGpuMemorySnapshot(include_external_tools);
    }
    virtual void WaitBetweenSamples(absl::Duration interval,
                                    const absl::Notification& stop_signal) {
      stop_signal.WaitForNotificationWithTimeout(interval);
    }
  };

  explicit GpuMemoryProfilingListener(
      int sampling_interval_ms = kDefaultSamplingIntervalMs,
      std::unique_ptr<Sampler> sampler = nullptr);
  ~GpuMemoryProfilingListener() override;

  GpuMemoryProfilingListener(const GpuMemoryProfilingListener&) = delete;
  GpuMemoryProfilingListener& operator=(const GpuMemoryProfilingListener&) =
      delete;

  // Starts sampling immediately and records the pre-initialization baseline
  // snapshot. Calling Start() before `BenchmarkModel::Init()` allows reporting
  // `Init GPU memory delta` and `Overall GPU memory delta` from the start of
  // initialization. If not called explicitly, `OnBenchmarkStart()` starts
  // sampling automatically prior to warmup/inference runs.
  void Start() ABSL_LOCKS_EXCLUDED(mutex_);

  // Stops the background sampling thread. Safe to call multiple times.
  void Stop() ABSL_LOCKS_EXCLUDED(mutex_);

  // Clears recorded snapshots and peaks for a fresh benchmark run.
  void Reset() ABSL_LOCKS_EXCLUDED(mutex_);

  void OnBenchmarkStart(const BenchmarkParams& params) override
      ABSL_LOCKS_EXCLUDED(mutex_);
  void OnBenchmarkEnd(const BenchmarkResults& results) override
      ABSL_LOCKS_EXCLUDED(mutex_);

  bool IsSupported() const { return is_supported_; }
  int num_samples() const ABSL_LOCKS_EXCLUDED(mutex_);
  std::optional<GpuMemorySnapshot> start_snapshot() const
      ABSL_LOCKS_EXCLUDED(mutex_);
  std::optional<GpuMemorySnapshot> init_snapshot() const
      ABSL_LOCKS_EXCLUDED(mutex_);
  std::optional<GpuMemorySnapshot> end_snapshot() const
      ABSL_LOCKS_EXCLUDED(mutex_);
  std::optional<GpuMemorySnapshot> peak_gpu_snapshot() const
      ABSL_LOCKS_EXCLUDED(mutex_);

  float peak_gpu_mb() const ABSL_LOCKS_EXCLUDED(mutex_);

  std::string FormatSummary() const ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  void StartThreadLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void StopThread() ABSL_LOCKS_EXCLUDED(mutex_);
  void RecordSampleLocked(const GpuMemorySnapshot& sample)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  static float BytesToMegabytes(int64_t bytes) {
    return bytes < 0 ? kInvalidMemUsageMB
                     : static_cast<float>(bytes / (1024.0 * 1024.0));
  }

  std::unique_ptr<Sampler> sampler_;
  bool is_supported_ = false;
  absl::Duration sampling_interval_;

  mutable absl::Mutex mutex_;
  std::unique_ptr<absl::Notification> stop_signal_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<std::thread> sampling_thread_ ABSL_GUARDED_BY(mutex_);
  bool has_ended_ ABSL_GUARDED_BY(mutex_) = false;

  int num_samples_ ABSL_GUARDED_BY(mutex_) = 0;
  std::optional<GpuMemorySnapshot> start_snapshot_ ABSL_GUARDED_BY(mutex_);
  std::optional<GpuMemorySnapshot> init_snapshot_ ABSL_GUARDED_BY(mutex_);
  std::optional<GpuMemorySnapshot> end_snapshot_ ABSL_GUARDED_BY(mutex_);
  std::optional<GpuMemorySnapshot> peak_gpu_snapshot_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_GPU_MEMORY_PROFILING_LISTENER_H_
