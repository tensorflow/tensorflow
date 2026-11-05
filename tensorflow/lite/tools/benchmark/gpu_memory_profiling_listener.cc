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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/lite/tools/benchmark/benchmark_listener.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/benchmark/gpu_memory_info.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace benchmark {
namespace {

std::string FormatMbOrNa(int64_t bytes) {
  if (bytes < 0) return "N/A";
  return absl::StrFormat("%.2f MB", bytes / (1024.0 * 1024.0));
}

std::string FormatSignedDeltaMbOrNa(int64_t current_bytes,
                                    int64_t baseline_bytes) {
  if (current_bytes < 0 || baseline_bytes < 0) return "N/A";
  return absl::StrFormat("%.2f MB",
                         (current_bytes - baseline_bytes) / (1024.0 * 1024.0));
}

}  // namespace

GpuMemoryProfilingListener::GpuMemoryProfilingListener(
    int sampling_interval_ms, std::unique_ptr<Sampler> sampler)
    : sampler_(sampler != nullptr ? std::move(sampler)
                                  : std::make_unique<Sampler>()),
      is_supported_(sampler_->IsSupported()),
      sampling_interval_(absl::Milliseconds(sampling_interval_ms > 0
                                                ? sampling_interval_ms
                                                : kDefaultSamplingIntervalMs)) {
}

GpuMemoryProfilingListener::~GpuMemoryProfilingListener() { StopThread(); }

void GpuMemoryProfilingListener::Start() {
  if (!is_supported_) return;
  {
    absl::MutexLock lock(mutex_);
    if (sampling_thread_ != nullptr) return;
  }

  const GpuMemorySnapshot initial =
      sampler_->Sample(/*include_external_tools=*/true);

  absl::MutexLock lock(mutex_);
  if (sampling_thread_ != nullptr) return;
  start_snapshot_ = initial;
  init_snapshot_.reset();
  end_snapshot_.reset();
  peak_gpu_snapshot_.reset();
  num_samples_ = 0;
  has_ended_ = false;
  RecordSampleLocked(initial);
  StartThreadLocked();
}

void GpuMemoryProfilingListener::Stop() {
  if (!is_supported_) return;
  StopThread();
}

void GpuMemoryProfilingListener::Reset() {
  StopThread();
  absl::MutexLock lock(mutex_);
  start_snapshot_.reset();
  init_snapshot_.reset();
  end_snapshot_.reset();
  peak_gpu_snapshot_.reset();
  num_samples_ = 0;
  has_ended_ = false;
}

void GpuMemoryProfilingListener::OnBenchmarkStart(
    const BenchmarkParams& params) {
  (void)params;
  if (!is_supported_) return;

  const GpuMemorySnapshot sample =
      sampler_->Sample(/*include_external_tools=*/true);

  absl::MutexLock lock(mutex_);
  if (has_ended_) {
    // External listener reused across multiple Run() invocations without an
    // explicit Start()/Reset() call in between.
    start_snapshot_.reset();
    end_snapshot_.reset();
    peak_gpu_snapshot_.reset();
    num_samples_ = 0;
    has_ended_ = false;
  }
  init_snapshot_ = sample;
  RecordSampleLocked(sample);
  if (sampling_thread_ == nullptr) {
    StartThreadLocked();
  }
}

void GpuMemoryProfilingListener::OnBenchmarkEnd(
    const BenchmarkResults& results) {
  (void)results;
  if (!is_supported_) {
    TFLITE_LOG(INFO) << "[GPU Memory Monitor] GPU memory monitoring is not "
                        "supported on this platform.";
    return;
  }

  StopThread();
  const GpuMemorySnapshot final_sample =
      sampler_->Sample(/*include_external_tools=*/true);
  {
    absl::MutexLock lock(mutex_);
    end_snapshot_ = final_sample;
    RecordSampleLocked(final_sample);
    has_ended_ = true;
  }

  const std::string summary = FormatSummary();
  for (absl::string_view line :
       absl::StrSplit(summary, '\n', absl::SkipEmpty())) {
    TFLITE_LOG(INFO) << line;
  }
}

void GpuMemoryProfilingListener::StartThreadLocked() {
  stop_signal_ = std::make_unique<absl::Notification>();
  absl::Notification* const stop_ptr = stop_signal_.get();
  sampling_thread_ = std::make_unique<std::thread>([this, stop_ptr]() {
    while (true) {
      sampler_->WaitBetweenSamples(sampling_interval_, *stop_ptr);
      if (stop_ptr->HasBeenNotified()) break;
      const GpuMemorySnapshot sample =
          sampler_->Sample(/*include_external_tools=*/false);
      absl::MutexLock lock(mutex_);
      RecordSampleLocked(sample);
    }
  });
}

void GpuMemoryProfilingListener::StopThread() {
  std::unique_ptr<std::thread> thread_to_join;
  std::unique_ptr<absl::Notification> signal_to_destroy;
  {
    absl::MutexLock lock(mutex_);
    if (sampling_thread_ == nullptr) return;
    stop_signal_->Notify();
    thread_to_join = std::move(sampling_thread_);
    signal_to_destroy = std::move(stop_signal_);
  }
  thread_to_join->join();
}

void GpuMemoryProfilingListener::RecordSampleLocked(
    const GpuMemorySnapshot& sample) {
  ++num_samples_;
  if (!sample.HasGpu()) {
    // On Linux, a pre-Init() sample before any DRM device is opened reports
    // `gpu_bytes == 0`. If opening the GPU device later reveals that the
    // kernel/driver does not expose GPU memory metrics (`kUnavailable`),
    // discard the pre-device-open zero peak so `kUnavailable` is not masked.
    if (peak_gpu_snapshot_.has_value() && peak_gpu_snapshot_->gpu_bytes == 0) {
      peak_gpu_snapshot_.reset();
    }
    return;
  }
  if (!peak_gpu_snapshot_.has_value() ||
      sample.gpu_bytes > peak_gpu_snapshot_->gpu_bytes) {
    peak_gpu_snapshot_ = sample;
  }
}

int GpuMemoryProfilingListener::num_samples() const {
  absl::MutexLock lock(mutex_);
  return num_samples_;
}

std::optional<GpuMemorySnapshot> GpuMemoryProfilingListener::start_snapshot()
    const {
  absl::MutexLock lock(mutex_);
  return start_snapshot_;
}

std::optional<GpuMemorySnapshot> GpuMemoryProfilingListener::init_snapshot()
    const {
  absl::MutexLock lock(mutex_);
  return init_snapshot_;
}

std::optional<GpuMemorySnapshot> GpuMemoryProfilingListener::end_snapshot()
    const {
  absl::MutexLock lock(mutex_);
  return end_snapshot_;
}

std::optional<GpuMemorySnapshot> GpuMemoryProfilingListener::peak_gpu_snapshot()
    const {
  absl::MutexLock lock(mutex_);
  return peak_gpu_snapshot_;
}

float GpuMemoryProfilingListener::peak_gpu_mb() const {
  absl::MutexLock lock(mutex_);
  return peak_gpu_snapshot_.has_value()
             ? BytesToMegabytes(peak_gpu_snapshot_->gpu_bytes)
             : kInvalidMemUsageMB;
}

std::string GpuMemoryProfilingListener::FormatSummary() const {
  absl::MutexLock lock(mutex_);
  if (!peak_gpu_snapshot_.has_value() || !peak_gpu_snapshot_->HasGpu()) {
    return "[GPU Memory Monitor] GPU memory metrics unavailable (restricted by "
           "sandbox or unsupported kernel).";
  }

  std::vector<std::string> lines;
  lines.push_back("[GPU Memory Monitor] GPU memory usage summary:");

  if (start_snapshot_.has_value()) {
    if (init_snapshot_.has_value()) {
      lines.push_back(
          absl::StrFormat("[GPU Memory Monitor] Init GPU memory delta:    %s",
                          FormatSignedDeltaMbOrNa(init_snapshot_->gpu_bytes,
                                                  start_snapshot_->gpu_bytes)));
    }
    if (end_snapshot_.has_value()) {
      lines.push_back(
          absl::StrFormat("[GPU Memory Monitor] Overall GPU memory delta: %s",
                          FormatSignedDeltaMbOrNa(end_snapshot_->gpu_bytes,
                                                  start_snapshot_->gpu_bytes)));
    }
  } else {
    if (init_snapshot_.has_value()) {
      lines.push_back(
          absl::StrFormat("[GPU Memory Monitor] Post-init GPU memory: %s",
                          FormatMbOrNa(init_snapshot_->gpu_bytes)));
    }
    if (end_snapshot_.has_value()) {
      lines.push_back(
          absl::StrFormat("[GPU Memory Monitor] Post-run GPU memory:  %s",
                          FormatMbOrNa(end_snapshot_->gpu_bytes)));
    }
  }

  lines.push_back(absl::StrFormat(
      "[GPU Memory Monitor] Peak GPU memory:          %s (samples=%d)",
      FormatMbOrNa(peak_gpu_snapshot_->gpu_bytes), num_samples_));

  if (!peak_gpu_snapshot_->gpu_breakdown.empty()) {
    std::vector<std::string> parts;
    parts.reserve(peak_gpu_snapshot_->gpu_breakdown.size());
    for (const auto& [source, bytes] : peak_gpu_snapshot_->gpu_breakdown) {
      parts.push_back(
          absl::StrFormat("%s=%.2f MB", source, bytes / (1024.0 * 1024.0)));
    }
    lines.push_back(
        absl::StrFormat("[GPU Memory Monitor] Peak GPU detail:          %s "
                        "(%s)",
                        FormatMbOrNa(peak_gpu_snapshot_->gpu_bytes),
                        absl::StrJoin(parts, ", ")));
  }

  return absl::StrJoin(lines, "\n");
}

}  // namespace benchmark
}  // namespace tflite
