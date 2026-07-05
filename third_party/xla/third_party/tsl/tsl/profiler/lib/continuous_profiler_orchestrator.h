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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_CONTINUOUS_PROFILER_ORCHESTRATOR_H_
#define TENSORFLOW_TSL_PROFILER_LIB_CONTINUOUS_PROFILER_ORCHESTRATOR_H_

#include <algorithm>
#include <any>
#include <cstddef>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"  // IWYU pragma: keep
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

template <typename ProfilerType>
class ContinuousProfilerOrchestrator : public ProfilerInterface {
 public:
  static constexpr absl::Duration kDefaultPollingInterval = absl::Seconds(1);

  explicit ContinuousProfilerOrchestrator(
      std::unique_ptr<ProfilerType> profiler)
      : profiler_(std::move(profiler)),
        is_running_(false),
        polling_interval_(kDefaultPollingInterval) {}

  ~ContinuousProfilerOrchestrator() override { StopIngestionThread(); }

  // Starts profiling and spawns background thread.
  absl::Status Start() override {
    {
      absl::MutexLock lock(mutex_);
      if (is_running_) {
        return absl::FailedPreconditionError(
            "ContinuousProfilerOrchestrator already started");
      }
    }
    TF_RETURN_IF_ERROR(profiler_->Start());

    {
      absl::MutexLock lock(mutex_);
      is_running_ = true;
    }
    ingestion_thread_ =
        std::unique_ptr<tsl::Thread>(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions{}, "ContinuousProfilerIngestion",
            [this]() { IngestionLoop(); }));
    return absl::OkStatus();
  }

  // Stops background thread and profiling.
  absl::Status Stop() override {
    StopIngestionThread();
    return profiler_->Stop();
  }

  absl::Status CollectData(tensorflow::profiler::XSpace* space) override {
    absl::Status status = Serialize({}, space);
    status.Update(profiler_->CollectData(space));
    return status;
  }

  absl::Status Serialize(std::any data,
                         tensorflow::profiler::XSpace* space) override {
    std::vector<std::any> chunks = PopBuffer();
    absl::Status status;
    for (auto& chunk : chunks) {
      status.Update(profiler_->Serialize(std::move(chunk), space));
    }
    return status;
  }

  // Returns the current polling interval (primarily for testing).
  absl::Duration polling_interval() const {
    absl::MutexLock lock(mutex_);
    return polling_interval_;
  }

  ProfilerType* profiler() { return profiler_.get(); }
  const ProfilerType* profiler() const { return profiler_.get(); }

 private:
  std::vector<std::any> PopBuffer() {
    absl::MutexLock lock(mutex_);
    std::vector<std::any> chunks;
    chunks.reserve(circular_buffer_.size());
    for (auto& item : circular_buffer_) {
      chunks.push_back(std::move(item));
    }
    circular_buffer_.clear();
    return chunks;
  }
  void IngestionLoop() {
    while (true) {
      absl::StatusOr<ConsumeResult> result = profiler_->Consume();

      absl::MutexLock lock(mutex_);
      if (!is_running_) break;

      if (result.ok()) {
        circular_buffer_.push_back(std::move(result->data));

        // Cap circular buffer to prevent infinite memory growth.
        if (circular_buffer_.size() > 100) {
          circular_buffer_.pop_front();
        }

        AdjustIntervalLocked(result->estimated_size_bytes);
      }

      // Wait using absl::CondVar on absl::Mutex
      cv_.WaitWithTimeout(&mutex_, polling_interval_);
      if (!is_running_) break;
    }
  }

  void StopIngestionThread() {
    {
      absl::MutexLock lock(mutex_);
      if (!is_running_) return;
      is_running_ = false;
      cv_.SignalAll();
    }
    ingestion_thread_.reset();
  }

  void AdjustIntervalLocked(size_t chunk_size_bytes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    constexpr size_t kHighWatermark = 512 * 1024 * 1024;  // 512MB
    constexpr size_t kLowWatermark = 5 * 1024 * 1024;     // 5MB

    if (chunk_size_bytes > kHighWatermark) {
      polling_interval_ =
          std::max(polling_interval_ / 2, absl::Milliseconds(100));
    } else if (chunk_size_bytes < kLowWatermark) {
      polling_interval_ = std::min(polling_interval_ * 2, absl::Seconds(5));
    }
  }

  std::unique_ptr<ProfilerType> profiler_;

  mutable absl::Mutex mutex_;
  absl::CondVar cv_;
  std::unique_ptr<tsl::Thread> ingestion_thread_;
  bool is_running_ ABSL_GUARDED_BY(mutex_);

  absl::Duration polling_interval_ ABSL_GUARDED_BY(mutex_);
  std::deque<std::any> circular_buffer_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_CONTINUOUS_PROFILER_ORCHESTRATOR_H_
