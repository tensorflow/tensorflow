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
#include <cstdint>
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
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

inline constexpr size_t kDefaultMaxBufferBytes =
    4ULL * 1024 * 1024 * 1024;  // 4GB

struct DrainedBuffer {
  std::vector<std::any> chunks;
  uint64_t cumulative_dropped_chunks = 0;
  uint64_t cumulative_dropped_bytes = 0;
};

template <typename ProfilerType>
class ContinuousProfilerOrchestrator : public ProfilerInterface {
 public:
  static constexpr absl::Duration kDefaultPollingInterval = absl::Seconds(2);
  static constexpr absl::Duration kMinPollingInterval = absl::Seconds(2);
  static constexpr absl::Duration kMaxPollingInterval = absl::Seconds(5);
  // Upper bound on how long Flush() will block waiting for the ingestion
  // thread. Generous relative to kMaxPollingInterval (which bounds how long
  // the ingestion thread sleeps) but small enough that a wedged Consume()
  // cannot hold an RPC handler hostage.
  static constexpr absl::Duration kFlushTimeout = absl::Seconds(30);

  explicit ContinuousProfilerOrchestrator(
      std::unique_ptr<ProfilerType> profiler,
      size_t max_buffer_bytes = kDefaultMaxBufferBytes)
      : profiler_(std::move(profiler)),
        max_buffer_bytes_(max_buffer_bytes),
        is_running_(false),
        polling_interval_(kDefaultPollingInterval) {}

  ~ContinuousProfilerOrchestrator() override { StopInternal().IgnoreError(); }

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
    absl::Status status = StopInternal();
    absl::StatusOr<ConsumeResult> result = profiler_->Consume();
    if (result.ok()) {
      absl::MutexLock lock(mutex_);
      PushChunkLocked(std::move(*result));
    } else if (!absl::IsUnimplemented(result.status())) {
      LOG(WARNING) << "Final Consume failed during Stop: " << result.status();
    }
    return status;
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

  std::vector<tensorflow::profiler::XSpace> SerializeChunks() {
    Flush();
    std::vector<std::any> chunks = PopBuffer();
    std::vector<tensorflow::profiler::XSpace> spaces;
    spaces.reserve(chunks.size());
    tensorflow::profiler::XSpace space;
    for (auto& chunk : chunks) {
      space.Clear();
      absl::Status status = profiler_->Serialize(std::move(chunk), &space);
      if (status.ok()) {
        spaces.push_back(std::move(space));
      } else {
        LOG(ERROR) << "Failed to serialize profiler chunk: " << status;
      }
    }
    return spaces;
  }

  // Returns the current polling interval (primarily for testing).
  absl::Duration polling_interval() const {
    absl::MutexLock lock(mutex_);
    return polling_interval_;
  }

  ProfilerType* profiler() { return profiler_.get(); }
  const ProfilerType* profiler() const { return profiler_.get(); }

  size_t max_buffer_bytes() const { return max_buffer_bytes_; }

  size_t total_buffered_bytes() const {
    absl::MutexLock lock(mutex_);
    return total_buffered_bytes_;
  }

  uint64_t dropped_chunks_count() const {
    absl::MutexLock lock(mutex_);
    return dropped_chunks_count_;
  }

  uint64_t dropped_bytes_count() const {
    absl::MutexLock lock(mutex_);
    return dropped_bytes_count_;
  }

  DrainedBuffer PopBufferWithTelemetry() {
    std::deque<std::any> local_buffer;
    uint64_t dropped_chunks = 0;
    uint64_t dropped_bytes = 0;
    {
      absl::MutexLock lock(mutex_);
      local_buffer.swap(circular_buffer_);
      chunk_sizes_.clear();
      total_buffered_bytes_ = 0;
      dropped_chunks = dropped_chunks_count_;
      dropped_bytes = dropped_bytes_count_;
    }

    std::vector<std::any> chunks;
    chunks.reserve(local_buffer.size());
    for (auto& item : local_buffer) {
      if (item.has_value()) {
        chunks.push_back(std::move(item));
      }
    }
    return DrainedBuffer{
        .chunks = std::move(chunks),
        .cumulative_dropped_chunks = dropped_chunks,
        .cumulative_dropped_bytes = dropped_bytes,
    };
  }

  std::vector<std::any> PopBuffer() { return PopBufferWithTelemetry().chunks; }

 private:
  void PushChunkLocked(ConsumeResult chunk)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (!chunk.data.has_value()) {
      return;
    }

    if (chunk.estimated_size_bytes > max_buffer_bytes_) {
      LOG_EVERY_N_SEC(WARNING, 30)
          << "ContinuousProfilerOrchestrator rejected chunk of size "
          << chunk.estimated_size_bytes << " bytes exceeding max buffer limit "
          << max_buffer_bytes_ << " bytes.";
      dropped_chunks_count_ += 1;
      dropped_bytes_count_ += chunk.estimated_size_bytes;
      return;
    }

    while (!circular_buffer_.empty() &&
           (total_buffered_bytes_ + chunk.estimated_size_bytes >
            max_buffer_bytes_)) {
      size_t front_size = chunk_sizes_.front();
      total_buffered_bytes_ = (total_buffered_bytes_ > front_size)
                                  ? total_buffered_bytes_ - front_size
                                  : 0;
      dropped_chunks_count_ += 1;
      dropped_bytes_count_ += front_size;
      circular_buffer_.pop_front();
      chunk_sizes_.pop_front();
    }

    total_buffered_bytes_ += chunk.estimated_size_bytes;
    circular_buffer_.push_back(std::move(chunk.data));
    chunk_sizes_.push_back(chunk.estimated_size_bytes);
  }

  void IngestionLoop() {
    LOG(INFO) << "ContinuousProfilerOrchestrator::IngestionLoop started";
    while (true) {
      uint64_t seq_before_consume = 0;
      {
        absl::MutexLock lock(mutex_);
        if (!is_running_) break;
        seq_before_consume = flush_requested_seq_;
      }
      absl::StatusOr<ConsumeResult> result = profiler_->Consume();

      absl::MutexLock lock(mutex_);
      if (result.ok()) {
        const size_t chunk_size = result->estimated_size_bytes;
        PushChunkLocked(std::move(*result));
        AdjustIntervalLocked(chunk_size);
      }

      if (seq_before_consume > flush_completed_seq_) {
        flush_completed_seq_ = seq_before_consume;
        cv_.SignalAll();
      }
      if (!is_running_) break;
      if (flush_requested_seq_ > flush_completed_seq_) {
        continue;  // Immediate consume requested while Consume() was in flight.
      }
      cv_.WaitWithTimeout(&mutex_, polling_interval_);
      if (!is_running_) break;
    }
  }

  // Asks the ingestion thread to run a Consume() cycle now and waits for it to
  // finish, so that everything recorded before this call is in the buffer.
  //
  // The wait is bounded by `kFlushTimeout`: a Consume() implementation that
  // wedges must not be able to block the caller (typically an RPC handler)
  // indefinitely. On timeout we simply return, and the caller sees whatever
  // was already buffered.
  //
  // Note this can cost up to two Consume() cycles in the worst case: a
  // Consume() that was already in flight when the flush was requested cannot
  // be credited against it, because it may have started before the data the
  // caller cares about was recorded.
  void Flush() {
    absl::MutexLock lock(mutex_);
    if (!is_running_) return;
    uint64_t target_seq = ++flush_requested_seq_;
    cv_.SignalAll();
    const absl::Time deadline = absl::Now() + kFlushTimeout;
    while (is_running_ && flush_completed_seq_ < target_seq) {
      if (cv_.WaitWithDeadline(&mutex_, deadline)) {
        LOG_EVERY_N_SEC(WARNING, 30)
            << "ContinuousProfilerOrchestrator::Flush timed out after "
            << kFlushTimeout
            << " waiting for the ingestion thread; returning partial data.";
        return;
      }
    }
  }

  absl::Status StopInternal() {
    {
      absl::MutexLock lock(mutex_);
      if (!is_running_) return absl::OkStatus();
      is_running_ = false;
      cv_.SignalAll();
    }
    absl::Status status = profiler_->Stop();
    ingestion_thread_.reset();
    return status;
  }

  void AdjustIntervalLocked(size_t chunk_size_bytes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    constexpr size_t kHighWatermark = 512 * 1024 * 1024;  // 512MB
    constexpr size_t kLowWatermark = 5 * 1024 * 1024;     // 5MB

    if (chunk_size_bytes > kHighWatermark) {
      polling_interval_ = std::max(polling_interval_ / 2, kMinPollingInterval);
    } else if (chunk_size_bytes < kLowWatermark) {
      polling_interval_ = std::min(polling_interval_ * 2, kMaxPollingInterval);
    }
  }

  std::unique_ptr<ProfilerType> profiler_;
  const size_t max_buffer_bytes_;

  mutable absl::Mutex mutex_;
  absl::CondVar cv_;
  std::unique_ptr<tsl::Thread> ingestion_thread_;
  bool is_running_ ABSL_GUARDED_BY(mutex_);

  absl::Duration polling_interval_ ABSL_GUARDED_BY(mutex_);
  std::deque<std::any> circular_buffer_ ABSL_GUARDED_BY(mutex_);
  std::deque<size_t> chunk_sizes_ ABSL_GUARDED_BY(mutex_);
  size_t total_buffered_bytes_ ABSL_GUARDED_BY(mutex_) = 0;
  uint64_t dropped_chunks_count_ ABSL_GUARDED_BY(mutex_) = 0;
  uint64_t dropped_bytes_count_ ABSL_GUARDED_BY(mutex_) = 0;
  uint64_t flush_requested_seq_ ABSL_GUARDED_BY(mutex_) = 0;
  uint64_t flush_completed_seq_ ABSL_GUARDED_BY(mutex_) = 0;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_CONTINUOUS_PROFILER_ORCHESTRATOR_H_
