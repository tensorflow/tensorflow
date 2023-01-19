/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_TFDATAZ_METRICS_H_
#define TENSORFLOW_CORE_DATA_TFDATAZ_METRICS_H_

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// Calculates the approximate average latency for past 1, 5 and 60 minutes.
// The implementation uses ring buffers to maintain the cumulative latency
// values and count for the past 60 minutes.
class ApproximateLatencyEstimator {
 public:
  enum class Duration {
    kMinute = 1,
    kFiveMinutes = 5,
    kSixtyMinutes = 60,
  };

  explicit ApproximateLatencyEstimator(const Env& env);

  // Records the latency with the current timestamp.
  void AddLatency(int64_t latency_usec);

  // Returns the average latency for the duration (1,5 and 60 minutes)
  // specified.
  double GetAverageLatency(Duration duration);

 private:
  static constexpr int64_t kSecondsPerMinute = 60;
  static constexpr int64_t kMinutesPerHour = 60;
  static constexpr int64_t kSlots = kMinutesPerHour;

  // Updates the latency value and count ring buffers with the latest cumulative
  // value and count. Resets the entire ring buffer with the last cumulative
  // values stored if the elapsed time duration is greater than 60 minutes.
  void UpdateRingBuffer() TF_LOCKS_EXCLUDED(mu_);
  // Moves the `next_slot_` to the next index in the ring buffer.
  void IncrementNextSlot() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Returns the slot index which is behind the current slot in ring buffer by
  // `steps` indices.
  int PrevSlot(int steps) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const Env& env_;

  // The time when the ring buffer was last updated.
  int64_t last_updated_time_mins_ TF_GUARDED_BY(mu_);

  mutex mu_;

  // Counters storing the cumulative sums of latency values and counts recorded
  // so far.
  int64_t latency_value_counter_ TF_GUARDED_BY(mu_);
  int64_t latency_count_counter_ TF_GUARDED_BY(mu_);

  // Next slot in the ring buffer.
  int next_slot_ TF_GUARDED_BY(mu_);

  // Ring buffer storing the cumulative sum of latency values and counts for the
  // last 60 minutes.
  int64_t latency_value_[kSlots] TF_GUARDED_BY(mu_);
  int64_t latency_count_[kSlots] TF_GUARDED_BY(mu_);
};

// Collects and exports the tf.data performance metrics to /tfdataz.
class TfDatazMetricsCollector {
 public:
  // Constructs a `TfDatazMetricsCollector`. `device_type` is one of the
  // devices defined in `types.h` (DEVICE_CPU, DEVICE_GPU, DEVICE_TPU, etc).
  // We only collect metrics for CPU devices. This is a heuristic to avoid
  // collecting metrics for device-side iterators created by the multi-device
  // iterator mechanism.
  TfDatazMetricsCollector(const std::string& device_type, const Env& env);

  // Records `GetNext` call latency.
  void RecordGetNextLatency(int64_t get_next_latency_usec);

  // Returns the average `GetNext` latency for past 1 minute.
  double GetAverageLatencyForLastOneMinute();

  // Returns the average `GetNext` latency for past 5 minutes.
  double GetAverageLatencyForLastFiveMinutes();

  // Returns the average `GetNext` latency for past 60 minutes.
  double GetAverageLatencyForLastSixtyMinutes();

 private:
  // One of the devices defined in `types.h`
  // (DEVICE_CPU, DEVICE_GPU, DEVICE_TPU, etc).
  const std::string device_type_;
  ApproximateLatencyEstimator latency_estimator_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_TFDATAZ_METRICS_H_
