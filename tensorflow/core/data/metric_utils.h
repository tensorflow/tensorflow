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
#ifndef TENSORFLOW_CORE_DATA_METRIC_UTILS_H_
#define TENSORFLOW_CORE_DATA_METRIC_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// Exports the metrics for `GetNext` calls by tf.data iterators. When the user
// calls `RecordStart` and `RecordStop`, it will export a latency sample. It
// also exports throughput, tf.data iterator life time, etc. This class is
// thread-safe. Example usage:
//
//   ```
//   IteratorMetricsCollector metrics_collector(DEVICE_CPU, env);
//   absl::Time start_time = metrics_collector.RecordStart();
//   auto status = iterator_->GetNext(IteratorContext(std::move(params)),
//                                    out_tensors, end_of_sequence);
//   metrics_collector.RecordStop(start_time, *out_tensors);
//   ```
class IteratorMetricsCollector {
 public:
  // Constructs a `IteratorMetricsCollector`. `device_type` is one of the
  // devices defined in `types.h` (DEVICE_CPU, DEVICE_GPU, DEVICE_TPU, etc).
  // We only collect metrics for CPU devices. This is a heuristic to avoid
  // collecting metrics for device-side iterators created by the multi-device
  // iterator mechanism.
  IteratorMetricsCollector(const std::string& device_type, const Env& env);

  // Starts the timer for the next `GetNext` call. Returns the start time.
  absl::Time RecordStart();

  // Records metrics for the most recent `GetNext` call, including the latency,
  // bytes fetched, iterator life time, etc. `start_time` is the start time
  // returned by `RecordStart`. `output` is the output of the `GetNext` call.
  void RecordStop(absl::Time start_time, const std::vector<Tensor>& output);

 private:
  // We only collect metrics for CPU devices.
  bool ShouldCollectMetrics() const;

  // One of the devices defined in `types.h`
  // (DEVICE_CPU, DEVICE_GPU, DEVICE_TPU, etc).
  const std::string device_type_;
  const Env& env_;

  mutex mu_;

  // Records the number of currently active `GetNext` calls.
  uint64_t num_active_calls_ TF_GUARDED_BY(mu_) = 0;

  // Records the start time (in microseconds) of the first `RecordStart()` call
  // that followed the last period of inactivity.
  uint64_t first_start_time_us_ TF_GUARDED_BY(mu_) = 0;

  // Records the end time (in microseconds) of the most recent `RecordStop()`
  // call.
  uint64_t end_time_us_ TF_GUARDED_BY(mu_) = 0;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_METRIC_UTILS_H_
