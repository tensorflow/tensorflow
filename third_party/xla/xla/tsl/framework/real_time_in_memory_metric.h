/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_TSL_FRAMEWORK_REAL_TIME_IN_MEMORY_METRIC_H_
#define XLA_TSL_FRAMEWORK_REAL_TIME_IN_MEMORY_METRIC_H_

#include <atomic>

namespace tsl {

// Represents a metric with backing storage in local RAM, for exporting real
// time metrics for consumers that live in the same process. It currently only
// supports a simple scalar value. The implementation of this class is lossy but
// minimizes overhead, because there is usually no requirement for metrics
// consumer to get the exact value for any specific time point, but the metrics
// update is usually placed at the critical path of some request. This class is
// a replacement for streamz metric for the above described use case, not
// complimentary.
//
// This class is thread-safe.
//
// NOTE: Only integer and floating point values are supported.
template <typename T>
class RealTimeInMemoryMetric {
 public:
  RealTimeInMemoryMetric() : value_(T{0}) {}

  // Gets the current value of this metric.
  T Get() const { return value_.load(std::memory_order_relaxed); }

  // Updates the current value of this metric.
  void Set(T new_value) { value_.store(new_value, std::memory_order_relaxed); }

  RealTimeInMemoryMetric(const RealTimeInMemoryMetric&) = delete;
  RealTimeInMemoryMetric& operator=(const RealTimeInMemoryMetric&) = delete;
  RealTimeInMemoryMetric(RealTimeInMemoryMetric&&) = delete;
  RealTimeInMemoryMetric& operator=(RealTimeInMemoryMetric&&) = delete;

  static_assert(std::is_arithmetic_v<T>);

 private:
  std::atomic<T> value_;
};

}  // namespace tsl

#endif  // XLA_TSL_FRAMEWORK_REAL_TIME_IN_MEMORY_METRIC_H_
