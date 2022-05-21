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
#include "tensorflow/core/data/metric_utils.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace data {
namespace {

// Safely subtracts `x` from `y` avoiding underflow.
uint64_t safe_sub(uint64_t x, uint64_t y) { return x >= y ? x - y : 0; }

}  // namespace

IteratorMetricsCollector::IteratorMetricsCollector(
    const std::string& device_type, const Env& env)
    : device_type_(device_type), env_(env) {}

void IteratorMetricsCollector::RecordStart() {
  if (!ShouldCollectMetrics()) {
    return;
  }

  const uint64_t start_time_us = env_.NowMicros();
  mutex_lock l(mu_);
  start_time_us_ = start_time_us;
  if (end_time_us_ == 0) {
    // We initialize `end_time_us_` to the start time of the first request to
    // make it possible to use the delta between `end_time_us_` and subsequent
    // `GetNext()` end time to incrementally collect the duration of the
    // iterator's lifetime.
    end_time_us_ = start_time_us_;
  }
  uint64_t gap_time_us = 0;
  if (num_active_calls_ == 0) {
    first_start_time_us_ = start_time_us_;
    gap_time_us = safe_sub(start_time_us_, end_time_us_);
  }
  metrics::RecordTFDataIteratorGap(gap_time_us);
  num_active_calls_++;
}

void IteratorMetricsCollector::RecordStop(const std::vector<Tensor>& output) {
  if (!ShouldCollectMetrics()) {
    return;
  }

  const uint64_t end_time_us = env_.NowMicros();
  IncrementThroughput(GetTotalBytes(output));
  mutex_lock l(mu_);
  AddLatencySample(safe_sub(end_time_us, start_time_us_));
  metrics::RecordTFDataIteratorLifetime(safe_sub(end_time_us, end_time_us_));
  end_time_us_ = std::max(end_time_us_, end_time_us);
  num_active_calls_--;
  if (num_active_calls_ == 0) {
    metrics::RecordTFDataIteratorBusy(
        safe_sub(end_time_us_, first_start_time_us_));
  }
}

bool IteratorMetricsCollector::ShouldCollectMetrics() const {
  return device_type_ == DEVICE_CPU;
}

}  // namespace data
}  // namespace tensorflow
