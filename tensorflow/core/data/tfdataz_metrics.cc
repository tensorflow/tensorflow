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
#include "tensorflow/core/data/tfdataz_metrics.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/time/time.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

ApproximateLatencyEstimator::ApproximateLatencyEstimator(const Env& env)
    : env_(env),
      last_updated_time_mins_(0),
      latency_value_counter_(0),
      latency_count_counter_(0),
      next_slot_(0) {
  for (int i = 0; i < kSlots; ++i) {
    latency_value_[i] = 0;
    latency_count_[i] = 0;
  }
}

void ApproximateLatencyEstimator::AddLatency(const int64_t latency_usec)
    TF_LOCKS_EXCLUDED(mu_) {
  UpdateRingBuffer();

  mutex_lock l(mu_);
  latency_value_counter_ += latency_usec;
  latency_count_counter_ += 1;
}

void ApproximateLatencyEstimator::UpdateRingBuffer() TF_LOCKS_EXCLUDED(mu_) {
  int64_t now_minutes =
      absl::ToInt64Minutes(absl::Microseconds(env_.NowMicros()));

  mutex_lock l(mu_);
  int64_t elapsed_minutes = now_minutes - last_updated_time_mins_;
  int64_t minutes_to_update = std::min(elapsed_minutes, kSlots);
  for (int i = 0; i < minutes_to_update; ++i) {
    latency_value_[next_slot_] = latency_value_counter_;
    latency_count_[next_slot_] = latency_count_counter_;
    IncrementNextSlot();
  }
  last_updated_time_mins_ = now_minutes;
}

void ApproximateLatencyEstimator::IncrementNextSlot()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  next_slot_ = (next_slot_ + 1) % kSlots;
}

int ApproximateLatencyEstimator::PrevSlot(int steps)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return (next_slot_ - steps + kSlots) % kSlots;
}

absl::Duration ApproximateLatencyEstimator::GetAverageLatency(Duration duration)
    TF_LOCKS_EXCLUDED(mu_) {
  UpdateRingBuffer();

  mutex_lock l(mu_);
  double interval_latency =
      static_cast<double>(latency_value_counter_ -
                          latency_value_[PrevSlot(static_cast<int>(duration))]);
  double interval_count =
      static_cast<double>(latency_count_counter_ -
                          latency_count_[PrevSlot(static_cast<int>(duration))]);
  if (interval_count == 0) {
    return absl::ZeroDuration();
  }
  return absl::Duration(absl::Microseconds(interval_latency)) / interval_count;
}

TfDatazMetricsCollector::TfDatazMetricsCollector(
    const Env& env, DatasetBaseIterator* iterator,
    std::shared_ptr<model::Model> model)
    : iterator_(iterator), model_(std::move(model)), latency_estimator_(env) {}

void TfDatazMetricsCollector::RecordGetNextLatency(
    int64_t get_next_latency_usec) {
  if (get_next_latency_usec > 0) {
    latency_estimator_.AddLatency(get_next_latency_usec);
  }
}

absl::Duration TfDatazMetricsCollector::GetAverageLatencyForLastOneMinute() {
  return latency_estimator_.GetAverageLatency(
      ApproximateLatencyEstimator::Duration::kMinute);
}

absl::Duration TfDatazMetricsCollector::GetAverageLatencyForLastFiveMinutes() {
  return latency_estimator_.GetAverageLatency(
      ApproximateLatencyEstimator::Duration::kFiveMinutes);
}

absl::Duration TfDatazMetricsCollector::GetAverageLatencyForLastSixtyMinutes() {
  return latency_estimator_.GetAverageLatency(
      ApproximateLatencyEstimator::Duration::kSixtyMinutes);
}

std::optional<std::string> TfDatazMetricsCollector::DatasetName() {
  auto options = iterator_->dataset()->options();
  if (options.has_dataset_name()) {
    return std::make_optional(options.dataset_name());
  }
  return std::nullopt;
}

int64_t TfDatazMetricsCollector::GetIteratorTotalMemoryUsage() {
  return iterator_->TotalBufferedBytes();
}

std::shared_ptr<model::Model> TfDatazMetricsCollector::GetModel() {
  return model_;
}

namespace {
static mutex* get_tfdataz_metrics_registry_lock() {
  static mutex tfdataz_metrics_registry_lock(LINKER_INITIALIZED);
  return &tfdataz_metrics_registry_lock;
}

using TfDatazMetricsCollectors =
    absl::flat_hash_set<std::shared_ptr<TfDatazMetricsCollector>>;
TfDatazMetricsCollectors& tfdataz_metric_collectors() {
  static auto& collectors = *new TfDatazMetricsCollectors();
  return collectors;
}
}  // namespace

void TfDatazMetricsRegistry::Register(
    std::shared_ptr<TfDatazMetricsCollector> collector) {
  mutex_lock l(*get_tfdataz_metrics_registry_lock());
  tfdataz_metric_collectors().insert(collector);
}

void TfDatazMetricsRegistry::Deregister(
    std::shared_ptr<TfDatazMetricsCollector> collector) {
  mutex_lock l(*get_tfdataz_metrics_registry_lock());
  tfdataz_metric_collectors().erase(collector);
}

absl::flat_hash_set<std::shared_ptr<TfDatazMetricsCollector>>
TfDatazMetricsRegistry::GetIteratorMetricCollectors() {
  mutex_lock l(*get_tfdataz_metrics_registry_lock());
  return tfdataz_metric_collectors();
}

}  // namespace data
}  // namespace tensorflow
