/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_LIB_MONITORING_COUNTER_GAUGE_H_
#define XLA_TSL_LIB_MONITORING_COUNTER_GAUGE_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "xla/tsl/lib/monitoring/label_array_utils.h"
#include "xla/tsl/lib/monitoring/metric_def.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {
namespace monitoring {

// CounterGaugeCell stores each value of an CounterGauge.
//
// This class is thread-safe.
class CounterGaugeCell {
 public:
  explicit CounterGaugeCell(int64_t value) : value_(value) {}
  CounterGaugeCell() = default;

  // Atomically increments the value by step. `step` can be any value.
  void IncrementBy(int64_t step);

  // Atomically increments the value by 1.
  void Increment();

  // Atomically decrements the value by 1.
  void Decrement();

  // Retrieves the current value.
  int64_t value() const;

 private:
  std::atomic<int64_t> value_;

  CounterGaugeCell(const CounterGaugeCell&) = delete;
  void operator=(const CounterGaugeCell&) = delete;
};

// A stateful class for updating a gauge integer metric.
//
// This class encapsulates a set of values (or a single value for a label-less
// metric). Each value is identified by a tuple of labels. The class allows the
// user to increment each value.
//
// Counter allocates storage and maintains a cell for each value. You can
// retrieve an individual cell using a label-tuple and update it separately.
// This improves performance since operations related to retrieval, like
// map-indexing and locking, are avoided.
//
// This class is thread-safe.
template <int NumLabels>
class CounterGauge {
 public:
  ~CounterGauge() {
    // Deleted here, before the metric_def is destroyed.
    registration_handle_.reset();
  }

  // Creates the metric based on the metric-definition arguments.
  //
  // Example:
  // auto* counter_with_label = CounterGauge<1>::New(
  //     "/tensorflow/counter", "Tensorflow counter", "MyLabelName");
  template <typename... MetricDefArgs>
  static CounterGauge* New(MetricDefArgs&&... metric_def_args);

  // Creates the metric based on the metric-definition arguments. Doesn't use
  // the heap; instead, allocates a static stack object whose destructor will
  // never be called. (See `absl::NoDestructor` documentation.)
  //
  // Example:
  // auto counter_with_label = CounterGauge<1>::MakeStatic(
  //     "/tensorflow/counter", "Tensorflow counter", "MyLabelName");
  template <typename... MetricDefArgs>
  static absl::NoDestructor<CounterGauge> MakeStatic(
      MetricDefArgs&&... metric_def_args);

  // Retrieves the cell for the specified labels, creating it on demand if
  // not already present.
  template <typename... Labels>
  CounterGaugeCell* GetCell(const Labels&... labels) ABSL_LOCKS_EXCLUDED(mu_);

  absl::Status GetStatus() { return status_; }

 private:
  friend class absl::NoDestructor<CounterGauge<NumLabels>>;

  explicit CounterGauge(
      const MetricDef<MetricKind::kGauge, int64_t, NumLabels>& metric_def)
      : metric_def_(metric_def),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [&](MetricCollectorGetter getter) {
              auto metric_collector = getter.Get(&metric_def_);

              absl::MutexLock l(mu_);
              for (const auto& cell : cells_) {
                metric_collector.CollectValue(cell.first, cell.second->value());
              }
            })) {
    if (registration_handle_) {
      status_ = absl::OkStatus();
    } else {
      status_ =
          absl::Status(absl::StatusCode::kAlreadyExists,
                       "Another metric with the same name already exists.");
    }
  }

  mutable absl::Mutex mu_;

  absl::Status status_;

  using LabelArray = std::array<std::string, NumLabels>;
  using LabelViewArray = std::array<absl::string_view, NumLabels>;

  LabelArrayMap<CounterGaugeCell, NumLabels> cells_ ABSL_GUARDED_BY(mu_);

  // The metric definition. This will be used to identify the metric when we
  // register it for collection.
  const MetricDef<MetricKind::kGauge, int64_t, NumLabels> metric_def_;

  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  CounterGauge(const CounterGauge&) = delete;
  void operator=(const CounterGauge&) = delete;
};

////
//  Implementation details follow. API readers may skip.
////

inline void CounterGaugeCell::IncrementBy(int64_t step) { value_ += step; }

inline int64_t CounterGaugeCell::value() const { return value_; }

inline void CounterGaugeCell::Increment() { IncrementBy(1); }

inline void CounterGaugeCell::Decrement() { IncrementBy(-1); }

template <int NumLabels>
template <typename... MetricDefArgs>
CounterGauge<NumLabels>* CounterGauge<NumLabels>::New(
    MetricDefArgs&&... metric_def_args) {
  return new CounterGauge<NumLabels>(
      MetricDef<MetricKind::kGauge, int64_t, NumLabels>(
          std::forward<MetricDefArgs>(metric_def_args)...));
}

template <int NumLabels>
template <typename... MetricDefArgs>
absl::NoDestructor<CounterGauge<NumLabels>> CounterGauge<NumLabels>::MakeStatic(
    MetricDefArgs&&... metric_def_args) {
  return absl::NoDestructor<CounterGauge<NumLabels>>(
      MetricDef<MetricKind::kGauge, int64_t, NumLabels>(
          std::forward<MetricDefArgs>(metric_def_args)...));
}

template <int NumLabels>
template <typename... Labels>
CounterGaugeCell* CounterGauge<NumLabels>::GetCell(const Labels&... labels)
    ABSL_LOCKS_EXCLUDED(mu_) {
  // Provides a more informative error message than the one during array
  // construction below.
  static_assert(sizeof...(Labels) == NumLabels,
                "Mismatch between CounterGauge<NumLabels> and number of labels "
                "provided in GetCell(...).");

  LabelViewArray label_view_array = {{labels...}};
  absl::MutexLock l(mu_);
  const auto found_it = cells_.find(label_view_array);
  if (found_it != cells_.end()) {
    return found_it->second.get();
  }
  return cells_
      .emplace(std::piecewise_construct,
               std::forward_as_tuple(LabelArray{std::string(labels)...}),
               std::forward_as_tuple(std::make_unique<CounterGaugeCell>()))
      .first->second.get();
}

}  // namespace monitoring
}  // namespace tsl

#endif  // XLA_TSL_LIB_MONITORING_COUNTER_GAUGE_H_
