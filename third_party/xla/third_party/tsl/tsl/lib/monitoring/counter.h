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

#ifndef TENSORFLOW_TSL_LIB_MONITORING_COUNTER_H_
#define TENSORFLOW_TSL_LIB_MONITORING_COUNTER_H_

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tsl/platform/platform.h"
// clang-format on

// We replace this implementation with a null implementation for mobile
// platforms.
#ifdef IS_MOBILE_PLATFORM

#include "tsl/platform/macros.h"
#include "tsl/platform/status.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace monitoring {

// CounterCell which has a null implementation.
class CounterCell {
 public:
  CounterCell() {}
  ~CounterCell() {}

  void IncrementBy(int64 step) {}
  int64 value() const { return 0; }

 private:
  CounterCell(const CounterCell&) = delete;
  void operator=(const CounterCell&) = delete;
};

// Counter which has a null implementation.
template <int NumLabels>
class Counter {
 public:
  ~Counter() {}

  template <typename... MetricDefArgs>
  static Counter* New(MetricDefArgs&&... metric_def_args) {
    return new Counter<NumLabels>();
  }

  template <typename... Labels>
  CounterCell* GetCell(const Labels&... labels) {
    return &default_counter_cell_;
  }

  Status GetStatus() { return OkStatus(); }

 private:
  Counter() {}

  CounterCell default_counter_cell_;

  Counter(const Counter&) = delete;
  void operator=(const Counter&) = delete;
};

}  // namespace monitoring
}  // namespace tsl

#else  // IS_MOBILE_PLATFORM

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <tuple>

#include "tsl/lib/monitoring/collection_registry.h"
#include "tsl/lib/monitoring/metric_def.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/status.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {
namespace monitoring {

// CounterCell stores each value of an Counter.
//
// A cell can be passed off to a module which may repeatedly update it without
// needing further map-indexing computations. This improves both encapsulation
// (separate modules can own a cell each, without needing to know about the map
// to which both cells belong) and performance (since map indexing and
// associated locking are both avoided).
//
// This class is thread-safe.
class CounterCell {
 public:
  explicit CounterCell(int64_t value) : value_(value) {}
  ~CounterCell() {}

  // Atomically increments the value by step.
  // REQUIRES: Step be non-negative.
  void IncrementBy(int64_t step);

  // Retrieves the current value.
  int64_t value() const;

 private:
  std::atomic<int64_t> value_;

  CounterCell(const CounterCell&) = delete;
  void operator=(const CounterCell&) = delete;
};

// A stateful class for updating a cumulative integer metric.
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
class Counter {
 public:
  ~Counter() {
    // Deleted here, before the metric_def is destroyed.
    registration_handle_.reset();
  }

  // Creates the metric based on the metric-definition arguments.
  //
  // Example;
  // auto* counter_with_label = Counter<1>::New("/tensorflow/counter",
  //   "Tensorflow counter", "MyLabelName");
  template <typename... MetricDefArgs>
  static Counter* New(MetricDefArgs&&... metric_def_args);

  // Retrieves the cell for the specified labels, creating it on demand if
  // not already present.
  template <typename... Labels>
  CounterCell* GetCell(const Labels&... labels) TF_LOCKS_EXCLUDED(mu_);

  Status GetStatus() { return status_; }

 private:
  explicit Counter(
      const MetricDef<MetricKind::kCumulative, int64_t, NumLabels>& metric_def)
      : metric_def_(metric_def),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [&](MetricCollectorGetter getter) {
              auto metric_collector = getter.Get(&metric_def_);

              mutex_lock l(mu_);
              for (const auto& cell : cells_) {
                metric_collector.CollectValue(cell.first, cell.second.value());
              }
            })) {
    if (registration_handle_) {
      status_ = OkStatus();
    } else {
      status_ = Status(absl::StatusCode::kAlreadyExists,
                       "Another metric with the same name already exists.");
    }
  }

  mutable mutex mu_;

  Status status_;

  using LabelArray = std::array<string, NumLabels>;
  std::map<LabelArray, CounterCell> cells_ TF_GUARDED_BY(mu_);

  // The metric definition. This will be used to identify the metric when we
  // register it for collection.
  const MetricDef<MetricKind::kCumulative, int64_t, NumLabels> metric_def_;

  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  Counter(const Counter&) = delete;
  void operator=(const Counter&) = delete;
};

////
//  Implementation details follow. API readers may skip.
////

inline void CounterCell::IncrementBy(const int64_t step) {
  DCHECK_LE(0, step) << "Must not decrement cumulative metrics.";
  value_ += step;
}

inline int64_t CounterCell::value() const { return value_; }

template <int NumLabels>
template <typename... MetricDefArgs>
Counter<NumLabels>* Counter<NumLabels>::New(
    MetricDefArgs&&... metric_def_args) {
  return new Counter<NumLabels>(
      MetricDef<MetricKind::kCumulative, int64_t, NumLabels>(
          std::forward<MetricDefArgs>(metric_def_args)...));
}

template <int NumLabels>
template <typename... Labels>
CounterCell* Counter<NumLabels>::GetCell(const Labels&... labels)
    TF_LOCKS_EXCLUDED(mu_) {
  // Provides a more informative error message than the one during array
  // construction below.
  static_assert(sizeof...(Labels) == NumLabels,
                "Mismatch between Counter<NumLabels> and number of labels "
                "provided in GetCell(...).");

  const LabelArray& label_array = {{labels...}};
  mutex_lock l(mu_);
  const auto found_it = cells_.find(label_array);
  if (found_it != cells_.end()) {
    return &(found_it->second);
  }
  return &(cells_
               .emplace(std::piecewise_construct,
                        std::forward_as_tuple(label_array),
                        std::forward_as_tuple(0))
               .first->second);
}

}  // namespace monitoring
}  // namespace tsl

#endif  // IS_MOBILE_PLATFORM
#endif  // TENSORFLOW_TSL_LIB_MONITORING_COUNTER_H_
