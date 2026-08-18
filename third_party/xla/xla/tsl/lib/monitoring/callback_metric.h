/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_TSL_LIB_MONITORING_CALLBACK_METRIC_H_
#define XLA_TSL_LIB_MONITORING_CALLBACK_METRIC_H_

#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "xla/tsl/lib/monitoring/gauge.h"
#include "xla/tsl/lib/monitoring/label_array_utils.h"
#include "xla/tsl/lib/monitoring/metric_def.h"

namespace tsl {
namespace monitoring {

// A base class for CallbackMetric to allow CallbackTrigger to manage multiple
// CallbackMetrics with different ValueTypes or NumLabels in a single vector.
class CallbackMetricBase {
 public:
  virtual ~CallbackMetricBase() = default;
  virtual void RegisterTrigger(std::function<void()> trigger_fn) = 0;
  virtual void UnregisterTrigger() = 0;
};

// A stateful class for updating a callback-based metric.
//
// This class is similar to Gauge, but its values are updated via a trigger
// callback that is executed when metrics are collected by CollectionRegistry.
//
// This class is thread-safe.
template <typename ValueType, int NumLabels>
class CallbackMetric : public CallbackMetricBase {
 public:
  ~CallbackMetric() override {
    // Deleted here, before the metric_def is destroyed.
    registration_handle_.reset();
  }

  // Creates the metric based on the metric-definition arguments.
  template <typename... MetricDefArgs>
  static CallbackMetric* New(MetricDefArgs&&... metric_def_args) {
    return new CallbackMetric(
        MetricDef<MetricKind::kGauge, ValueType, NumLabels>(
            std::forward<MetricDefArgs>(metric_def_args)...));
  }

  // Creates the metric based on the metric-definition arguments without using
  // the heap.
  template <typename... MetricDefArgs>
  static absl::NoDestructor<CallbackMetric> NoDestructor(
      MetricDefArgs&&... metric_def_args) {
    return absl::NoDestructor<CallbackMetric>(
        MetricDef<MetricKind::kGauge, ValueType, NumLabels>(
            std::forward<MetricDefArgs>(metric_def_args)...));
  }

  // Retrieves the cell for the specified labels, creating it on demand if not
  // already present.
  template <typename... Labels>
  GaugeCell<ValueType>* GetCell(const Labels&... labels)
      ABSL_LOCKS_EXCLUDED(mu_) {
    static_assert(sizeof...(Labels) == NumLabels,
                  "Mismatch between CallbackMetric<ValueType, NumLabels> and "
                  "number of labels provided in GetCell(...).");

    LabelViewArray label_view_array = {{labels...}};
    absl::MutexLock l(&mu_);
    const auto found_it = cells_.find(label_view_array);
    if (found_it != cells_.end()) {
      return found_it->second.get();
    }
    return cells_
        .emplace(std::piecewise_construct,
                 std::forward_as_tuple(LabelArray{std::string(labels)...}),
                 std::forward_as_tuple(
                     std::make_unique<GaugeCell<ValueType>>(ValueType())))
        .first->second.get();
  }

  // Sets the value for the cell with the specified labels.
  template <typename... Labels>
  void Set(const ValueType& value, const Labels&... labels) {
    GetCell(labels...)->Set(value);
  }

  void RegisterTrigger(std::function<void()> trigger_fn) override {
    absl::MutexLock l(&mu_);
    trigger_fn_ = std::move(trigger_fn);
  }

  void UnregisterTrigger() override {
    absl::MutexLock l(&mu_);
    trigger_fn_ = nullptr;
  }

  absl::Status GetStatus() { return status_; }

 private:
  friend class absl::NoDestructor<CallbackMetric<ValueType, NumLabels>>;

  explicit CallbackMetric(
      const MetricDef<MetricKind::kGauge, ValueType, NumLabels>& metric_def)
      : metric_def_(metric_def),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [this](MetricCollectorGetter getter) {
              std::function<void()> trigger_copy;
              {
                absl::MutexLock l(&mu_);
                trigger_copy = trigger_fn_;
              }
              if (trigger_copy) {
                trigger_copy();
              }

              auto metric_collector = getter.Get(&metric_def_);
              absl::MutexLock l(&mu_);
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

  LabelArrayMap<GaugeCell<ValueType>, NumLabels> cells_ ABSL_GUARDED_BY(mu_);
  std::function<void()> trigger_fn_ ABSL_GUARDED_BY(mu_);

  const MetricDef<MetricKind::kGauge, ValueType, NumLabels> metric_def_;
  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  CallbackMetric(const CallbackMetric&) = delete;
  void operator=(const CallbackMetric&) = delete;
};

// RAII helper to register a trigger function with a set of CallbackMetrics.
class CallbackTrigger {
 public:
  CallbackTrigger(std::function<void()> trigger_fn,
                  std::initializer_list<CallbackMetricBase*> metrics)
      : CallbackTrigger(
            std::move(trigger_fn),
            std::vector<CallbackMetricBase*>(metrics.begin(), metrics.end())) {}

  CallbackTrigger(std::function<void()> trigger_fn,
                  std::vector<CallbackMetricBase*> metrics)
      : metrics_(std::move(metrics)) {
    for (auto* metric : metrics_) {
      if (metric != nullptr) {
        metric->RegisterTrigger(trigger_fn);
      }
    }
  }

  ~CallbackTrigger() {
    for (auto* metric : metrics_) {
      if (metric != nullptr) {
        metric->UnregisterTrigger();
      }
    }
  }

  CallbackTrigger(const CallbackTrigger&) = delete;
  CallbackTrigger& operator=(const CallbackTrigger&) = delete;

 private:
  std::vector<CallbackMetricBase*> metrics_;
};

}  // namespace monitoring
}  // namespace tsl

#endif  // XLA_TSL_LIB_MONITORING_CALLBACK_METRIC_H_
