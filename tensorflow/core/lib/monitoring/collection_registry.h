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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
#define TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_

#include <map>
#include <memory>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace monitoring {

namespace test_util {
class CollectionRegistryTestAccess;
}  // namespace test_util

namespace internal {
class Collector;
}  // namespace internal

// Metric implementations would get an instance of this class using the
// MetricCollectorGetter in the collection-function lambda, so that their values
// can be collected.
//
// Read the documentation on CollectionRegistry::Register() for more details.
//
// For example:
//   auto metric_collector = metric_collector_getter->Get(&metric_def);
//   metric_collector.CollectValue(some_labels, some_value);
//   metric_collector.CollectValue(others_labels, other_value);
//
// This class is NOT thread-safe.
template <MetricKind metric_kind, typename Value, int NumLabels>
class MetricCollector {
 public:
  ~MetricCollector() = default;

  // Collects the value with these labels.
  void CollectValue(const std::array<string, NumLabels>& labels,
                    const Value& value);

 private:
  friend class internal::Collector;

  MetricCollector(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def,
      const uint64 registration_time_millis,
      internal::Collector* const collector, PointSet* const point_set)
      : metric_def_(metric_def),
        registration_time_millis_(registration_time_millis),
        collector_(collector),
        point_set_(point_set) {
    point_set_->metric_name = std::string(metric_def->name());
  }

  const MetricDef<metric_kind, Value, NumLabels>* const metric_def_;
  const uint64 registration_time_millis_;
  internal::Collector* const collector_;
  PointSet* const point_set_;

  // This is made copyable because we can't hand out references of this class
  // from MetricCollectorGetter because this class is templatized, and we need
  // MetricCollectorGetter not to be templatized and hence MetricCollectorGetter
  // can't own an instance of this class.
};

// Returns a MetricCollector with the same template parameters as the
// metric-definition, so that the values of a metric can be collected.
//
// The collection-function defined by a metric takes this as a parameter.
//
// Read the documentation on CollectionRegistry::Register() for more details.
class MetricCollectorGetter {
 public:
  // Returns the MetricCollector with the same template parameters as the
  // metric_def.
  template <MetricKind metric_kind, typename Value, int NumLabels>
  MetricCollector<metric_kind, Value, NumLabels> Get(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def);

 private:
  friend class internal::Collector;

  MetricCollectorGetter(internal::Collector* const collector,
                        const AbstractMetricDef* const allowed_metric_def,
                        const uint64 registration_time_millis)
      : collector_(collector),
        allowed_metric_def_(allowed_metric_def),
        registration_time_millis_(registration_time_millis) {}

  internal::Collector* const collector_;
  const AbstractMetricDef* const allowed_metric_def_;
  const uint64 registration_time_millis_;
};

// A collection registry for metrics.
//
// Metrics are registered here so that their state can be collected later and
// exported.
//
// This class is thread-safe.
class CollectionRegistry {
 public:
  ~CollectionRegistry() = default;

  // Returns the default registry for the process.
  //
  // This registry belongs to this library and should never be deleted.
  static CollectionRegistry* Default();

  using CollectionFunction = std::function<void(MetricCollectorGetter getter)>;

  // Registers the metric and the collection-function which can be used to
  // collect its values. Returns a Registration object, which when upon
  // destruction would cause the metric to be unregistered from this registry.
  //
  // IMPORTANT: Delete the handle before the metric-def is deleted.
  //
  // Example usage;
  // CollectionRegistry::Default()->Register(
  //   &metric_def,
  //   [&](MetricCollectorGetter getter) {
  //     auto metric_collector = getter.Get(&metric_def);
  //     for (const auto& cell : cells) {
  //       metric_collector.CollectValue(cell.labels(), cell.value());
  //     }
  //   });
  class RegistrationHandle;
  std::unique_ptr<RegistrationHandle> Register(
      const AbstractMetricDef* metric_def,
      const CollectionFunction& collection_function)
      LOCKS_EXCLUDED(mu_) TF_MUST_USE_RESULT;

  // Options for collecting metrics.
  struct CollectMetricsOptions {
    CollectMetricsOptions() {}
    bool collect_metric_descriptors = true;
  };
  // Goes through all the registered metrics, collects their definitions
  // (optionally) and current values and returns them in a standard format.
  std::unique_ptr<CollectedMetrics> CollectMetrics(
      const CollectMetricsOptions& options) const;

 private:
  friend class test_util::CollectionRegistryTestAccess;
  friend class internal::Collector;

  CollectionRegistry(Env* env);

  // Unregisters the metric from this registry. This is private because the
  // public interface provides a Registration handle which automatically calls
  // this upon destruction.
  void Unregister(const AbstractMetricDef* metric_def) LOCKS_EXCLUDED(mu_);

  // TF environment, mainly used for timestamping.
  Env* const env_;

  mutable mutex mu_;

  // Information required for collection.
  struct CollectionInfo {
    const AbstractMetricDef* const metric_def;
    CollectionFunction collection_function;
    uint64 registration_time_millis;
  };
  std::map<StringPiece, CollectionInfo> registry_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(CollectionRegistry);
};

////
// Implementation details follow. API readers may skip.
////

class CollectionRegistry::RegistrationHandle {
 public:
  RegistrationHandle(CollectionRegistry* const export_registry,
                     const AbstractMetricDef* const metric_def)
      : export_registry_(export_registry), metric_def_(metric_def) {}

  ~RegistrationHandle() { export_registry_->Unregister(metric_def_); }

 private:
  CollectionRegistry* const export_registry_;
  const AbstractMetricDef* const metric_def_;
};

namespace internal {

template <typename Value>
void CollectValue(const Value& value, Point* point);

template <>
inline void CollectValue(const int64& value, Point* const point) {
  point->value_type = ValueType::kInt64;
  point->int64_value = value;
}

template <>
inline void CollectValue(const string& value, Point* const point) {
  point->value_type = ValueType::kString;
  point->string_value = value;
}

template <>
inline void CollectValue(const bool& value, Point* const point) {
  point->value_type = ValueType::kBool;
  point->bool_value = value;
}

template <>
inline void CollectValue(const HistogramProto& value, Point* const point) {
  point->value_type = ValueType::kHistogram;
  // This is inefficient. If and when we hit snags, we can change the API to do
  // this more efficiently.
  point->histogram_value = value;
}

// Used by the CollectionRegistry class to collect all the values of all the
// metrics in the registry. This is an implementation detail of the
// CollectionRegistry class, please do not depend on this.
//
// This cannot be a private nested class because we need to forward declare this
// so that the MetricCollector and MetricCollectorGetter classes can be friends
// with it.
//
// This class is thread-safe.
class Collector {
 public:
  Collector(const uint64 collection_time_millis)
      : collected_metrics_(new CollectedMetrics()),
        collection_time_millis_(collection_time_millis) {}

  template <MetricKind metric_kind, typename Value, int NumLabels>
  MetricCollector<metric_kind, Value, NumLabels> GetMetricCollector(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def,
      const uint64 registration_time_millis,
      internal::Collector* const collector) LOCKS_EXCLUDED(mu_) {
    auto* const point_set = [&]() {
      mutex_lock l(mu_);
      return collected_metrics_->point_set_map
          .insert(std::make_pair(std::string(metric_def->name()),
                                 std::unique_ptr<PointSet>(new PointSet())))
          .first->second.get();
    }();
    return MetricCollector<metric_kind, Value, NumLabels>(
        metric_def, registration_time_millis, collector, point_set);
  }

  uint64 collection_time_millis() const { return collection_time_millis_; }

  void CollectMetricDescriptor(const AbstractMetricDef* const metric_def)
      LOCKS_EXCLUDED(mu_);

  void CollectMetricValues(
      const CollectionRegistry::CollectionInfo& collection_info);

  std::unique_ptr<CollectedMetrics> ConsumeCollectedMetrics()
      LOCKS_EXCLUDED(mu_);

 private:
  mutable mutex mu_;
  std::unique_ptr<CollectedMetrics> collected_metrics_ GUARDED_BY(mu_);
  const uint64 collection_time_millis_;

  TF_DISALLOW_COPY_AND_ASSIGN(Collector);
};

// Write the timestamps for the point based on the MetricKind.
//
// Gauge metrics will have start and end timestamps set to the collection time.
//
// Cumulative metrics will have the start timestamp set to the time when the
// collection function was registered, while the end timestamp will be set to
// the collection time.
template <MetricKind kind>
void WriteTimestamps(const uint64 registration_time_millis,
                     const uint64 collection_time_millis, Point* const point);

template <>
inline void WriteTimestamps<MetricKind::kGauge>(
    const uint64 registration_time_millis, const uint64 collection_time_millis,
    Point* const point) {
  point->start_timestamp_millis = collection_time_millis;
  point->end_timestamp_millis = collection_time_millis;
}

template <>
inline void WriteTimestamps<MetricKind::kCumulative>(
    const uint64 registration_time_millis, const uint64 collection_time_millis,
    Point* const point) {
  point->start_timestamp_millis = registration_time_millis;
  // There's a chance that the clock goes backwards on the same machine, so we
  // protect ourselves against that.
  point->end_timestamp_millis =
      registration_time_millis < collection_time_millis
          ? collection_time_millis
          : registration_time_millis;
}

}  // namespace internal

template <MetricKind metric_kind, typename Value, int NumLabels>
void MetricCollector<metric_kind, Value, NumLabels>::CollectValue(
    const std::array<string, NumLabels>& labels, const Value& value) {
  point_set_->points.emplace_back(new Point());
  auto* const point = point_set_->points.back().get();
  const std::vector<string> label_descriptions =
      metric_def_->label_descriptions();
  point->labels.reserve(NumLabels);
  for (int i = 0; i < NumLabels; ++i) {
    point->labels.push_back({});
    auto* const label = &point->labels.back();
    label->name = label_descriptions[i];
    label->value = labels[i];
  }
  internal::CollectValue(value, point);
  internal::WriteTimestamps<metric_kind>(
      registration_time_millis_, collector_->collection_time_millis(), point);
}

template <MetricKind metric_kind, typename Value, int NumLabels>
MetricCollector<metric_kind, Value, NumLabels> MetricCollectorGetter::Get(
    const MetricDef<metric_kind, Value, NumLabels>* const metric_def) {
  if (allowed_metric_def_ != metric_def) {
    LOG(FATAL) << "Expected collection for: " << allowed_metric_def_->name()
               << " but instead got: " << metric_def->name();
  }

  return collector_->GetMetricCollector(metric_def, registration_time_millis_,
                                        collector_);
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
