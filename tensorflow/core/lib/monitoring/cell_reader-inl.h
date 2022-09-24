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
#ifndef TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_INL_H_
#define TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_INL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/test_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace monitoring {
namespace testing {
namespace internal {

// Returns a snapshot of the metrics collected at the time of calling.
std::unique_ptr<CollectedMetrics> CollectMetrics();

// Returns whether this is a cumulative or gauge metric.
MetricKind GetMetricKind(const CollectedMetrics& metrics,
                         const std::string& metric_name);

// Returns the points collected for `metric_name` associated with the `labels`.
// A `Point` represents a data point collected for the metric. For example,
// suppose a counter is incremented 3 times, then its points are 1, 2, 3.
//
// If the metric does not exist, it returns a `NotFound` error. If the number of
// labels does not match the metric definition, it returns an `InvalidArgument`
// error.
StatusOr<std::vector<Point>> GetPoints(const CollectedMetrics& metrics,
                                       const std::string& metric_name,
                                       const std::vector<std::string>& labels);

// Returns the `Point` that corresponds to the latest data point collected for
// `metric_name`, associated with the `labels`.
//
// If the metric does not exist, it returns a `NotFound` error. If the metric
// exists but no data is collected, it returns an `Unavailable` error. If the
// number of labels does not match the metric definition, it returns an
// `InvalidArgument` error.
StatusOr<Point> GetLatestPoint(const CollectedMetrics& metrics,
                               const std::string& metric_name,
                               const std::vector<std::string>& labels);

// Returns the value of `point`. Currently, only int64_t (counter) values are
// supported.
template <typename ValueType>
ValueType GetValue(const Point& point) {
  LOG(FATAL) << "Invalid argument: Tensorflow CellReader does not support type "
             << typeid(ValueType).name();
}

template <>
int64_t GetValue(const Point& point);

template <>
std::string GetValue(const Point& point);

template <>
bool GetValue(const Point& point);

template <>
Histogram GetValue(const Point& point);

template <>
Percentiles GetValue(const Point& point);

// Returns the latest value for `metric_name`, associated with the `labels`. If
// the metric has not collected any data, it returns a default value appropriate
// for `ValueType`. If the metric does not exist, or the wrong number of labels
// is provided, it will crash.
template <typename ValueType>
ValueType GetLatestValueOrDefault(const CollectedMetrics& metrics,
                                  const std::string& metric_name,
                                  const std::vector<std::string>& labels,
                                  const ValueType default_value = ValueType()) {
  StatusOr<Point> latest_point = GetLatestPoint(metrics, metric_name, labels);
  if (errors::IsUnavailable(latest_point.status())) {
    return std::move(default_value);
  }
  if (!latest_point.ok()) {
    LOG(FATAL) << "Failed to read from tfstreamz: " << latest_point.status();
  }
  return GetValue<ValueType>(*latest_point);
}

// Returns the difference between two values. Currently, only int64_t (counter)
// values are supported.
template <typename ValueType>
ValueType GetDelta(const ValueType& a, const ValueType& b) {
  LOG(FATAL) << "Invalid argument: Tensorflow CellReader does not support type "
             << typeid(ValueType).name();
}

template <>
int64_t GetDelta(const int64_t& a, const int64_t& b);

template <>
Histogram GetDelta(const Histogram& a, const Histogram& b);

template <>
Percentiles GetDelta(const Percentiles& a, const Percentiles& b);

template <>
std::string GetDelta(const std::string& a, const std::string& b);

template <>
bool GetDelta(const bool& a, const bool& b);

}  // namespace internal
}  // namespace testing
}  // namespace monitoring
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_INL_H_
