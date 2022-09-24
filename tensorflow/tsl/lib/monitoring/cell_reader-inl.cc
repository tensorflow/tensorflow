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
#include "tensorflow/tsl/lib/monitoring/cell_reader-inl.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/tsl/lib/monitoring/collected_metrics.h"
#include "tensorflow/tsl/lib/monitoring/collection_registry.h"
#include "tensorflow/tsl/lib/monitoring/metric_def.h"
#include "tensorflow/tsl/lib/monitoring/test_utils.h"
#include "tensorflow/tsl/lib/monitoring/types.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tsl {
namespace monitoring {
namespace testing {
namespace internal {
namespace {

// Returns the labels of `point` as a vector of strings.
std::vector<std::string> GetLabels(const monitoring::Point& point) {
  std::vector<std::string> labels;
  labels.reserve(point.labels.size());
  for (const monitoring::Point::Label& label : point.labels) {
    labels.push_back(label.value);
  }
  return labels;
}
}  // namespace

std::unique_ptr<CollectedMetrics> CollectMetrics() {
  CollectionRegistry::CollectMetricsOptions options;
  return CollectionRegistry::Default()->CollectMetrics(options);
}

MetricKind GetMetricKind(const CollectedMetrics& metrics,
                         const std::string& metric_name) {
  auto metric_descriptor = metrics.metric_descriptor_map.find(metric_name);
  if (metric_descriptor == metrics.metric_descriptor_map.end()) {
    return MetricKind::kCumulative;
  }
  return metric_descriptor->second->metric_kind;
}

StatusOr<std::vector<Point>> GetPoints(const CollectedMetrics& metrics,
                                       const std::string& metric_name,
                                       const std::vector<std::string>& labels) {
  auto metric_descriptor = metrics.metric_descriptor_map.find(metric_name);
  if (metric_descriptor == metrics.metric_descriptor_map.end()) {
    return errors::NotFound("Metric descriptor is not found for metric ",
                            metric_name, ".");
  }
  const std::vector<string>& label_names =
      metric_descriptor->second->label_names;
  if (label_names.size() != labels.size()) {
    return errors::InvalidArgument(
        "Metric ", metric_name, " has ", label_names.size(), " labels: [",
        absl::StrJoin(label_names, ", "), "]. Got label values [",
        absl::StrJoin(labels, ", "), "].");
  }
  auto point_set = metrics.point_set_map.find(metric_name);
  if (point_set == metrics.point_set_map.end()) {
    return errors::NotFound("Metric point set is not found for metric ",
                            metric_name, ".");
  }

  std::vector<Point> points;
  for (const std::unique_ptr<Point>& point : point_set->second->points) {
    if (GetLabels(*point) == labels) {
      points.push_back(*point);
    }
  }
  return points;
}

StatusOr<Point> GetLatestPoint(const CollectedMetrics& metrics,
                               const std::string& metric_name,
                               const std::vector<std::string>& labels) {
  TF_ASSIGN_OR_RETURN(std::vector<Point> points,
                      GetPoints(metrics, metric_name, labels));
  if (points.empty()) {
    return errors::Unavailable("No data collected for metric ", metric_name,
                               " with labels [", absl::StrJoin(labels, ", "),
                               "].");
  }

  bool same_start_time =
      std::all_of(points.begin(), points.end(), [&points](const Point& point) {
        return point.start_timestamp_millis == points[0].start_timestamp_millis;
      });
  if (!same_start_time) {
    return errors::Internal(
        "Collected cumulative metrics should have the same start timestamp "
        "(the registration timestamp). This error implies a bug in the "
        "`tensorflow::monitoring::testing::CellReader` library.");
  }

  std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
    return a.end_timestamp_millis < b.end_timestamp_millis;
  });
  return points.back();
}

template <>
int64_t GetValue(const Point& point) {
  return point.int64_value;
}

template <>
std::string GetValue(const Point& point) {
  return point.string_value;
}

template <>
bool GetValue(const Point& point) {
  return point.bool_value;
}

template <>
Histogram GetValue(const Point& point) {
  return Histogram(point.histogram_value);
}

template <>
Percentiles GetValue(const Point& point) {
  return Percentiles(point.percentiles_value);
}

template <>
int64_t GetDelta(const int64_t& a, const int64_t& b) {
  return a - b;
}

template <>
Histogram GetDelta(const Histogram& a, const Histogram& b) {
  StatusOr<Histogram> result = a.Subtract(b);
  if (!result.ok()) {
    LOG(FATAL) << "Failed to compute the delta between histograms: "
               << result.status();
  }
  return *result;
}

template <>
Percentiles GetDelta(const Percentiles& a, const Percentiles& b) {
  return a.Subtract(b);
}

template <>
std::string GetDelta(const std::string& a, const std::string& b) {
  LOG(FATAL) << "`CellReader<std::string>` does not support `Delta`. "
             << "Please use `Read` instead.";
}

template <>
bool GetDelta(const bool& a, const bool& b) {
  LOG(FATAL) << "`CellReader<bool>` does not support `Delta`. "
             << "Please use `Read` instead.";
}

}  // namespace internal
}  // namespace testing
}  // namespace monitoring
}  // namespace tsl
