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
#ifndef TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_H_
#define TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/lib/monitoring/cell_reader-inl.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"

namespace tensorflow {
namespace monitoring {
namespace testing {

// `CellReader` is a testing class which allows a user to read the current value
// of a tfstreamz cell.
//
// For tfstreamz metrics like the following:
//
// ```
// auto* test_counter = monitoring::Counter<1>::New(
//    "/tensorflow/monitoring/test/counter", "label",
//    "Test tfstreamz counter.");
// auto* test_sampler = monitoring::Sampler<2>::New(
//    "/tensorflow/monitoring/test/sampler", "label1", "label2",
//    "Test tfstreamz sampler.");
// auto* test_string_gauge = monitoring::Gauge<2>::New(
//    "/tensorflow/monitoring/test/gauge", "label1", "label2",
//    "Test tfstreamz gauge.");
// auto* test_percentiles = monitoring::PercentileSampler<2>::New(
//    {"/tensorflow/monitoring/test/percentiles", "Test percentiles.",
//     "label1", "label2"},
//     /*percentiles=*/{25.0, 50.0, 80.0, 90.0, 95.0, 99.0},
//     /*max_samples=*/1024,
//     monitoring::UnitOfMeasure::kNumber);
// ```
//
// one could read the exported tfstreamz values using a `CellReader` like this:
//
// ```
// using tensorflow::monitoring::testing::Histogram;
// using tensorflow::monitoring::testing::Percentiles;
//
// CellReader<int64_t> counter_reader("/tensorflow/monitoring/test/counter");
// CellReader<Histogram> sampler_reader("/tensorflow/monitoring/test/sampler");
// CellReader<std::string> gauge_reader("/tensorflow/monitoring/test/gauge");
// CellReader<Percentiles> percentiles_reader(
//     "/tensorflow/monitoring/test/percentiles");
// EXPECT_EQ(counter_reader.Delta("label_value"), 0);
// EXPECT_FLOAT_EQ(sampler_reader.Delta("x", "y").num(), 0.0);
// EXPECT_EQ(gauge_reader.Delta("x", "y"), "");
// EXPECT_EQ(percentiles_reader.Delta("x", "y").num(), 0);
//
// CodeThatUpdateMetrics();
// EXPECT_EQ(counter_reader.Delta("label_value"), 5);
// Histogram histogram = sampler_reader.Delta("x", "y");
// EXPECT_FLOAT_EQ(histogram.num(), 5.0);
// EXPECT_GT(histogram.sum(), 0.0);
// EXPECT_EQ(gauge_reader.Delta("x", "y"), "gauge value");
// EXPECT_EQ(percentiles_reader.Delta("x", "y").num(), 5);
// ```
template <typename ValueType>
class CellReader {
 public:
  // Constructs a `CellReader` that reads values exported for `metric_name`.
  //
  // REQUIRES: a tfstreamz with `metric_name` exists. Otherwise, the
  // `CellReader` will construct without issue, but the `Read` and `Delta` calls
  // will CHECK-fail.
  explicit CellReader(const std::string& metric_name);
  virtual ~CellReader() = default;
  CellReader(const CellReader&) = delete;
  CellReader& operator=(const CellReader&) = delete;

  // Returns the current value of the cell with the given `labels`. A metric can
  // have zero or more labels, depending on its definition. If the metric has
  // not been modified, it returns a default value appropriate for `ValueType`.
  //
  // REQUIRES: The tfstreamz exists, and `labels` contains a correct number of
  // labels per tfstreamz definition. Otherwise, it will CHECK-fail.
  template <typename... LabelType>
  ValueType Read(const LabelType&... labels);

  // Returns the difference in the value of this cell since the last time
  // `Delta()` was called for this cell, or when the `CellReader` was created,
  // whichever was most recent. If the metric has not been modified, it returns
  // a default value appropriate for `ValueType`. `Delta` is not supported for
  // string and bool gauges.
  //
  // REQUIRES: The tfstreamz exists, `labels` contains a correct number of
  // labels per tfstreamz definition, and the ValueType is not string or bool.
  // Otherwise, it will CHECK-fail.
  template <typename... LabelType>
  ValueType Delta(const LabelType&... labels);

 private:
  const std::string metric_name_;

  // Metrics collected at the time of construction. It is needed because data
  // may have been collected when this object is constructed. The initial values
  // need to be subtracted from the result of the `Read()` call to compute the
  // correct values.
  std::unique_ptr<CollectedMetrics> initial_metrics_;

  // Records the value of the cells since the last time `Delta()` was called.
  // This is used to compute the next delta value.
  absl::flat_hash_map<std::vector<std::string>, ValueType> delta_map_;
};

template <typename ValueType>
CellReader<ValueType>::CellReader(const std::string& metric_name)
    : metric_name_(metric_name), initial_metrics_(internal::CollectMetrics()) {}

template <typename ValueType>
template <typename... LabelType>
ValueType CellReader<ValueType>::Read(const LabelType&... labels) {
  std::vector<std::string> labels_list{labels...};
  std::unique_ptr<CollectedMetrics> metrics = internal::CollectMetrics();
  ValueType value = internal::GetLatestValueOrDefault<ValueType>(
      *metrics, metric_name_, labels_list);
  if (internal::GetMetricKind(*metrics, metric_name_) == MetricKind::kGauge) {
    return value;
  }
  ValueType initial_value = internal::GetLatestValueOrDefault<ValueType>(
      *initial_metrics_, metric_name_, labels_list);
  return internal::GetDelta<ValueType>(value, initial_value);
}

template <typename ValueType>
template <typename... LabelType>
ValueType CellReader<ValueType>::Delta(const LabelType&... labels) {
  std::vector<std::string> labels_list{labels...};
  std::unique_ptr<CollectedMetrics> metrics = internal::CollectMetrics();
  ValueType value = internal::GetLatestValueOrDefault<ValueType>(
      *metrics, metric_name_, labels_list);
  ValueType initial_value = internal::GetLatestValueOrDefault<ValueType>(
      *initial_metrics_, metric_name_, labels_list);
  if (delta_map_.contains(labels_list)) {
    initial_value = delta_map_[labels_list];
  }
  delta_map_[labels_list] = value;
  return internal::GetDelta<ValueType>(value, initial_value);
}

}  // namespace testing
}  // namespace monitoring
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_H_
