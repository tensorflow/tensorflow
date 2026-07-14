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
#include "xla/tsl/lib/monitoring/test_utils.h"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ostream>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/metric_def.h"
#include "xla/tsl/lib/monitoring/types.h"
#include "xla/tsl/protobuf/histogram.pb.h"

namespace tsl {
namespace monitoring {
namespace testing {

double Histogram::num() const { return histogram_proto_.num(); }

double Histogram::num(size_t bucket) const {
  if (bucket >= histogram_proto_.bucket().size()) {
    return 0;
  }
  return histogram_proto_.bucket(bucket);
}

double Histogram::sum() const { return histogram_proto_.sum(); }
double Histogram::sum_squares() const { return histogram_proto_.sum_squares(); }

absl::StatusOr<Histogram> Histogram::Subtract(const Histogram& other) const {
  HistogramProto histogram_proto = histogram_proto_;
  if (other.histogram_proto_.bucket_limit().empty() &&
      other.histogram_proto_.bucket().empty()) {
    return Histogram(histogram_proto);
  }

  if (!absl::c_equal(histogram_proto.bucket_limit(),
                     other.histogram_proto_.bucket_limit())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Subtracting a histogram with different buckets. Left: [",
        absl::StrJoin(histogram_proto.bucket_limit(), ", "), "], right: [",
        absl::StrJoin(other.histogram_proto_.bucket_limit(), ", "), "]."));
  }

  histogram_proto.set_num(histogram_proto.num() - other.histogram_proto_.num());
  histogram_proto.set_sum(histogram_proto.sum() - other.histogram_proto_.sum());
  histogram_proto.set_sum_squares(histogram_proto.sum_squares() -
                                  other.histogram_proto_.sum_squares());
  for (size_t i = 0; i < histogram_proto.bucket().size(); ++i) {
    histogram_proto.set_bucket(
        i, histogram_proto.bucket(i) - other.histogram_proto_.bucket(i));
  }

  const bool histogram_is_valid =
      histogram_proto.num() >= 0 &&
      absl::c_all_of(histogram_proto.bucket(),
                     [](const double num) { return num >= 0; });
  if (!histogram_is_valid) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to subtract a histogram by a larger histogram. Left operand: ",
        histogram_proto.ShortDebugString(),
        ", right operand: ", other.histogram_proto_.ShortDebugString()));
  }
  return Histogram(histogram_proto);
}

size_t Percentiles::num() const { return percentiles_.total_samples; }

double Percentiles::sum() const {
  return std::isnan(percentiles_.accumulator) ? 0 : percentiles_.accumulator;
}

Percentiles Percentiles::Subtract(const Percentiles& other) const {
  tsl::monitoring::Percentiles delta;
  delta.unit_of_measure = percentiles_.unit_of_measure;
  delta.total_samples = num() - other.num();
  delta.accumulator = sum() - other.sum();
  return Percentiles(delta);
}

}  // namespace testing

namespace {
absl::string_view Indent(int width) {
  static constexpr int kMaxIndentWidth = 32;
  static constexpr const char kIndentBuffer[kMaxIndentWidth + 1] =
      "                                ";
  static constexpr absl::string_view kMaxIndent(kIndentBuffer);

  return kMaxIndent.substr(0, width);
}
}  // namespace

std::ostream& operator<<(std::ostream& os, ValueType value_type) {
  switch (value_type) {
    case ValueType::kInt64:
      return os << "ValueType::kInt64";
    case ValueType::kHistogram:
      return os << "ValueType::kHistogram";
    case ValueType::kString:
      return os << "ValueType::kString";
    case ValueType::kBool:
      return os << "ValueType::kBool";
    case ValueType::kPercentiles:
      return os << "ValueType::kPercentiles";
    case ValueType::kDouble:
      return os << "ValueType::kDouble";
  }
  return os << "(Invalid ValueType)";
}

std::ostream& operator<<(std::ostream& os, UnitOfMeasure unit_of_measure) {
  switch (unit_of_measure) {
    case UnitOfMeasure::kNumber:
      return os << "UnitOfMeasure::kNumber";
    case UnitOfMeasure::kTime:
      return os << "UnitOfMeasure::kTime";
    case UnitOfMeasure::kBytes:
      return os << "UnitOfMeasure::kBytes";
  }
  return os << "(Invalid UnitOfMeasure)";
}

std::ostream& operator<<(std::ostream& os, const PercentilePoint& point) {
  os << "{ percentile: " << point.percentile  //
     << ", value: " << point.value << " }";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Percentiles& percentiles) {
  constexpr size_t kMaxPointsToPrint = 3;

  int base_indentation = os.width();
  absl::string_view indent_0 = Indent(base_indentation);
  absl::string_view indent_1 = Indent(base_indentation + 2);
  absl::string_view indent_2 = Indent(base_indentation + 4);
  os << std::setw(0);

  os << indent_0 << "{\n"
     << indent_1 << "unit_of_measure: " << percentiles.unit_of_measure << ",\n"
     << indent_1 << "start_nstime: " << percentiles.start_nstime << ",\n"
     << indent_1 << "end_nstime: " << percentiles.end_nstime << ",\n"
     << indent_1 << "min_value: " << percentiles.min_value << ",\n"
     << indent_1 << "max_value: " << percentiles.max_value << ",\n"
     << indent_1 << "mean: " << percentiles.mean << ",\n"
     << indent_1 << "stddev: " << percentiles.stddev << ",\n"
     << indent_1 << "num_samples: " << percentiles.num_samples << ",\n"
     << indent_1 << "total_samples: " << percentiles.total_samples << ",\n"
     << indent_1 << "accumulator: " << percentiles.accumulator << ",\n"
     << indent_1 << "points: {";
  if (!percentiles.points.empty()) {
    os << "\n";
    int points_printed = 0;
    for (const auto& point : percentiles.points) {
      if (points_printed >= kMaxPointsToPrint) {
        os << indent_2 << "...\n";
        break;
      }
      os << indent_2 << point << ",\n";
      ++points_printed;
    }
    os << indent_1;
  }
  os << "},\n"  //
     << indent_0 << " }";

  return os;
}

std::ostream& operator<<(std::ostream& os, const Point& point) {
  int base_indent_width = os.width();
  absl::string_view indent_0 = Indent(base_indent_width);
  absl::string_view indent_1 = Indent(base_indent_width + 2);
  absl::string_view indent_2 = Indent(base_indent_width + 4);
  os << std::setw(0);

  os << indent_0 << "{\n"  //
     << indent_1 << "labels: {";
  if (!point.labels.empty()) {
    os << "\n";
    for (const auto& label : point.labels) {
      os << indent_2 << label << ",\n";
    }
    os << indent_1;
  }
  os << "},\n"  //
     << indent_1 << "value_type: " << point.value_type << ",\n";
  switch (point.value_type) {
    case ValueType::kInt64:
      os << indent_1 << "int64_value: " << point.int64_value << ",\n";
      break;
    case ValueType::kHistogram:
      os << indent_1 << "histogram_value: { ... },\n";
      break;
    case ValueType::kString:
      os << indent_1 << "string_value: " << point.string_value << ",\n";
      break;
    case ValueType::kBool:
      os << indent_1 << "bool_value: " << point.bool_value << ",\n";
      break;
    case ValueType::kPercentiles:
      os << indent_1 << "percentiles_value: " << point.percentiles_value
         << ",\n";
      break;
    case ValueType::kDouble:
      os << indent_1 << "double_value: " << point.double_value << ",\n";
      break;
  }
  os << indent_1 << "start_timestamp_millis: " << point.start_timestamp_millis
     << ",\n"
     << indent_1 << "end_timestamp_millis: " << point.end_timestamp_millis
     << ",\n"
     << indent_0 << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Point::Label& label) {
  os << "{ name: \"" << label.name << "\", value: \"" << label.value << "\" }";
  return os;
}

}  // namespace monitoring
}  // namespace tsl
