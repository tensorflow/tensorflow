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
#include "tensorflow/core/lib/monitoring/test_utils.h"

#include <cmath>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
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

StatusOr<Histogram> Histogram::Subtract(const Histogram& other) const {
  HistogramProto histogram_proto = histogram_proto_;
  if (other.histogram_proto_.bucket_limit().empty() &&
      other.histogram_proto_.bucket().empty()) {
    return Histogram(histogram_proto);
  }

  if (!absl::c_equal(histogram_proto.bucket_limit(),
                     other.histogram_proto_.bucket_limit())) {
    return errors::InvalidArgument(
        "Subtracting a histogram with different buckets. Left: [",
        absl::StrJoin(histogram_proto.bucket_limit(), ", "), "], right: [",
        absl::StrJoin(other.histogram_proto_.bucket_limit(), ", "), "].");
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
    return errors::InvalidArgument(
        "Failed to subtract a histogram by a larger histogram. Left operand: ",
        histogram_proto.ShortDebugString(),
        ", right operand: ", other.histogram_proto_.ShortDebugString());
  }
  return Histogram(histogram_proto);
}

size_t Percentiles::num() const { return percentiles_.total_samples; }

double Percentiles::sum() const {
  return std::isnan(percentiles_.accumulator) ? 0 : percentiles_.accumulator;
}

Percentiles Percentiles::Subtract(const Percentiles& other) const {
  tensorflow::monitoring::Percentiles delta;
  delta.unit_of_measure = percentiles_.unit_of_measure;
  delta.total_samples = num() - other.num();
  delta.accumulator = sum() - other.sum();
  return Percentiles(delta);
}

}  // namespace testing
}  // namespace monitoring
}  // namespace tensorflow
