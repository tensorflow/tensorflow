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
#ifndef XLA_TSL_LIB_MONITORING_TEST_UTILS_H_
#define XLA_TSL_LIB_MONITORING_TEST_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/tsl/lib/monitoring/types.h"
#include "tsl/platform/statusor.h"
#include "tsl/protobuf/histogram.pb.h"

namespace tsl {
namespace monitoring {
namespace testing {
using tensorflow::HistogramProto;
// Represents a `HistogramProto` but with a restricted API. This is used by
// a `CellReader` to return collected histograms in unit tests.
// Refer to core/framework/summary.proto for documentation of relevant fields.
class Histogram final {
 public:
  Histogram() = default;
  explicit Histogram(const HistogramProto& histogram_proto)
      : histogram_proto_(histogram_proto) {}

  // Returns the number of samples.
  double num() const;

  // Returns the number of samples in the `bucket`-th bucket.
  //
  // The range for a bucket is:
  //   bucket == 0:  -DBL_MAX .. bucket_limit(0)
  //   bucket != 0:  bucket_limit(bucket - 1) .. bucket_limit(bucket)
  double num(size_t bucket) const;

  // Returns the sum of the samples.
  double sum() const;

  // Returns the sum of squares of the samples.
  double sum_squares() const;

  // Subtracts the histogram by `other`. This is used by `CellReader` to compute
  // the delta of the metrics.
  //
  // REQUIRES:
  //   - The histograms have the same bucket boundaries.
  //   - This histogram has more or equal number of samples than `other` in
  //     every bucket.
  // Returns an InvalidArgument error if the requirements are violated.
  absl::StatusOr<Histogram> Subtract(const Histogram& other) const;

 private:
  HistogramProto histogram_proto_;
};

// Represents a collected `Percentiles` but with a restricted API. Subtracting
// two `Percentiles` does not produce a meaningful `Percentiles`, so we only
// expose a limited API that supports testing the number and sum of the samples.
class Percentiles final {
 public:
  Percentiles() = default;
  explicit Percentiles(const tsl::monitoring::Percentiles& percentiles)
      : percentiles_(percentiles) {}

  // Returns the number of samples.
  size_t num() const;

  // Returns the sum of samples.
  double sum() const;

  // Subtracts the percentiles by `other`. This is used by `CellReader` to
  // compute the delta of the metrics.
  Percentiles Subtract(const Percentiles& other) const;

 private:
  tsl::monitoring::Percentiles percentiles_;
};

}  // namespace testing
}  // namespace monitoring
}  // namespace tsl

#endif  // XLA_TSL_LIB_MONITORING_TEST_UTILS_H_
