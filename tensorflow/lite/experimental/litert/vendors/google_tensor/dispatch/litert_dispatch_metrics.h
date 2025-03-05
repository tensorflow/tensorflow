// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_METRICS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_METRICS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

class LiteRtDispatchMetricsT {
 public:
  // Construct a LiteRtDispatchMetricsT object using C-style arrays and strings.
  // `metric_names` is an array of C-style strings representing metric names.
  // `metric_values` is an array of int64_t values representing metric values.
  // Both `metric_names` and `metric_values` have `num_metrics` elements.
  //
  // NOTE: The values in the arrays are copied into the LiteRtDispatchMetricsT.
  LiteRtDispatchMetricsT(int num_metrics, const char** metric_names,
                         const int64_t* metric_values)
      : metric_names_(metric_names, metric_names + num_metrics),
        metric_values_(metric_values, metric_values + num_metrics) {}
  int GetNumMetrics() const { return metric_names_.size(); }
  LiteRtMetric GetMetric(int metric_index) const {
    if (metric_index < 0 || metric_index >= GetNumMetrics()) {
      return LiteRtMetric{.name = "invalid_metric",
                          .value = LiteRtAny{.type = kLiteRtAnyTypeNone}};
    }
    return LiteRtMetric{
        .name = metric_names_[metric_index].c_str(),
        .value =
            LiteRtAny{
                .type = kLiteRtAnyTypeInt,
                .int_value = metric_values_[metric_index],
            },
    };
  }

 private:
  const std::vector<std::string> metric_names_;
  const std::vector<int64_t> metric_values_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_METRICS_H_
