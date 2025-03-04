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
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

class LiteRtDispatchMetricsT {
 public:
  LiteRtDispatchMetricsT(int num_metrics, const char** metric_names,
                         const int64_t* metric_values)
      : metrics_(CreateMetrics(num_metrics, metric_names, metric_values)) {}
  int GetNumMetrics() const { return metrics_.size(); }
  LiteRtMetric GetMetric(int metric_index) const {
    return metrics_[metric_index];
  }

 private:
  static std::vector<LiteRtMetric> CreateMetrics(int num_metrics,
                                                 const char** metric_names,
                                                 const int64_t* metric_values) {
    std::vector<LiteRtMetric> metrics;
    metrics.reserve(num_metrics);
    for (int i = 0; i < num_metrics; ++i) {
      metrics.push_back(LiteRtMetric{
          .name = metric_names[i],
          .value =
              LiteRtAny{
                  .type = kLiteRtAnyTypeInt,
                  .int_value = metric_values[i],
              },
      });
    }
    return metrics;
  }
  const std::vector<LiteRtMetric> metrics_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_METRICS_H_
