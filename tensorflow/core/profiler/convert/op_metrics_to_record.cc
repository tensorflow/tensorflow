/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"

#include <iterator>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {

std::vector<const OpMetrics*> SortedOpMetricsDb(const OpMetricsDb& metrics_db,
                                                int max_records) {
  std::vector<const OpMetrics*> result;
  result.reserve(metrics_db.metrics_db_size());
  for (const OpMetrics& metrics : metrics_db.metrics_db()) {
    result.push_back(&metrics);
  }

  auto comp = [](const OpMetrics* a, const OpMetrics* b) {
    return std::make_tuple(a->self_time_ps(), b->name()) >
           std::make_tuple(b->self_time_ps(), a->name());
  };
  int result_size = result.size();
  if (max_records != -1 && result_size > max_records) {
    absl::c_partial_sort(result, result.begin() + max_records, comp);
    result.resize(max_records);
  } else {
    absl::c_sort(result, comp);
  }
  return result;
}

}  // namespace profiler
}  // namespace tensorflow
