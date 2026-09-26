/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/batch_cost_utils.h"

#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/lib/monitoring/sampler.h"

namespace tensorflow {
namespace serving {

void RecordBatchCosts(const std::string& model_name,
                      const int64_t processed_size,
                      const absl::string_view cost_type,
                      const absl::Duration total_cost) {
  static auto* cell = tensorflow::monitoring::Sampler<3>::New(
      {"/tensorflow/serving/batching/costs",
       "Tracks the batch costs (in microseconds) by model name and processed "
       "size.",
       "model_name", "processed_size", "cost_type"},
      // It's 27 buckets with the last bucket being 2^26 to DBL_MAX;
      // so the limits are [1, 2, 4, 8, ..., 64 * 1024 * 1024 (~64s), DBL_MAX].
      monitoring::Buckets::Exponential(1, 2, 27));
  cell->GetCell(model_name, std::to_string(processed_size),
                std::string(cost_type))
      ->Add(absl::ToDoubleMicroseconds(total_cost));
}

}  // namespace serving
}  // namespace tensorflow
