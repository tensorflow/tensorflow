/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/run_handler_util.h"

#include <algorithm>
#include <cmath>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

void ComputeInterOpSchedulingRanges(int num_active_requests, int num_threads,
                                    int min_threads_per_request,
                                    std::vector<std::uint_fast32_t>* start_vec,
                                    std::vector<std::uint_fast32_t>* end_vec) {
  // Each request is expected to have weight W[i] = num_active_requests - i.
  // Therefore, total_weight = sum of all request weights.
  float total_weight = 0.5f * num_active_requests * (num_active_requests + 1);
  float demand_factor = static_cast<float>(num_threads) / total_weight;
  float last_cumulative_weight = 0.0;
  min_threads_per_request = std::max(1, min_threads_per_request);
  for (int i = 0; i != num_active_requests; i++) {
    float cumulative_weight =
        static_cast<float>(i + 1) *
        (num_active_requests - static_cast<float>(i) * 0.5f);
    float weight = cumulative_weight - last_cumulative_weight;
    // Quantize thread_demand by rounding up, and also satisfying
    // `min_threads_per_request` constraint.
    // Note: We subtract a small epsilon (0.00001) to prevent ceil(..) from
    // rounding weights like 4.0 to 5.
    int demand =
        std::max(min_threads_per_request,
                 static_cast<int>(ceil(weight * demand_factor - 0.00001f)));
    // For the quantized range [start, end); compute the floor of real start,
    // and expand downwards from there with length `demand` and adjust for
    // boundary conditions.
    int start = last_cumulative_weight * demand_factor;
    int end = std::min(num_threads, start + demand);
    start = std::max(0, std::min(start, end - demand));
    start_vec->at(i) = start;
    end_vec->at(i) = end;
    last_cumulative_weight = cumulative_weight;
  }
}
}  // namespace tensorflow
