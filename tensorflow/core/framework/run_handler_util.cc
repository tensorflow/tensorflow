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

#include <cmath>

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {

double ParamFromEnvWithDefault(const std::string& var_name,
                               double default_value) {
  const char* val = std::getenv(var_name.c_str());
  double num;
  return (val && strings::safe_strtod(val, &num)) ? num : default_value;
}

std::vector<double> ParamFromEnvWithDefault(const std::string& var_name,
                                            std::vector<double> default_value) {
  const char* val = std::getenv(var_name.c_str());
  if (!val) {
    return default_value;
  }
  std::vector<string> splits = str_util::Split(val, ",");
  std::vector<double> result;
  result.reserve(splits.size());
  for (auto& split : splits) {
    double num;
    if (strings::safe_strtod(split, &num)) {
      result.push_back(num);
    } else {
      LOG(ERROR) << "Wrong format for " << var_name << ". Use default value.";
      return default_value;
    }
  }
  return result;
}

std::vector<int> ParamFromEnvWithDefault(const std::string& var_name,
                                         std::vector<int> default_value) {
  const char* val = std::getenv(var_name.c_str());
  if (!val) {
    return default_value;
  }
  std::vector<string> splits = str_util::Split(val, ",");
  std::vector<int> result;
  result.reserve(splits.size());
  for (auto& split : splits) {
    int num;
    if (strings::safe_strto32(split, &num)) {
      result.push_back(num);
    } else {
      LOG(ERROR) << "Wrong format for " << var_name << ". Use default value.";
      return default_value;
    }
  }
  return result;
}

bool ParamFromEnvBoolWithDefault(const std::string& var_name,
                                 bool default_value) {
  const char* val = std::getenv(var_name.c_str());
  return (val) ? str_util::Lowercase(val) == "true" : default_value;
}

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
    int demand = std::max(
        min_threads_per_request,
        static_cast<int>(std::ceil(weight * demand_factor - 0.00001f)));
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

void ComputeInterOpStealingRanges(int num_threads, int min_threads_per_domain,
                                  std::vector<std::uint_fast32_t>* start_vec,
                                  std::vector<std::uint_fast32_t>* end_vec) {
  int steal_domain_size = std::min(min_threads_per_domain, num_threads);
  unsigned steal_start = 0, steal_end = steal_domain_size;
  for (int i = 0; i < num_threads; ++i) {
    if (i >= steal_end) {
      if (steal_end + steal_domain_size < num_threads) {
        steal_start = steal_end;
        steal_end += steal_domain_size;
      } else {
        steal_end = num_threads;
        steal_start = steal_end - steal_domain_size;
      }
    }
    start_vec->at(i) = steal_start;
    end_vec->at(i) = steal_end;
  }
}

std::vector<int> ChooseRequestsWithExponentialDistribution(
    int num_active_requests, int num_threads) {
  // Fraction of the total threads that will be evenly distributed across
  // requests. The rest of threads will be exponentially distributed across
  // requests.
  static const double kCapacityFractionForEvenDistribution =
      ParamFromEnvWithDefault("TF_RUN_HANDLER_EXP_DIST_EVEN_FRACTION", 0.5);

  // For the threads that will be exponentially distributed across requests,
  // a request will get allocated (kPowerBase - 1) times as much threads as
  // threads allocated to all requests that arrive after it. For example, the
  // oldest request will be allocated num_threads*(kPowerBase-1)/kPowerBase
  // number of threads.
  static const double kPowerBase =
      ParamFromEnvWithDefault("TF_RUN_HANDLER_EXP_DIST_POWER_BASE", 2.0);

  std::vector<int> request_idx_list;
  request_idx_list.resize(num_threads);
  // Each request gets at least this number of threads that steal from it first.
  int min_threads_per_request =
      num_threads * kCapacityFractionForEvenDistribution / num_active_requests;
  min_threads_per_request =
      std::max(static_cast<int>(ParamFromEnvWithDefault(
                   "TF_RUN_HANDLER_EXP_DIST_MIN_EVEN_THREADS", 1)),
               min_threads_per_request);
  min_threads_per_request =
      std::min(static_cast<int>(ParamFromEnvWithDefault(
                   "TF_RUN_HANDLER_EXP_DIST_MAX_EVEN_THREADS", 3)),
               min_threads_per_request);

  int num_remaining_threads =
      std::max(0, num_threads - num_active_requests * min_threads_per_request);
  int request_idx = -1;
  int num_threads_next_request = 0;

  for (int tid = 0; tid < num_threads; ++tid) {
    if (num_threads_next_request <= 0) {
      request_idx = std::min(num_active_requests - 1, request_idx + 1);
      int num_extra_threads_next_request =
          std::ceil(num_remaining_threads * (kPowerBase - 1.0) / kPowerBase);
      num_remaining_threads -= num_extra_threads_next_request;
      num_threads_next_request =
          num_extra_threads_next_request + min_threads_per_request;
    }
    num_threads_next_request--;
    request_idx_list[tid] = request_idx;
  }
  return request_idx_list;
}

}  // namespace tensorflow
