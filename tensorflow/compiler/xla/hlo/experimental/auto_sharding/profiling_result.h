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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_PROFILING_RESULT_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_PROFILING_RESULT_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"

namespace xla {
namespace spmd {

// Store the profiling results of communication and computation.
class ProfilingResult {
 public:
  // TODO (zhuohan): loading the profiling result.
  ProfilingResult() {
    if (all_reduce_cost_dict_.empty()) {
      enabled_ = false;
    } else {
      enabled_ = true;
    }
  }

  bool Enabled() const { return enabled_; }

  double EstimateAllGatherCost(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    if (all_gather_cost_dict_.empty()) {
      // Use all-reduce to approximate all-gather.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype,
                            all_gather_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_gather_cost_dict_);
  }

  double EstimateAllReduceCost(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_reduce_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_reduce_cost_dict_);
  }

  double EstimateReduceScatterCost(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    if (reduce_scatter_cost_dict_.empty()) {
      // Use all-reduce to approximate reduce-scatter.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype,
                            reduce_scatter_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype,
                            reduce_scatter_cost_dict_);
  }

  double EstimateAllToAllCost(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    // A penalty factor to make the theoretical cost match the
    // empirical cost on v100 + nvlink.
    int64_t num_devices = replica_groups.front().size();
    double penalty_factor = static_cast<double>(num_devices) / 2.0;
    // Use all-gather to approximate all-to-all.
    return EstimateAllGatherCost(replica_groups, size / num_devices, dtype) *
           penalty_factor;
  }

  std::string ToString() {
    std::string str;
    for (const auto& item : all_reduce_cost_dict_) {
      absl::StrAppend(&str, item.first.first, " ", item.first.second, "\n");
    }
    return str;
  }

 private:
  // pair<group, dtype>
  using Key = std::pair<std::string, std::string>;
  // vector<pair<size, time>>
  using Value = std::vector<std::pair<int64_t, double>>;

  // Estimate the cost by linear interpolation between the two closest points.
  double EstimateInternal(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype,
      const StableHashMap<Key, Value>& cost_dict) const {
    Key key(Group2Str(replica_groups), dtype);
    Value cost_list = cost_dict.at(key);

    CHECK(!cost_list.empty());

    size_t i;
    if (size > cost_list.back().first) {
      i = cost_list.size() - 2;
    } else if (size < cost_list.front().first) {
      i = 0;
    } else {
      for (i = 0; i < cost_list.size() - 1; ++i) {
        if (cost_list[i].first <= size && size <= cost_list[i + 1].first) {
          break;
        }
      }
    }

    int64_t left_size = cost_list[i].first;
    double left_cost = cost_list[i].second;
    int64_t right_size = cost_list[i + 1].first;
    double right_cost = cost_list[i + 1].second;

    return 1.0 * (size - left_size) / (right_size - left_size) *
               (right_cost - left_cost) +
           left_cost;
  }

  // Make a string key of a replica_groups.
  std::string Group2Str(
      const std::vector<std::vector<int64_t>>& replica_groups) const {
    std::string str("(");
    for (const auto& group : replica_groups) {
      absl::StrAppend(&str, "(", absl::StrJoin(group, ","), ")");
    }
    absl::StrAppend(&str, ")");

    return str;
  }

  bool enabled_;
  StableHashMap<Key, Value> all_reduce_cost_dict_;
  StableHashMap<Key, Value> all_gather_cost_dict_;
  StableHashMap<Key, Value> reduce_scatter_cost_dict_;
};

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_PROFILING_RESULT_H_
