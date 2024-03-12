/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/flags.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

std::vector<int> ConvertBinarySplitsToBucketSplits(int64 split,
                                                   int max_division_level) {
  std::vector<int> bucket_splits;
  uint32 current_index = 0;
  while (split > 0) {
    if (split % 2 == 1) {
      int split_level = absl::bit_width(current_index + 1) - 1;
      int split_offset = current_index - (1 << split_level) + 1;
      int split_size = 1 << (max_division_level - 1 - split_level);
      bucket_splits.push_back(split_size + split_offset * split_size * 2);
    }
    split >>= 1;
    current_index += 1;
  }
  absl::c_sort(bucket_splits);
  return bucket_splits;
}

int64 ConvertBucketSplitsToBinarySplits(std::vector<int> bucket_splits,
                                        int max_division_level) {
  int64 binary_splits = 0;
  for (auto& bucket_split : bucket_splits) {
    int split_level = max_division_level - 1;
    while (bucket_split > 0 && bucket_split % 2 == 0) {
      --split_level;
      bucket_split = bucket_split >> 1;
    }
    binary_splits |= (1LL << ((1 << split_level) - 1 + bucket_split / 2));
  }
  return binary_splits;
}

Status ValidateInputCombiner(const std::string& combiner) {
  if (combiner != "sum" && combiner != "mean" && combiner != "sqrtn") {
    return absl::InvalidArgumentError(
        "Invalid combiner: only \"sum\", \"mean\", and "
        "\"sqrtn\" are supported.");
  }
  return absl::OkStatus();
}

std::function<float(float)> GetCombinerScaleContributionFunction(
    absl::string_view combiner) {
  if (combiner == "sum") {
    return [](float x) -> float { return 1.f; };
  } else if (combiner == "mean") {
    return [](float x) -> float { return x; };
  } else {  // combiner == "sqrtn"
    return [](float x) -> float { return x * x; };
  }
}

std::function<float(float)> GetCombinerScaleTransformFunction(
    absl::string_view combiner) {
  if (combiner == "sum") {
    return [](float x) -> float { return 1; };
  } else if (combiner == "mean") {
    return [](float x) -> float { return x == 0.0f ? 0.0f : 1.0 / x; };
  } else {  // combiner == "sqrtn"
    return
        [](float x) -> float { return x == 0.0f ? 0.0f : 1.0 / std::sqrt(x); };
  }
}

Status GetMaxIdsAndUniquesExternal(const std::string& program_key,
                                   const std::string& table_name,
                                   int64_t num_samples_per_sparse_core,
                                   int64_t feature_width,
                                   int64_t* max_ids_per_partition,
                                   int64_t* max_unique_ids_per_partition) {
  SparseCore_GetMaxIdsAndUniques_Params params;
  params.program_key = program_key.c_str();
  params.table_name = table_name.c_str();
  params.num_samples_per_sparse_core = num_samples_per_sparse_core;
  params.feature_width = feature_width;
  StatusHelper status;
  params.status = status.c_status;

  stream_executor::tpu::OpsApiFn()->SparseCore_GetMaxIdsAndUniquesFn(&params);
  *max_ids_per_partition = params.max_ids_per_partition;
  *max_unique_ids_per_partition = params.max_unique_ids_per_partition;
  return status.status();
}

std::vector<std::vector<std::string>> GetTableStacks(
    const std::vector<int64_t>& table_height,
    const std::vector<int64_t>& table_width,
    const std::vector<int64_t>& table_num_samples,
    const std::vector<int64_t>& table_group,
    const std::vector<std::string>& table_names, int64_t num_tpu_chips) {
  if (GetDisableTableStacking()) {
    std::vector<std::vector<std::string>> stacks(table_names.size());
    for (int i = 0; i < table_names.size(); ++i) stacks[i] = {table_names[i]};
    return stacks;
  }

  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>>
      table_data(table_height.size());
  for (int i = 0; i < table_height.size(); ++i)
    table_data[i] =
        std::make_tuple(table_height[i], table_width[i], table_num_samples[i],
                        table_group[i], table_names[i]);

  // Sort tables by name so that we have a deterministic stacking.
  std::sort(table_data.begin(), table_data.end(), [](auto& lh, auto& rh) {
    return std::get<4>(lh) < std::get<4>(rh);
  });

  absl::flat_hash_map<int64_t, std::vector<std::vector<std::string>>>
      stacks_by_group;
  absl::flat_hash_map<int64_t, std::vector<int64_t>> stacks_height_by_group;
  absl::flat_hash_map<int64_t, std::vector<int64_t>> stacks_width_by_group;
  absl::flat_hash_map<int64_t, std::vector<int64_t>> stacks_samples_by_group;

  const int64_t mem_limit = GetXlaSparseCoreStackingMemLimit();
  const int64_t table_shard_limit = GetXlaSparseCoreStackingTableShardLimit();

  for (const auto& table : table_data) {
    int64_t height;
    int64_t width;
    int64_t num_samples;
    int64_t group;
    std::string name;
    std::tie(height, width, num_samples, group, name) = table;

    // Want per SparseCore samples.
    num_samples /= 4;

    // Find a stack to fit in. We need to stay under the limit on activation
    // sizes (if set) and the limit on table shard sizes (if set).
    int64_t stack_id = 0;
    for (; stack_id < stacks_by_group[group].size(); ++stack_id)
      if (((mem_limit == 0) ||
           (sizeof(float) * width *
                (num_samples + stacks_samples_by_group[group][stack_id]) <
            mem_limit)) &&
          ((table_shard_limit == 0) ||
           (sizeof(float) * (height + stacks_height_by_group[group][stack_id]) *
                width / num_tpu_chips <
            table_shard_limit)))
        break;

    // Create a new stack if we didn't find a stack to join.
    if (stack_id == stacks_by_group[group].size()) {
      stacks_by_group[group].resize(stacks_by_group[group].size() + 1);
      stacks_height_by_group[group].push_back(0);
      stacks_width_by_group[group].push_back(width);
      stacks_samples_by_group[group].push_back(0);
    }

    // Add the table to the stack and track the number of samples and height
    // of the table.
    stacks_by_group[group][stack_id].push_back(name);
    stacks_height_by_group[group][stack_id] += height;
    stacks_samples_by_group[group][stack_id] += num_samples;
  }

  // Merge all the stacks into one list.
  std::vector<std::vector<std::string>> table_stacks;
  for (const auto& [group, stacks] : stacks_by_group)
    table_stacks.insert(table_stacks.end(), stacks.begin(), stacks.end());

  return table_stacks;
}

ABSL_ATTRIBUTE_WEAK int GetMinibatchMaxDivisionLevel() {
  XlaSparseCoreFlags* sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_minibatch_max_division_level;
}

ABSL_ATTRIBUTE_WEAK bool GetDisableTableStacking() {
  XlaSparseCoreFlags* sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_disable_table_stacking;
}

ABSL_ATTRIBUTE_WEAK int64_t GetXlaSparseCoreStackingMemLimit() {
  XlaSparseCoreFlags* sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_stacking_mem_limit_bytes;
}

ABSL_ATTRIBUTE_WEAK int64_t GetXlaSparseCoreStackingTableShardLimit() {
  XlaSparseCoreFlags* sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_stacking_table_shard_limit_bytes;
}

}  // namespace tensorflow
