/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/mlrt/kernel/shard_restore_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"

namespace tensorflow {
namespace tf_mlrt {

// Shard variables into cluster of roughly the same size.
// The returned result is a vector of clusters and each cluster is represented
// as a vector of variable indices.
std::vector<std::vector<int>> ShardVariables(
    int num_shards, absl::Span<int64_t> variable_sizes) {
  DCHECK_GT(num_shards, 0);
  // Shard the variables into clusters of roughly the same size. This is done
  // via a greedy algorithm: add the next largest sized variable tensor to the
  // smallest cluster.

  // First, create a list of variables sorted by size in descending order.
  struct IndexSize {
    int index;
    int64_t size;
  };
  std::vector<IndexSize> variable_index_sizes;
  variable_index_sizes.reserve(variable_sizes.size());
  for (int i = 0; i < variable_sizes.size(); ++i) {
    variable_index_sizes.push_back({.index = i, .size = variable_sizes[i]});
  }
  std::sort(
      variable_index_sizes.begin(), variable_index_sizes.end(),
      [&](const IndexSize& a, const IndexSize& b) { return a.size > b.size; });

  // Then shards the variables into clusters of roughly the same size.
  // Each cluster is also roughly equally distributed in size b/c the variables
  // are sorted by size in the previous step.
  struct RestoreVariableCluster {
    std::vector<int> indices;
    size_t total_size = 0;
  };

  auto cmp = [](const RestoreVariableCluster& a,
                const RestoreVariableCluster& b) {
    return a.total_size > b.total_size;
  };
  std::priority_queue<RestoreVariableCluster,
                      std::vector<RestoreVariableCluster>, decltype(cmp)>
      min_heap;
  for (int i = 0; i < num_shards; ++i) {
    min_heap.push(RestoreVariableCluster());
  }
  for (int i = 0; i < variable_index_sizes.size(); ++i) {
    RestoreVariableCluster min_cluster = min_heap.top();
    min_heap.pop();
    min_cluster.total_size += variable_index_sizes[i].size;
    min_cluster.indices.push_back(variable_index_sizes[i].index);
    min_heap.push(std::move(min_cluster));
  }

  std::vector<std::vector<int>> shards;
  shards.reserve(min_heap.size());
  while (!min_heap.empty()) {
    auto& min_cluster = min_heap.top();
    if (min_cluster.total_size > 0) {
      shards.push_back(min_cluster.indices);
    }
    min_heap.pop();
  }
  return shards;
}

}  // namespace  tf_mlrt
}  // namespace tensorflow
