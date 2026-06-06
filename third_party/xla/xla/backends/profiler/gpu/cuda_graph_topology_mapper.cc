/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cuda_graph_topology_mapper.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace xla {
namespace profiler {

size_t CudaGraphTopologyMapper::CalculateMergedSize(
    uint32_t graph_id, const absl::flat_hash_map<uint32_t, size_t>& base_sizes,
    const absl::flat_hash_map<uint32_t, std::vector<ChildGraphEntry>>&
        child_graphs,
    absl::flat_hash_map<uint32_t, size_t>* merged_sizes) {
  auto merged_it = merged_sizes->find(graph_id);
  if (merged_it != merged_sizes->end()) {
    return merged_it->second;
  }

  size_t size = 0;
  auto size_it = base_sizes.find(graph_id);
  if (size_it != base_sizes.end()) {
    size = size_it->second;
  }

  auto children_it = child_graphs.find(graph_id);
  if (children_it != child_graphs.end()) {
    for (const auto& child : children_it->second) {
      size_t child_size = CalculateMergedSize(child.child_graph_id, base_sizes,
                                              child_graphs, merged_sizes);
      size += child_size;
      if (!child.is_conditional) {
        size -= 1;  // Child graph node itself is replaced for inline children.
      }
    }
  }

  (*merged_sizes)[graph_id] = size;
  return size;
}

std::pair<uint32_t, uint32_t> CudaGraphTopologyMapper::ResolveMergedNode(
    uint32_t graph_id, uint32_t node_index,
    const absl::flat_hash_map<uint32_t, size_t>& base_sizes,
    const absl::flat_hash_map<uint32_t, std::vector<ChildGraphEntry>>&
        child_graphs,
    absl::flat_hash_map<uint32_t, size_t>* merged_sizes) {
  auto children_it = child_graphs.find(graph_id);
  if (children_it == child_graphs.end() || children_it->second.empty()) {
    return {graph_id, node_index};
  }

  // Calculate total inline size.
  size_t total_inline_size = 0;
  auto size_it = base_sizes.find(graph_id);
  if (size_it != base_sizes.end()) {
    total_inline_size = size_it->second;
  }
  for (const auto& child : children_it->second) {
    if (!child.is_conditional) {
      size_t child_size = CalculateMergedSize(child.child_graph_id, base_sizes,
                                              child_graphs, merged_sizes);
      total_inline_size += (child_size - 1);
    }
  }

  if (node_index < total_inline_size) {
    // 1. Resolve inline nodes
    uint32_t inline_offset = 0;
    for (const auto& child : children_it->second) {
      if (child.is_conditional) {
        continue;
      }
      size_t child_size = CalculateMergedSize(child.child_graph_id, base_sizes,
                                              child_graphs, merged_sizes);
      uint32_t child_start = child.insertion_point + inline_offset;
      uint32_t child_end = child_start + child_size - 1;

      if (node_index >= child_start && node_index <= child_end) {
        uint32_t local_index = node_index - child_start;
        return ResolveMergedNode(child.child_graph_id, local_index, base_sizes,
                                 child_graphs, merged_sizes);
      }

      if (node_index < child_start) {
        return {graph_id, node_index - inline_offset};
      }

      inline_offset += (child_size - 1);
    }
    return {graph_id, node_index - inline_offset};
  }

  // 2. Resolve conditional nodes
  uint32_t conditional_offset = 0;
  for (const auto& child : children_it->second) {
    if (!child.is_conditional) {
      continue;
    }
    size_t child_size = CalculateMergedSize(child.child_graph_id, base_sizes,
                                            child_graphs, merged_sizes);
    uint32_t child_start = total_inline_size + conditional_offset;
    uint32_t child_end = child_start + child_size - 1;

    if (node_index >= child_start && node_index <= child_end) {
      uint32_t local_index = node_index - child_start;
      return ResolveMergedNode(child.child_graph_id, local_index, base_sizes,
                               child_graphs, merged_sizes);
    }

    conditional_offset += child_size;
  }

  return {graph_id, node_index};
}

}  // namespace profiler
}  // namespace xla
