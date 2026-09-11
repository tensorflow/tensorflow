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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUDA_GRAPH_TOPOLOGY_MAPPER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUDA_GRAPH_TOPOLOGY_MAPPER_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace xla {
namespace profiler {

// CudaGraphTopologyMapper is a pure, state-less C++ helper library designed to
// mirror the layout and node-flattening sequences that the NVIDIA CUDA driver
// uses to assign executable flat LocalNodeId sequences to CUPTI activity traces
// during the instantiation of hierarchical and conditional nested CUDA Graphs.
//
// Topology Examples:
//
// 1. Standard Inline Child Graph (`is_conditional = false`)
// In this scenario, the nodes of the nested child graph completely replace the
// placeholder node in the parent graph.
// Layout Example:
//   Parent Graph (Size 3): [P0, P_Child, P2]
//   Child Graph (Size 2): [C0, C1]
//   Expected Flattened Execution sequence: P0 -> C0 -> C1 -> P2
//   Expected Flat indices: P0 (0), C0 (1), C1 (2), P2 (3)
//
// 2. Conditional Nested Graph (`is_conditional = true`)
// In this scenario, the child graph's nodes are executed as a separate entity
// at runtime, appended to the end of the parent graph's sequence. The
// placeholder node in the parent retains its original index in the flattened
// space.
// Layout Example:
//   Parent Graph (Size 3): [P0, P1_Cond, P2]
//   Child Graph (Size 2): [C0, C1]
//   Expected Flattened Sequence (Kernels only): P0 -> C0 -> C1 -> P2
//   Expected Flat indices: P0 (0), P1_Cond (1), P2 (2), C0 (3), C1 (4)
//   (Note that the CUPTI trace for `C0` reports local node index `3`, while the
//    original placeholder `P2` reports index `2`).
//
// All APIs in this library are pure, static, and have zero CUDA SDK
// dependencies, allowing them to run identically on the host CPU for unit
// testing.
class CudaGraphTopologyMapper {
 public:
  struct ChildGraphEntry {
    uint32_t child_graph_id;
    uint32_t insertion_point;
    bool is_conditional;

    bool operator<(const ChildGraphEntry& other) const {
      return insertion_point < other.insertion_point;
    }
  };

  // Recursively calculates the total merged size of a graph, caching results in
  // the provided mutable merged_sizes map.
  static size_t CalculateMergedSize(
      uint32_t graph_id,
      const absl::flat_hash_map<uint32_t, size_t>& base_sizes,
      const absl::flat_hash_map<uint32_t, std::vector<ChildGraphEntry>>&
          child_graphs,
      absl::flat_hash_map<uint32_t, size_t>* merged_sizes);

  // Resolves a flat, merged node index into its target (template_graph_id,
  // local_node_index) pair.
  static std::pair<uint32_t, uint32_t> ResolveMergedNode(
      uint32_t graph_id, uint32_t node_index,
      const absl::flat_hash_map<uint32_t, size_t>& base_sizes,
      const absl::flat_hash_map<uint32_t, std::vector<ChildGraphEntry>>&
          child_graphs,
      absl::flat_hash_map<uint32_t, size_t>* merged_sizes);

 private:
  CudaGraphTopologyMapper() = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUDA_GRAPH_TOPOLOGY_MAPPER_H_
