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

#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry_pre_13_4.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry.h"
#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"

// Selected by BUILD only with CUDA 13.2+ headers; see
// cuda_command_buffer_telemetry_pre_13_4_stub.cc for older ones.
namespace stream_executor::gpu::command_buffer_telemetry {
namespace {

using Registry = CudaGraphAnnotationRegistry;
using KeySpace = Registry::KeySpace;

void ProjectGraphHierarchyDfs(CUgraph graph, uint32_t exec_id,
                              uint32_t* next_flat_local_id) {
  // Every node of `graph` occupies an ordinal whether or not it can be
  // registered. If `cuGraphGetId` fails, the nodes (and their subgraphs) are
  // still counted, just not registered, so that later ordinals don't shift
  // onto the wrong nodes. An empty graph contributes no ordinals.
  //
  // Not unit-tested for the failure case: `cuGraphGetId` can't fail on a valid
  // graph, and stubbing the driver API is out of proportion for this branch.
  const std::optional<uint32_t> src_graph_id = GetGraphId(graph);
  const std::vector<CUgraphNode> nodes = GetGraphNodes(graph);

  // First pass: CUDA assigns contiguous 0-based flat local IDs
  // [base_offset, base_offset + num_nodes) to all nodes directly in `graph`
  // before recursing into its child and conditional subgraphs.
  Registry& registry = Registry::Default();
  for (CUgraphNode node : nodes) {
    const uint32_t flat_local_id = (*next_flat_local_id)++;
    if (!src_graph_id.has_value()) {
      continue;
    }
    const uint64_t exec_node_id = (uint64_t{exec_id} << 32) | flat_local_id;
    if (std::optional<uint64_t> src_tools_id = GetToolsId(node)) {
      registry.CopyNodeAnnotation(exec_id, exec_node_id, *src_graph_id,
                                  *src_tools_id, /*fallback=*/"",
                                  KeySpace::kPre134ExecGraph);
    }
  }

  // Second pass: recurse in depth-first pre-order into embedded child graphs
  // and conditional body graphs.
  for (CUgraphNode node : nodes) {
    ForEachSubgraphPair(
        node, nullptr, [&](CUgraph subgraph, CUgraph /*orig_subgraph*/) {
          ProjectGraphHierarchyDfs(subgraph, exec_id, next_flat_local_id);
        });
  }
}

}  // namespace

std::optional<uint32_t> GetGraphExecId(CUgraphExec exec) {
  if (exec == nullptr) {
    return std::nullopt;
  }
  unsigned int exec_id = 0;
  if (cuGraphExecGetId(exec, &exec_id) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(exec_id);
}

void RegisterPre134ExecGraphNodes(CUgraph root_graph, CUgraphExec exec) {
  // The caller (`MaybeFinalizeGraphTelemetry`) already checked that recording
  // is on.
  if (root_graph == nullptr || exec == nullptr) {
    return;
  }
  const std::optional<uint32_t> exec_id = GetGraphExecId(exec);
  if (!exec_id.has_value()) {
    return;
  }

  // Attach the exec graph's cleanup to `root_graph`. `exec` doesn't hold this
  // reference: it is attached after instantiation. So the entry lives exactly
  // as long as `root_graph`, which outlives `exec` because
  // `~CudaCommandBuffer()` destroys `graph_exec_` before `graph_`.
  if (!TrackAndAttach(*exec_id, root_graph, KeySpace::kPre134ExecGraph)
           .has_value()) {
    return;
  }

  uint32_t next_flat_local_id = 0;
  ProjectGraphHierarchyDfs(root_graph, *exec_id, &next_flat_local_id);
}

}  // namespace stream_executor::gpu::command_buffer_telemetry
