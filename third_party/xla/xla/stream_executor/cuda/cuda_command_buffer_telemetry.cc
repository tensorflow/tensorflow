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

#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"
#include "xla/stream_executor/cuda/cuda_user_object.h"

// Selected by BUILD only with CUDA 13.2+ headers; see
// cuda_command_buffer_telemetry_stub.cc for older ones.
namespace stream_executor::gpu::command_buffer_telemetry {
namespace {

using Registry = CudaGraphAnnotationRegistry;
using KeySpace = CudaGraphAnnotationRegistry::KeySpace;
using TrackMode = CudaGraphAnnotationRegistry::TrackMode;

// The object that `TrackAndAttach` attaches to a graph as a `CUuserObject`.
// Its destructor unregisters the graph's entry once CUDA destroys the user
// object.
class GraphTelemetryCleanup {
 public:
  GraphTelemetryCleanup(uint32_t graph_id, Registry::Token token,
                        KeySpace space)
      : graph_id_(graph_id), token_(token), space_(space) {}

  // A moved-from cleanup holds the invalid token, so only one of the two
  // unregisters the entry.
  GraphTelemetryCleanup(GraphTelemetryCleanup&& other) noexcept
      : graph_id_(std::exchange(other.graph_id_, 0)),
        token_(std::exchange(other.token_, Registry::kInvalidToken)),
        space_(other.space_) {}
  GraphTelemetryCleanup& operator=(GraphTelemetryCleanup&&) = delete;

  // Because of `CU_USER_OBJECT_NO_DESTRUCTOR_SYNC`, this runs on a CUDA driver
  // thread, possibly long after the graph was destroyed: it must not call
  // CUDA. `UnregisterGraph` makes no CUDA calls, doesn't log, and ignores the
  // call if a newer graph with the same ID has replaced the entry.
  ~GraphTelemetryCleanup() {
    if (token_ != Registry::kInvalidToken) {
      Registry::Default().UnregisterGraph(graph_id_, token_, space_);
    }
  }

 private:
  uint32_t graph_id_;
  Registry::Token token_;
  KeySpace space_;
};

}  // namespace

std::optional<uint32_t> GetGraphId(CUgraph graph) {
  unsigned int graph_id = 0;
  if (graph == nullptr || cuGraphGetId(graph, &graph_id) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(graph_id);
}

std::optional<uint64_t> GetToolsId(CUgraphNode node) {
  unsigned long long tools_id = 0;  // NOLINT(runtime/int): CUDA API type.
  if (node == nullptr ||
      cuGraphNodeGetToolsId(node, &tools_id) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
  // Container nodes never produce CUPTI activity records, and excluding them
  // keeps annotations consistent at every nesting level.
  if (cuGraphNodeGetType(node, &type) == CUDA_SUCCESS &&
      (type == CU_GRAPH_NODE_TYPE_GRAPH ||
       type == CU_GRAPH_NODE_TYPE_CONDITIONAL)) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(tools_id);
}

std::optional<CUgraph> GetContainingGraph(CUgraphNode node) {
  CUgraph graph = nullptr;
  if (node == nullptr ||
      cuGraphNodeGetContainingGraph(node, &graph) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  return graph;
}

std::vector<CUgraphNode> GetGraphNodes(CUgraph graph) {
  size_t num_nodes = 0;
  if (graph == nullptr ||
      cuGraphGetNodes(graph, nullptr, &num_nodes) != CUDA_SUCCESS ||
      num_nodes == 0) {
    return {};
  }
  std::vector<CUgraphNode> nodes(num_nodes);
  if (cuGraphGetNodes(graph, nodes.data(), &num_nodes) != CUDA_SUCCESS) {
    return {};
  }
  nodes.resize(num_nodes);
  return nodes;
}

void ForEachSubgraphPair(
    CUgraphNode node, CUgraphNode original_node,
    absl::FunctionRef<void(CUgraph subgraph, CUgraph orig_subgraph)> fn) {
  CUgraphNodeType type;
  if (node == nullptr || cuGraphNodeGetType(node, &type) != CUDA_SUCCESS) {
    return;
  }
  CUgraphNodeType orig_type = CU_GRAPH_NODE_TYPE_EMPTY;
  const bool orig_matches_type =
      original_node != nullptr &&
      cuGraphNodeGetType(original_node, &orig_type) == CUDA_SUCCESS &&
      orig_type == type;

  if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
    CUgraph child_graph = nullptr;
    if (cuGraphChildGraphNodeGetGraph(node, &child_graph) == CUDA_SUCCESS &&
        child_graph != nullptr) {
      CUgraph orig_child = nullptr;
      if (orig_matches_type) {
        cuGraphChildGraphNodeGetGraph(original_node, &orig_child);
      }
      fn(child_graph, orig_child);
    }
  } else if (type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    CUgraphNodeParams params{};
    if (cuGraphNodeGetParams(node, &params) == CUDA_SUCCESS) {
      CUgraphNodeParams orig_params{};
      const bool has_orig_bodies =
          orig_matches_type &&
          cuGraphNodeGetParams(original_node, &orig_params) == CUDA_SUCCESS &&
          orig_params.conditional.size == params.conditional.size;
      for (unsigned int i = 0; i < params.conditional.size; ++i) {
        if (CUgraph body_graph = params.conditional.phGraph_out[i];
            body_graph != nullptr) {
          CUgraph orig_body = has_orig_bodies
                                  ? orig_params.conditional.phGraph_out[i]
                                  : nullptr;
          fn(body_graph, orig_body);
        }
      }
    }
  }
}

std::optional<TrackedGraph> TrackAndAttach(uint32_t graph_id, CUgraph attach_to,
                                           TrackMode mode, KeySpace space) {
  Registry& registry = Registry::Default();
  const auto [token, newly_tracked] =
      registry.TrackGraph(graph_id, mode, space);
  if (!newly_tracked) {
    // Whoever created the entry attached its cleanup.
    return TrackedGraph{graph_id, token, space};
  }
  const absl::Status status =
      CudaUserObject<GraphTelemetryCleanup>::Create(
          GraphTelemetryCleanup(graph_id, token, space), attach_to)
          .status();
  if (!status.ok()) {
    // Without a cleanup the entry would outlive the graph, so remove it now,
    // together with anything other threads registered in the meantime. If
    // `cuUserObjectCreate` failed, destroying the cleanup already removed it.
    // If only the attach failed, `CudaUserObject` released the user object,
    // and its destructor, which may run later, finds a stale token.
    registry.UnregisterGraph(graph_id, token, space);
    LOG_EVERY_N_SEC(WARNING, 60)
        << "Command buffer telemetry is disabled for CUDA graph " << graph_id
        << ": failed to attach the cleanup of its annotations: " << status;
    return std::nullopt;
  }
  return TrackedGraph{graph_id, token, space};
}

void RegisterNode(const TrackedGraph& graph, CUgraphNode node,
                  absl::string_view annotation) {
  if (annotation.empty()) {
    return;
  }
  if (std::optional<uint64_t> tools_id = GetToolsId(node)) {
    Registry::Default().RegisterNodeAnnotation(
        graph.graph_id, graph.token, *tools_id, annotation, graph.space);
  }
}

void RegisterGraphNodes(CUgraph graph, const TrackedGraph& tracked,
                        CUgraph original, absl::string_view fallback,
                        TrackMode child_mode) {
  const std::optional<uint32_t> original_id = GetGraphId(original);
  if (!original_id.has_value() && fallback.empty()) {
    return;
  }

  const std::vector<CUgraphNode> nodes = GetGraphNodes(graph);
  if (nodes.empty()) {
    return;
  }
  const std::vector<CUgraphNode> original_nodes =
      original_id.has_value() ? GetGraphNodes(original)
                              : std::vector<CUgraphNode>();
  const bool can_copy_from_original =
      original_id.has_value() && original_nodes.size() == nodes.size();
  if (!can_copy_from_original && fallback.empty()) {
    return;
  }

  Registry& registry = Registry::Default();
  for (size_t i = 0; i < nodes.size(); ++i) {
    const CUgraphNode node = nodes[i];
    const CUgraphNode orig_node =
        can_copy_from_original ? original_nodes[i] : nullptr;
    if (std::optional<uint64_t> tools_id = GetToolsId(node)) {
      std::optional<uint64_t> orig_tools_id =
          can_copy_from_original ? GetToolsId(orig_node) : std::nullopt;
      if (orig_tools_id.has_value()) {
        registry.CopyNodeAnnotation(tracked.graph_id, tracked.token, *tools_id,
                                    *original_id, *orig_tools_id, fallback,
                                    tracked.space);
      } else {
        registry.RegisterNodeAnnotation(tracked.graph_id, tracked.token,
                                        *tools_id, fallback, tracked.space);
      }
    }

    ForEachSubgraphPair(
        node, orig_node, [&](CUgraph child, CUgraph orig_child) {
          // A copy-only walk can only copy from a tracked original, so don't
          // track the child (and attach a cleanup) for nothing.
          if (fallback.empty()) {
            const std::optional<uint32_t> orig_child_id =
                GetGraphId(orig_child);
            if (!orig_child_id.has_value() ||
                !registry.IsGraphTracked(*orig_child_id)) {
              return;
            }
          }
          const std::optional<uint32_t> child_id = GetGraphId(child);
          if (!child_id.has_value() ||
              (child_mode == TrackMode::kGetOrCreate &&
               registry.IsGraphTracked(*child_id, tracked.space))) {
            return;
          }
          if (std::optional<TrackedGraph> child_tracked =
                  TrackAndAttach(*child_id, child, child_mode, tracked.space)) {
            RegisterGraphNodes(child, *child_tracked, orig_child, fallback,
                               child_mode);
          }
        });
  }
}

}  // namespace stream_executor::gpu::command_buffer_telemetry
