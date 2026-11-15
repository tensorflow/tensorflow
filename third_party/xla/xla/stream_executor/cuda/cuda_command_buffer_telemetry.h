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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"

// Producer-side helpers that `CudaCommandBuffer` uses to record node
// annotations in `CudaGraphAnnotationRegistry::Default()`. All of them make
// CUDA calls, so callers check `CudaCommandBuffer::ShouldRecordAnnotations()`
// first.
//
// Built from cuda_command_buffer_telemetry.cc with CUDA 13.2+ headers, which
// declare the graph and node ID APIs it uses. With older headers, BUILD selects
// cuda_command_buffer_telemetry_stub.cc instead, where every function does
// nothing; it is never reached because `SupportsNodeAnnotations()` is false.
//
// Entry lifetime is owned by XLA, not by CUDA: each root `CudaCommandBuffer`
// records the entries it creates in `telemetry_graphs_`, retains nested
// entries installed on child nodes via `RetainGraphs`, and unregisters them on
// update or destruction. CUDA user objects are deliberately not used for this:
// an executable graph that owns any user object has a much smaller per-stream
// in-flight launch limit (b/578493713).
namespace stream_executor::gpu::command_buffer_telemetry {

// A graph tracked in `CudaGraphAnnotationRegistry::Default()`, with its
// KeySpace.
using TrackedGraph = CudaGraphAnnotationRegistry::GraphRef;

// Returns `cuGraphGetId(graph)`, or `std::nullopt` if `graph` is null or the
// call fails.
std::optional<uint32_t> GetGraphId(CUgraph graph);

// Returns `cuGraphNodeGetToolsId(node)`, or `std::nullopt` if `node` is null or
// the call fails.
std::optional<uint64_t> GetToolsId(CUgraphNode node);

// Returns the graph that contains `node` from `cuGraphNodeGetContainingGraph`,
// or `std::nullopt` if `node` is null or the call fails.
std::optional<CUgraph> GetContainingGraph(CUgraphNode node);

// Returns the nodes of `graph` from `cuGraphGetNodes`, or an empty vector if
// `graph` is null, empty, or the query fails.
std::vector<CUgraphNode> GetGraphNodes(CUgraph graph);

// Invokes `fn` on each subgraph owned by `node` (child graph or conditional
// bodies), paired with the matching subgraph of `original_node`, or nullptr.
void ForEachSubgraphPair(
    CUgraphNode node, CUgraphNode original_node,
    absl::FunctionRef<void(CUgraph subgraph, CUgraph orig_subgraph)> fn);

// Tracks `graph_id` in `space` unless it is tracked already. If this call
// created the entry, appends it to `owned`: the owner of that list must
// unregister it (UnregisterTrackedGraphs) when the graph is destroyed.
TrackedGraph Track(uint32_t graph_id, std::vector<TrackedGraph>& owned,
                   CudaGraphAnnotationRegistry::KeySpace space =
                       CudaGraphAnnotationRegistry::KeySpace::kTemplateGraph);

// Unregisters `graphs` from `CudaGraphAnnotationRegistry::Default()`, under a
// single registry lock. While a harvest hold is active, their annotations stay
// resolvable until it is released. Makes no CUDA calls.
void UnregisterTrackedGraphs(absl::Span<const TrackedGraph> graphs);

// Registers `annotation` for `node`, a node of `graph`, unless the node has an
// annotation already.
void RegisterNode(const TrackedGraph& graph, CUgraphNode node,
                  absl::string_view annotation);

// Registers every node of `graph` (recursing into subgraphs) using `fallback`
// or a copy from `original`. Relies on `cuGraphGetNodes` returning nodes of a
// cloned graph in the same order as its original's. Skips subgraphs that are
// tracked already, and appends the subgraphs it tracks to `owned`; subgraphs
// live as long as `graph`, so they share its owner.
void RegisterGraphNodes(CUgraph graph, const TrackedGraph& tracked,
                        CUgraph original, absl::string_view fallback,
                        std::vector<TrackedGraph>& owned);

}  // namespace stream_executor::gpu::command_buffer_telemetry

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_H_
