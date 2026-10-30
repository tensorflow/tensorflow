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

// Selected by BUILD instead of cuda_command_buffer_telemetry.cc with CUDA
// headers older than 13.2, which lack the graph and node ID APIs it uses. There
// `CudaCommandBuffer::SupportsNodeAnnotations()` is false, so these functions
// are never reached; they only keep the library linkable.

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry.h"
#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"

namespace stream_executor::gpu::command_buffer_telemetry {

std::optional<uint32_t> GetGraphId(CUgraph /*graph*/) { return std::nullopt; }

std::optional<uint64_t> GetToolsId(CUgraphNode /*node*/) {
  return std::nullopt;
}

std::optional<CUgraph> GetContainingGraph(CUgraphNode /*node*/) {
  return std::nullopt;
}

std::vector<CUgraphNode> GetGraphNodes(CUgraph /*graph*/) { return {}; }

void ForEachSubgraphPair(
    CUgraphNode /*node*/, CUgraphNode /*original_node*/,
    absl::FunctionRef<void(CUgraph subgraph, CUgraph orig_subgraph)> /*fn*/) {}

std::optional<TrackedGraph> TrackAndAttach(
    uint32_t /*graph_id*/, CUgraph /*attach_to*/,
    CudaGraphAnnotationRegistry::KeySpace /*space*/) {
  return std::nullopt;
}

void RegisterNode(const TrackedGraph& /*graph*/, CUgraphNode /*node*/,
                  absl::string_view /*annotation*/) {}

void RegisterGraphNodes(CUgraph /*graph*/, const TrackedGraph& /*tracked*/,
                        CUgraph /*original*/, absl::string_view /*fallback*/) {}

}  // namespace stream_executor::gpu::command_buffer_telemetry
