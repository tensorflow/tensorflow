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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUDA_GRAPH_ANNOTATION_REGISTRY_H_
#define XLA_BACKENDS_PROFILER_GPU_CUDA_GRAPH_ANNOTATION_REGISTRY_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/profiler/gpu/cuda_driver_graph_interface.h"
#include "xla/backends/profiler/gpu/cuda_graph_topology_mapper.h"

struct CUgraph_st;
typedef struct CUgraph_st* CUgraph;
struct CUgraphNode_st;
typedef struct CUgraphNode_st* CUgraphNode;
struct CUgraphExec_st;
typedef struct CUgraphExec_st* CUgraphExec;

namespace xla {
namespace profiler {

class ScopedCudaDriverOverrideForTesting;
class CudaGraphAnnotationRegistryTest;

// CudaGraphAnnotationRegistry is a host-side registry that maps CUDA Graph
// nodes (CUgraphNode) to their corresponding XLA HLO metadata annotations.
//
// Thread Safety:
// All public APIs are thread-safe and guarded by a global mutex.
class CudaGraphAnnotationRegistry {
 public:
  // Register an annotation for a raw (CUgraph, CUgraphNode) pair.
  // Internally queries their CUDA IDs and stores them in the global registry.
  static void RegisterNodeAnnotation(CUgraph graph, CUgraphNode node,
                                     absl::string_view annotation);

  // Register the size of a template graph.
  static void RegisterGraphSize(CUgraph graph, size_t size);

  // Register a child graph nested under a parent graph at a specific node.
  static void RegisterChildGraph(CUgraph parent_graph, CUgraph child_graph,
                                 CUgraphNode child_node, bool is_conditional);

  // Register a graph execution instance and pre-compute node mappings.
  static void RegisterGraphExec(CUgraphExec graph_exec, CUgraph graph);

  // Unregister all node mappings associated with a specific CUgraphExec.
  static void UnregisterGraphExec(CUgraphExec graph_exec);

  // Unregister all node annotations associated with a specific CUgraph when it
  // is destroyed to prevent memory leaks.
  static void UnregisterGraphAnnotations(CUgraph graph);

  // Lookup the annotation for a given graph ID and Tools ID.
  // Used by the CUPTI collector during trace processing.
  static std::string LookupAnnotation(uint32_t graph_id, uint64_t tools_id);

  // Register a mapping from instantiated graph ID to template graph ID.
  static void RegisterGraphMapping(uint32_t instantiated_graph_id,
                                   uint32_t template_graph_id);

 private:
  CudaGraphAnnotationRegistry() = delete;

  friend class CudaGraphAnnotationRegistryTest;

  // For testing only. Resets the registry to a clean state.
  static void ResetForTesting();

  static absl::Mutex mutex_;

  // graphId -> {toolsNodeId -> HLO Name}
  static absl::NoDestructor<
      absl::flat_hash_map<uint32_t, absl::flat_hash_map<uint64_t, std::string>>>
      registry_ ABSL_GUARDED_BY(mutex_);

  // instantiated_graph_id -> template_graph_id
  static absl::NoDestructor<absl::flat_hash_map<uint32_t, uint32_t>>
      graph_id_map_ ABSL_GUARDED_BY(mutex_);

  // instantiated_graph_exec_id -> {node_index -> template_tools_id}
  static absl::NoDestructor<
      absl::flat_hash_map<uint32_t, absl::flat_hash_map<uint32_t, uint64_t>>>
      tools_id_map_ ABSL_GUARDED_BY(mutex_);

  // template_graph_id -> size
  static absl::NoDestructor<absl::flat_hash_map<uint32_t, size_t>> graph_sizes_
      ABSL_GUARDED_BY(mutex_);

  // template_graph_id -> merged_size (cached)
  static absl::NoDestructor<absl::flat_hash_map<uint32_t, size_t>> merged_sizes_
      ABSL_GUARDED_BY(mutex_);

  // template_parent_graph_id -> list of child graphs
  static absl::NoDestructor<absl::flat_hash_map<
      uint32_t, std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>>
      child_graphs_ ABSL_GUARDED_BY(mutex_);

  static const CudaDriverGraphInterface* driver_;

  friend class ScopedCudaDriverOverrideForTesting;
  static const CudaDriverGraphInterface* ExchangeCudaDriver(
      const CudaDriverGraphInterface* driver);
  static const CudaDriverGraphInterface* GetDriver();

  static size_t CalculateMergedSize(uint32_t graph_id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  static std::pair<uint32_t, uint32_t> ResolveMergedNodeReadOnly(
      uint32_t graph_id, uint32_t node_index)
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUDA_GRAPH_ANNOTATION_REGISTRY_H_
