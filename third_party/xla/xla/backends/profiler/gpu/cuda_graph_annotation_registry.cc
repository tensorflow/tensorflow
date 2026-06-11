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

#include "xla/backends/profiler/gpu/cuda_graph_annotation_registry.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/profiler/gpu/cuda_driver_graph_interface.h"
#include "xla/backends/profiler/gpu/cuda_graph_topology_mapper.h"

#if CUDA_VERSION >= 13010
#include <algorithm>
#include <utility>

#include "xla/debug_options_flags.h"
#endif

namespace xla {
namespace profiler {

// Static members definition.
// Guarded by kConstInit to avoid dynamic initialization order issues.
absl::Mutex CudaGraphAnnotationRegistry::mutex_(absl::kConstInit);

// absl::NoDestructor prevents destructor call at exit, complying with Google
// style.
absl::NoDestructor<
    absl::flat_hash_map<uint32_t, absl::flat_hash_map<uint64_t, std::string>>>
    CudaGraphAnnotationRegistry::registry_;

absl::NoDestructor<absl::flat_hash_map<uint32_t, uint32_t>>
    CudaGraphAnnotationRegistry::graph_id_map_;

absl::NoDestructor<
    absl::flat_hash_map<uint32_t, absl::flat_hash_map<uint32_t, uint64_t>>>
    CudaGraphAnnotationRegistry::tools_id_map_;

absl::NoDestructor<absl::flat_hash_map<uint32_t, size_t>>
    CudaGraphAnnotationRegistry::graph_sizes_;

absl::NoDestructor<absl::flat_hash_map<uint32_t, size_t>>
    CudaGraphAnnotationRegistry::merged_sizes_;

absl::NoDestructor<absl::flat_hash_map<
    uint32_t, std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>>
    CudaGraphAnnotationRegistry::child_graphs_;

const CudaDriverGraphInterface* CudaGraphAnnotationRegistry::driver_ =
    CudaDriverGraphInterface::GetDefault();

namespace {
thread_local const CudaDriverGraphInterface* tls_driver = nullptr;
}  // namespace

const CudaDriverGraphInterface* CudaGraphAnnotationRegistry::GetDriver() {
  if (tls_driver != nullptr) {
    return tls_driver;
  }
  return driver_;
}

const CudaDriverGraphInterface* CudaGraphAnnotationRegistry::ExchangeCudaDriver(
    const CudaDriverGraphInterface* driver) {
  const CudaDriverGraphInterface* old = tls_driver;
  tls_driver = driver;
  return old;
}

void CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
    CUgraph graph, CUgraphNode node, absl::string_view annotation) {
#if CUDA_VERSION >= 13010
  unsigned int graph_id = 0;
  if (GetDriver()->GetGraphId(graph, &graph_id) != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query CUDA Graph ID";
    return;
  }

  uint64_t tools_id = 0;
  if (GetDriver()->GetNodeToolsId(node, &tools_id) != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query CUDA Graph Node Tools ID";
    return;
  }

  VLOG(5) << "[REGISTRY] Registering CUDA Graph Node Annotation: graph_id="
          << graph_id << ", tools_id=" << tools_id
          << ", annotation=" << annotation;

  absl::MutexLock lock(mutex_);
  (*registry_)[graph_id][tools_id] = std::string(annotation);
#else
  static bool log_once = []() {
    LOG(WARNING) << "CUDA Graphs telemetry is disabled (requires CUDA 13.1+)";
    return true;
  }();
  (void)log_once;
#endif
}

void CudaGraphAnnotationRegistry::RegisterGraphSize(CUgraph graph,
                                                    size_t size) {
#if CUDA_VERSION >= 13010
  unsigned int graph_id = 0;
  if (GetDriver()->GetGraphId(graph, &graph_id) != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query CUDA Graph ID during size registration";
    return;
  }
  VLOG(5) << "[REGISTRY] RegisterGraphSize: graph_id=" << graph_id
          << ", size=" << size;
  absl::MutexLock lock(mutex_);
  (*graph_sizes_)[graph_id] = size;
#endif
}

void CudaGraphAnnotationRegistry::RegisterChildGraph(CUgraph parent_graph,
                                                     CUgraph child_graph,
                                                     CUgraphNode child_node,
                                                     bool is_conditional) {
#if CUDA_VERSION >= 13010
  unsigned int parent_graph_id = 0;
  if (GetDriver()->GetGraphId(parent_graph, &parent_graph_id) != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query parent CUDA Graph ID";
    return;
  }
  unsigned int child_graph_id = 0;
  if (GetDriver()->GetGraphId(child_graph, &child_graph_id) != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query child CUDA Graph ID";
    return;
  }
  uint64_t child_node_tools_id = 0;
  auto res = GetDriver()->GetNodeToolsId(child_node, &child_node_tools_id);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query child CUDA Graph Node Tools ID, err=" << res;
    return;
  }
  uint32_t insertion_point = child_node_tools_id & 0xFFFFFFFF;

  VLOG(5) << "[REGISTRY] RegisterChildGraph: parent=" << parent_graph_id
          << ", child=" << child_graph_id << ", raw_tools_id=0x" << std::hex
          << child_node_tools_id << std::dec
          << ", insertion_point=" << insertion_point
          << ", is_conditional=" << is_conditional;

  absl::MutexLock lock(mutex_);
  auto& children = (*child_graphs_)[parent_graph_id];
  children.push_back({child_graph_id, insertion_point, is_conditional});
  std::sort(children.begin(), children.end());
#endif
}

void CudaGraphAnnotationRegistry::RegisterGraphExec(CUgraphExec graph_exec,
                                                    CUgraph graph) {
#if CUDA_VERSION >= 13010
  unsigned int exec_id = 0;
  if (GetDriver()->GetExecId(graph_exec, &exec_id) != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query CUDA Graph Exec ID";
    return;
  }
  unsigned int template_id = 0;
  if (GetDriver()->GetGraphId(graph, &template_id) != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query CUDA Graph ID from template";
    return;
  }

  VLOG(5) << "[REGISTRY] RegisterGraphExec: exec_id=" << exec_id
          << ", template_id=" << template_id;

  RegisterGraphMapping(exec_id, template_id);

  size_t merged_size = 0;
  {
    absl::MutexLock lock(mutex_);
    merged_size = CalculateMergedSize(template_id);
  }

  VLOG(5) << "[REGISTRY] Pre-computing merged mappings for exec_id=" << exec_id
          << ", template_id=" << template_id << ", merged_size=" << merged_size;

  std::vector<std::pair<uint32_t, uint64_t>> local_mappings;
  local_mappings.reserve(merged_size);
  {
    absl::ReaderMutexLock lock(mutex_);
    for (uint32_t i = 0; i < merged_size; ++i) {
      auto resolved = ResolveMergedNodeReadOnly(template_id, i);
      uint64_t template_tools_id =
          (static_cast<uint64_t>(resolved.first) << 32) | resolved.second;
      local_mappings.push_back({i, template_tools_id});
    }
  }

  {
    absl::MutexLock lock(mutex_);
    auto& exec_map = (*tools_id_map_)[exec_id];
    for (const auto& mapping : local_mappings) {
      exec_map[mapping.first] = mapping.second;
      VLOG(3) << "[REGISTRY] Pre-computed mapping: " << exec_id << " | "
              << mapping.first << " -> " << mapping.second;
    }
  }
#endif
}

void CudaGraphAnnotationRegistry::UnregisterGraphAnnotations(CUgraph graph) {
#if CUDA_VERSION >= 13010
  unsigned int graph_id = 0;
  if (GetDriver()->GetGraphId(graph, &graph_id) != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query CUDA Graph ID during destruction";
    return;
  }

  VLOG(3) << "Unregistering CUDA Graph Annotations for graph_id=" << graph_id;

  absl::MutexLock lock(mutex_);
  registry_->erase(graph_id);
  graph_sizes_->erase(graph_id);
  child_graphs_->erase(graph_id);
  merged_sizes_->erase(graph_id);
#endif
}

void CudaGraphAnnotationRegistry::RegisterGraphMapping(
    uint32_t instantiated_graph_id, uint32_t template_graph_id) {
#if CUDA_VERSION >= 13010
  absl::MutexLock lock(mutex_);
  (*graph_id_map_)[instantiated_graph_id] = template_graph_id;
  VLOG(5) << "[REGISTRY] RegisterGraphMapping: " << instantiated_graph_id
          << " -> " << template_graph_id;
#endif
}

void CudaGraphAnnotationRegistry::UnregisterGraphExec(CUgraphExec graph_exec) {
#if CUDA_VERSION >= 13010
  unsigned int exec_id = 0;
  if (GetDriver()->GetExecId(graph_exec, &exec_id) != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to query CUDA Graph Exec ID during destruction";
    return;
  }

  VLOG(3) << "Unregistering CUDA Graph Exec mappings for exec_id=" << exec_id;

  absl::MutexLock lock(mutex_);
  graph_id_map_->erase(exec_id);
  tools_id_map_->erase(exec_id);
#endif
}

std::string CudaGraphAnnotationRegistry::LookupAnnotation(uint32_t graph_id,
                                                          uint64_t tools_id) {
#if CUDA_VERSION >= 13010
  absl::MutexLock lock(mutex_);

  uint32_t target_graph_id = graph_id;
  uint64_t target_tools_id = tools_id;
  uint32_t target_node_index = tools_id & 0xFFFFFFFF;

  // 1. Try to map tools_id
  auto exec_it = tools_id_map_->find(graph_id);
  if (exec_it != tools_id_map_->end()) {
    auto tools_it = exec_it->second.find(target_node_index);
    if (tools_it != exec_it->second.end()) {
      target_tools_id = tools_it->second;
      target_graph_id = target_tools_id >> 32;
      target_node_index = target_tools_id & 0xFFFFFFFF;
      VLOG(5) << "[REGISTRY] Lookup mapped ToolsID: " << tools_id << " -> "
              << target_tools_id << " (Graph " << graph_id << " -> "
              << target_graph_id << ")";
    }
  } else {
    // 2. Fallback to map graph_id only
    auto graph_it = graph_id_map_->find(graph_id);
    if (graph_it != graph_id_map_->end()) {
      target_graph_id = graph_it->second;
      // Reconstruct target_tools_id using mapped graph_id and original node_id
      target_tools_id =
          (static_cast<uint64_t>(target_graph_id) << 32) | target_node_index;
      VLOG(5) << "[REGISTRY] Lookup mapped GraphID only: " << graph_id << " -> "
              << target_graph_id
              << ", reconstructed ToolsID: " << target_tools_id;
    }
  }

  // 3. Dynamic Resolution for nested graphs
  auto child_it = child_graphs_->find(target_graph_id);
  if (child_it != child_graphs_->end()) {
    bool is_conditional_node = false;
    for (const auto& child : child_it->second) {
      if (child.insertion_point == target_node_index && child.is_conditional) {
        is_conditional_node = true;
        break;
      }
    }

    if (is_conditional_node) {
      VLOG(5) << "[REGISTRY] Lookup mapped to conditional node: "
              << target_tools_id << ", returning placeholder";
      return "Thunk:#hlo_op=driver_conditional_helper#";
    }

    auto resolved = CudaGraphTopologyMapper::ResolveMergedNode(
        target_graph_id, target_node_index, *graph_sizes_, *child_graphs_,
        merged_sizes_.get());
    if (resolved.first != target_graph_id ||
        resolved.second != target_node_index) {
      target_graph_id = resolved.first;
      target_node_index = resolved.second;
      target_tools_id =
          (static_cast<uint64_t>(target_graph_id) << 32) | target_node_index;
      VLOG(5) << "[REGISTRY] Resolved merged ToolsID: " << tools_id << " -> "
              << target_tools_id << " (Graph " << target_graph_id << ")";
    }
  }

  std::string kernel_annotation;
  auto it = registry_->find(target_graph_id);
  if (it != registry_->end()) {
    auto node_it = it->second.find(target_tools_id);
    if (node_it != it->second.end()) {
      kernel_annotation = node_it->second;
    }
  }

  if (!kernel_annotation.empty()) {
    VLOG(5) << "[REGISTRY] Lookup successful: graph_id=" << graph_id
            << ", tools_id=" << tools_id
            << " (mapped to graph=" << target_graph_id
            << ", tools=" << target_tools_id << ") -> " << kernel_annotation;
    return kernel_annotation;
  } else {
    VLOG(5) << "[REGISTRY] Lookup failed (kernel annotation missing): graph_id="
            << graph_id << ", tools_id=" << tools_id
            << " (mapped to graph=" << target_graph_id
            << ", tools=" << target_tools_id << ")";
  }
#else
  VLOG(5) << "[REGISTRY] Lookup bypassed (CUDA < 13.1): graph_id=" << graph_id
          << ", tools_id=" << tools_id;
#endif
  return "";
}

#if CUDA_VERSION >= 13010
size_t CudaGraphAnnotationRegistry::CalculateMergedSize(uint32_t graph_id) {
  return CudaGraphTopologyMapper::CalculateMergedSize(
      graph_id, *graph_sizes_, *child_graphs_, merged_sizes_.get());
}

std::pair<uint32_t, uint32_t>
CudaGraphAnnotationRegistry::ResolveMergedNodeReadOnly(uint32_t graph_id,
                                                       uint32_t node_index) {
  auto* mutable_merged_sizes =
      const_cast<absl::flat_hash_map<uint32_t, size_t>*>(merged_sizes_.get());
  return CudaGraphTopologyMapper::ResolveMergedNode(
      graph_id, node_index, *graph_sizes_, *child_graphs_,
      mutable_merged_sizes);
}
#endif

void CudaGraphAnnotationRegistry::ResetForTesting() {
  absl::MutexLock lock(mutex_);
  registry_->clear();
  graph_id_map_->clear();
  tools_id_map_->clear();
  graph_sizes_->clear();
  child_graphs_->clear();
  merged_sizes_->clear();
}

}  // namespace profiler
}  // namespace xla
