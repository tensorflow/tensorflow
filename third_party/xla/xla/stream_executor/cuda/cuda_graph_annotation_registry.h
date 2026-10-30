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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_ANNOTATION_REGISTRY_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_ANNOTATION_REGISTRY_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::gpu {

// Process-wide map from `(KeySpace, graph_id)` to the HLO annotation active
// when XLA created each CUDA graph node. `CudaCommandBuffer` populates entries
// when command buffer telemetry is enabled and attaches a `CUuserObject` to
// unregister them when the graph is destroyed (see
// cuda_command_buffer_telemetry.h); `CuptiTraceCollector` resolves them during
// harvest.
//
// While a `HarvestHold` is active, unregistered entries are marked `destroyed`
// ("parked", capped at `max_parked_bytes`) instead of erased so graphs
// destroyed mid-session still resolve at harvest.
//
// Graph IDs (`cuGraphGetId` / `cuGraphExecGetId`) are monotonically increasing
// 32-bit IDs that are not recycled on graph destruction (verified by
// `GraphAndGraphExecIdsAreStrictlyIncreasingAcrossDestruction` in
// cuda_command_buffer_telemetry_pre_13_4_test.cc) and are assumed not to wrap
// around (b/515425329). `mu_` is a leaf lock: no CUDA calls or logging while
// held, and erased entries are destroyed after unlock.
class CudaGraphAnnotationRegistry {
 public:
  // Disjoint key spaces for template (`CUgraph`) and pre-13.4 executable
  // (`CUgraphExec`) graph IDs, so their entries never collide.
  enum class KeySpace : uint8_t {
    kTemplateGraph = 0,
    // TODO(b/483450622): Delete once CUDA < 13.4 support is retired.
    kPre134ExecGraph = 1,
  };

  // Which graph and node IDs `CudaCommandBuffer` registers, and in which
  // KeySpace. Readers look up by the IDs each CUPTI record carries.
  enum class TelemetryStrategy : uint8_t {
    kDisabled = 0,
    // CUDA 13.2-13.3: `(graphId, graphNodeId)` in `kPre134ExecGraph`.
    // TODO(b/483450622): Delete once CUDA < 13.4 support is retired.
    kPre134ExecGraph,
    // CUDA 13.4+: `(sourceGraphId, sourceGraphNodeId)` in `kTemplateGraph`.
    // P2P memcpy records lack these fields and keep the launch annotation.
    kCuda134SourceGraph,
  };

  // Picks the telemetry strategy from the CUDA toolkit XLA was compiled with
  // and the driver and runtime it runs on. A strategy depends only on the
  // compile-time toolkit version, the driver version, and the runtime version.
  static TelemetryStrategy SelectTelemetryStrategy(
      SemanticVersion compiled_toolkit, SemanticVersion driver,
      SemanticVersion runtime);

  // Byte counts are sums of annotation sizes, without container overhead.
  // Walks every entry; for diagnostics only.
  struct Stats {
    size_t live_graphs = 0;
    size_t live_nodes = 0;
    size_t live_bytes = 0;
    size_t parked_graphs = 0;
    size_t parked_nodes = 0;
    size_t parked_bytes = 0;
    int active_holds = 0;

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Stats& stats) {
      absl::Format(
          &sink,
          "live_graphs=%d live_nodes=%d live_bytes=%d parked_graphs=%d "
          "parked_nodes=%d parked_bytes=%d active_holds=%d",
          stats.live_graphs, stats.live_nodes, stats.live_bytes,
          stats.parked_graphs, stats.parked_nodes, stats.parked_bytes,
          stats.active_holds);
    }
  };

  // Keeps entries of graphs destroyed during a profiling session until the
  // session's records are harvested. Move-only; releases the hold when
  // destroyed.
  class HarvestHold {
   public:
    HarvestHold(HarvestHold&& other) noexcept
        : registry_(std::exchange(other.registry_, nullptr)) {}
    HarvestHold& operator=(HarvestHold&& other) = delete;
    HarvestHold(const HarvestHold&) = delete;
    HarvestHold& operator=(const HarvestHold&) = delete;
    ~HarvestHold();

   private:
    friend class CudaGraphAnnotationRegistry;
    explicit HarvestHold(CudaGraphAnnotationRegistry* registry)
        : registry_(registry) {}

    CudaGraphAnnotationRegistry* registry_ = nullptr;
  };

  // Default for `max_parked_bytes`: 256 MiB.
  static constexpr size_t kDefaultMaxParkedBytes = size_t{256} << 20;

  // Returns the process-wide registry. It is never destroyed, so command
  // buffers destroyed during process exit can still unregister.
  static CudaGraphAnnotationRegistry& Default();

  explicit CudaGraphAnnotationRegistry(
      size_t max_parked_bytes = kDefaultMaxParkedBytes);

  CudaGraphAnnotationRegistry(const CudaGraphAnnotationRegistry&) = delete;
  CudaGraphAnnotationRegistry& operator=(const CudaGraphAnnotationRegistry&) =
      delete;

  // Starts tracking `graph_id` in `space` unless it has an entry already.
  // Returns true if this call created the entry: the caller then attaches the
  // `CUuserObject` that unregisters it when the graph is destroyed. A parked
  // entry is not revived, since its graph is destroyed.
  bool TrackGraph(uint32_t graph_id, KeySpace space = KeySpace::kTemplateGraph)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Returns whether `graph_id` has a live entry in `space`. Parked entries
  // don't count: they belong to destroyed graphs.
  bool IsGraphTracked(uint32_t graph_id,
                      KeySpace space = KeySpace::kTemplateGraph) const
      ABSL_LOCKS_EXCLUDED(mu_);

  // First registration wins. Returns false if empty, untracked, parked, or
  // present.
  bool RegisterNodeAnnotation(uint32_t graph_id, uint64_t node_id,
                              absl::string_view annotation,
                              KeySpace space = KeySpace::kTemplateGraph)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Like RegisterNodeAnnotation, with a copy of a template node's annotation
  // (live or parked), or fallback. One lock, so the source can't be purged in
  // between.
  bool CopyNodeAnnotation(uint32_t graph_id, uint64_t node_id,
                          uint32_t source_graph_id, uint64_t source_node_id,
                          absl::string_view fallback,
                          KeySpace space = KeySpace::kTemplateGraph)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Calls visitor with the annotation (live or parked) under the reader lock;
  // the visitor must copy and must not re-enter.
  bool VisitNodeAnnotation(uint32_t graph_id, uint64_t node_id,
                           absl::FunctionRef<void(absl::string_view)> visitor,
                           KeySpace space = KeySpace::kTemplateGraph) const
      ABSL_LOCKS_EXCLUDED(mu_);

  // Unregisters `graph_id` in `space`, parking the entry if a harvest hold is
  // active or erasing it otherwise. Called from the CUuserObject destructor on
  // a CUDA driver thread; makes no CUDA calls and does not log.
  void UnregisterGraph(uint32_t graph_id,
                       KeySpace space = KeySpace::kTemplateGraph)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Takes a harvest hold. Holds are counted, so with overlapping sessions the
  // parked entries are purged only when the last hold is released.
  [[nodiscard]] HarvestHold AcquireHarvestHold() ABSL_LOCKS_EXCLUDED(mu_);

  // Returns whether there are neither live nor parked entries. Takes no lock,
  // so consumers can skip lookups cheaply while telemetry is off.
  bool empty() const;

  Stats GetStats() const ABSL_LOCKS_EXCLUDED(mu_);

  // For tests that can't reach the recorded CUgraph nodes: returns a copy of
  // every live and parked annotation in `space`, sorted.
  std::vector<std::string> GetNodeAnnotationsForTesting(KeySpace space) const
      ABSL_LOCKS_EXCLUDED(mu_);

  // For tests: drops all live and parked entries, but keeps the active holds.
  void ClearForTesting() ABSL_LOCKS_EXCLUDED(mu_);

 private:
  using GraphKey = uint64_t;
  static constexpr GraphKey MakeGraphKey(KeySpace space, uint32_t graph_id) {
    return (static_cast<uint64_t>(space) << 32) | graph_id;
  }

  struct GraphEntry {
    // Sum of the annotation sizes.
    size_t bytes = 0;
    absl::flat_hash_map<uint64_t, std::string> nodes;
    // Parked: the graph was destroyed while a hold was active. The entry only
    // serves lookups until the last hold is released.
    bool destroyed = false;
  };
  using GraphMap = absl::flat_hash_map<GraphKey, GraphEntry>;

  // Decrements the hold count. The last release purges the parked entries and
  // warns if the session dropped any.
  void ReleaseHarvestHold() ABSL_LOCKS_EXCLUDED(mu_);

  // Returns the live (not parked) entry of `graph_key`, or null.
  GraphEntry* FindLiveEntryLocked(GraphKey graph_key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Returns the annotation of the node, from a live or parked entry.
  const std::string* FindNodeLocked(GraphKey graph_key, uint64_t node_id) const
      ABSL_SHARED_LOCKS_REQUIRED(mu_);
  void UpdateNumGraphsLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const size_t max_parked_bytes_;

  mutable absl::Mutex mu_;
  // Live and parked entries.
  GraphMap graphs_ ABSL_GUARDED_BY(mu_);
  int active_holds_ ABSL_GUARDED_BY(mu_) = 0;
  size_t parked_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  uint64_t park_dropped_in_session_ ABSL_GUARDED_BY(mu_) = 0;
  // `graphs_.size()`, readable without the lock by `empty()`.
  std::atomic<size_t> num_graphs_{0};
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_ANNOTATION_REGISTRY_H_
