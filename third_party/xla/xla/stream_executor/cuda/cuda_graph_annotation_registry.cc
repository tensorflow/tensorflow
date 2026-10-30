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

#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::gpu {

CudaGraphAnnotationRegistry::HarvestHold::~HarvestHold() {
  if (registry_ != nullptr) {
    registry_->ReleaseHarvestHold();
  }
}

CudaGraphAnnotationRegistry& CudaGraphAnnotationRegistry::Default() {
  static absl::NoDestructor<CudaGraphAnnotationRegistry> instance;
  return *instance;
}

CudaGraphAnnotationRegistry::CudaGraphAnnotationRegistry(
    size_t max_parked_bytes)
    : max_parked_bytes_(max_parked_bytes) {}

bool CudaGraphAnnotationRegistry::TrackGraph(uint32_t graph_id,
                                             KeySpace space) {
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  bool parked = false;
  {
    absl::MutexLock lock(mu_);
    // An existing entry is kept, even a parked one: graph IDs are not reused,
    // so it belongs to this graph.
    auto [it, inserted] = graphs_.try_emplace(graph_key);
    if (inserted) {
      UpdateNumGraphsLocked();
      return true;
    }
    parked = it->second.destroyed;
  }
  if (parked) {
    LOG_FIRST_N(WARNING, 1)
        << "CudaGraphAnnotationRegistry::TrackGraph saw a parked entry for "
           "graph_id="
        << graph_id << " (possible 32-bit graph ID wraparound, b/515425329).";
  }
  return false;
}

bool CudaGraphAnnotationRegistry::IsGraphTracked(uint32_t graph_id,
                                                 KeySpace space) const {
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  absl::ReaderMutexLock lock(mu_);
  auto it = graphs_.find(graph_key);
  return it != graphs_.end() && !it->second.destroyed;
}

bool CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
    uint32_t graph_id, uint64_t node_id, absl::string_view annotation,
    KeySpace space) {
  if (annotation.empty()) {
    return false;
  }
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  absl::MutexLock lock(mu_);
  GraphEntry* entry = FindLiveEntryLocked(graph_key);
  if (entry == nullptr) {
    return false;
  }
  if (!entry->nodes.try_emplace(node_id, annotation).second) {
    return false;
  }
  entry->bytes += annotation.size();
  return true;
}

bool CudaGraphAnnotationRegistry::CopyNodeAnnotation(
    uint32_t graph_id, uint64_t node_id, uint32_t source_graph_id,
    uint64_t source_node_id, absl::string_view fallback, KeySpace space) {
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  const GraphKey source_graph_key =
      MakeGraphKey(KeySpace::kTemplateGraph, source_graph_id);
  absl::MutexLock lock(mu_);
  GraphEntry* entry = FindLiveEntryLocked(graph_key);
  if (entry == nullptr) {
    return false;
  }
  const std::string* source = FindNodeLocked(source_graph_key, source_node_id);
  // Copy before inserting: the source may be in `entry->nodes` itself, which
  // the insertion may rehash.
  std::string annotation = source != nullptr ? *source : std::string(fallback);
  if (annotation.empty()) {
    return false;
  }
  auto [it, inserted] =
      entry->nodes.try_emplace(node_id, std::move(annotation));
  if (!inserted) {
    return false;
  }
  entry->bytes += it->second.size();
  return true;
}

bool CudaGraphAnnotationRegistry::VisitNodeAnnotation(
    uint32_t graph_id, uint64_t node_id,
    absl::FunctionRef<void(absl::string_view)> visitor, KeySpace space) const {
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  absl::ReaderMutexLock lock(mu_);
  const std::string* annotation = FindNodeLocked(graph_key, node_id);
  if (annotation == nullptr) {
    return false;
  }
  visitor(*annotation);
  return true;
}

void CudaGraphAnnotationRegistry::UnregisterGraph(uint32_t graph_id,
                                                  KeySpace space) {
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  // Declared before the lock, so destroyed after it is released: freeing a
  // large graph's strings must not lengthen the critical section.
  std::optional<GraphEntry> garbage;
  absl::MutexLock lock(mu_);
  auto it = graphs_.find(graph_key);
  if (it == graphs_.end() || it->second.destroyed) {
    // `ClearForTesting()` already removed the entry, or it is parked already.
    return;
  }
  GraphEntry& entry = it->second;
  // Without a hold nobody will resolve the entry's nodes any more, and an
  // entry without nodes has nothing to resolve.
  if (active_holds_ > 0 && !entry.nodes.empty()) {
    if (parked_bytes_ + entry.bytes <= max_parked_bytes_) {
      entry.destroyed = true;
      parked_bytes_ += entry.bytes;
      return;
    }
    ++park_dropped_in_session_;
  }
  garbage.emplace(std::move(entry));
  graphs_.erase(it);
  UpdateNumGraphsLocked();
}

CudaGraphAnnotationRegistry::HarvestHold
CudaGraphAnnotationRegistry::AcquireHarvestHold() {
  absl::MutexLock lock(mu_);
  if (active_holds_++ == 0) {
    park_dropped_in_session_ = 0;
  }
  return HarvestHold(this);
}

void CudaGraphAnnotationRegistry::ReleaseHarvestHold() {
  // Declared before the lock, so destroyed after it is released.
  std::vector<GraphEntry> purged;
  uint64_t dropped = 0;
  {
    absl::MutexLock lock(mu_);
    DCHECK_GT(active_holds_, 0);
    if (--active_holds_ > 0) {
      return;
    }
    // Entries are only moved out and erased, so iteration order doesn't
    // matter.
    // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
    for (auto it = graphs_.begin(); it != graphs_.end();) {
      if (it->second.destroyed) {
        purged.push_back(std::move(it->second));
        graphs_.erase(it++);
      } else {
        ++it;
      }
    }
    parked_bytes_ = 0;
    dropped = std::exchange(park_dropped_in_session_, 0);
    UpdateNumGraphsLocked();
  }
  // Warn once per session, on the thread that ends the harvest, rather than
  // once per dropped graph in `UnregisterGraph`.
  if (dropped > 0) {
    LOG(WARNING) << "Command buffer telemetry dropped the annotations of "
                 << dropped
                 << " graph(s) destroyed during the profiling session (parked "
                    "annotations are capped at "
                 << max_parked_bytes_
                 << " bytes). Their CUPTI events keep the command buffer "
                    "launch annotation.";
  }
}

bool CudaGraphAnnotationRegistry::empty() const {
  return num_graphs_.load(std::memory_order_acquire) == 0;
}

CudaGraphAnnotationRegistry::Stats CudaGraphAnnotationRegistry::GetStats()
    const {
  absl::ReaderMutexLock lock(mu_);
  Stats stats;
  stats.parked_bytes = parked_bytes_;
  stats.active_holds = active_holds_;
  // Order-independent integer sums, so iteration order doesn't matter.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [id, entry] : graphs_) {
    if (entry.destroyed) {
      ++stats.parked_graphs;
      stats.parked_nodes += entry.nodes.size();
    } else {
      ++stats.live_graphs;
      stats.live_nodes += entry.nodes.size();
      stats.live_bytes += entry.bytes;
    }
  }
  return stats;
}

std::vector<std::string>
CudaGraphAnnotationRegistry::GetNodeAnnotationsForTesting(
    KeySpace space) const {
  std::vector<std::string> annotations;
  absl::ReaderMutexLock lock(mu_);
  // The result is sorted below, so iteration order doesn't matter.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [key, entry] : graphs_) {
    if ((key >> 32) != static_cast<uint64_t>(space)) {
      continue;
    }
    // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
    for (const auto& [node_id, annotation] : entry.nodes) {
      annotations.push_back(annotation);
    }
  }
  std::sort(annotations.begin(), annotations.end());
  return annotations;
}

void CudaGraphAnnotationRegistry::ClearForTesting() {
  // Declared before the lock, so destroyed after it is released.
  GraphMap graphs;
  absl::MutexLock lock(mu_);
  graphs.swap(graphs_);
  parked_bytes_ = 0;
  park_dropped_in_session_ = 0;
  UpdateNumGraphsLocked();
}

CudaGraphAnnotationRegistry::GraphEntry*
CudaGraphAnnotationRegistry::FindLiveEntryLocked(GraphKey graph_key) {
  auto it = graphs_.find(graph_key);
  if (it == graphs_.end() || it->second.destroyed) {
    return nullptr;
  }
  return &it->second;
}

const std::string* CudaGraphAnnotationRegistry::FindNodeLocked(
    GraphKey graph_key, uint64_t node_id) const {
  auto graph_it = graphs_.find(graph_key);
  if (graph_it == graphs_.end()) {
    return nullptr;
  }
  auto node_it = graph_it->second.nodes.find(node_id);
  if (node_it == graph_it->second.nodes.end()) {
    return nullptr;
  }
  return &node_it->second;
}

void CudaGraphAnnotationRegistry::UpdateNumGraphsLocked() {
  num_graphs_.store(graphs_.size(), std::memory_order_release);
}

CudaGraphAnnotationRegistry::TelemetryStrategy
CudaGraphAnnotationRegistry::SelectTelemetryStrategy(
    SemanticVersion compiled_toolkit, SemanticVersion driver,
    SemanticVersion runtime) {
  SemanticVersion effective = driver < runtime ? driver : runtime;
  if (compiled_toolkit < SemanticVersion{13, 2, 0} ||
      effective < SemanticVersion{13, 2, 0}) {
    return TelemetryStrategy::kDisabled;
  }
  if (compiled_toolkit >= SemanticVersion{13, 4, 0} &&
      effective >= SemanticVersion{13, 4, 0}) {
    return TelemetryStrategy::kCuda134SourceGraph;
  }
  return TelemetryStrategy::kPre134ExecGraph;
}

}  // namespace stream_executor::gpu
