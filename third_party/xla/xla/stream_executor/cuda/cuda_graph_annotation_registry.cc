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
#include <initializer_list>
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

CudaGraphAnnotationRegistry::TrackResult
CudaGraphAnnotationRegistry::TrackGraph(uint32_t graph_id, TrackMode mode,
                                        KeySpace space) {
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  // Declared before the lock, so destroyed after it is released.
  Garbage garbage;
  absl::MutexLock lock(mu_);
  auto [it, inserted] = live_.try_emplace(graph_key);
  if (!inserted) {
    if (mode == TrackMode::kGetOrCreate) {
      return {it->second.token, /*newly_tracked=*/false};
    }
    // The caller just created this graph, so the entry belongs to a destroyed
    // graph with the same ID whose cleanup hasn't run yet. Once it runs, its
    // token no longer matches.
    RetireLocked(it, garbage);
    // `RetireLocked` erased `it`, so insert a fresh entry for the new graph.
    it = live_.try_emplace(graph_key).first;
  }
  const Token token = next_token_++;
  it->second.token = token;
  UpdateNumGraphsLocked();
  return {token, /*newly_tracked=*/true};
}

bool CudaGraphAnnotationRegistry::IsGraphTracked(uint32_t graph_id,
                                                 KeySpace space) const {
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  absl::ReaderMutexLock lock(mu_);
  return live_.contains(graph_key);
}

bool CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
    uint32_t graph_id, Token token, uint64_t node_id,
    absl::string_view annotation, KeySpace space) {
  if (annotation.empty()) {
    return false;
  }
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  absl::MutexLock lock(mu_);
  GraphEntry* entry = FindLiveEntryLocked(graph_key, token);
  if (entry == nullptr) {
    return false;
  }
  if (!entry->nodes.try_emplace(node_id, annotation).second) {
    return false;
  }
  AddLiveNodeLocked(*entry, annotation.size());
  return true;
}

bool CudaGraphAnnotationRegistry::CopyNodeAnnotation(
    uint32_t graph_id, Token token, uint64_t node_id, uint32_t source_graph_id,
    uint64_t source_node_id, absl::string_view fallback, KeySpace space) {
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  const GraphKey source_graph_key =
      MakeGraphKey(KeySpace::kTemplateGraph, source_graph_id);
  absl::MutexLock lock(mu_);
  GraphEntry* entry = FindLiveEntryLocked(graph_key, token);
  if (entry == nullptr || entry->nodes.contains(node_id)) {
    return false;
  }
  const std::string* source = FindNodeLocked(source_graph_key, source_node_id);
  // Copy before inserting: the source may be in `entry->nodes` itself, which
  // the insertion may rehash.
  std::string annotation = source != nullptr ? *source : std::string(fallback);
  if (annotation.empty()) {
    return false;
  }
  const size_t size = annotation.size();
  entry->nodes.emplace(node_id, std::move(annotation));
  AddLiveNodeLocked(*entry, size);
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
                                                  Token token, KeySpace space) {
  const GraphKey graph_key = MakeGraphKey(space, graph_id);
  // Declared before the lock, so destroyed after it is released: freeing a
  // large graph's strings must not lengthen the driver thread's critical
  // section.
  Garbage garbage;
  absl::MutexLock lock(mu_);
  auto it = live_.find(graph_key);
  if (it == live_.end() || it->second.token != token) {
    // `TrackGraph(kNewGraph)` or `Clear()` already removed this graph's entry,
    // and the ID may now belong to a newer graph.
    return;
  }
  RetireLocked(it, garbage);
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
  GraphMap purged;
  uint64_t dropped = 0;
  {
    absl::MutexLock lock(mu_);
    DCHECK_GT(active_holds_, 0);
    if (--active_holds_ > 0) {
      return;
    }
    purged.swap(parked_);
    parked_bytes_ = 0;
    dropped = std::exchange(park_dropped_in_session_, 0);
    UpdateNumGraphsLocked();
  }
  // Warn here, on the thread that ends the harvest: `UnregisterGraph`, which
  // drops the entries, runs on CUDA driver threads and must not log.
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
  stats.live_graphs = live_.size();
  stats.parked_graphs = parked_.size();
  stats.parked_bytes = parked_bytes_;
  stats.active_holds = active_holds_;
  // Order-independent integer sums, so iteration order doesn't matter.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [id, entry] : live_) {
    stats.live_nodes += entry.nodes.size();
    stats.live_bytes += entry.bytes;
  }
  // Order-independent integer sums, so iteration order doesn't matter.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [id, entry] : parked_) {
    stats.parked_nodes += entry.nodes.size();
  }
  return stats;
}

std::vector<std::string>
CudaGraphAnnotationRegistry::GetNodeAnnotationsForTesting(
    KeySpace space) const {
  std::vector<std::string> annotations;
  absl::ReaderMutexLock lock(mu_);
  for (const GraphMap* graphs : {&live_, &parked_}) {
    // The result is sorted below, so iteration order doesn't matter.
    // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
    for (const auto& [key, entry] : *graphs) {
      if ((key >> 32) != static_cast<uint64_t>(space)) {
        continue;
      }
      // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
      for (const auto& [node_id, annotation] : entry.nodes) {
        annotations.push_back(annotation);
      }
    }
  }
  std::sort(annotations.begin(), annotations.end());
  return annotations;
}

void CudaGraphAnnotationRegistry::ClearForTesting() {
  // Declared before the lock, so destroyed after it is released.
  GraphMap live;
  GraphMap parked;
  absl::MutexLock lock(mu_);
  live.swap(live_);
  parked.swap(parked_);
  parked_bytes_ = 0;
  park_dropped_in_session_ = 0;
  // `next_token_` keeps counting, so that the cleanups of graphs tracked before
  // the `Clear()` stay stale.
  UpdateNumGraphsLocked();
}

CudaGraphAnnotationRegistry::GraphEntry*
CudaGraphAnnotationRegistry::FindLiveEntryLocked(GraphKey graph_key,
                                                 Token token) {
  auto it = live_.find(graph_key);
  if (it == live_.end() || it->second.token != token) {
    return nullptr;
  }
  return &it->second;
}

const std::string* CudaGraphAnnotationRegistry::FindNodeLocked(
    GraphKey graph_key, uint64_t node_id) const {
  // Live entries first: a parked entry with the same key belongs to an older
  // graph.
  for (const GraphMap* graphs : {&live_, &parked_}) {
    auto graph_it = graphs->find(graph_key);
    if (graph_it == graphs->end()) {
      continue;
    }
    auto node_it = graph_it->second.nodes.find(node_id);
    if (node_it != graph_it->second.nodes.end()) {
      return &node_it->second;
    }
  }
  return nullptr;
}

void CudaGraphAnnotationRegistry::AddLiveNodeLocked(GraphEntry& entry,
                                                    size_t size) {
  entry.bytes += size;
}

void CudaGraphAnnotationRegistry::RetireLocked(GraphMap::iterator it,
                                               Garbage& garbage) {
  const GraphKey graph_key = it->first;
  GraphEntry entry = std::move(it->second);
  live_.erase(it);
  // Without a hold nobody will resolve the entry's nodes any more, and an
  // entry without nodes has nothing to resolve.
  if (active_holds_ == 0 || entry.nodes.empty()) {
    garbage.push_back(std::move(entry));
    return;
  }
  // Check the cap after accounting for any older parked entry with the same
  // key that this entry would replace. If the new entry still exceeds the cap,
  // drop the new entry and keep the older parked entry intact.
  auto parked_it = parked_.find(graph_key);
  const size_t replaced_bytes =
      parked_it != parked_.end() ? parked_it->second.bytes : 0;
  if (parked_bytes_ - replaced_bytes + entry.bytes > max_parked_bytes_) {
    ++park_dropped_in_session_;
    garbage.push_back(std::move(entry));
    return;
  }
  // Read before `entry` is moved from below.
  const size_t entry_bytes = entry.bytes;
  if (parked_it != parked_.end()) {
    // Replace the older parked entry in place; it goes to `garbage`, so its
    // strings are freed after `mu_` is released.
    parked_bytes_ -= replaced_bytes;
    ++park_dropped_in_session_;
    garbage.push_back(std::move(parked_it->second));
    parked_it->second = std::move(entry);
  } else {
    parked_.try_emplace(graph_key, std::move(entry));
  }
  parked_bytes_ += entry_bytes;
}

void CudaGraphAnnotationRegistry::UpdateNumGraphsLocked() {
  num_graphs_.store(live_.size() + parked_.size(), std::memory_order_release);
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
