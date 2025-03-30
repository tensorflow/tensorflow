/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/resource_requests.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla {
namespace gpu {

namespace {

// A container for per-process persistent cliques.
struct PersistentCliquesMap {
  absl::Mutex mutex;
  AcquiredCliquesMap cliques_map ABSL_GUARDED_BY(mutex);
};

static PersistentCliquesMap& GetPersistentCliquesMap() {
  static auto* persistent_cliques = new PersistentCliquesMap();
  return *persistent_cliques;
}
}  // namespace

absl::Status ResourceRequests::AddClique(const GpuCliqueKey& clique_key) {
  VLOG(5) << "Add collective clique request: " << clique_key.ToString();

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // clique requests simply based on the number of already recorded requests.
  int64_t id = cliques_.size();
  cliques_.try_emplace(clique_key, CliqueRequest{clique_key, id});
  return absl::OkStatus();
}

absl::StatusOr<Thunk::CollectiveCliques>
ResourceRequests::AcquireCollectiveCliques(
    const Thunk::CollectiveExecuteParams& params, bool use_persistent_cliques) {
  if (cliques_.empty()) return Thunk::CollectiveCliques();

  VLOG(2) << "Acquire " << cliques_.size()
          << " collective cliques for global device id "
          << params.global_device_id.value()
          << "; run_id=" << params.run_id.ToInt()
          << "; max number of channels for collectives "
          << params.collective_max_nchannels
          << "; max number of channels for p2p " << params.p2p_max_nchannels
          << "; use_persistent_cliques=" << use_persistent_cliques;

  std::vector<CliqueRequest> ordered_cliques = GetOrderedCliqueRequests();
  for (size_t i = 0; i < ordered_cliques.size(); ++i) {
    const CliqueRequest& r = ordered_cliques[i];
    VLOG(2) << "  clique #" << i << " (for global device id "
            << params.global_device_id.value() << ")"
            << ": num_local_participants=" << r.key.num_local_participants()
            << "; id=" << r.id << "; key=" << r.key.ToString();
  }

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "AcquireCollectiveCliques",
        {{"num_cliques", cliques_.size()},
         {"use_persistent_cliques", use_persistent_cliques}});
  });

  auto start_micros = tsl::Env::Default()->NowMicros();

  AcquiredCliquesMap cliques_map;
  int32_t num_transient_cliques = 0;

  for (const CliqueRequest& r : ordered_cliques) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);

    if (!rank.has_value()) {
      return absl::InternalError(absl::StrCat(
          "Can't find global device id ", params.global_device_id.value(),
          " in clique key ", r.key.ToString()));
    }

    TF_ASSIGN_OR_RETURN(const CliqueIdCallback* clique_id_callback,
                        params.collectives->GetCliqueIdCallback(
                            params.nccl_clique_id_callback, r.key.is_local()));

    int64_t max_channels = r.key.stream_kind() == AsyncStreamKind::kCollective
                               ? params.collective_max_nchannels
                               : params.p2p_max_nchannels;

    // Check if we have a persistent clique for this key.
    if (use_persistent_cliques) {
      auto& pc = GetPersistentCliquesMap();
      absl::MutexLock lock(&pc.mutex);

      if (auto it = pc.cliques_map.find(r.key); it != pc.cliques_map.end()) {
        VLOG(2) << "Found persistent clique for key " << r.key.ToString();
        cliques_map[r.key] = it->second;
        continue;
      }
    }

    // If we don't have a persistent clique we have to acquire a transient
    // one.
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<LockableGpuClique::Lock> clique,
        AcquireGpuClique(params.collectives, params.executor, params.run_id,
                         r.key, *clique_id_callback, *rank, cliques_map,
                         max_channels));
    ++num_transient_cliques;

    // Take a copy of the clique lock, so that we can reuse it. This is
    // potentially unsafe in the case when we have multiple racing executions
    // of XLA, as we might observe partial state and some of the replicas will
    // use persistent clique, and others will try to acquire a new one.
    //
    // However given that persistent cliques is an unsafe escape hatch, any
    // racing execution together with persistent cliques will lead to
    // deadlocks anyway, so we don't bother to fix this. If anyone is doing
    // it, it's 100% their fault and they will suffer.
    if (use_persistent_cliques) {
      auto& pc = GetPersistentCliquesMap();
      absl::MutexLock lock(&pc.mutex);
      pc.cliques_map[r.key] = clique;
    }

    cliques_map[r.key] = std::move(clique);
  }

  auto end_micros = tsl::Env::Default()->NowMicros();
  VLOG(2) << "Acquired " << cliques_map.size()
          << " collective cliques for global device id "
          << params.global_device_id.value() << " in "
          << (end_micros - start_micros) << " μs"
          << "; run_id=" << params.run_id.ToInt()
          << "; num_transient_cliques=" << num_transient_cliques;

  return Thunk::CollectiveCliques(std::move(cliques_map),
                                  num_transient_cliques);
}

std::vector<ResourceRequests::CliqueRequest>
ResourceRequests::GetOrderedCliqueRequests() {
  std::vector<CliqueRequest> cliques;
  cliques.reserve(cliques_.size());
  for (const auto& [_, request] : cliques_) cliques.push_back(request);

  absl::c_sort(cliques, [](const CliqueRequest& a, const CliqueRequest& b) {
    // Acquire larger cliques first to be able to split them later.
    if (a.key.devices().size() > b.key.devices().size()) return true;
    if (b.key.devices().size() > a.key.devices().size()) return false;

    // If cliques have the same size prefer cliques with smaller stream id.
    if (a.key.stream_id().value() < b.key.stream_id().value()) return true;
    if (b.key.stream_id().value() < a.key.stream_id().value()) return false;

    // Prefer cliques with smaller id (comes earlier in execution order).
    return a.id < b.id;
  });

  return cliques;
}
}  // namespace gpu
}  // namespace xla
