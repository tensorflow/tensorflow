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

#include "xla/backends/gpu/collectives/gpu_cliques.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/service/lockable.h"
#include "xla/service/rendezvous.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/hash.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

using DeviceRank = Collectives::DeviceRank;

//===----------------------------------------------------------------------===//
// GpuClique Acquire and Initialization Timeouts
//===----------------------------------------------------------------------===//

// We rely on rendezvous (all participating threads arriving to a rendezvous
// barrier at the same time) to guarantee that GPU communicators used in a way
// that does not lead to a deadlock. This itself can create new deadlocks if
// thread pools sized incorrectly. To prevent hard to debug deadlocks with WARN
// and terminate when detect that rendezvous runs for too long.

static absl::Duration WarnStuckTimeout() { return absl::Seconds(10); }

static absl::Duration TerminateTimeout() {
  static const int64_t terminate_timeout =
      xla::GetDebugOptionsFromFlags()
          .xla_gpu_nccl_termination_timeout_seconds();
  return (terminate_timeout >= 0) ? absl::Seconds(terminate_timeout)
                                  : absl::InfiniteDuration();
}

static bool TerminateOnCollectivesError() {
  return xla::GetDebugOptionsFromFlags().xla_gpu_nccl_terminate_on_error();
}

//===----------------------------------------------------------------------===//
// ProcessGpuCliques
//===----------------------------------------------------------------------===//

namespace {
// Container for initialized and ready to use local (in-process) GPU cliques.
struct ProcessGpuCliques {
  absl::Mutex mu;
  absl::node_hash_map<GpuCliqueKey, LockableGpuClique> map ABSL_GUARDED_BY(mu);
};
}  // namespace

// Returns process-local GPU cliques.
static ProcessGpuCliques& GetProcessGpuCliques() {
  static auto* cliques = new ProcessGpuCliques;
  return *cliques;
}

//===----------------------------------------------------------------------===//
// GpuClique Heart Beat Monitor
//===----------------------------------------------------------------------===//

// Runs an async error check for a `comm` and aborts it if it is in the
// error state. It will free resources that are allocated to a communicator
// and abort any uncompleted operations before destroying the communicator.
static absl::Status CheckComm(Communicator* comm) {
  absl::Status health = comm->HealthCheck();
  if (!health.ok()) {
    LOG(ERROR) << "Aborting communicator: " << comm
               << " due to error: " << health;
    TF_RETURN_IF_ERROR(comm->Abort());
  }
  return health;
}

// Runs async check on all communicators in a clique.
static void CheckClique(const GpuCliqueKey& clique_key,
                        LockableGpuClique& lockable_clique) {
  if (TerminateOnCollectivesError()) {
    absl::Status status = lockable_clique.HealthCheck();
    if (!status.ok()) {
      LOG(FATAL) << "Terminating process due to async GPU clique error: "
                 << status;
    }
    return;
  }
  if (LockableGpuClique::Lock clique = lockable_clique.TryAcquire()) {
    VLOG(5) << "Checking GPU clique " << clique_key.ToString()
            << " for async errors; num_communicators="
            << clique->num_communicators();
    clique->ForEachComm([](RankId rank, Communicator* comm) {
      if (auto status = CheckComm(comm); !status.ok()) LOG(ERROR) << status;
    });
  } else {
    VLOG(5) << "Skip checking in-use GPU clique " << clique_key.ToString();
  }
}

// TODO(ezhulenev): We need a mechanism to destroy whole clique when one of the
// communicators is aborted to be able to recover from errors.
static void GpuCliqueHeartBeatMonitorThread() {
  VLOG(5) << "Starting GPU cliques heart beat monitor";
  while (true) {
    absl::SleepFor(absl::Seconds(30));
    ProcessGpuCliques& cliques = GetProcessGpuCliques();
    absl::MutexLock lock(&cliques.mu);
    VLOG(5) << "Checking GPU communicators for errors"
            << "; num_cliques=" << cliques.map.size();
    for (auto& [clique_key, lockable_clique] : cliques.map) {
      CheckClique(clique_key, lockable_clique);
    }
  }
}

static void StartGpuCliqueHeartBeatMonitor() {
  static auto* monitor_thread = tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "gpu_clique_heart_beat_monitor",
      GpuCliqueHeartBeatMonitorThread);
  (void)monitor_thread;  // suppress unused variable warning
}

//===----------------------------------------------------------------------===//
// GpuClique Initialization
//===----------------------------------------------------------------------===//

// GpuClique initialization must be executed together by all participants, and
// we rely on rendezvous to guarantee that all ranks are ready to initialize
// GPU communicators. In general collective operations are expected to be
// executed concurrently by all participating ranks, and when some ranks do not
// join the operation it  leads to deadlocks. We use a combination of rendezvous
// and locking to guarantee that all collective operations in XLA have a well
// defined order and do not deadlock inside underlying collective communication
// library.

static auto DeviceRanksToString(absl::Span<const DeviceRank> ranks) {
  return absl::StrJoin(ranks, ",", [](std::string* str, auto& rank) {
    str->append(std::to_string(rank.rank.value()));
  });
}

// Returns true if peer access is possible between all devices in `ranks`. As a
// side effect, enables peer access even if it was not enabled before.
static absl::StatusOr<bool> EnablePeerAccess(
    const GpuCliqueKey& key, absl::Span<const DeviceRank> ranks) {
  if (key.devices().size() != ranks.size()) {
    // The clique is not local, so we can't enable peer access.
    return false;
  }

  std::vector<se::StreamExecutor*> devices;
  devices.reserve(ranks.size());
  for (int64_t i = 0; i < ranks.size(); ++i) {
    TF_ASSIGN_OR_RETURN(auto device, GpuCollectives::TryCast(ranks[i].device));
    devices.push_back(device->stream_executor());
  }

  for (int64_t i = 0; i < devices.size(); ++i) {
    for (int64_t j = 0; j < devices.size(); ++j) {
      // An attempt to enable peer access to itself will fail.
      if (i == j) continue;

      // To check if peer access is possible, we need to enable it and check
      // the result. OkStatus means that peer access is possible.
      auto status = devices[i]->EnablePeerAccessTo(devices[j]);
      if (!status.ok()) {
        return false;
      }
    }
  }

  return true;
}

// Joins a GpuClique initialization rendezvous for a `clique_key` and returns
// a lock that gives an access to initialized clique (access is shared between
// all participating ranks that own a shared pointer).
static absl::StatusOr<std::shared_ptr<LockableGpuClique::Lock>>
InitializeGpuClique(GpuCollectives* collectives, se::StreamExecutor* device,
                    RunId run_id, const GpuCliqueKey& clique_key,
                    const GpuCollectives::CliqueIdCallback& clique_id_callback,
                    int32_t num_local_participants, RankId rank,
                    const GpuCollectives::Config& config) {
  VLOG(3) << "Initialize GPU clique " << clique_key.ToString() << " rank #"
          << rank << "; num_local_participants=" << num_local_participants;

  // Start GPU clique heart beat monitor when create a first clique.
  StartGpuCliqueHeartBeatMonitor();

  using RendezvousArg = std::pair<DeviceRank, /*synchronized=*/bool>;

  // Check how many roots are needed to initialize the GpuClique
  static const int64_t nccl_init_rank_per_root_ratio =
      xla::GetDebugOptionsFromFlags()
          .xla_gpu_nccl_init_max_rank_per_root_ratio();
  int64_t nranks = clique_key.num_devices();
  int64_t nroots = nccl_init_rank_per_root_ratio != 0
                       ? CeilOfRatio(nranks, nccl_init_rank_per_root_ratio)
                       : 1;

  // Initializes a GpuClique for given device ranks and returns a lock that
  // gives access to clique communicators.
  auto initialize = [&](absl::Span<const RendezvousArg* const> args)
      -> absl::StatusOr<LockableGpuClique::Lock> {
    tsl::profiler::TraceMe trace("InitializeGpuClique");

    CliqueIds clique_ids;
    const auto& subkeys = clique_key.GetSubKeys(nroots);
    for (const auto& subkey : subkeys) {
      VLOG(3) << absl::StreamFormat(
          "Get CliqueId for sub clique key %s; nroots=%lld", subkey.ToString(),
          nroots);
      TF_ASSIGN_OR_RETURN(auto clique_id, clique_id_callback(subkey));
      clique_ids.Add(clique_id);
    }

    // Check that all ranks successfully synchronized device activity before
    // trying to instantiate GPU communicators.
    for (const RendezvousArg* arg : args) {
      if (auto& [device_rank, synchronized] = *arg; !synchronized) {
        return Internal(
            "Failed to synchronize device activity on rank %d. Do not attempt "
            "to initialize GPU clique.",
            device_rank.rank.value());
      }
    }

    std::vector<DeviceRank> ranks;
    ranks.reserve(args.size());
    for (auto* arg : args) ranks.emplace_back(arg->first);

    // Sort device ranks, mainly to get more readable logs below.
    absl::c_sort(ranks, [](auto& a, auto& b) { return a.rank < b.rank; });

    // Check if peer access is possible between all devices in the clique.
    TF_ASSIGN_OR_RETURN(bool peer_access_enabled,
                        EnablePeerAccess(clique_key, ranks));

    VLOG(3) << absl::StreamFormat(
        "Create GPU communicators for clique %s; ranks=[%s]; "
        "nroots=%lld; fingerprint(id)=%d, peer_access_enabled=%d",
        clique_key.ToString(), DeviceRanksToString(ranks), nroots,
        clique_ids.fingerprint(), peer_access_enabled);

    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<Communicator>> created_comms,
        collectives->CreateCommunicators(clique_key, clique_ids, ranks,
                                         config));

    absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
    for (size_t i = 0; i < ranks.size(); ++i) {
      comms[ranks[i].rank] = std::move(created_comms[i]);
    }

    VLOG(3) << absl::StreamFormat(
        "Created GPU communicators for clique %s; ranks=[%s]; "
        "nroots=%lld; fingerprint(id)=%d, peer_access_enabled=%d",
        clique_key.ToString(), DeviceRanksToString(ranks), nroots,
        clique_ids.fingerprint(), peer_access_enabled);

    ProcessGpuCliques& cliques = GetProcessGpuCliques();
    absl::MutexLock lock(&cliques.mu);

    // Create a new clique with given clique key and communicators.
    auto emplaced =
        cliques.map.try_emplace(clique_key, clique_key, clique_ids,
                                std::move(comms), peer_access_enabled);

    // We can have a race to create a clique for a given key, the winner
    // inserts it into a map and the looser destroys all communicators.
    if (!emplaced.second) {
      VLOG(3) << "Clique already exists: "
              << emplaced.first->second.DebugString();
    } else {
      VLOG(3) << "Created new clique: " << emplaced.first->second.DebugString();
    }

    return emplaced.first->second.Acquire();
  };

  // We include `run_id` to a rendezvous key to make sure that multiple
  // concurrent initializations will not join the same rendezvous. The winner
  // will update cliques state, and others will destroy unused communicators.
  auto rendezvous_key = std::make_tuple(run_id, clique_key);
  auto initialization_rendezvous_name =
      absl::StrFormat("initialize clique for rank %d; clique=%s; run_id=%d",
                      rank.value(), clique_key.ToString(), run_id.ToInt());

  GpuCollectives::Device gpu_device(device);
  GpuCollectives::DeviceRank device_rank = {&gpu_device, rank};
  bool synchronized = device->SynchronizeAllActivity();

  // We choose not to exit early on failed synchronization, because it will lead
  // to a deadlock, as not all participants will arrive to a rendezvous point,
  // instead we check synchronization result in the initialization callback.
  //
  // Unfortunately we can't share synchronization result across different
  // processes, so we still might end up in a deadlock situation when some
  // processes are not able to synchronize device activity.
  RendezvousArg rendezvous_arg = std::make_pair(device_rank, synchronized);

  return Rendezvous<absl::StatusOr<LockableGpuClique::Lock>>(
      initialization_rendezvous_name, rendezvous_key, rendezvous_arg,
      num_local_participants, initialize, WarnStuckTimeout(),
      TerminateTimeout());
}

// Computes a unique GPU communicator split color from a clique key. We use a
// deterministic hash function to guarantee that all participating processes get
// the same color value for a clique.
static int32_t GetCommSplitColor(const GpuCliqueKey& clique_key) {
  std::vector<int64_t> global_device_ids;
  global_device_ids.reserve(clique_key.devices().size());

  for (GlobalDeviceId id : clique_key.devices()) {
    global_device_ids.push_back(id.value());
  }

  return abs(static_cast<int32_t>(
      tsl::Hash32(reinterpret_cast<char*>(global_device_ids.data()),
                  sizeof(int64_t) * global_device_ids.size(), 0)));
}

// Joins a GpuClique initialization rendezvous for a `clique_key` and returns
// a lock that gives an access to clique created by splitting already acquired
// `parent_clique` clique (access is shared between all participating ranks that
// own a shared pointer).
static absl::StatusOr<std::shared_ptr<LockableGpuClique::Lock>>
InitializeGpuClique(GpuCollectives* collectives, se::StreamExecutor* device,
                    RunId run_id, const GpuCliqueKey& clique_key,
                    std::shared_ptr<LockableGpuClique::Lock> parent_clique,
                    int32_t num_local_participants, RankId rank,
                    const GpuCollectives::Config& config) {
  // Find our rank in the parent clique.
  const GpuCliqueKey& parent_clique_key = (*parent_clique)->key();
  RankId parent_rank =
      *parent_clique_key.rank(clique_key.devices()[rank.value()]);

  VLOG(3) << "Initialize GPU clique " << clique_key.ToString() << " rank #"
          << rank << " by splitting rank #" << parent_rank.value()
          << " in parent clique " << parent_clique_key.ToString()
          << "; num_local_participants=" << num_local_participants;

  using RankPair = std::pair<RankId, DeviceRank>;
  GpuCollectives::Device gpu_device(device);
  GpuCollectives::DeviceRank device_rank = {&gpu_device, rank};
  RankPair rank_pair = {parent_rank, device_rank};

  // Current approach for communicator splitting works because of XLAs SPMD
  // programming model where all collective operations have replica groups that
  // include all ranks. This property guarantees that we'll split each
  // communicator exactly once with a unique color computed from rank mapping
  // and each communicator in the parent clique will become a member of exactly
  // one new clique. Clique splitting happens concurrently for multiple
  // non-overlapping clique and this guarantees forward progress even with
  // implicit synchronization inside GPU collectives (i.e. NCCL).

  // Initializes a GpuClique for given device ranks and returns a lock that
  // gives access to clique communicators.
  auto split = [&](absl::Span<const RankPair* const> rank_pairs)
      -> absl::StatusOr<LockableGpuClique::Lock> {
    tsl::profiler::TraceMe trace("SplitGpuClique");

    // Collect mapping from ranks in parent clique to ranks in a new clique.
    absl::btree_map<RankId, RankId> rank_mapping;
    for (auto* rank_pair : rank_pairs) {
      rank_mapping[rank_pair->first] = rank_pair->second.rank;
    }

    auto rank_mapping_formatter = [](std::string* str, auto mapping) {
      absl::StrAppend(str, mapping.first.value(), "->", mapping.second.value());
    };

    // Collect parent communicators we'll be splitting from and keys for
    // creating new communicators.
    std::vector<Communicator*> parent_comms;
    std::vector<RankId> keys;

    for (auto& [parent_rank, split_rank] : rank_mapping) {
      auto parent_comm = (*parent_clique)->comm(parent_rank);
      if (!parent_comm.has_value()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Parent clique %s does not have a communicator for rank %d",
            parent_clique_key.ToString(), parent_rank.value()));
      }

      parent_comms.push_back(*parent_comm);
      keys.push_back(split_rank);
    }

    // Get a globally consistent color value for newly created clique.
    int32_t color = GetCommSplitColor(clique_key);

    bool peer_access_enabled = false;
    if ((*parent_clique)->IsLocal()) {
      // The parent clique is local, we can be sure that peer access was already
      // enabled.
      peer_access_enabled = (*parent_clique)->peer_access_enabled();
    } else {
      // The parent clique is not local, but this clique can be local. We need
      // to check if peer access is possible between all devices in this clique.
      std::vector<DeviceRank> ranks;
      ranks.reserve(rank_pairs.size());
      for (auto& rank_pair : rank_pairs) {
        ranks.emplace_back(rank_pair->second);
      }
      TF_ASSIGN_OR_RETURN(peer_access_enabled,
                          EnablePeerAccess(clique_key, ranks));
    }

    VLOG(3) << absl::StreamFormat(
        "Create GPU communicators for clique %s; parent=%s; color=%d; "
        "peer_access_enabled=%d; rank_mapping=[%s]",
        clique_key.ToString(), parent_clique_key.ToString(), color,
        peer_access_enabled,
        absl::StrJoin(rank_mapping, ",", rank_mapping_formatter));

    TF_ASSIGN_OR_RETURN(
        auto splitted_comms,
        collectives->SplitCommunicators(parent_comms, color, keys, config));

    absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
    for (size_t i = 0; i < splitted_comms.size(); ++i) {
      comms[keys[i]] = std::move(splitted_comms[i]);
    }

    VLOG(3) << absl::StreamFormat(
        "Created GPU communicators for clique %s; parent=%s; color=%d; "
        "peer_access_enabled=%d; "
        "rank_mapping=[%s]",
        clique_key.ToString(), parent_clique_key.ToString(), color,
        peer_access_enabled,
        absl::StrJoin(rank_mapping, ",", rank_mapping_formatter));

    ProcessGpuCliques& cliques = GetProcessGpuCliques();
    absl::MutexLock lock(&cliques.mu);

    // Create a new clique with given clique key and communicators.
    auto emplaced =
        cliques.map.try_emplace(clique_key, clique_key, std::nullopt,
                                std::move(comms), peer_access_enabled);

    // We can have a race to create a clique for a given key, the winner
    // inserts it into a map and the looser destroys all communicators.
    if (!emplaced.second) {
      VLOG(3) << "Clique already exists: "
              << emplaced.first->second.DebugString();
    } else {
      VLOG(3) << "Created new clique: " << emplaced.first->second.DebugString();
    }

    return emplaced.first->second.Acquire();
  };

  // We include `run_id` to a rendezvous key to make sure that multiple
  // concurrent initializations will not join the same rendezvous. The winner
  // will update cliques state, and others will destroy unused communicators.
  auto rendezvous_key = std::make_tuple(run_id, clique_key, parent_clique_key);
  auto initialization_rendezvous_name = absl::StrFormat(
      "initialize clique for rank %d; clique=%s; run_id=%d; parent=%s",
      rank.value(), clique_key.ToString(), run_id.ToInt(),
      parent_clique_key.ToString());

  return Rendezvous<absl::StatusOr<LockableGpuClique::Lock>>(
      initialization_rendezvous_name, rendezvous_key, rank_pair,
      num_local_participants, split, WarnStuckTimeout(), TerminateTimeout());
}

//===----------------------------------------------------------------------===//

absl::StatusOr<std::shared_ptr<LockableGpuClique::Lock>> AcquireGpuClique(
    GpuCollectives* collectives, se::StreamExecutor* device, RunId run_id,
    const GpuCliqueKey& clique_key,
    const GpuCollectives::CliqueIdCallback& clique_id_callback, RankId rank,
    size_t num_local_participants, const AcquiredCliquesMap& acquired_cliques,
    int64_t max_nchannels) {
  VLOG(2) << "Acquire GPU clique " << clique_key.ToString() << "; run"
          << run_id.ToString() << "; rank " << rank
          << "; num_local_participants=" << num_local_participants
          << "; acquired_cliques=" << acquired_cliques.size()
          << "; max_nchannels=" << max_nchannels;

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "AcquireGpuClique", {{"rank", rank.value()},
                             {"num_local_participants", num_local_participants},
                             {"clique_key", clique_key.ToString()}});
  });

  // Get the clique lock via the rendezvous to guarantee that all clique
  // members participate in XLA run.
  auto rendezvous_key = std::make_tuple(run_id, clique_key);
  auto rendezvous_name =
      absl::StrFormat("acquire clique for rank %d; clique=%s; run_id=%d",
                      rank.value(), clique_key.ToString(), run_id.ToInt());

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<LockableGpuClique::Lock> clique,
      Rendezvous<absl::StatusOr<LockableGpuClique::Lock>>(
          rendezvous_name, rendezvous_key, num_local_participants,
          [&] {
            tsl::profiler::TraceMe trace("LockGpuClique");
            ProcessGpuCliques& cliques = GetProcessGpuCliques();

            // Returns nullptr if we do not have a clique for `clique_key`.
            auto lockable_clique = [&]() -> LockableGpuClique* {
              absl::MutexLock lock(&cliques.mu);
              auto it = cliques.map.find(clique_key);
              return it == cliques.map.end() ? nullptr : &it->second;
            }();

            return lockable_clique ? lockable_clique->Acquire()
                                   : LockableGpuClique::Lock();
          },
          WarnStuckTimeout(), TerminateTimeout()));

  // If lock is not null return it to the caller.
  if (*clique) return clique;

  // Maybe find if we acquired a clique with communicators that we can split.
  static const int64_t enable_nccl_comm_splitting =
      xla::GetDebugOptionsFromFlags().xla_gpu_enable_nccl_comm_splitting();

  // We enable resource sharing between parent and split communicators by
  // default because that's the only reason why we use comm splitting.
  GpuCollectives::Config config;
  config.split_share = true;
  config.max_nchannels = max_nchannels;

  if (enable_nccl_comm_splitting) {
    for (auto& [acquired_clique_key, acquired_clique] : acquired_cliques) {
      if (clique_key.IsSubsetOf(acquired_clique_key)) {
        return InitializeGpuClique(collectives, device, run_id, clique_key,
                                   acquired_clique, num_local_participants,
                                   rank, config);
      }
    }
  }

  // If we can't split any of the acquired cliques, create a new one.
  return InitializeGpuClique(collectives, device, run_id, clique_key,
                             clique_id_callback, num_local_participants, rank,
                             config);
}

}  // namespace xla::gpu
