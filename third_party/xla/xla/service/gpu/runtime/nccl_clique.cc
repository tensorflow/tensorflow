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

#include "xla/service/gpu/runtime/nccl_clique.h"

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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/lockable.h"
#include "xla/service/rendezvous.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/hash.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// NcclCliqueIdCallback
//===----------------------------------------------------------------------===//

bool IsGlobalNcclConfig() {
  static const char* const nccl_comm_id = std::getenv("NCCL_COMM_ID");
  return nccl_comm_id != nullptr;
}

absl::StatusOr<const NcclCliqueIdCallback*> GetNcclCliqueIdCallback(
    const NcclCliqueIdCallback* clique_id_callback, bool is_local) {
  if (clique_id_callback != nullptr) return clique_id_callback;

  TF_RET_CHECK(is_local || IsGlobalNcclConfig())
      << "If non-local devices are taking part of a collective API on "
         "GPU, the nccl_clique_id_callback must be provided by the client.";

  static auto* local_callback = new NcclCliqueIdCallback(
      [](const NcclCliqueKey&) { return NcclApi::Default()->GetUniqueId(); });
  return local_callback;
}

//===----------------------------------------------------------------------===//
// NcclClique Acquire and Initialization Timeouts
//===----------------------------------------------------------------------===//

// We rely on rendezvous (all participating threads arriving to a rendezvous
// barrier at the same time) to guarantee that NCCL communicators used in a way
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

static bool TerminateOnNcclError() {
  return xla::GetDebugOptionsFromFlags().xla_gpu_nccl_terminate_on_error();
}

//===----------------------------------------------------------------------===//
// NcclClique
//===----------------------------------------------------------------------===//

NcclCliqueCommunicators::NcclCliqueCommunicators(
    NcclCliqueKey clique_key, std::optional<NcclCliqueId> clique_id,
    absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators)
    : clique_key_(std::move(clique_key)),
      clique_id_(std::move(clique_id)),
      communicators_(std::move(communicators)) {}

std::optional<Communicator*> NcclCliqueCommunicators::comm(RankId rank) {
  if (auto it = communicators_.find(rank); it != communicators_.end()) {
    return it->second.get();
  }
  return std::nullopt;
}

bool NcclCliqueCommunicators::IsLocal() const {
  return communicators_.size() == clique_key_.devices().size();
}

void NcclCliqueCommunicators::ForEachComm(
    absl::FunctionRef<void(RankId, Communicator*)> fn) {
  for (auto& [rank, comm] : communicators_) {
    fn(rank, comm.get());
  }
}

std::string NcclCliqueCommunicators::DebugString() const {
  std::string out = absl::StrFormat(
      "clique_key: %s; fingerprint(id): %d; size: %d; communicators: ",
      clique_key_.ToString(),
      clique_id_.has_value() ? clique_id_->fingerprint() : 0,
      communicators_.size());
  int32_t cnt = 0;
  for (const auto& [rank, comm] : communicators_) {
    if (cnt++) absl::StrAppend(&out, ", ");
    absl::StrAppendFormat(&out, "[rank=%d, comm=%p]", rank.value(), comm.get());
  }
  return out;
}

std::string NcclClique::DebugString() const {
  return absl::StrFormat("NcclClique: %s", value().DebugString());
}

namespace {
// Container for initialized and ready to use local (in-process) NCCL cliques.
struct NcclCliques {
  absl::Mutex mu;
  absl::node_hash_map<NcclCliqueKey, NcclClique> map ABSL_GUARDED_BY(mu);
};
}  // namespace

// Returns local (in-process) NcclCliques container.
static NcclCliques& GetNcclCliques() {
  static auto* cliques = new NcclCliques;
  return *cliques;
}

//===----------------------------------------------------------------------===//
// NcclClique Heart Beat Monitor
//===----------------------------------------------------------------------===//

// Runs an async error check for a `comm` and aborts it if it is in the
// error state. It will free resources that are allocated to a communicator
// and abort any uncompleted operations before destroying the communicator.
static absl::Status CheckComm(Communicator* comm) {
  absl::Status async_err = NcclApi::Default()->CommGetAsyncError(comm);
  if (!async_err.ok()) {
    LOG(ERROR) << "Aborting communicator: " << comm
               << " due to async NCCL error: " << async_err;
    TF_RETURN_IF_ERROR(NcclApi::Default()->CommAbort(comm));
  }
  return async_err;
}

// Runs async check on all communicators in a clique.
static void CheckClique(const NcclCliqueKey& clique_key,
                        NcclClique& lockable_clique) {
  if (TerminateOnNcclError()) {
    absl::Status status = lockable_clique.CheckAsyncErrors();
    if (!status.ok()) {
      LOG(FATAL) << "Terminating process due to async NCCL error: " << status;
    }
    return;
  }
  if (NcclClique::Lock clique = lockable_clique.TryAcquire()) {
    VLOG(5) << "Checking NCCL clique " << clique_key.ToString()
            << " for async errors; num_communicators="
            << clique->num_communicators();
    clique->ForEachComm([](RankId rank, Communicator* comm) {
      if (auto status = CheckComm(comm); !status.ok()) LOG(ERROR) << status;
    });
  } else {
    VLOG(5) << "Skip checking in-use NCCL clique " << clique_key.ToString();
  }
}

// TODO(ezhulenev): We need a mechanism to destroy whole clique when one of the
// communicators is aborted to be able to recover from errors.
static void NcclCliqueHeartBeatMonitorThread() {
  VLOG(5) << "Starting NCCL clique heart beat monitor";
  while (true) {
    absl::SleepFor(absl::Seconds(30));
    NcclCliques& cliques = GetNcclCliques();
    absl::MutexLock lock(&cliques.mu);
    VLOG(5) << "Checking NCCL communicators for async errors"
            << "; num_cliques=" << cliques.map.size();
    for (auto& [clique_key, lockable_clique] : cliques.map) {
      CheckClique(clique_key, lockable_clique);
    }
  }
}

static void StartNcclCliqueHeartBeatMonitor() {
  static auto* monitor_thread = tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "nccl_clique_heart_beat_monitor",
      NcclCliqueHeartBeatMonitorThread);
  (void)monitor_thread;  // suppress unused variable warning
}

//===----------------------------------------------------------------------===//
// NcclClique Initialization
//===----------------------------------------------------------------------===//

// NcclClique initialization must be executed together by all participants, and
// we rely on rendezvous to guarantee that all ranks are ready to initialize
// NCCL communicators. In general collective operations are expected to be
// executed concurrently by all participating ranks, and when some ranks do not
// join the operation it  leads to deadlocks. We use a combination of rendezvous
// and locking to guarantee that all collective operations in XLA have a well
// defined order and do not deadlock inside underlying collective communication
// library.

static auto DeviceRanksToString(absl::Span<const NcclApi::DeviceRank> ranks) {
  return absl::StrJoin(ranks, ",", [](std::string* str, auto& rank) {
    str->append(std::to_string(rank.rank.value()));
  });
}

// Joins a NcclClique initialization rendezvous for a `clique_key` and returns
// a lock that gives an access to initialized clique (access is shared between
// all participating ranks that own a shared pointer).
static absl::StatusOr<std::shared_ptr<NcclClique::Lock>> InitializeNcclClique(
    se::StreamExecutor* device, RunId run_id, NcclCliqueKey clique_key,
    const NcclCliqueIdCallback& clique_id_callback,
    int32_t num_local_participants, RankId rank, NcclApi::Config& config) {
  int nranks = clique_key.devices().size();
  VLOG(3) << "Initialize NCCL clique " << clique_key.ToString() << " rank #"
          << rank << "; num_local_participants=" << num_local_participants;

  // Start NCCL clique heart beat monitor when create a first clique.
  StartNcclCliqueHeartBeatMonitor();

  using RendezvousArg = std::pair<NcclApi::DeviceRank, /*synchronized=*/bool>;

  // Initializes a NcclClique for given device ranks and returns a lock that
  // gives access to clique communicators.
  auto initialize = [&](absl::Span<const RendezvousArg* const> args)
      -> absl::StatusOr<NcclClique::Lock> {
    TF_ASSIGN_OR_RETURN(auto clique_id, clique_id_callback(clique_key));

    // Check that all ranks successfully synchronized device activity before
    // trying to instantiate NCCL communicators.
    for (const RendezvousArg* arg : args) {
      if (auto& [device_rank, synchronized] = *arg; !synchronized) {
        return Internal(
            "Failed to synchronize device activity on rank %d. Do not attempt "
            "to initialize NCCL clique.",
            device_rank.rank.value());
      }
    }

    std::vector<NcclApi::DeviceRank> ranks;
    ranks.reserve(args.size());
    for (auto* arg : args) ranks.emplace_back(arg->first);

    // Sort device ranks, mainly to get more readable logs below, NCCL does
    // not care in what order ranks are initialized.
    absl::c_sort(ranks, [](auto& a, auto& b) { return a.rank < b.rank; });

    VLOG(3) << absl::StreamFormat(
        "Create NCCL communicators for clique %s; ranks=[%s]; "
        "fingerprint(id)=%d",
        clique_key.ToString(), DeviceRanksToString(ranks),
        clique_id.fingerprint());

    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<Communicator>> created_comms,
        NcclApi::Default()->CommInitRanks(nranks, clique_id, ranks, config));

    absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
    for (size_t i = 0; i < ranks.size(); ++i) {
      comms[ranks[i].rank] = std::move(created_comms[i]);
    }

    VLOG(3) << absl::StreamFormat(
        "Created NCCL communicators for clique %s; ranks=[%s]; "
        "fingerprint(id)=%d",
        clique_key.ToString(), DeviceRanksToString(ranks),
        clique_id.fingerprint());

    NcclCliques& cliques = GetNcclCliques();
    absl::MutexLock lock(&cliques.mu);

    // Create a new clique with given clique key and communicators.
    auto emplaced = cliques.map.try_emplace(clique_key, clique_key, clique_id,
                                            std::move(comms));

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

  NcclApi::DeviceRank device_rank = {device, rank};
  bool synchronized = device->SynchronizeAllActivity();

  // We choose not to exit early on failed synchronization, because it will lead
  // to a deadlock, as not all participants will arrive to a rendezvous point,
  // instead we check synchronization result in the initialization callback.
  //
  // Unfortunately we can't share synchronization result across different
  // processes, so we still might end up in a deadlock situation when some
  // processes are not able to synchronize device activity.
  RendezvousArg rendezvous_arg = std::make_pair(device_rank, synchronized);

  return RendezvousSingle<absl::StatusOr<NcclClique::Lock>>(
      initialization_rendezvous_name, rendezvous_key, rendezvous_arg,
      num_local_participants, initialize, WarnStuckTimeout(),
      TerminateTimeout());
}

// Computes a unique NCCL communicator split color from a clique key. We use a
// deterministic hash function to guarantee that all participating processes get
// the same color value for a clique.
static int32_t GetCommSplitColor(const NcclCliqueKey& clique_key) {
  std::vector<int64_t> global_device_ids;
  global_device_ids.reserve(clique_key.devices().size());

  for (GlobalDeviceId id : clique_key.devices()) {
    global_device_ids.push_back(id.value());
  }

  return abs(static_cast<int32_t>(
      tsl::Hash32(reinterpret_cast<char*>(global_device_ids.data()),
                  sizeof(int64_t) * global_device_ids.size(), 0)));
}

// Joins a NcclClique initialization rendezvous for a `clique_key` and returns
// a lock that gives an access to clique created by splitting already acquired
// `parent_clique` clique (access is shared between all participating ranks that
// own a shared pointer).
static absl::StatusOr<std::shared_ptr<NcclClique::Lock>> InitializeNcclClique(
    se::StreamExecutor* device, RunId run_id, NcclCliqueKey clique_key,
    std::shared_ptr<NcclClique::Lock> parent_clique,
    int32_t num_local_participants, RankId rank, NcclApi::Config& config) {
  // Find our rank in the parent clique.
  const NcclCliqueKey& parent_clique_key = (*parent_clique)->clique_key();
  RankId parent_rank =
      *parent_clique_key.rank(clique_key.devices()[rank.value()]);

  VLOG(3) << "Initialize NCCL clique " << clique_key.ToString() << " rank #"
          << rank << " by splitting rank #" << parent_rank.value()
          << " in parent clique " << parent_clique_key.ToString()
          << "; num_local_participants=" << num_local_participants;

  using RankPair = std::pair<RankId, RankId>;
  RankPair rank_pair = {parent_rank, rank};

  // Current approach for communicator splitting works because of XLAs SPMD
  // programming model where all collective operations have replica groups that
  // include all ranks. This property guarantees that we'll split each
  // communicator exactly once with a unique color computed from rank mapping
  // and each communicator in the parent clique will become a member of exactly
  // one new clique. Clique splitting happens concurrently for multiple
  // non-overlapping clique and this guarantees forward progress even with
  // implicit synchronization inside NCCL.

  // Initializes a NcclClique for given device ranks and returns a lock that
  // gives access to clique communicators.
  auto split = [&](absl::Span<const RankPair* const> rank_pairs)
      -> absl::StatusOr<NcclClique::Lock> {
    // Collect mapping from ranks in parent clique to ranks in a new clique.
    absl::btree_map<RankId, RankId> rank_mapping;
    for (auto* rank_pair : rank_pairs) {
      rank_mapping[rank_pair->first] = rank_pair->second;
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

    VLOG(3) << absl::StreamFormat(
        "Create NCCL communicators for clique %s; parent=%s; color=%d; "
        "rank_mapping=[%s]",
        clique_key.ToString(), parent_clique_key.ToString(), color,
        absl::StrJoin(rank_mapping, ",", rank_mapping_formatter));

    TF_ASSIGN_OR_RETURN(
        auto splitted_comms,
        NcclApi::Default()->CommSplit(parent_comms, color, keys, config));

    absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
    for (size_t i = 0; i < splitted_comms.size(); ++i) {
      comms[keys[i]] = std::move(splitted_comms[i]);
    }

    VLOG(3) << absl::StreamFormat(
        "Created NCCL communicators for clique %s; parent=%s; color=%d; "
        "rank_mapping=[%s]",
        clique_key.ToString(), parent_clique_key.ToString(), color,
        absl::StrJoin(rank_mapping, ",", rank_mapping_formatter));

    NcclCliques& cliques = GetNcclCliques();
    absl::MutexLock lock(&cliques.mu);

    // Create a new clique with given clique key and communicators.
    auto emplaced = cliques.map.try_emplace(clique_key, clique_key,
                                            std::nullopt, std::move(comms));

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

  return RendezvousSingle<absl::StatusOr<NcclClique::Lock>>(
      initialization_rendezvous_name, rendezvous_key, rank_pair,
      num_local_participants, split, WarnStuckTimeout(), TerminateTimeout());
}

//===----------------------------------------------------------------------===//

using AcquiredCliquesMap = NcclClique::AcquiredCliquesMap;

absl::StatusOr<std::shared_ptr<NcclClique::Lock>> AcquireNcclClique(
    se::StreamExecutor* device, RunId run_id, NcclCliqueKey clique_key,
    const NcclCliqueIdCallback& clique_id_callback, RankId rank,
    size_t num_local_participants, const AcquiredCliquesMap& acquired_cliques,
    int64_t max_nchannels) {
  VLOG(2) << "Acquire NCCL clique " << clique_key.ToString() << "; run"
          << run_id.ToString() << "; rank " << rank
          << "; num_local_participants=" << num_local_participants
          << "; acquired_cliques=" << acquired_cliques.size();

  // Get the clique lock via the rendezvous to guarantee that all clique
  // members participate in XLA run.
  auto rendezvous_key = std::make_tuple(run_id, clique_key);
  auto rendezvous_name =
      absl::StrFormat("acquire clique for rank %d; clique=%s; run_id=%d",
                      rank.value(), clique_key.ToString(), run_id.ToInt());

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<NcclClique::Lock> clique,
      RendezvousSingle<absl::StatusOr<NcclClique::Lock>>(
          rendezvous_name, rendezvous_key, num_local_participants,
          [&] {
            NcclCliques& cliques = GetNcclCliques();
            absl::MutexLock lock(&cliques.mu);
            // Returns empty lock if we do not have a clique for `clique_key`.
            auto it = cliques.map.find(clique_key);
            return it == cliques.map.end() ? NcclClique::Lock()
                                           : it->second.Acquire();
          },
          WarnStuckTimeout(), TerminateTimeout()));

  // If lock is not null return it to the caller.
  if (*clique) return clique;

  // Maybe find if we acquired a clique with communicators that we can split.
  static const int64_t enable_nccl_comm_splitting =
      xla::GetDebugOptionsFromFlags().xla_gpu_enable_nccl_comm_splitting();

  // We enable resource sharing between parent and split communicators by
  // default because that's the only reason why we use comm splitting.
  NcclApi::Config config;
  config.split_share = true;
  config.max_nchannels = max_nchannels;

  if (enable_nccl_comm_splitting) {
    for (auto& [acquired_clique_key, acquired_clique] : acquired_cliques) {
      if (clique_key.IsSubsetOf(acquired_clique_key)) {
        return InitializeNcclClique(device, run_id, clique_key, acquired_clique,
                                    num_local_participants, rank, config);
      }
    }
  }

  // If we can't split any of the acquired cliques, create a new one.
  return InitializeNcclClique(device, run_id, clique_key, clique_id_callback,
                              num_local_participants, rank, config);
}

absl::Status NcclClique::CheckAsyncErrors() {
  return async_error_checker_.Check();
}

absl::Status NcclCliqueCommunicators::AsyncErrorChecker::Check() {
  absl::Status status = absl::OkStatus();
  communicators_.ForEachComm([&status](RankId rank, Communicator* comm) {
    // Do not overwrite previous errors.
    if (!status.ok()) return;
    status = NcclApi::Default()->CommGetAsyncError(comm);
    if (!status.ok()) {
      LOG(ERROR) << "NCCL async error (rank " << rank << "): " << status;
    }
  });
  return status;
}

}  // namespace xla::gpu
