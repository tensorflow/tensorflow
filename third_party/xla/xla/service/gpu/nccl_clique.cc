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

#include "xla/service/gpu/nccl_clique.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/lockable.h"
#include "xla/service/rendezvous.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
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

//===----------------------------------------------------------------------===//
// NcclClique
//===----------------------------------------------------------------------===//

NcclCliqueCommunicators::NcclCliqueCommunicators(
    NcclCliqueKey clique_key, NcclCliqueId clique_id,
    absl::flat_hash_map<int32_t, NcclApi::OwnedNcclComm> communicators)
    : clique_key_(std::move(clique_key)),
      clique_id_(std::move(clique_id)),
      communicators_(std::move(communicators)) {}

std::optional<NcclApi::NcclCommHandle> NcclCliqueCommunicators::comm(
    int32_t rank) {
  if (auto it = communicators_.find(rank); it != communicators_.end()) {
    return it->second.get();
  }
  return std::nullopt;
}

void NcclCliqueCommunicators::ForEachComm(
    absl::FunctionRef<void(int32_t, NcclApi::NcclCommHandle)> fn) {
  for (auto& [rank, comm] : communicators_) {
    fn(rank, comm.get());
  }
}

std::string NcclCliqueCommunicators::DebugString() const {
  std::string out = absl::StrFormat(
      "clique_key: %s; hash(id): %d; size: %d; communicators: ",
      clique_key_.ToString(), absl::HashOf(clique_id_), communicators_.size());
  int32_t cnt = 0;
  for (const auto& [rank, comm] : communicators_) {
    if (cnt++) absl::StrAppend(&out, ", ");
    absl::StrAppendFormat(&out, "[rank=%d, comm=%p]", rank, comm.get());
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
static absl::Status CheckComm(NcclApi::NcclCommHandle comm) {
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
  if (NcclClique::Lock clique = lockable_clique.TryAcquire()) {
    VLOG(5) << "Checking NCCL clique " << clique_key.ToString()
            << " for async errors; num_communicators=" << clique->size();
    clique->ForEachComm([](int32_t rank, NcclApi::NcclCommHandle comm) {
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
    str->append(std::to_string(rank.rank));
  });
}

// Joins a NcclClique initialization rendezvous for a `clique_key` and returns
// a lock that gives an access to initialized clique (access is shared between
// all participating ranks that own a shared pointer).
static absl::StatusOr<std::shared_ptr<NcclClique::Lock>> InitializeNcclClique(
    se::StreamExecutor* device, RunId run_id, NcclCliqueKey clique_key,
    const NcclCliqueIdCallback& clique_id_callback,
    int32_t num_local_participants, int32_t rank) {
  int nranks = clique_key.devices().size();
  VLOG(3) << "Initialize NCCL clique " << clique_key.ToString() << " rank #"
          << rank << " of " << nranks
          << "; num_local_participants=" << num_local_participants;

  // Start NCCL clique heart beat monitor when create a first clique.
  StartNcclCliqueHeartBeatMonitor();

  // Initializes a NcclClique for given device ranks and returns a lock that
  // gives access to clique communicators.
  auto initialize = [&](absl::Span<const NcclApi::DeviceRank* const> args)
      -> absl::StatusOr<NcclClique::Lock> {
    TF_ASSIGN_OR_RETURN(auto clique_id, clique_id_callback(clique_key));

    std::vector<NcclApi::DeviceRank> ranks;
    ranks.reserve(args.size());
    for (auto* arg : args) ranks.emplace_back(*arg);

    VLOG(3) << absl::StreamFormat(
        "Create NCCL communicators for clique %s; ranks=[%s]; hash(id)=%d",
        clique_key.ToString(), DeviceRanksToString(ranks),
        absl::HashOf(clique_id));

    TF_ASSIGN_OR_RETURN(
        std::vector<NcclApi::OwnedNcclComm> created_comms,
        NcclApi::Default()->CommInitRanks(nranks, clique_id, ranks));

    absl::flat_hash_map<int32_t, NcclApi::OwnedNcclComm> comms;
    for (size_t i = 0; i < ranks.size(); ++i) {
      comms[ranks[i].rank] = std::move(created_comms[i]);
    }

    VLOG(3) << absl::StreamFormat(
        "Created NCCL communicators for clique %s; ranks=[%s]; hash(id)=%d",
        clique_key.ToString(), DeviceRanksToString(ranks),
        absl::HashOf(clique_id));

    NcclCliques& cliques = GetNcclCliques();
    absl::MutexLock lock(&cliques.mu);

    // Create a new clique with given clique id and communicators.
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
  auto initialization_rendezvous_name = absl::StrFormat(
      "create clique initialization state for rank %d; clique=%s; run_id=%d",
      rank, clique_key.ToString(), run_id.ToInt());

  NcclApi::DeviceRank device_rank = {device, rank};

  return RendezvousSingle<absl::StatusOr<NcclClique::Lock>>(
      initialization_rendezvous_name, rendezvous_key, device_rank,
      num_local_participants, initialize, WarnStuckTimeout(),
      TerminateTimeout());
}

//===----------------------------------------------------------------------===//

absl::StatusOr<std::shared_ptr<NcclClique::Lock>> AcquireNcclClique(
    se::StreamExecutor* device, RunId run_id, NcclCliqueKey clique_key,
    const NcclCliqueIdCallback& clique_id_callback, int32_t rank,
    size_t num_local_participants) {
  VLOG(2) << "Acquire NCCL clique " << clique_key.ToString() << "; run"
          << run_id.ToString() << "; rank " << rank
          << "; num_local_participants=" << num_local_participants;

  // Get the clique lock via the rendezvous to guarantee that all clique
  // members participate in XLA run.
  auto rendezvous_key = std::make_tuple(run_id, clique_key);
  auto rendezvous_name =
      absl::StrFormat("acquire clique for rank %d; clique=%s; run_id=%d", rank,
                      clique_key.ToString(), run_id.ToInt());

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

  // If NCCL clique is not found try to initialize a new one for a given key.
  return InitializeNcclClique(device, run_id, clique_key, clique_id_callback,
                              num_local_participants, rank);
}

}  // namespace xla::gpu
