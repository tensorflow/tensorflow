/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/lockable.h"
#include "xla/service/rendezvous.h"
#include "xla/status_macros.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// NcclUniqueId
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
      [](const NcclCliqueKey&) { return NcclApi::GetUniqueId(); });
  return local_callback;
}

//===----------------------------------------------------------------------===//
// NcclClique
//===----------------------------------------------------------------------===//

namespace {

struct NcclCliqueState {
  NcclCliqueId clique_id;
  int64_t run_id = -1;

  // `mu` guards `communicators` and `status` during initialization.
  // Once `ready` has been notified, the communicators may be accessed without
  // synchronization.
  absl::Mutex mu;
  absl::Notification ready;
  absl::Status status;
  absl::flat_hash_map<int, std::unique_ptr<NcclComm>> communicators;
};

using NcclClique = Lockable<NcclCliqueState>;

struct NcclCliques {
  NcclClique& operator[](const NcclCliqueKey& key) {
    absl::MutexLock lock(&mu);
    return cliques[key];
  }

  absl::Mutex mu;
  absl::node_hash_map<NcclCliqueKey, NcclClique> cliques ABSL_GUARDED_BY(mu);
};

std::shared_ptr<absl::StatusOr<NcclClique::Lock>> AcquireNcclClique(
    RunId run_id, OpId op_id, NcclCliqueKey clique_key,
    const NcclCliqueIdCallback& clique_id_callback,
    size_t num_local_participants, bool may_skip_rendezvous) {
  static auto& cliques = *new NcclCliques;

  VLOG(2) << "AcquireNcclClique Rendezvous key (clique_key: "
          << clique_key.ToString() << ", run" << run_id.ToString() << ", op"
          << op_id.value() << ")";

  // RendezvousSingle should only be used to guard nccl communicator
  // initialization. Return the clique state when we are done with such
  // initialization.
  //
  // TODO(bixia): enable this unconditionally after fixing a deadlock issue.
  if (may_skip_rendezvous) {
    // Destruct clique if it hasn't been notified.
    NcclClique::Lock clique = cliques[clique_key].Acquire();
    if (clique->ready.HasBeenNotified() && clique->run_id == run_id.ToInt()) {
      return std::make_shared<absl::StatusOr<NcclClique::Lock>>(
          std::move(clique));
    }
  }

  auto rendezvous_key = std::make_tuple(run_id, op_id, std::move(clique_key));

  int64_t terminate_timeout = xla::GetDebugOptionsFromFlags()
                                  .xla_gpu_nccl_termination_timeout_seconds();

  return RendezvousSingle<absl::StatusOr<NcclClique::Lock>>(
      rendezvous_key, num_local_participants,
      [&]() -> absl::StatusOr<NcclClique::Lock> {
        const NcclCliqueKey& clique_key = std::get<2>(rendezvous_key);
        NcclClique::Lock clique = cliques[clique_key].Acquire();
        if (clique->run_id < 0) {
          TF_ASSIGN_OR_RETURN(clique->clique_id,
                              clique_id_callback(clique_key));
        }
        // If multiple executable are running simultaneously while using
        // multiple hosts, it is possible that different executables could
        // acquire the same clique on different hosts. We protect against this
        // by checking that the run ID increases monotonically.
        bool is_local = clique_key.devices().size() == num_local_participants;
        TF_RET_CHECK(is_local || (run_id.ToInt() >= clique->run_id));
        clique->run_id = run_id.ToInt();
        return clique;
      },
      /*warn_stuck_timeout=*/absl::Seconds(10),
      (terminate_timeout >= 0) ? absl::Seconds(terminate_timeout)
                               : absl::InfiniteDuration());
}

// Adds NCCL communicator to a global per-process state that tracks NCCL
// communicators health.
void TrackNcclCommunicatorHealth(NcclComm* comm) {
  struct AllCommunicators {
    absl::Mutex mu;
    std::vector<NcclComm*> communicators ABSL_GUARDED_BY(mu);
  };

  static auto* all_communicators = new AllCommunicators();

  absl::MutexLock lock(&all_communicators->mu);
  all_communicators->communicators.push_back(comm);

  // Runs an async error check for a `comm` and aborts it if it is in the error
  // state. It will free resources that are allocated to a communicator and
  // abort any uncompleted operations before destroying the communicator.
  auto check_nccl_async_error = [](NcclComm* lockable_comm) -> absl::Status {
    NcclApi::NcclCommHandle comm = *lockable_comm->Acquire();
    if (comm == nullptr) return absl::OkStatus();

    absl::Status async_err = NcclApi::CommGetAsyncError(comm);
    if (!async_err.ok()) {
      LOG(ERROR) << "Aborting communicator: " << comm
                 << " due to async NCCL error: " << async_err;
      TF_RETURN_IF_ERROR(NcclApi::CommAbort(comm));
    }

    return async_err;
  };

  // Launch a thread that periodically checks all NCCL communicators for
  // asynchronous errors. If an asynchronous error is observed, the communicator
  // is aborted and an error message logged.
  static auto check_async_error_thread = tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "nccl_async_error_thread", [&] {
        while (true) {
          absl::SleepFor(absl::Seconds(30));
          absl::MutexLock lock(&all_communicators->mu);
          VLOG(5) << "Checking NCCL communicators for async errors"
                  << "; num_communicators="
                  << all_communicators->communicators.size();
          for (NcclComm* comm : all_communicators->communicators) {
            if (auto status = check_nccl_async_error(comm); !status.ok()) {
              LOG(ERROR) << status;
            }
          }
        }
      });
  (void)check_async_error_thread;  // Silence unused variable warning.
}

}  // namespace

absl::StatusOr<NcclComm::Lock> AcquireNcclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    size_t num_local_participants,
    const NcclCliqueIdCallback& clique_id_callback, int32_t rank,
    int64_t stream_id, bool enable_clique_optimization) {
  // Ensure that this group of threads have exclusive access to the clique to
  // prevent threads from different groups locking communicators in the clique.
  // The enable_clique_optimization value is only used for asynchronous
  // collective stream currently. For synchronous collectives, we should always
  // enable the optimization. For P2P stream, we currently have to always enable
  // the optimization, because we initially implement this optimization to
  // workaround an NCCL bug related to P2P operations.
  NcclCliqueKey clique_key(std::move(participants), stream_id);

  std::shared_ptr<absl::StatusOr<NcclClique::Lock>> clique = AcquireNcclClique(
      run_id, op_id, clique_key, clique_id_callback, num_local_participants,
      enable_clique_optimization ||
          stream_id !=
              GetStreamId(/*is_async=*/true, AsyncStreamKind::kCollective));

  TF_RETURN_IF_ERROR(clique->status());
  NcclCliqueState& state = *clique->value();

  if (!state.ready.HasBeenNotified()) {
    int nranks = clique_key.devices().size();

    absl::StatusOr<NcclApi::NcclCommHandle> comm =
        NcclApi::CommInitRank(nranks, state.clique_id, rank);

    size_t num_initialized = [&] {
      absl::MutexLock lock(&state.mu);
      if (comm.ok()) {
        state.communicators[rank] = std::make_unique<NcclComm>(*comm);
      } else {
        state.status.Update(comm.status());
        state.communicators[rank] = std::make_unique<NcclComm>(nullptr);
      }
      return state.communicators.size();
    }();

    // Wait for all communicators to initialize before allowing any progress.
    // Otherwise we may get deadlocks, because ncclCommInitRank may allocate,
    // which may block on the completion of device activity on a peer device,
    // which may depend on the completion of this collective if we do not have a
    // barrier to prevent it.
    if (num_initialized == num_local_participants) {
      state.ready.Notify();
    } else {
      TF_RETURN_IF_ERROR(comm.status());
      state.ready.WaitForNotification();
    }

    // Register initialized communicator with pre-process health tracking.
    TrackNcclCommunicatorHealth(state.communicators[rank].get());
  }

  TF_RETURN_IF_ERROR(state.status);
  return state.communicators[rank]->Acquire();
}

}  // namespace xla::gpu
