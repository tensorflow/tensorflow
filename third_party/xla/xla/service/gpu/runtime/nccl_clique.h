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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_CLIQUE_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_CLIQUE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/executable_run_options.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/lockable.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// NCCL clique (collective clique) is a set of devices that execute collective
// operations (e.g. all-reduce). It is notoriously easy to misuse NCCL
// communicators (see link below) and get a dead lock at run time, so in XLA we
// take extra care to order all collective operations in a way that would not
// lead to a deadlock.
//
// We rely on exclusive access to a NCCL clique (using Lockable<T> mechanism) to
// guarantee that only a set of threads executing a particular collective
// operation can schedule new work using communicators belonging to a clique.
//
// In XLA process we have multiple cliques for different combinations of
// participating devices and properties of collective operations launched on
// them, e.g. mixing NCCL operations launched from CUDA graphs with regularly
// launched operations is prone to dead locks, and we keep them separate. See
// NcclCliqueKey for details.
//
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently

//===----------------------------------------------------------------------===//
// NcclUniqueId
//===----------------------------------------------------------------------===//

// Returns true if the NCCL config is global (NCCL_COMM_ID env variable is set).
bool IsGlobalNcclConfig();

// Returns a clique id callback passed as an argument if it's not null or a
// default callback to get create a clique id if we are running in local mode.
absl::StatusOr<const NcclCliqueIdCallback*> GetNcclCliqueIdCallback(
    const NcclCliqueIdCallback* clique_id_callback,  // may be null
    bool is_local);

//===----------------------------------------------------------------------===//
// NcclClique
//===----------------------------------------------------------------------===//

// A group of NCCL communicators making up a clique. With NCCL it's notoriously
// easy to get a deadlock, so we take extra care by grouping communicators into
// cliques and making sure that we have a well defined order of all collective
// operations that does not lead to deadlocks.
class NcclCliqueCommunicators {
 public:
  class AsyncErrorChecker {
   public:
    absl::Status Check();

   private:
    friend class NcclCliqueCommunicators;
    AsyncErrorChecker(NcclCliqueCommunicators& comms) : communicators_(comms) {}

    NcclCliqueCommunicators& communicators_;
  };

  NcclCliqueCommunicators(
      NcclCliqueKey clique_key, std::optional<NcclCliqueId> clique_id,
      absl::btree_map<int32_t, NcclApi::OwnedNcclComm> communicators);

  // Returns a NCCL communicator for a given rank if it's in a clique.
  std::optional<NcclApi::NcclCommHandle> comm(int32_t rank);

  // Return true if clique is local: all communicators belong to current
  // process. Non-local cliques spans multiple processes (typically hosts).
  bool IsLocal() const;

  // Calls `fn` for each communicator in the clique.
  void ForEachComm(
      absl::FunctionRef<void(int32_t, NcclApi::NcclCommHandle)> fn);

  const NcclCliqueKey& clique_key() const { return clique_key_; }
  const std::optional<NcclCliqueId>& clique_id() const { return clique_id_; }
  size_t num_communicators() const { return communicators_.size(); }

  std::string DebugString() const;

  AsyncErrorChecker GetChecker() { return AsyncErrorChecker(*this); }

 private:
  NcclCliqueKey clique_key_;
  std::optional<NcclCliqueId> clique_id_;

  // TODO(ezhulenev): Switch this map to GlobalDeviceId key.
  absl::btree_map<int32_t, NcclApi::OwnedNcclComm> communicators_;
};

struct NcclCliqueName {
  static std::string ToString(const NcclCliqueCommunicators& comms) {
    return absl::StrFormat("lockable clique %s", comms.clique_key().ToString());
  }
};

class NcclClique : public Lockable<NcclCliqueCommunicators, NcclCliqueName> {
 public:
  // We keep acquired cliques in a sorted container to guarantee that all
  // participants iterate over cliques in the same order.
  using AcquiredCliquesMap =
      absl::btree_map<NcclCliqueKey, std::shared_ptr<NcclClique::Lock>,
                      std::greater<NcclCliqueKey>>;

  // Construct the lockable clique.
  // Note that async errors can be checked without acquiring the lock.
  // To get the lock-free reference to the communicators for the async
  // error checks, the constructor intentionally leaks the reference
  // to the communicators from an acquired lock.
  NcclClique(NcclCliqueKey clique_key, std::optional<NcclCliqueId> clique_id,
             absl::btree_map<int32_t, NcclApi::OwnedNcclComm> communicators)
      : Lockable(std::move(clique_key), clique_id, std::move(communicators)),
        async_error_checker_(Acquire()->GetChecker()) {}

  std::string DebugString() const;

  // Checks for async errors for all the communicators in the clique without
  // taking the lock. If at least one of the communicators has an async error,
  // it returns one of the errors.
  absl::Status CheckAsyncErrors();

 private:
  NcclCliqueCommunicators::AsyncErrorChecker async_error_checker_;
};

// Acquires an shared access to a NCCL clique (NcclClique::Lock collectively
// owned by `num_local_participants` threads). XLA uses this lock to serialize
// execution of all collective operations sharing a `clique_id`.
//
// If clique for a given key does not exist it will be initialized from newly
// created communicators or maybe created by splitting of the already acquired
// cliques.
absl::StatusOr<std::shared_ptr<NcclClique::Lock>> AcquireNcclClique(
    se::StreamExecutor* device, RunId run_id, NcclCliqueKey clique_key,
    const NcclCliqueIdCallback& clique_id_callback, int32_t rank,
    size_t num_local_participants,
    const NcclClique::AcquiredCliquesMap& acquired_cliques,
    int64_t max_nchannels = 0);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_CLIQUE_H_
