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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUES_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "absl/container/btree_map.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/service/lockable.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep

namespace xla::gpu {

// A sorted container of acquired cliques. We keep cliques ordered by the key,
// so that all participants are guaranteed to iterate over the cliques in the
// same order, because otherwise we could get deadlocks when different
// participants try to split cliques in different orders.
class AcquiredCliquesMap
    : public absl::btree_map<GpuCliqueKey,
                             std::shared_ptr<LockableGpuClique::Lock>,
                             std::greater<GpuCliqueKey>> {};

// Acquires a "shared exclusive" access to a GPU clique (exclusive in a sense
// that the clique is locked for exclusive use by `num_local_participants`
// threads holding the shared lock object). XLA uses this lock to serialize
// execution of all collective operations sharing a `clique_id`.
//
// We rely on exclusive access to a GPU clique (using Lockable<T> mechanism) to
// guarantee that only a set of threads executing a particular collective
// operation can schedule new work using communicators belonging to a clique.
//
// If clique for a given key does not exist it will be initialized from newly
// created communicators or maybe created by splitting of the already acquired
// cliques.
//
// WARNING: This is a collective operation that must be executed by all local
// participants of the clique key concurrently (it must be called from an
// appropriately sized thread pool to avoid deadlocks). Implementation relies on
// the rendezvous mechanism to ensure that all participants join clique
// acquisition, with a rendezvous key derived from the clique key.
absl::StatusOr<std::shared_ptr<LockableGpuClique::Lock>> AcquireGpuClique(
    GpuCollectives* collectives, se::StreamExecutor* device, RunId run_id,
    const GpuCliqueKey& clique_key,
    const GpuCollectives::CliqueIdCallback& clique_id_callback, RankId rank,
    const AcquiredCliquesMap& acquired_cliques, int64_t max_nchannels = 0);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUES_H_
