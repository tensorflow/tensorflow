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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/lockable.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep

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
// GpuCliqueKey for details.
//
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently

//===----------------------------------------------------------------------===//
// NcclClique
//===----------------------------------------------------------------------===//

// TODO(b/380457503): Delete this alias.
using NcclClique = LockableGpuClique;

// Acquires an shared access to a NCCL clique (NcclClique::Lock collectively
// owned by `num_local_participants` threads). XLA uses this lock to serialize
// execution of all collective operations sharing a `clique_id`.
//
// If clique for a given key does not exist it will be initialized from newly
// created communicators or maybe created by splitting of the already acquired
// cliques.
absl::StatusOr<std::shared_ptr<NcclClique::Lock>> AcquireNcclClique(
    se::StreamExecutor* device, RunId run_id, GpuCliqueKey clique_key,
    const CliqueIdCallback& clique_id_callback, RankId rank,
    size_t num_local_participants,
    const NcclClique::AcquiredCliquesMap& acquired_cliques,
    int64_t max_nchannels = 0);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_CLIQUE_H_
