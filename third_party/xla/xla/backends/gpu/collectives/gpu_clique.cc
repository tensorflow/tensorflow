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

#include "xla/backends/gpu/collectives/gpu_clique.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/clique.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/lockable.h"
#include "tsl/platform/logging.h"

namespace xla::gpu {

GpuClique::GpuClique(
    GpuCliqueKey key, std::optional<CliqueId> id,
    absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators)
    : Clique(std::move(communicators)), key_(key), id_(id) {}

std::string GpuClique::DebugString() const {
  std::string out =
      absl::StrFormat("key: %s; fingerprint(id): %d; size: %d; communicators: ",
                      key_.ToString(), id_.has_value() ? id_->fingerprint() : 0,
                      num_communicators());
  int32_t cnt = 0;
  ForEachComm([&](RankId rank, Communicator* comm) {
    if (cnt++) absl::StrAppend(&out, ", ");
    absl::StrAppendFormat(&out, "[rank=%d, comm=%p]", rank.value(), comm);
  });
  return out;
}

absl::Status GpuClique::HealthCheck() const {
  absl::Status health_check = absl::OkStatus();
  ForEachComm([&health_check](RankId rank, Communicator* comm) {
    if (auto s = comm->HealthCheck(); !s.ok()) {
      LOG(ERROR) << "GPU communicator error (rank " << rank << "): " << s;
      if (health_check.ok()) health_check = std::move(s);  // return first error
    }
  });
  return health_check;
}

std::string GpuClique::LockableName::ToString(const GpuClique& clique) {
  return absl::StrFormat("lockable clique %s", clique.key().ToString());
}

LockableGpuClique::LockableGpuClique(
    GpuCliqueKey clique_key, std::optional<CliqueId> clique_id,
    absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators)
    : Lockable(std::move(clique_key), clique_id, std::move(communicators)) {}

absl::Status LockableGpuClique::HealthCheck() const {
  return value().HealthCheck();
}

std::string LockableGpuClique::DebugString() const {
  return absl::StrFormat("LockableGpuClique: %s", value().DebugString());
}

}  // namespace xla::gpu
