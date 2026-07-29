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

#include "xla/backends/gpu/collectives/oneccl_collectives.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "oneapi/ccl.h"
#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/oneccl_communicator.h"
#include "xla/backends/gpu/collectives/oneccl_errors.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

onecclConfig_t AsOnecclConfig(const Collectives::Config& config) {
  onecclConfig_t comm_config = ONECCL_CONFIG_INITIALIZER;
  comm_config.multiThreaded = 1;
  return comm_config;
}

static absl::StatusOr<onecclUniqueId> AsOnecclUniqueId(
    const CliqueId& clique_id) {
  if (clique_id.size() != ONECCL_UNIQUE_ID_BYTES) {
    return Internal(
        "CliqueId size is not equal to ONECCL_UNIQUE_ID_BYTES: %d vs %d",
        clique_id.size(), ONECCL_UNIQUE_ID_BYTES);
  }
  onecclUniqueId id;
  absl::c_copy(clique_id.data(), id.data);
  return id;
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
OnecclCollectives::CreateCommunicators(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, const Collectives::Config& config) {
  const auto& gpu_config =
      tsl::down_cast<const GpuCollectives::Config&>(config);
  onecclConfig_t comm_config = AsOnecclConfig(gpu_config);

  auto make_comm = [&](int i) -> absl::StatusOr<onecclComm_t> {
    VLOG(1) << "Initialize oneCCL communicator for rank #" << ranks[i].rank
            << "of " << clique_key.num_devices()
            << "; fingerprint(id)=" << clique_ids->fingerprint()
            << "; size(id)=" << clique_ids->data().size();
    auto* device = absl::down_cast<GpuCollectives::Device*>(ranks[i].device);
    int32_t device_ordinal = device->stream_executor()->device_ordinal();
    XLA_ONECCL_RETURN_IF_ERROR(onecclSetDevice(device_ordinal));
    ASSIGN_OR_RETURN(auto oneccl_unique_id,
                     AsOnecclUniqueId(clique_ids->at(0)));
    onecclComm_t comm;
    XLA_ONECCL_RETURN_IF_ERROR(onecclCommInitRankConfig(
        &comm, clique_key.num_devices(), oneccl_unique_id,
        ranks[i].rank.value(), &comm_config));
    return comm;
  };

  // Create all communicators. Each communicator is crated on its own thread.
  std::vector<std::unique_ptr<Communicator>> comms(ranks.size());
  absl::Status status;
  absl::once_flag once;

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(),
                                 "CreateOnecclCommunicators", ranks.size());
    for (size_t i = 0; i < ranks.size(); ++i) {
      pool.Schedule([&, i]() {
        absl::StatusOr<std::unique_ptr<OnecclCommunicator>> comm =
            OnecclCommunicator::Create(std::bind(make_comm, i),
                                       gpu_config.async_execution);

        if (!comm.ok()) {
          absl::call_once(once, [&] { status = comm.status(); });
          return;
        }
        comms[i] = *std::move(comm);
      });
    }
  }
  RETURN_IF_ERROR(status);
  return comms;
}
}  // namespace xla::gpu

XLA_COLLECTIVES_REGISTER("SYCL", "oneccl", 100,
                         std::make_unique<xla::gpu::OnecclCollectives>());
