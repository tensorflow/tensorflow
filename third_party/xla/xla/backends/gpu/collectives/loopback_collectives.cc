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

#include "xla/backends/gpu/collectives/loopback_collectives.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/loopback_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

absl::StatusOr<CliqueId> LoopbackCollectives::CreateUniqueCliqueId() const {
  VLOG(1) << "Created loopback clique id";
  return CliqueId();
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
LoopbackCollectives::CreateCommunicators(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, const Collectives::Config& config) {
  size_t num_ranks = ranks.size();
  VLOG(1) << absl::StreamFormat(
      "Creating %d loopback communicators for clique %v", num_ranks,
      clique_key);
  std::vector<std::unique_ptr<Communicator>> comms(num_ranks);

  for (size_t i = 0; i < num_ranks; ++i) {
    auto* device = tsl::down_cast<GpuCollectives::Device*>(ranks[i].device);
    comms[i] = std::make_unique<LoopbackCommunicator>(device->stream_executor(),
                                                      num_ranks, i);
    VLOG(3) << absl::StreamFormat(
        "Created loopback communicator rank=%d/%d for clique %v", i, num_ranks,
        clique_key);
  }

  return comms;
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
LoopbackCollectives::SplitCommunicators(
    absl::Span<const Communicator* const> comms, int32_t color,
    absl::Span<const RankId> keys, const Collectives::Config& config,
    absl::Span<const DeviceRank> ranks) {
  size_t num_ranks = ranks.size();
  VLOG(1) << absl::StreamFormat(
      "Splitting %d loopback communicators with color=%d", num_ranks, color);
  std::vector<std::unique_ptr<Communicator>> split_comms(num_ranks);

  for (size_t i = 0; i < num_ranks; ++i) {
    auto* device = tsl::down_cast<GpuCollectives::Device*>(ranks[i].device);
    split_comms[i] = std::make_unique<LoopbackCommunicator>(
        device->stream_executor(), num_ranks, i);
  }

  return split_comms;
}

}  // namespace xla::gpu

XLA_COLLECTIVES_REGISTER("gpu", "loopback", 0,
                         std::make_unique<xla::gpu::LoopbackCollectives>());
