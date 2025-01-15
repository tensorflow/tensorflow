/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/cpu/collectives/in_process_collectives.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/in_process_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
InProcessCollectives::CreateCommunicators(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, const Config& config) {
  std::vector<std::unique_ptr<Communicator>> communicators;
  communicators.reserve(ranks.size());

  for (auto& device_rank : ranks) {
    size_t rank = device_rank.rank.value();
    communicators.push_back(std::make_unique<InProcessCommunicator>(
        rank, clique_key.num_devices()));
  }

  return communicators;
}

}  // namespace xla::cpu
