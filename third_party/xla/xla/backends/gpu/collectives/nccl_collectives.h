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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COLLECTIVES_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"

namespace xla::gpu {

// XLA host-initiated collectives implemented on top of NCCL.
class NcclCollectives : public GpuCollectives {
 public:
  bool IsImplemented() const final { return true; }

  bool IsGlobalConfig() const final;

  absl::StatusOr<const CliqueIdCallback*> GetCliqueIdCallback(
      const CliqueIdCallback* clique_id_callback, bool is_local) final;

  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const final;

  absl::Status GroupStart() final;
  absl::Status GroupEnd() final;

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Collectives::Config& config) final;

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> SplitCommunicators(
      absl::Span<const Communicator* const> comms, int32_t color,
      absl::Span<const RankId> keys, const Collectives::Config& config) final;

  absl::StatusOr<void*> Allocate(uint64_t bytes) final;

  absl::Status Deallocate(void* location) final;

  absl::Status InitializeTopology(Topology topology) final;

  // Adds the provided communicator to the current NCCL group, if there is one.
  // If there is no active group, JoinGroup is a noop. JoinGroup returns true if
  // the communicator was added to a group, and false otherwise.
  bool JoinGroup(const Communicator* communicator);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COLLECTIVES_H_
