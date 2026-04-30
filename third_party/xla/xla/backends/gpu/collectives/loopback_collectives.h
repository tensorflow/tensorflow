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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_LOOPBACK_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_LOOPBACK_COLLECTIVES_H_

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

// A loopback collectives implementation that creates LoopbackCommunicators
// which simulate collective operations using local memory copies instead of
// real inter-device communication (e.g. NCCL). This is useful for testing
// high-level graph logic without needing a multi-node cluster, or for handling
// cases where a model expects a distributed environment but is running on a
// single local device.
//
// Loopback collectives behave as if every rank in the communicator holds
// identical data. No actual inter-rank communication takes place; all
// operations are implemented via device-local memcpy on the stream associated
// with the executor.
//
// Semantic summary of each collective operation:
//
//   AllReduce:         memcpy(recv, send) — other ranks contribute identity
//   Broadcast:         memcpy(recv, send)
//   ReduceScatter:     memcpy this rank's chunk from send to recv
//   AllGather:         replicate send into every rank's slot in recv
//   AllToAll:          memcpy each send buffer to the corresponding recv buffer
//   CollectivePermute: memcpy(recv, send) — data loops back to self
//   Send:              no-op — data "sent" successfully
//   Recv:              zero-fill recv buffer to avoid non-finite values
//
// This way we minimize the chance of producing nans or complete garbage data,
// as all collective results stay "reasonable".
class LoopbackCollectives : public GpuCollectives {
 public:
  bool IsImplemented() const final { return true; }

  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const final;

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Collectives::Config& config) final;

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> SplitCommunicators(
      absl::Span<const Communicator* const> comms, int32_t color,
      absl::Span<const RankId> keys, const Collectives::Config& config,
      absl::Span<const DeviceRank> ranks) final;

  absl::StatusOr<std::unique_ptr<Communicator>> CreateCommunicator() final {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::StatusOr<void*> Allocate(uint64_t bytes) final {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::Status Deallocate(void* buffer) final {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::StatusOr<CliqueIdCallback> InitializeTopology(
      const Topology& topology) final {
    return CliqueIdCallback([](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
      return CliqueIds(CliqueId());
    });
  }
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_LOOPBACK_COLLECTIVES_H_
