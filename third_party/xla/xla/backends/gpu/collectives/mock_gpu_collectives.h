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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_MOCK_GPU_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_MOCK_GPU_COLLECTIVES_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"

namespace xla::gpu {

// Implements GpuCollectives for testing.
class MockGpuCollectives : public GpuCollectives {
 public:
  MockGpuCollectives() = default;

  MOCK_METHOD(absl::StatusOr<CliqueId>, CreateUniqueCliqueId, (),
              (const, override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>,
              CreateCommunicators,
              (const CliqueKey& clique_key,
               const std::optional<CliqueIds>& clique_ids,
               absl::Span<const DeviceRank> ranks,
               const Collectives::Config& config),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>,
              SplitCommunicators,
              (absl::Span<const Communicator* const> comms, int32_t color,
               absl::Span<const RankId> keys, const Collectives::Config& config,
               absl::Span<const DeviceRank> ranks),
              (override));
  MOCK_METHOD(absl::StatusOr<CliqueIdCallback>, InitializeTopology,
              (const Topology& topology), (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>,
              CreateCommunicatorsWithCancel,
              (const CliqueKey& clique_key,
               const std::optional<CliqueIds>& clique_ids,
               absl::Span<const DeviceRank> ranks,
               const Collectives::Config& config,
               std::shared_ptr<CancellationToken> cancel),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>,
              SplitCommunicatorsWithCancel,
              (absl::Span<const Communicator* const> comms, int32_t color,
               absl::Span<const RankId> keys, const Collectives::Config& config,
               absl::Span<const DeviceRank> ranks,
               std::shared_ptr<CancellationToken> cancel),
              (override));
  MOCK_METHOD(absl::Status, MaybeAttachGxlCommunicators,
              (const CliqueKey& clique_key, absl::Span<const DeviceRank> ranks,
               absl::Span<Communicator* const> comms),
              (override));
  MOCK_METHOD(bool, IsImplemented, (), (const, override));
  MOCK_METHOD(absl::Status, GroupLaunch,
              (absl::Span<const GpuCommunicator* const> comms,
               absl::FunctionRef<absl::Status()> group),
              (override));
  MOCK_METHOD(absl::StatusOr<void*>, Allocate, (uint64_t bytes), (override));
  MOCK_METHOD(absl::Status, Deallocate, (void* buffer), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Communicator>>, CreateCommunicator,
              (), (override));
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MOCK_GPU_COLLECTIVES_H_
