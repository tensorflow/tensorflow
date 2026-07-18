/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/collectives/cpu_cliques.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_clique_key.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/collectives/in_process_collectives.h"
#include "xla/backends/cpu/collectives/in_process_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

class TrackingCommunicator : public InProcessCommunicator {
 public:
  TrackingCommunicator(size_t rank, size_t num_ranks, bool* destroyed)
      : InProcessCommunicator(rank, num_ranks), destroyed_(destroyed) {}
  ~TrackingCommunicator() override { *destroyed_ = true; }

 private:
  bool* destroyed_;
};

class TrackingCollectives : public CpuCollectives {
 public:
  explicit TrackingCollectives(bool* destroyed) : destroyed_(destroyed) {}

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Config& config) override {
    std::vector<std::unique_ptr<Communicator>> communicators;
    communicators.reserve(ranks.size());

    for (auto& device_rank : ranks) {
      size_t rank = device_rank.rank.value();
      communicators.push_back(std::make_unique<TrackingCommunicator>(
          rank, clique_key.num_devices(), destroyed_));
    }

    return communicators;
  }

 private:
  bool* destroyed_;
};

TEST(CpuCliques, InvalidateAcquiredCommunicators) {
  GlobalDeviceId d0(0);
  GlobalDeviceId d1(1);

  CpuCliqueKey clique_key({d0, d1});

  bool comm0_destroyed = false;
  auto collectives0 = std::make_unique<TrackingCollectives>(&comm0_destroyed);

  // Check that communicator instance is cached.
  ASSERT_OK_AND_ASSIGN(auto* comm0, AcquireCommunicator(collectives0.get(),
                                                        clique_key, RankId(0)));
  ASSERT_OK_AND_ASSIGN(auto* comm1, AcquireCommunicator(collectives0.get(),
                                                        clique_key, RankId(0)));
  EXPECT_EQ(comm0, comm1);

  EXPECT_FALSE(comm0_destroyed);

  // Destroy collectives0. This should destroy the communicators.
  collectives0.reset();

  EXPECT_TRUE(comm0_destroyed);

  bool comm1_destroyed = false;
  auto collectives1 = std::make_unique<TrackingCollectives>(&comm1_destroyed);

  // Acquire communicator from a new instance of collectives.
  ASSERT_OK_AND_ASSIGN(auto* comm2, AcquireCommunicator(collectives1.get(),
                                                        clique_key, RankId(0)));
  ASSERT_OK_AND_ASSIGN(auto* comm3, AcquireCommunicator(collectives1.get(),
                                                        clique_key, RankId(0)));
  EXPECT_EQ(comm2, comm3);

  EXPECT_FALSE(comm1_destroyed);

  collectives1.reset();
  EXPECT_TRUE(comm1_destroyed);
}

}  // namespace
}  // namespace xla::cpu
