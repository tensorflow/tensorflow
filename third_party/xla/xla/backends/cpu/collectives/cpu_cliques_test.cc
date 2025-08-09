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

#include <memory>

#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_clique_key.h"
#include "xla/backends/cpu/collectives/in_process_collectives.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/global_device_id.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(CpuCliques, InvalidateAcquiredCommunicators) {
  GlobalDeviceId d0(0);
  GlobalDeviceId d1(1);

  CpuCliqueKey clique_key({d0, d1});

  auto collectives0 = std::make_unique<InProcessCollectives>();
  auto collectives1 = std::make_unique<InProcessCollectives>();

  // Check that communicator instance is cached.
  TF_ASSERT_OK_AND_ASSIGN(
      auto* comm0, AcquireCommunicator(&*collectives0, clique_key, RankId(0)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto* comm1, AcquireCommunicator(&*collectives0, clique_key, RankId(0)));
  EXPECT_EQ(comm0, comm1);

  // Destroy communicators created for `collectives0`.
  collectives0.reset();

  // Acquire communicator from a new instance of collectives.
  TF_ASSERT_OK_AND_ASSIGN(
      auto* comm2, AcquireCommunicator(&*collectives1, clique_key, RankId(0)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto* comm3, AcquireCommunicator(&*collectives1, clique_key, RankId(0)));
  EXPECT_EQ(comm2, comm3);

  // Check that we acquired new communicators.
  EXPECT_NE(comm0, comm2);

  // Destroy communicators created for `collectives1`.
  collectives1.reset();
}

}  // namespace
}  // namespace xla::cpu
