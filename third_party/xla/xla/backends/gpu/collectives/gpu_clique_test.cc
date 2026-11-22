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

#include "xla/backends/gpu/collectives/gpu_clique.h"

#include <memory>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/btree_map.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/mock_gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/mock_stream_executor.h"

namespace xla::gpu {
namespace {

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

TEST(GpuCliqueTest, ReleaseSymmetricMemoryOnlyOnMatchingDevice) {
  NiceMock<se::MockStreamExecutor> executor0;
  NiceMock<se::MockStreamExecutor> executor1;
  ON_CALL(executor0, device_ordinal()).WillByDefault(Return(0));
  ON_CALL(executor1, device_ordinal()).WillByDefault(Return(1));

  auto comm0 = std::make_unique<StrictMock<MockGpuCommunicator>>();
  auto comm1 = std::make_unique<StrictMock<MockGpuCommunicator>>();
  auto comm_without_executor =
      std::make_unique<StrictMock<MockGpuCommunicator>>();
  EXPECT_CALL(*comm0, stream_executor()).WillRepeatedly(Return(&executor0));
  EXPECT_CALL(*comm1, stream_executor()).WillRepeatedly(Return(&executor1));
  EXPECT_CALL(*comm_without_executor, stream_executor())
      .WillRepeatedly(Return(nullptr));

  se::DeviceAddressBase addr(reinterpret_cast<void*>(0x1000), 1024);
  EXPECT_CALL(*comm1, ReleaseSymmetricMemory(addr));

  absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
  comms[RankId(0)] = std::move(comm0);
  comms[RankId(1)] = std::move(comm1);
  comms[RankId(2)] = std::move(comm_without_executor);
  GpuClique clique(
      GpuCliqueKey({GlobalDeviceId(0), GlobalDeviceId(1), GlobalDeviceId(2)},
                   /*num_local_participants=*/3),
      /*ids=*/std::nullopt, std::move(comms), /*peer_access_enabled=*/false,
      std::make_shared<CancellationToken>());

  clique.ReleaseSymmetricMemory(LocalDeviceId(1), addr);
}

}  // namespace
}  // namespace xla::gpu
