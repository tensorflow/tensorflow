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

#include "xla/pjrt/gpu/cross_host_transfer_clique_key.h"

#include <array>
#include <optional>

#include <gtest/gtest.h>
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/runtime/device_id.h"

namespace xla {
namespace {

TEST(CrossHostTransferCliqueKeyTest, MissingSnapshotMatchesHistoricalKey) {
  const GlobalDeviceId src(1);
  const GlobalDeviceId dst(2);
  const gpu::GpuCliqueKey key =
      CrossHostTransferCliqueKey(src, dst, std::nullopt);
  const gpu::GpuCliqueKey historical({src, dst}, /*num_local_participants=*/1,
                                     gpu::CommunicationId(0));

  EXPECT_EQ(key, historical);
  EXPECT_TRUE(key.incarnations().empty());
}

TEST(CrossHostTransferCliqueKeyTest, SortsIncarnationsForBothHosts) {
  const GlobalDeviceId src(1);
  const GlobalDeviceId dst(2);
  const gpu::GpuCliqueKey high_then_low = CrossHostTransferCliqueKey(
      src, dst,
      std::array<IncarnationId, 2>{IncarnationId(20), IncarnationId(10)});
  const gpu::GpuCliqueKey low_then_high = CrossHostTransferCliqueKey(
      src, dst,
      std::array<IncarnationId, 2>{IncarnationId(10), IncarnationId(20)});

  EXPECT_EQ(high_then_low, low_then_high);
  ASSERT_EQ(high_then_low.incarnations().size(), 2);
  EXPECT_EQ(high_then_low.incarnations()[0], IncarnationId(10));
  EXPECT_EQ(high_then_low.incarnations()[1], IncarnationId(20));
  ASSERT_EQ(high_then_low.devices().size(), 2);
  EXPECT_EQ(high_then_low.devices()[0], src);
  EXPECT_EQ(high_then_low.devices()[1], dst);
}

TEST(CrossHostTransferCliqueKeyTest, DedupsEqualIncarnations) {
  const gpu::GpuCliqueKey key = CrossHostTransferCliqueKey(
      GlobalDeviceId(3), GlobalDeviceId(4),
      std::array<IncarnationId, 2>{IncarnationId(7), IncarnationId(7)});

  ASSERT_EQ(key.incarnations().size(), 1);
  EXPECT_EQ(key.incarnations()[0], IncarnationId(7));
}

}  // namespace
}  // namespace xla
