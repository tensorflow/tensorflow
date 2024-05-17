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

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/bind_front.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_test_util.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;

class XlaShardingSerDesTest : public test_util::ShardingTest {};

TEST_P(XlaShardingSerDesTest, HloShardingRoundTrip) {
  auto device_list = GetDevices({0, 1});
  auto xla_hlo_sharding = xla::HloSharding::Tile(
      xla::TileAssignment((absl::Span<const int64_t>){2, 1}));
  auto sharding = HloSharding::Create(device_list, MemoryKind("abc"),
                                      /*xla_hlo_sharding=*/xla_hlo_sharding);

  TF_ASSERT_OK_AND_ASSIGN(auto serialized, Serialize(*sharding));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding,
      Deserialize<HloSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(
                          absl::bind_front(&Client::LookupDevice, client()))));

  EXPECT_THAT(out_sharding->devices(), ElementsAreArray(sharding->devices()));
  EXPECT_EQ(out_sharding->xla_hlo_sharding(), sharding->xla_hlo_sharding());
}

INSTANTIATE_TEST_SUITE_P(NumDevices, XlaShardingSerDesTest,
                         testing::Values(test_util::ShardingTestParam{
                             .num_devices = 2, .num_addressable_devices = 2}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
