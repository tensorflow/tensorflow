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

#include <memory>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;

using XlaShardingSerDesTestParam =
    std::tuple<SerDesVersion, test_util::DeviceTestParam>;

class XlaShardingSerDesTest
    : public testing::TestWithParam<XlaShardingSerDesTestParam> {
 public:
  XlaShardingSerDesTest()
      : version_(std::get<0>(GetParam())), fixture_(std::get<1>(GetParam())) {}

  SerDesVersion version() const { return version_; }

  Client* client() { return fixture_.client(); }
  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return fixture_.GetDevices(device_indices);
  }

 private:
  SerDesVersion version_;
  test_util::DeviceTestFixture fixture_;
};

TEST_P(XlaShardingSerDesTest, HloShardingRoundTrip) {
  auto device_list = GetDevices({0, 1});
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({2, 1}));
  auto sharding = HloSharding::Create(device_list, MemoryKind("abc"),
                                      /*xla_hlo_sharding=*/xla_hlo_sharding);

  auto options = std::make_unique<SerializeOptions>(version());
  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*sharding, std::move(options)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding,
      Deserialize<HloSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(client())));

  EXPECT_THAT(out_sharding->devices()->devices(),
              ElementsAreArray(sharding->devices()->devices()));
  EXPECT_EQ(out_sharding->xla_hlo_sharding(), sharding->xla_hlo_sharding());
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion_NumDevices, XlaShardingSerDesTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(test_util::DeviceTestParam{
                         /*num_devices=*/2, /*num_addressable_devices=*/2})));

}  // namespace
}  // namespace ifrt
}  // namespace xla
