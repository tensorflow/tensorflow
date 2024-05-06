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

#include "xla/python/ifrt/support/sharding_conversions.h"

#include <memory>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_test_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace support {
namespace {

using ::tsl::testing::StatusIs;
using xla::HloSharding;

absl::StatusOr<HloSharding> ToHloShardingViaOpSharding(
    const ShardingParam& sharding_param, absl::Span<const int> device_list) {
  TF_ASSIGN_OR_RETURN(xla::OpSharding op_sharding,
                      ToOpSharding(sharding_param, device_list));
  return HloSharding::FromProto(op_sharding);
}

TEST(ShardingConversionsTest, Replicated) {
  ShardingParam expected_sharding_param{
      /*dim_shards=*/{1, 1, 1},
      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_EXPECT_OK(expected_sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_iota_sharding,
                          ToHloSharding(expected_sharding_param));
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding hlo_sharding,
      ToHloShardingViaOpSharding(expected_sharding_param, {0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(hlo_sharding.ToString(), "{replicated}");
  EXPECT_EQ(hlo_sharding, hlo_iota_sharding);
  TF_ASSERT_OK_AND_ASSIGN(auto sharding_param,
                          ToShardingParam(hlo_iota_sharding, 3, 6));
  // We do not compare expected_sharding_param and sharding_param because they
  // haven't been canonicalized (1x1x1 to [0, 1] on 2x3 vs. 1x1x1 to [0] on 6).
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding actual_hlo_sharding,
                          ToHloSharding(sharding_param));
  EXPECT_EQ(hlo_iota_sharding, actual_hlo_sharding);
}

TEST(ShardingConversionsTest, SingleDeviceReplicated) {
  ShardingParam expected_sharding_param{
      /*dim_shards=*/{1, 1}, {/*permutation=*/{0}, /*axis_sizes=*/{1}}};
  TF_EXPECT_OK(expected_sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_iota_sharding,
                          ToHloSharding(expected_sharding_param));
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding hlo_sharding,
      ToHloShardingViaOpSharding(expected_sharding_param, {0}));
  EXPECT_EQ(hlo_sharding.ToString(), "{replicated}");
  EXPECT_EQ(hlo_sharding, hlo_iota_sharding);
  TF_ASSERT_OK_AND_ASSIGN(auto sharding_param,
                          ToShardingParam(hlo_iota_sharding, 2, 1));
  EXPECT_EQ(expected_sharding_param, sharding_param);
}

TEST(ShardingConversionsTest, Permutation) {
  ShardingParam expected_sharding_param{
      /*dim_shards=*/{2, 1, 3},
      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_EXPECT_OK(expected_sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_iota_sharding,
                          ToHloSharding(expected_sharding_param));
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding hlo_sharding,
      ToHloShardingViaOpSharding(expected_sharding_param, {0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(hlo_sharding.ToString(), "{devices=[2,1,3]0,3,1,4,2,5}");
  EXPECT_EQ(hlo_sharding, hlo_iota_sharding);
  TF_ASSERT_OK_AND_ASSIGN(auto sharding_param,
                          ToShardingParam(hlo_iota_sharding, 3, 6));
  EXPECT_EQ(expected_sharding_param, sharding_param);
}

TEST(ShardingConversionsTest, Partial) {
  ShardingParam expected_sharding_param{
      /*dim_shards=*/{2, 1}, {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_EXPECT_OK(expected_sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_iota_sharding,
                          ToHloSharding(expected_sharding_param));
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding hlo_sharding,
      ToHloShardingViaOpSharding(expected_sharding_param, {0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(hlo_sharding.ToString(),
            "{devices=[2,1,3]0,1,2,3,4,5 last_tile_dim_replicate}");
  EXPECT_EQ(hlo_sharding, hlo_iota_sharding);
  TF_ASSERT_OK_AND_ASSIGN(auto sharding_param,
                          ToShardingParam(hlo_iota_sharding, 2, 6));
  // We do not compare expected_sharding_param and sharding_param because they
  // haven't been canonicalized (2x1 to [0, 1] on 2x3 vs. 2x1 to [0] on 6).
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding actual_hlo_sharding,
                          ToHloSharding(sharding_param));
  EXPECT_EQ(hlo_iota_sharding, actual_hlo_sharding);
}

TEST(ShardingConversionsTest, OneDimToTwoAxes) {
  ShardingParam expected_sharding_param{
      /*dim_shards=*/{4}, {/*permutation=*/{1, 0}, /*axis_sizes=*/{2, 2}}};
  TF_EXPECT_OK(expected_sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_iota_sharding,
                          ToHloSharding(expected_sharding_param));
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding hlo_sharding,
      ToHloShardingViaOpSharding(expected_sharding_param, {0, 1, 2, 3}));
  EXPECT_EQ(hlo_sharding.ToString(), "{devices=[4]0,2,1,3}");
  EXPECT_EQ(hlo_sharding, hlo_iota_sharding);
  TF_ASSERT_OK_AND_ASSIGN(auto sharding_param,
                          ToShardingParam(hlo_iota_sharding, 1, 4));
  EXPECT_EQ(expected_sharding_param, sharding_param);
}

TEST(ShardingConversionsTest, NonTrivialDeviceAssignment) {
  ShardingParam expected_sharding_param{
      /*dim_shards=*/{2, 1, 3},
      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_EXPECT_OK(expected_sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding hlo_sharding,
      ToHloShardingViaOpSharding(expected_sharding_param, {6, 5, 4, 3, 2, 1}));
  EXPECT_EQ(hlo_sharding.ToString(), "{devices=[2,1,3]6,3,5,2,4,1}");
}

TEST(ShardingConversionsTest, VerifyIncorrectShardings) {
  ShardingParam different_permutation_and_axis{
      /*dim_shards=*/{1, 1}, {/*permutation=*/{0, 1}, /*axis_sizes=*/{2}}};
  EXPECT_FALSE(different_permutation_and_axis.verify().ok());
  ShardingParam too_many_slices{/*dim_shards=*/{2, 2},
                                {/*permutation=*/{0}, /*axis_sizes=*/{2}}};
  EXPECT_FALSE(too_many_slices.verify().ok());
  ShardingParam incorrect_permutation{
      /*dim_shards=*/{4, 1},
      {/*permutation=*/{0, 1, 1}, /*axis_sizes=*/{2, 2, 2}}};
  EXPECT_FALSE(incorrect_permutation.verify().ok());
}

TEST(ShardingConversionsTest, ErrorOnDeviceAssignment) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 1, 3},
                               {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_EXPECT_OK(sharding_param.verify());
  EXPECT_THAT(
      ToHloShardingViaOpSharding(sharding_param, {6, 5, 4, 3, 2}),
      StatusIs(absl::StatusCode::kOutOfRange,
               ::testing::HasSubstr("Can't map device with logical id 5")));
}

struct HloShardingTestStruct {
  HloSharding hlo_sharding;
  int rank;
  int num_devices;
};

using HloShardingToShardingParamTest =
    ::testing::TestWithParam<HloShardingTestStruct>;

TEST_P(HloShardingToShardingParamTest, HloShardingToShardingParam) {
  const auto& param = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto sharding_param,
      ToShardingParam(param.hlo_sharding, param.rank, param.num_devices));
  EXPECT_TRUE(sharding_param.verify().ok());
  TF_ASSERT_OK_AND_ASSIGN(auto actual_hlo_sharding,
                          ToHloSharding(sharding_param));
  EXPECT_EQ(param.hlo_sharding, actual_hlo_sharding);
  // Verify that the conversion to OpSharding is also correct.
  std::vector<int> device_ids(param.num_devices);
  std::iota(device_ids.begin(), device_ids.end(), 0);
  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_via_op_sharding,
      ToHloShardingViaOpSharding(sharding_param, device_ids));
  EXPECT_EQ(param.hlo_sharding, hlo_via_op_sharding);
}

INSTANTIATE_TEST_SUITE_P(
    HloShardingConversionTests, HloShardingToShardingParamTest,
    testing::ValuesIn<HloShardingTestStruct>({
        {HloSharding::IotaTile({4, 2}), 2, 8},
        {HloSharding::IotaTile({2, 4}, {4, 2}, {1, 0}), 2, 8},
        {HloSharding::IotaTile({8, 1}), 2, 8},
        {HloSharding::IotaTile({8, 1}, {4, 2}, {1, 0}), 2, 8},
        {HloSharding::PartialTile(TileAssignment({4, 1, 2}, {8}, {0})), 2, 8},
        {HloSharding::PartialTile(TileAssignment({2, 1, 4}, {4, 2}, {1, 0})), 2,
         8},
        {HloSharding::PartialTile(TileAssignment({1, 4, 2}, {8}, {0})), 2, 8},
        {HloSharding::PartialTile(TileAssignment({1, 2, 4}, {4, 2}, {1, 0})), 2,
         8},
        {HloSharding::PartialTile(TileAssignment({4, 3, 2}, {2, 3, 4},
                                                 {2, 1, 0})),
         2, 24},
        {HloSharding::PartialTile(TileAssignment({4, 2, 3}, {6, 4}, {1, 0})), 2,
         24},
        {HloSharding::PartialTile(TileAssignment({6, 1, 4}, {24}, {0})), 2, 24},
        {HloSharding::PartialTile(TileAssignment({12, 1, 2}, {2, 12}, {1, 0})),
         2, 24},
        {HloSharding::PartialTile(TileAssignment({8, 1, 3}, {6, 4}, {1, 0})), 2,
         24},
        {HloSharding::PartialTile(TileAssignment({2, 1, 12}, {24}, {0})), 2,
         24},
        {HloSharding::PartialTile(TileAssignment({3, 1, 8}, {2, 3, 4},
                                                 {1, 0, 2})),
         2, 24},
        {HloSharding::PartialTile(TileAssignment({1, 4, 6}, {6, 4}, {1, 0})), 2,
         24},
        {HloSharding::PartialTile(TileAssignment({1, 12, 2}, {2, 12}, {1, 0})),
         2, 24},

        {HloSharding::PartialTile(TileAssignment({3, 2, 1, 4}, {2, 3, 4},
                                                 {1, 0, 2})),
         3, 24},
        {HloSharding::PartialTile(TileAssignment({2, 4, 1, 3}, {2, 3, 4},
                                                 {0, 2, 1})),
         3, 24},
        {HloSharding::PartialTile(TileAssignment({4, 3, 1, 2}, {2, 3, 4},
                                                 {2, 1, 0})),
         3, 24},
        {HloSharding::PartialTile(TileAssignment({12, 1, 1, 2}, {2, 12},
                                                 {1, 0})),
         3, 24},
    }));

class ShardingConversionsEquivalentTest : public test_util::ShardingTest {
 public:
  void AssertSameTiling(const ShardingParam& sharding_param,
                        const HloSharding& hlo_sharding, const Shape& shape) {
    auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
    TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const Sharding> sharding,
                            ShardingParamSharding::Create(
                                sharding_param, device_list, MemoryKind()));
    const xla::Shape xla_shape(PrimitiveType::F16, shape.dims(), {}, {});

    TF_ASSERT_OK_AND_ASSIGN(const std::vector<IndexDomain> index_domains,
                            sharding->IndexDomains(shape));
    ASSERT_EQ(index_domains.size(),
              hlo_sharding.tile_assignment().num_elements());
    const xla::Shape xla_tile_shape = hlo_sharding.TileShape(xla_shape);
    for (int i = 0; i < index_domains.size(); ++i) {
      SCOPED_TRACE(absl::StrCat("on device ", i));
      EXPECT_EQ(index_domains[i].origin().elements(),
                hlo_sharding.TileOffsetForDevice(xla_shape, i));
      EXPECT_EQ(index_domains[i].shape().dims(), xla_tile_shape.dimensions());
    }
  }

 private:
  std::shared_ptr<Client> client_;
};

TEST_P(ShardingConversionsEquivalentTest, ShardingParamFullySharded) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 3},
                               {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_EXPECT_OK(sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding hlo_sharding,
      ToHloShardingViaOpSharding(sharding_param, {0, 1, 2, 3, 4, 5}));
  AssertSameTiling(sharding_param, hlo_sharding, Shape({6, 6}));
}

TEST_P(ShardingConversionsEquivalentTest, ShardingParamWithPermutation) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 3},
                               {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_EXPECT_OK(sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding hlo_sharding,
      ToHloShardingViaOpSharding(sharding_param, {0, 1, 2, 3, 4, 5}));
  AssertSameTiling(sharding_param, hlo_sharding, Shape({6, 6}));
}

TEST_P(ShardingConversionsEquivalentTest, ShardingParamWithReplication) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 1},
                               {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_EXPECT_OK(sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding hlo_sharding,
      ToHloShardingViaOpSharding(sharding_param, {0, 1, 2, 3, 4, 5}));
  AssertSameTiling(sharding_param, hlo_sharding, Shape({6, 6}));
}

TEST_P(ShardingConversionsEquivalentTest, OpShardingReplicated) {
  OpSharding op_sharding;
  op_sharding.set_type(OpSharding::REPLICATED);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          HloSharding::FromProto(op_sharding));
  TF_ASSERT_OK_AND_ASSIGN(auto actual, ToShardingParam(hlo_sharding, 2, 6));
  ShardingParam expected{/*dim_shards=*/{1, 1},
                         {/*permutation=*/{0}, /*axis_sizes=*/{6}}};
  TF_EXPECT_OK(expected.verify());
  EXPECT_EQ(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(NumDevices, ShardingConversionsEquivalentTest,
                         testing::Values(test_util::ShardingTestParam{
                             .num_devices = 6, .num_addressable_devices = 4}));

}  // namespace
}  // namespace support
}  // namespace ifrt
}  // namespace xla
