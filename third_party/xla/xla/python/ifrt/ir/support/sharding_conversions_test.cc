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

#include "xla/python/ifrt/ir/support/sharding_conversions.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/shape.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace support {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Return;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using ::xla::HloSharding;

absl::StatusOr<HloSharding> ToHloShardingViaOpSharding(
    const ShardingParam& sharding_param) {
  TF_ASSIGN_OR_RETURN(xla::OpSharding op_sharding,
                      ToOpSharding(sharding_param));
  return HloSharding::FromProto(op_sharding);
}

// Internal state of a client for sharding conversion tests.
struct ShardingConversionTestClientState {
  absl::flat_hash_map<DeviceId, std::unique_ptr<Device>> device_map;
  std::vector<Device*> devices;
};

// Creates a mock client for sharding tests. The client will have a specified
// number of fake devices. Client implements `devices()`, and Device implements
// `Id()`, with iota device ids assignment.
std::shared_ptr<MockClient> MakeTestClient(int num_devices) {
  auto state = std::make_shared<ShardingConversionTestClientState>();
  state->devices.reserve(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    auto device = std::make_unique<MockDevice>();
    ON_CALL(*device, Id).WillByDefault(Return(DeviceId(i)));
    ON_CALL(*device, IsAddressable).WillByDefault(Return(true));
    state->devices.push_back(device.get());
    state->device_map.insert({DeviceId(i), std::move(device)});
  }
  auto client = std::make_shared<MockClient>();
  ON_CALL(*client, devices)
      .WillByDefault(
          [state]() -> absl::Span<Device* const> { return state->devices; });
  ON_CALL(*client, MakeDeviceList)
      .WillByDefault([](absl::Span<Device* const> devices) -> DeviceListRef {
        return BasicDeviceList::Create(devices);
      });
  return client;
}

class ShardingConversionsTest : public testing::TestWithParam<int> {
 public:
  void SetUp() override { client_ = MakeTestClient(GetParam()); }

  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return test_util::GetDevices(client_.get(), device_indices).value();
  }

  void AssertSameTiling(const ShardingParam& sharding_param,
                        const HloSharding& hlo_sharding, const Shape& shape) {
    auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
    TF_ASSERT_OK_AND_ASSIGN(ShardingRef sharding,
                            ShardingParamSharding::Create(
                                sharding_param, device_list, MemoryKind()));
    const xla::Shape xla_shape(PrimitiveType::F16, shape.dims());

    TF_ASSERT_OK_AND_ASSIGN(
        const std::vector<IndexDomain> index_domains,
        sharding->IndexDomains(
            shape, xla::ifrt::SingleDeviceShardSemantics::kAllShards));
    ASSERT_EQ(index_domains.size(), hlo_sharding.num_devices());
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

struct ShardingParamToHloShardingTestParam {
  ShardingParam sharding_param;
  std::string expected_hlo_sharding_str;
};

using ShardingParamToHloShardingTest =
    ::testing::TestWithParam<ShardingParamToHloShardingTestParam>;

TEST_P(ShardingParamToHloShardingTest, ShardingParamToHloSharding) {
  const ShardingParamToHloShardingTestParam& param = GetParam();
  TF_EXPECT_OK(param.sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_iota_sharding,
                          ToHloSharding(param.sharding_param));
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_sharding,
                          ToHloShardingViaOpSharding(param.sharding_param));
  EXPECT_EQ(hlo_sharding.ToString(), param.expected_hlo_sharding_str);
  EXPECT_EQ(hlo_sharding, hlo_iota_sharding);
  TF_ASSERT_OK_AND_ASSIGN(
      auto sharding_param,
      ToShardingParam(hlo_iota_sharding,
                      param.sharding_param.dim_shards().size(),
                      param.sharding_param.NumDevices()));

  if (param.sharding_param.minor_to_major().axis_sizes.size() ==
          hlo_sharding.num_dimensions() &&
      hlo_sharding.subgroup_types().size() < 2) {
    // We compare expected_sharding_param and sharding_param if the HloSharding
    // has not canonicalized the axes and if there is at most one subgroup type.
    // HloShardings with multiple subgroup types may have canonicalized axes
    // that have multiple valid expansions.
    EXPECT_EQ(sharding_param, param.sharding_param);
  }
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding actual_hlo_sharding,
                          ToHloSharding(sharding_param));
  EXPECT_EQ(hlo_iota_sharding, actual_hlo_sharding);
}

INSTANTIATE_TEST_SUITE_P(
    ShardingParamToHloSharding, ShardingParamToHloShardingTest,
    testing::ValuesIn<ShardingParamToHloShardingTestParam>(
        {{ShardingParam{/*dim_shards=*/{1, 1, 1},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          "{replicated}"},
         {ShardingParam{/*dim_shards=*/{1, 1},
                        {/*permutation=*/{0}, /*axis_sizes=*/{1}}},
          "{replicated}"},
         {ShardingParam{/*dim_shards=*/{1, 1, 1},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}},
                        /*unreduced_axes=*/{0, 1}},
          "{unreduced}"},
         {ShardingParam{/*dim_shards=*/{},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}},
                        /*unreduced_axes=*/{0, 1}},
          "{unreduced}"},
         {ShardingParam{/*dim_shards=*/{2, 1, 3},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}},
          "{devices=[2,1,3]<=[2,3]T(1,0)}"},
         {ShardingParam{/*dim_shards=*/{4},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{2, 2}}},
          "{devices=[4]<=[2,2]T(1,0)}"},
         {ShardingParam{/*dim_shards=*/{2, 1},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          "{devices=[2,1,3]<=[6] last_tile_dim_replicate}"},
         {ShardingParam{/*dim_shards=*/{1, 1, 3},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{3, 2}},
                        /*unreduced_axes=*/{1}},
          "{devices=[1,1,3,2]<=[6] last_tile_dims={unreduced}}"},
         {ShardingParam{/*dim_shards=*/{3, 1},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}},
                        /*unreduced_axes=*/{1}},
          "{devices=[3,1,2]<=[2,3]T(1,0) last_tile_dims={unreduced}}"},
         {ShardingParam{/*dim_shards=*/{6, 1},
                        {/*permutation=*/{3, 1, 2, 0},
                         /*axis_sizes=*/{3, 5, 2, 7}},
                        /*unreduced_axes=*/{1, 3}},
          "{devices=[6,1,35]<=[7,10,3]T(2,1,0) last_tile_dims={unreduced}}"},
         {ShardingParam{/*dim_shards=*/{1, 1, 1},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}},
                        /*unreduced_axes=*/{0}},
          "{devices=[1,1,1,2,3]<=[3,2]T(1,0) "
          "last_tile_dims={unreduced, replicated}}"},
         {ShardingParam{/*dim_shards=*/{3},
                        {/*permutation=*/{1, 0, 2},
                         /*axis_sizes=*/{2, 2, 3}},
                        /*unreduced_axes=*/{0}},
          "{devices=[3,2,2]<=[3,2,2]T(0,2,1) "
          "last_tile_dims={unreduced, replicated}}"},
         {ShardingParam{/*dim_shards=*/{3, 1},
                        {/*permutation=*/{3, 1, 2, 0},
                         /*axis_sizes=*/{3, 5, 2, 7}},
                        /*unreduced_axes=*/{2}},
          "{devices=[3,1,2,35]<=[7,2,5,3]T(3,1,0,2) "
          "last_tile_dims={unreduced, replicated}}"},
         {ShardingParam{/*dim_shards=*/{2, 3},
                        {/*permutation=*/{6, 1, 3, 7, 2, 0, 5, 4},
                         /*axis_sizes=*/{3, 3, 6, 5, 2, 3, 4, 3}},
                        /*unreduced_axes=*/{0, 1, 3}},
          "{devices=[2,3,45,72]<=[12,3,2,5,6,3,3]T(2,1,6,3,5,0,4) "
          "last_tile_dims={unreduced, replicated}}"},
         {ShardingParam{/*dim_shards=*/{2, 3},
                        {/*permutation=*/{6, 1, 3, 7, 2, 0, 5, 4},
                         /*axis_sizes=*/{3, 3, 6, 5, 2, 3, 4, 3}},
                        /*unreduced_axes=*/{0, 3}},
          "{devices=[2,3,15,216]<=[12,3,2,5,18,3]T(2,1,5,3,0,4) "
          "last_tile_dims={unreduced, replicated}}"}}));

TEST_P(ShardingConversionsTest, VerifyIncorrectShardings) {
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
  ShardingParam unreduced_repeated_axis{
      /*dim_shards=*/{1, 1},
      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 2}},
      /*unreduced_axes=*/{0, 1, 0}};
  EXPECT_FALSE(unreduced_repeated_axis.verify().ok());
  ShardingParam unreduced_out_of_bounds_axis{/*dim_shards=*/{1, 1},
                                             {/*permutation=*/{0, 1},
                                              /*axis_sizes=*/{2, 2}},
                                             /*unreduced_axes=*/{2}};
  EXPECT_FALSE(unreduced_out_of_bounds_axis.verify().ok());
}

TEST_P(ShardingConversionsTest, ShardingParamWithWrongUnreducedAxisError) {
  ShardingParam sharding_param{/*dim_shards=*/{3, 1},
                               {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}},
                               /*unreduced_axes=*/{0}};
  auto hlo_sharding = ToHloShardingViaOpSharding(sharding_param);
  EXPECT_THAT(hlo_sharding, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(ShardingConversionsTest, ShardingParamFullySharded) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 3},
                               {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_EXPECT_OK(sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_sharding,
                          ToHloShardingViaOpSharding(sharding_param));
  AssertSameTiling(sharding_param, hlo_sharding, Shape({6, 6}));
}

TEST_P(ShardingConversionsTest, ShardingParamWithPermutation) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 3},
                               {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_EXPECT_OK(sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_sharding,
                          ToHloShardingViaOpSharding(sharding_param));
  AssertSameTiling(sharding_param, hlo_sharding, Shape({6, 6}));
}

TEST_P(ShardingConversionsTest, ShardingParamWithReplication) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 1},
                               {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_EXPECT_OK(sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_sharding,
                          ToHloShardingViaOpSharding(sharding_param));
  AssertSameTiling(sharding_param, hlo_sharding, Shape({6, 6}));
}

TEST_P(ShardingConversionsTest, OpShardingReplicated) {
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

TEST_P(ShardingConversionsTest, OpShardingUnreduced) {
  OpSharding op_sharding;
  op_sharding.set_type(OpSharding::UNREDUCED);
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_sharding,
                          HloSharding::FromProto(op_sharding));
  TF_ASSERT_OK_AND_ASSIGN(ShardingParam actual,
                          ToShardingParam(hlo_sharding, 2, 6));
  ShardingParam expected{/*dim_shards=*/{1, 1},
                         {/*permutation=*/{0}, /*axis_sizes=*/{6}},
                         /*unreduced_axes=*/{0}};
  TF_EXPECT_OK(expected.verify());
  EXPECT_EQ(actual, expected);
}

TEST_P(ShardingConversionsTest, OpShardingWithUnreduced) {
  OpSharding op_sharding = ParseTextProtoOrDie<OpSharding>(R"pb(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 5
    tile_assignment_dimensions: 3
    iota_reshape_dims: 5
    iota_reshape_dims: 3
    iota_reshape_dims: 2
    iota_transpose_perm: 2
    iota_transpose_perm: 0
    iota_transpose_perm: 1
    last_tile_dims: UNREDUCED
    last_tile_dims: REPLICATED
  )pb");
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_sharding,
                          HloSharding::FromProto(op_sharding));
  TF_ASSERT_OK_AND_ASSIGN(ShardingParam actual,
                          ToShardingParam(hlo_sharding, 3, 30));
  ShardingParam expected{/*dim_shards=*/{1, 1, 2},
                         {/*permutation=*/{1, 2, 0}, /*axis_sizes=*/{2, 3, 5}},
                         /*unreduced_axes=*/{2}};
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding expected_hlo_sharding,
                          ToHloSharding(expected));
  TF_EXPECT_OK(expected.verify());
  TF_EXPECT_OK(actual.verify());
  EXPECT_EQ(hlo_sharding, expected_hlo_sharding);
}

INSTANTIATE_TEST_SUITE_P(NumDevices, ShardingConversionsTest,
                         testing::Values(7));

struct HloShardingTestStruct {
  HloSharding hlo_sharding;
  int rank;
  int num_devices;
};

class HloShardingToShardingParamTest
    : public testing::TestWithParam<HloShardingTestStruct> {
 public:
  void SetUp() override {
    const auto& param = GetParam();
    client_ = MakeTestClient(param.num_devices);
  }

  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return test_util::GetDevices(client_.get(), device_indices).value();
  }

 private:
  std::shared_ptr<Client> client_;
};

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
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_via_op_sharding,
                          ToHloShardingViaOpSharding(sharding_param));
  EXPECT_EQ(param.hlo_sharding, hlo_via_op_sharding);
}

INSTANTIATE_TEST_SUITE_P(
    HloShardingConversionTests, HloShardingToShardingParamTest,
    testing::ValuesIn<HloShardingTestStruct>(
        {{HloSharding::IotaTile({4, 2}), 2, 8},
         {HloSharding::IotaTile({2, 4}, {4, 2}, {1, 0}), 2, 8},
         {HloSharding::IotaTile({8, 1}), 2, 8},
         {HloSharding::IotaTile({8, 1}, {4, 2}, {1, 0}), 2, 8},
         {HloSharding::PartialTile(TileAssignment({4, 1, 2}, {8}, {0})), 2, 8},
         {HloSharding::PartialTile(TileAssignment({2, 1, 4}, {4, 2}, {1, 0})),
          2, 8},
         {HloSharding::PartialTile(TileAssignment({1, 4, 2}, {8}, {0})), 2, 8},
         {HloSharding::PartialTile(TileAssignment({1, 2, 4}, {4, 2}, {1, 0})),
          2, 8},
         {HloSharding::PartialTile(TileAssignment({4, 3, 2}, {2, 3, 4},
                                                  {2, 1, 0})),
          2, 24},
         {HloSharding::PartialTile(TileAssignment({4, 2, 3}, {6, 4}, {1, 0})),
          2, 24},
         {HloSharding::PartialTile(TileAssignment({6, 1, 4}, {24}, {0})), 2,
          24},
         {HloSharding::PartialTile(TileAssignment({12, 1, 2}, {2, 12}, {1, 0})),
          2, 24},
         {HloSharding::PartialTile(TileAssignment({8, 1, 3}, {6, 4}, {1, 0})),
          2, 24},
         {HloSharding::PartialTile(TileAssignment({2, 1, 12}, {24}, {0})), 2,
          24},
         {HloSharding::PartialTile(TileAssignment({3, 1, 8}, {2, 3, 4},
                                                  {1, 0, 2})),
          2, 24},
         {HloSharding::PartialTile(TileAssignment({1, 4, 6}, {6, 4}, {1, 0})),
          2, 24},
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
         {HloSharding::Unreduced(), 3, 24},
         {HloSharding::Subgroup(TileAssignment({1, 1, 8}, {4, 2}, {1, 0}),
                                {OpSharding::UNREDUCED}),
          2, 8},
         {HloSharding::Subgroup(TileAssignment({4, 1, 2}, {8}, {0}),
                                {OpSharding::UNREDUCED}),
          2, 8},
         {HloSharding::Subgroup(TileAssignment({2, 1, 4}, {4, 2}, {1, 0}),
                                {OpSharding::UNREDUCED}),
          2, 8},
         {HloSharding::Subgroup(TileAssignment({4, 3, 2}, {2, 3, 4}, {2, 1, 0}),
                                {OpSharding::UNREDUCED}),
          2, 24},
         {HloSharding::Subgroup(TileAssignment({12, 1, 2}, {2, 12}, {1, 0}),
                                {OpSharding::UNREDUCED}),
          2, 24},
         {HloSharding::Subgroup(TileAssignment({8, 1, 3}, {6, 4}, {1, 0}),
                                {OpSharding::UNREDUCED}),
          2, 24},
         {HloSharding::Subgroup(TileAssignment({2, 1, 12}, {24}, {0}),
                                {OpSharding::UNREDUCED}),
          2, 24},
         {HloSharding::Subgroup(TileAssignment({3, 1, 8}, {2, 3, 4}, {1, 0, 2}),
                                {OpSharding::UNREDUCED}),
          2, 24},
         {HloSharding::Subgroup(TileAssignment({1, 12, 2}, {2, 12}, {1, 0}),
                                {OpSharding::UNREDUCED}),
          2, 24},
         {HloSharding::Subgroup(TileAssignment({3, 2, 1, 4}, {2, 3, 4},
                                               {1, 0, 2}),
                                {OpSharding::UNREDUCED}),
          3, 24},
         {HloSharding::Subgroup(TileAssignment({2, 4, 1, 3}, {2, 3, 4},
                                               {0, 2, 1}),
                                {OpSharding::UNREDUCED}),
          3, 24},
         {HloSharding::Subgroup(TileAssignment({4, 3, 1, 2}, {2, 3, 4},
                                               {2, 1, 0}),
                                {OpSharding::UNREDUCED}),
          3, 24},
         {HloSharding::Subgroup(TileAssignment({12, 1, 1, 2}, {2, 12}, {1, 0}),
                                {OpSharding::UNREDUCED}),
          3, 24},
         {HloSharding::Subgroup(TileAssignment({1, 1, 2, 4}, {4, 2}, {1, 0}),
                                {OpSharding::REPLICATED,
                                 OpSharding::UNREDUCED}),
          2, 8},
         {HloSharding::Subgroup(TileAssignment({4, 2, 3}, {6, 4}, {1, 0}),
                                {OpSharding::REPLICATED,
                                 OpSharding::UNREDUCED}),
          1, 24},
         {HloSharding::Subgroup(TileAssignment({6, 1, 4}, {24}, {0}),
                                {OpSharding::UNREDUCED,
                                 OpSharding::REPLICATED}),
          1, 24},
         {HloSharding::Subgroup(TileAssignment({1, 4, 2}, {8}, {0}),
                                {OpSharding::UNREDUCED,
                                 OpSharding::REPLICATED}),
          1, 8},
         {HloSharding::Subgroup(
              TileAssignment({3, 2, 5, 7}, {7, 10, 3}, {2, 1, 0}),
              {OpSharding::UNREDUCED, OpSharding::REPLICATED}),
          2, 210},
         {HloSharding::Subgroup(TileAssignment({1, 4, 3, 2}, {6, 4}, {1, 0}),
                                {OpSharding::REPLICATED,
                                 OpSharding::UNREDUCED}),
          2, 24},
         {HloSharding::Subgroup(TileAssignment({2, 3, 4}, {2, 3, 4}, {0, 1, 2}),
                                {OpSharding::REPLICATED,
                                 OpSharding::UNREDUCED}),
          1, 24},
         {HloSharding::Subgroup(
              TileAssignment({4, 1, 3, 2}, {2, 3, 4}, {2, 1, 0}),
              {OpSharding::REPLICATED, OpSharding::UNREDUCED}),
          2, 24}}));

struct ShardingParamToHloShardingWithDeviceIdsTestParam {
  ShardingParam sharding_param;
  std::vector<int> device_ids;
  std::string expected_hlo_sharding_str;
};

using ShardingParamToHloShardingWithDeviceIdsTest =
    ::testing::TestWithParam<ShardingParamToHloShardingWithDeviceIdsTestParam>;

TEST_P(ShardingParamToHloShardingWithDeviceIdsTest,
       ShardingParamToHloShardingWithDeviceIds) {
  const auto& param = GetParam();
  TF_EXPECT_OK(param.sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding hlo_sharding,
                          ToHloSharding(param.sharding_param,
                                        llvm::ArrayRef<int>(param.device_ids)));
  EXPECT_EQ(hlo_sharding.ToString(), param.expected_hlo_sharding_str);
  if (hlo_sharding.IsTiled()) {
    EXPECT_FALSE(hlo_sharding.tile_assignment().iota().has_value());
  }
}

INSTANTIATE_TEST_SUITE_P(
    ShardingParamToHloShardingWithDeviceIds,
    ShardingParamToHloShardingWithDeviceIdsTest,
    testing::ValuesIn<ShardingParamToHloShardingWithDeviceIdsTestParam>(
        {{ShardingParam{/*dim_shards=*/{1, 1, 1},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          {5, 4, 3, 2, 1, 0},
          "{replicated}"},
         {ShardingParam{/*dim_shards=*/{2, 3},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          {5, 4, 3, 2, 1, 0},
          "{devices=[2,3]5,4,3,2,1,0}"},
         {ShardingParam{/*dim_shards=*/{2, 1, 3},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}},
          {3, 0, 5, 4, 1, 2},
          "{devices=[2,1,3]3,4,0,1,5,2}"},
         {ShardingParam{/*dim_shards=*/{2, 1},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          {5, 4, 3, 2, 1, 0},
          "{devices=[2,1,3]3,4,5,0,1,2 last_tile_dim_replicate}"},
         {ShardingParam{/*dim_shards=*/{4},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{2, 2}}},
          {3, 1, 2, 0},
          "{devices=[4]3,2,1,0}"},
         {ShardingParam{/*dim_shards=*/{2, 3},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          {0, 1, 2, 3, 4, 5},
          "{devices=[2,3]0,1,2,3,4,5}"}}));

struct HloShardingV1RoundtripTestParam {
  ShardingParam sharding_param;
  std::vector<int> device_ids;
  int rank;
  int num_devices;
};

using HloShardingV1RoundtripTest =
    ::testing::TestWithParam<HloShardingV1RoundtripTestParam>;

TEST_P(HloShardingV1RoundtripTest, Roundtrip) {
  const auto& param = GetParam();
  TF_EXPECT_OK(param.sharding_param.verify());
  TF_ASSERT_OK_AND_ASSIGN(const HloSharding v1_sharding,
                          ToHloSharding(param.sharding_param,
                                        llvm::ArrayRef<int>(param.device_ids)));
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingParamWithDeviceIds result,
      ToShardingParamAndDevices(v1_sharding, param.rank, param.num_devices));
  std::optional<llvm::ArrayRef<int>> logical_device_ids = std::nullopt;
  if (result.logical_device_ids.has_value()) {
    logical_device_ids.emplace(*result.logical_device_ids);
  }
  TF_ASSERT_OK_AND_ASSIGN(
      const HloSharding roundtrip_sharding,
      ToHloSharding(result.sharding_param, logical_device_ids));
  EXPECT_EQ(roundtrip_sharding.ToString(), v1_sharding.ToString());
}

INSTANTIATE_TEST_SUITE_P(
    HloShardingV1Roundtrip, HloShardingV1RoundtripTest,
    testing::ValuesIn<HloShardingV1RoundtripTestParam>(
        {{ShardingParam{/*dim_shards=*/{2, 3},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          {5, 4, 3, 2, 1, 0},
          2,
          6},
         {ShardingParam{/*dim_shards=*/{2, 1, 3},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}},
          {3, 0, 5, 4, 1, 2},
          3,
          6},
         {ShardingParam{/*dim_shards=*/{2, 1},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          {5, 4, 3, 2, 1, 0},
          2,
          6},
         {ShardingParam{/*dim_shards=*/{4},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{2, 2}}},
          {3, 1, 2, 0},
          1,
          4},
         {ShardingParam{/*dim_shards=*/{2, 3},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          {0, 1, 2, 3, 4, 5},
          2,
          6},
         {ShardingParam{/*dim_shards=*/{1, 1, 1},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}},
          {5, 4, 3, 2, 1, 0},
          3,
          6}}));

}  // namespace
}  // namespace support
}  // namespace ifrt
}  // namespace xla
