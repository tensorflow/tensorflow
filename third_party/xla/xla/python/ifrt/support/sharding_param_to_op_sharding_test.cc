/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/python/ifrt/support/sharding_param_to_op_sharding.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_test_util.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace support {
namespace {

using ::tsl::testing::StatusIs;

StatusOr<xla::HloSharding> ToHloSharding(const ShardingParam& sharding_param,
                                         absl::Span<const int> device_list) {
  TF_ASSIGN_OR_RETURN(xla::OpSharding op_sharding,
                      ToOpSharding(sharding_param, device_list));
  return xla::HloSharding::FromProto(op_sharding);
}

TEST(ShardingParamToOpShardingTest, Replicated) {
  ShardingParam sharding_param{/*dim_shards=*/{1, 1, 1},
                               {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(const xla::HloSharding actual,
                          ToHloSharding(sharding_param, {0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(actual.ToString(), "{replicated}");
}

TEST(ShardingParamToOpShardingTest, Maximal) {
  ShardingParam sharding_param{/*dim_shards=*/{1, 1},
                               {/*permutation=*/{0}, /*axis_sizes=*/{1}}};
  TF_ASSERT_OK_AND_ASSIGN(const xla::HloSharding actual,
                          ToHloSharding(sharding_param, {0}));
  EXPECT_EQ(actual.ToString(), "{maximal device=0}");
}

TEST(ShardingParamToOpShardingTest, Permutation) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 1, 3},
                               {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(const xla::HloSharding actual,
                          ToHloSharding(sharding_param, {0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(actual.ToString(), "{devices=[2,1,3]0,3,1,4,2,5}");
}

TEST(ShardingParamToOpShardingTest, Partial) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 1},
                               {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(const xla::HloSharding actual,
                          ToHloSharding(sharding_param, {0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(actual.ToString(),
            "{devices=[2,1,3]0,1,2,3,4,5 last_tile_dim_replicate}");
}

TEST(ShardingParamToOpShardingTest, OneDimToTwoAxes) {
  ShardingParam sharding_param{/*dim_shards=*/{4},
                               {/*permutation=*/{1, 0}, /*axis_sizes=*/{2, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(const xla::HloSharding actual,
                          ToHloSharding(sharding_param, {0, 1, 2, 3}));
  EXPECT_EQ(actual.ToString(), "{devices=[4]0,2,1,3}");
}

TEST(ShardingParamToOpShardingTest, NonTrivialDeviceAssignment) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 1, 3},
                               {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(const xla::HloSharding actual,
                          ToHloSharding(sharding_param, {6, 5, 4, 3, 2, 1}));
  EXPECT_EQ(actual.ToString(), "{devices=[2,1,3]6,3,5,2,4,1}");
}

TEST(ShardingParamToOpShardingTest, ErrorOnDeviceAssignment) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 1, 3},
                               {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  EXPECT_THAT(ToHloSharding(sharding_param, {6, 5, 4, 3, 2}),
              StatusIs(tsl::error::OUT_OF_RANGE, "Can't map device 5"));
}

class ShardingParamToOpShardingEquivalentTest : public test_util::ShardingTest {
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

TEST_P(ShardingParamToOpShardingEquivalentTest, FullySharded) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 3},
                               {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(const xla::HloSharding hlo_sharding,
                          ToHloSharding(sharding_param, {0, 1, 2, 3, 4, 5}));
  AssertSameTiling(sharding_param, hlo_sharding, Shape({6, 6}));
}

TEST_P(ShardingParamToOpShardingEquivalentTest, WithPermutation) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 3},
                               {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(const xla::HloSharding hlo_sharding,
                          ToHloSharding(sharding_param, {0, 1, 2, 3, 4, 5}));
  AssertSameTiling(sharding_param, hlo_sharding, Shape({6, 6}));
}

TEST_P(ShardingParamToOpShardingEquivalentTest, WithReplication) {
  ShardingParam sharding_param{/*dim_shards=*/{2, 1},
                               {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(const xla::HloSharding hlo_sharding,
                          ToHloSharding(sharding_param, {0, 1, 2, 3, 4, 5}));
  AssertSameTiling(sharding_param, hlo_sharding, Shape({6, 6}));
}

INSTANTIATE_TEST_SUITE_P(NumDevices, ShardingParamToOpShardingEquivalentTest,
                         testing::Values(test_util::ShardingTestParam{
                             .num_devices = 6, .num_addressable_devices = 4}));

}  // namespace
}  // namespace support
}  // namespace ifrt
}  // namespace xla
