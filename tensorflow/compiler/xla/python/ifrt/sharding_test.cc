/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/sharding.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/Casting.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/ifrt/ir/sharding_param.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

DeviceList CreateDummyDevices(int count) {
  DeviceList::Devices devices;
  devices.reserve(count);
  for (int i = 0; i < count; ++i) {
    devices.push_back(reinterpret_cast<Device*>(i + 1));
  }
  return DeviceList(std::move(devices));
}

TEST(SingleDeviceShardingTest, IndexDomains) {
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(reinterpret_cast<Device*>(1));

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, sharding->IndexDomains(shape));
  EXPECT_THAT(index_domains, ElementsAre(IndexDomain(shape)));
}

TEST(SingleDeviceShardingTest, Disassemble) {
  auto device = reinterpret_cast<Device*>(1);
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto disassembled, sharding->Disassemble(shape));

  ASSERT_THAT(disassembled, SizeIs(1));
  const auto& [result_shape, result_sharding] = disassembled[0];
  ASSERT_EQ(shape, result_shape);
  ASSERT_TRUE(llvm::isa<SingleDeviceSharding>(*result_sharding));
  EXPECT_THAT(result_sharding->devices().devices(), ElementsAre(device));
}

TEST(OpaqueShardingTest, FailedToDisassemble) {
  DeviceList device_list = CreateDummyDevices(2);
  std::shared_ptr<const Sharding> sharding =
      OpaqueSharding::Create(device_list);

  EXPECT_THAT(
      sharding->Disassemble(Shape({30})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("OpaqueSharding does not have shard shape information")));
}

TEST(OpaqueShardingTest, IndexDomainsFails) {
  DeviceList device_list = CreateDummyDevices(2);
  std::shared_ptr<const Sharding> sharding =
      OpaqueSharding::Create(device_list);

  EXPECT_THAT(
      sharding->IndexDomains(Shape({30})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("OpaqueSharding does not have index domain information")));
}

TEST(ConcreteShardingTest, Disassemble) {
  DeviceList device_list = CreateDummyDevices(2);
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  std::shared_ptr<const Sharding> sharding =
      ConcreteSharding::Create(device_list, Shape({30}), shard_shapes);

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                          sharding->Disassemble(Shape({30})));
  ASSERT_THAT(disassembled, SizeIs(2));
  for (int i = 0; i < 2; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, shard_shapes[i]);
    EXPECT_TRUE(llvm::isa<SingleDeviceSharding>(*sharding));
    EXPECT_THAT(sharding->devices().devices(),
                ElementsAre(device_list.devices()[i]));
  }
}

TEST(ConcreteShardingTest, DisassembleFailsForUnexpectedShape) {
  DeviceList device_list = CreateDummyDevices(2);
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  std::shared_ptr<const Sharding> sharding =
      ConcreteSharding::Create(device_list, Shape({30}), shard_shapes);

  EXPECT_THAT(sharding->Disassemble(Shape({40})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteSharding can only disassemble")));
}

TEST(ConcreteShardingTest, IndexDomainsFails) {
  DeviceList device_list = CreateDummyDevices(2);
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  std::shared_ptr<const Sharding> sharding =
      ConcreteSharding::Create(device_list, Shape({30}), shard_shapes);

  EXPECT_THAT(sharding->IndexDomains(Shape({30})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteSharding does not have index "
                                 "domain information")));
}

TEST(ConcreteEvenShardingTest, Disassemble) {
  DeviceList device_list = CreateDummyDevices(2);
  std::shared_ptr<const Sharding> sharding =
      ConcreteEvenSharding::Create(device_list, Shape({30}), Shape({15}));

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                          sharding->Disassemble(Shape({30})));
  ASSERT_THAT(disassembled, SizeIs(2));
  for (int i = 0; i < 2; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({15}));
    EXPECT_TRUE(llvm::isa<SingleDeviceSharding>(*sharding));
    EXPECT_THAT(sharding->devices().devices(),
                ElementsAre(device_list.devices()[i]));
  }
}

TEST(ConcreteEvenShardingTest, DisassembleFailsForUnexpectedShape) {
  DeviceList device_list = CreateDummyDevices(2);
  std::shared_ptr<const Sharding> sharding =
      ConcreteEvenSharding::Create(device_list, Shape({30}), Shape({15}));

  EXPECT_THAT(sharding->Disassemble(Shape({40})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteEvenSharding can only disassemble")));
}

TEST(ConcreteEvenShardingTest, IndexDomainsFails) {
  DeviceList device_list = CreateDummyDevices(2);
  std::vector<Shape> shard_shapes;
  std::shared_ptr<const Sharding> sharding =
      ConcreteEvenSharding::Create(device_list, Shape({30}), Shape({15}));

  EXPECT_THAT(
      sharding->IndexDomains(Shape({30})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "ConcreteEvenSharding does not have index domain information")));
}

TEST(ShardingParamShardingTest, CreateFailsWhenDeviceCountNotMatch) {
  DeviceList device_list = CreateDummyDevices(2);
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};

  EXPECT_THAT(ShardingParamSharding::Create(param, device_list),
              StatusIs(tsl::error::FAILED_PRECONDITION,
                       HasSubstr("Device counts don't match. From "
                                 "ShardingParam 6 vs from DeviceList 2")));
}

TEST(ShardingParamShardingTest, Disassemble) {
  DeviceList device_list = CreateDummyDevices(6);
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, CreateDummyDevices(6)));

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                          param_sharding->Disassemble(Shape({6, 6})));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({3, 2}));
    EXPECT_TRUE(llvm::isa<SingleDeviceSharding>(*sharding));
    EXPECT_THAT(sharding->devices().devices(),
                ElementsAre(device_list.devices()[i]));
  }
}

TEST(ShardingParamShardingTest, DisassembleFailsWhenRankNotMatch) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, CreateDummyDevices(6)));

  EXPECT_THAT(
      param_sharding->Disassemble(Shape({6, 6, 6})),
      StatusIs(tsl::error::FAILED_PRECONDITION,
               HasSubstr(
                   "Ranks don't match. From Shape 3 vs from ShardingParam 2")));
}

TEST(ShardingParamShardingTest, DisassembleFailsForUnevenSharding) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, CreateDummyDevices(6)));

  EXPECT_THAT(
      param_sharding->Disassemble(Shape({7, 6})),
      StatusIs(
          tsl::error::FAILED_PRECONDITION,
          HasSubstr("Uneven shard is not supported. dim: 7, dim_shards: 2")));
}

TEST(ShardingParamShardingTest, IndexDomain) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, CreateDummyDevices(6)));

  TF_ASSERT_OK_AND_ASSIGN(auto index_domains,
                          param_sharding->IndexDomains(Shape({6, 6})));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 2})),
                          IndexDomain(Index({0, 2}), Shape({3, 2})),
                          IndexDomain(Index({0, 4}), Shape({3, 2})),
                          IndexDomain(Index({3, 0}), Shape({3, 2})),
                          IndexDomain(Index({3, 2}), Shape({3, 2})),
                          IndexDomain(Index({3, 4}), Shape({3, 2}))));
}

TEST(ShardingParamShardingTest, IndexDomainWithPermutation) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, CreateDummyDevices(6)));

  TF_ASSERT_OK_AND_ASSIGN(auto index_domains,
                          param_sharding->IndexDomains(Shape({6, 6})));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 2})),
                          IndexDomain(Index({0, 4}), Shape({3, 2})),
                          IndexDomain(Index({3, 2}), Shape({3, 2})),
                          IndexDomain(Index({0, 2}), Shape({3, 2})),
                          IndexDomain(Index({3, 0}), Shape({3, 2})),
                          IndexDomain(Index({3, 4}), Shape({3, 2}))));
}

TEST(ShardingParamShardingTest, IndexDomainWithReplication) {
  ShardingParam param{/*dim_shards=*/{2, 1},
                      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, CreateDummyDevices(6)));

  TF_ASSERT_OK_AND_ASSIGN(auto index_domains,
                          param_sharding->IndexDomains(Shape({6, 6})));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 6})),
                          IndexDomain(Index({0, 0}), Shape({3, 6})),
                          IndexDomain(Index({0, 0}), Shape({3, 6})),
                          IndexDomain(Index({3, 0}), Shape({3, 6})),
                          IndexDomain(Index({3, 0}), Shape({3, 6})),
                          IndexDomain(Index({3, 0}), Shape({3, 6}))));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
