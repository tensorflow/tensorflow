/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/sharding.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

class SingleDeviceShardingTest : public test_util::ShardingTest {};
class OpaqueShardingTest : public test_util::ShardingTest {};
class ConcreteShardingTest : public test_util::ShardingTest {};
class ConcreteEvenShardingTest : public test_util::ShardingTest {};
class ShardingParamShardingTest : public test_util::ShardingTest {};

TEST_P(SingleDeviceShardingTest, IndexDomains) {
  auto device_list = GetDevices({0});
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device_list.devices().front(), MemoryKind());

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, sharding->IndexDomains(shape));
  EXPECT_THAT(index_domains, ElementsAre(IndexDomain(shape)));
}

TEST_P(SingleDeviceShardingTest, Disassemble) {
  auto device_list = GetDevices({0});
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device_list.devices().front(), MemoryKind());

  {  // Disassemble static shape.
    Shape shape({10, 20});
    TF_ASSERT_OK_AND_ASSIGN(auto disassembled, sharding->Disassemble(shape));

    ASSERT_THAT(disassembled, SizeIs(1));
    const auto& [result_shape, result_sharding] = disassembled[0];
    ASSERT_EQ(shape, result_shape);
    ASSERT_TRUE(llvm::isa<SingleDeviceSharding>(*result_sharding));
    EXPECT_THAT(result_sharding->devices().devices(),
                ElementsAreArray(device_list.devices()));
  }
  {  // Disassemble dynamic shape.
    TF_ASSERT_OK_AND_ASSIGN(
        DynamicShape dynamic_shape,
        DynamicShape::Create(Shape({10, 20}),
                             BoundedDynamicShapeTag({true, true})));
    TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                            sharding->Disassemble(dynamic_shape));

    ASSERT_THAT(disassembled, SizeIs(1));
    const auto& [result_shape, result_sharding] = disassembled[0];
    ASSERT_EQ(dynamic_shape, result_shape);
    ASSERT_TRUE(llvm::isa<SingleDeviceSharding>(*result_sharding));
    EXPECT_THAT(result_sharding->devices().devices(),
                ElementsAreArray(device_list.devices()));
  }
}

TEST_P(OpaqueShardingTest, FailedToDisassemble) {
  auto device_list = GetDevices({0, 1});
  std::shared_ptr<const Sharding> sharding =
      OpaqueSharding::Create(device_list, MemoryKind());

  EXPECT_THAT(
      sharding->Disassemble(Shape({30})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("OpaqueSharding does not have shard shape information")));

  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape dynamic_shape,
      DynamicShape::Create(Shape({30}), BoundedDynamicShapeTag({true})));
  EXPECT_THAT(
      sharding->Disassemble(dynamic_shape),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("OpaqueSharding does not have shard shape information")));
}

TEST_P(OpaqueShardingTest, IndexDomainsFails) {
  auto device_list = GetDevices({0, 1});
  std::shared_ptr<const Sharding> sharding =
      OpaqueSharding::Create(device_list, MemoryKind());

  EXPECT_THAT(
      sharding->IndexDomains(Shape({30})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("OpaqueSharding does not have index domain information")));
}

TEST_P(ConcreteShardingTest, Disassemble) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  std::shared_ptr<const Sharding> sharding = ConcreteSharding::Create(
      device_list, MemoryKind(), Shape({30}), shard_shapes);

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

TEST_P(ConcreteShardingTest, DisassembleDynamicShape) {
  DeviceList device_list = GetDevices({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape dynamic_shape,
      DynamicShape::Create(Shape({10}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape1,
      DynamicShape::Create(Shape({3}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape2,
      DynamicShape::Create(Shape({7}), BoundedDynamicShapeTag({true})));
  std::vector<DynamicShape> shard_dynamic_shapes{
      std::move(shard_dynamic_shape1), std::move(shard_dynamic_shape2)};
  auto sharding = ConcreteSharding::Create(device_list, MemoryKind(),
                                           dynamic_shape, shard_dynamic_shapes);
  EXPECT_THAT(sharding->Disassemble(Shape({10})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteSharding holds dynamic shape")));
  TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                          sharding->Disassemble(DynamicShape(dynamic_shape)));
  ASSERT_THAT(disassembled, SizeIs(2));
  for (int i = 0; i < disassembled.size(); ++i) {
    const auto& [dynamic_shape, sharding] = disassembled[i];
    EXPECT_EQ(dynamic_shape, shard_dynamic_shapes[i]);
    EXPECT_TRUE(llvm::isa<SingleDeviceSharding>(*sharding));
    EXPECT_THAT(sharding->devices().devices(),
                ElementsAre(device_list.devices()[i]));
  }
}

TEST_P(ConcreteShardingTest, DisassembleFailsForUnexpectedShape) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  std::shared_ptr<const Sharding> sharding = ConcreteSharding::Create(
      device_list, MemoryKind(), Shape({30}), shard_shapes);

  EXPECT_THAT(sharding->Disassemble(Shape({40})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteSharding can only disassemble")));
}

TEST_P(ConcreteShardingTest, IndexDomainsFails) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  std::shared_ptr<const Sharding> sharding = ConcreteSharding::Create(
      device_list, MemoryKind(), Shape({30}), shard_shapes);

  EXPECT_THAT(sharding->IndexDomains(Shape({30})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteSharding does not have index "
                                 "domain information")));
}

TEST_P(ConcreteEvenShardingTest, Disassemble) {
  auto device_list = GetDevices({0, 1});
  std::shared_ptr<const Sharding> sharding = ConcreteEvenSharding::Create(
      device_list, MemoryKind(), Shape({30}), Shape({15}));

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

TEST_P(ConcreteEvenShardingTest, DisassembleFailsForUnexpectedShape) {
  auto device_list = GetDevices({0, 1});
  std::shared_ptr<const Sharding> sharding = ConcreteEvenSharding::Create(
      device_list, MemoryKind(), Shape({30}), Shape({15}));

  EXPECT_THAT(sharding->Disassemble(Shape({40})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteEvenSharding can only disassemble")));
}

TEST_P(ConcreteEvenShardingTest, IndexDomainsFails) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes;
  std::shared_ptr<const Sharding> sharding = ConcreteEvenSharding::Create(
      device_list, MemoryKind(), Shape({30}), Shape({15}));

  EXPECT_THAT(
      sharding->IndexDomains(Shape({30})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "ConcreteEvenSharding does not have index domain information")));
}

TEST_P(ShardingParamShardingTest, CreateFailsWhenDeviceCountNotMatch) {
  auto device_list = GetDevices({0, 1});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};

  EXPECT_THAT(ShardingParamSharding::Create(param, device_list, MemoryKind()),
              StatusIs(tsl::error::FAILED_PRECONDITION,
                       HasSubstr("Device counts don't match. From "
                                 "ShardingParam 6 vs from DeviceList 2")));
}

TEST_P(ShardingParamShardingTest, Disassemble) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

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

TEST_P(ShardingParamShardingTest, DisassembleFailsWhenRankNotMatch) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

  EXPECT_THAT(
      param_sharding->Disassemble(Shape({6, 6, 6})),
      StatusIs(tsl::error::FAILED_PRECONDITION,
               HasSubstr(
                   "Ranks don't match. From Shape 3 vs from ShardingParam 2")));
}

TEST_P(ShardingParamShardingTest, DisassembleFailsForUnevenSharding) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

  EXPECT_THAT(
      param_sharding->Disassemble(Shape({7, 6})),
      StatusIs(
          tsl::error::FAILED_PRECONDITION,
          HasSubstr("Uneven shard is not supported. dim: 7, dim_shards: 2")));
}

TEST_P(ShardingParamShardingTest, IndexDomain) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

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

TEST_P(ShardingParamShardingTest, IndexDomainWithPermutation) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

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

TEST_P(ShardingParamShardingTest, IndexDomainWithReplication) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 1},
                      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const Sharding> param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

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

INSTANTIATE_TEST_SUITE_P(NumDevices, SingleDeviceShardingTest,
                         testing::Values(test_util::ShardingTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/6}));
INSTANTIATE_TEST_SUITE_P(NumDevices, OpaqueShardingTest,
                         testing::Values(test_util::ShardingTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/6}));
INSTANTIATE_TEST_SUITE_P(NumDevices, ConcreteShardingTest,
                         testing::Values(test_util::ShardingTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/6}));
INSTANTIATE_TEST_SUITE_P(NumDevices, ConcreteEvenShardingTest,
                         testing::Values(test_util::ShardingTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/6}));
INSTANTIATE_TEST_SUITE_P(NumDevices, ShardingParamShardingTest,
                         testing::Values(test_util::ShardingTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/4}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
