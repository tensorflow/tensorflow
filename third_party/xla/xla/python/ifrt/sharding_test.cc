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
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

class ShardingTest : public testing::TestWithParam<test_util::DeviceTestParam> {
 public:
  ShardingTest() : fixture_(GetParam()) {}

  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return fixture_.GetDevices(device_indices);
  }

 private:
  test_util::DeviceTestFixture fixture_;
};

class SingleDeviceShardingTest : public ShardingTest {};
class OpaqueShardingTest : public ShardingTest {};
class ConcreteShardingTest : public ShardingTest {};
class ConcreteEvenShardingTest : public ShardingTest {};
class ShardingParamShardingTest : public ShardingTest {};

TEST_P(SingleDeviceShardingTest, CreateWithBadDevice) {
  EXPECT_DEATH(SingleDeviceSharding::Create(nullptr, MemoryKind()), "");
}

TEST_P(SingleDeviceShardingTest, IsFullyReplicated) {
  auto device_list = GetDevices({0});
  ShardingRef sharding = SingleDeviceSharding::Create(
      device_list->devices().front(), MemoryKind());
  EXPECT_TRUE(sharding->IsFullyReplicated());
}

TEST_P(SingleDeviceShardingTest, GetShardShape) {
  auto device_list = GetDevices({0});
  ShardingRef sharding = SingleDeviceSharding::Create(
      device_list->devices().front(), MemoryKind());
  EXPECT_THAT(sharding->GetShardShape(Shape({10, 20})),
              IsOkAndHolds(Shape({10, 20})));
}

TEST_P(SingleDeviceShardingTest, HasSamePartitioning) {
  auto device_list0 = GetDevices({0});
  ShardingRef sharding0 = SingleDeviceSharding::Create(
      device_list0->devices().front(), MemoryKind());

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  {
    auto device_list1 = GetDevices({1});
    ShardingRef sharding1 = SingleDeviceSharding::Create(
        device_list1->devices().front(), MemoryKind());
    EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding1));
  }
}

TEST_P(SingleDeviceShardingTest, WithDeviceAssignment) {
  auto device_list0 = GetDevices({0});
  ShardingRef sharding0 = SingleDeviceSharding::Create(
      device_list0->devices().front(), MemoryKind());
  {
    auto device_list1 = GetDevices({1});
    ShardingRef sharding1 = SingleDeviceSharding::Create(
        device_list1->devices().front(), MemoryKind());
    TF_ASSERT_OK_AND_ASSIGN(
        auto new_sharding,
        sharding0->WithDeviceAssignment(device_list1,
                                        /*memory_kind=*/std::nullopt));
    EXPECT_EQ(*new_sharding, *sharding1);
  }
  {
    auto device_list1 = GetDevices({0, 1});
    EXPECT_THAT(sharding0->WithDeviceAssignment(device_list1,
                                                /*memory_kind=*/std::nullopt),
                StatusIs(tsl::error::INVALID_ARGUMENT,
                         HasSubstr("SingleDeviceSharding can only have one "
                                   "device, but was asked to have 2 devices")));
  }
}

TEST_P(SingleDeviceShardingTest, IndexDomains) {
  auto device_list = GetDevices({0});
  ShardingRef sharding = SingleDeviceSharding::Create(
      device_list->devices().front(), MemoryKind());

  Shape shape({10, 20});
  {
    TF_ASSERT_OK_AND_ASSIGN(auto index_domains, sharding->IndexDomains(shape));
    EXPECT_THAT(index_domains, ElementsAre(IndexDomain(shape)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto index_domains,
        sharding->IndexDomains(shape, SingleDeviceShardSemantics::kAllShards));
    EXPECT_THAT(index_domains, ElementsAre(IndexDomain(shape)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto index_domains,
        sharding->IndexDomains(shape,
                               SingleDeviceShardSemantics::kAddressableShards));
    EXPECT_THAT(index_domains, ElementsAre(IndexDomain(shape)));
  }
}

TEST_P(SingleDeviceShardingTest, Disassemble) {
  auto device_list = GetDevices({0});
  ShardingRef sharding = SingleDeviceSharding::Create(
      device_list->devices().front(), MemoryKind());

  {  // Disassemble static shape.
    Shape shape({10, 20});
    {
      TF_ASSERT_OK_AND_ASSIGN(auto disassembled, sharding->Disassemble(shape));
      ASSERT_THAT(disassembled, SizeIs(1));
      const auto& [result_shape, result_sharding] = disassembled[0];
      EXPECT_EQ(shape, result_shape);
      EXPECT_EQ(*result_sharding, *sharding);
    }
    {
      TF_ASSERT_OK_AND_ASSIGN(
          auto disassembled,
          sharding->Disassemble(shape, SingleDeviceShardSemantics::kAllShards));
      ASSERT_THAT(disassembled, SizeIs(1));
      const auto& [result_shape, result_sharding] = disassembled[0];
      EXPECT_EQ(shape, result_shape);
      EXPECT_EQ(*result_sharding, *sharding);
    }
    {
      TF_ASSERT_OK_AND_ASSIGN(
          auto disassembled,
          sharding->Disassemble(
              shape, SingleDeviceShardSemantics::kAddressableShards));
      ASSERT_THAT(disassembled, SizeIs(1));
      const auto& [result_shape, result_sharding] = disassembled[0];
      EXPECT_EQ(shape, result_shape);
      EXPECT_EQ(*result_sharding, *sharding);
    }
  }
  {  // Disassemble dynamic shape.
    TF_ASSERT_OK_AND_ASSIGN(
        DynamicShape dynamic_shape,
        DynamicShape::Create(Shape({10, 20}),
                             BoundedDynamicShapeTag({true, true})));
    {
      TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                              sharding->Disassemble(dynamic_shape));
      ASSERT_THAT(disassembled, SizeIs(1));
      const auto& [result_shape, result_sharding] = disassembled[0];
      EXPECT_EQ(dynamic_shape, result_shape);
      EXPECT_EQ(*result_sharding, *sharding);
    }
    {
      TF_ASSERT_OK_AND_ASSIGN(
          auto disassembled,
          sharding->Disassemble(dynamic_shape,
                                SingleDeviceShardSemantics::kAllShards));
      ASSERT_THAT(disassembled, SizeIs(1));
      const auto& [result_shape, result_sharding] = disassembled[0];
      EXPECT_EQ(dynamic_shape, result_shape);
      EXPECT_EQ(*result_sharding, *sharding);
    }
    {
      TF_ASSERT_OK_AND_ASSIGN(
          auto disassembled,
          sharding->Disassemble(
              dynamic_shape, SingleDeviceShardSemantics::kAddressableShards));
      ASSERT_THAT(disassembled, SizeIs(1));
      const auto& [result_shape, result_sharding] = disassembled[0];
      EXPECT_EQ(dynamic_shape, result_shape);
      EXPECT_EQ(*result_sharding, *sharding);
    }
  }
}

TEST_P(SingleDeviceShardingTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      *SingleDeviceSharding::Create(GetDevices({0})->devices().front(),
                                    MemoryKind()),
      *SingleDeviceSharding::Create(GetDevices({1})->devices().front(),
                                    MemoryKind()),
  }));
}

TEST_P(OpaqueShardingTest, CreateWithBadDeviceList) {
  EXPECT_DEATH(OpaqueSharding::Create(DeviceListRef(), MemoryKind()), "");

  EXPECT_DEATH(OpaqueSharding::Create(GetDevices({}), MemoryKind()), "");
}

TEST_P(OpaqueShardingTest, IsFullyReplicated) {
  auto device_list = GetDevices({0, 1});
  ShardingRef sharding = OpaqueSharding::Create(device_list, MemoryKind());
  EXPECT_FALSE(sharding->IsFullyReplicated());
}

TEST_P(OpaqueShardingTest, GetShardShape) {
  auto device_list = GetDevices({0, 1});
  ShardingRef sharding = OpaqueSharding::Create(device_list, MemoryKind());
  EXPECT_THAT(sharding->GetShardShape(Shape({10, 20})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("OpaqueSharding does not have shard shape")));
}

TEST_P(OpaqueShardingTest, HasSamePartitioning) {
  auto device_list0 = GetDevices({0, 1});
  ShardingRef sharding0 = OpaqueSharding::Create(device_list0, MemoryKind());

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  {
    auto device_list1 = GetDevices({2, 3});
    ShardingRef sharding1 = OpaqueSharding::Create(device_list0, MemoryKind());
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
}

TEST_P(OpaqueShardingTest, WithDeviceAssignment) {
  auto device_list0 = GetDevices({0, 1});
  ShardingRef sharding0 = OpaqueSharding::Create(device_list0, MemoryKind());
  {
    auto device_list1 = GetDevices({2, 3});
    ShardingRef sharding1 = OpaqueSharding::Create(device_list0, MemoryKind());
    TF_ASSERT_OK_AND_ASSIGN(
        auto new_sharding,
        sharding0->WithDeviceAssignment(device_list1,
                                        /*memory_kind=*/std::nullopt));
    // For OpaqueSharding, we cannot use an equality test.
    ASSERT_TRUE(llvm::isa<OpaqueSharding>(*new_sharding));
    EXPECT_THAT(new_sharding->devices()->devices(),
                ElementsAreArray(device_list1->devices()));
  }
  {
    auto device_list1 = GetDevices({0, 1, 2, 3});
    EXPECT_THAT(
        sharding0->WithDeviceAssignment(device_list1,
                                        /*memory_kind=*/std::nullopt),
        StatusIs(tsl::error::INVALID_ARGUMENT,
                 HasSubstr("OpaqueSharding should have the same number of "
                           "devices as the current sharding, but was asked to "
                           "have 4 devices")));
  }
}

TEST_P(OpaqueShardingTest, FailedToDisassemble) {
  auto device_list = GetDevices({0, 1});
  ShardingRef sharding = OpaqueSharding::Create(device_list, MemoryKind());

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
  ShardingRef sharding = OpaqueSharding::Create(device_list, MemoryKind());

  EXPECT_THAT(
      sharding->IndexDomains(Shape({30})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("OpaqueSharding does not have index domain information")));
}

TEST_P(OpaqueShardingTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      *OpaqueSharding::Create(GetDevices({0, 1}), MemoryKind()),
      *OpaqueSharding::Create(GetDevices({2, 3}), MemoryKind()),
  }));
}

TEST_P(ConcreteShardingTest, CreateWithBadDeviceList) {
  EXPECT_DEATH(ConcreteSharding::Create(DeviceListRef(), MemoryKind(),
                                        Shape({}), {Shape({})}),
               "");

  EXPECT_DEATH(ConcreteSharding::Create(GetDevices({}), MemoryKind(), Shape({}),
                                        {Shape({})}),
               "");
}

TEST_P(ConcreteShardingTest, IsFullyReplicated) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  ShardingRef sharding = ConcreteSharding::Create(device_list, MemoryKind(),
                                                  Shape({30}), shard_shapes);
  EXPECT_FALSE(sharding->IsFullyReplicated());
}

TEST_P(ConcreteShardingTest, GetShardShapeSuccess) {
  auto device_list = GetDevices({0, 1});
  Shape shard_shape({30});
  std::vector<Shape> shard_shapes(2, shard_shape);
  ShardingRef sharding = ConcreteSharding::Create(device_list, MemoryKind(),
                                                  Shape({30}), shard_shapes);
  EXPECT_THAT(sharding->GetShardShape(Shape({30})), IsOkAndHolds(shard_shape));
}

TEST_P(ConcreteShardingTest, GetShardShapeFailure) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  ShardingRef sharding = ConcreteSharding::Create(device_list, MemoryKind(),
                                                  Shape({30}), shard_shapes);
  EXPECT_THAT(
      sharding->GetShardShape(Shape({30})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("ConcreteSharding does not have a fixed shard shape")));
}

TEST_P(ConcreteShardingTest, HasSamePartitioning) {
  auto device_list0 = GetDevices({0, 1});
  std::vector<Shape> shard_shapes0;
  shard_shapes0.reserve(2);
  shard_shapes0.push_back(Shape({10}));
  shard_shapes0.push_back(Shape({20}));
  ShardingRef sharding0 = ConcreteSharding::Create(device_list0, MemoryKind(),
                                                   Shape({30}), shard_shapes0);

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  {
    auto device_list1 = GetDevices({2, 3});
    std::vector<Shape> shard_shapes1;
    shard_shapes1.reserve(2);
    shard_shapes1.push_back(Shape({10}));
    shard_shapes1.push_back(Shape({20}));
    ShardingRef sharding1 = ConcreteSharding::Create(
        device_list1, MemoryKind(), Shape({30}), shard_shapes1);
    EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different number of shards.
  {
    auto device_list1 = GetDevices({2, 3, 4});
    std::vector<Shape> shard_shapes1;
    shard_shapes1.reserve(3);
    shard_shapes1.push_back(Shape({10}));
    shard_shapes1.push_back(Shape({20}));
    shard_shapes1.push_back(Shape({30}));
    ShardingRef sharding1 = ConcreteSharding::Create(
        device_list1, MemoryKind(), Shape({60}), shard_shapes1);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Difference shape.
  {
    auto device_list1 = GetDevices({2, 3});
    std::vector<Shape> shard_shapes1;
    shard_shapes1.reserve(2);
    shard_shapes1.push_back(Shape({10}));
    shard_shapes1.push_back(Shape({20}));
    ShardingRef sharding1 = ConcreteSharding::Create(
        device_list1, MemoryKind(), Shape({40}), shard_shapes1);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different shard shapes.
  {
    auto device_list1 = GetDevices({2, 3});
    std::vector<Shape> shard_shapes1;
    shard_shapes1.reserve(2);
    shard_shapes1.push_back(Shape({10000}));
    shard_shapes1.push_back(Shape({20}));
    ShardingRef sharding1 = ConcreteSharding::Create(
        device_list1, MemoryKind(), Shape({30}), shard_shapes1);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
}

TEST_P(ConcreteShardingTest, WithDeviceAssignment) {
  auto device_list0 = GetDevices({0, 1});
  std::vector<Shape> shard_shapes0;
  shard_shapes0.reserve(2);
  shard_shapes0.push_back(Shape({10}));
  shard_shapes0.push_back(Shape({20}));
  ShardingRef sharding0 = ConcreteSharding::Create(device_list0, MemoryKind(),
                                                   Shape({30}), shard_shapes0);
  {
    auto device_list1 = GetDevices({0, 1});
    std::vector<Shape> shard_shapes1;
    shard_shapes1.reserve(2);
    shard_shapes1.push_back(Shape({10}));
    shard_shapes1.push_back(Shape({20}));
    ShardingRef sharding1 = ConcreteSharding::Create(
        device_list1, MemoryKind(), Shape({30}), shard_shapes1);
    TF_ASSERT_OK_AND_ASSIGN(
        auto new_sharding,
        sharding0->WithDeviceAssignment(device_list1,
                                        /*memory_kind=*/std::nullopt));
    EXPECT_EQ(*new_sharding, *sharding1);
  }
  {
    auto device_list1 = GetDevices({0, 1, 2, 3});
    EXPECT_THAT(
        sharding0->WithDeviceAssignment(device_list1,
                                        /*memory_kind=*/std::nullopt),
        StatusIs(tsl::error::INVALID_ARGUMENT,
                 HasSubstr("ConcreteSharding should have the same number of "
                           "devices as the current sharding, but was asked to "
                           "have 4 devices")));
  }
}

TEST_P(ConcreteShardingTest, Disassemble) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(4);
  shard_shapes.push_back(Shape({3}));
  shard_shapes.push_back(Shape({7}));
  shard_shapes.push_back(Shape({3}));
  shard_shapes.push_back(Shape({7}));
  ShardingRef sharding = ConcreteSharding::Create(device_list, MemoryKind(),
                                                  Shape({20}), shard_shapes);

  {
    EXPECT_THAT(
        sharding->Disassemble(Shape({20})),
        StatusIs(
            tsl::error::INVALID_ARGUMENT,
            HasSubstr("SingleDeviceShardSemantics::kAllShards was requested, "
                      "but the ConcreteSharding contains non-addressable "
                      "devices. Saw 6 devices, with 4 addressable devices")));
  }
  {
    EXPECT_THAT(
        sharding->Disassemble(Shape({20}),
                              SingleDeviceShardSemantics::kAllShards),
        StatusIs(
            tsl::error::INVALID_ARGUMENT,
            HasSubstr("SingleDeviceShardSemantics::kAllShards was requested, "
                      "but the ConcreteSharding contains non-addressable "
                      "devices. Saw 6 devices, with 4 addressable devices")));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto disassembled,
        sharding->Disassemble(Shape({20}),
                              SingleDeviceShardSemantics::kAddressableShards));
    // The first 4 devices are addressable.
    ASSERT_THAT(disassembled, SizeIs(4));
    for (int i = 0; i < 4; ++i) {
      const auto& [shape, sharding] = disassembled[i];
      EXPECT_EQ(shape, shard_shapes[i]);
      EXPECT_EQ(*sharding, *SingleDeviceSharding::Create(
                               device_list->devices()[i], MemoryKind()));
    }
  }
}

TEST_P(ConcreteShardingTest, DisassembleDynamicShape) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape dynamic_shape,
      DynamicShape::Create(Shape({20}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape0,
      DynamicShape::Create(Shape({3}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape1,
      DynamicShape::Create(Shape({7}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape2,
      DynamicShape::Create(Shape({3}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape3,
      DynamicShape::Create(Shape({7}), BoundedDynamicShapeTag({true})));
  std::vector<DynamicShape> shard_dynamic_shapes{
      std::move(shard_dynamic_shape0),
      std::move(shard_dynamic_shape1),
      std::move(shard_dynamic_shape2),
      std::move(shard_dynamic_shape3),
  };
  auto sharding = ConcreteSharding::Create(device_list, MemoryKind(),
                                           dynamic_shape, shard_dynamic_shapes);
  EXPECT_THAT(sharding->Disassemble(Shape({20})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteSharding holds dynamic shape")));
  {
    EXPECT_THAT(
        sharding->Disassemble(DynamicShape(dynamic_shape)),
        StatusIs(
            tsl::error::INVALID_ARGUMENT,
            HasSubstr("SingleDeviceShardSemantics::kAllShards was requested, "
                      "but the ConcreteSharding contains non-addressable "
                      "devices. Saw 6 devices, with 4 addressable devices")));
  }
  {
    EXPECT_THAT(
        sharding->Disassemble(DynamicShape(dynamic_shape),
                              SingleDeviceShardSemantics::kAllShards),
        StatusIs(
            tsl::error::INVALID_ARGUMENT,
            HasSubstr("SingleDeviceShardSemantics::kAllShards was requested, "
                      "but the ConcreteSharding contains non-addressable "
                      "devices. Saw 6 devices, with 4 addressable devices")));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto disassembled,
        sharding->Disassemble(DynamicShape(dynamic_shape),
                              SingleDeviceShardSemantics::kAddressableShards));
    // The first 4 devices are addressable.
    ASSERT_THAT(disassembled, SizeIs(4));
    for (int i = 0; i < 4; ++i) {
      const auto& [dynamic_shape, sharding] = disassembled[i];
      EXPECT_EQ(dynamic_shape, shard_dynamic_shapes[i]);
      EXPECT_EQ(*sharding, *SingleDeviceSharding::Create(
                               device_list->devices()[i], MemoryKind()));
    }
  }
}

TEST_P(ConcreteShardingTest, DisassembleFailsForUnexpectedShape) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  ShardingRef sharding = ConcreteSharding::Create(device_list, MemoryKind(),
                                                  Shape({30}), shard_shapes);

  EXPECT_THAT(sharding->Disassemble(Shape({40})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteSharding can only disassemble")));
}

TEST_P(ConcreteShardingTest, IndexDomainsFullyAddressable) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes = {Shape({1}), Shape({2})};
  std::vector<IndexDomain> index_domains{
      IndexDomain(Index({0}), Shape({1})),
      IndexDomain(Index({1}), Shape({2})),
  };
  ShardingRef sharding = ConcreteSharding::Create(
      device_list, MemoryKind(), Shape({15}), shard_shapes, index_domains);

  EXPECT_THAT(sharding->IndexDomains(Shape({15})),
              IsOkAndHolds(ElementsAreArray(index_domains)));
  EXPECT_THAT(sharding->IndexDomains(Shape({15}),
                                     SingleDeviceShardSemantics::kAllShards),
              IsOkAndHolds(ElementsAreArray(index_domains)));
  EXPECT_THAT(sharding->IndexDomains(
                  Shape({15}), SingleDeviceShardSemantics::kAddressableShards),
              IsOkAndHolds(ElementsAreArray(index_domains)));
}

TEST_P(ConcreteShardingTest, IndexDomainsNonAddressable) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  std::vector<Shape> shard_shapes = {
      Shape({1}),
      Shape({2}),
      Shape({3}),
      Shape({4}),
  };
  std::vector<IndexDomain> index_domains{
      IndexDomain(Index({0}), Shape({1})),
      IndexDomain(Index({1}), Shape({2})),
      IndexDomain(Index({3}), Shape({3})),
      IndexDomain(Index({6}), Shape({4})),
  };
  ShardingRef sharding = ConcreteSharding::Create(
      device_list, MemoryKind(), Shape({15}), shard_shapes, index_domains);

  EXPECT_THAT(
      sharding->IndexDomains(Shape({15}),
                             SingleDeviceShardSemantics::kAllShards),
      StatusIs(tsl::error::INVALID_ARGUMENT,
               HasSubstr("SingleDeviceShardSemantics::kAllShards was "
                         "requested, but the ConcreteSharding contains index "
                         "domains from non-addressable devices")));
  EXPECT_THAT(sharding->IndexDomains(
                  Shape({15}), SingleDeviceShardSemantics::kAddressableShards),
              IsOkAndHolds(ElementsAreArray(index_domains)));
}

TEST_P(ConcreteShardingTest, IndexDomainsMissing) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes;
  shard_shapes.reserve(2);
  shard_shapes.push_back(Shape({10}));
  shard_shapes.push_back(Shape({20}));
  ShardingRef sharding = ConcreteSharding::Create(device_list, MemoryKind(),
                                                  Shape({30}), shard_shapes);

  EXPECT_THAT(sharding->IndexDomains(Shape({30})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteSharding does not have index "
                                 "domain information")));
}

TEST_P(ConcreteShardingTest, IndexDomainsFails) {
  auto device_list = GetDevices({0, 1, 2, 3, 4});
  std::vector<Shape> shard_shapes(5, Shape({1}));
  std::vector<IndexDomain> index_domains(5,
                                         IndexDomain(Index({0}), Shape({1})));
  ShardingRef sharding = ConcreteSharding::Create(
      device_list, MemoryKind(), Shape({30}), shard_shapes, index_domains);

  EXPECT_THAT(sharding->IndexDomains(Shape({1})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteSharding must have the same number "
                                 "of index domains and addressable devices")));
}

TEST_P(ConcreteShardingTest, Hash) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto dynamic_shape,
      DynamicShape::Create(Shape({30}), BoundedDynamicShapeTag({true})));
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      *ConcreteSharding::Create(GetDevices({0, 1}), MemoryKind(), Shape({30}),
                                {Shape({10}), Shape({20})}),
      *ConcreteSharding::Create(GetDevices({0, 1}), MemoryKind(), dynamic_shape,
                                {dynamic_shape, dynamic_shape}),
  }));
}

TEST_P(ConcreteEvenShardingTest, CreateWithBadDeviceList) {
  EXPECT_DEATH(ConcreteEvenSharding::Create(DeviceListRef(), MemoryKind(),
                                            Shape({}), Shape({}),
                                            /*is_fully_replicated=*/true),
               "");

  EXPECT_DEATH(ConcreteEvenSharding::Create(GetDevices({}), MemoryKind(),
                                            Shape({}), Shape({}),
                                            /*is_fully_replicated=*/true),
               "");
}

TEST_P(ConcreteEvenShardingTest, IsFullyReplicated) {
  auto device_list = GetDevices({0, 1});
  {
    // Fully replicated.
    ShardingRef sharding =
        ConcreteEvenSharding::Create(device_list, MemoryKind(), Shape({30}),
                                     Shape({15}), /*is_fully_replicated=*/true);
    EXPECT_TRUE(sharding->IsFullyReplicated());
  }
  {
    // Not fully replicated.
    ShardingRef sharding = ConcreteEvenSharding::Create(
        device_list, MemoryKind(), Shape({30}), Shape({15}),
        /*is_fully_replicated=*/false);
    EXPECT_FALSE(sharding->IsFullyReplicated());
  }
}

TEST_P(ConcreteEvenShardingTest, GetShardShape) {
  auto device_list = GetDevices({0, 1});
  ShardingRef sharding =
      ConcreteEvenSharding::Create(device_list, MemoryKind(), Shape({30}),
                                   Shape({15}), /*is_fully_replicated=*/true);
  EXPECT_THAT(sharding->GetShardShape(Shape({30})), IsOkAndHolds(Shape({15})));
  EXPECT_THAT(
      sharding->GetShardShape(Shape({45})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("ConcreteEvenSharding has a shard shape for shape [30], "
                    "but was asked to get a shard shape for shape [45]")));
}

TEST_P(ConcreteEvenShardingTest, HasSamePartitioning) {
  auto device_list0 = GetDevices({0, 1});
  ShardingRef sharding0 =
      ConcreteEvenSharding::Create(device_list0, MemoryKind(), Shape({30}),
                                   Shape({15}), /*is_fully_replicated=*/true);

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  {
    auto device_list1 = GetDevices({2, 3});
    ShardingRef sharding1 =
        ConcreteEvenSharding::Create(device_list1, MemoryKind(), Shape({30}),
                                     Shape({15}), /*is_fully_replicated=*/true);
    EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different number of shards.
  {
    auto device_list1 = GetDevices({2, 3, 4});
    ShardingRef sharding1 =
        ConcreteEvenSharding::Create(device_list1, MemoryKind(), Shape({30}),
                                     Shape({15}), /*is_fully_replicated=*/true);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Difference shape.
  {
    auto device_list1 = GetDevices({2, 3});
    ShardingRef sharding1 =
        ConcreteEvenSharding::Create(device_list1, MemoryKind(), Shape({45}),
                                     Shape({15}), /*is_fully_replicated=*/true);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different shard shape.
  {
    auto device_list1 = GetDevices({2, 3});
    ShardingRef sharding1 =
        ConcreteEvenSharding::Create(device_list1, MemoryKind(), Shape({30}),
                                     Shape({10}), /*is_fully_replicated=*/true);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different is_fully_replicated.
  {
    auto device_list1 = GetDevices({2, 3});
    ShardingRef sharding1 = ConcreteEvenSharding::Create(
        device_list1, MemoryKind(), Shape({30}), Shape({15}),
        /*is_fully_replicated=*/false);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
}

TEST_P(ConcreteEvenShardingTest, WithDeviceAssignment) {
  auto device_list0 = GetDevices({0, 1});
  ShardingRef sharding0 =
      ConcreteEvenSharding::Create(device_list0, MemoryKind(), Shape({30}),
                                   Shape({15}), /*is_fully_replicated=*/true);
  {
    auto device_list1 = GetDevices({2, 3});
    ShardingRef sharding1 =
        ConcreteEvenSharding::Create(device_list1, MemoryKind(), Shape({30}),
                                     Shape({15}), /*is_fully_replicated=*/true);
    TF_ASSERT_OK_AND_ASSIGN(
        auto new_sharding,
        sharding0->WithDeviceAssignment(device_list1,
                                        /*memory_kind=*/std::nullopt));
    EXPECT_EQ(*new_sharding, *sharding1);
  }
  {
    auto device_list1 = GetDevices({0, 1, 2, 3});
    EXPECT_THAT(
        sharding0->WithDeviceAssignment(device_list1,
                                        /*memory_kind=*/std::nullopt),
        StatusIs(
            tsl::error::INVALID_ARGUMENT,
            HasSubstr("ConcreteEvenSharding should have the same number of "
                      "devices as the current sharding, but was asked to "
                      "have 4 devices")));
  }
}

TEST_P(ConcreteEvenShardingTest, Disassemble) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingRef sharding =
      ConcreteEvenSharding::Create(device_list, MemoryKind(), Shape({30}),
                                   Shape({5}), /*is_fully_replicated=*/false);

  {
    TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                            sharding->Disassemble(Shape({30})));
    ASSERT_THAT(disassembled, SizeIs(6));
    for (int i = 0; i < 6; ++i) {
      const auto& [shape, sharding] = disassembled[i];
      EXPECT_EQ(shape, Shape({5}));
      EXPECT_EQ(*sharding, *SingleDeviceSharding::Create(
                               device_list->devices()[i], MemoryKind()));
    }
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto disassembled,
        sharding->Disassemble(Shape({30}),
                              SingleDeviceShardSemantics::kAllShards));
    ASSERT_THAT(disassembled, SizeIs(6));
    for (int i = 0; i < 6; ++i) {
      const auto& [shape, sharding] = disassembled[i];
      EXPECT_EQ(shape, Shape({5}));
      EXPECT_EQ(*sharding, *SingleDeviceSharding::Create(
                               device_list->devices()[i], MemoryKind()));
    }
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto disassembled,
        sharding->Disassemble(Shape({30}),
                              SingleDeviceShardSemantics::kAddressableShards));
    // The first 4 devices are addressable.
    ASSERT_THAT(disassembled, SizeIs(4));
    for (int i = 0; i < 4; ++i) {
      const auto& [shape, sharding] = disassembled[i];
      EXPECT_EQ(shape, Shape({5}));
      EXPECT_EQ(*sharding, *SingleDeviceSharding::Create(
                               device_list->devices()[i], MemoryKind()));
    }
  }
}

TEST_P(ConcreteEvenShardingTest, DisassembleFailsForUnexpectedShape) {
  auto device_list = GetDevices({0, 1});
  ShardingRef sharding =
      ConcreteEvenSharding::Create(device_list, MemoryKind(), Shape({30}),
                                   Shape({15}), /*is_fully_replicated=*/false);

  EXPECT_THAT(sharding->Disassemble(Shape({40})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("ConcreteEvenSharding can only disassemble")));
}

TEST_P(ConcreteEvenShardingTest, IndexDomainsFails) {
  auto device_list = GetDevices({0, 1});
  std::vector<Shape> shard_shapes;
  ShardingRef sharding =
      ConcreteEvenSharding::Create(device_list, MemoryKind(), Shape({30}),
                                   Shape({5}), /*is_fully_replicated=*/false);

  EXPECT_THAT(
      sharding->IndexDomains(Shape({30})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "ConcreteEvenSharding does not have index domain information")));
}

TEST_P(ConcreteEvenShardingTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      *ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                    Shape({30}), Shape({30}),
                                    /*is_fully_replicated=*/true),
      *ConcreteEvenSharding::Create(GetDevices({2, 3}), MemoryKind(),
                                    Shape({30}), Shape({15}),
                                    /*is_fully_replicated=*/false),
  }));
}

TEST_P(ShardingParamShardingTest, CreateWithBadDeviceList) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  EXPECT_DEATH(
      ShardingParamSharding::Create(param, DeviceListRef(), MemoryKind())
          .value(),
      "");

  EXPECT_DEATH(
      ShardingParamSharding::Create(param, GetDevices({}), MemoryKind())
          .value(),
      "");
}

TEST_P(ShardingParamShardingTest, CreateFailsWhenDeviceCountNotMatch) {
  auto device_list = GetDevices({0, 1});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};

  EXPECT_THAT(ShardingParamSharding::Create(param, device_list, MemoryKind()),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("Device counts don't match. From "
                                 "ShardingParam 6 vs from DeviceList 2")));
}

TEST_P(ShardingParamShardingTest, IsFullyReplicated) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  {
    // Fully replicated.
    ShardingParam param{/*dim_shards=*/{1, 1},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
    TF_ASSERT_OK_AND_ASSIGN(
        ShardingRef param_sharding,
        ShardingParamSharding::Create(param, device_list, MemoryKind()));
    EXPECT_TRUE(param_sharding->IsFullyReplicated());
  }
  {
    // Not fully replicated.
    ShardingParam param{/*dim_shards=*/{1, 6},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
    TF_ASSERT_OK_AND_ASSIGN(
        ShardingRef param_sharding,
        ShardingParamSharding::Create(param, device_list, MemoryKind()));
    EXPECT_FALSE(param_sharding->IsFullyReplicated());
  }
  {
    // Not fully replicated.
    ShardingParam param{/*dim_shards=*/{2, 3},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
    TF_ASSERT_OK_AND_ASSIGN(
        ShardingRef param_sharding,
        ShardingParamSharding::Create(param, device_list, MemoryKind()));
    EXPECT_FALSE(param_sharding->IsFullyReplicated());
  }
}

TEST_P(ShardingParamShardingTest, GetShardShape) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));
  EXPECT_THAT(sharding->GetShardShape(Shape({6, 6})),
              IsOkAndHolds(Shape({3, 2})));
  EXPECT_THAT(sharding->GetShardShape(Shape({6, 6, 6})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("Numbers of dimensions don't match. From "
                                 "Shape 3 vs from ShardingParam 2")));
}

TEST_P(ShardingParamShardingTest, HasSamePartitioning) {
  auto device_list0 = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param0{/*dim_shards=*/{2, 3},
                       {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding0,
      ShardingParamSharding::Create(param0, device_list0, MemoryKind()));

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  {
    auto device_list1 = GetDevices({3, 4, 5, 0, 1, 2});
    ShardingParam param1{/*dim_shards=*/{2, 3},
                         {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
    TF_ASSERT_OK_AND_ASSIGN(
        ShardingRef sharding1,
        ShardingParamSharding::Create(param1, device_list1, MemoryKind()));
    EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different number of shards.
  {
    auto device_list1 = GetDevices({3, 4, 5});
    ShardingParam param1{/*dim_shards=*/{3, 1},
                         {/*permutation=*/{1, 0}, /*axis_sizes=*/{1, 3}}};
    TF_ASSERT_OK_AND_ASSIGN(
        ShardingRef sharding1,
        ShardingParamSharding::Create(param1, device_list1, MemoryKind()));
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different sharding param.
  {
    auto device_list1 = GetDevices({3, 4, 5, 0, 1, 2});
    ShardingParam param1{/*dim_shards=*/{3, 2},
                         {/*permutation=*/{0, 1}, /*axis_sizes=*/{3, 2}}};
    TF_ASSERT_OK_AND_ASSIGN(
        ShardingRef sharding1,
        ShardingParamSharding::Create(param1, device_list1, MemoryKind()));
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
}

TEST_P(ShardingParamShardingTest, WithDeviceAssignment) {
  auto device_list0 = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param0{/*dim_shards=*/{2, 3},
                       {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding0,
      ShardingParamSharding::Create(param0, device_list0, MemoryKind()));
  {
    auto device_list1 = GetDevices({3, 4, 5, 0, 1, 2});
    ShardingParam param1{/*dim_shards=*/{2, 3},
                         {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
    TF_ASSERT_OK_AND_ASSIGN(
        ShardingRef sharding1,
        ShardingParamSharding::Create(param1, device_list1, MemoryKind()));
    TF_ASSERT_OK_AND_ASSIGN(
        auto new_sharding,
        sharding0->WithDeviceAssignment(device_list1,
                                        /*memory_kind=*/std::nullopt));
    EXPECT_EQ(*new_sharding, *sharding1);
  }
  {
    auto device_list1 = GetDevices({0, 1, 2});
    EXPECT_THAT(
        sharding0->WithDeviceAssignment(device_list1,
                                        /*memory_kind=*/std::nullopt),
        StatusIs(
            tsl::error::INVALID_ARGUMENT,
            HasSubstr("ShardingParamSharding should have the same number of "
                      "devices as the current sharding, but was asked to "
                      "have 3 devices")));
  }
}

TEST_P(ShardingParamShardingTest, Disassemble) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

  {
    TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                            param_sharding->Disassemble(Shape({6, 6})));
    ASSERT_THAT(disassembled, SizeIs(6));
    for (int i = 0; i < 6; ++i) {
      const auto& [shape, sharding] = disassembled[i];
      EXPECT_EQ(shape, Shape({3, 2}));
      EXPECT_EQ(*sharding, *SingleDeviceSharding::Create(
                               device_list->devices()[i], MemoryKind()));
    }
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto disassembled,
        param_sharding->Disassemble(Shape({6, 6}),
                                    SingleDeviceShardSemantics::kAllShards));
    ASSERT_THAT(disassembled, SizeIs(6));
    for (int i = 0; i < 6; ++i) {
      const auto& [shape, sharding] = disassembled[i];
      EXPECT_EQ(shape, Shape({3, 2}));
      EXPECT_EQ(*sharding, *SingleDeviceSharding::Create(
                               device_list->devices()[i], MemoryKind()));
    }
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto disassembled,
        param_sharding->Disassemble(
            Shape({6, 6}), SingleDeviceShardSemantics::kAddressableShards));
    // The first 4 devices are addressable.
    ASSERT_THAT(disassembled, SizeIs(4));
    for (int i = 0; i < 4; ++i) {
      const auto& [shape, sharding] = disassembled[i];
      EXPECT_EQ(shape, Shape({3, 2}));
      EXPECT_EQ(*sharding, *SingleDeviceSharding::Create(
                               device_list->devices()[i], MemoryKind()));
    }
  }
}

TEST_P(ShardingParamShardingTest, DisassembleFailsWhenRankNotMatch) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

  EXPECT_THAT(param_sharding->Disassemble(Shape({6, 6, 6})),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("Numbers of dimensions don't match. From "
                                 "Shape 3 vs from ShardingParam 2")));
}

TEST_P(ShardingParamShardingTest, DisassembleFailsForUnevenSharding) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

  EXPECT_THAT(
      param_sharding->Disassemble(Shape({7, 6})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("Uneven shard is not supported. dim: 7, dim_shards: 2")));
}

TEST_P(ShardingParamShardingTest, IndexDomain) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

  {
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
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto index_domains,
        param_sharding->IndexDomains(Shape({6, 6}),
                                     SingleDeviceShardSemantics::kAllShards));
    EXPECT_THAT(index_domains,
                ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 2})),
                            IndexDomain(Index({0, 2}), Shape({3, 2})),
                            IndexDomain(Index({0, 4}), Shape({3, 2})),
                            IndexDomain(Index({3, 0}), Shape({3, 2})),
                            IndexDomain(Index({3, 2}), Shape({3, 2})),
                            IndexDomain(Index({3, 4}), Shape({3, 2}))));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto index_domains,
        param_sharding->IndexDomains(
            Shape({6, 6}), SingleDeviceShardSemantics::kAddressableShards));
    // The first 4 devices are addressable.
    EXPECT_THAT(index_domains,
                ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 2})),
                            IndexDomain(Index({0, 2}), Shape({3, 2})),
                            IndexDomain(Index({0, 4}), Shape({3, 2})),
                            IndexDomain(Index({3, 0}), Shape({3, 2}))));
  }
}

TEST_P(ShardingParamShardingTest, IndexDomainWithPermutation) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

  {
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
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto index_domains,
        param_sharding->IndexDomains(Shape({6, 6}),
                                     SingleDeviceShardSemantics::kAllShards));
    EXPECT_THAT(index_domains,
                ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 2})),
                            IndexDomain(Index({0, 4}), Shape({3, 2})),
                            IndexDomain(Index({3, 2}), Shape({3, 2})),
                            IndexDomain(Index({0, 2}), Shape({3, 2})),
                            IndexDomain(Index({3, 0}), Shape({3, 2})),
                            IndexDomain(Index({3, 4}), Shape({3, 2}))));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto index_domains,
        param_sharding->IndexDomains(
            Shape({6, 6}), SingleDeviceShardSemantics::kAddressableShards));
    // The first 4 devices are addressable.
    EXPECT_THAT(index_domains,
                ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 2})),
                            IndexDomain(Index({0, 4}), Shape({3, 2})),
                            IndexDomain(Index({3, 2}), Shape({3, 2})),
                            IndexDomain(Index({0, 2}), Shape({3, 2}))));
  }
}

TEST_P(ShardingParamShardingTest, IndexDomainWithReplication) {
  auto device_list = GetDevices({0, 1, 2, 3, 4, 5});
  ShardingParam param{/*dim_shards=*/{2, 1},
                      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef param_sharding,
      ShardingParamSharding::Create(param, device_list, MemoryKind()));

  {
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
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto index_domains,
        param_sharding->IndexDomains(Shape({6, 6}),
                                     SingleDeviceShardSemantics::kAllShards));
    EXPECT_THAT(index_domains,
                ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 6})),
                            IndexDomain(Index({0, 0}), Shape({3, 6})),
                            IndexDomain(Index({0, 0}), Shape({3, 6})),
                            IndexDomain(Index({3, 0}), Shape({3, 6})),
                            IndexDomain(Index({3, 0}), Shape({3, 6})),
                            IndexDomain(Index({3, 0}), Shape({3, 6}))));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto index_domains,
        param_sharding->IndexDomains(
            Shape({6, 6}), SingleDeviceShardSemantics::kAddressableShards));
    // The first 4 devices are addressable.
    EXPECT_THAT(index_domains,
                ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 6})),
                            IndexDomain(Index({0, 0}), Shape({3, 6})),
                            IndexDomain(Index({0, 0}), Shape({3, 6})),
                            IndexDomain(Index({3, 0}), Shape({3, 6}))));
  }
}

TEST_P(ShardingParamShardingTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      *ShardingParamSharding::Create(
           ShardingParam{/*dim_shards=*/{2, 3},
                         {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}},
           GetDevices({0, 1, 2, 3, 4, 5}), MemoryKind())
           .value(),
      *ShardingParamSharding::Create(
           ShardingParam{/*dim_shards=*/{2, 3},
                         {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}},
           GetDevices({3, 4, 5, 0, 1, 2}), MemoryKind())
           .value(),
  }));
}

INSTANTIATE_TEST_SUITE_P(NumDevices, SingleDeviceShardingTest,
                         testing::Values(test_util::DeviceTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/4}));
INSTANTIATE_TEST_SUITE_P(NumDevices, OpaqueShardingTest,
                         testing::Values(test_util::DeviceTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/4}));
INSTANTIATE_TEST_SUITE_P(NumDevices, ConcreteShardingTest,
                         testing::Values(test_util::DeviceTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/4}));
INSTANTIATE_TEST_SUITE_P(NumDevices, ConcreteEvenShardingTest,
                         testing::Values(test_util::DeviceTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/4}));
INSTANTIATE_TEST_SUITE_P(NumDevices, ShardingParamShardingTest,
                         testing::Values(test_util::DeviceTestParam{
                             /*num_devices=*/6,
                             /*num_addressable_devices=*/4}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
