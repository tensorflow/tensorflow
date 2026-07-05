/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/pjrt_layout.h"

#include <memory>
#include <optional>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Optional;
using ::testing::Return;

TEST(PjRtLayoutTest, Create) {
  EXPECT_EQ(PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                                   xla::LayoutUtil::MakeDescendingLayout(2)))
                ->pjrt_layout()
                ->xla_layout(),
            xla::LayoutUtil::MakeDescendingLayout(2));
}

TEST(PjRtLayoutTest, ByteSize) {
  EXPECT_THAT(
      PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(xla::Layout()))
          ->ByteSize(DType(DType::kToken), Shape({})),
      IsOkAndHolds(std::nullopt));
  EXPECT_THAT(
      PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(xla::Layout()))
          ->ByteSize(DType(DType::kOpaque), Shape({})),
      IsOkAndHolds(std::nullopt));

  EXPECT_THAT(PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                                     xla::LayoutUtil::MakeDescendingLayout(0)))
                  ->ByteSize(DType(DType::kS32), Shape({})),
              IsOkAndHolds(Optional(4)));
  EXPECT_THAT(PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                                     xla::LayoutUtil::MakeDescendingLayout(2)))
                  ->ByteSize(DType(DType::kS32), Shape({3, 2})),
              IsOkAndHolds(Optional(24)));
  EXPECT_THAT(
      PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(xla::Layout(
                             /*minor_to_major=*/{1, 0},
                             /*tiles=*/{xla::Tile({2, 128})},
                             /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
                             /*element_primitive_type=*/PRIMITIVE_TYPE_INVALID,
                             /*tail_padding_alignment_in_elements=*/1,
                             /*element_size_in_bits=*/32)))
          ->ByteSize(DType(DType::kS32), Shape({1, 127})),
      IsOkAndHolds(Optional(4 * (2 * 128))));
  EXPECT_THAT(
      PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(xla::Layout(
                             /*minor_to_major=*/{1, 0},
                             /*tiles=*/{xla::Tile({2, 1024})},
                             /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
                             /*element_primitive_type=*/PRIMITIVE_TYPE_INVALID,
                             /*tail_padding_alignment_in_elements=*/1,
                             /*element_size_in_bits=*/4)))
          ->ByteSize(DType(DType::kS4), Shape({1, 1023})),
      IsOkAndHolds(Optional(2 * 1024 / 2)));
}

TEST(PjRtLayoutTest, ByteSizeStatic) {
  auto client = std::make_shared<MockClient>();
  ON_CALL(*client, MakeDeviceList)
      .WillByDefault([](absl::Span<Device* const> devices) -> DeviceListRef {
        return BasicDeviceList::Create(devices);
      });
  Shape shape({6, 2});
  Shape shard_shape({3, 2});
  auto device0 = std::make_unique<MockDevice>();
  auto device1 = std::make_unique<MockDevice>();
  ON_CALL(*device0, client()).WillByDefault(Return(client.get()));
  ON_CALL(*device1, client()).WillByDefault(Return(client.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list,
      client->MakeDeviceList({device0.get(), device1.get()}));
  ShardingRef sharding = ConcreteEvenSharding::Create(
      device_list, MemoryKind(), shape, shard_shape,
      /*is_fully_replicated=*/false);

  auto pjrt_layout = std::make_shared<const xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(2));

  // Using a custom layout.
  EXPECT_THAT(
      PjRtLayout::ByteSize(DType(DType::kS32), shape, sharding, pjrt_layout),
      IsOkAndHolds(Optional(24)));

  // Using a default layout.
  EXPECT_CALL(*client,
              GetDefaultPjRtLayout(DType(DType::kS32), shard_shape.dims(),
                                   device0.get(), MemoryKind()))
      .WillOnce(Return(pjrt_layout));

  EXPECT_THAT(PjRtLayout::ByteSize(DType(DType::kS32), shape, sharding,
                                   /*pjrt_layout=*/
                                   std::shared_ptr<const xla::PjRtLayout>()),
              IsOkAndHolds(Optional(24)));
}

TEST(PjRtLayoutTest, NulloptByteSizeStatic) {
  auto client = std::make_shared<MockClient>();
  ON_CALL(*client, MakeDeviceList)
      .WillByDefault([](absl::Span<Device* const> devices) -> DeviceListRef {
        return BasicDeviceList::Create(devices);
      });
  auto device0 = std::make_unique<MockDevice>();
  auto device1 = std::make_unique<MockDevice>();
  ON_CALL(*device0, client()).WillByDefault(Return(client.get()));
  ON_CALL(*device1, client()).WillByDefault(Return(client.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list,
      client->MakeDeviceList({device0.get(), device1.get()}));

  {
    Shape shape({});
    Shape shard_shape({});
    ShardingRef sharding = ConcreteEvenSharding::Create(
        device_list, MemoryKind(), shape, shard_shape,
        /*is_fully_replicated=*/false);
    auto pjrt_layout = std::make_shared<const xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(0));
    EXPECT_THAT(PjRtLayout::ByteSize(DType(DType::kToken), shape, sharding,
                                     pjrt_layout),
                IsOkAndHolds(std::nullopt));
  }
  {
    Shape shape({5, 2});
    ShardingRef sharding = ConcreteSharding::Create(
        device_list, MemoryKind(), shape,
        /*shard_shapes=*/{Shape({3, 2}), Shape({2, 2})});
    auto pjrt_layout = std::make_shared<const xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(2));
    EXPECT_THAT(
        PjRtLayout::ByteSize(DType(DType::kS32), shape, sharding, pjrt_layout),
        IsOkAndHolds(std::nullopt));
  }
}

TEST(PjRtLayoutTest, ToPjRtLayout) {
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto layout,
        ToPjRtLayout(DType(DType::kS32), Shape({3, 2}),
                     PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                         xla::LayoutUtil::MakeDescendingLayout(2)))));
    EXPECT_EQ(layout->xla_layout(), xla::LayoutUtil::MakeDescendingLayout(2));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout,
                            ToPjRtLayout(DType(DType::kS32), Shape({3, 2}),
                                         CompactLayout::CreateCOrder(2)));
    EXPECT_EQ(layout->xla_layout(), xla::LayoutUtil::MakeDescendingLayout(2));
  }

  {
    auto client = std::make_shared<MockClient>();
    auto device = std::make_unique<MockDevice>();
    Shape shape({3, 2});
    ON_CALL(*device, client).WillByDefault(Return(client.get()));
    EXPECT_CALL(*client, GetDefaultPjRtLayout)
        .With(std::make_tuple(DType(DType::kS32), shape.dims(),
                              static_cast<Device*>(device.get()), MemoryKind()))
        .WillOnce(Return(absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>>(
            std::make_shared<xla::PjRtLayout>(
                xla::LayoutUtil::MakeDescendingLayout(2)))));
    TF_ASSERT_OK_AND_ASSIGN(
        auto layout, ToPjRtLayout(DType(DType::kS32), shape, device.get(),
                                  MemoryKind(), /*layout=*/nullptr));
    EXPECT_EQ(layout->xla_layout(), xla::LayoutUtil::MakeDescendingLayout(2));
  }
  {
    auto client = std::make_shared<MockClient>();
    auto device = std::make_unique<MockDevice>();
    TF_ASSERT_OK_AND_ASSIGN(
        auto layout,
        ToPjRtLayout(DType(DType::kS32), Shape({3, 2}), device.get(),
                     MemoryKind(),
                     PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                         xla::LayoutUtil::MakeDescendingLayout(2)))));
    EXPECT_EQ(layout->xla_layout(), xla::LayoutUtil::MakeDescendingLayout(2));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
