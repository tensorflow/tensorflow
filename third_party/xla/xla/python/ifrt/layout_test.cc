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

#include "xla/python/ifrt/layout.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::Return;
using ::testing::ReturnRef;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(CompactLayoutTest, Create) {
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->major_to_minor(), ElementsAre());
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({1, 0}));
    EXPECT_THAT(layout->major_to_minor(), ElementsAre(1, 0));
  }
}

TEST(CompactLayoutTest, CreateCOrder) {
  EXPECT_THAT(CompactLayout::CreateCOrder(0)->major_to_minor(), ElementsAre());
  EXPECT_THAT(CompactLayout::CreateCOrder(2)->major_to_minor(),
              ElementsAre(0, 1));
}

TEST(CompactLayoutTest, ByteSize) {
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kToken), Shape({})),
                IsOkAndHolds(std::nullopt));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kOpaque), Shape({})),
                IsOkAndHolds(std::nullopt));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kString), Shape({})),
                IsOkAndHolds(std::nullopt));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kS8), Shape({})),
                IsOkAndHolds(Optional(1)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kS32), Shape({})),
                IsOkAndHolds(Optional(4)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({1, 0}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kS32), Shape({3, 2})),
                IsOkAndHolds(Optional(24)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({1, 0}));
    EXPECT_THAT(layout->ByteSize(DType(DType::kS4), Shape({3, 2})),
                IsOkAndHolds(Optional(3)));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({}));
    EXPECT_THAT(
        layout->ByteSize(DType(DType::kS32), Shape({3, 2})),
        StatusIs(tsl::error::INVALID_ARGUMENT,
                 HasSubstr(
                     "CompactLayout expects Shape with the same number of "
                     "dimensions as major_to_minor [], but got shard_shape=")));
  }
}

TEST(LayoutTest, EquivalentLayouts) {
  auto client = std::make_unique<MockClient>();
  ON_CALL(*client, MakeDeviceList)
      .WillByDefault([](absl::Span<Device* const> devices) -> DeviceListRef {
        return BasicDeviceList::Create(devices);
      });

  Shape shape({3, 2});

  auto memory0 = std::make_unique<MockMemory>();
  auto memory1 = std::make_unique<MockMemory>();
  auto memory2 = std::make_unique<MockMemory>();
  MemoryKind memory_kind0("memory kind 0");
  ON_CALL(*memory0, Kind()).WillByDefault(ReturnRef(memory_kind0));
  ON_CALL(*memory1, Kind()).WillByDefault(ReturnRef(memory_kind0));
  ON_CALL(*memory2, Kind()).WillByDefault(ReturnRef(memory_kind0));

  auto device0 = std::make_unique<MockDevice>();
  auto device1 = std::make_unique<MockDevice>();
  auto device2 = std::make_unique<MockDevice>();
  ON_CALL(*device0, client()).WillByDefault(Return(client.get()));
  ON_CALL(*device1, client()).WillByDefault(Return(client.get()));
  ON_CALL(*device2, client()).WillByDefault(Return(client.get()));
  ON_CALL(*device0, DefaultMemory()).WillByDefault(Return(memory0.get()));
  ON_CALL(*device1, DefaultMemory()).WillByDefault(Return(memory1.get()));
  ON_CALL(*device2, DefaultMemory()).WillByDefault(Return(memory2.get()));

  ON_CALL(*client, GetDefaultPjRtLayout)
      .With(std::make_tuple(DType(DType::kS32), shape.dims(),
                            static_cast<Device*>(device0.get()), memory_kind0))
      .WillByDefault(
          [](DType dtype, absl::Span<const int64_t> dims, Device* device,
             MemoryKind memory_kind)
              -> absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> {
            return std::make_shared<xla::PjRtLayout>(
                xla::LayoutUtil::MakeDescendingLayout(2));
          });
  ON_CALL(*client, GetDefaultPjRtLayout)
      .With(std::make_tuple(DType(DType::kS32), shape.dims(),
                            static_cast<Device*>(device1.get()), memory_kind0))
      .WillByDefault(
          [](DType dtype, absl::Span<const int64_t> dims, Device* device,
             MemoryKind memory_kind)
              -> absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> {
            return std::make_shared<xla::PjRtLayout>(
                xla::LayoutUtil::MakeDescendingLayout(2));
          });

  ON_CALL(*client, GetDefaultPjRtLayout)
      .With(std::make_tuple(DType(DType::kS32), shape.dims(),
                            static_cast<Device*>(device2.get()), memory_kind0))
      .WillByDefault(
          [](DType dtype, absl::Span<const int64_t> dims, Device* device,
             MemoryKind memory_kind)
              -> absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> {
            return std::make_shared<xla::PjRtLayout>(
                xla::LayoutUtil::MakeAscendingLayout(2));
          });

  // A concrete layout and a default layout are not equivalent.
  {
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout0, CompactLayout::Create({1, 0}));
    LayoutRef layout1 = nullptr;
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(false));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1,
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0),
        IsOkAndHolds(false));
  }

  // Two same concrete layouts are equivalent.
  {
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout0, CompactLayout::Create({1, 0}));
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout1, CompactLayout::Create({1, 0}));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(true));
  }
  // Two different concrete layouts are not equivalent.
  {
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout0, CompactLayout::Create({1, 0}));
    TF_ASSERT_OK_AND_ASSIGN(LayoutRef layout1, CompactLayout::Create({0, 1}));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(false));
  }

  // Default layouts are equivalent if they resolve to the same concrete layout.
  {
    LayoutRef layout0 = nullptr;
    LayoutRef layout1 = nullptr;
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout1),
        IsOkAndHolds(true));
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device1.get(), MemoryKind()), layout1),
        IsOkAndHolds(true));
  }
  // Default layouts are not equivalent if they resolve to different concrete
  // layouts.
  {
    LayoutRef layout0 = nullptr;
    LayoutRef layout1 = nullptr;
    EXPECT_THAT(
        EquivalentLayouts(
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device0.get(), MemoryKind()), layout0,
            DType(DType::kS32), shape,
            SingleDeviceSharding::Create(device2.get(), MemoryKind()), layout1),
        IsOkAndHolds(false));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
