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

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::Optional;
using ::testing::Return;
using ::tsl::testing::IsOkAndHolds;

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
                             /*tail_padding_alignment_in_elements=*/1,
                             /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
                             /*element_primitive_type=*/PRIMITIVE_TYPE_INVALID,
                             /*element_size_in_bits=*/32)))
          ->ByteSize(DType(DType::kS32), Shape({1, 127})),
      IsOkAndHolds(Optional(4 * (2 * 128))));
  EXPECT_THAT(
      PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(xla::Layout(
                             /*minor_to_major=*/{1, 0},
                             /*tiles=*/{xla::Tile({2, 1024})},
                             /*tail_padding_alignment_in_elements=*/1,
                             /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
                             /*element_primitive_type=*/PRIMITIVE_TYPE_INVALID,
                             /*element_size_in_bits=*/4)))
          ->ByteSize(DType(DType::kS4), Shape({1, 1023})),
      IsOkAndHolds(Optional((2 * 1024) / 2)));
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
    EXPECT_CALL(*client, GetDefaultLayout)
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
