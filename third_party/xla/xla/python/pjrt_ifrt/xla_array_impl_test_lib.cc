/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"

namespace xla {
namespace ifrt {
namespace {

class XlaArrayImplHashTest : public ::testing::TestWithParam<Client::HashMode> {
};

TEST_P(XlaArrayImplHashTest, HashValuesDifferentLayouts) {
  ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  if (client->addressable_devices().size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 devices";
  }
  Client::HashMode hash_mode = GetParam();

  DType dtype(DType::kF32);
  Shape shape({2, 3});

  std::vector<float> data(6);
  absl::c_iota(data, 0.0f);

  // Array 0: Row-major layout
  std::shared_ptr<const xla::ifrt::PjRtLayout> layout0 =
      xla::ifrt::PjRtLayout::Create(std::make_shared<const xla::PjRtLayout>(
          xla::LayoutUtil::MakeDescendingLayout(/*num_dims=*/2)));
  ASSERT_OK_AND_ASSIGN(
      ArrayRef array0,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt,
          SingleDeviceSharding::Create(client->addressable_devices().at(0),
                                       MemoryKind()),
          layout0, Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> actual_layout0 =
      array0->pjrt_layout();
  if (absl::IsUnimplemented(actual_layout0.status())) {
    GTEST_SKIP() << "Custom layout not supported";
  }
  ASSERT_OK(actual_layout0.status());

  // Array 1: Column-major layout
  std::shared_ptr<const xla::ifrt::PjRtLayout> layout1 =
      xla::ifrt::PjRtLayout::Create(std::make_shared<const xla::PjRtLayout>(
          xla::LayoutUtil::MakeAscendingLayout(/*num_dims=*/2)));
  ASSERT_OK_AND_ASSIGN(
      ArrayRef array1,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt,
          SingleDeviceSharding::Create(client->addressable_devices().at(0),
                                       MemoryKind()),
          layout1, Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> actual_layout1 =
      array1->pjrt_layout();
  if (absl::IsUnimplemented(actual_layout1.status())) {
    GTEST_SKIP() << "Custom layout not supported";
  }
  ASSERT_OK(actual_layout1.status());

  absl::StatusOr<std::vector<uint64_t>> result0 =
      client->HashValues({array0}, hash_mode).Await();
  if (absl::IsUnimplemented(result0.status())) {
    GTEST_SKIP() << "HashValues not implemented";
  }
  ASSERT_OK(result0.status());

  ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> result1,
                       client->HashValues({array1}, hash_mode).Await());

  switch (hash_mode) {
    case Client::HashMode::kPhysical:
      EXPECT_NE(result0->front(), result1.front());
      break;
    case Client::HashMode::kLogical:
      EXPECT_EQ(result0->front(), result1.front());
      break;
  }
}

INSTANTIATE_TEST_SUITE_P(
    XlaArrayImplHashTests, XlaArrayImplHashTest,
    ::testing::Values(Client::HashMode::kPhysical, Client::HashMode::kLogical),
    [](const ::testing::TestParamInfo<Client::HashMode>& info) {
      switch (info.param) {
        case Client::HashMode::kPhysical:
          return "Physical";
        case Client::HashMode::kLogical:
          return "Logical";
      }
    });

}  // namespace
}  // namespace ifrt
}  // namespace xla
