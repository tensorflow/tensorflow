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

#include "xla/python/pjrt_ifrt/pjrt_layout_migration_util.h"

#include <cstdint>
#include <memory>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::Return;
using ::testing::ReturnRef;

class PjRtLayoutMigrationUtilTest : public testing::Test {
 public:
  PjRtLayoutMigrationUtilTest() : shape_(Shape({3, 2})) {
    client_ = std::make_unique<MockClient>();
    ON_CALL(*client_, MakeDeviceList)
        .WillByDefault([](absl::Span<Device* const> devices) -> DeviceListRef {
          return BasicDeviceList::Create(devices);
        });

    memory_ = std::make_unique<MockMemory>();
    memory_kind_ = MemoryKind("memory kind");
    ON_CALL(*memory_, Kind()).WillByDefault(ReturnRef(memory_kind_));

    device_ = std::make_unique<MockDevice>();
    ON_CALL(*device_, client()).WillByDefault(Return(client_.get()));
    ON_CALL(*device_, DefaultMemory()).WillByDefault(Return(memory_.get()));

    ON_CALL(*client_, GetDefaultPjRtLayout)
        .With(std::make_tuple(DType(DType::kS32), shape_.dims(),
                              static_cast<Device*>(device_.get()),
                              memory_kind_))
        .WillByDefault(
            [](DType dtype, absl::Span<const int64_t> dims, Device* device,
               MemoryKind memory_kind)
                -> absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> {
              // Typically the default layout is descending. We use an ascending
              // layout here for testing purposes.
              return std::make_shared<xla::PjRtLayout>(
                  xla::LayoutUtil::MakeAscendingLayout(2));
            });

    ON_CALL(*client_, GetDefaultPjRtLayout)
        .With(std::make_tuple(DType(DType::kF32), shape_.dims(),
                              static_cast<Device*>(device_.get()),
                              memory_kind_))
        .WillByDefault(
            [](DType dtype, absl::Span<const int64_t> dims, Device* device,
               MemoryKind memory_kind)
                -> absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> {
              return absl::UnimplementedError("Unimplemented");
            });
    ON_CALL(*client_, GetDefaultLayout)
        .WillByDefault(
            [this](DType dtype, const Shape& shape, const ShardingRef& sharding)
                -> absl::StatusOr<CustomLayoutRef> {
              return GetDefaultLayoutUsingDefaultPjRtLayout(
                  client_.get(), dtype, shape, sharding);
            });

    sharding_ = SingleDeviceSharding::Create(device_.get(), memory_kind_);
  }

 protected:
  std::unique_ptr<MockClient> client_;
  std::unique_ptr<MockDevice> device_;
  std::unique_ptr<MockMemory> memory_;
  MemoryKind memory_kind_;
  Shape shape_;
  ShardingRef sharding_;
};

TEST_F(PjRtLayoutMigrationUtilTest, GetDefaultLayoutFromDefaultPjRtLayout) {
  TF_ASSERT_OK_AND_ASSIGN(
      CustomLayoutRef layout,
      client_->GetDefaultLayout(DType(DType::kS32), shape_, sharding_));
  ASSERT_TRUE(llvm::isa<PjRtLayout>(*layout));
  EXPECT_EQ(llvm::cast<PjRtLayout>(*layout).pjrt_layout()->xla_layout(),
            xla::LayoutUtil::MakeAscendingLayout(2));
}

TEST_F(PjRtLayoutMigrationUtilTest, GetArrayLayoutFromPjRtLayout) {
  {
    // If `Array::pjrt_layout()` is implemented, we use it for
    // `Array::layout()`.
    auto array = tsl::MakeRef<MockArray>();
    ON_CALL(*array, client()).WillByDefault(Return(client_.get()));
    ON_CALL(*array, dtype()).WillByDefault(Return(DType(DType::kS32)));
    ON_CALL(*array, shape()).WillByDefault(ReturnRef(shape_));
    ON_CALL(*array, shared_ptr_sharding()).WillByDefault(Return(sharding_));
    ON_CALL(*array, pjrt_layout())
        .WillByDefault(Return(std::make_shared<xla::PjRtLayout>(
            xla::LayoutUtil::MakeAscendingLayout(2))));
    ON_CALL(*array, layout())
        .WillByDefault([array = array.get()]() -> CustomLayoutRef {
          return GetArrayLayoutUsingPjRtLayout(array);
        });

    CustomLayoutRef layout = array->layout();
    ASSERT_TRUE(llvm::isa<PjRtLayout>(*layout));
    EXPECT_EQ(llvm::cast<PjRtLayout>(*layout).pjrt_layout()->xla_layout(),
              xla::LayoutUtil::MakeAscendingLayout(2));
  }
  {
    // If `Array::pjrt_layout()` is unimplemented, we fall back to querying the
    // client for the default layout.
    auto array = tsl::MakeRef<MockArray>();
    ON_CALL(*array, client()).WillByDefault(Return(client_.get()));
    ON_CALL(*array, dtype()).WillByDefault(Return(DType(DType::kS32)));
    ON_CALL(*array, shape()).WillByDefault(ReturnRef(shape_));
    ON_CALL(*array, shared_ptr_sharding()).WillByDefault(Return(sharding_));
    ON_CALL(*array, pjrt_layout())
        .WillByDefault(Return(absl::UnimplementedError("Unimplemented")));
    ON_CALL(*array, layout())
        .WillByDefault([array = array.get()]() -> CustomLayoutRef {
          return GetArrayLayoutUsingPjRtLayout(array);
        });

    CustomLayoutRef layout = array->layout();
    ASSERT_TRUE(llvm::isa<PjRtLayout>(*layout));
    EXPECT_EQ(llvm::cast<PjRtLayout>(*layout).pjrt_layout()->xla_layout(),
              xla::LayoutUtil::MakeAscendingLayout(2));
  }
  {
    // If the client default layout is also unimplemented, we fall back to a
    // descending layout to match the PjRt behavior for unimplemented layout
    // methods.
    auto array = tsl::MakeRef<MockArray>();
    ON_CALL(*array, client()).WillByDefault(Return(client_.get()));
    ON_CALL(*array, dtype()).WillByDefault(Return(DType(DType::kF32)));
    ON_CALL(*array, shape()).WillByDefault(ReturnRef(shape_));
    ON_CALL(*array, shared_ptr_sharding()).WillByDefault(Return(sharding_));
    ON_CALL(*array, pjrt_layout())
        .WillByDefault(Return(absl::UnimplementedError("Unimplemented")));
    ON_CALL(*array, layout())
        .WillByDefault([array = array.get()]() -> CustomLayoutRef {
          return GetArrayLayoutUsingPjRtLayout(array);
        });

    CustomLayoutRef layout = array->layout();
    ASSERT_TRUE(llvm::isa<PjRtLayout>(*layout));
    EXPECT_EQ(llvm::cast<PjRtLayout>(*layout).pjrt_layout()->xla_layout(),
              xla::LayoutUtil::MakeDescendingLayout(2));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
