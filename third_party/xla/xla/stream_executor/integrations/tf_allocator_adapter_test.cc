/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/integrations/tf_allocator_adapter.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace {

// Each allocation will have an incrementing address.
class TestAllocator : public tsl::Allocator {
 public:
  explicit TestAllocator(
      size_t start_address,
      std::shared_ptr<absl::flat_hash_set<void*>> allocations = nullptr)
      : start_address_(start_address), allocations_(allocations) {
    if (allocations_ == nullptr) {
      allocations_ = std::make_shared<absl::flat_hash_set<void*>>();
    }
  }

  std::string Name() override { return "test"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    void* ptr = reinterpret_cast<void*>(++start_address_);
    allocations_->insert(ptr);
    return ptr;
  }

  void DeallocateRaw(void* ptr) override {
    auto it = allocations_->find(ptr);
    if (it == allocations_->end()) {
      ADD_FAILURE() << "Allocation not found (double free?)";
    } else {
      allocations_->erase(it);
    }
  }

 private:
  size_t start_address_;
  std::shared_ptr<absl::flat_hash_set<void*>> allocations_;
};

TEST(MultiDeviceAdapter, UsesCorrectAllocator) {
  TF_ASSERT_OK_AND_ASSIGN(auto* platform,
                          xla::PlatformUtil::GetDefaultPlatform());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<StreamExecutor*> executors,
                          xla::PlatformUtil::GetStreamExecutors(platform));
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executors[0]->CreateStream());

  std::vector<MultiDeviceAdapter::AllocatorInfo> infos;
  infos.emplace_back(std::make_unique<TestAllocator>(0x1000), stream.get(),
                     /*memory_space=*/0, /*device_ordinal=*/0);
  infos.emplace_back(std::make_unique<TestAllocator>(0x2000), stream.get(),
                     /*memory_space=*/0, /*device_ordinal=*/1);
  infos.emplace_back(std::make_unique<TestAllocator>(0x3000), stream.get(),
                     /*memory_space=*/1, /*device_ordinal=*/0);
  infos.emplace_back(std::make_unique<TestAllocator>(0x4000), stream.get(),
                     /*memory_space=*/1, /*device_ordinal=*/1);
  std::unique_ptr<DeviceAddressAllocator> allocator =
      std::make_unique<MultiDeviceAdapter>(platform, std::move(infos));

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedDeviceAddress<uint8_t> buff0,
      allocator->Allocate(/*device_ordinal=*/0, 4, false, /*memory_space=*/0));
  CHECK_EQ(reinterpret_cast<size_t>(buff0->opaque()), 0x1001);
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedDeviceAddress<uint8_t> buff1,
      allocator->Allocate(/*device_ordinal=*/0, 4, false, /*memory_space=*/0));
  CHECK_EQ(reinterpret_cast<size_t>(buff1->opaque()), 0x1002);
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedDeviceAddress<uint8_t> buff2,
      allocator->Allocate(/*device_ordinal=*/0, 4, false, /*memory_space=*/1));
  CHECK_EQ(reinterpret_cast<size_t>(buff2->opaque()), 0x3001);
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedDeviceAddress<uint8_t> buff3,
      allocator->Allocate(/*device_ordinal=*/1, 4, false, /*memory_space=*/0));
  CHECK_EQ(reinterpret_cast<size_t>(buff3->opaque()), 0x2001);
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedDeviceAddress<uint8_t> buff4,
      allocator->Allocate(/*device_ordinal=*/1, 4, false, /*memory_space=*/1));
  CHECK_EQ(reinterpret_cast<size_t>(buff4->opaque()), 0x4001);
}

TEST(MultiDeviceAdapter, DeallocationWithDifferentAllocator) {
  TF_ASSERT_OK_AND_ASSIGN(auto* platform,
                          xla::PlatformUtil::GetDefaultPlatform());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<StreamExecutor*> executors,
                          xla::PlatformUtil::GetStreamExecutors(platform));
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executors[0]->CreateStream());

  std::shared_ptr<absl::flat_hash_set<void*>> allocations =
      std::make_shared<absl::flat_hash_set<void*>>();
  std::vector<MultiDeviceAdapter::AllocatorInfo> info_allocator;
  info_allocator.emplace_back(
      std::make_unique<TestAllocator>(0x1000, allocations), stream.get(),
      /*memory_space=*/0, /*device_ordinal=*/0);

  std::unique_ptr<DeviceAddressAllocator> allocator =
      std::make_unique<MultiDeviceAdapter>(platform, std::move(info_allocator));

  std::vector<MultiDeviceAdapter::AllocatorInfo> info_deallocator;
  info_deallocator.emplace_back(
      std::make_unique<TestAllocator>(0x1000, allocations), stream.get(),
      /*memory_space=*/0, /*device_ordinal=*/0);
  std::unique_ptr<DeviceAddressAllocator> deallocator =
      std::make_unique<MultiDeviceAdapter>(platform,
                                           std::move(info_deallocator));

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedDeviceAddress<uint8_t> buff0,
      allocator->Allocate(/*device_ordinal=*/0, 4, false, /*memory_space=*/0));
  CHECK_EQ(allocations->size(), 1);
  CHECK_EQ(reinterpret_cast<size_t>(buff0->opaque()), 0x1001);

  CHECK_OK(deallocator->Deallocate(/*device_ordinal=*/0, buff0.cref()));
  CHECK_EQ(allocations->size(), 0);

  // Place back memory pointer to remove it during with ScopedDeviceAddress
  // destruction.
  allocations->insert(buff0->opaque());
}

TEST(MemoryAllocationError, IsMemoryAllocationError) {
  EXPECT_TRUE(IsMemoryAllocationError(
      MemoryAllocationError(100, /*is_host_mem=*/false)));
  EXPECT_FALSE(IsMemoryAllocationError(absl::OkStatus()));
  EXPECT_FALSE(IsMemoryAllocationError(absl::InternalError("")));
}

}  // namespace
}  // namespace stream_executor
