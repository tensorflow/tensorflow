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

#include "xla/stream_executor/rocm/rocm_device_address_vmm_allocator.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace {

using ::absl_testing::IsOk;

class DeviceAddressVmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("ROCM");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "ROCM platform not available";
    }
    platform_ = platform_or.value();

    auto executor_or = platform_->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "ROCM executor not available: " << executor_or.status();
    }
    executor_ = executor_or.value();

    auto stream_or = executor_->CreateStream();
    if (!stream_or.ok()) {
      GTEST_SKIP() << "Failed to create stream";
    }
    stream_ = std::move(stream_or.value());

    auto probe =
        gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get());
    if (!probe.ok()) {
      GTEST_SKIP() << "RocmDeviceAddressVmmAllocator not supported: "
                   << probe.status();
    }
  }

  Platform* platform_ = nullptr;
  StreamExecutor* executor_ = nullptr;
  std::unique_ptr<Stream> stream_;
};

TEST_F(DeviceAddressVmmAllocatorTest, AllocateAndDeallocate) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  EXPECT_FALSE(scoped_address.is_null());
  EXPECT_EQ(scoped_address->size(), 1024);
  EXPECT_NE(
      allocator->GetRawAllocation(executor_->device_ordinal(), *scoped_address),
      nullptr);
  EXPECT_NE(
      allocator->GetReservation(executor_->device_ordinal(), *scoped_address),
      nullptr);
}

TEST_F(DeviceAddressVmmAllocatorTest, AllocateZeroSize) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 0,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  EXPECT_TRUE(scoped_address.is_null());
}

TEST_F(DeviceAddressVmmAllocatorTest, AllocateMultiple) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr1,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr2,
      allocator->Allocate(executor_->device_ordinal(), 2048,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  EXPECT_FALSE(addr1.is_null());
  EXPECT_FALSE(addr2.is_null());
  EXPECT_NE(addr1->opaque(), addr2->opaque());
  EXPECT_EQ(addr1.cref().size(), 1024);
  EXPECT_EQ(addr2.cref().size(), 2048);
}

TEST_F(DeviceAddressVmmAllocatorTest, MemoryReadWrite) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_NE(scoped_address->opaque(), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());

  constexpr uint64_t kTestValue = 0xDEADBEEFCAFEBABE;
  DeviceAddressBase addr = scoped_address.cref();
  EXPECT_THAT(stream->Memcpy(&addr, &kTestValue, sizeof(kTestValue)),
              absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  uint64_t read_value = 0;
  EXPECT_THAT(stream->Memcpy(&read_value, addr, sizeof(read_value)),
              absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  EXPECT_EQ(read_value, kTestValue);
}

TEST_F(DeviceAddressVmmAllocatorTest, GetStream) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  TF_ASSERT_OK_AND_ASSIGN(Stream * stream,
                          allocator->GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream_.get());

  TF_ASSERT_OK_AND_ASSIGN(Stream * stream2,
                          allocator->GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream2);
}

TEST_F(DeviceAddressVmmAllocatorTest, GetStreamExecutor) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      StreamExecutor * se,
      allocator->GetStreamExecutor(executor_->device_ordinal()));
  EXPECT_EQ(se, executor_);
}

TEST_F(DeviceAddressVmmAllocatorTest, AllowsAsynchronousDeallocation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  EXPECT_TRUE(allocator->AllowsAsynchronousDeallocation());
}

TEST_F(DeviceAddressVmmAllocatorTest, ExplicitDeallocate) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_NE(scoped_address->opaque(), nullptr);
  DeviceAddressBase addr = scoped_address.cref();

  EXPECT_THAT(allocator->Deallocate(executor_->device_ordinal(), addr),
              absl_testing::IsOk());

  scoped_address.Release();
}

TEST_F(DeviceAddressVmmAllocatorTest, DeallocateNull) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  DeviceAddressBase null_addr;
  EXPECT_THAT(allocator->Deallocate(executor_->device_ordinal(), null_addr),
              absl_testing::IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       PendingDeallocationReusesSameVirtualAddress) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  constexpr uint64_t kSize = 1024;

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr1,
      allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  void* const va = addr1->opaque();

  DeviceAddressBase raw = addr1.cref();
  addr1.Release();
  ASSERT_THAT(allocator->Deallocate(ordinal, raw), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr2,
      allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  EXPECT_EQ(addr2->opaque(), va);

  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       DeferredDeallocationSafeWhileGpuWritesData) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, sizeof(uint64_t), /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  constexpr uint64_t kPattern = 0xCAFEBABEDEADBEEFULL;
  DeviceAddressBase dev_addr = addr.cref();
  ASSERT_THAT(stream_->Memcpy(&dev_addr, &kPattern, sizeof(kPattern)), IsOk());

  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());

  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MultipleSeqnosAllCompleteAfterStreamSync) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  constexpr int kCount = 8;
  constexpr uint64_t kSize = 1024;

  for (int i = 0; i < kCount; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  }

  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());

  for (int i = 0; i < kCount; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    EXPECT_FALSE(addr.is_null());
  }

  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       DestructorWithPendingDeallocationsDoesNotCrash) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();

  for (int i = 0; i < 4; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, 1024, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  }

  allocator.reset();
}

TEST_F(DeviceAddressVmmAllocatorTest, UnknownDeviceOrdinalReturnsError) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int unknown_ordinal = 9999;
  EXPECT_FALSE(allocator
                   ->Allocate(unknown_ordinal, 1024, /*retry_on_failure=*/true,
                              static_cast<int64_t>(MemorySpace::kCollective))
                   .ok());
  EXPECT_THAT(allocator->Deallocate(unknown_ordinal, DeviceAddressBase{}),
              absl_testing::IsOk());
  DeviceAddressBase fake_addr(reinterpret_cast<void*>(0x1000), 64);
  EXPECT_FALSE(allocator->Deallocate(unknown_ordinal, fake_addr).ok());
  EXPECT_FALSE(allocator->GetStream(unknown_ordinal).ok());
  EXPECT_FALSE(allocator->GetStreamExecutor(unknown_ordinal).ok());
}

class MultiDeviceVmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("ROCM");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "ROCM platform not available";
    }
    platform_ = platform_or.value();

    if (platform_->VisibleDeviceCount() < 2) {
      GTEST_SKIP() << "Fewer than two ROCm devices available";
    }

    for (int i = 0; i < 2; ++i) {
      auto executor_or = platform_->ExecutorForDevice(i);
      if (!executor_or.ok()) {
        GTEST_SKIP() << "ROCM executor not available for device " << i;
      }
      executors_.push_back(executor_or.value());

      auto stream_or = executors_.back()->CreateStream();
      if (!stream_or.ok()) {
        GTEST_SKIP() << "Failed to create stream for device " << i;
      }
      streams_.push_back(std::move(stream_or.value()));
    }

    auto probe = gpu::RocmDeviceAddressVmmAllocator::Create(executors_[0],
                                                            streams_[0].get());
    if (!probe.ok()) {
      GTEST_SKIP() << "RocmDeviceAddressVmmAllocator not supported: "
                   << probe.status();
    }
  }

  Platform* platform_ = nullptr;
  std::vector<StreamExecutor*> executors_;
  std::vector<std::unique_ptr<Stream>> streams_;
};

TEST_F(MultiDeviceVmmAllocatorTest, AllocateOnBothDevices) {
  std::vector<DeviceAddressVmmAllocator::DeviceConfig> configs;
  for (int i = 0; i < 2; ++i) {
    configs.push_back({executors_[i], streams_[i].get()});
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(platform_, configs));

  for (int i = 0; i < 2; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(executors_[i]->device_ordinal(), 1024,
                            /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    EXPECT_FALSE(addr.is_null());
    EXPECT_EQ(addr->size(), 1024);
    EXPECT_NE(
        allocator->GetRawAllocation(executors_[i]->device_ordinal(), *addr),
        nullptr);
    EXPECT_NE(allocator->GetReservation(executors_[i]->device_ordinal(), *addr),
              nullptr);
    TF_ASSERT_OK_AND_ASSIGN(
        StreamExecutor * se,
        allocator->GetStreamExecutor(executors_[i]->device_ordinal()));
    EXPECT_EQ(se, executors_[i]);
    TF_ASSERT_OK_AND_ASSIGN(
        Stream * stream, allocator->GetStream(executors_[i]->device_ordinal()));
    EXPECT_EQ(stream, streams_[i].get());
  }
}

TEST_F(MultiDeviceVmmAllocatorTest, AllocationOnOneDeviceDoesNotAffectOther) {
  std::vector<DeviceAddressVmmAllocator::DeviceConfig> configs;
  for (int i = 0; i < 2; ++i) {
    configs.push_back({executors_[i], streams_[i].get()});
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::RocmDeviceAddressVmmAllocator::Create(platform_, configs));

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr0,
      allocator->Allocate(executors_[0]->device_ordinal(), 4096,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  EXPECT_FALSE(addr0.is_null());

  EXPECT_EQ(
      allocator->GetRawAllocation(executors_[1]->device_ordinal(), *addr0),
      nullptr);
}

}  // namespace
}  // namespace stream_executor
