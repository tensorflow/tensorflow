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

#include "xla/stream_executor/device_address_vmm_allocator.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace {

using ::absl_testing::StatusIs;
using ::testing::NiceMock;
using ::testing::Return;

constexpr uint64_t kGranularity = 64;

uint64_t RoundUpTestSize(uint64_t size) {
  return ((size + kGranularity - 1) / kGranularity) * kGranularity;
}

class TestMemoryAllocation final : public MemoryAllocation {
 public:
  explicit TestMemoryAllocation(uint64_t size)
      : storage_(std::make_unique<uint8_t[]>(size)), size_(size) {}

  DeviceAddressBase address() const override {
    return DeviceAddressBase(storage_.get(), size_);
  }

 private:
  std::unique_ptr<uint8_t[]> storage_;
  uint64_t size_;
};

class TestMemoryReservation final : public MemoryReservation {
 public:
  explicit TestMemoryReservation(uint64_t size)
      : storage_(std::make_unique<uint8_t[]>(size)), size_(size) {}

  DeviceAddressBase address() const override {
    return DeviceAddressBase(storage_.get(), size_);
  }

  int active_mapping_count() const { return active_mapping_count_; }

 private:
  absl::Status Map(size_t reservation_offset, size_t allocation_offset,
                   size_t size, MemoryAllocation& allocation) override {
    if (reservation_offset > size_ || size > size_ - reservation_offset ||
        allocation_offset > allocation.address().size() ||
        size > allocation.address().size() - allocation_offset) {
      return absl::InvalidArgumentError("mapping range is out of bounds");
    }
    ++active_mapping_count_;
    return absl::OkStatus();
  }

  absl::Status SetAccess(uint64_t /*reservation_offset*/,
                         size_t /*size*/) override {
    return absl::OkStatus();
  }

  absl::Status UnMap(size_t /*reservation_offset*/, size_t /*size*/) override {
    if (active_mapping_count_ == 0) {
      return absl::FailedPreconditionError("reservation is not mapped");
    }
    --active_mapping_count_;
    return absl::OkStatus();
  }

  std::unique_ptr<uint8_t[]> storage_;
  uint64_t size_;
  int active_mapping_count_ = 0;
};

class TestDeviceAddressVmmAllocator final : public DeviceAddressVmmAllocator {
 public:
  static absl::StatusOr<std::unique_ptr<TestDeviceAddressVmmAllocator>> Create(
      const Platform* platform, absl::Span<const DeviceConfig> devices,
      uint64_t physical_size_padding = 0,
      std::function<void(int)> on_device_destroy = nullptr) {
    auto allocator = std::unique_ptr<TestDeviceAddressVmmAllocator>(
        new TestDeviceAddressVmmAllocator(platform, physical_size_padding,
                                          on_device_destroy));
    absl::Status status = PopulateDevices(allocator.get(), devices);
    if (!status.ok()) {
      return status;
    }
    return allocator;
  }

  int allocation_count() const { return allocation_count_; }

 protected:
  absl::Status InitializeDeviceState(PerDeviceState& state) override {
    state.allocation_granularity = kGranularity;
    auto* timeline = new uint64_t(0);
    state.pinned_timeline = timeline;
    int ordinal = state.executor->device_ordinal();
    state.destroy_fn = [timeline, ordinal,
                        on_device_destroy = on_device_destroy_] {
      delete timeline;
      if (on_device_destroy) {
        on_device_destroy(ordinal);
      }
    };
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> CreateAllocation(
      StreamExecutor* /*executor*/, uint64_t size) override {
    ++allocation_count_;
    return std::make_unique<TestMemoryAllocation>(RoundUpTestSize(size) +
                                                  physical_size_padding_);
  }

  absl::StatusOr<std::unique_ptr<MemoryReservation>> CreateReservation(
      StreamExecutor* /*executor*/, uint64_t size) override {
    return std::make_unique<TestMemoryReservation>(RoundUpTestSize(size) +
                                                   physical_size_padding_);
  }

  absl::Status EnqueueDeferredDeallocation(PerDeviceState& state,
                                           uint64_t seqno) override {
    __atomic_store_n(state.pinned_timeline, seqno, __ATOMIC_RELEASE);
    return absl::OkStatus();
  }

 private:
  TestDeviceAddressVmmAllocator(const Platform* platform,
                                uint64_t physical_size_padding,
                                std::function<void(int)> on_device_destroy)
      : DeviceAddressVmmAllocator(platform),
        physical_size_padding_(physical_size_padding),
        on_device_destroy_(on_device_destroy) {}

  uint64_t physical_size_padding_;
  std::function<void(int)> on_device_destroy_;
  int allocation_count_ = 0;
};

class DeviceAddressVmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ON_CALL(executor_, device_ordinal()).WillByDefault(Return(0));
    ON_CALL(executor_, SynchronizeAllActivity()).WillByDefault(Return(true));
    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
  }

  DeviceAddressVmmAllocator::DeviceConfig Config(uint64_t pa_budget) {
    return {&executor_, &stream_, pa_budget};
  }

  NiceMock<MockPlatform> platform_;
  NiceMock<MockStreamExecutor> executor_;
  NiceMock<MockStream> stream_;
};

TEST_F(DeviceAddressVmmAllocatorTest,
       DestructorSynchronizesExecutorWithoutPendingOperations) {
  EXPECT_CALL(executor_, SynchronizeAllActivity()).WillOnce(Return(true));
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {Config(UINT64_MAX)}));

  allocator.reset();
}

TEST_F(DeviceAddressVmmAllocatorTest,
       DestructorSynchronizesAllExecutorsBeforeReleasingDeviceResources) {
  bool device_0_synchronized = false;
  bool device_1_synchronized = false;
  EXPECT_CALL(executor_, SynchronizeAllActivity()).WillOnce([&] {
    device_0_synchronized = true;
    return true;
  });

  NiceMock<MockStreamExecutor> executor_1;
  NiceMock<MockStream> stream_1;
  ON_CALL(executor_1, device_ordinal()).WillByDefault(Return(1));
  ON_CALL(stream_1, parent()).WillByDefault(Return(&executor_1));
  EXPECT_CALL(executor_1, SynchronizeAllActivity()).WillOnce([&] {
    device_1_synchronized = true;
    return true;
  });

  auto reservation_0 = std::make_unique<TestMemoryReservation>(kGranularity);
  auto reservation_1 = std::make_unique<TestMemoryReservation>(kGranularity);
  int destroyed_devices = 0;
  auto on_device_destroy = [&](int ordinal) {
    EXPECT_TRUE(device_0_synchronized);
    EXPECT_TRUE(device_1_synchronized);
    if (ordinal == 0) {
      EXPECT_EQ(reservation_0->active_mapping_count(), 0);
    } else {
      EXPECT_EQ(ordinal, 1);
      EXPECT_EQ(reservation_1->active_mapping_count(), 0);
    }
    ++destroyed_devices;
  };
  const DeviceAddressVmmAllocator::DeviceConfig config_0 = Config(UINT64_MAX);
  const DeviceAddressVmmAllocator::DeviceConfig config_1 = {
      &executor_1, &stream_1, UINT64_MAX};
  ASSERT_OK_AND_ASSIGN(auto allocator,
                       TestDeviceAddressVmmAllocator::Create(
                           &platform_, {config_0, config_1},
                           /*physical_size_padding=*/0, on_device_destroy));

  ASSERT_OK_AND_ASSIGN(
      auto address_0,
      allocator->Allocate(
          /*device_ordinal=*/0, /*allocation_size=*/kGranularity,
          /*retry_on_failure=*/true, /*memory_space=*/0, reservation_0.get(),
          /*reservation_offset=*/0, /*mapping_size=*/kGranularity));
  ASSERT_OK_AND_ASSIGN(
      auto address_1,
      allocator->Allocate(
          /*device_ordinal=*/1, /*allocation_size=*/kGranularity,
          /*retry_on_failure=*/true, /*memory_space=*/0, reservation_1.get(),
          /*reservation_offset=*/0, /*mapping_size=*/kGranularity));
  EXPECT_EQ(reservation_0->active_mapping_count(), 1);
  EXPECT_EQ(reservation_1->active_mapping_count(), 1);
  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, address_0.Release()),
              absl_testing::IsOk());
  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/1, address_1.Release()),
              absl_testing::IsOk());

  allocator.reset();
  EXPECT_EQ(destroyed_devices, 2);
}

TEST_F(DeviceAddressVmmAllocatorTest, RetryFlagDoesNotDisablePendingReclaim) {
  const DeviceAddressVmmAllocator::DeviceConfig config =
      Config(2 * kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {config}));

  ASSERT_OK_AND_ASSIGN(
      auto first,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/true, /*memory_space=*/0));
  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, first.Release()),
              absl_testing::IsOk());

  ASSERT_OK_AND_ASSIGN(
      auto retried,
      allocator->Allocate(/*device_ordinal=*/0, 2 * kGranularity,
                          /*retry_on_failure=*/false, /*memory_space=*/0));
  EXPECT_EQ(allocator->allocation_count(), 2);
}

TEST_F(DeviceAddressVmmAllocatorTest,
       RetryDisabledStillReusesCompatiblePendingAllocation) {
  const DeviceAddressVmmAllocator::DeviceConfig config = Config(kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {config}));

  ASSERT_OK_AND_ASSIGN(
      auto first,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/true, /*memory_space=*/0));
  void* first_address = first->opaque();
  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, first.Release()),
              absl_testing::IsOk());

  ASSERT_OK_AND_ASSIGN(
      auto reused,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/false, /*memory_space=*/0));
  EXPECT_EQ(reused->opaque(), first_address);
  EXPECT_EQ(allocator->allocation_count(), 1);
}

TEST_F(DeviceAddressVmmAllocatorTest,
       RetryFlagDoesNotDisableMappedPendingReclaim) {
  auto reservation = std::make_unique<TestMemoryReservation>(2 * kGranularity);
  const DeviceAddressVmmAllocator::DeviceConfig config =
      Config(2 * kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {config}));

  ASSERT_OK_AND_ASSIGN(
      auto first,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/true, /*memory_space=*/0));
  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, first.Release()),
              absl_testing::IsOk());

  ASSERT_OK_AND_ASSIGN(
      auto retried,
      allocator->Allocate(
          /*device_ordinal=*/0, /*allocation_size=*/2 * kGranularity,
          /*retry_on_failure=*/false, /*memory_space=*/0, reservation.get(),
          /*reservation_offset=*/0, /*mapping_size=*/2 * kGranularity));
  EXPECT_EQ(reservation->active_mapping_count(), 1);
  EXPECT_EQ(allocator->allocation_count(), 2);
}

TEST_F(DeviceAddressVmmAllocatorTest,
       PhysicalAllocationSizeControlsBudgetAccounting) {
  const DeviceAddressVmmAllocator::DeviceConfig config =
      Config(2 * kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator,
                       TestDeviceAddressVmmAllocator::Create(
                           &platform_, {config},
                           /*physical_size_padding=*/kGranularity));

  ASSERT_OK_AND_ASSIGN(
      auto first,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/true, /*memory_space=*/0));
  ASSERT_NE(allocator->GetRawAllocation(/*device_ordinal=*/0, first.cref()),
            nullptr);
  EXPECT_EQ(allocator->GetRawAllocation(/*device_ordinal=*/0, first.cref())
                ->address()
                .size(),
            2 * kGranularity);
  // The first allocation consumes the full budget based on the physical size,
  // even though its requested size was one granularity unit.
  EXPECT_THAT(allocator->Allocate(/*device_ordinal=*/0, 2 * kGranularity,
                                  /*retry_on_failure=*/false,
                                  /*memory_space=*/0),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(allocator->allocation_count(), 1);

  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, first.Release()),
              absl_testing::IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(/*device_ordinal=*/0),
              absl_testing::IsOk());
  ASSERT_OK_AND_ASSIGN(
      auto second,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/false, /*memory_space=*/0));
  EXPECT_EQ(allocator->allocation_count(), 2);
}

TEST_F(DeviceAddressVmmAllocatorTest,
       BatchedUnmapAndDeallocateReclaimSelectedAllocation) {
  auto backing = std::make_unique<TestMemoryReservation>(kGranularity);
  auto alias = std::make_unique<TestMemoryReservation>(kGranularity);
  const DeviceAddressVmmAllocator::DeviceConfig config = Config(kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {config}));

  // The mapped overload returns the reservation slice as the allocator address,
  // so the record is kAllocateAndMap. A later plain Allocate() cannot satisfy
  // itself from such a record by reuse, which forces it through reclaim below.
  ASSERT_OK_AND_ASSIGN(
      auto mapped, allocator->Allocate(
                       /*device_ordinal=*/0, /*allocation_size=*/kGranularity,
                       /*retry_on_failure=*/false, /*memory_space=*/0,
                       backing.get(), /*reservation_offset=*/0,
                       /*mapping_size=*/kGranularity));
  ASSERT_THAT(allocator->Map(/*device_ordinal=*/0, mapped.cref(), alias.get(),
                             /*reservation_offset=*/0, kGranularity),
              absl_testing::IsOk());
  EXPECT_EQ(alias->active_mapping_count(), 1);

  // Queue the alias teardown and the allocation teardown back to back so both
  // land in the same open batch and share one sequence number.
  ASSERT_THAT(allocator->UnMap(/*device_ordinal=*/0, alias.get(),
                               /*reservation_offset=*/0, kGranularity),
              absl_testing::IsOk());
  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, mapped.Release()),
              absl_testing::IsOk());

  // Reclaim skips kMap entries, so it must select the allocation entry rather
  // than the map entry carrying the same sequence number, and must then
  // complete the paired stale mapping instead of leaving the alias mapped.
  ASSERT_OK_AND_ASSIGN(
      auto replacement,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/false, /*memory_space=*/0));
  EXPECT_EQ(alias->active_mapping_count(), 0);
  EXPECT_EQ(allocator->allocation_count(), 2);
}

}  // namespace
}  // namespace stream_executor
