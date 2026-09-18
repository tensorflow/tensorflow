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
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
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
  int mapping_count() const { return mapping_count_; }

 private:
  absl::Status Map(size_t reservation_offset, size_t allocation_offset,
                   size_t size, MemoryAllocation& allocation) override {
    if (reservation_offset > size_ || size > size_ - reservation_offset ||
        allocation_offset > allocation.address().size() ||
        size > allocation.address().size() - allocation_offset) {
      return absl::InvalidArgumentError("mapping range is out of bounds");
    }
    ++active_mapping_count_;
    ++mapping_count_;
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
  int mapping_count_ = 0;
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
  int timeline_write_count() const { return timeline_write_count_; }

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
    ++timeline_write_count_;
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
  int timeline_write_count_ = 0;
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

TEST_F(DeviceAddressVmmAllocatorTest,
       OverlapLookupPreservesRangeBoundariesForActiveAndStaleAddresses) {
  for (bool alias : {false, true}) {
    for (bool stale : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "alias=" << alias << " stale=" << stale);
      TestMemoryReservation reservation(16 * kGranularity);
      ASSERT_OK_AND_ASSIGN(auto allocator,
                           TestDeviceAddressVmmAllocator::Create(
                               &platform_, {Config(UINT64_MAX)}));
      ASSERT_OK_AND_ASSIGN(
          auto tracked,
          alias ? allocator->Allocate(/*device_ordinal=*/0, 4 * kGranularity)
                : allocator->Allocate(
                      /*device_ordinal=*/0, 4 * kGranularity,
                      /*retry_on_failure=*/false, /*memory_space=*/0,
                      &reservation, 4 * kGranularity, 4 * kGranularity));
      if (alias) {
        ASSERT_THAT(allocator->Map(0, tracked.cref(), &reservation,
                                   4 * kGranularity, 4 * kGranularity),
                    absl_testing::IsOk());
      }
      if (stale) {
        if (alias) {
          ASSERT_THAT(allocator->UnMap(0, &reservation, 4 * kGranularity,
                                       4 * kGranularity),
                      absl_testing::IsOk());
        } else {
          ASSERT_THAT(allocator->Deallocate(0, tracked.Release()),
                      absl_testing::IsOk());
        }
      }
      ASSERT_OK_AND_ASSIGN(auto source,
                           allocator->Allocate(0, 16 * kGranularity));
      struct Range {
        uint64_t offset;
        uint64_t size;
      };
      // Intersect the successor, lie inside the predecessor, intersect its
      // end, share its start with a different size, or contain the whole range.
      const Range partial_overlaps[] = {{2, 3}, {5, 1}, {7, 2},
                                        {4, 2}, {4, 5}, {2, 8}};
      for (Range range : partial_overlaps) {
        SCOPED_TRACE(testing::Message()
                     << "offset=" << range.offset << " size=" << range.size);
        EXPECT_THAT(allocator->Map(0, source.cref(), &reservation,
                                   range.offset * kGranularity,
                                   range.size * kGranularity),
                    StatusIs(absl::StatusCode::kFailedPrecondition));
        EXPECT_THAT(allocator->Allocate(
                        0, range.size * kGranularity, false, 0, &reservation,
                        range.offset * kGranularity, range.size * kGranularity),
                    StatusIs(absl::StatusCode::kFailedPrecondition));
        // Rejecting an overlap must not drain the stale mapping.
        EXPECT_EQ(reservation.active_mapping_count(), 1);
      }

      // Exactly adjacent ranges are disjoint on either side of the record.
      ASSERT_THAT(
          allocator->Map(0, source.cref(), &reservation, 0, 4 * kGranularity),
          absl_testing::IsOk());
      ASSERT_OK_AND_ASSIGN(
          auto right,
          allocator->Allocate(0, 4 * kGranularity, false, 0, &reservation,
                              8 * kGranularity, 4 * kGranularity));
      auto exact =
          allocator->Allocate(0, 4 * kGranularity, false, 0, &reservation,
                              4 * kGranularity, 4 * kGranularity);
      if (stale) {
        ASSERT_THAT(exact, absl_testing::IsOk());
      } else {
        EXPECT_THAT(exact, StatusIs(absl::StatusCode::kAlreadyExists));
      }
      ASSERT_THAT(allocator->UnMap(0, &reservation, 0, 4 * kGranularity),
                  absl_testing::IsOk());
      if (alias && !stale) {
        ASSERT_THAT(allocator->UnMap(0, &reservation, 4 * kGranularity,
                                     4 * kGranularity),
                    absl_testing::IsOk());
      }
    }
  }
}

TEST_F(DeviceAddressVmmAllocatorTest,
       ManyReservationAliasesReuseAndRemoveOrderedRecords) {
  constexpr int kCount = 512;
  TestMemoryReservation reservation((2 * kCount + 1) * kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {Config(UINT64_MAX)}));
  std::vector<ScopedDeviceAddress<uint8_t>> sources;
  for (int i = 0; i < kCount; ++i) {
    ASSERT_OK_AND_ASSIGN(auto source, allocator->Allocate(0, kGranularity));
    sources.push_back(std::move(source));
  }
  // Insert in an order that interleaves low and high addresses, leaving gaps
  // between aliases. Records must remain valid across index rebalancing.
  for (int n = 0; n < kCount; ++n) {
    int i = (n % 2 == 0) ? n / 2 : kCount - 1 - n / 2;
    ASSERT_THAT(allocator->Map(0, sources[i].cref(), &reservation,
                               (2 * i + 1) * kGranularity, kGranularity),
                absl_testing::IsOk());
  }
  for (int i = 0; i < kCount; ++i) {
    ASSERT_THAT(allocator->UnMap(0, &reservation, (2 * i + 1) * kGranularity,
                                 kGranularity),
                absl_testing::IsOk());
  }
  for (int i = kCount - 1; i >= 0; --i) {
    ASSERT_THAT(allocator->Map(0, sources[i].cref(), &reservation,
                               (2 * i + 1) * kGranularity, kGranularity),
                absl_testing::IsOk());
  }
  EXPECT_EQ(reservation.active_mapping_count(), kCount);
  EXPECT_EQ(reservation.mapping_count(), kCount);
  EXPECT_EQ(allocator->allocation_count(), kCount);

  // A query starting in a gap must still find the next occupied range.
  ASSERT_OK_AND_ASSIGN(auto source, allocator->Allocate(0, 4 * kGranularity));
  EXPECT_THAT(allocator->Map(0, source.cref(), &reservation,
                             kCount * kGranularity, 4 * kGranularity),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  for (int i = 0; i < kCount; ++i) {
    ASSERT_THAT(allocator->UnMap(0, &reservation, (2 * i + 1) * kGranularity,
                                 kGranularity),
                absl_testing::IsOk());
  }
  ASSERT_THAT(allocator->SynchronizePendingOperations(0), absl_testing::IsOk());
  EXPECT_EQ(reservation.active_mapping_count(), 0);
  // The removed alias ranges can now be covered by a larger mapping.
  ASSERT_THAT(allocator->Map(0, source.cref(), &reservation,
                             kCount * kGranularity, 4 * kGranularity),
              absl_testing::IsOk());
  ASSERT_THAT(allocator->UnMap(0, &reservation, kCount * kGranularity,
                               4 * kGranularity),
              absl_testing::IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       StaleAliasReuseValidatesSourceAndFullReservationRange) {
  TestMemoryReservation reservation(2 * kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {Config(UINT64_MAX)}));
  ASSERT_OK_AND_ASSIGN(auto source, allocator->Allocate(0, 2 * kGranularity));
  ASSERT_THAT(allocator->Map(0, source.cref(), &reservation, 0, kGranularity),
              absl_testing::IsOk());
  ASSERT_THAT(allocator->UnMap(0, &reservation, 0, kGranularity),
              absl_testing::IsOk());

  // Matching the start is insufficient: a larger range must still be rejected,
  // without retiring the stale alias or its pending operation.
  EXPECT_THAT(
      allocator->Map(0, source.cref(), &reservation, 0, 2 * kGranularity),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(reservation.active_mapping_count(), 1);
  EXPECT_EQ(allocator->timeline_write_count(), 0);

  ASSERT_THAT(allocator->Map(0, source.cref(), &reservation, 0, kGranularity),
              absl_testing::IsOk());
  EXPECT_THAT(allocator->Map(0, source.cref(), &reservation, 0, kGranularity),
              StatusIs(absl::StatusCode::kAlreadyExists));
  ASSERT_THAT(allocator->SynchronizePendingOperations(0), absl_testing::IsOk());
  EXPECT_EQ(reservation.mapping_count(), 1);
  EXPECT_EQ(reservation.active_mapping_count(), 1);
  EXPECT_EQ(allocator->timeline_write_count(), 0);

  ASSERT_THAT(allocator->UnMap(0, &reservation, 0, kGranularity),
              absl_testing::IsOk());
  DeviceAddressBase stale_source = source.Release();
  ASSERT_THAT(allocator->Deallocate(0, stale_source), absl_testing::IsOk());
  EXPECT_THAT(allocator->Map(0, stale_source, &reservation, 0, kGranularity),
              StatusIs(absl::StatusCode::kNotFound));
  ASSERT_THAT(allocator->SynchronizePendingOperations(0), absl_testing::IsOk());
  EXPECT_EQ(reservation.active_mapping_count(), 0);
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MixedPendingOperationsCanBeCancelledAndRequeuedInAnyOrder) {
  constexpr int kCount = 4;
  TestMemoryReservation reservation(2 * kCount * kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {Config(UINT64_MAX)}));
  std::vector<ScopedDeviceAddress<uint8_t>> temps;
  std::vector<ScopedDeviceAddress<uint8_t>> sources;
  for (int i = 0; i < kCount; ++i) {
    ASSERT_OK_AND_ASSIGN(
        auto temp, allocator->Allocate(0, kGranularity, false, 0, &reservation,
                                       i * kGranularity, kGranularity));
    temps.push_back(std::move(temp));
    ASSERT_OK_AND_ASSIGN(auto source, allocator->Allocate(0, kGranularity));
    ASSERT_THAT(allocator->Map(0, source.cref(), &reservation,
                               (kCount + i) * kGranularity, kGranularity),
                absl_testing::IsOk());
    sources.push_back(std::move(source));
  }

  for (int step = 0; step < 3; ++step) {
    SCOPED_TRACE(step);
    // Interleave allocator deallocations and alias unmaps in the same batch.
    for (int i = 0; i < kCount; ++i) {
      ASSERT_THAT(allocator->Deallocate(0, temps[i].Release()),
                  absl_testing::IsOk());
      ASSERT_THAT(allocator->UnMap(0, &reservation, (kCount + i) * kGranularity,
                                   kGranularity),
                  absl_testing::IsOk());
    }
    // Cancel from the middle, head, tail, and finally the remaining entries.
    // The following iteration requeues the same record-owned nodes.
    for (int i : {1, 0, 3, 2}) {
      ASSERT_OK_AND_ASSIGN(
          temps[i], allocator->Allocate(0, kGranularity, false, 0, &reservation,
                                        i * kGranularity, kGranularity));
      ASSERT_THAT(allocator->Map(0, sources[i].cref(), &reservation,
                                 (kCount + i) * kGranularity, kGranularity),
                  absl_testing::IsOk());
    }
    ASSERT_THAT(allocator->SynchronizePendingOperations(0),
                absl_testing::IsOk());
    EXPECT_EQ(allocator->timeline_write_count(), 0);
    EXPECT_EQ(allocator->allocation_count(), 2 * kCount);
    EXPECT_EQ(reservation.mapping_count(), 2 * kCount);
    EXPECT_EQ(reservation.active_mapping_count(), 2 * kCount);
  }

  // Leave one alias pending between cancelled entries. Draining must release
  // only that alias, then permit a fresh mapping at the same address.
  for (int i = 0; i < kCount; ++i) {
    ASSERT_THAT(allocator->UnMap(0, &reservation, (kCount + i) * kGranularity,
                                 kGranularity),
                absl_testing::IsOk());
  }
  for (int i : {3, 0, 2}) {
    ASSERT_THAT(allocator->Map(0, sources[i].cref(), &reservation,
                               (kCount + i) * kGranularity, kGranularity),
                absl_testing::IsOk());
  }
  ASSERT_THAT(allocator->SynchronizePendingOperations(0), absl_testing::IsOk());
  EXPECT_EQ(reservation.active_mapping_count(), 2 * kCount - 1);
  EXPECT_EQ(allocator->timeline_write_count(), 1);
  ASSERT_THAT(allocator->Map(0, sources[1].cref(), &reservation,
                             (kCount + 1) * kGranularity, kGranularity),
              absl_testing::IsOk());

  for (int i = 0; i < kCount; ++i) {
    ASSERT_THAT(allocator->UnMap(0, &reservation, (kCount + i) * kGranularity,
                                 kGranularity),
                absl_testing::IsOk());
  }
  temps.clear();
  ASSERT_THAT(allocator->SynchronizePendingOperations(0), absl_testing::IsOk());
  EXPECT_EQ(reservation.active_mapping_count(), 0);
  EXPECT_EQ(allocator->timeline_write_count(), 2);
}

}  // namespace
}  // namespace stream_executor
