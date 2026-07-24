/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_device_address_vmm_allocator.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/service/computation_placer.h"
#include "xla/stream_executor/cuda/cuda_memory_reservation.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_space.h"
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
    auto platform_or = PlatformManager::PlatformWithName("CUDA");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "CUDA platform not available";
    }
    platform_ = platform_or.value();

    auto executor_or = platform_->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "CUDA executor not available";
    }
    executor_ = executor_or.value();

    auto stream_or = executor_->CreateStream();
    if (!stream_or.ok()) {
      GTEST_SKIP() << "Failed to create stream";
    }
    stream_ = std::move(stream_or.value());

    // Probe for cuStreamWriteValue64 support (requires CC >= 7.0).
    auto probe =
        gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get());
    if (absl::IsUnimplemented(probe.status())) {
      GTEST_SKIP() << "Device does not support cuStreamWriteValue64 "
                      "(requires compute capability >= 7.0): "
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
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate memory.
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

  // The ScopedDeviceAddress will automatically deallocate when it goes out of
  // scope.
}

TEST_F(DeviceAddressVmmAllocatorTest, AllocateZeroSize) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate zero-size memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 0,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  // Zero-size allocation should return a null address.
  EXPECT_TRUE(scoped_address.is_null());
}

TEST_F(DeviceAddressVmmAllocatorTest, AllocateMultiple) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate multiple memory regions.
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

  // Both allocations should be valid and distinct.
  EXPECT_FALSE(addr1.is_null());
  EXPECT_FALSE(addr2.is_null());
  EXPECT_NE(addr1->opaque(), addr2->opaque());
  EXPECT_EQ(addr1.cref().size(), 1024);
  EXPECT_EQ(addr2.cref().size(), 2048);
}

TEST_F(DeviceAddressVmmAllocatorTest, MemoryReadWrite) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_NE(scoped_address->opaque(), nullptr);

  // Create a stream for memory operations.
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());

  // Write data to the allocated memory.
  constexpr uint64_t kTestValue = 0xDEADBEEFCAFEBABE;
  DeviceAddressBase addr = scoped_address.cref();
  EXPECT_THAT(stream->Memcpy(&addr, &kTestValue, sizeof(kTestValue)),
              absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  // Read data back.
  uint64_t read_value = 0;
  EXPECT_THAT(stream->Memcpy(&read_value, addr, sizeof(read_value)),
              absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  EXPECT_EQ(read_value, kTestValue);
}

TEST_F(DeviceAddressVmmAllocatorTest, GetStream) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Get the stream - should return the same stream that was provided at
  // construction.
  TF_ASSERT_OK_AND_ASSIGN(Stream * stream,
                          allocator->GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream_.get());

  // Getting the stream again should return the same pointer.
  TF_ASSERT_OK_AND_ASSIGN(Stream * stream2,
                          allocator->GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream2);
}

TEST_F(DeviceAddressVmmAllocatorTest, GetStreamExecutor) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      StreamExecutor * se,
      allocator->GetStreamExecutor(executor_->device_ordinal()));
  EXPECT_EQ(se, executor_);
}

TEST_F(DeviceAddressVmmAllocatorTest, AllowsAsynchronousDeallocation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Virtual address allocator supports asynchronous deallocation via
  // GPU timeline-based processing.
  EXPECT_TRUE(allocator->AllowsAsynchronousDeallocation());
}

TEST_F(DeviceAddressVmmAllocatorTest, ExplicitDeallocate) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_NE(scoped_address->opaque(), nullptr);
  DeviceAddressBase addr = scoped_address.cref();

  // Explicitly deallocate.
  EXPECT_THAT(allocator->Deallocate(executor_->device_ordinal(), addr),
              absl_testing::IsOk());

  // Release ownership to prevent double-free.
  scoped_address.Release();
}

TEST_F(DeviceAddressVmmAllocatorTest, SynchronizePendingOperationsDrainsQueue) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(ordinal, 1024, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  DeviceAddressBase addr = scoped_address.cref();
  scoped_address.Release();
  ASSERT_THAT(allocator->Deallocate(ordinal, addr), IsOk());
  EXPECT_NE(allocator->GetRawAllocation(ordinal, addr), nullptr);

  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, addr), nullptr);
}

TEST_F(DeviceAddressVmmAllocatorTest, MapAndUnMapReservationAlias) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_THAT(allocator->Map(ordinal, source.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_THAT(allocator->Deallocate(ordinal, source.cref()),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MapRejectsPartialOverlapWithActiveReservationMapping) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);
  const uint64_t mapping_size = 2 * granularity;

  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, mapping_size));
  ASSERT_OK_AND_ASSIGN(
      auto full_range_source,
      allocator->Allocate(ordinal, mapping_size, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_OK_AND_ASSIGN(
      auto partial_range_source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_THAT(
      allocator->Map(ordinal, full_range_source.cref(), reservation.get(),
                     /*reservation_offset=*/0, mapping_size),
      IsOk());

  EXPECT_THAT(
      allocator->Map(ordinal, partial_range_source.cref(), reservation.get(),
                     /*reservation_offset=*/granularity, granularity),
      absl_testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::HasSubstr("partially overlaps active reservation")));

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, mapping_size),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, full_range_source.Release()),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, partial_range_source.Release()),
              IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MapReactivatesPendingUnMapForSameRawAndRange) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  auto* raw_allocation = allocator->GetRawAllocation(ordinal, source.cref());
  ASSERT_NE(raw_allocation, nullptr);
  DeviceAddressBase reservation_address = reservation->address().GetByteSlice(
      /*offset_bytes=*/0, granularity);

  ASSERT_THAT(allocator->Map(ordinal, source.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, reservation_address), nullptr);

  ASSERT_THAT(allocator->Map(ordinal, source.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, reservation_address),
            raw_allocation);
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, reservation_address),
            raw_allocation);

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, source.Release()), IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MapWaitsForPendingUnMapBeforeMappingSameSourceToDifferentReservation) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(auto old_reservation, gpu::CudaMemoryReservation::Create(
                                                 executor_, granularity));
  ASSERT_OK_AND_ASSIGN(auto new_reservation, gpu::CudaMemoryReservation::Create(
                                                 executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  auto* raw_allocation = allocator->GetRawAllocation(ordinal, source.cref());
  ASSERT_NE(raw_allocation, nullptr);
  DeviceAddressBase old_reservation_address =
      old_reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);
  DeviceAddressBase new_reservation_address =
      new_reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);
  ASSERT_NE(old_reservation_address.opaque(), new_reservation_address.opaque());

  ASSERT_THAT(allocator->Map(ordinal, source.cref(), old_reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  constexpr uint64_t kPattern = 0x123456789ABCDEF0ULL;
  ASSERT_THAT(
      stream_->Memcpy(&old_reservation_address, &kPattern, sizeof(kPattern)),
      IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, old_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, old_reservation_address),
            nullptr);

  ASSERT_THAT(allocator->Map(ordinal, source.cref(), new_reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, new_reservation_address),
            raw_allocation);
  uint64_t read_value = 0;
  ASSERT_THAT(
      stream_->Memcpy(&read_value, new_reservation_address, sizeof(read_value)),
      IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
  EXPECT_EQ(read_value, kPattern);

  ASSERT_THAT(allocator->UnMap(ordinal, new_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, source.Release()), IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MapWaitsForPendingDestinationAndSourceAliases) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(
      auto source_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto destination_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_OK_AND_ASSIGN(
      auto destination_occupant,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  auto* source_raw_allocation =
      allocator->GetRawAllocation(ordinal, source.cref());
  ASSERT_NE(source_raw_allocation, nullptr);
  DeviceAddressBase source_reservation_address =
      source_reservation->address().GetByteSlice(/*offset_bytes=*/0,
                                                 granularity);
  DeviceAddressBase destination_reservation_address =
      destination_reservation->address().GetByteSlice(/*offset_bytes=*/0,
                                                      granularity);

  ASSERT_THAT(allocator->Map(ordinal, source.cref(), source_reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  constexpr uint64_t kPattern = 0x123456789ABCDEF0ULL;
  ASSERT_THAT(
      stream_->Memcpy(&source_reservation_address, &kPattern, sizeof(kPattern)),
      IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, source_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());

  ASSERT_THAT(allocator->Map(ordinal, destination_occupant.cref(),
                             destination_reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, destination_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());

  ASSERT_THAT(
      allocator->Map(ordinal, source.cref(), destination_reservation.get(),
                     /*reservation_offset=*/0, granularity),
      IsOk());
  EXPECT_EQ(
      allocator->GetRawAllocation(ordinal, destination_reservation_address),
      source_raw_allocation);
  uint64_t read_value = 0;
  ASSERT_THAT(stream_->Memcpy(&read_value, destination_reservation_address,
                              sizeof(read_value)),
              IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
  EXPECT_EQ(read_value, kPattern);

  ASSERT_THAT(allocator->UnMap(ordinal, destination_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, source.Release()), IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, destination_occupant.Release()),
              IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MapRejectsActiveDestinationWithoutDrainingStaleSourceAlias) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(
      auto source_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto destination_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_OK_AND_ASSIGN(
      auto destination_occupant,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_THAT(allocator->Map(ordinal, source.cref(), source_reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, source_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Map(ordinal, destination_occupant.cref(),
                             destination_reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());

  EXPECT_THAT(
      allocator->Map(ordinal, source.cref(), destination_reservation.get(),
                     /*reservation_offset=*/0, granularity),
      absl_testing::StatusIs(
          absl::StatusCode::kAlreadyExists,
          ::testing::HasSubstr("reservation range is already tracked")));
  EXPECT_THAT(
      allocator->UnMap(ordinal, source_reservation.get(),
                       /*reservation_offset=*/0, granularity),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             ::testing::HasSubstr("already pending UnMap()")));

  ASSERT_THAT(allocator->UnMap(ordinal, destination_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, source.Release()), IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, destination_occupant.Release()),
              IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MapRejectsPartialOverlapWithStaleReservationMapping) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);
  const uint64_t mapping_size = 2 * granularity;

  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, mapping_size));
  ASSERT_OK_AND_ASSIGN(
      auto full_range_source,
      allocator->Allocate(ordinal, mapping_size, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_OK_AND_ASSIGN(
      auto partial_range_source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_THAT(
      allocator->Map(ordinal, full_range_source.cref(), reservation.get(),
                     /*reservation_offset=*/0, mapping_size),
      IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, mapping_size),
              IsOk());

  EXPECT_THAT(
      allocator->Map(ordinal, partial_range_source.cref(), reservation.get(),
                     /*reservation_offset=*/granularity, granularity),
      absl_testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::HasSubstr("partially overlaps stale reservation")));

  ASSERT_THAT(allocator->Deallocate(ordinal, full_range_source.Release()),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, partial_range_source.Release()),
              IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MapRemapsPendingUnMapForDifferentRawAllocation) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto old_source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_OK_AND_ASSIGN(
      auto new_source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  auto* old_raw_allocation =
      allocator->GetRawAllocation(ordinal, old_source.cref());
  auto* new_raw_allocation =
      allocator->GetRawAllocation(ordinal, new_source.cref());
  ASSERT_NE(old_raw_allocation, nullptr);
  ASSERT_NE(new_raw_allocation, nullptr);
  ASSERT_NE(old_raw_allocation, new_raw_allocation);
  DeviceAddressBase reservation_address = reservation->address().GetByteSlice(
      /*offset_bytes=*/0, granularity);

  ASSERT_THAT(allocator->Map(ordinal, old_source.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  constexpr uint64_t kPattern = 0x123456789ABCDEF0ULL;
  DeviceAddressBase device_address = reservation_address;
  ASSERT_THAT(stream_->Memcpy(&device_address, &kPattern, sizeof(kPattern)),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, reservation_address), nullptr);

  ASSERT_THAT(allocator->Map(ordinal, new_source.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, reservation_address),
            new_raw_allocation);

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, old_source.Release()), IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, new_source.Release()), IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       MapConflictPreservesUnrelatedPendingUnMap) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(
      auto unrelated_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto conflict_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto unrelated_source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_OK_AND_ASSIGN(
      auto old_source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_OK_AND_ASSIGN(
      auto new_source,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_THAT(allocator->Map(ordinal, unrelated_source.cref(),
                             unrelated_reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, unrelated_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());

  ASSERT_THAT(
      allocator->Map(ordinal, old_source.cref(), conflict_reservation.get(),
                     /*reservation_offset=*/0, granularity),
      IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, conflict_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());

  ASSERT_THAT(
      allocator->Map(ordinal, new_source.cref(), conflict_reservation.get(),
                     /*reservation_offset=*/0, granularity),
      IsOk());
  EXPECT_THAT(
      allocator->UnMap(ordinal, unrelated_reservation.get(),
                       /*reservation_offset=*/0, granularity),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             ::testing::HasSubstr("already pending UnMap()")));

  ASSERT_THAT(allocator->UnMap(ordinal, conflict_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, unrelated_source.Release()),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, old_source.Release()), IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, new_source.Release()), IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       AllocateIntoReservationReturnsReservationAddress) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto address,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective),
                          reservation.get(), /*reservation_offset=*/0,
                          granularity));

  EXPECT_EQ(address->opaque(), reservation->address().opaque());
  EXPECT_NE(allocator->GetRawAllocation(ordinal, address.cref()), nullptr);

  ASSERT_THAT(allocator->Deallocate(ordinal, address.Release()), IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest,
       AllocateIntoReservationReusesPendingReservationAddress) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  DeviceAddressBase reservation_address = reservation->address().GetByteSlice(
      /*offset_bytes=*/0, granularity);
  ASSERT_OK_AND_ASSIGN(
      auto address,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective),
                          reservation.get(), /*reservation_offset=*/0,
                          granularity));
  auto* raw_allocation = allocator->GetRawAllocation(ordinal, address.cref());
  ASSERT_NE(raw_allocation, nullptr);

  ASSERT_THAT(allocator->Deallocate(ordinal, address.Release()), IsOk());

  ASSERT_OK_AND_ASSIGN(
      auto reused,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective),
                          reservation.get(), /*reservation_offset=*/0,
                          granularity));
  EXPECT_TRUE(reused.cref().IsSameAs(reservation_address));
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, reused.cref()),
            raw_allocation);
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, reused.cref()),
            raw_allocation);

  ASSERT_THAT(allocator->Deallocate(ordinal, reused.Release()), IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

// Simulates back-to-back executable runs over a shared reservation layout:
// each step allocates the same reservation slices, runs stream work through
// the reservation addresses, and defers deallocation at teardown. Later steps
// allocate again immediately, with no synchronization in between and with the
// previous step's work potentially still in flight. Every later step must
// reactivate the previous step's stale mappings — same reservation VA backed
// by the same physical allocation — and the stream-ordered data flow must
// stay correct.
TEST_F(DeviceAddressVmmAllocatorTest,
       BackToBackMappedAllocationsReuseMappingsAcrossSteps) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  // Two slices at fixed offsets emulate a module's deterministic reservation
  // layout for two temp allocations.
  constexpr int kSlices = 2;
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, kSlices * granularity));

  constexpr int kSteps = 4;
  uint64_t last_pattern = 0;
  std::array<MemoryAllocation*, kSlices> first_step_raw = {nullptr, nullptr};
  std::vector<ScopedDeviceAddress<uint8_t>> addresses;
  for (int step = 0; step < kSteps; ++step) {
    addresses.clear();
    for (int slice = 0; slice < kSlices; ++slice) {
      ASSERT_OK_AND_ASSIGN(
          auto address,
          allocator->Allocate(
              ordinal, granularity, /*retry_on_failure=*/true,
              static_cast<int64_t>(MemorySpace::kCollective), reservation.get(),
              /*reservation_offset=*/slice * granularity, granularity));
      EXPECT_TRUE(address.cref().IsSameAs(reservation->address().GetByteSlice(
          slice * granularity, granularity)));
      auto* raw_allocation =
          allocator->GetRawAllocation(ordinal, address.cref());
      ASSERT_NE(raw_allocation, nullptr);
      if (step == 0) {
        first_step_raw[slice] = raw_allocation;
      } else {
        // Reuse must reactivate the previous step's mapping instead of
        // creating a new physical allocation.
        EXPECT_EQ(raw_allocation, first_step_raw[slice])
            << "step=" << step << " slice=" << slice;
      }
      addresses.push_back(std::move(address));
    }

    // Stream work through the reservation addresses: write a per-step pattern
    // into slice 0 and copy it into slice 1.
    last_pattern = 0xC0FFEE00ULL + step;
    DeviceAddressBase slice0 = addresses[0].cref();
    DeviceAddressBase slice1 = addresses[1].cref();
    ASSERT_THAT(stream_->Memcpy(&slice0, &last_pattern, sizeof(last_pattern)),
                IsOk());
    ASSERT_THAT(stream_->MemcpyD2D(&slice1, slice0, sizeof(last_pattern)),
                IsOk());

    if (step < kSteps - 1) {
      // Teardown: defer deallocation while the copies may still be in flight.
      for (auto& address : addresses) {
        ASSERT_THAT(allocator->Deallocate(ordinal, address.Release()), IsOk());
      }
    }
  }

  // One sync at the very end: all steps' copies executed in stream order
  // through mappings that were never torn down.
  uint64_t read_value = 0;
  ASSERT_THAT(
      stream_->Memcpy(&read_value, addresses[1].cref(), sizeof(read_value)),
      IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
  EXPECT_EQ(read_value, last_pattern);

  for (auto& address : addresses) {
    ASSERT_THAT(allocator->Deallocate(ordinal, address.Release()), IsOk());
  }
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

// A request with a different size at the same reservation offset does not
// match the stale mapping: it must be rejected as a partial overlap instead
// of remapping underneath the pending teardown, and the stale mapping must
// stay reusable for a later matching request.
TEST_F(DeviceAddressVmmAllocatorTest,
       AllocateIntoReservationRejectsSizeChangeAtSameAddress) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  const uint64_t granularity = allocator->GetAllocationGranularity(executor_);
  ASSERT_GT(granularity, 0);

  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, 2 * granularity));
  ASSERT_OK_AND_ASSIGN(
      auto address,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective),
                          reservation.get(), /*reservation_offset=*/0,
                          granularity));
  auto* raw_allocation = allocator->GetRawAllocation(ordinal, address.cref());
  ASSERT_NE(raw_allocation, nullptr);
  ASSERT_THAT(allocator->Deallocate(ordinal, address.Release()), IsOk());

  // A larger mapping at the same offset partially overlaps the stale
  // granularity-sized mapping and must be rejected.
  EXPECT_FALSE(allocator
                   ->Allocate(ordinal, 2 * granularity,
                              /*retry_on_failure=*/true,
                              static_cast<int64_t>(MemorySpace::kCollective),
                              reservation.get(), /*reservation_offset=*/0,
                              2 * granularity)
                   .ok());

  // The failed attempt must leave the stale mapping intact: a matching-size
  // request still reuses it.
  ASSERT_OK_AND_ASSIGN(
      auto reused,
      allocator->Allocate(ordinal, granularity, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective),
                          reservation.get(), /*reservation_offset=*/0,
                          granularity));
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, reused.cref()),
            raw_allocation);

  ASSERT_THAT(allocator->Deallocate(ordinal, reused.Release()), IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(ordinal), IsOk());
}

TEST_F(DeviceAddressVmmAllocatorTest, DeallocateNull) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Deallocating null address should succeed.
  DeviceAddressBase null_addr;
  EXPECT_THAT(allocator->Deallocate(executor_->device_ordinal(), null_addr),
              absl_testing::IsOk());
}

// --- Timeline / sequence-number design tests ---
//
// These tests exercise the cuStreamWriteValue64-based deferred deallocation
// mechanism. Each pending Deallocate() call records an increasing seqno and
// enqueues a GPU timeline write; the CPU checks the pinned counter to decide
// when memory is safe to free.

// Verifies that TryReusePendingDeallocation returns the same virtual address
// when a new allocation of the same rounded size is requested immediately
// after a Deallocate. The reuse is safe because stream ordering guarantees
// all prior GPU work finishes before any new work submitted after Allocate.
TEST_F(DeviceAddressVmmAllocatorTest,
       PendingDeallocationReusesSameVirtualAddress) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  constexpr uint64_t kSize = 1024;

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr1,
      allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  void* const va = addr1->opaque();

  // Deallocate: timeline write is enqueued but VA is not freed yet.
  DeviceAddressBase raw = addr1.cref();
  addr1.Release();
  ASSERT_THAT(allocator->Deallocate(ordinal, raw), IsOk());

  // Allocate the same size — TryReusePendingDeallocation should match the
  // pending entry and return the identical virtual address.
  TF_ASSERT_OK_AND_ASSIGN(
      auto addr2,
      allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  EXPECT_EQ(addr2->opaque(), va);

  // Sync to drain all pending GPU timeline writes before the allocator
  // is destroyed.
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}

// Verifies that deallocating memory while the GPU is still writing to it is
// safe. The timeline write for the deallocation is enqueued on the stream
// AFTER the memcpy, so the physical memory is not freed until the GPU finishes.
TEST_F(DeviceAddressVmmAllocatorTest,
       DeferredDeallocationSafeWhileGpuWritesData) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, sizeof(uint64_t), /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  // Enqueue a memcpy to the allocated buffer on stream_.
  constexpr uint64_t kPattern = 0xCAFEBABEDEADBEEFULL;
  DeviceAddressBase dev_addr = addr.cref();
  ASSERT_THAT(stream_->Memcpy(&dev_addr, &kPattern, sizeof(kPattern)), IsOk());

  // Deallocate while the memcpy is still queued. The seqno timeline write is
  // appended to the stream AFTER the memcpy, so the VA cannot be reused until
  // the GPU advances past it.
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());

  // Sync: both the memcpy and the timeline write execute in order.
  // No crash here means the physical memory was not freed prematurely.
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}

// Allocates and deallocates N buffers, recording a distinct seqno for each.
// After a single stream sync all seqnos have been written by the GPU, so
// re-allocating the same size should succeed by reusing the pending entries.
TEST_F(DeviceAddressVmmAllocatorTest,
       MultipleSeqnosAllCompleteAfterStreamSync) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  constexpr int kCount = 8;
  constexpr uint64_t kSize = 1024;

  // Allocate kCount buffers and immediately queue their deallocation.
  // Each Deallocate increments next_seqno and enqueues a timeline write.
  for (int i = 0; i < kCount; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  }

  // Sync the stream: all kCount timeline writes (seqnos 1..kCount) complete.
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());

  // Each new Allocate call finds a matching pending entry via
  // TryReusePendingDeallocation (or via ProcessCompletedPendingDeallocations
  // once the pending queue is exhausted).
  for (int i = 0; i < kCount; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    EXPECT_FALSE(addr.is_null());
  }

  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}

// Verifies that the destructor correctly spin-waits on the pinned timeline
// counter until all pending GPU timeline writes complete, then frees the
// physical memory without crashing.
TEST_F(DeviceAddressVmmAllocatorTest,
       DestructorWithPendingDeallocationsDoesNotCrash) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();

  // Queue several deallocations without syncing the stream first.
  for (int i = 0; i < 4; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, 1024, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  }

  // Destroy without an explicit stream sync. The destructor must spin on the
  // pinned_timeline until the GPU writes all pending seqnos, then free
  // each virtual address safely.
  allocator.reset();  // Must not crash or leak.
}

TEST_F(DeviceAddressVmmAllocatorTest, UnknownDeviceOrdinalReturnsError) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int unknown_ordinal = 9999;
  EXPECT_FALSE(allocator
                   ->Allocate(unknown_ordinal, 1024, /*retry_on_failure=*/true,
                              static_cast<int64_t>(MemorySpace::kCollective))
                   .ok());
  // Null Deallocate always succeeds (early return before ordinal lookup).
  EXPECT_THAT(allocator->Deallocate(unknown_ordinal, DeviceAddressBase{}),
              absl_testing::IsOk());
  // Non-null address on unknown ordinal returns error.
  DeviceAddressBase fake_addr(reinterpret_cast<void*>(0x1000), 64);
  EXPECT_FALSE(allocator->Deallocate(unknown_ordinal, fake_addr).ok());
  EXPECT_FALSE(allocator->GetStream(unknown_ordinal).ok());
  EXPECT_FALSE(allocator->GetStreamExecutor(unknown_ordinal).ok());
}

// Multi-device fixture: skips the test if fewer than two CUDA devices are
// available.
class MultiDeviceVmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("CUDA");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "CUDA platform not available";
    }
    platform_ = platform_or.value();

    if (platform_->VisibleDeviceCount() < 2) {
      GTEST_SKIP() << "Fewer than two CUDA devices available";
    }

    for (int i = 0; i < 2; ++i) {
      auto executor_or = platform_->ExecutorForDevice(i);
      if (!executor_or.ok()) {
        GTEST_SKIP() << "CUDA executor not available for device " << i;
      }
      executors_.push_back(executor_or.value());

      auto stream_or = executors_.back()->CreateStream();
      if (!stream_or.ok()) {
        GTEST_SKIP() << "Failed to create stream for device " << i;
      }
      streams_.push_back(std::move(stream_or.value()));
    }

    // Probe for cuStreamWriteValue64 support using device 0 (requires CC
    // >= 7.0).
    auto probe = gpu::CudaDeviceAddressVmmAllocator::Create(executors_[0],
                                                            streams_[0].get());
    if (absl::IsUnimplemented(probe.status())) {
      GTEST_SKIP() << "Device does not support cuStreamWriteValue64 "
                      "(requires compute capability >= 7.0): "
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
      gpu::CudaDeviceAddressVmmAllocator::Create(platform_, configs));

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
      gpu::CudaDeviceAddressVmmAllocator::Create(platform_, configs));

  // Allocate on device 0.
  TF_ASSERT_OK_AND_ASSIGN(
      auto addr0,
      allocator->Allocate(executors_[0]->device_ordinal(), 4096,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  EXPECT_FALSE(addr0.is_null());

  // GetRawAllocation for addr0's pointer on device 1 should return nullptr
  // (different per-device map).
  EXPECT_EQ(
      allocator->GetRawAllocation(executors_[1]->device_ordinal(), *addr0),
      nullptr);
}

TEST_F(DeviceAddressVmmAllocatorTest, MultiDeviceTagIsolatesReuse) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));
  const int ordinal = executor_->device_ordinal();
  constexpr uint64_t kSize = 1024;

  xla::DeviceAssignment multi_device_assignment(/*replica_count=*/2,
                                                /*computation_count=*/1);
  multi_device_assignment(0, 0) = 0;
  multi_device_assignment(1, 0) = 1;

  void* multi_device_ptr = nullptr;
  {
    DeviceAddressVmmAllocator::DeviceAssignmentScope scope(
        &multi_device_assignment);
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr1,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            /*memory_space=*/0));
    multi_device_ptr = addr1->opaque();
    ASSERT_THAT(allocator->Deallocate(ordinal, addr1.Release()), IsOk());

    TF_ASSERT_OK_AND_ASSIGN(
        auto addr2,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            /*memory_space=*/0));
    EXPECT_EQ(addr2->opaque(), multi_device_ptr);
    ASSERT_THAT(allocator->Deallocate(ordinal, addr2.Release()), IsOk());
  }

  // Outside scope: single-device alloc must not reuse multi-device entry.
  TF_ASSERT_OK_AND_ASSIGN(
      auto addr3, allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                                      /*memory_space=*/0));
  EXPECT_NE(addr3->opaque(), multi_device_ptr);
}

}  // namespace
}  // namespace stream_executor
