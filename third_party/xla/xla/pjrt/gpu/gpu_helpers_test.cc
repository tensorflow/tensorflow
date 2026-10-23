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

#include "xla/pjrt/gpu/gpu_helpers.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/gpu/allocator_config.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::NiceMock;
using ::testing::UnorderedElementsAre;

constexpr int64_t kMiB = 1 << 20;

// Sizing tests construct allocators without requesting physical memory.
class SizingReservation : public se::MemoryReservation {
 public:
  SizingReservation(size_t size, size_t granularity)
      : size_((size + granularity - 1) / granularity * granularity),
        granularity_(granularity) {}

  se::DeviceAddressBase address() const override {
    return se::DeviceAddressBase(reinterpret_cast<void*>(0x10000000), size_);
  }
  size_t granularity() const override { return granularity_; }

 private:
  absl::Status Map(size_t, size_t, size_t, se::MemoryAllocation&) override {
    return absl::UnimplementedError("No physical allocations in sizing tests");
  }
  absl::Status SetAccess(uint64_t, size_t) override {
    return absl::UnimplementedError("No physical allocations in sizing tests");
  }
  absl::Status UnMap(size_t, size_t) override {
    return absl::UnimplementedError("No physical allocations in sizing tests");
  }

  size_t size_;
  size_t granularity_;
};

class SizingExecutor : public NiceMock<se::MockStreamExecutor> {
 public:
  SizingExecutor() {
    ON_CALL(*this, DeviceMemoryUsage(_, _))
        .WillByDefault([this](int64_t* free, int64_t* total) {
          *free = *total = total_memory_;
          return true;
        });
  }

  absl::StatusOr<std::unique_ptr<se::MemoryReservation>>
  CreateMemoryReservation(uint64_t size) override {
    ++reservations_;
    return std::make_unique<SizingReservation>(size, granularity_);
  }

  int64_t total_memory_ = 64 * kMiB;
  size_t granularity_ = 4 * kMiB;
  int reservations_ = 0;
};

class BfcMemorySizingTest : public ::testing::Test {
 protected:
  absl::StatusOr<std::shared_ptr<tsl::BFCAllocator>> Create(
      const MemFraction& fraction,
      std::optional<int64_t> initial_bytes = std::nullopt) {
    return CreateBFCAllocator(&executor_, fraction, /*preallocate=*/true,
                              initial_bytes,
                              /*sub_allocator_alloc_visitors=*/{},
                              /*sub_allocator_free_visitors=*/{},
                              /*enable_spatial_partitioning=*/true);
  }

  SizingExecutor executor_;
};

TEST_F(BfcMemorySizingTest, RejectsNarrowFractionalRangeAfterRounding) {
  executor_.total_memory_ = int64_t{80} << 30;
  EXPECT_THAT(
      Create(FlexMemFraction{0.750001, 0.750002}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("growth cap")));
}

TEST_F(BfcMemorySizingTest, RejectsInitialSizeAboveAlignedCap) {
  // Initial 9 MiB rounds to 12 MiB; cap 11 MiB rounds down to 8 MiB.
  for (bool explicit_initial : {false, true}) {
    SCOPED_TRACE(explicit_initial);
    EXPECT_THAT(
        Create(
            FlexMemFraction{9.0 / 64, 11.0 / 64},
            explicit_initial ? std::optional<int64_t>(9 * kMiB) : std::nullopt),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("growth cap")));
  }
}

TEST_F(BfcMemorySizingTest, RoundsPhysicalCapDownDespiteVaPadding) {
  // Reserving 11 MiB requires 12 MiB of VA, but only 8 MiB may be mapped.
  ASSERT_OK_AND_ASSIGN(auto allocator,
                       Create(FlexMemFraction{8.0 / 64, 11.0 / 64}));
  EXPECT_EQ(executor_.reservations_, 1);
  ASSERT_TRUE(allocator->GetStats().has_value());
  EXPECT_EQ(allocator->GetStats()->bytes_limit, 8 * kMiB);
}

TEST_F(BfcMemorySizingTest, AcceptsExactlyAlignedInitialSizeAndCap) {
  ASSERT_OK_AND_ASSIGN(auto allocator,
                       Create(FlexMemFraction{8.0 / 64, 12.0 / 64}));
  EXPECT_EQ(allocator->GetStats()->bytes_limit, 12 * kMiB);
}

TEST_F(BfcMemorySizingTest, RejectsInvalidEqualEndpoints) {
  for (double fraction : {0.0, -0.5, std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(fraction);
    EXPECT_THAT(Create(FlexMemFraction{fraction, fraction}),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST_F(BfcMemorySizingTest, EqualEndpointsMatchFixedPolicy) {
  // 75% of 66 MiB is not aligned. It must follow fixed-policy rounding,
  // rather than trying to grow with incompatible initial/cap rounding.
  executor_.total_memory_ = 66 * kMiB;
  ASSERT_OK_AND_ASSIGN(auto fixed, Create(FixedMemFraction{0.75}));
  ASSERT_OK_AND_ASSIGN(auto equal, Create(FlexMemFraction{0.75, 0.75}));
  EXPECT_EQ(equal->GetStats()->bytes_limit, fixed->GetStats()->bytes_limit);
  EXPECT_EQ(executor_.reservations_, 0);
}

constexpr size_t kPoolLimit = 24 * kMiB;

class GpuBFCAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* value = std::getenv("TF_ENABLE_GPU_GARBAGE_COLLECTION");
    if (value != nullptr) {
      original_garbage_collection_ = value;
    }
    ASSERT_EQ(tsl::unsetenv("TF_ENABLE_GPU_GARBAGE_COLLECTION"), 0);
  }

  void TearDown() override {
    if (original_garbage_collection_.has_value()) {
      EXPECT_EQ(tsl::setenv("TF_ENABLE_GPU_GARBAGE_COLLECTION",
                            original_garbage_collection_->c_str(), 1),
                0);
    } else {
      EXPECT_EQ(tsl::unsetenv("TF_ENABLE_GPU_GARBAGE_COLLECTION"), 0);
    }
  }

  GpuBFCAllocatorTest() {
    ON_CALL(executor_, DeviceMemoryUsage(_, _))
        .WillByDefault([](int64_t* free, int64_t* total) {
          *free = kPoolLimit;
          *total = kPoolLimit;
          return true;
        });
    // Fake, non-dereferenceable device addresses. Separate regions cannot be
    // coalesced, just as with ordinary device allocations in a growing pool.
    ON_CALL(executor_, Allocate(_, _))
        .WillByDefault([this](uint64_t size, int64_t) {
          void* ptr = reinterpret_cast<void*>(next_address_);
          next_address_ += size + kMiB;
          allocations_[ptr] = size;
          return se::DeviceAddressBase(ptr, size);
        });
    ON_CALL(executor_, Deallocate(_))
        .WillByDefault([this](se::DeviceAddressBase* memory) {
          freed_sizes_.push_back(allocations_.at(memory->opaque()));
          freed_addresses_.push_back(memory->opaque());
          allocations_.erase(memory->opaque());
        });
  }

  NiceMock<se::MockStreamExecutor> executor_;
  uintptr_t next_address_ = uintptr_t{1} << 40;
  std::unordered_map<void*, size_t> allocations_;
  std::vector<size_t> freed_sizes_;
  std::vector<void*> freed_addresses_;
  std::optional<std::string> original_garbage_collection_;
};

class GpuBFCGarbageCollectionTest
    : public GpuBFCAllocatorTest,
      public ::testing::WithParamInterface<const char*> {
 protected:
  void SetUp() override {
    GpuBFCAllocatorTest::SetUp();
    if (GetParam() != nullptr) {
      ASSERT_EQ(tsl::setenv("TF_ENABLE_GPU_GARBAGE_COLLECTION", GetParam(), 1),
                0);
    }
  }
};

TEST_P(GpuBFCGarbageCollectionTest,
       GrowingPoolReclaimsWholeFreeRegionsBeforeOom) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      CreateBFCAllocator(&executor_, FixedMemFraction{1.0},
                         /*preallocate=*/false, std::nullopt, {}, {}));
  constexpr size_t kAlignment = tsl::Allocator::kAllocatorAlignment;
  tsl::AllocationAttributes no_retry;
  no_retry.retry_on_failure = false;

  void* live = allocator->AllocateRaw(kAlignment, 2 * kMiB, no_retry);
  ASSERT_NE(live, nullptr);
  void* temporary = allocator->AllocateRaw(kAlignment, 4 * kMiB, no_retry);
  ASSERT_NE(temporary, nullptr);
  allocator->DeallocateRaw(temporary);
  temporary = allocator->AllocateRaw(kAlignment, 8 * kMiB, no_retry);
  ASSERT_NE(temporary, nullptr);
  allocator->DeallocateRaw(temporary);
  ASSERT_EQ(*allocator->GetStats()->pool_bytes, 14 * kMiB);
  EXPECT_THAT(freed_sizes_, IsEmpty());

  // The 16 MiB request fits neither the largest free region (8 MiB) nor the
  // remaining growth allowance (10 MiB). Returning the 4 and 8 MiB regions
  // permits it without releasing the live allocation or increasing the cap.
  void* large = allocator->AllocateRaw(kAlignment, 16 * kMiB, no_retry);
  if (GetParam() != nullptr && std::string(GetParam()) == "false") {
    EXPECT_EQ(large, nullptr);
    EXPECT_THAT(freed_sizes_, IsEmpty());
    EXPECT_EQ(*allocator->GetStats()->pool_bytes, 14 * kMiB);
    EXPECT_EQ(allocator->RequestedSize(live), 2 * kMiB);
    allocator->DeallocateRaw(live);
    return;
  }
  ASSERT_NE(large, nullptr);
  EXPECT_THAT(freed_sizes_, UnorderedElementsAre(4 * kMiB, 8 * kMiB));
  EXPECT_THAT(freed_addresses_, ::testing::Not(::testing::Contains(live)));
  EXPECT_EQ(allocator->RequestedSize(live), 2 * kMiB);
  EXPECT_EQ(allocator->GetStats()->bytes_in_use, 18 * kMiB);
  EXPECT_LE(*allocator->GetStats()->pool_bytes, kPoolLimit);
  // Reclamation must not evict live regions or exceed the configured limit.
  EXPECT_EQ(allocator->AllocateRaw(kAlignment, 8 * kMiB, no_retry), nullptr);
  EXPECT_THAT(freed_sizes_, UnorderedElementsAre(4 * kMiB, 8 * kMiB));
  EXPECT_EQ(allocator->RequestedSize(live), 2 * kMiB);
  allocator->DeallocateRaw(large);
  allocator->DeallocateRaw(live);
}

TEST_P(GpuBFCGarbageCollectionTest, PreallocatedPoolStaysReserved) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      CreateBFCAllocator(&executor_, FixedMemFraction{1.0},
                         /*preallocate=*/true, std::nullopt, {}, {}));
  void* ptr =
      allocator->AllocateRaw(tsl::Allocator::kAllocatorAlignment, 2 * kMiB);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(*allocator->GetStats()->pool_bytes, kPoolLimit);
  allocator->DeallocateRaw(ptr);
  EXPECT_EQ(*allocator->GetStats()->pool_bytes, kPoolLimit);
  EXPECT_THAT(freed_sizes_, IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(GarbageCollection, GpuBFCGarbageCollectionTest,
                         ::testing::Values(nullptr, "true", "false"));

TEST_F(GpuBFCAllocatorTest, GrowingPoolRejectsInvalidGarbageCollectionSetting) {
  ASSERT_EQ(tsl::setenv("TF_ENABLE_GPU_GARBAGE_COLLECTION", "invalid", 1), 0);
  EXPECT_THAT(CreateBFCAllocator(&executor_, FixedMemFraction{1.0},
                                 /*preallocate=*/false, std::nullopt, {}, {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("TF_ENABLE_GPU_GARBAGE_COLLECTION")));
}

TEST_F(GpuBFCAllocatorTest, PreallocatedPoolIgnoresGarbageCollectionSetting) {
  ASSERT_EQ(tsl::setenv("TF_ENABLE_GPU_GARBAGE_COLLECTION", "invalid", 1), 0);
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      CreateBFCAllocator(&executor_, FixedMemFraction{1.0},
                         /*preallocate=*/true, std::nullopt, {}, {}));
}

}  // namespace
}  // namespace xla
