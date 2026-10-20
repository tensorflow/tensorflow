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
#include <limits>
#include <memory>
#include <optional>

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
#include "xla/tsl/framework/bfc_allocator.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::NiceMock;

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

}  // namespace
}  // namespace xla
