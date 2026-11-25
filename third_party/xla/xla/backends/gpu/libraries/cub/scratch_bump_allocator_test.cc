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

#include "xla/backends/gpu/libraries/cub/scratch_bump_allocator.h"

#include <cstdint>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(ScratchBumpAllocatorTest, EmptyAllocatorHasZeroTotalBytes) {
  ScratchBumpAllocator alloc(/*alignment_bytes=*/256);
  EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(0));
}

TEST(ScratchBumpAllocatorTest, AlignsSequentialAllocationsAndTotalBytes) {
  ScratchBumpAllocator alloc(/*alignment_bytes=*/256);

  EXPECT_THAT(alloc.Allocate(1), IsOkAndHolds(ScratchRange{0, 1}));
  EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(256));

  EXPECT_THAT(alloc.Allocate(256), IsOkAndHolds(ScratchRange{256, 256}));
  EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(512));

  EXPECT_THAT(alloc.Allocate(17), IsOkAndHolds(ScratchRange{512, 17}));
  EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(768));
}

TEST(ScratchBumpAllocatorTest, ZeroByteAllocationsDoNotAdvanceCursor) {
  ScratchBumpAllocator alloc(/*alignment_bytes=*/64);

  EXPECT_THAT(alloc.Allocate(0), IsOkAndHolds(ScratchRange{0, 0}));
  EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(0));

  EXPECT_THAT(alloc.Allocate(10), IsOkAndHolds(ScratchRange{0, 10}));
  EXPECT_THAT(alloc.Allocate(0), IsOkAndHolds(ScratchRange{0, 0}));
  EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(64));

  EXPECT_THAT(alloc.Allocate(5), IsOkAndHolds(ScratchRange{64, 5}));
  EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(128));
}

TEST(ScratchBumpAllocatorTest, UnitAlignmentPacksTightly) {
  ScratchBumpAllocator alloc(/*alignment_bytes=*/1);

  EXPECT_THAT(alloc.Allocate(3), IsOkAndHolds(ScratchRange{0, 3}));
  EXPECT_THAT(alloc.Allocate(5), IsOkAndHolds(ScratchRange{3, 5}));
  EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(8));
}

TEST(ScratchBumpAllocatorTest, RejectsNegativeSizeWithoutMutatingState) {
  ScratchBumpAllocator alloc(/*alignment_bytes=*/64);
  EXPECT_THAT(alloc.Allocate(10), IsOkAndHolds(ScratchRange{0, 10}));
  EXPECT_THAT(alloc.Allocate(-1), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(64));
  EXPECT_THAT(alloc.Allocate(12), IsOkAndHolds(ScratchRange{64, 12}));
}

TEST(ScratchBumpAllocatorTest, RejectsNonPositiveAlignment) {
  for (int64_t bad_align : {int64_t{0}, int64_t{-1}, int64_t{-256},
                            std::numeric_limits<int64_t>::min()}) {
    ScratchBumpAllocator alloc(bad_align);
    EXPECT_THAT(alloc.Allocate(0),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(alloc.Allocate(16),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(alloc.TotalBytes(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(ScratchBumpAllocatorTest, DetectsAdditionAndAlignmentOverflow) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

  // Addition overflow after an initial allocation.
  {
    ScratchBumpAllocator alloc(/*alignment_bytes=*/256);
    EXPECT_THAT(alloc.Allocate(256), IsOkAndHolds(ScratchRange{0, 256}));
    EXPECT_THAT(alloc.Allocate(kMax - 255),
                StatusIs(absl::StatusCode::kInvalidArgument));
    // State remains valid after failed allocation.
    EXPECT_THAT(alloc.TotalBytes(), IsOkAndHolds(256));
  }

  // Alignment rounding overflow on next allocation and on TotalBytes().
  {
    ScratchBumpAllocator alloc(/*alignment_bytes=*/256);
    EXPECT_THAT(alloc.Allocate(kMax - 1),
                IsOkAndHolds(ScratchRange{0, kMax - 1}));
    EXPECT_THAT(alloc.Allocate(1),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(alloc.TotalBytes(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
}  // namespace xla::gpu
