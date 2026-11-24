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

#include "xla/backends/gpu/runtime/buffer_allocations.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

namespace xla::gpu {
namespace {

using ::testing::_;
using ::testing::AnyOf;
using ::testing::HasSubstr;
using ::testing::Optional;

TEST(BufferAllocationsTest, FindAllocationIndexReturnsCorrectIndex) {
  // Two 64-byte buffers at distinct addresses.
  char buf0[64];
  char buf1[64];
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(buf0, 64),
      se::DeviceAddressBase(buf1, 64),
  };

  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  // Address at the start of buf0 → index 0.
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf0, 1)), 0);

  // Address in the middle of buf1 → index 1.
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf1 + 32, 1)), 1);
}

TEST(BufferAllocationsTest, FindAllocationIndexReturnsNulloptForUnknown) {
  char buf[64];
  char other[64];
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(buf, 64),
  };

  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(other, 1)),
            std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexExcludesEndBoundary) {
  char buf[64];
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(buf, 64),
  };

  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  // Address at exactly buf + size should NOT match (strict less-than).
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf + 64, 1)),
            std::nullopt);

  // Address at buf + size - 1 should still match.
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf + 63, 1)), 0);
}

// The tests below use synthetic device addresses that are never dereferenced.
constexpr uintptr_t kBase = 0x7f0000000000;

se::DeviceAddressBase Addr(uintptr_t ptr, uint64_t size) {
  return se::DeviceAddressBase(tsl::safe_reinterpret_cast<void*>(ptr), size);
}

TEST(BufferAllocationsTest, FindAllocationIndexAdjacentAllocations) {
  std::vector<se::DeviceAddressBase> buffers = {
      Addr(kBase, 64),
      Addr(kBase + 64, 64),
  };
  BufferAllocations allocs(buffers, 0, nullptr);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, _)).Times(0);
  log.StartCapturingLogs();

  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase, 64)), 0);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 63, 1)), 0);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 64, 64)), 1);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 63, 2)), std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexRejectsNestedOverlap) {
  std::vector<se::DeviceAddressBase> buffers = {
      Addr(kBase, 256),        // 0: outer
      Addr(kBase + 64, 64),    // 1: nested in 0
      Addr(kBase + 4096, 64),  // 2: disjoint
  };
  BufferAllocations allocs(buffers, 0, nullptr);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       HasSubstr("Overlapping buffer allocations")))
      .Times(1);
  log.StartCapturingLogs();

  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 64, 64)), std::nullopt);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase, 64)), std::nullopt);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 4096, 1)), std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexRejectsPartialOverlap) {
  std::vector<se::DeviceAddressBase> buffers = {
      Addr(kBase, 256),        // 0
      Addr(kBase + 128, 256),  // 1: partially overlaps 0
      Addr(kBase + 4096, 64),  // 2: disjoint
  };
  BufferAllocations allocs(buffers, 0, nullptr);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       HasSubstr("Overlapping buffer allocations")))
      .Times(1);
  log.StartCapturingLogs();

  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 128, 128)), std::nullopt);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 4096, 1)), std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexRejectsSameStartDifferentSize) {
  std::vector<se::DeviceAddressBase> buffers = {
      Addr(kBase, 32),         // 0
      Addr(kBase, 32),         // 1: duplicate of 0
      Addr(kBase, 64),         // 2: same start, different size
      Addr(kBase + 4096, 64),  // 3: disjoint
  };
  BufferAllocations allocs(buffers, 0, nullptr);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       HasSubstr("Overlapping buffer allocations")))
      .Times(1);
  log.StartCapturingLogs();

  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase, 16)), std::nullopt);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 4096, 1)), std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexDuplicateAllocations) {
  // The same buffer passed twice, with no other overlap.
  std::vector<se::DeviceAddressBase> buffers = {
      Addr(kBase + 1024, 64), Addr(kBase, 64), Addr(kBase, 64)};
  BufferAllocations allocs(buffers, 0, nullptr);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, _)).Times(0);
  log.StartCapturingLogs();

  EXPECT_THAT(allocs.FindAllocationIndex(Addr(kBase + 8, 1)),
              Optional(AnyOf(1, 2)));
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 1024, 1)), 0);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 64, 1)), std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexNearMaxAddress) {
  std::vector<se::DeviceAddressBase> buffers = {Addr(UINTPTR_MAX - 1023, 1024)};
  BufferAllocations allocs(buffers, 0, nullptr);

  EXPECT_EQ(allocs.FindAllocationIndex(Addr(UINTPTR_MAX, 1)), 0);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(UINTPTR_MAX - 15, 32)),
            std::nullopt);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(UINTPTR_MAX - 1024, 1)),
            std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexSkipsEmptyAllocations) {
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(),  // 0: unassigned
      Addr(kBase, 0),           // 1: zero-size
      Addr(kBase, 64),          // 2
  };
  BufferAllocations allocs(buffers, 0, nullptr);

  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase, 1)), 2);
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase()), std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexSeesMutatedAddress) {
  std::vector<se::DeviceAddressBase> buffers = {Addr(kBase, 64)};
  BufferAllocations allocs(buffers, 0, nullptr);

  // Build the index, then move allocation 0.
  ASSERT_EQ(allocs.FindAllocationIndex(Addr(kBase, 1)), 0);
  allocs.GetMutableDeviceAddress(0) = Addr(kBase + 1024, 64);

  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase, 1)), std::nullopt);
  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase + 1024, 1)), 0);
}

TEST(BufferAllocationsTest, FindAllocationIndexSeesAddressAssignedAfterLookup) {
  std::vector<se::DeviceAddressBase> buffers = {se::DeviceAddressBase(),
                                                Addr(kBase, 0)};
  BufferAllocations allocs(buffers, 0, nullptr);

  ASSERT_EQ(allocs.FindAllocationIndex(Addr(kBase, 1)), std::nullopt);
  allocs.GetMutableDeviceAddress(0) = Addr(kBase, 64);

  EXPECT_EQ(allocs.FindAllocationIndex(Addr(kBase, 1)), 0);
}

TEST(BufferAllocationsTest, FindAllocationIndexAfterMove) {
  std::vector<se::DeviceAddressBase> buffers = {Addr(kBase, 64)};
  BufferAllocations allocs(buffers, 0, nullptr);
  ASSERT_EQ(allocs.FindAllocationIndex(Addr(kBase, 1)), 0);

  BufferAllocations moved(std::move(allocs));
  EXPECT_EQ(moved.FindAllocationIndex(Addr(kBase, 1)), 0);

  BufferAllocations assigned({Addr(kBase + 1024, 64)}, 0, nullptr);
  ASSERT_EQ(assigned.FindAllocationIndex(Addr(kBase + 1024, 1)), 0);
  assigned = std::move(moved);
  EXPECT_EQ(assigned.FindAllocationIndex(Addr(kBase, 1)), 0);
  EXPECT_EQ(assigned.FindAllocationIndex(Addr(kBase + 1024, 1)), std::nullopt);
}

}  // namespace
}  // namespace xla::gpu
