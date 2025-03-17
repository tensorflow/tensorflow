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

#include "xla/stream_executor/generic_memory_allocator.h"

#include <array>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace {

TEST(GenericMemoryAllocatorTest, AllocateReturnsCorrectMemoryAllocation) {
  std::array<char, 64> array;
  int deleter_called = 0;
  auto allocator = GenericMemoryAllocator(
      [&array, &deleter_called](
          uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        EXPECT_EQ(array.size(), 64);
        return std::make_unique<GenericMemoryAllocation>(
            array.data(), array.size(), [&deleter_called](void*, uint64_t) {
              EXPECT_EQ(deleter_called, 0);
              ++deleter_called;
            });
      });
  TF_ASSERT_OK_AND_ASSIGN(auto allocation, allocator.Allocate(64));
  EXPECT_EQ(deleter_called, 0);
  allocation.reset();
  EXPECT_EQ(deleter_called, 1);
}

TEST(GenericMemoryAllocatorTest, AllocateReturnsError) {
  auto allocator = GenericMemoryAllocator(
      [](uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        return absl::InternalError("Failed to allocate memory");
      });
  EXPECT_THAT(allocator.Allocate(64), testing::Not(tsl::testing::IsOk()));
}

}  // namespace
}  // namespace stream_executor
