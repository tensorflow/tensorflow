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

#include "xla/stream_executor/integrations/stream_executor_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"

namespace stream_executor {
namespace {

TEST(StreamExecutorAllocatorTest, NoMemoryReturnsNullptr) {
  auto allocator = std::make_unique<GenericMemoryAllocator>(
      [](uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        return absl::InternalError("Failed to allocate memory");
      });

  StreamExecutorAllocator stream_executor_allocator(
      std::move(allocator), MemoryType::kHost, /*index=*/0,
      /*alloc_visitors=*/{},
      /*free_visitors=*/{});
  size_t bytes_received = 0;
  EXPECT_EQ(stream_executor_allocator.Alloc(/*alignment=*/1, /*num_bytes=*/64,
                                            &bytes_received),
            nullptr);
  EXPECT_EQ(bytes_received, 0);
}

TEST(StreamExecutorAllocatorTest, DoesntSupportCoalescing) {
  auto allocator = std::make_unique<GenericMemoryAllocator>(
      [](uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        return absl::InternalError("Failed to allocate memory");
      });
  StreamExecutorAllocator stream_executor_allocator(
      std::move(allocator), MemoryType::kHost, /*index=*/0,
      /*alloc_visitors=*/{},
      /*free_visitors=*/{});
  EXPECT_FALSE(stream_executor_allocator.SupportsCoalescing());
}

TEST(StreamExecutorAllocatorTest, GetMemoryTypeReturnsHostPinnedForHostMemory) {
  auto allocator = std::make_unique<GenericMemoryAllocator>(
      [](uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        return absl::InternalError("Failed to allocate memory");
      });
  StreamExecutorAllocator stream_executor_allocator(
      std::move(allocator), MemoryType::kHost, /*index=*/0,
      /*alloc_visitors=*/{},
      /*free_visitors=*/{});
  EXPECT_EQ(tsl::AllocatorMemoryType::kHostPinned,
            stream_executor_allocator.GetMemoryType());
}

TEST(StreamExecutorAllocatorTest, GetMemoryTypeReturnsDeviceForDeviceMemory) {
  auto allocator = std::make_unique<GenericMemoryAllocator>(
      [](uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        return absl::InternalError("Failed to allocate memory");
      });
  StreamExecutorAllocator stream_executor_allocator(
      std::move(allocator), MemoryType::kDevice, /*index=*/0,
      /*alloc_visitors=*/{},
      /*free_visitors=*/{});
  EXPECT_EQ(tsl::AllocatorMemoryType::kDevice,
            stream_executor_allocator.GetMemoryType());
}

TEST(StreamExecutorAllocatorTest,
     MemoryAllocationWorksAndVisitsAppropriateVisitor) {
  MemoryAllocation* allocation = nullptr;
  auto allocator = std::make_unique<GenericMemoryAllocator>(
      [&allocation](
          uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        auto new_allocation = std::make_unique<GenericMemoryAllocation>(
            new char[64], 64, [&allocation](void* ptr, uint64_t size) {
              EXPECT_EQ(ptr, allocation->opaque());
              EXPECT_EQ(size, 64);
              char* char_ptr = static_cast<char*>(ptr);
              delete[] char_ptr;
            });
        allocation = new_allocation.get();
        return new_allocation;
      });

  bool alloc_visitor_called = false;
  auto alloc_visitor = [&allocation, &alloc_visitor_called](
                           void* ptr, int index, uint64_t size) {
    EXPECT_EQ(ptr, allocation->opaque());
    EXPECT_EQ(index, 0);
    EXPECT_EQ(size, 64);
    alloc_visitor_called = true;
  };

  bool free_visitor_called = false;
  auto free_visitor = [&allocation, &free_visitor_called](void* ptr, int index,
                                                          uint64_t size) {
    EXPECT_EQ(ptr, allocation->opaque());
    EXPECT_EQ(index, 0);
    EXPECT_EQ(size, 64);
    free_visitor_called = true;
  };
  StreamExecutorAllocator stream_executor_allocator(
      std::move(allocator), MemoryType::kDevice, /*index=*/0, {alloc_visitor},
      {free_visitor});
  EXPECT_FALSE(free_visitor_called);
  EXPECT_FALSE(alloc_visitor_called);
  size_t bytes_received = 0;
  EXPECT_EQ(stream_executor_allocator.Alloc(/*alignment=*/1, /*num_bytes=*/64,
                                            &bytes_received),
            allocation->opaque());
  EXPECT_TRUE(alloc_visitor_called);
  stream_executor_allocator.Free(allocation->opaque(), 64);
  EXPECT_TRUE(free_visitor_called);
}

}  // namespace
}  // namespace stream_executor
