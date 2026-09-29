/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/utils/buffer_pool.h"

#include "xla/tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

TEST(BufferPoolTest, GetOrCreateBufferAlloc) {
  constexpr size_t kBufferSizeInBytes = 32 * 1024;
  BufferPool buffer_pool(kBufferSizeInBytes);
  uint8_t* first_buffer = buffer_pool.GetOrCreateBuffer();
  EXPECT_NE(first_buffer, nullptr);

  // Checks that a second unique buffer is allocated.
  uint8_t* second_buffer = buffer_pool.GetOrCreateBuffer();
  EXPECT_NE(second_buffer, first_buffer);

  for (size_t idx = 0; idx < kBufferSizeInBytes; ++idx) {
    // Check that buffer is writable, msan will warn if not.
    first_buffer[idx] = 0xAB;
  }

  // Clean up buffers held by the test.
  buffer_pool.ReclaimBuffer(first_buffer);
  buffer_pool.ReclaimBuffer(second_buffer);
}

TEST(BufferPoolTest, GetOrCreateBufferReuse) {
  constexpr size_t kBufferSizeInBytes = 32 * 1024;
  BufferPool buffer_pool(kBufferSizeInBytes);
  uint8_t* buffer = buffer_pool.GetOrCreateBuffer();
  EXPECT_NE(buffer, nullptr);
  // Write a dummy value to the buffer.
  buffer[0] = 0xFF;

  uint8_t* previous_buffer = buffer;
  buffer_pool.ReclaimBuffer(buffer);

  // Check that we can retrieved a recently reclaimed buffer.
  uint8_t* reused_buffer = buffer_pool.GetOrCreateBuffer();
  EXPECT_EQ(reused_buffer, previous_buffer);

  for (size_t idx = 0; idx < kBufferSizeInBytes; ++idx) {
    // Check that reused buffer is writable, msan will warn if not.
    reused_buffer[idx] = 0xCD;
  }

  // Clean up buffers held by the test.
  buffer_pool.ReclaimBuffer(reused_buffer);
}

TEST(BufferPoolTest, DestroyAllBuffers) {
  constexpr size_t kBufferSizeInBytes = 32 * 1024;
  BufferPool buffer_pool(kBufferSizeInBytes);
  uint8_t* first_buffer = buffer_pool.GetOrCreateBuffer();
  EXPECT_NE(first_buffer, nullptr);

  // Check that first buffer (not reclaimed) is still writable after
  // DestroyAllBuffers.
  buffer_pool.DestroyAllBuffers();
  for (size_t idx = 0; idx < kBufferSizeInBytes; ++idx) {
    first_buffer[idx] = 0xEF;
  }

  uint8_t* second_buffer = buffer_pool.GetOrCreateBuffer();
  for (size_t idx = 0; idx < kBufferSizeInBytes; ++idx) {
    // Check that second buffer is writable.
    second_buffer[idx] = 0xAB;
  }

  // Clean up buffers held by the test.
  buffer_pool.ReclaimBuffer(first_buffer);
  buffer_pool.ReclaimBuffer(second_buffer);
}

TEST(BufferPoolWrapperTest, DefaultPreallocationIsZeroed) {
  constexpr size_t kBufferSizeInBytes = 32 * 1024;
  BufferPoolWrapper wrapper(kBufferSizeInBytes);
  EXPECT_EQ(wrapper.GetPreallocationCount(), 8);
  EXPECT_EQ(wrapper.GetBufferSizeInBytes(), kBufferSizeInBytes);

  std::vector<uint8_t*> buffers;
  for (size_t i = 0; i < 8; ++i) {
    uint8_t* buffer = wrapper.GetOrCreateBuffer();
    ASSERT_NE(buffer, nullptr);
    for (size_t j = 0; j < kBufferSizeInBytes; ++j) {
      ASSERT_EQ(buffer[j], 0);
    }
    buffers.push_back(buffer);
  }

  for (uint8_t* buffer : buffers) {
    wrapper.ReclaimBuffer(buffer);
  }
}

TEST(BufferPoolWrapperTest, CustomPreallocationCount) {
  constexpr size_t kBufferSizeInBytes = 1024;
  constexpr size_t kPreallocationCount = 4;
  BufferPoolWrapper wrapper(kBufferSizeInBytes, kPreallocationCount);
  EXPECT_EQ(wrapper.GetPreallocationCount(), kPreallocationCount);

  std::vector<uint8_t*> buffers;
  for (size_t i = 0; i < kPreallocationCount; ++i) {
    uint8_t* buffer = wrapper.GetOrCreateBuffer();
    ASSERT_NE(buffer, nullptr);
    for (size_t j = 0; j < kBufferSizeInBytes; ++j) {
      ASSERT_EQ(buffer[j], 0);
    }
    buffers.push_back(buffer);
  }

  for (uint8_t* buffer : buffers) {
    wrapper.ReclaimBuffer(buffer);
  }
}

TEST(BufferPoolWrapperTest, ReclaimZeroesBufferAndPutsBackToFreeList) {
  constexpr size_t kBufferSizeInBytes = 1024;
  BufferPoolWrapper wrapper(kBufferSizeInBytes, /*preallocation_count=*/1);

  uint8_t* buffer = wrapper.GetOrCreateBuffer();
  ASSERT_NE(buffer, nullptr);
  std::memset(buffer, 0xAB, kBufferSizeInBytes);

  // Queue is now empty (0 < 1), so reclaiming will zero and put back in free
  // list.
  wrapper.ReclaimBuffer(buffer);

  // When retrieved again, buffer should be zeroed.
  uint8_t* reused_buffer = wrapper.GetOrCreateBuffer();
  EXPECT_EQ(reused_buffer, buffer);
  for (size_t j = 0; j < kBufferSizeInBytes; ++j) {
    ASSERT_EQ(reused_buffer[j], 0);
  }

  wrapper.ReclaimBuffer(reused_buffer);
}

TEST(BufferPoolWrapperTest,
     ReclaimDropsBufferWhenQueueAtOrExceedsPreallocation) {
  constexpr size_t kBufferSizeInBytes = 1024;
  BufferPoolWrapper wrapper(kBufferSizeInBytes, /*preallocation_count=*/2);
  EXPECT_EQ(wrapper.GetBufferPool().GetFreeBuffersCount(), 2);

  uint8_t* b1 = wrapper.GetOrCreateBuffer();
  uint8_t* b2 = wrapper.GetOrCreateBuffer();
  // 3rd buffer exceeds preallocation and is dynamically allocated.
  uint8_t* b3 = wrapper.GetOrCreateBuffer();
  EXPECT_EQ(wrapper.GetBufferPool().GetFreeBuffersCount(), 0);

  // Reclaim b1: queue size was 0 (< 2), so b1 is zeroed and added to queue.
  wrapper.ReclaimBuffer(b1);
  EXPECT_EQ(wrapper.GetBufferPool().GetFreeBuffersCount(), 1);

  // Reclaim b2: queue size was 1 (< 2), so b2 is zeroed and added to queue.
  wrapper.ReclaimBuffer(b2);
  EXPECT_EQ(wrapper.GetBufferPool().GetFreeBuffersCount(), 2);

  // Reclaim b3: queue size is now 2 (>= preallocation_count of 2), so b3 is
  // DROPPED (freed).
  wrapper.ReclaimBuffer(b3);
  EXPECT_EQ(wrapper.GetBufferPool().GetFreeBuffersCount(), 2);

  // Re-pop the 2 buffers from the queue; both should be zeroed.
  uint8_t* popped1 = wrapper.GetOrCreateBuffer();
  uint8_t* popped2 = wrapper.GetOrCreateBuffer();
  for (size_t j = 0; j < kBufferSizeInBytes; ++j) {
    ASSERT_EQ(popped1[j], 0);
    ASSERT_EQ(popped2[j], 0);
  }

  wrapper.ReclaimBuffer(popped1);
  wrapper.ReclaimBuffer(popped2);
}

TEST(BufferPoolWrapperTest, DynamicAllocationBeyondPreallocationIsZeroed) {
  constexpr size_t kBufferSizeInBytes = 1024;
  BufferPoolWrapper wrapper(kBufferSizeInBytes, /*preallocation_count=*/2);

  uint8_t* b1 = wrapper.GetOrCreateBuffer();
  uint8_t* b2 = wrapper.GetOrCreateBuffer();
  // 3rd buffer exceeds pre-allocation and must be dynamically allocated.
  uint8_t* b3 = wrapper.GetOrCreateBuffer();
  ASSERT_NE(b3, nullptr);
  for (size_t j = 0; j < kBufferSizeInBytes; ++j) {
    ASSERT_EQ(b3[j], 0);
  }

  wrapper.ReclaimBuffer(b1);
  wrapper.ReclaimBuffer(b2);
  wrapper.ReclaimBuffer(b3);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
