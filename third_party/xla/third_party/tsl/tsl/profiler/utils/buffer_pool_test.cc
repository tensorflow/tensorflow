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

#include "tsl/profiler/utils/buffer_pool.h"

#include "tsl/platform/test.h"

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

}  // namespace
}  // namespace profiler
}  // namespace tsl
