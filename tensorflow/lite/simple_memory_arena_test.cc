/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/simple_memory_arena.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace {

void ReportError(TfLiteContext* context, const char* format, ...) {}

class ReallocatingTestAllocator {
 public:
  explicit ReallocatingTestAllocator(bool fail_reallocate = false)
      : fail_reallocate_(fail_reallocate),
        allocator_{this, AllocateCallback, ReallocateCallback,
                   DeallocateCallback} {}

  TfLiteAllocator* GetAllocator() { return &allocator_; }

  int allocate_calls() const { return allocate_calls_; }
  int reallocate_calls() const { return reallocate_calls_; }
  int deallocate_calls() const { return deallocate_calls_; }
  bool arguments_valid() const { return arguments_valid_; }

 private:
  static constexpr size_t kAlignment = 64;
  static constexpr size_t kCapacity = 512;

  static void* AllocateCallback(void* data, size_t bytes, size_t alignment) {
    auto* self = static_cast<ReallocatingTestAllocator*>(data);
    ++self->allocate_calls_;
    const bool valid = self->current_pointer_ == nullptr && bytes > 0 &&
                       bytes <= kCapacity && alignment == kAlignment;
    self->arguments_valid_ = self->arguments_valid_ && valid;
    if (!valid) {
      return nullptr;
    }
    self->current_pointer_ = self->initial_storage_;
    self->current_bytes_ = bytes;
    return self->current_pointer_;
  }

  static void* ReallocateCallback(void* data, void* ptr, size_t old_bytes,
                                  size_t new_bytes, size_t alignment) {
    auto* self = static_cast<ReallocatingTestAllocator*>(data);
    ++self->reallocate_calls_;
    const bool valid = ptr == self->current_pointer_ &&
                       old_bytes == self->current_bytes_ && new_bytes > 0 &&
                       new_bytes <= kCapacity && alignment == kAlignment;
    self->arguments_valid_ = self->arguments_valid_ && valid;
    if (!valid || self->fail_reallocate_) {
      return nullptr;
    }
    std::memcpy(self->resized_storage_, self->current_pointer_,
                std::min(old_bytes, new_bytes));
    self->current_pointer_ = self->resized_storage_;
    self->current_bytes_ = new_bytes;
    return self->current_pointer_;
  }

  static void DeallocateCallback(void* data, void* ptr, size_t bytes,
                                 size_t alignment) {
    auto* self = static_cast<ReallocatingTestAllocator*>(data);
    ++self->deallocate_calls_;
    const bool valid = ptr == self->current_pointer_ &&
                       bytes == self->current_bytes_ && alignment == kAlignment;
    self->arguments_valid_ = self->arguments_valid_ && valid;
    if (valid) {
      self->current_pointer_ = nullptr;
      self->current_bytes_ = 0;
    }
  }

  bool fail_reallocate_;
  TfLiteAllocator allocator_;
  int allocate_calls_ = 0;
  int reallocate_calls_ = 0;
  int deallocate_calls_ = 0;
  bool arguments_valid_ = true;
  void* current_pointer_ = nullptr;
  size_t current_bytes_ = 0;
  alignas(kAlignment) char initial_storage_[kCapacity];
  alignas(kAlignment) char resized_storage_[kCapacity];
};

TEST(SimpleMemoryArenaTest, BasicArenaOperations) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[6];

  arena.Allocate(&context, 32, 2047, 0, 1, 3, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 2, 5, &allocs[1]);
  arena.Allocate(&context, 32, 2047, 2, 3, 6, &allocs[2]);
  arena.Allocate(&context, 32, 2047, 3, 5, 6, &allocs[3]);
  arena.Allocate(&context, 32, 1023, 4, 4, 6, &allocs[4]);
  arena.Allocate(&context, 32, 1023, 5, 6, 6, &allocs[5]);

  EXPECT_EQ(allocs[0].offset, 0);
  EXPECT_EQ(allocs[1].offset, 2048);
  EXPECT_EQ(allocs[2].offset, 4096);
  EXPECT_EQ(allocs[3].offset, 0);
  EXPECT_EQ(allocs[4].offset, 6144);
  EXPECT_EQ(allocs[5].offset, 2048);
}

TEST(ResizableAlignedBufferTest, FailedResizePreservesOriginalBuffer) {
  ResizableAlignedBuffer buffer(/*alignment=*/64, /*subgraph_index=*/0);

  bool reallocated = false;
  ASSERT_EQ(buffer.Resize(/*new_size=*/128, &reallocated), kTfLiteOk);
  ASSERT_TRUE(reallocated);
  char* const original_ptr = buffer.GetPtr();
  const size_t original_size = buffer.GetSize();

  reallocated = true;
  EXPECT_EQ(buffer.Resize(std::numeric_limits<size_t>::max(), &reallocated),
            kTfLiteError);
  EXPECT_FALSE(reallocated);
  EXPECT_EQ(buffer.GetPtr(), original_ptr);
  EXPECT_EQ(buffer.GetSize(), original_size);
}

TEST(ResizableAlignedBufferTest, UsesCustomAllocatorReallocateCallback) {
  ReallocatingTestAllocator allocator;
  {
    ResizableAlignedBuffer buffer(/*alignment=*/64, /*subgraph_index=*/0,
                                  allocator.GetAllocator());

    bool reallocated = false;
    ASSERT_EQ(buffer.Resize(/*new_size=*/128, &reallocated), kTfLiteOk);
    ASSERT_TRUE(reallocated);
    ASSERT_NE(buffer.GetPtr(), nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.GetPtr()) % 64, 0);
    EXPECT_EQ(allocator.allocate_calls(), 1);
    EXPECT_EQ(allocator.reallocate_calls(), 0);
    std::memset(buffer.GetPtr(), 0x5a, buffer.GetSize());

    char* const original_ptr = buffer.GetPtr();
    ASSERT_EQ(buffer.Resize(/*new_size=*/256, &reallocated), kTfLiteOk);
    EXPECT_TRUE(reallocated);
    EXPECT_NE(buffer.GetPtr(), original_ptr);
    EXPECT_EQ(buffer.GetSize(), 256);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.GetPtr()) % 64, 0);
    EXPECT_EQ(allocator.allocate_calls(), 1);
    EXPECT_EQ(allocator.reallocate_calls(), 1);
    EXPECT_TRUE(std::all_of(buffer.GetPtr(), buffer.GetPtr() + 128,
                            [](char value) { return value == 0x5a; }));
    EXPECT_TRUE(allocator.arguments_valid());
  }
  EXPECT_EQ(allocator.deallocate_calls(), 1);
  EXPECT_TRUE(allocator.arguments_valid());
}

TEST(ResizableAlignedBufferTest,
     FailedCustomReallocatePreservesOriginalBuffer) {
  ReallocatingTestAllocator allocator(/*fail_reallocate=*/true);
  {
    ResizableAlignedBuffer buffer(/*alignment=*/64, /*subgraph_index=*/0,
                                  allocator.GetAllocator());

    bool reallocated = false;
    ASSERT_EQ(buffer.Resize(/*new_size=*/128, &reallocated), kTfLiteOk);
    std::memset(buffer.GetPtr(), 0x5a, buffer.GetSize());
    char* const original_ptr = buffer.GetPtr();
    const size_t original_size = buffer.GetSize();

    reallocated = true;
    EXPECT_EQ(buffer.Resize(/*new_size=*/256, &reallocated), kTfLiteError);
    EXPECT_FALSE(reallocated);
    EXPECT_EQ(buffer.GetPtr(), original_ptr);
    EXPECT_EQ(buffer.GetSize(), original_size);
    EXPECT_TRUE(std::all_of(buffer.GetPtr(), buffer.GetPtr() + original_size,
                            [](char value) { return value == 0x5a; }));
    EXPECT_EQ(allocator.allocate_calls(), 1);
    EXPECT_EQ(allocator.reallocate_calls(), 1);
    EXPECT_EQ(allocator.deallocate_calls(), 0);
    EXPECT_TRUE(allocator.arguments_valid());
  }
  EXPECT_EQ(allocator.deallocate_calls(), 1);
  EXPECT_TRUE(allocator.arguments_valid());
}

TEST(SimpleMemoryArenaTest, BasicZeroAlloc) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval alloc;

  // Zero-sized allocs should have a 0 offset and size.
  ASSERT_EQ(arena.Allocate(&context, 32, 0, 0, 1, 2, &alloc), kTfLiteOk);
  EXPECT_EQ(alloc.offset, 0);
  EXPECT_EQ(alloc.size, 0);

  // The zero-sized alloc should resolve to null.
  char* resolved_ptr = nullptr;
  bool reallocated = false;
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  EXPECT_FALSE(reallocated);  // Don't allocate when zero bytes are needed.
  EXPECT_EQ(resolved_ptr, nullptr);
}

TEST(SimpleMemoryArenaTest, InterleavedZeroAlloc) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[4];

  // Interleave some zero and non-zero-sized allocations and deallocations.
  ASSERT_EQ(arena.Allocate(&context, 32, 2047, 0, 0, 4, &allocs[0]), kTfLiteOk);
  ASSERT_EQ(arena.Allocate(&context, 32, 0, 1, 1, 2, &allocs[1]), kTfLiteOk);
  ASSERT_EQ(arena.Allocate(&context, 32, 1023, 2, 1, 2, &allocs[2]), kTfLiteOk);
  ASSERT_EQ(arena.Allocate(&context, 32, 2047, 3, 3, 4, &allocs[3]), kTfLiteOk);

  // Deallocation of a zero-sized alloc should not impact the allocator offsets.
  EXPECT_EQ(allocs[0].offset, 0);
  EXPECT_EQ(allocs[1].offset, 0);
  EXPECT_EQ(allocs[2].offset, 2048);
  EXPECT_EQ(allocs[3].offset, 2048);
}

TEST(SimpleMemoryArenaTest, TestClearPlan) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[9];

  arena.Allocate(&context, 32, 2047, 0, 0, 2, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 1, 2, &allocs[1]);
  arena.Allocate(&context, 32, 2047, 2, 1, 2, &allocs[2]);
  bool reallocated = false;
  arena.Commit(&reallocated);
  ASSERT_TRUE(reallocated);

  EXPECT_EQ(allocs[0].offset, 0);
  EXPECT_EQ(allocs[1].offset, 2048);
  EXPECT_EQ(allocs[2].offset, 4096);

  arena.ClearPlan();

  // Test with smaller allocs.
  arena.Allocate(&context, 32, 1023, 3, 0, 2, &allocs[3]);
  arena.Allocate(&context, 32, 1023, 4, 1, 2, &allocs[4]);
  arena.Allocate(&context, 32, 1023, 5, 1, 2, &allocs[5]);
  arena.Commit(&reallocated);
  ASSERT_FALSE(reallocated);

  EXPECT_EQ(allocs[3].offset, 0);
  EXPECT_EQ(allocs[4].offset, 1024);
  EXPECT_EQ(allocs[5].offset, 2048);

  arena.ClearPlan();

  // Test larger allocs which should require a reallocation.
  arena.Allocate(&context, 32, 4095, 6, 0, 2, &allocs[6]);
  arena.Allocate(&context, 32, 4095, 7, 1, 2, &allocs[7]);
  arena.Allocate(&context, 32, 4095, 8, 1, 2, &allocs[8]);
  arena.Commit(&reallocated);
  ASSERT_TRUE(reallocated);

  EXPECT_EQ(allocs[6].offset, 0);
  EXPECT_EQ(allocs[7].offset, 4096);
  EXPECT_EQ(allocs[8].offset, 8192);
}

TEST(SimpleMemoryArenaTest, TestPurgeAllocs) {
  TfLiteContext context;
  context.ReportError = ReportError;
  SimpleMemoryArena arena(/*arena_alignment=*/64);
  ArenaAllocWithUsageInterval allocs[5];

  arena.Allocate(&context, /*alignment=*/32, /*size=*/2047, /*tensor=*/0,
                 /*first_node=*/0, /*last_node=*/2, &allocs[0]);
  arena.Allocate(&context, /*alignment=*/32, /*size=*/2047, /*tensor=*/1,
                 /*first_node=*/1, /*last_node=*/2, &allocs[1]);
  arena.Allocate(&context, /*alignment=*/32, /*size=*/2047, /*tensor=*/2,
                 /*first_node=*/2, /*last_node=*/3, &allocs[2]);

  bool reallocated = false;
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  ASSERT_TRUE(reallocated);
  char* resolved_ptr0 = nullptr;
  char* resolved_ptr1 = nullptr;
  char* resolved_ptr2 = nullptr;
  char* resolved_ptr3 = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr0), kTfLiteOk);
  EXPECT_NE(resolved_ptr0, nullptr);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[1], &resolved_ptr1), kTfLiteOk);
  EXPECT_EQ(resolved_ptr1, resolved_ptr0 + 2048);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[2], &resolved_ptr2), kTfLiteOk);
  EXPECT_EQ(resolved_ptr2, resolved_ptr1 + 2048);

  /* This is the expected arena. Tensors are available in the range [first_node,
   * last_node] meaning that tensor2 must be stacked on top of tensors 0+1,
   * despite these tensors being deallocated at node 2.
   *              |xxxxx| tensor2
   *        |xxxxx| tensor1
   *  |xxxxxxxxxxx| tensor0
   *  -------------------
   *  |     |     |     |
   *  0     1     2     3
   *  ___________________
   */
  /* Delete all alloc information for tensors deallocated before node 4. This
   * information is not required for computing allocations anymore, it only
   * slows down the calculation as irrelevant allocs must be checked.
   */
  arena.PurgeActiveAllocs(4);
  arena.Allocate(&context, /*alignment=*/32, /*size=*/13, /*tensor=*/3,
                 /*first_node=*/4, /*last_node=*/5, &allocs[4]);
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[4], &resolved_ptr3), kTfLiteOk);
  /* no tensors are allocated at node 4, so tensor 3's offset should be zero.*/
  ASSERT_EQ(allocs[4].offset, 0);
  /* TFLite executes graphs from node 0 to node N-1 in order. Once a node has
   * been executed, the use of the memory arena means that there is no
   * guarantee that it's input data is still available, the arena will allocate
   * memory as efficiently as possible, potentially re-using memory as soon as
   * it becomes available again. This means that we cannot allocate a tensor at
   * node K after node K has been executed, the data is not guaranteed to be
   * available anymore. So to test purge, we re-allocate tensor 2 which is
   * active between nodes 0 and 2. If old alloc information has been purged, it
   * will no longer be stacked on top of tensors 0+1.
   * This is the expected arena after tensors 2 and 4 have been allocated.
   *
   *      tensor0 |xxxxx|     |xxxxx|tensor4
   *  -------------------------------
   *  |     |     |     |     |     |
   *  0     1     2     3     4     5
   *  ___________________
   */
  arena.Allocate(&context, /*alignment=*/32, /*size=*/2047, /*tensor=*/0,
                 /*first_node=*/0, /*last_node=*/2, &allocs[0]);
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[3], &resolved_ptr3), kTfLiteOk);
  ASSERT_EQ(allocs[0].offset, 0);
}

TEST(SimpleMemoryArenaTest, TestResetAllocs) {
  TfLiteContext context;
  context.ReportError = ReportError;
  SimpleMemoryArena arena(/*arena_alignment=*/64);
  ArenaAllocWithUsageInterval allocs[9];

  arena.Allocate(&context, /*alignment=*/32, /*size=*/2047, /*tensor=*/0,
                 /*first_node=*/0, /*last_node=*/2, &allocs[0]);
  arena.Allocate(&context, /*alignment=*/32, /*size=*/2047, /*tensor=*/1,
                 /*first_node=*/1, /*last_node=*/2, &allocs[1]);
  arena.Allocate(&context, /*alignment=*/32, /*size=*/2047, /*tensor=*/2,
                 /*first_node=*/2, /*last_node=*/3, &allocs[2]);

  bool reallocated = false;
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  ASSERT_TRUE(reallocated);
  char* resolved_ptr0 = nullptr;
  char* resolved_ptr1 = nullptr;
  char* resolved_ptr2 = nullptr;
  char* resolved_ptr3 = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr0), kTfLiteOk);
  EXPECT_NE(resolved_ptr0, nullptr);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[1], &resolved_ptr1), kTfLiteOk);
  EXPECT_EQ(resolved_ptr1, resolved_ptr0 + 2048);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[2], &resolved_ptr2), kTfLiteOk);
  EXPECT_EQ(resolved_ptr2, resolved_ptr1 + 2048);

  /* This is the expected arena. Tensors are available in the range [first_node,
   * last_node] meaning that tensor2 must be stacked on top of tensors 0+1,
   * despite these tensors being deallocated at node 2.
   *              |xxxxx| tensor2
   *        |xxxxx| tensor1
   *  |xxxxxxxxxxx| tensor0
   *  -------------------
   *  |     |     |     |
   *  0     1     2     3
   *  ___________________
   */
  /* Allocate tensor 3 after already commiting. It should increase the size of
   * the arena, triggering a realloc and alloc[3] should be placed on top of 0,
   * 1, 2
   */
  arena.Allocate(&context, /*alignment=*/32, /*size=*/13, /*tensor=*/0,
                 /*first_node=*/0, /*last_node=*/3, &allocs[3]);
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  /* This is the expected arena after tensor3 has been allocated.
   * |xxxxxxxxxxxxxxxxx| tensor3
   *             |xxxxx| tensor2
   *       |xxxxx| tensor1
   * |xxxxxxxxxxx| tensor0
   * -------------------
   * |     |     |     |
   * 0     1     2     3
   * ___________________
   */
  ASSERT_TRUE(reallocated);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr0), kTfLiteOk);
  EXPECT_NE(resolved_ptr0, nullptr);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[1], &resolved_ptr1), kTfLiteOk);
  EXPECT_EQ(resolved_ptr1, resolved_ptr0 + 2048);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[2], &resolved_ptr2), kTfLiteOk);
  EXPECT_EQ(resolved_ptr2, resolved_ptr1 + 2048);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[3], &resolved_ptr3), kTfLiteOk);
  EXPECT_EQ(resolved_ptr3, resolved_ptr2 + 2048);

  /* Resetting the arena clears all allocs. If tensor reallocation is required,
   * then the arena should be reset before graph execution.
   */
  arena.ResetAllocs();
  arena.Allocate(&context, /*alignment=*/32, /*size=*/13, /*tensor=*/0,
                 /*first_node=*/0, /*last_node=*/2, &allocs[3]);
  /* This is the expected arena after tensor3 has been allocated.
   * tensors 0, 1, 2 have been deleted.
   * |xxxxxxxxxxxxxxxxx| tensor3
   * -------------------
   * |     |     |     |
   * 0     1     2     3
   * ___________________
   */

  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[3], &resolved_ptr3), kTfLiteOk);
  ASSERT_EQ(allocs[3].offset, 0);
}

TEST(SimpleMemoryArenaTest, TestClearBuffer) {
  TfLiteContext context;
  context.ReportError = ReportError;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[9];

  arena.Allocate(&context, 32, 2047, 0, 0, 2, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 1, 2, &allocs[1]);

  // Should be a no-op.
  ASSERT_EQ(arena.ReleaseBuffer(), kTfLiteOk);

  // Commit and ensure resolved pointers are not null.
  bool reallocated = false;
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  ASSERT_TRUE(reallocated);
  char* resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);
  resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[1], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);

  ASSERT_EQ(arena.ReleaseBuffer(), kTfLiteOk);
  // Base pointer should be null.
  ASSERT_EQ(arena.BasePointer(), 0);

  // Tensors cannot be resolved after ClearBuffer().
  ASSERT_NE(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr), kTfLiteOk);

  // Commit again and ensure resolved pointers are not null.
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  ASSERT_TRUE(reallocated);
  ASSERT_NE(arena.BasePointer(), 0);
  resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);
  resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[1], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);
}

TEST(SimpleMemoryArenaTest, FailedCommitLeavesArenaUncommitted) {
  TfLiteContext context;
  context.ReportError = ReportError;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval alloc;

  ASSERT_EQ(
      arena.Allocate(&context, /*alignment=*/32,
                     /*size=*/std::numeric_limits<size_t>::max(),
                     /*tensor=*/0, /*first_node=*/0, /*last_node=*/0, &alloc),
      kTfLiteOk);

  bool reallocated = true;
  EXPECT_EQ(arena.Commit(&reallocated), kTfLiteError);

  char* resolved_ptr = nullptr;
  EXPECT_NE(arena.ResolveAlloc(&context, alloc, &resolved_ptr), kTfLiteOk);
}

// Test parameterized by whether ClearBuffer() is called before ClearPlan(), or
// vice versa.
class BufferAndPlanClearingTest : public ::testing::Test,
                                  public ::testing::WithParamInterface<bool> {};

TEST_P(BufferAndPlanClearingTest, TestClearBufferAndClearPlan) {
  TfLiteContext context;
  context.ReportError = ReportError;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[9];

  arena.Allocate(&context, 32, 2047, 0, 0, 2, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 1, 2, &allocs[1]);

  bool reallocated = false;
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  ASSERT_TRUE(reallocated);

  if (GetParam()) {
    ASSERT_EQ(arena.ReleaseBuffer(), kTfLiteOk);
    ASSERT_EQ(arena.ClearPlan(), kTfLiteOk);
  } else {
    ASSERT_EQ(arena.ClearPlan(), kTfLiteOk);
    ASSERT_EQ(arena.ReleaseBuffer(), kTfLiteOk);
  }

  // Just committing won't work, allocations need to be made again.
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  // There was no allocation, the buffer has 0 bytes (was released) and the high
  // water mark is 0 (plan was cleared).
  EXPECT_FALSE(reallocated);
  char* resolved_ptr = nullptr;
  ASSERT_NE(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr), kTfLiteOk);

  // Re-allocate tensors & commit.
  arena.Allocate(&context, 32, 2047, 0, 0, 2, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 1, 2, &allocs[1]);
  ASSERT_EQ(arena.Commit(&reallocated), kTfLiteOk);
  ASSERT_TRUE(reallocated);

  // Pointer-resolution now works.
  resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[1], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);
}

INSTANTIATE_TEST_SUITE_P(BufferAndPlanClearingTest, BufferAndPlanClearingTest,
                         ::testing::Values(true, false));

}  // namespace
}  // namespace tflite
