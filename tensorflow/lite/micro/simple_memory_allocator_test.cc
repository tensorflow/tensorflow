/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/simple_memory_allocator.h"

#include <cstdint>

#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestAdjustHead) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  // First allocation from head.
  {
    uint8_t* result = allocator.AdjustHead(100, 1);
    TF_LITE_MICRO_EXPECT(arena == result);
    TF_LITE_MICRO_EXPECT(arena + 100 == allocator.GetHead());
  }
  // Second allocation doesn't require as much space so head pointer didn't
  // move.
  {
    uint8_t* result = allocator.AdjustHead(10, 1);
    TF_LITE_MICRO_EXPECT(arena == result);
    TF_LITE_MICRO_EXPECT(arena + 100 == allocator.GetHead());
  }
  // Third allocation increase head memory usage.
  {
    uint8_t* result = allocator.AdjustHead(1000, 1);
    TF_LITE_MICRO_EXPECT(arena == result);
    TF_LITE_MICRO_EXPECT(arena + 1000 == allocator.GetHead());
  }
}

TF_LITE_MICRO_TEST(TestJustFits) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* result = allocator.AllocateFromTail(arena_size, 1);
  TF_LITE_MICRO_EXPECT(nullptr != result);
}

TF_LITE_MICRO_TEST(TestAligned) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* result = allocator.AllocateFromTail(1, 1);
  TF_LITE_MICRO_EXPECT(nullptr != result);

  result = allocator.AllocateFromTail(16, 4);
  TF_LITE_MICRO_EXPECT(nullptr != result);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          reinterpret_cast<std::uintptr_t>(result) & 3);
}

TF_LITE_MICRO_TEST(TestMultipleTooLarge) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* result = allocator.AllocateFromTail(768, 1);
  TF_LITE_MICRO_EXPECT(nullptr != result);

  result = allocator.AllocateFromTail(768, 1);
  TF_LITE_MICRO_EXPECT(nullptr == result);
}

TF_LITE_MICRO_TEST(TestTempAllocations) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* temp1 = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp1);

  uint8_t* temp2 = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp2);

  // Expect that the next micro allocation is 100 bytes away from each other.
  TF_LITE_MICRO_EXPECT_EQ(temp2 - temp1, 100);
}

TF_LITE_MICRO_TEST(TestResetTempAllocations) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* temp1 = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp1);

  allocator.ResetTempAllocations();

  uint8_t* temp2 = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp2);

  // Reset temp allocations should have the same start address:
  TF_LITE_MICRO_EXPECT_EQ(temp2 - temp1, 0);
}

TF_LITE_MICRO_TEST(TestAllocateHeadWithoutResettingTemp) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* temp = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp);

  // Allocation should be null since temp allocation was not followed by a call
  // to ResetTempAllocations().
  uint8_t* head = allocator.AdjustHead(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr == head);

  allocator.ResetTempAllocations();

  head = allocator.AdjustHead(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != head);

  // The most recent head allocation should be in the same location as the
  // original temp allocation pointer.
  TF_LITE_MICRO_EXPECT(temp == head);
}

// TODO(b/161171251): Add more coverage to this test - specifically around -1
// alignments and other odd allocation requests.

TF_LITE_MICRO_TESTS_END
