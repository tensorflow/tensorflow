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

TF_LITE_MICRO_TEST(TestEnsureHeadSizeSimpleAlignment) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  // First head adjustment
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator.EnsureHeadSize(/*size=*/100, /*alignment=*/1));
  TF_LITE_MICRO_EXPECT(arena + 100 == allocator.GetHead());

  // Second head adjusment is smaller, head size should still be 100.
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator.EnsureHeadSize(/*size=*/10, /*alignment=*/1));
  TF_LITE_MICRO_EXPECT(arena + 100 == allocator.GetHead());

  // Third head adjustment re-increases the head size:
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator.EnsureHeadSize(/*size=*/1000, /*alignment=*/1));
  TF_LITE_MICRO_EXPECT(arena + 1000 == allocator.GetHead());
}

TF_LITE_MICRO_TEST(TestAdjustHeadSizeMisalignment) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  // First head adjustment of 100 bytes (aligned 12):
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator.EnsureHeadSize(/*size=*/100, /*alignment=*/12));

  // Offset alignment of 12 can lead to allocation within 8 byte range of
  // requested bytes based to arena alignment at runtime:
  TF_LITE_MICRO_EXPECT_GE(allocator.GetHead(), arena + 100);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetHead(), arena + 100 + 11);

  // Second head adjusment shrinks the head size (aligned at 12), head size
  // should still be 100:
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator.EnsureHeadSize(/*size=*/10, /*alignment=*/12));
  TF_LITE_MICRO_EXPECT_GE(allocator.GetHead(), arena + 100);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetHead(), arena + 100 + 11);

  // Third head adjustment re-increases the head size (aligned at 12):
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator.EnsureHeadSize(/*size=*/1000, /*alignment=*/12));
  TF_LITE_MICRO_EXPECT_GE(allocator.GetHead(), arena + 1000);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetHead(), arena + 1000 + 11);
}

TF_LITE_MICRO_TEST(TestAdjustHeadSizeMisalignedHandlesCorrectBytesAvailable) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  // First head adjustment of 100 bytes (aligned 12):
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator.EnsureHeadSize(/*size=*/100, /*alignment=*/12));

  // allocator.GetAvailableMemory() should also report the actual amount of
  // memory available based on a requested offset (12):
  size_t aligned_available_bytes =
      allocator.GetAvailableMemory(/*alignment=*/12);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - 100);
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - 100 - 24);

  // Second head adjusment shrinks the head size (aligned at 12), head size
  // should still be 100:
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator.EnsureHeadSize(/*size=*/10, /*alignment=*/12));
  aligned_available_bytes = allocator.GetAvailableMemory(/*alignment=*/12);

  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - 100);
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - 100 - 24);

  // Third head adjustment re-increases the head size (aligned at 12):
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator.EnsureHeadSize(/*size=*/1000, /*alignment=*/12));
  aligned_available_bytes = allocator.GetAvailableMemory(/*alignment=*/12);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - 1000);
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - 1000 - 24);
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

TF_LITE_MICRO_TEST(TestEnsureHeadSizeWithoutResettingTemp) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* temp = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp);

  // Adjustment to head should fail since temp allocation was not followed by a
  // call to ResetTempAllocations().
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, allocator.EnsureHeadSize(100, 1));

  allocator.ResetTempAllocations();

  // Reduce head size back to zero.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, allocator.EnsureHeadSize(0, 1));

  // The most recent head allocation should be in the same location as the
  // original temp allocation pointer.
  TF_LITE_MICRO_EXPECT(temp == allocator.GetHead());
}

TF_LITE_MICRO_TESTS_END
