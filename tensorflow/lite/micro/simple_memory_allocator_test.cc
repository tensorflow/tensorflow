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

TF_LITE_MICRO_TEST(TestJustFits) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* result = allocator.AllocateFromTail(arena_size, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);
}

TF_LITE_MICRO_TEST(TestAligned) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* result = allocator.AllocateFromTail(1, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);

  result = allocator.AllocateFromTail(16, 4);
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);
  TF_LITE_MICRO_EXPECT_EQ(0, reinterpret_cast<std::uintptr_t>(result) & 3);
}

TF_LITE_MICRO_TEST(TestMultipleTooLarge) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SimpleMemoryAllocator allocator(micro_test::reporter, arena,
                                          arena_size);

  uint8_t* result = allocator.AllocateFromTail(768, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, result);

  result = allocator.AllocateFromTail(768, 1);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, result);
}

TF_LITE_MICRO_TESTS_END
