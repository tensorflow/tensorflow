/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/core/arena.h"

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {
namespace {

// Write random data to allocated memory
static void TestMemory(void* mem, int size) {
  // Check that we can memset the entire memory
  memset(mem, 0xaa, size);

  // Do some memory allocation to check that the arena doesn't mess up
  // the internal memory allocator
  char* tmp[100];
  for (size_t i = 0; i < TF_ARRAYSIZE(tmp); i++) {
    tmp[i] = new char[i * i + 1];
  }

  memset(mem, 0xcc, size);

  // Free up the allocated memory;
  for (size_t i = 0; i < TF_ARRAYSIZE(tmp); i++) {
    delete[] tmp[i];
  }

  // Check that we can memset the entire memory
  memset(mem, 0xee, size);
}

TEST(ArenaTest, TestBasicArena) {
  Arena a(1024);
  char* memory = a.Alloc(100);
  ASSERT_NE(memory, nullptr);
  TestMemory(memory, 100);

  // Allocate again
  memory = a.Alloc(100);
  ASSERT_NE(memory, nullptr);
  TestMemory(memory, 100);
}

TEST(ArenaTest, TestAlignment) {
  Arena a(1024);
  char* byte0 = a.Alloc(1);
  char* alloc_aligned8 = a.AllocAligned(17, 8);
  EXPECT_EQ(alloc_aligned8 - byte0, 8);
  char* alloc_aligned8_b = a.AllocAligned(8, 8);
  EXPECT_EQ(alloc_aligned8_b - alloc_aligned8, 24);
  char* alloc_aligned8_c = a.AllocAligned(16, 8);
  EXPECT_EQ(alloc_aligned8_c - alloc_aligned8_b, 8);
  char* alloc_aligned8_d = a.AllocAligned(8, 1);
  EXPECT_EQ(alloc_aligned8_d - alloc_aligned8_c, 16);
}

TEST(ArenaTest, TestVariousArenaSizes) {
  {
    Arena a(1024);

    // Allocate blocksize
    char* memory = a.Alloc(1024);
    ASSERT_NE(memory, nullptr);
    TestMemory(memory, 1024);

    // Allocate another blocksize
    char* memory2 = a.Alloc(1024);
    ASSERT_NE(memory2, nullptr);
    TestMemory(memory2, 1024);
  }

  // Allocate an arena and allocate two blocks
  // that together exceed a block size
  {
    Arena a(1024);

    //
    char* memory = a.Alloc(768);
    ASSERT_NE(memory, nullptr);
    TestMemory(memory, 768);

    // Allocate another blocksize
    char* memory2 = a.Alloc(768);
    ASSERT_NE(memory2, nullptr);
    TestMemory(memory2, 768);
  }

  // Allocate larger than a blocksize
  {
    Arena a(1024);

    char* memory = a.Alloc(10240);
    ASSERT_NE(memory, nullptr);
    TestMemory(memory, 10240);

    // Allocate another blocksize
    char* memory2 = a.Alloc(1234);
    ASSERT_NE(memory2, nullptr);
    TestMemory(memory2, 1234);
  }
}

}  // namespace
}  // namespace core
}  // namespace tensorflow
