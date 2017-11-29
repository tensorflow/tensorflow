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
#include "tensorflow/contrib/lite/simple_memory_arena.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace {

TEST(SimpleMemoryArenaTest, BasicArenaOperations) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAlloc allocs[6];

  arena.Allocate(&context, 32, 2047, &allocs[0]);
  arena.Allocate(&context, 32, 2047, &allocs[1]);
  arena.Allocate(&context, 32, 2047, &allocs[2]);
  arena.Deallocate(&context, allocs[0]);
  arena.Allocate(&context, 32, 1023, &allocs[3]);
  arena.Allocate(&context, 32, 2047, &allocs[4]);
  arena.Deallocate(&context, allocs[1]);
  arena.Allocate(&context, 32, 1023, &allocs[5]);

  EXPECT_EQ(allocs[0].offset, 0);
  EXPECT_EQ(allocs[1].offset, 2048);
  EXPECT_EQ(allocs[2].offset, 4096);
  EXPECT_EQ(allocs[3].offset, 0);
  EXPECT_EQ(allocs[4].offset, 6144);
  EXPECT_EQ(allocs[5].offset, 1024);
}

TEST(SimpleMemoryArenaTest, TestAfterClear) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAlloc allocs[9];

  arena.Allocate(&context, 32, 2047, &allocs[0]);
  arena.Allocate(&context, 32, 2047, &allocs[1]);
  arena.Allocate(&context, 32, 2047, &allocs[2]);
  arena.Commit(&context);

  EXPECT_EQ(allocs[0].offset, 0);
  EXPECT_EQ(allocs[1].offset, 2048);
  EXPECT_EQ(allocs[2].offset, 4096);

  arena.Clear();

  // Test with smaller allocs.
  arena.Allocate(&context, 32, 1023, &allocs[3]);
  arena.Allocate(&context, 32, 1023, &allocs[4]);
  arena.Allocate(&context, 32, 1023, &allocs[5]);
  arena.Commit(&context);

  EXPECT_EQ(allocs[3].offset, 0);
  EXPECT_EQ(allocs[4].offset, 1024);
  EXPECT_EQ(allocs[5].offset, 2048);

  arena.Clear();

  // Test larger allocs which should require a reallocation.
  arena.Allocate(&context, 32, 4095, &allocs[6]);
  arena.Allocate(&context, 32, 4095, &allocs[7]);
  arena.Allocate(&context, 32, 4095, &allocs[8]);
  arena.Commit(&context);

  EXPECT_EQ(allocs[6].offset, 0);
  EXPECT_EQ(allocs[7].offset, 4096);
  EXPECT_EQ(allocs[8].offset, 8192);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
