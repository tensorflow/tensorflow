/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/profiling/memory_info.h"

#include <memory>
#include <new>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace tflite {
namespace profiling {
namespace memory {

TEST(MemoryUsage, AddAndSub) {
  MemoryUsage mem1, mem2;
  mem1.mem_footprint_kb = 5;
  mem1.total_allocated_bytes = 7000;
  mem1.in_use_allocated_bytes = 2000;

  mem2.mem_footprint_kb = 3;
  mem2.total_allocated_bytes = 7000;
  mem2.in_use_allocated_bytes = 4000;

  const auto add_mem = mem1 + mem2;
  EXPECT_EQ(8, add_mem.mem_footprint_kb);
  EXPECT_EQ(14000, add_mem.total_allocated_bytes);
  EXPECT_EQ(6000, add_mem.in_use_allocated_bytes);

  const auto sub_mem = mem1 - mem2;
  EXPECT_EQ(2, sub_mem.mem_footprint_kb);
  EXPECT_EQ(0, sub_mem.total_allocated_bytes);
  EXPECT_EQ(-2000, sub_mem.in_use_allocated_bytes);
}

TEST(MemoryUsage, GetMemoryUsage) {
  MemoryUsage result;
  EXPECT_EQ(MemoryUsage::kValueNotSet, result.mem_footprint_kb);
  EXPECT_EQ(MemoryUsage::kValueNotSet, result.total_allocated_bytes);
  EXPECT_EQ(MemoryUsage::kValueNotSet, result.in_use_allocated_bytes);

#if defined(__linux__) || defined(__APPLE__)
  // Just allocate some space in heap so that we have some meaningful
  // memory usage to report.
  constexpr int size = 10 * 1024 * 1024;
  std::unique_ptr<unsigned char[]> byte_array(new unsigned char[size]);
  for (int i = 0; i < size; ++i) {
    byte_array[i] = i % 256;
  }

  result = GetMemoryUsage();

  // Use the heap object that we allocated, so that the compiler can't
  // (so easily) optimize it away.
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(byte_array[i], i % 256);
  }

  EXPECT_GE(result.mem_footprint_kb, size / 1024);
  EXPECT_GE(result.total_allocated_bytes, size);
  EXPECT_GE(result.in_use_allocated_bytes, size);
#endif
}

// The main aim of this test is just to exercise the code for
// the ostream operator << and verify that it doesn't crash.
// There's not much that we can usefully assert about the resulting message
// here without making the test too brittle, so we just verify that
// it generates a non-empty message.
TEST(MemoryUsage, OutputMemoryUsageToStream) {
  MemoryUsage memory_usage = GetMemoryUsage();
  std::stringstream stream;
  stream << memory_usage;
  std::string message = stream.str();
  EXPECT_STRNE(message.c_str(), "");
}

TEST(MemoryUsage, IsSupported) {
#if defined(__linux__) || defined(__APPLE__)
  EXPECT_TRUE(MemoryUsage::IsSupported());
#else
  EXPECT_FALSE(MemoryUsage::IsSupported());
#endif
}

}  // namespace memory
}  // namespace profiling
}  // namespace tflite
