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

#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace profiling {
namespace memory {

TEST(MemoryUsage, AddAndSub) {
  MemoryUsage mem1, mem2;
  mem1.max_rss_kb = 5;
  mem1.total_allocated_bytes = 7000;
  mem1.in_use_allocated_bytes = 2000;

  mem2.max_rss_kb = 3;
  mem2.total_allocated_bytes = 7000;
  mem2.in_use_allocated_bytes = 4000;

  const auto add_mem = mem1 + mem2;
  EXPECT_EQ(8, add_mem.max_rss_kb);
  EXPECT_EQ(14000, add_mem.total_allocated_bytes);
  EXPECT_EQ(6000, add_mem.in_use_allocated_bytes);

  const auto sub_mem = mem1 - mem2;
  EXPECT_EQ(2, sub_mem.max_rss_kb);
  EXPECT_EQ(0, sub_mem.total_allocated_bytes);
  EXPECT_EQ(-2000, sub_mem.in_use_allocated_bytes);
}

TEST(MemoryUsage, GetMemoryUsage) {
  MemoryUsage result;
  EXPECT_EQ(MemoryUsage::kValueNotSet, result.max_rss_kb);
  EXPECT_EQ(MemoryUsage::kValueNotSet, result.total_allocated_bytes);
  EXPECT_EQ(MemoryUsage::kValueNotSet, result.in_use_allocated_bytes);

#ifdef __linux__
  // Just allocate some space in heap so that we could meaningful memory usage
  // report.
  std::unique_ptr<int[]> int_array(new int[1204]);
  for (int i = 0; i < 1024; ++i) int_array[i] = i;
  result = GetMemoryUsage();

  // As the getrusage call may fail, we might not be able to get max_rss_kb.
  EXPECT_NE(MemoryUsage::kValueNotSet, result.total_allocated_bytes);
#endif
}

}  // namespace memory
}  // namespace profiling
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
