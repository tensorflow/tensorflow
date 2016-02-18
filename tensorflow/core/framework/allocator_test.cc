/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

TEST(CPUAllocatorTest, Simple) {
  Allocator* a = cpu_allocator();
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a->AllocateRaw(1, s);
    ptrs.push_back(raw);
  }
  std::sort(ptrs.begin(), ptrs.end());
  for (size_t i = 0; i < ptrs.size(); i++) {
    if (i > 0) {
      CHECK_NE(ptrs[i], ptrs[i - 1]);  // No dups
    }
    a->DeallocateRaw(ptrs[i]);
  }
  float* t1 = a->Allocate<float>(1024);
  double* t2 = a->Allocate<double>(1048576);
  a->Deallocate(t1, 1024);
  a->Deallocate(t2, 1048576);
}

// Define a struct that we will use to observe behavior in the unit tests
struct TestStruct {
  int x;  // not used just want to make sure sizeof(TestStruct) > 1
};

TEST(CPUAllocatorTest, CheckStructSize) { CHECK_GT(sizeof(TestStruct), 1); }

TEST(CPUAllocatorTest, AllocateOverflowMaxSizeT) {
  Allocator* a = cpu_allocator();

  // The maximum size_t value will definitely overflow.
  size_t count_to_allocate = std::numeric_limits<size_t>::max();
  TestStruct* const test_pointer = a->Allocate<TestStruct>(count_to_allocate);

  CHECK_EQ(test_pointer, reinterpret_cast<TestStruct*>(NULL));
}

TEST(CPUAllocatorTest, AllocateOverflowSmallest) {
  Allocator* a = cpu_allocator();

  // count_to_allocate is the smallest count that will cause overflow.
  const size_t count_to_allocate =
      (std::numeric_limits<size_t>::max() / sizeof(TestStruct)) + 1;
  TestStruct* const test_pointer = a->Allocate<TestStruct>(count_to_allocate);

  CHECK_EQ(test_pointer, reinterpret_cast<TestStruct*>(NULL));
}

TEST(CPUAllocatorTest, Sizes) {
  Allocator* a = cpu_allocator();

  EXPECT_EQ(false, a->TracksAllocationSizes());
}

namespace {

AllocatorAttributes DeviceAllocatorAttribute() {
  AllocatorAttributes attr;
  attr.value |= (0x1 << 24);
  return attr;
}

bool HasDeviceAllocatorAttribute(const AllocatorAttributes& attr) {
  return attr.value & (0x1 << 24);
}

}  // namespace

TEST(CustomAllocatorAttributes, TestSetterAndGetter) {
  AllocatorAttributes attr = DeviceAllocatorAttribute();
  EXPECT_TRUE(HasDeviceAllocatorAttribute(attr));
  EXPECT_FALSE(HasDeviceAllocatorAttribute(AllocatorAttributes()));
}

static void BM_Allocation(int iters) {
  Allocator* a = cpu_allocator();
  // Exercise a few different allocation sizes
  std::vector<int> sizes = {256, 4096, 16384, 524288, 512, 1048576};
  int size_index = 0;

  while (--iters > 0) {
    int bytes = sizes[size_index++ % sizes.size()];
    void* p = a->AllocateRaw(1, bytes);
    a->DeallocateRaw(p);
  }
}
BENCHMARK(BM_Allocation);

}  // namespace tensorflow
