/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/allocator.h"

#include <gtest/gtest.h>

namespace ruy {
namespace {

TEST(AllocatorTest, ReturnsValidMemory) {
  Allocator allocator;
  int *p;
  allocator.Allocate(1, &p);
  ASSERT_NE(p, nullptr);

  // If this is bogus memory, ASan will cause this test to fail.
  *p = 42;

  allocator.FreeAll();
}

TEST(AllocatorTest, NoLeak) {
  Allocator allocator;
  // Allocate and free some ridiculously large total amount of memory, so
  // that a leak will hopefully cause some sort of resource exhaustion.
  //
  // Despite the large number of allocations, this test is actually quite
  // fast, since our fast-path allocation logic is very fast.
  constexpr int kNumAllocations = 100 * 1024;
  constexpr int kAllocationSize = 1024 * 1024;
  for (int i = 0; i < kNumAllocations; i++) {
    char *p;
    allocator.Allocate(kAllocationSize, &p);
    allocator.FreeAll();
  }
}

TEST(AllocatorTest, IncreasingSizes) {
  Allocator allocator;
  // Allocate sizes that increase by small amounts across FreeAll calls.
  for (int i = 1; i < 100 * 1024; i++) {
    char *p;
    allocator.Allocate(i, &p);
    allocator.FreeAll();
  }
}

TEST(AllocatorTest, ManySmallAllocations) {
  Allocator allocator;
  // Allocate many small allocations between FreeAll calls.
  for (int i = 0; i < 10 * 1024; i += 100) {
    for (int j = 0; j < i; j++) {
      char *p;
      allocator.Allocate(1, &p);
    }
    allocator.FreeAll();
  }
}

TEST(AllocatorTest, DestructorHandlesMainBumpPtr) {
  // This is a white-box test.
  Allocator allocator;
  allocator.AllocateBytes(1);
  allocator.FreeAll();
  // After the call to FreeAll, the allocator will consolidate all of the memory
  // into the main bump-ptr allocator's block, which we then expect to be freed
  // in the destructor.
  //
  // We have no test assertions -- we primarily expect that this trigger a leak
  // checker and cause the test to fail.
}

TEST(AllocatorTest, DestructorHandlesFallbackBlocks) {
  // This is a white-box test.
  Allocator allocator;
  // Since we just created the allocator, this will allocate a fallback block,
  // which we then expect to be freed in the destructor.
  //
  // We have no test assertions -- we primarily expect that this trigger a leak
  // checker and cause the test to fail.
  allocator.AllocateBytes(1);
}

}  // namespace
}  // namespace ruy

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
