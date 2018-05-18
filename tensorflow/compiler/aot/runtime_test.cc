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

#include "tensorflow/compiler/aot/runtime.h"

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tfcompile {
namespace runtime {
namespace {

TEST(Runtime, AlignmentValue) {
  // We've chosen 64 byte alignment for the tfcompile runtime to mimic the
  // regular tensorflow allocator, which was chosen to play nicely with Eigen.
  // The tfcompile runtime also has a requirement that comes from the xla
  // generated code, on the relation: buffer_size >= 16 ? 2 * sizeof(void*) : 8
  // So any value that we choose must abide by that constraint as well.
  EXPECT_EQ(kAlign, Allocator::kAllocatorAlignment);
}

TEST(Runtime, AlignedBufferBytes) {
  EXPECT_EQ(aligned_buffer_bytes(nullptr, 0), 0);

  static constexpr intptr_t sizesA[1] = {-1};
  EXPECT_EQ(aligned_buffer_bytes(sizesA, 1), 0);

  static constexpr intptr_t sizesB[1] = {3};
  EXPECT_EQ(aligned_buffer_bytes(sizesB, 1), 64);

  static constexpr intptr_t sizesC[1] = {32};
  EXPECT_EQ(aligned_buffer_bytes(sizesC, 1), 64);

  static constexpr intptr_t sizesD[7] = {1, -1, 32, -1, 64, 2, 3};
  EXPECT_EQ(aligned_buffer_bytes(sizesD, 7), 320);
}

void* add_ptr(void* base, uintptr_t delta) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + delta);
}

// To test MallocContiguousBuffers and FreeContiguous, we just check for
// expected nullptrs, and write to each byte of allocated memory.  We rely on
// the leak checker to tell us if there's an inconsistency between malloc and
// free.  We also check the contiguous property.
TEST(Runtime, MallocFreeContiguousBuffers) {
  // Test empty sizes.
  void* base = MallocContiguousBuffers(nullptr, 0, nullptr, false);
  EXPECT_EQ(base, nullptr);
  FreeContiguous(base);

  // Test non-empty sizes with 0 sum.
  static constexpr intptr_t sizesA[1] = {-1};
  void* bufA[1];
  base = MallocContiguousBuffers(sizesA, 1, bufA, false);
  EXPECT_EQ(base, nullptr);
  EXPECT_EQ(bufA[0], nullptr);
  FreeContiguous(base);

  // Test non-empty sizes with non-0 sum.
  static constexpr intptr_t sizesB[1] = {3};
  void* bufB[1];
  base = MallocContiguousBuffers(sizesB, 1, bufB, false);
  EXPECT_NE(base, nullptr);
  EXPECT_EQ(bufB[0], add_ptr(base, 0));
  char* bufB0_bytes = static_cast<char*>(bufB[0]);
  bufB0_bytes[0] = 'A';
  bufB0_bytes[1] = 'B';
  bufB0_bytes[2] = 'C';
  FreeContiguous(base);

  // Test non-empty sizes with non-0 sum, and annotate_initialized.
  static constexpr intptr_t sizesC[1] = {3};
  void* bufC[1];
  base = MallocContiguousBuffers(sizesC, 1, bufC, true);
  EXPECT_NE(base, nullptr);
  EXPECT_EQ(bufC[0], add_ptr(base, 0));
  char* bufC0_bytes = static_cast<char*>(bufC[0]);
  bufC0_bytes[0] = 'A';
  bufC0_bytes[1] = 'B';
  bufC0_bytes[2] = 'C';
  FreeContiguous(base);

  // Test mixed sizes.
  static constexpr intptr_t sizesD[7] = {1, -1, 32, -1, 64, 2, 3};
  void* bufD[7];
  base = MallocContiguousBuffers(sizesD, 7, bufD, false);
  EXPECT_NE(base, nullptr);
  EXPECT_EQ(bufD[0], add_ptr(base, 0));
  EXPECT_EQ(bufD[1], nullptr);
  EXPECT_EQ(bufD[2], add_ptr(base, 64));
  EXPECT_EQ(bufD[3], nullptr);
  EXPECT_EQ(bufD[4], add_ptr(base, 128));
  EXPECT_EQ(bufD[5], add_ptr(base, 192));
  EXPECT_EQ(bufD[6], add_ptr(base, 256));
  for (int i = 0; i < 7; ++i) {
    const intptr_t size = sizesD[i];
    if (size != -1) {
      char* bufD_bytes = static_cast<char*>(bufD[i]);
      for (size_t j = 0; j < size; ++j) {
        bufD_bytes[j] = 'A' + j;
      }
    }
  }
  FreeContiguous(base);
}

}  // namespace
}  // namespace runtime
}  // namespace tfcompile
}  // namespace tensorflow
