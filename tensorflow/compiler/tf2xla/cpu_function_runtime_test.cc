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

#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::xla::cpu_function_runtime::BufferInfo;

TEST(XlaCompiledCpuFunctionTest, AlignmentValue) {
  // We've chosen 64 byte alignment for the tfcompile runtime to mimic the
  // regular tensorflow allocator, which was chosen to play nicely with Eigen.
  // The tfcompile runtime also has a requirement that comes from the xla
  // generated code, on the relation: buffer_size >= 16 ? 2 * sizeof(void*) : 8
  // So any value that we choose must abide by that constraint as well.
  EXPECT_EQ(xla::cpu_function_runtime::Align(), Allocator::kAllocatorAlignment);
  EXPECT_LE(xla::cpu_function_runtime::MinAlign(),
            Allocator::kAllocatorAlignment);
}

std::vector<BufferInfo> SizesToBufferInfos(const intptr_t* sizes, size_t n) {
  std::vector<BufferInfo> buffer_infos;
  std::transform(sizes, sizes + n, std::back_inserter(buffer_infos),
                 [&](intptr_t size) {
                   if (size == -1) {
                     // Use a dummy on-stack buffer allocation to indicat the
                     // the current slot does not need an allocation.
                     int64_t on_stack_buffer_size = 4;
                     return BufferInfo::MakeOnStackBuffer(on_stack_buffer_size);
                   }
                   return BufferInfo::MakeTempBuffer(size);
                 });
  return buffer_infos;
}

// Simple wrappers to make writing tests more ergonomic.

size_t AlignedBufferBytesFromSizes(const intptr_t* sizes, size_t n) {
  std::vector<BufferInfo> buffer_infos = SizesToBufferInfos(sizes, n);
  return AlignedBufferBytes(buffer_infos.data(), n,
                            /*allocate_entry_params=*/false);
}

void* MallocContiguousBuffersFromSizes(const intptr_t* sizes, size_t n,
                                       void** bufs, bool annotate_initialized) {
  std::vector<BufferInfo> buffer_infos = SizesToBufferInfos(sizes, n);
  return MallocContiguousBuffers(buffer_infos.data(), n,
                                 /*allocate_entry_params=*/false, bufs,
                                 annotate_initialized);
}

TEST(XlaCompiledCpuFunctionTest, AlignedBufferBytes) {
  EXPECT_EQ(AlignedBufferBytesFromSizes(nullptr, 0), 0);

  static constexpr intptr_t sizesA[1] = {-1};
  EXPECT_EQ(AlignedBufferBytesFromSizes(sizesA, 1), 0);

  static constexpr intptr_t sizesB[1] = {3};
  EXPECT_EQ(AlignedBufferBytesFromSizes(sizesB, 1), 64);

  static constexpr intptr_t sizesC[1] = {32};
  EXPECT_EQ(AlignedBufferBytesFromSizes(sizesC, 1), 64);

  static constexpr intptr_t sizesD[7] = {1, -1, 32, -1, 64, 2, 3};
  EXPECT_EQ(AlignedBufferBytesFromSizes(sizesD, 7), 320);
}

void* add_ptr(void* base, uintptr_t delta) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + delta);
}

// To test MallocContiguousBuffers and FreeContiguous, we just check for
// expected nullptrs, and write to each byte of allocated memory.  We rely on
// the leak checker to tell us if there's an inconsistency between malloc and
// free.  We also check the contiguous property.
TEST(XlaCompiledCpuFunctionTest, MallocFreeContiguousBuffers) {
  // Test empty sizes.
  void* base = MallocContiguousBuffersFromSizes(nullptr, 0, nullptr, false);
  EXPECT_EQ(base, nullptr);
  xla::cpu_function_runtime::FreeContiguous(base);

  // Test non-empty sizes with 0 sum.
  static constexpr intptr_t sizesA[1] = {-1};
  void* bufA[1];
  base = MallocContiguousBuffersFromSizes(sizesA, 1, bufA, false);
  EXPECT_EQ(base, nullptr);
  EXPECT_EQ(bufA[0], nullptr);
  xla::cpu_function_runtime::FreeContiguous(base);

  // Test non-empty sizes with non-0 sum.
  static constexpr intptr_t sizesB[1] = {3};
  void* bufB[1];
  base = MallocContiguousBuffersFromSizes(sizesB, 1, bufB, false);
  EXPECT_NE(base, nullptr);
  EXPECT_EQ(bufB[0], add_ptr(base, 0));
  char* bufB0_bytes = static_cast<char*>(bufB[0]);
  bufB0_bytes[0] = 'A';
  bufB0_bytes[1] = 'B';
  bufB0_bytes[2] = 'C';
  xla::cpu_function_runtime::FreeContiguous(base);

  // Test non-empty sizes with non-0 sum, and annotate_initialized.
  static constexpr intptr_t sizesC[1] = {3};
  void* bufC[1];
  base = MallocContiguousBuffersFromSizes(sizesC, 1, bufC, true);
  EXPECT_NE(base, nullptr);
  EXPECT_EQ(bufC[0], add_ptr(base, 0));
  char* bufC0_bytes = static_cast<char*>(bufC[0]);
  bufC0_bytes[0] = 'A';
  bufC0_bytes[1] = 'B';
  bufC0_bytes[2] = 'C';
  xla::cpu_function_runtime::FreeContiguous(base);

  // Test mixed sizes.
  static constexpr intptr_t sizesD[7] = {1, -1, 32, -1, 64, 2, 3};
  void* bufD[7];
  base = MallocContiguousBuffersFromSizes(sizesD, 7, bufD, false);
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
  xla::cpu_function_runtime::FreeContiguous(base);
}

void CheckRoundTripIsOk(const BufferInfo& buffer_info) {
  BufferInfo round_trip(buffer_info.EncodeOld());
  ASSERT_EQ(round_trip, buffer_info);
}

TEST(XlaCompiledCpuFunctionTest, BufferInfoTest) {
  CheckRoundTripIsOk(BufferInfo::MakeTempBuffer(0));
  CheckRoundTripIsOk(BufferInfo::MakeTempBuffer(4));
  CheckRoundTripIsOk(BufferInfo::MakeOnStackBuffer(0));
  CheckRoundTripIsOk(BufferInfo::MakeOnStackBuffer(4));
  CheckRoundTripIsOk(BufferInfo::MakeConstant(0));
  CheckRoundTripIsOk(BufferInfo::MakeConstant(4));
  CheckRoundTripIsOk(
      BufferInfo::MakeEntryParameter(/*size=*/0, /*param_number=*/4));
  CheckRoundTripIsOk(
      BufferInfo::MakeEntryParameter(/*size=*/4, /*param_number=*/0));
}

}  // namespace
}  // namespace tensorflow
