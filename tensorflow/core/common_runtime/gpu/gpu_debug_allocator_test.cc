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

#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

TEST(GPUDebugAllocatorTest, OverwriteDetection_None) {
  const PlatformGpuId platform_gpu_id(0);
  GPUDebugAllocator a(new GPUBFCAllocator(platform_gpu_id, 1 << 30, ""),
                      platform_gpu_id);
  auto stream_exec =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();

  for (int s : {8}) {
    std::vector<int64> cpu_array(s);
    memset(&cpu_array[0], 0, cpu_array.size() * sizeof(int64));
    int64* gpu_array = a.Allocate<int64>(cpu_array.size());
    se::DeviceMemory<int64> gpu_array_ptr{se::DeviceMemoryBase{gpu_array}};
    ASSERT_TRUE(stream_exec->SynchronousMemcpy(&gpu_array_ptr, &cpu_array[0],
                                               s * sizeof(int64)));
    EXPECT_TRUE(a.CheckHeader(gpu_array));
    EXPECT_TRUE(a.CheckFooter(gpu_array));

    // Confirm no error on free.
    a.DeallocateRaw(gpu_array);
  }
}

TEST(GPUDebugAllocatorTest, OverwriteDetection_Header) {
  for (int s : {8, 211}) {
    EXPECT_DEATH(
        {
          const PlatformGpuId platform_gpu_id(0);
          GPUDebugAllocator a(new GPUBFCAllocator(platform_gpu_id, 1 << 30, ""),
                              platform_gpu_id);
          auto stream_exec =
              GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();

          std::vector<int64> cpu_array(s);
          memset(&cpu_array[0], 0, cpu_array.size() * sizeof(int64));
          int64* gpu_array = a.Allocate<int64>(cpu_array.size());

          se::DeviceMemory<int64> gpu_array_ptr{
              se::DeviceMemoryBase{gpu_array}};
          ASSERT_TRUE(stream_exec->SynchronousMemcpy(
              &gpu_array_ptr, &cpu_array[0], cpu_array.size() * sizeof(int64)));

          se::DeviceMemory<int64> gpu_hdr_ptr{
              se::DeviceMemoryBase{gpu_array - 1}};
          // Clobber first word of the header.
          float pi = 3.1417;
          ASSERT_TRUE(
              stream_exec->SynchronousMemcpy(&gpu_hdr_ptr, &pi, sizeof(float)));

          // Expect error on free.
          a.DeallocateRaw(gpu_array);
        },
        "");
  }
}

TEST(GPUDebugAllocatorTest, OverwriteDetection_Footer) {
  for (int s : {8, 22}) {
    EXPECT_DEATH(
        {
          const PlatformGpuId platform_gpu_id(0);
          GPUDebugAllocator a(new GPUBFCAllocator(platform_gpu_id, 1 << 30, ""),
                              platform_gpu_id);
          auto stream_exec =
              GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();

          std::vector<int64> cpu_array(s);
          memset(&cpu_array[0], 0, cpu_array.size() * sizeof(int64));
          int64* gpu_array = a.Allocate<int64>(cpu_array.size());

          se::DeviceMemory<int64> gpu_array_ptr{
              se::DeviceMemoryBase{gpu_array}};
          ASSERT_TRUE(stream_exec->SynchronousMemcpy(
              &gpu_array_ptr, &cpu_array[0], cpu_array.size() * sizeof(int64)));

          // Clobber word of the footer.
          se::DeviceMemory<int64> gpu_ftr_ptr{
              se::DeviceMemoryBase{gpu_array + s}};
          float pi = 3.1417;
          ASSERT_TRUE(
              stream_exec->SynchronousMemcpy(&gpu_ftr_ptr, &pi, sizeof(float)));

          // Expect error on free.
          a.DeallocateRaw(gpu_array);
        },
        "");
  }
}

TEST(GPUDebugAllocatorTest, ResetToNan) {
  const PlatformGpuId platform_gpu_id(0);
  GPUNanResetAllocator a(new GPUBFCAllocator(platform_gpu_id, 1 << 30, ""),
                         platform_gpu_id);
  auto stream_exec =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();

  std::vector<float> cpu_array(1024);
  std::vector<float> cpu_array_result(1024);

  // Allocate 1024 floats
  float* gpu_array = a.Allocate<float>(cpu_array.size());
  se::DeviceMemory<float> gpu_array_ptr{se::DeviceMemoryBase{gpu_array}};
  ASSERT_TRUE(stream_exec->SynchronousMemcpy(&cpu_array[0], gpu_array_ptr,
                                             cpu_array.size() * sizeof(float)));
  for (float f : cpu_array) {
    ASSERT_FALSE(std::isfinite(f));
  }

  // Set one of the fields to 1.0.
  cpu_array[0] = 1.0;
  ASSERT_TRUE(stream_exec->SynchronousMemcpy(&gpu_array_ptr, &cpu_array[0],
                                             cpu_array.size() * sizeof(float)));
  // Copy the data back and verify.
  ASSERT_TRUE(
      stream_exec->SynchronousMemcpy(&cpu_array_result[0], gpu_array_ptr,
                                     cpu_array_result.size() * sizeof(float)));
  ASSERT_EQ(1.0, cpu_array_result[0]);

  // Free the array
  a.DeallocateRaw(gpu_array);

  // All values should be reset to nan.
  ASSERT_TRUE(
      stream_exec->SynchronousMemcpy(&cpu_array_result[0], gpu_array_ptr,
                                     cpu_array_result.size() * sizeof(float)));
  for (float f : cpu_array_result) {
    ASSERT_FALSE(std::isfinite(f));
  }
}

TEST(GPUDebugAllocatorTest, ResetToNanWithHeaderFooter) {
  const PlatformGpuId platform_gpu_id(0);
  // NaN reset must be the outer-most allocator.
  GPUNanResetAllocator a(
      new GPUDebugAllocator(new GPUBFCAllocator(platform_gpu_id, 1 << 30, ""),
                            platform_gpu_id),
      platform_gpu_id);
  auto stream_exec =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();

  std::vector<float> cpu_array(1024);
  std::vector<float> cpu_array_result(1024);

  // Allocate 1024 floats
  float* gpu_array = a.Allocate<float>(cpu_array.size());
  se::DeviceMemory<float> gpu_array_ptr{se::DeviceMemoryBase{gpu_array}};
  ASSERT_TRUE(stream_exec->SynchronousMemcpy(&cpu_array[0], gpu_array_ptr,
                                             cpu_array.size() * sizeof(float)));
  for (float f : cpu_array) {
    ASSERT_FALSE(std::isfinite(f));
  }

  // Set one of the fields to 1.0.
  cpu_array[0] = 1.0;
  ASSERT_TRUE(stream_exec->SynchronousMemcpy(&gpu_array_ptr, &cpu_array[0],
                                             cpu_array.size() * sizeof(float)));
  // Copy the data back and verify.
  ASSERT_TRUE(
      stream_exec->SynchronousMemcpy(&cpu_array_result[0], gpu_array_ptr,
                                     cpu_array_result.size() * sizeof(float)));
  ASSERT_EQ(1.0, cpu_array_result[0]);

  // Free the array
  a.DeallocateRaw(gpu_array);

  // All values should be reset to nan.
  ASSERT_TRUE(
      stream_exec->SynchronousMemcpy(&cpu_array_result[0], gpu_array_ptr,
                                     cpu_array_result.size() * sizeof(float)));
  for (float f : cpu_array_result) {
    ASSERT_FALSE(std::isfinite(f));
  }
}

TEST(GPUDebugAllocatorTest, TracksSizes) {
  const PlatformGpuId platform_gpu_id(0);
  GPUDebugAllocator a(new GPUBFCAllocator(platform_gpu_id, 1 << 30, ""),
                      platform_gpu_id);
  EXPECT_EQ(true, a.TracksAllocationSizes());
}

TEST(GPUDebugAllocatorTest, AllocatedVsRequested) {
  const PlatformGpuId platform_gpu_id(0);
  GPUNanResetAllocator a(
      new GPUDebugAllocator(new GPUBFCAllocator(platform_gpu_id, 1 << 30, ""),
                            platform_gpu_id),
      platform_gpu_id);
  float* t1 = a.Allocate<float>(1);
  EXPECT_EQ(4, a.RequestedSize(t1));
  EXPECT_EQ(256, a.AllocatedSize(t1));
  a.DeallocateRaw(t1);
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
