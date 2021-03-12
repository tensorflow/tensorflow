/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"

#if CUDA_VERSION >= 10020

#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

using ::stream_executor::gpu::GpuContext;
using ::stream_executor::gpu::GpuDevicePtr;
using ::stream_executor::gpu::GpuDriver;

// Empirically the min allocation granularity.
constexpr size_t k2MiB{2 << 20};

// Creates an allocator with 8 MiB of virtual address space.
std::unique_ptr<GpuVirtualMemAllocator> CreateAllocator() {
  PlatformDeviceId gpu_id(0);
  auto executor =
      DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(), gpu_id)
          .ValueOrDie();
  GpuContext* gpu_context = reinterpret_cast<GpuContext*>(
      executor->implementation()->GpuContextHack());
  return GpuVirtualMemAllocator::Create(
             {}, {}, *gpu_context, gpu_id,
             /*virtual_address_space_size=*/4 * k2MiB, {})
      .ValueOrDie();
}

TEST(GpuVirtualMemAllocatorTest, SimpleAlloc) {
  PlatformDeviceId gpu_id(0);
  auto executor =
      DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(), gpu_id)
          .ValueOrDie();
  GpuContext* gpu_context = reinterpret_cast<GpuContext*>(
      executor->implementation()->GpuContextHack());
  auto allocator = GpuVirtualMemAllocator::Create(
                       {}, {}, *gpu_context, gpu_id,
                       /*virtual_address_space_size=*/4 * k2MiB, {})
                       .ValueOrDie();
  size_t bytes_received;  // Ignored in this test.
  void* gpu_block =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(gpu_block, nullptr);

  constexpr size_t kBufSize{256};
  void* host_mem[2] = {GpuDriver::HostAllocate(gpu_context, kBufSize),
                       GpuDriver::HostAllocate(gpu_context, kBufSize)};
  std::memset(host_mem[0], 'z', kBufSize);
  std::memset(host_mem[1], 0, kBufSize);

  GpuDevicePtr gpu_buf = reinterpret_cast<GpuDevicePtr>(gpu_block) + 2048;
  ASSERT_TRUE(GpuDriver::SynchronousMemcpyH2D(gpu_context, gpu_buf, host_mem[0],
                                              kBufSize)
                  .ok());
  ASSERT_TRUE(GpuDriver::SynchronousMemcpyD2H(gpu_context, host_mem[1], gpu_buf,
                                              kBufSize)
                  .ok());
  for (int i = 0; i < kBufSize; ++i) {
    ASSERT_EQ('z', reinterpret_cast<const char*>(host_mem[1])[i]);
  }
}

TEST(GpuVirtualMemAllocatorTest, AllocPaddedUp) {
  auto allocator = CreateAllocator();
  size_t bytes_received;
  void* gpu_block =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/256, &bytes_received);
  ASSERT_NE(gpu_block, nullptr);
  ASSERT_EQ(bytes_received, k2MiB);
}

TEST(GpuVirtualMemAllocatorTest, AllocsContiguous) {
  auto allocator = CreateAllocator();
  size_t bytes_received;  // Ignored in this test.
  void* first_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(first_alloc, nullptr);
  void* second_alloc = allocator->Alloc(
      /*alignment=*/0, /*num_bytes=*/2 * k2MiB, &bytes_received);
  ASSERT_NE(second_alloc, nullptr);

  ASSERT_EQ(second_alloc, reinterpret_cast<const char*>(first_alloc) + k2MiB);

  void* third_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(third_alloc, nullptr);

  ASSERT_EQ(third_alloc,
            reinterpret_cast<const char*>(second_alloc) + 2 * k2MiB);
}

TEST(GpuVirtualMemAllocator, OverAllocate) {
  auto allocator = CreateAllocator();
  size_t bytes_received;  // Ignored in this test.
  void* first_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(first_alloc, nullptr);
  void* over_alloc = allocator->Alloc(/*alignment=*/0, /*num_bytes=*/4 * k2MiB,
                                      &bytes_received);
  ASSERT_EQ(over_alloc, nullptr);
}

TEST(GpuVirtualMemAllocatorTest, FreeAtEnd) {
  auto allocator = CreateAllocator();
  size_t bytes_received;  // Ignored in this test.
  void* first_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(first_alloc, nullptr);
  void* second_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(second_alloc, nullptr);

  allocator->Free(second_alloc, k2MiB);

  void* re_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_EQ(re_alloc, second_alloc);
}

TEST(GpuVirtualMemAllocatorTest, FreeHole) {
  auto allocator = CreateAllocator();
  size_t bytes_received;  // Ignored in this test.
  void* first_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(first_alloc, nullptr);
  void* second_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(second_alloc, nullptr);

  allocator->Free(first_alloc, k2MiB);

  void* third_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(third_alloc, nullptr);

  // Expect that allocation still happens at the end.
  ASSERT_EQ(third_alloc, reinterpret_cast<const char*>(second_alloc) + k2MiB);
}

TEST(GpuVirtualMemAllocatorTest, FreeRange) {
  auto allocator = CreateAllocator();
  size_t bytes_received;  // Ignored in this test.
  void* first_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(first_alloc, nullptr);
  void* second_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(second_alloc, nullptr);
  void* third_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(third_alloc, nullptr);

  allocator->Free(first_alloc, 3 * k2MiB);

  void* re_alloc =
      allocator->Alloc(/*alignment=*/0, /*num_bytes=*/k2MiB, &bytes_received);
  ASSERT_NE(re_alloc, nullptr);
  ASSERT_EQ(re_alloc, first_alloc);
}

}  // namespace
}  // namespace tensorflow

#endif
