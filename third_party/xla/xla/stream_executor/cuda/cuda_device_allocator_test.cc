/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_device_allocator.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {
namespace {

TEST(CudaDeviceAllocatorTest, AllocateAndFree) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaDeviceAllocator allocator(executor);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(1024));
  ASSERT_NE(allocation, nullptr);
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
}

TEST(CudaDeviceAllocatorTest, AllocateZeroBytes) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaDeviceAllocator allocator(executor);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(0));
  ASSERT_NE(allocation, nullptr);
  EXPECT_EQ(allocation->address().opaque(), nullptr);
}

TEST(CudaDeviceAllocatorTest, MemcpyRoundTrip) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());

  CudaDeviceAllocator allocator(executor);

  constexpr int kSize = 1024;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(kSize));

  // Write a pattern to host buffer and copy to device.
  std::vector<uint8_t> host_src(kSize);
  for (int i = 0; i < kSize; i++) {
    host_src[i] = static_cast<uint8_t>(i);
  }

  DeviceAddress<uint8_t> addr(allocation->address());
  ASSERT_OK(stream->MemcpyH2D(absl::MakeConstSpan(host_src), &addr));

  // Copy back from device to host.
  std::vector<uint8_t> host_dst(kSize, 0);
  ASSERT_OK(stream->MemcpyD2H(addr, absl::MakeSpan(host_dst)));
  ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_EQ(host_src, host_dst);
}

}  // namespace
}  // namespace stream_executor::gpu
