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

#include "xla/stream_executor/cuda/cuda_host_allocator.h"

#include <cstdint>
#include <cstring>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_device_allocator.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/numa.h"

namespace stream_executor::gpu {
namespace {

class CudaHostAllocatorTest : public ::testing::TestWithParam<int32_t> {};

TEST_P(CudaHostAllocatorTest, AllocateAndFree) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaHostAllocator allocator(executor, GetParam());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(1024));
  ASSERT_NE(allocation, nullptr);
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);

  // Verify the host memory is accessible.
  std::memset(allocation->address().opaque(), 0xAB, 1024);
}

TEST_P(CudaHostAllocatorTest, AllocateZeroBytes) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaHostAllocator allocator(executor, GetParam());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(0));
  ASSERT_NE(allocation, nullptr);
  EXPECT_EQ(allocation->address().opaque(), nullptr);
}

TEST_P(CudaHostAllocatorTest, MemcpyRoundTrip) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());

  CudaHostAllocator host_allocator(executor, GetParam());
  CudaDeviceAllocator device_allocator(executor);

  constexpr int kSize = 1024;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> host_alloc,
                       host_allocator.Allocate(kSize));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> device_alloc,
                       device_allocator.Allocate(kSize));

  // Create a span over pinned host memory and write a pattern.
  absl::Span<uint8_t> host_span(
      static_cast<uint8_t*>(host_alloc->address().opaque()), kSize);
  for (int i = 0; i < kSize; i++) {
    host_span[i] = static_cast<uint8_t>(i);
  }

  // Copy pinned host memory to device.
  DeviceAddress<uint8_t> device_addr(device_alloc->address());
  ASSERT_OK(
      stream->MemcpyH2D(absl::Span<const uint8_t>(host_span), &device_addr));

  // Zero the host buffer and copy back from device.
  std::memset(host_span.data(), 0, kSize);
  ASSERT_OK(stream->MemcpyD2H(device_addr, host_span));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Verify the data roundtripped correctly.
  for (int i = 0; i < kSize; i++) {
    EXPECT_EQ(host_span[i], static_cast<uint8_t>(i))
        << "mismatch at byte " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(NumaAffinity, CudaHostAllocatorTest,
                         ::testing::Values(tsl::port::kNUMANoAffinity, 0),
                         [](const auto& info) {
                           return info.param == tsl::port::kNUMANoAffinity
                                      ? "NoAffinity"
                                      : "Numa0";
                         });

}  // namespace
}  // namespace stream_executor::gpu
