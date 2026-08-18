/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/host_offloading/gpu_host_offloading_allocator.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "xla/core/host_offloading/host_offloading_allocator.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/status_matchers.h"

namespace xla::gpu {

namespace {

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

TEST(GpuHostOffloadingAllocatorTest, AllocateTransferBuffer) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  auto allocator = CreateGpuHostOffloadingAllocator(stream_executor);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer, allocator->AllocateTransferBuffer(1024));
  EXPECT_EQ(buffer->size_bytes(), 1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto memory_type,
      stream_executor->GetPointerMemorySpace(buffer->untyped_data()));
  EXPECT_EQ(memory_type, stream_executor::MemorySpace::kHost);
}

TEST(GpuHostOffloadingAllocatorTest, AllocateStagingBuffer) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  auto allocator = CreateGpuHostOffloadingAllocator(stream_executor);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer, allocator->AllocateStagingBuffer(1024));
  EXPECT_EQ(buffer->size_bytes(), 1024);

  auto memory_type_or_status =
      stream_executor->GetPointerMemorySpace(buffer->untyped_data());

  // Staging buffers are allocated with operators new/delete, so they will be
  // considered invalid arguments to stream calls.
  EXPECT_TRUE(absl::IsInternal(memory_type_or_status.status()));
}

TEST(GpuHostOffloadingAllocatorTest, LargeTransferBufferCanBeCopied) {
  constexpr size_t kSmallBufferSize = 32 << 10;
  constexpr int kNumSmallBuffers = 32;
  constexpr size_t kLargeBufferSize = 4 << 20;

  se::StreamExecutor* executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());
  std::unique_ptr<HostOffloadingAllocator> allocator =
      CreateGpuHostOffloadingAllocator(executor);

  // Fill 1 MiB of BFC's initial 2 MiB region, leaving a free tail. With
  // suballocator coalescing enabled, BFC can combine that tail with the next
  // 4 MiB HostMemoryAllocate region and return a large transfer buffer spanning
  // two independently registered pinned host allocations.
  std::vector<std::unique_ptr<HostOffloadingAllocator::Buffer>> small_buffers;
  small_buffers.reserve(kNumSmallBuffers);
  for (int i = 0; i < kNumSmallBuffers; ++i) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HostOffloadingAllocator::Buffer> buffer,
        allocator->AllocateTransferBuffer(kSmallBufferSize));
    small_buffers.push_back(std::move(buffer));
  }

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostOffloadingAllocator::Buffer> large_buffer,
      allocator->AllocateTransferBuffer(kLargeBufferSize));
  se::DeviceAddressBase device_buffer = executor->Allocate(kLargeBufferSize);
  ASSERT_FALSE(device_buffer.is_null());

  ASSERT_OK(stream->Memcpy(&device_buffer, large_buffer->untyped_data(),
                           kLargeBufferSize));
  ASSERT_OK(stream->Memcpy(large_buffer->untyped_data(), device_buffer,
                           kLargeBufferSize));
  ASSERT_OK(stream->BlockHostUntilDone());

  executor->Deallocate(&device_buffer);
}

}  // namespace

}  // namespace xla::gpu
