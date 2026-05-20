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

#include "xla/stream_executor/cuda/cuda_vmm_allocator.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {
namespace {

CudaVmmAllocator::Options MakeTestOptions(bool enable_rdma) {
  CudaVmmAllocator::Options options;
  options.enable_rdma = enable_rdma;
  return options;
}

class CudaVmmAllocatorTest : public ::testing::TestWithParam<bool> {};

TEST_P(CudaVmmAllocatorTest, AllocateAndFree) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaVmmAllocator allocator(executor, MakeTestOptions(GetParam()));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(1024));
  ASSERT_NE(allocation, nullptr);
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_GE(allocation->address().size(), 1024);
}

TEST_P(CudaVmmAllocatorTest, AllocateZeroBytes) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaVmmAllocator allocator(executor, MakeTestOptions(GetParam()));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(0));
  ASSERT_NE(allocation, nullptr);
  EXPECT_EQ(allocation->address().opaque(), nullptr);
}

TEST_P(CudaVmmAllocatorTest, MemcpyRoundTrip) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());

  CudaVmmAllocator allocator(executor, MakeTestOptions(GetParam()));

  constexpr int kSize = 1024;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(kSize));

  // Write a pattern to host buffer and copy to device.
  std::vector<uint8_t> host_src(kSize);
  for (int i = 0; i < kSize; i++) {
    host_src[i] = static_cast<uint8_t>(i);
  }

  // Use a DeviceAddress sized to kSize (not the padded allocation size) so
  // the memcpy transfers exactly the bytes we care about.
  DeviceAddress<uint8_t> addr(
      DeviceAddressBase(allocation->address().opaque(), kSize));
  ASSERT_OK(stream->MemcpyH2D(absl::MakeConstSpan(host_src), &addr));

  // Copy back from device to host.
  std::vector<uint8_t> host_dst(kSize, 0);
  ASSERT_OK(stream->MemcpyD2H(addr, absl::MakeSpan(host_dst)));
  ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_EQ(host_src, host_dst);
}

TEST_P(CudaVmmAllocatorTest, PeerAccessEnabled) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaVmmAllocator::Options options = MakeTestOptions(GetParam());
  options.enable_peer_access = true;
  CudaVmmAllocator allocator(executor, options);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(4096));
  ASSERT_NE(allocation, nullptr);
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_GE(allocation->address().size(), 4096);
}

TEST_P(CudaVmmAllocatorTest, HopperNoWarningCheck) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaComputeCapability cc =
      executor->GetDeviceDescription().cuda_compute_capability();
  if (!cc.IsAtLeastHopper()) {
    GTEST_SKIP() << "Test only runs on H100+";
  }

  auto* cuda_executor = static_cast<CudaExecutor*>(executor);

  CudaVmmAllocator::Options options = MakeTestOptions(GetParam());
  options.enable_fabric_handle = cuda_executor->is_fabric_supported();
  options.enable_posix_fd_handle = true;

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, ::testing::_,
                       ::testing::HasSubstr("FABRIC+POSIX_FD")))
      .Times(0);
  log.StartCapturingLogs();

  CudaVmmAllocator allocator(executor, options);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(1024));
  EXPECT_NE(allocation, nullptr);

  log.StopCapturingLogs();
}

INSTANTIATE_TEST_SUITE_P(RdmaSupport, CudaVmmAllocatorTest, ::testing::Bool(),
                         [](const auto& info) {
                           return info.param ? "RdmaEnabled" : "RdmaDisabled";
                         });

}  // namespace
}  // namespace stream_executor::gpu
