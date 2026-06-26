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

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/statusor.h"
#include "absl/status/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_device_allocator.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {
namespace {

absl::StatusOr<StreamExecutor*> GetGpuExecutor(int64_t device_ordinal) {
  ASSIGN_OR_RETURN(Platform * platform,
                   PlatformManager::PlatformWithName(GpuPlatformName()));
  ASSIGN_OR_RETURN(StreamExecutor * executor,
                   platform->ExecutorForDevice(device_ordinal));
  return executor;
}

TEST(CudaDeviceAllocatorTest, HopperNoWarningCheck) {
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor, GetGpuExecutor(0));
  CudaComputeCapability cc =
      executor->GetDeviceDescription().cuda_compute_capability();
  if (!cc.IsAtLeastHopper()) {
    GTEST_SKIP() << "Test only runs on H100+";
  }

  auto* cuda_executor = static_cast<CudaExecutor*>(executor);

  CudaDeviceAllocator::Options options;
  options.enable_fabric_handle = cuda_executor->is_fabric_supported();
  options.enable_posix_fd_handle = true;

  if (!options.enable_fabric_handle || !cuda_executor->GetDeviceDescription()
                                            .device_interconnect_info()
                                            .is_in_cluster()) {
    GTEST_SKIP()
        << "Test requires fabric support and an active fabric cluster.";
  }

  // We want to verify that no warning is logged when attempting to allocate
  // with FABRIC+POSIX_FD handle types on Hopper when fabric is supported and
  // cluster is active.
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, ::testing::_,
                       ::testing::HasSubstr("FABRIC+POSIX_FD")))
      .Times(0);
  log.StartCapturingLogs();

  CudaDeviceAllocator allocator(executor, options);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(1024));
  EXPECT_NE(allocation, nullptr);

  log.StopCapturingLogs();
}

}  // namespace
}  // namespace stream_executor::gpu
