/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_collectives.h"

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {
using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;

TEST(CudaCollectivesTest, CollectiveMemoryAllocation) {
  auto* collectives = xla::gpu::GpuCollectives::Default();
  if (collectives->IsImplemented()) {
    GTEST_SKIP() << "Compiled without NCCL support";
  }

  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  constexpr size_t kAllocateSize = 1024;
  TF_ASSERT_OK_AND_ASSIGN(
      void* memory,
      CudaCollectives::CollectiveMemoryAllocate(executor, kAllocateSize));

  EXPECT_THAT(executor->GetPointerMemorySpace(memory),
              IsOkAndHolds(MemoryType::kDevice));

  EXPECT_THAT(CudaCollectives::CollectiveMemoryDeallocate(executor, memory),
              IsOk());
}

}  // namespace
}  // namespace stream_executor::gpu
