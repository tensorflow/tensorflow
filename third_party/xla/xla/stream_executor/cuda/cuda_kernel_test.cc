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

#include <gtest/gtest.h>
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {
using testing::Ge;
using tsl::testing::IsOkAndHolds;

TEST(CudaKernelTest, GetMaxOccupiedBlocksPerCore) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  TF_ASSERT_OK_AND_ASSIGN(auto cuda_kernel,
                          executor->LoadKernel(GetAddI32KernelSpec()));

  EXPECT_EQ(cuda_kernel->Arity(), 3);
  EXPECT_THAT(cuda_kernel->GetMaxOccupiedBlocksPerCore(
                  ThreadDim(1, 1, 1), /*dynamic_shared_memory_bytes=*/0),
              IsOkAndHolds(Ge(1)));
}

}  // namespace
}  // namespace stream_executor::gpu
