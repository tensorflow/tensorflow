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

#include "xla/stream_executor/rocm/gpu_test_kernels_rocm_embedded.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::rocm {
namespace {

using tsl::testing::IsOk;

TEST(GpuTestKernelsRocmEmbeddedTest, LoadEmbeddedKernel) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          stream_executor::PlatformManager::PlatformWithId(
                              stream_executor::rocm::kROCmPlatformId));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  KernelLoaderSpec spec = KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      kFatbin_gpu_test_kernels_rocm_embedded, "AddI32", /*arity=*/3);
  EXPECT_THAT(executor->LoadKernel(spec), IsOk());
}

}  // namespace
}  // namespace stream_executor::rocm
