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

#include "xla/backends/gpu/codegen/tensor_ir/support.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace xla::gpu::tensor_ir {
namespace {

using ::testing::HasSubstr;

TEST(IsSupportedComputeCapabilityTest, AllowsHopper) {
  EXPECT_TRUE(IsSupportedComputeCapability(
                  se::GpuComputeCapability(se::CudaComputeCapability::Hopper()))
                  .IsAllowed());
}

TEST(IsSupportedComputeCapabilityTest, AllowsNewerThanHopper) {
  EXPECT_TRUE(
      IsSupportedComputeCapability(
          se::GpuComputeCapability(se::CudaComputeCapability::Blackwell()))
          .IsAllowed());
}

TEST(IsSupportedComputeCapabilityTest, ForbidsAmpere) {
  CodegenDecision decision = IsSupportedComputeCapability(
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere()));
  EXPECT_FALSE(decision.IsAllowed());
  EXPECT_THAT(decision.Explain(), HasSubstr("9.0"));
  EXPECT_THAT(decision.Explain(), HasSubstr("8.0"));
}

TEST(IsSupportedComputeCapabilityTest, ForbidsNonCuda) {
  CodegenDecision decision = IsSupportedComputeCapability(
      se::GpuComputeCapability(se::RocmComputeCapability("gfx942")));
  EXPECT_FALSE(decision.IsAllowed());
  EXPECT_THAT(decision.Explain(), HasSubstr("CUDA"));
}

}  // namespace
}  // namespace xla::gpu::tensor_ir
