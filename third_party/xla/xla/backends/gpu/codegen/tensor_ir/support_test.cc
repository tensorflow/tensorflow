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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
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

class IsSupportedFusionComputationTest : public HloHardwareIndependentTestBase {
};

TEST_F(IsSupportedFusionComputationTest, AllowsMultiOutputTupleRoot) {
  constexpr absl::string_view kHloText = R"(
    HloModule m

    fused_computation {
      p0 = f32[8] parameter(0)
      neg = f32[8] negate(p0)
      abs = f32[8] abs(p0)
      ROOT tuple = (f32[8], f32[8]) tuple(neg, abs)
    }

    ENTRY entry {
      p0 = f32[8] parameter(0)
      ROOT fusion = (f32[8], f32[8]) fusion(p0), kind=kLoop,
          calls=fused_computation
    })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  EXPECT_TRUE(IsSupportedFusionComputation(
                  *module->GetComputationWithName("fused_computation"))
                  .IsAllowed());
}

TEST_F(IsSupportedFusionComputationTest, ForbidsTupleShapedNonTupleRoot) {
  constexpr absl::string_view kHloText = R"(
    HloModule m

    compare {
      p0.lhs = f32[] parameter(0)
      p0.rhs = f32[] parameter(1)
      p1.lhs = f32[] parameter(2)
      p1.rhs = f32[] parameter(3)
      ROOT lt = pred[] compare(p0.lhs, p0.rhs), direction=LT
    }

    fused_computation {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      ROOT sort = (f32[8], f32[8]) sort(p0, p1), dimensions={0}, to_apply=compare
    }

    ENTRY entry {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      ROOT fusion = (f32[8], f32[8]) fusion(p0, p1), kind=kLoop,
          calls=fused_computation
    })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  CodegenDecision decision = IsSupportedFusionComputation(
      *module->GetComputationWithName("fused_computation"));
  EXPECT_FALSE(decision.IsAllowed());
  EXPECT_THAT(decision.Explain(), HasSubstr("non-array shape"));
}

}  // namespace
}  // namespace xla::gpu::tensor_ir
