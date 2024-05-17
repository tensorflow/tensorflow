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

#include "xla/service/gpu/gpu_algebraic_simplifier.h"

#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class GpuAlgebraicSimplifierTest : public HloTestBase {};

TEST_F(GpuAlgebraicSimplifierTest, VectorVectorDotShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[32, 5] parameter(0)
  p1 = f32[32, 5] parameter(1)
  ROOT dot = f32[32] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  AlgebraicSimplifierOptions options;
  options.set_enable_dot_strength_reduction(true);
  se::CudaComputeCapability ampere(8, 0);
  GpuAlgebraicSimplifier simplifier(options, ampere);
  GpuAlgebraicSimplifierVisitor visitor(options, ampere, &simplifier);
  EXPECT_TRUE(visitor.ShouldStrengthReduceDotToReduce(dot));
}

TEST_F(GpuAlgebraicSimplifierTest, MatrixVectorDotShouldNotBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[32, 5, 7] parameter(0)
  p1 = f32[32, 5] parameter(1)
  ROOT dot = f32[32,7] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  AlgebraicSimplifierOptions options;
  options.set_enable_dot_strength_reduction(true);
  se::CudaComputeCapability ampere(8, 0);
  GpuAlgebraicSimplifier simplifier(options, ampere);
  GpuAlgebraicSimplifierVisitor visitor(options, ampere, &simplifier);
  EXPECT_FALSE(visitor.ShouldStrengthReduceDotToReduce(dot));
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotWithTypeUnsupportedByGemmFusionShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = c64[32, 5, 7] parameter(0)
  p1 = c64[32, 5] parameter(1)
  ROOT dot = c64[32,7] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  AlgebraicSimplifierOptions options;
  options.set_enable_dot_strength_reduction(true);
  se::CudaComputeCapability ampere(8, 0);
  GpuAlgebraicSimplifier simplifier(options, ampere);
  GpuAlgebraicSimplifierVisitor visitor(options, ampere, &simplifier);
  EXPECT_TRUE(visitor.ShouldStrengthReduceDotToReduce(dot));
}

}  // namespace
}  // namespace xla::gpu
