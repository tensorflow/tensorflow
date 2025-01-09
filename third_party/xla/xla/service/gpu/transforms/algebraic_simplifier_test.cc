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

#include "xla/service/gpu/transforms/algebraic_simplifier.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class GpuAlgebraicSimplifierTest : public HloTestBase {
 public:
  se::CudaComputeCapability Ampere() {
    return se::CudaComputeCapability::Ampere();
  }
  se::CudaComputeCapability Hopper() {
    return se::CudaComputeCapability::Hopper();
  }
};

TEST_F(GpuAlgebraicSimplifierTest, SinkBroadcastOperandsOfChainedAdds) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      in = bf16[1,3,3,1] parameter(0)
      filter = bf16[2,2,1,1] constant({{{{1.1}}, {{2.1}}},
                                      {{{3.1}}, {{4.1}}}})
      conv = bf16[1,2,2,1] convolution(in, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      const0 = bf16[2] constant({0, 0.25})
      bcast0 = bf16[1,2,2,1] broadcast(const0), dimensions={1}
      add0 = bf16[1,2,2,1] add(conv, bcast0)
      const1 = bf16[2] constant({1, 1.25})
      bcast1 = bf16[1,2,2,1] broadcast(const1), dimensions={1}
      ROOT add1 = bf16[1,2,2,1] add(add0, bcast1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(true);
  ASSERT_TRUE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::AddAnyOrder(
                  m::Broadcast(m::Add(m::Constant(), m::Constant())),
                  m::Convolution(m::Op(), m::Op()))));
}

TEST_F(GpuAlgebraicSimplifierTest,
       DoNotSinkBroadcastOperandsOfChainedAddsWhenDisabled) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      in = bf16[1,3,3,1] parameter(0)
      filter = bf16[2,2,1,1] constant({{{{1.1}}, {{2.1}}},
                                      {{{3.1}}, {{4.1}}}})
      conv = bf16[1,2,2,1] convolution(in, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      const0 = bf16[2] constant({0, 0.25})
      bcast0 = bf16[1,2,2,1] broadcast(const0), dimensions={1}
      add0 = bf16[1,2,2,1] add(conv, bcast0)
      const1 = bf16[2] constant({1, 1.25})
      bcast1 = bf16[1,2,2,1] broadcast(const1), dimensions={1}
      ROOT add1 = bf16[1,2,2,1] add(add0, bcast1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(false);
  EXPECT_FALSE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DoNotSinkBroadcastOperandsOfChainedAddsWithoutConvolution) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      p = bf16[4, 4] parameter(0)
      const0 = bf16[4] constant({0, 0.25, 0.5, 0.75})
      bcast0 = bf16[4,4] broadcast(const0), dimensions={0}
      add0 = bf16[4,4] add(p, bcast0)
      const1 = bf16[4] constant({1, 1.25, 1.5, 1.75})
      bcast1 = bf16[4,4] broadcast(const1), dimensions={0}
      ROOT add1 = bf16[4,4] add(add0, bcast1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(true);
  EXPECT_FALSE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
}

TEST_F(
    GpuAlgebraicSimplifierTest,
    DoNotSinkBroadcastOperandsOfChainedAddsWithMismatchedBroadcastDimensions) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      in = bf16[1,3,3,1] parameter(0)
      filter = bf16[2,2,1,1] constant({{{{1.1}}, {{2.1}}},
                                      {{{3.1}}, {{4.1}}}})
      conv = bf16[1,2,2,1] convolution(in, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      const0 = bf16[2] constant({0, 0.25})
      bcast0 = bf16[1,2,2,1] broadcast(const0), dimensions={1}
      add0 = bf16[1,2,2,1] add(conv, bcast0)
      const1 = bf16[2] constant({1, 1.25})
      bcast1 = bf16[1,2,2,1] broadcast(const1), dimensions={2}
      ROOT add1 = bf16[1,2,2,1] add(add0, bcast1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(true);
  EXPECT_FALSE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
}

TEST_F(GpuAlgebraicSimplifierTest, VectorVectorDotShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[32, 500] parameter(0)
  p1 = f32[32, 500] parameter(1)
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
  p0 = f32[32, 5000, 7000] parameter(0)
  p1 = f32[32, 5000] parameter(1)
  ROOT dot = f32[32,7000] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1},
    algorithm=dot_bf16_bf16_f32_x6
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
  p0 = c64[32, 5000, 7000] parameter(0)
  p1 = c64[32, 5000] parameter(1)
  ROOT dot = c64[32,7000] dot(p0, p1), lhs_batch_dims={0},
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

TEST_F(GpuAlgebraicSimplifierTest, SmallDotShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[32, 50, 70] parameter(0)
  p1 = f32[32, 50] parameter(1)
  ROOT dot = f32[32,70] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1},
    algorithm=dot_bf16_bf16_f32_x6
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

TEST_F(GpuAlgebraicSimplifierTest, SmallDotShouldBeStrengthReduced2) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[2000, 3000] parameter(0)
  p1 = f32[2000] parameter(1)
  ROOT dot = f32[3000] dot(p0, p1), lhs_contracting_dims={0},
    rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_f32_x6
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

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_F32_F32_F32_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_f32_f32_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotStrengthReductionWith_F32_F32_F32_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,2]{1,0} parameter(0)
      b = f32[2]{0} parameter(1)
      ROOT dot = f32[128]{0} dot(a, b),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0},
        algorithm=dot_f32_f32_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_BF16_BF16_F32_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_bf16_bf16_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_BF16_BF16_F32_X3_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_bf16_bf16_f32_x3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_BF16_BF16_F32_X6_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_bf16_bf16_f32_x6
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

}  // namespace
}  // namespace xla::gpu
