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

#include "xla/backends/gpu/transforms/algebraic_simplifier.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class GpuAlgebraicSimplifierTest : public HloHardwareIndependentTestBase {
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

TEST_F(
    GpuAlgebraicSimplifierTest,
    DotToMultiplyRewriteForZeroContractingDimWith_BF16_BF16_F32_X6_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT dot = f32[] dot(a, b),
      algorithm=dot_bf16_bf16_f32_x6
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
  constexpr absl::string_view kPattern = R"(
    CHECK-COUNT-6: %[[partial_result:.*]] = bf16[] multiply
  )";
  TF_ASSERT_OK_AND_ASSIGN(bool matched, RunFileCheck(m->ToString(), kPattern));
  EXPECT_TRUE(matched);
}

}  // namespace
}  // namespace xla::gpu
