/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/cublas_pad_for_gemms.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace m = ::xla::match;

namespace xla {
namespace gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class CublasGemmPadForTensorCoresTest : public HloHardwareIndependentTestBase {
 protected:
  bool PadForF16Gemms(HloModule* module) {
    return CublasPadForGemms(se::CudaComputeCapability(7, 0))
        .Run(module)
        .value();
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
    // Some pads would not be added if we detect that Triton will handle the
    // given dot operation.
    debug_options.set_xla_gpu_triton_gemm_any(false);
    return debug_options;
  }
};

TEST_F(CublasGemmPadForTensorCoresTest, OneDotRootComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[2048,1024] parameter(0)
    %param2 = f16[1024,33708] parameter(1)
    ROOT %dot.2309 = f16[2048,33708]{1,0} dot(f16[2048,1024]{1,0} %param1,
                f16[1024,33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
                })")
                    .value();

  EXPECT_TRUE(PadForF16Gemms(module.get()));
  SCOPED_TRACE(module->ToString());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(
          m::Slice(m::Dot(m::Parameter().WithShape(F16, {2048, 1024}),
                          m::Pad(m::Parameter().WithShape(F16, {1024, 33708}),
                                 m::Constant().WithShape(F16, {}))
                              .WithShape(F16, {1024, 33712}))
                       .WithShape(F16, {2048, 33712})
                       .WithContractingDims(/*lhs_contracting_dims=*/{1},
                                            /*rhs_contracting_dims=*/{0}))
              .WithShape(F16, {2048, 33708})));
}

TEST_F(CublasGemmPadForTensorCoresTest, OneDotS8RootComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = s8[2047,1023] parameter(0)
    %param2 = s8[1023,33707] parameter(1)
    ROOT %dot.2309 = s32[2047,33707]{1,0} dot(s8[2047,1023]{1,0} %param1,
                s8[1023,33707]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
                })")
                    .value();

  EXPECT_THAT(
      CublasPadForGemms(se::CudaComputeCapability(7, 0)).Run(module.get()),
      IsOkAndHolds(true));
  SCOPED_TRACE(module->ToString());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(
          m::Slice(m::Dot(m::Pad(m::Parameter().WithShape(S8, {2047, 1023}),
                                 m::Constant().WithShape(S8, {}))
                              .WithShape(S8, {2048, 1024}),
                          m::Pad(m::Parameter().WithShape(S8, {1023, 33707}),
                                 m::Constant().WithShape(S8, {}))
                              .WithShape(S8, {1024, 33708}))
                       .WithShape(S32, {2048, 33708})
                       .WithContractingDims(/*lhs_contracting_dims=*/{1},
                                            /*rhs_contracting_dims=*/{0}))
              .WithShape(S32, {2047, 33707})));
}

TEST_F(CublasGemmPadForTensorCoresTest, TwoDotsComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[2048, 1024] parameter(0)
    %param2 = f16[1024, 33708] parameter(1)
    %param3 = f16[33708, 2] parameter(2)
    %dot1 = f16[2048, 33708]{1,0} dot(f16[2048, 1024]{1,0} %param1,
                f16[1024, 33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT %dot2 = f16[2048, 2]{1,0} dot(f16[2048, 33708]{1,0} %dot1,
                f16[33708, 2]{0,1} %param3),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })")
                    .value();

  EXPECT_TRUE(PadForF16Gemms(module.get()));
  SCOPED_TRACE(module->ToString());

  auto* root = module->entry_computation()->root_instruction();
  const HloInstruction* dot2 = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(
          m::Slice(m::Dot(m::Pad(m::Slice(m::Dot(&dot2,
                                                 m::Parameter().WithShape(
                                                     F16, {2048, 1024}),
                                                 m::Pad().WithShape(
                                                     F16, {1024, 33712}))
                                              .WithContractingDims(
                                                  /*lhs_contracting_dims=*/{1},
                                                  /*rhs_contracting_dims=*/{0})
                                              .WithShape(F16, {2048, 33712}))
                                     .WithShape(F16, {2048, 33708}),
                                 m::Constant().WithShape(F16, {}))
                              .WithShape(F16, {2048, 33712}),

                          m::Pad(m::Parameter().WithShape(F16, {33708, 2}),
                                 m::Constant().WithShape(F16, {}))
                              .WithShape(F16, {33712, 8}))
                       .WithShape(F16, {2048, 8})
                       .WithContractingDims(/*lhs_contracting_dims=*/{1},
                                            /*rhs_contracting_dims=*/{0}))
              .WithShape(F16, {2048, 2})));

  EXPECT_THAT(
      dot2,
      GmockMatch(m::Dot(m::Parameter().WithShape(F16, {2048, 1024}),
                        m::Pad(m::Parameter().WithShape(F16, {1024, 33708}),
                               m::Constant().WithShape(F16, {}))
                            .WithShape(F16, {1024, 33712}))
                     .WithContractingDims(/*lhs_contracting_dims=*/{1},
                                          /*rhs_contracting_dims=*/{0})));
}

TEST_F(CublasGemmPadForTensorCoresTest, DotWithBatchDimensions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[3, 5, 2048, 1024] parameter(0)
    %param2 = f16[3, 5, 1024, 33708] parameter(1)
    ROOT %dot.2309 = f16[3, 5, 2048, 33708]{3, 2, 1,0} dot(f16[3, 5, 2048, 1024]{3, 2, 1,0} %param1,
                f16[3, 5, 1024, 33708]{2, 3, 0,1} %param2), lhs_batch_dims={0, 1}, rhs_batch_dims={0, 1}, lhs_contracting_dims={3}, rhs_contracting_dims={2}})")
                    .value();

  EXPECT_TRUE(PadForF16Gemms(module.get()));
  SCOPED_TRACE(module->ToString());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(
          m::Slice(
              m::Dot(m::Parameter().WithShape(F16, {3, 5, 2048, 1024}),
                     m::Pad(m::Parameter().WithShape(F16, {3, 5, 1024, 33708}),
                            m::Constant().WithShape(F16, {}))
                         .WithShape(F16, {3, 5, 1024, 33712}))
                  .WithShape(F16, {3, 5, 2048, 33712})
                  .WithContractingDims(/*lhs_contracting_dims=*/{3},
                                       /*rhs_contracting_dims=*/{2}))
              .WithShape(F16, {3, 5, 2048, 33708})));
}

TEST_F(CublasGemmPadForTensorCoresTest, NoDotComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %x = f32[] parameter(0)
    %y = f32[] parameter(1)
    ROOT %maximum = f32[] maximum(f32[] %x, f32[] %y)
  })")
                    .value();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

TEST_F(CublasGemmPadForTensorCoresTest, F32DotComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f32[2048,1024] parameter(0)
    %param2 = f32[1024,33708] parameter(1)
    ROOT %dot.2309 = f32[2048,33708]{1,0} dot(f32[2048,1024]{1,0} %param1,
                f32[1024,33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}})")
                    .value();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

TEST_F(CublasGemmPadForTensorCoresTest, F64DotComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f64[2048,1024] parameter(0)
    %param2 = f64[1024,33708] parameter(1)
    ROOT %dot.2309 = f64[2048,33708]{1,0} dot(f64[2048,1024]{1,0} %param1,
                f64[1024,33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}})")
                    .value();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

TEST_F(CublasGemmPadForTensorCoresTest, MultiplesOf8DotComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[2048,1024] parameter(0)
    %param2 = f16[1024,33712] parameter(1)
    ROOT %dot.2309 = f16[2048,33712]{1,0} dot(f16[2048,1024]{1,0} %param1,
                f16[1024,33712]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}})")
                    .value();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

TEST_F(CublasGemmPadForTensorCoresTest, CheckSavingMetadata) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[2048,1024] parameter(0)
    %param2 = f16[1024,33708] parameter(1)
    ROOT %dot.2309 = f16[2048,33708]{1,0} dot(f16[2048,1024]{1,0} %param1,
                f16[1024,33708]{0,1} %param2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0},
                metadata={op_type="MatMul" op_name="transformer_v2/Transformer/decode/embedding_shared_weights_1/presoftmax_linear/MatMul"}
                })")
                    .value();

  SCOPED_TRACE(module->ToString());

  EXPECT_TRUE(PadForF16Gemms(module.get()));
  auto metadata = module->entry_computation()->root_instruction()->metadata();
  EXPECT_EQ("MatMul", metadata.op_type());
  EXPECT_EQ(
      "transformer_v2/Transformer/decode/embedding_shared_weights_1/"
      "presoftmax_linear/MatMul",
      metadata.op_name());
}

TEST_F(CublasGemmPadForTensorCoresTest, NotCanonicalizedDot) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    %param1 = f16[3, 5, 2048, 1024] parameter(0)
    %param2 = f16[3, 5, 1024, 33708] parameter(1)
    ROOT %dot.2309 = f16[3,2048, 33708]{2, 1, 0} dot(f16[3, 5, 2048, 1024]{3, 2, 1, 0} %param1, f16[3, 5, 1024, 33708]{3, 2, 1, 0} %param2), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={3, 1}, rhs_contracting_dims={2, 1}})")
                    .value();

  EXPECT_FALSE(PadForF16Gemms(module.get()));
}

TEST_F(CublasGemmPadForTensorCoresTest, PadsBf16OnlyFromAmpere) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = bf16[16,15] parameter(0)
      p1 = bf16[15,16] parameter(1)
      ROOT dot = bf16[16,16] dot(p0, p1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })"));

  EXPECT_THAT(
      CublasPadForGemms(se::CudaComputeCapability::Volta()).Run(module.get()),
      IsOkAndHolds(false));
  EXPECT_THAT(
      CublasPadForGemms(se::CudaComputeCapability::Ampere()).Run(module.get()),
      IsOkAndHolds(true));
  // Only the contracting dimension is padded, so the result needs no slice.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Pad().WithShape(BF16, {16, 16}),
                                m::Pad().WithShape(BF16, {16, 16}))
                             .WithShape(BF16, {16, 16})));
}

TEST_F(CublasGemmPadForTensorCoresTest, DoesNothingOnRocm) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = f16[15,15] parameter(0)
      p1 = f16[15,15] parameter(1)
      ROOT dot = f16[15,15] dot(p0, p1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })"));

  EXPECT_THAT(CublasPadForGemms(
                  se::GpuComputeCapability(se::RocmComputeCapability("gfx942")))
                  .Run(module.get()),
              IsOkAndHolds(false));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
