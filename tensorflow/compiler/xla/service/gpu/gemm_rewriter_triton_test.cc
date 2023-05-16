/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"

#include <string>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;

namespace m = ::xla::match;

using GemmRewriterTritonTest = HloTestBase;

TEST_F(GemmRewriterTritonTest, TransposeSubdimensionGroup) {
  // This HLO is artificial because unnecessary reshapes get optimized
  // out during compilation. It tests the ability of GemmRewriterTriton
  // to handle transposes of groups of subdimensions.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = f32[32,3] parameter(0)
  t1 = f32[3,32] transpose(p0), dimensions={1,0}
  r1 = f32[3,8,4] reshape(t1)
  r0 = f32[3,32] reshape(r1)
  p1 = f16[32,7] parameter(1)
  c1 = f32[32,7] convert(p1)
  ROOT d = f32[3,7] dot(r0, c1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                    .value();
  GpuVersion gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  EXPECT_TRUE(GemmRewriterTriton(gpu_version).Run(module.get()).value());
}

TEST_F(GemmRewriterTritonTest, BitcastChain) {
  // This HLO is artificial because unnecessary reshapes get optimized
  // out during compilation. It tests the ability of GemmRewriterTriton
  // to handle various kinds of bitcasts.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = s8[60,5] parameter(0)
  r0 = s8[3,20,5] reshape(p0)
  c0 = f16[3,20,5] convert(r0)
  p1 = f16[3,200] parameter(1)
  r12 = f16[600] reshape(p1)
  r11 = f16[30,20] reshape(r12)
  r1 = f16[3,10,20] reshape(r11)
  ROOT d = f16[3,5,10] dot(c0, r1),
    lhs_contracting_dims={1}, rhs_contracting_dims={2},
    lhs_batch_dims={0}, rhs_batch_dims={0}
})")
                    .value();
  GpuVersion gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  EXPECT_TRUE(GemmRewriterTriton(gpu_version).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

using TritonDotAnalysisTest = HloTestBase;

TEST_F(TritonDotAnalysisTest, NopBitcasts) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = s8[48,4]{1,0} parameter(0)
  bitcast.18 = s8[1,48,4]{2,1,0} bitcast(param_0.1)
  bitcast.19 = s8[48,4]{1,0} bitcast(bitcast.18)
  convert.4 = bf16[48,4]{1,0} convert(bitcast.19)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[48,3]{1,0} dot(convert.4, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[48,4]{1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[48,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
  ROOT bitcast.2 = bf16[1,8,6,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* dot = dot_computation->root_instruction();
  const DotFusionAnalysis analysis(dot);
  EXPECT_EQ(analysis.OperandToParameter(0),
            dot_computation->parameter_instruction(0));
  EXPECT_EQ(analysis.OperandToParameter(1),
            dot_computation->parameter_instruction(1));
  EXPECT_THAT(
      analysis.IterSpec(0, 0),
      ElementsAre(FieldsAre(/*stride=*/4, /*count=*/48, ElementsAre(48))));
  EXPECT_THAT(
      analysis.IterSpec(0, 1),
      ElementsAre(FieldsAre(/*stride=*/1, /*count=*/4, ElementsAre(4))));
  EXPECT_THAT(
      analysis.IterSpec(1, 0),
      ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4, ElementsAre(4))));
  EXPECT_THAT(
      analysis.IterSpec(1, 1),
      ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3, ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, Merge) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = s8[1,8,6,4]{3,2,1,0} parameter(0)
  bitcast.18 = s8[48,4]{1,0} bitcast(param_0.1)
  convert.4 = bf16[48,4]{1,0} convert(bitcast.18)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[48,3]{1,0} dot(convert.4, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[1,8,6,4]{3,2,1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[48,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
  ROOT bitcast.2 = bf16[1,8,6,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* dot = dot_computation->root_instruction();
  const DotFusionAnalysis analysis(dot);
  EXPECT_EQ(analysis.OperandToParameter(0),
            dot_computation->parameter_instruction(0));
  EXPECT_EQ(analysis.OperandToParameter(1),
            dot_computation->parameter_instruction(1));
  EXPECT_THAT(analysis.IterSpec(0, 0),
              ElementsAre(FieldsAre(/*stride=*/4, /*count=*/6 * 8,
                                    /*subfragments=*/ElementsAre(6, 8))));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3,
                                    /*subfragments=*/ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, Split) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  %parameter_1 = f32[24000,2]{1,0} parameter(1)
  %convert.15 = f16[24000,2]{1,0} convert(%parameter_1)
  %parameter_0 = f16[4]{0} parameter(0)
  %bitcast.45 = f16[2,2]{1,0} bitcast(%parameter_0)
  ROOT %dot.26 = f16[24000,2]{1,0} dot(%convert.15, %bitcast.45),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[4]{0} parameter(0)
  p1 = f32[24000,2]{1,0} parameter(1)
  ROOT r = f16[24000,2]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  const HloInstruction* dot = dot_computation->root_instruction();
  const DotFusionAnalysis analysis(dot);
  EXPECT_EQ(analysis.OperandToParameter(0),
            dot_computation->parameter_instruction(1));
  EXPECT_EQ(analysis.OperandToParameter(1),
            dot_computation->parameter_instruction(0));
  EXPECT_THAT(analysis.IterSpec(0, 0),
              ElementsAre(FieldsAre(/*stride=*/2, /*count=*/24000,
                                    /*subfragments=*/ElementsAre(24000))));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/2,
                                    /*subfragments=*/ElementsAre(2))));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/2, /*count=*/2,
                                    /*subfragments=*/ElementsAre(2))));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/2,
                                    /*subfragments=*/ElementsAre(2))));
}

TEST_F(TritonDotAnalysisTest, TransposeMerge) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = s8[1,4,8,6]{3,2,1,0} parameter(0)
  transpose.3 = s8[1,8,6,4]{3,2,1,0} transpose(param_0.1), dimensions={0,2,3,1}
  bitcast.18 = s8[48,4]{1,0} bitcast(transpose.3)
  convert.4 = bf16[48,4]{1,0} convert(bitcast.18)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[48,3]{1,0} dot(convert.4, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[1,4,8,6]{3,2,1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[48,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
  ROOT bitcast.2 = bf16[1,8,6,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* dot = dot_computation->root_instruction();
  const DotFusionAnalysis analysis(dot);
  EXPECT_EQ(analysis.OperandToParameter(0),
            dot_computation->parameter_instruction(0));
  EXPECT_EQ(analysis.OperandToParameter(1),
            dot_computation->parameter_instruction(1));
  EXPECT_THAT(analysis.IterSpec(0, 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8 * 6,
                                    /*subfragments=*/ElementsAre(6, 8))));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/8 * 6, /*count=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3,
                                    /*subfragments=*/ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, CopyMerge) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = s8[1,4,8,6]{3,2,1,0} parameter(0)
  bitcast.99 = s8[1,8,6,4]{2,1,3,0} bitcast(param_0.1)
  copy.3 = s8[1,8,6,4]{3,2,1,0} copy(bitcast.99)
  bitcast.18 = s8[48,4]{1,0} bitcast(copy.3)
  convert.4 = bf16[48,4]{1,0} convert(bitcast.18)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[48,3]{1,0} dot(convert.4, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[1,4,8,6]{3,2,1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[48,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
  ROOT bitcast.2 = bf16[1,8,6,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* dot = dot_computation->root_instruction();
  const DotFusionAnalysis analysis(dot);
  EXPECT_EQ(analysis.OperandToParameter(0),
            dot_computation->parameter_instruction(0));
  EXPECT_EQ(analysis.OperandToParameter(1),
            dot_computation->parameter_instruction(1));
  EXPECT_THAT(analysis.IterSpec(0, 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8 * 6,
                                    /*subfragments=*/ElementsAre(6, 8))));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/8 * 6, /*count=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3,
                                    /*subfragments=*/ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, TransposeMergeNCN) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = bf16[3,4,8,1]{3,2,1,0} parameter(0)
  transpose.3 = bf16[3,8,1,4]{3,2,1,0} transpose(param_0.1), dimensions={0,2,3,1}
  bitcast.18 = bf16[24,4]{1,0} bitcast(transpose.3)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[24,3]{1,0} dot(bitcast.18, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = bf16[3,4,8,1]{3,2,1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[24,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton", called_computations={triton_dot}
  ROOT bitcast.2 = bf16[3,8,1,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* dot = dot_computation->root_instruction();
  const DotFusionAnalysis analysis(dot);
  EXPECT_EQ(analysis.OperandToParameter(0),
            dot_computation->parameter_instruction(0));
  EXPECT_EQ(analysis.OperandToParameter(1),
            dot_computation->parameter_instruction(1));
  EXPECT_THAT(analysis.IterSpec(0, 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8,
                                    /*subfragments=*/ElementsAre(8)),
                          FieldsAre(/*stride=*/4 * 8, /*count=*/3,
                                    /*subfragments=*/ElementsAre(3))));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/8, /*count=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3,
                                    /*subfragments=*/ElementsAre(3))));
}

using SplitKTest = HloTestBase;

TEST_F(SplitKTest, MakeSplitK) {
  const std::string hlo_text = R"(
HloModule t

triton_gemm_dot {
  parameter_0 = s8[3,128,5,32]{3,2,1,0} parameter(0)
  bitcast.1 = s8[3,5,32,128]{2,1,3,0} bitcast(parameter_0)
  copy.1 = s8[3,5,32,128]{3,2,1,0} copy(bitcast.1)
  reshape.5 = s8[480,128]{1,0} reshape(copy.1)
  convert.8 = bf16[480,128]{1,0} convert(reshape.5)
  parameter_1 = bf16[16,128]{1,0} parameter(1)
  ROOT dot.0 = bf16[480,16]{1,0} dot(convert.8, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = s8[3,128,5,32]{3,2,1,0} parameter(0)
  p1 = bf16[16,128]{1,0} parameter(1)
  ROOT fusion = bf16[480,16]{1,0} fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot, backend_config="__triton_gemm"
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  tensorflow::AutotuneResult::TritonGemmKey key;
  key.set_block_m(16);
  key.set_block_n(16);
  key.set_block_k(16);
  key.set_split_k(4);
  key.set_num_stages(1);
  key.set_num_warps(4);
  TF_EXPECT_OK(
      MakeDotSplitKBatch(module->entry_computation()->root_instruction(), key));
  EXPECT_TRUE(VerifyHloModule(module.get(), /*layout_sensitive=*/true,
                              /*allow_mixed_precision=*/false)
                  .ok());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kReduce);
}

TEST_F(SplitKTest, MakeSplitKWithNonStandardOutputLayout) {
  const std::string kHloText = R"(
HloModule t

triton_gemm_dot {
  parameter_0 = s8[3,128,5,32]{3,2,1,0} parameter(0)
  bitcast.1 = s8[3,5,32,128]{2,1,3,0} bitcast(parameter_0)
  copy.1 = s8[3,5,32,128]{3,2,1,0} copy(bitcast.1)
  reshape.5 = s8[480,128]{1,0} reshape(copy.1)
  convert.8 = bf16[480,128]{1,0} convert(reshape.5)
  parameter_1 = bf16[16,128]{1,0} parameter(1)
  ROOT dot.0 = bf16[480,16]{0,1} dot(convert.8, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = s8[3,128,5,32]{3,2,1,0} parameter(0)
  p1 = bf16[16,128]{1,0} parameter(1)
  ROOT fusion = bf16[480,16]{0,1} fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot, backend_config="__triton_gemm"
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  tensorflow::AutotuneResult::TritonGemmKey key;
  key.set_block_m(16);
  key.set_block_n(16);
  key.set_block_k(16);
  key.set_split_k(4);
  key.set_num_stages(1);
  key.set_num_warps(4);

  TF_EXPECT_OK(
      MakeDotSplitKBatch(module->entry_computation()->root_instruction(), key));

  TF_EXPECT_OK(VerifyHloModule(module.get(), /*layout_sensitive=*/true,
                               /*allow_mixed_precision=*/false));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kReduce);
  EXPECT_EQ(module->entry_computation()->root_instruction()->shape().layout(),
            Layout({0, 1}));
}

TEST_F(SplitKTest, MakeSplitKWithExistingBatchDim) {
  const std::string hlo_text = R"(
HloModule m

triton_gemm_dot.24 {
  parameter_1 = bf16[1,1,800,5,128]{4,3,2,1,0} parameter(1)
  bitcast.3 = bf16[800,5,128]{2,1,0} bitcast(parameter_1)
  convert.3 = f32[800,5,128]{2,1,0} convert(bitcast.3)
  parameter_0 = f32[1,5,700,800]{3,2,1,0} parameter(0)
  bitcast.2 = f32[5,700,800]{2,1,0} bitcast(parameter_0)
  ROOT dot.26 = f32[5,128,700]{2,1,0} dot(convert.3, bitcast.2),
    lhs_batch_dims={1}, lhs_contracting_dims={0},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY e {
  tmp_3 = f32[1,5,700,800]{3,2,1,0} parameter(0)
  tmp_0 = bf16[1,1,800,5,128]{4,3,2,1,0} parameter(1)
  ROOT triton_gemm_dot.24 = f32[5,128,700]{2,1,0} fusion(tmp_3, tmp_0),
    kind=kCustom, calls=triton_gemm_dot.24,
    backend_config="__triton_gemm"
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  tensorflow::AutotuneResult::TritonGemmKey key;
  key.set_block_m(32);
  key.set_block_n(64);
  key.set_block_k(64);
  key.set_split_k(8);
  key.set_num_stages(1);
  key.set_num_warps(4);
  TF_EXPECT_OK(
      MakeDotSplitKBatch(module->entry_computation()->root_instruction(), key));
  EXPECT_TRUE(VerifyHloModule(module.get(), /*layout_sensitive=*/true,
                              /*allow_mixed_precision=*/false)
                  .ok());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kReduce);
}

TEST_F(SplitKTest, SkipIndivisible) {
  const std::string hlo_text = R"(
HloModule t

triton_gemm_dot {
  parameter_0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  bitcast.1 = s8[3,5,32,129]{2,1,3,0} bitcast(parameter_0)
  copy.1 = s8[3,5,32,129]{3,2,1,0} copy(bitcast.1)
  reshape.5 = s8[480,129]{1,0} reshape(copy.1)
  convert.8 = bf16[480,129]{1,0} convert(reshape.5)
  parameter_1 = bf16[16,129]{1,0} parameter(1)
  ROOT dot.0 = bf16[480,16]{1,0} dot(convert.8, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = s8[3,129,5,32]{3,2,1,0} parameter(0)
  p1 = bf16[16,129]{1,0} parameter(1)
  ROOT fusion = bf16[480,16]{1,0} fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot, backend_config="__triton_gemm"
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  tensorflow::AutotuneResult::TritonGemmKey key;
  key.set_block_m(16);
  key.set_block_n(16);
  key.set_block_k(16);
  key.set_split_k(4);
  key.set_num_stages(1);
  key.set_num_warps(4);
  EXPECT_THAT(
      MakeDotSplitKBatch(module->entry_computation()->root_instruction(), key),
      tsl::testing::StatusIs(tsl::error::CANCELLED,
                             "Contracting dimension is too fragmented."));
}

TEST_F(SplitKTest, SkipSmallK) {
  const std::string hlo_text = R"(
HloModule t

triton_gemm_dot {
  parameter_0 = s8[3,64,5,32]{3,2,1,0} parameter(0)
  bitcast.1 = s8[3,5,32,64]{2,1,3,0} bitcast(parameter_0)
  copy.1 = s8[3,5,32,64]{3,2,1,0} copy(bitcast.1)
  reshape.5 = s8[480,64]{1,0} reshape(copy.1)
  convert.8 = bf16[480,64]{1,0} convert(reshape.5)
  parameter_1 = bf16[16,64]{1,0} parameter(1)
  ROOT dot.0 = bf16[480,16]{1,0} dot(convert.8, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = s8[3,64,5,32]{3,2,1,0} parameter(0)
  p1 = bf16[16,64]{1,0} parameter(1)
  ROOT fusion = bf16[480,16]{1,0} fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot, backend_config="__triton_gemm"
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  tensorflow::AutotuneResult::TritonGemmKey key;
  key.set_block_m(16);
  key.set_block_n(16);
  key.set_block_k(128);
  key.set_split_k(4);
  key.set_num_stages(1);
  key.set_num_warps(4);
  EXPECT_THAT(
      MakeDotSplitKBatch(module->entry_computation()->root_instruction(), key),
      tsl::testing::StatusIs(
          tsl::error::CANCELLED,
          "Too small divisible part of the contracting dimension."));
}

TEST_F(SplitKTest, FragmentedKSupported) {
  const std::string hlo_text = R"(
HloModule t

triton_gemm_dot {
  p0 = f16[7,2,16,4,20] parameter(0)
  t0 = f16[2,16,4,20,7] transpose(p0), dimensions={1,2,3,4,0}
  b0 = f16[2560,7] bitcast(t0)
  a1 = f16[2560,5] parameter(1)
  ROOT r = f16[7,5] dot(b0, a1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[7,2,16,4,20] parameter(0)
  p1 = f16[2560,5] parameter(1)
  ROOT fusion = f16[7,5] fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot, backend_config="__triton_gemm"
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  tensorflow::AutotuneResult::TritonGemmKey key;
  key.set_block_m(32);
  key.set_block_n(32);
  key.set_block_k(16);
  key.set_num_stages(1);
  key.set_num_warps(4);

  // 5 divides the contracting dimension, but not its major subdimensions.
  key.set_split_k(5);
  EXPECT_THAT(
      MakeDotSplitKBatch(module->entry_computation()->root_instruction(), key),
      tsl::testing::StatusIs(tsl::error::CANCELLED,
                             "Contracting dimension is too fragmented."));

  // 8 fits the constraints.
  key.set_split_k(8);
  TF_EXPECT_OK(
      MakeDotSplitKBatch(module->entry_computation()->root_instruction(), key));
  TF_EXPECT_OK(VerifyHloModule(module.get(), /*layout_sensitive=*/true,
                               /*allow_mixed_precision=*/false));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReduce);
  DotFusionAnalysis analysis(root->operand(0)->fused_expression_root(),
                             key.split_k());
  EXPECT_THAT(analysis.IterSpec(0, 0),
              ElementsAre(FieldsAre(/*stride=*/320, /*count=*/8,
                                    /*subfragments=*/ElementsAre(4, 2))));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/320,
                                    /*subfragments=*/ElementsAre(20, 4, 4))));
}

TEST_F(SplitKTest, FragmentedKUnsupported) {
  const std::string hlo_text = R"(
HloModule t

triton_gemm_dot {
  p0 = f32[3,128,77] parameter(0)
  b0 = f32[384,77] bitcast(p0)
  a1 = f32[384,25] parameter(1)
  ROOT r = f32[77,25] dot(b0, a1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[3,128,77] parameter(0)
  p1 = f32[384,25] parameter(1)
  ROOT fusion = f32[77,25] fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot, backend_config="__triton_gemm"
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  tensorflow::AutotuneResult::TritonGemmKey key;
  key.set_block_m(16);
  key.set_block_n(16);
  key.set_block_k(16);
  key.set_num_stages(1);
  key.set_num_warps(4);
  key.set_split_k(4);
  EXPECT_THAT(
      MakeDotSplitKBatch(module->entry_computation()->root_instruction(), key),
      tsl::testing::StatusIs(tsl::error::CANCELLED,
                             "Contracting dimension is too fragmented."));
}
}  // namespace
}  // namespace gpu
}  // namespace xla
