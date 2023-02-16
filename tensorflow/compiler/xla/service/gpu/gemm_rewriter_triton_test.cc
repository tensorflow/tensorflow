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
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using TritonDotAnalysisTest = HloTestBase;
using ::testing::ElementsAre;
using ::testing::FieldsAre;

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
  EXPECT_THAT(analysis.IterSpec(0, 0),
              ElementsAre(FieldsAre(/*stride=*/4, /*count=*/48)));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3)));
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
              ElementsAre(FieldsAre(/*stride=*/4, /*count=*/6 * 8)));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3)));
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
              ElementsAre(FieldsAre(/*stride=*/2, /*count=*/24000)));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/2)));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/2, /*count=*/2)));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/2)));
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
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8 * 6)));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/8 * 6, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3)));
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
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8 * 6)));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/8 * 6, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3)));
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
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8),
                          FieldsAre(/*stride=*/4 * 8, /*count=*/3)));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/8, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4)));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3)));
}

TEST_F(TritonDotAnalysisTest, SplitRhsContracting) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  p0 = s8[5,8] parameter(0)
  c0 = f16[5,8] convert(p0)
  p1 = f16[2,3,4] parameter(1)
  t1 = f16[3,2,4] transpose(p1), dimensions={1,0,2}
  r1 = f16[3,8] reshape(t1)
  ROOT _ = f16[5,3] dot(c0, r1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = s8[5,8] parameter(0)
  p1 = f16[2,3,4] parameter(1)
  ROOT custom-call = f16[5,3] custom-call(p0, p1),
    custom_call_target="__triton", called_computations={triton_dot}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  const HloInstruction* dot = dot_computation->root_instruction();
  const DotFusionAnalysis analysis(dot);
  EXPECT_EQ(analysis.OperandToParameter(0),
            dot_computation->parameter_instruction(0));
  EXPECT_EQ(analysis.OperandToParameter(1),
            dot_computation->parameter_instruction(1));
  EXPECT_THAT(analysis.IterSpec(0, 0),
              ElementsAre(FieldsAre(/*stride=*/8, /*count=*/5)));
  EXPECT_THAT(analysis.IterSpec(0, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8)));
  EXPECT_THAT(analysis.IterSpec(1, 0),
              ElementsAre(FieldsAre(/*stride=*/4, /*count=*/3)));
  EXPECT_THAT(analysis.IterSpec(1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/4),
                          FieldsAre(/*stride=*/12, /*count=*/2)));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
