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

#include "xla/service/hlo_unstacker.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using UnstackerTest = HloTestBase;

int64_t GetInstrCountWithOpcodeInEntry(HloModule* module, HloOpcode opcode) {
  int64_t instr_with_opcode_count = 0;
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() == opcode) {
      instr_with_opcode_count++;
    }
  }
  return instr_with_opcode_count;
}

TEST_F(UnstackerTest, UnstackDSFusionPattern) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] p1, i), kind=kLoop, calls=%fused_computation.slice
    conv = bf16[8,128] convolution(bf16[8,128] p0, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, conv, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 3);
  // Check that the bitcast is unfused and there are not fusions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kFusion),
            0);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, NotUnstackDSFusionPattern) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %fused_computation.tuple {
    %param_0.51117 = s8[3,128,128] parameter(0)
    mult = multiply(param_0.51117, param_0.51117)
    ROOT out = tuple(param_0.51117, mult)
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] p1, i), kind=kLoop, calls=%fused_computation.slice
    conv = bf16[8,128] convolution(bf16[8,128] p0, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    fusion_mult = (s8[3,128,128], s8[3,128,128]) fusion(s8[3,128,128] p1), kind=kLoop, calls=%fused_computation.tuple
    mult = s8[3,128,128] get-tuple-element(fusion_mult), index=1
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, conv, mult)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_FALSE(unstacked);
}

TEST_F(UnstackerTest, UnstackDSFusionPatternMultipleLoopRootUse) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p2 = s8[3,128,128] get-tuple-element(wide_p), index=3
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] p2, i), kind=kLoop, calls=%fused_computation.slice
    conv = bf16[8,128] convolution(bf16[8,128] p0, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    ROOT out = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) tuple(inc, conv, p2, p2)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    zero = s8[] constant(0)
    buffer = s8[3,128,128] broadcast(zero), dimensions={}
    while.input = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) tuple(init, p1, p0, buffer)
    while.out = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 6);
  // Check that the bitcast is unfused and there are not fusions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kFusion),
            0);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackDSFusionPatternWithUnusedOperand) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] p1, i), kind=kLoop, calls=%fused_computation.slice
    conv = bf16[8,128] convolution(bf16[8,128] p0, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    ROOT out = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) tuple(inc, conv, p1, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    zero = s8[] constant(0)
    buffer = s8[3,128,128] broadcast(zero), dimensions={}
    while.input = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) tuple(init, p1, p0, buffer)
    while.out = (s32[], bf16[8,128], s8[3,128,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 6);
  // Check that the bitcast is unfused and there are not fusions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kFusion),
            0);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackReduceFusionPattern) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  dynamic-slice.609.reduce_sub_computation {
    lhs.53 = s8[] parameter(0)
    rhs.53 = s8[] parameter(1)
    ROOT add.3090 = s8[] add(lhs.53, rhs.53)
  }

  fused_computation.1096.clone {
    param_0.5572 = s8[3,128,128] parameter(0)
    param_1.6711 = s32[]{:T(128)} parameter(1)
    constant.12008 = s32[]{:T(128)} constant(0)
    dynamic-slice.1545 = s8[1,128,128] dynamic-slice(param_0.5572, param_1.6711, constant.12008, constant.12008), dynamic_slice_sizes={1,128, 128}
    constant.12009 = s8[] constant(-0)
    ROOT reduce.919 = s8[128,128] reduce(dynamic-slice.1545, constant.12009), dimensions={0}, to_apply=dynamic-slice.609.reduce_sub_computation
  } // fused_computation.1096.clone

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] p1, i), kind=kLoop, calls=%fused_computation.1096.clone
    conv = bf16[8,128] convolution(bf16[8,128] p0, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, conv, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackDSFusionPatternNoBitcast) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[1,128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    ROOT %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67830 = s8[1,128,128] fusion(s8[3,128,128] p1, i), kind=kLoop, calls=%fused_computation.slice
    bitcast.102 = s8[128,128] bitcast(s8[1,128,128] %fusion.67830)
    conv = bf16[8,128] convolution(bf16[8,128] p0, s8[128,128] bitcast.102), dim_labels=bf_io->bf
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, conv, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 3);
  // Check that all the fusions are removed.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kFusion),
            0);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackDSFusionPatternNoBitcastKeepFused) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[1,128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    ROOT %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67830 = s8[1,128,128] fusion(s8[3,128,128] p1, i), kind=kLoop, calls=%fused_computation.slice
    bitcast.102 = s8[128,128] bitcast(s8[1,128,128] %fusion.67830)
    conv = bf16[8,128] convolution(bf16[8,128] p0, s8[128,128] bitcast.102), dim_labels=bf_io->bf
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, conv, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  auto unfuse = [](HloInstruction* instruction) { return false; };
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked,
                          HloUnstacker(unfuse).Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 0);
  // Check that dynamic-slices are still fused.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kFusion),
            3);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackDSFusionPatternKeepFused) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT out = s8[128,128] bitcast(%dynamic-slice.22040)
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] p1, i), kind=kLoop, calls=%fused_computation.slice
    conv = bf16[8,128] convolution(bf16[8,128] p0, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, conv, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  auto unfuse = [](HloInstruction* instruction) { return false; };
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked,
                          HloUnstacker(unfuse).Run(module.get()));
  EXPECT_FALSE(unstacked);
}

TEST_F(UnstackerTest, UnstackDSFusionPatternWithDifferentLayout) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.30.clone (param_0.153: bf16[32,4,64,64,3], param_1.123: s32[]) -> bf16[64,4,64,3] {
    %param_0.153 = bf16[32,4,64,64,3]{2,1,4,3,0} parameter(0)
    %param_1.123 = s32[]{:T(128)} parameter(1)
    %constant.227 = s32[]{:T(128)} constant(0)
    %dynamic-slice.5 = bf16[1,4,64,64,3]{2,1,4,3,0} dynamic-slice(bf16[32,4,64,64,3]{2,1,4,3,0} %param_0.153, s32[]{:T(128)} %param_1.123, s32[]{:T(128)} %constant.227, s32[]{:T(128)} %constant.227, s32[]{:T(128)} %constant.227, /*index=5*/s32[]{:T(128)} %constant.227), dynamic_slice_sizes={1,4,64,64,3}
    ROOT %bitcast.102 = bf16[64,4,64,3]{0,1,3,2} bitcast(bf16[1,4,64,64,3]{2,1,4,3,0} %dynamic-slice.5)
  }

  %while.body (wide_param: (s32[], bf16[8,128], bf16[32,4,64,64,3])) -> (s32[], bf16[8,128], bf16[32,4,64,64,3]) {
    wide_p = (s32[], bf16[8,128], bf16[32,4,64,64,3]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = bf16[32,4,64,64,3]{2,1,4,3,0} get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67830 = bf16[64,4,64,3]{0,1,3,2} fusion(p1, i), kind=kLoop, calls=%fused_computation.30.clone
    ROOT out = (s32[], bf16[8,128], bf16[32,4,64,64,3]) tuple(inc, p0, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], bf16[32,4,64,64,3])) -> pred[] {
    wide_p = (s32[], bf16[8,128], bf16[32,4,64,64,3]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(32)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = bf16[32,4,64,64,3] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], bf16[32,4,64,64,3]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], bf16[32,4,64,64,3]) while(while.input), condition=%while.cond , body=%while.body
    while_use = bf16[32,4,64,64,3] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice),
            32);
  // Check that dynamic-slices are still fused.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kFusion),
            0);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt));
}

TEST_F(UnstackerTest, UnstackNestedDSFusionPattern) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %fused_computation.inner (param_0.34523: bf16[8,128], param_1.30691: s8[3,128,128], p2: s32[]) -> bf16[8,128] {
    %param_0.34523 = bf16[8,128] parameter(0)
    %param_1.30691 = s8[3,128,128] parameter(1)
    p2 = s32[] parameter(2)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice
    ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    fusion.conv = bf16[8,128] fusion(p0, p1, i), kind=kOutput, calls=%fused_computation.inner
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, fusion.conv, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 3);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

// Instead of slicing the entire shape, this test slices only even elements from
// the first parameter.
TEST_F(UnstackerTest, UnstackNestedDSFusionPatternWithDynamicIndex) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[6,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[6,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[6,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %fused_computation.inner (param_0.34523: bf16[8,128], param_1.30691: s8[6,128,128], p2: s32[]) -> bf16[8,128] {
    %param_0.34523 = bf16[8,128] parameter(0)
    %param_1.30691 = s8[6,128,128] parameter(1)
    p2 = s32[] parameter(2)
    %fusion.67830 = s8[128,128] fusion(s8[6,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice
    ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[6,128,128])) -> (s32[], bf16[8,128], s8[6,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[6,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[6,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    two = s32[] constant(2)
    mult = s32[] multiply(i, two)
    fusion.conv = bf16[8,128] fusion(p0, p1, mult), kind=kOutput, calls=%fused_computation.inner
    ROOT out = (s32[], bf16[8,128], s8[6,128,128]) tuple(inc, fusion.conv, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[6,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[6,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[6,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[6,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[6,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[6,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackNestedDSFusionPatternWithMultipleIndex) {
  std::string hlo_string = R"(
    HloModule SimpleLoop
    %fused_computation.slice.1 (param_0.51117: s8[4,128,128], p1: s32[]) -> s8[128,128] {
      %param_0.51117 = s8[4,128,128] parameter(0)
      p1 = s32[] parameter(1)
      %constant.85694 = s32[] constant(0)
      %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[4,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
      ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
    }

    %fused_computation.slice.2 (param_0.51117: s8[4,128,128], p1: s32[]) -> s8[128,128] {
      %param_0.51117 = s8[4,128,128] parameter(0)
      p1 = s32[] parameter(1)
      %constant.85694 = s32[] constant(0)
      %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[4,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
      ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
    }

    %fused_computation.inner.1 (param_0.34523: bf16[8,128], param_1.30691: s8[4,128,128], p2: s32[]) -> bf16[8,128] {
      %param_0.34523 = bf16[8,128] parameter(0)
      %param_1.30691 = s8[4,128,128] parameter(1)
      p2 = s32[] parameter(2)
      %fusion.67830 = s8[128,128] fusion(s8[4,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice.1
      ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    }

    %fused_computation.inner.2 (param_0.34523: bf16[8,128], param_1.30691: s8[4,128,128], p2: s32[]) -> bf16[8,128] {
      %param_0.34523 = bf16[8,128] parameter(0)
      %param_1.30691 = s8[4,128,128] parameter(1)
      p2 = s32[] parameter(2)
      %fusion.67830 = s8[128,128] fusion(s8[4,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice.2
      ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    }

    %while.body (wide_param: (s32[], bf16[8,128], s8[4,128,128], s8[4,128,128])) -> (s32[], bf16[8,128], s8[4,128,128], s8[4,128,128]) {
      wide_p = (s32[], bf16[8,128], s8[4,128,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      p0 = bf16[8,128] get-tuple-element(wide_p), index=1
      // to_be_sliced_while_gte
      p1 = s8[4,128,128] get-tuple-element(wide_p), index=2
      p2 = s8[4,128,128] get-tuple-element(wide_p), index=3
      one = s32[] constant(1)
      inc = s32[] add(i, one)
      fusion.conv.1 = bf16[8,128] fusion(p0, p1, i), kind=kOutput, calls=%fused_computation.inner.1
      fusion.conv.2 = bf16[8,128] fusion(p0, p2, i), kind=kOutput, calls=%fused_computation.inner.2
      plus = bf16[8,128] add(fusion.conv.1, fusion.conv.2)
      ROOT out = (s32[], bf16[8,128], s8[4,128,128], s8[4,128,128]) tuple(inc, plus, p1, p2)
    }

    %while.cond (wide_param: (s32[], bf16[8,128], s8[4,128,128], s8[4,128,128])) -> pred[] {
      wide_p = (s32[], bf16[8,128], s8[4,128,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      %constant.12857 = s32[] constant(4)
      ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
    }

    ENTRY main {
      p0 = s8[4,128,128] parameter(0)
      p1 = s8[4,128,128] parameter(1)
      p2 = bf16[8,128] parameter(2)
      init = s32[] constant(0)
      while.input = (s32[], bf16[8,128], s8[4,128,128], s8[4,128,128]) tuple(init, p2, p0, p1)
      while.out = (s32[], bf16[8,128], s8[4,128,128], s8[4,128,128]) while(while.input), condition=%while.cond , body=%while.body
      ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions. For each unstacked input, we
  // create 4 slices, 8 in total.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 8);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackNestedDSFusionPatternWithDiffereOperandsOrder) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %fused_computation.inner (param_1.30691: s8[3,128,128], p2: s32[], param_0.34523: bf16[8,128]) -> bf16[8,128] {
    %param_0.34523 = bf16[8,128] parameter(2)
    %param_1.30691 = s8[3,128,128] parameter(0)
    p2 = s32[] parameter(1)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice
    ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    fusion.conv = bf16[8,128] fusion(p1, i, p0), kind=kOutput, calls=%fused_computation.inner
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, fusion.conv, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 3);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackNestedDSFusionPatternWithSameUnstackingComps) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice.1 (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %fused_computation.slice.2 (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %fused_computation.inner.1 (param_0.34523: bf16[8,128], param_1.30691: s8[3,128,128], p2: s32[]) -> bf16[8,128] {
    %param_0.34523 = bf16[8,128] parameter(0)
    %param_1.30691 = s8[3,128,128] parameter(1)
    p2 = s32[] parameter(2)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice.1
    ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
  }

  %fused_computation.inner.2 (param_0.34523: bf16[8,128], param_1.30691: s8[3,128,128], p2: s32[]) -> bf16[8,128] {
    %param_0.34523 = bf16[8,128] parameter(0)
    %param_1.30691 = s8[3,128,128] parameter(1)
    p2 = s32[] parameter(2)
    %fusion.67830 = s8[128,128] fusion(s8[3,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice.2
    ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    fusion.conv1 = bf16[8,128] fusion(p0, p1, i), kind=kOutput, calls=%fused_computation.inner.1
    fusion.conv2 = bf16[8,128] fusion(p0, p1, i), kind=kOutput, calls=%fused_computation.inner.2
    add = bf16[8,128] add(fusion.conv1, fusion.conv2)
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, add, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 3);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest,
       NotUnstackNestedDSFusionPatternWithDifferentUnstackingComps) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice.1 (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[1,128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    ROOT %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
  }

  %fused_computation.slice.2 (param_0.51117: s8[3,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[3,128,128] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[3,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> (s32[], bf16[8,128], s8[3,128,128]) {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[3,128,128] get-tuple-element(wide_p), index=2
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    %fusion.67831 = s8[128,128] fusion(p1, i), kind=kLoop, calls=%fused_computation.slice.2
    %fusion.67830 = s8[1,128,128] fusion(p1, i), kind=kLoop, calls=%fused_computation.slice.1
    %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %fusion.67830)
    ROOT out = (s32[], bf16[8,128], s8[3,128,128]) tuple(inc, p0, p1)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[3,128,128])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[3,128,128]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[3,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[3,128,128]) tuple(init, p1, p0)
    while.out = (s32[], bf16[8,128], s8[3,128,128]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[3,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Currently, we don't unroll if there are multiple nested ds fusions.
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_FALSE(unstacked);
}

TEST_F(UnstackerTest, UnstackNestedDSFusionPatternSingleNestedLoop) {
  std::string hlo_string = R"(
    HloModule SimpleLoop
    %fused_computation.slice (param_0.51117: s8[4,128,128], p1: s32[]) -> s8[128,128] {
      %param_0.51117 = s8[4,128,128] parameter(0)
      p1 = s32[] parameter(1)
      %constant.85694 = s32[] constant(0)
      %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[4,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
      ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
    }

    %fused_computation.inner (param_0.34523: bf16[8,128], param_1.30691: s8[4,128,128], p2: s32[]) -> bf16[8,128] {
      %param_0.34523 = bf16[8,128] parameter(0)
      %param_1.30691 = s8[4,128,128] parameter(1)
      p2 = s32[] parameter(2)
      %fusion.67830 = s8[128,128] fusion(s8[4,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice
      ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    }

    %while.body.inner (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> (s32[], bf16[8,128], s8[4,128,128]) {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      inner_param_0 = bf16[8,128] get-tuple-element(wide_p), index=1
      inner_param_1 = s8[4,128,128] get-tuple-element(wide_p), index=2
      one = s32[] constant(1)
      inc = s32[] add(i, one)
      fusion.conv = bf16[8,128] fusion(inner_param_0, inner_param_1, i), kind=kOutput, calls=%fused_computation.inner
      ROOT out = (s32[], bf16[8,128], s8[4,128,128]) tuple(inc, fusion.conv, inner_param_1)
    }

    %while.cond.inner (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> pred[] {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      %constant.12857 = s32[] constant(4)
      ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
    }

    %while.body (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> (s32[], bf16[8,128], s8[4,128,128]) {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      param0 = bf16[8,128] get-tuple-element(wide_p), index=1
      param1 = s8[4,128,128] get-tuple-element(wide_p), index=2
      one = s32[] constant(2)
      zero = s32[] constant(0)
      mult = s32[] multiply(i, one)
      inner.in = (s32[], bf16[8,128], s8[4,128,128]) tuple(zero, param0, param1)
      inner.out = (s32[], bf16[8,128], s8[4,128,128]) while(inner.in), condition=%while.cond.inner, body=%while.body.inner
      fusion.conv.inner = bf16[8,128] get-tuple-element(inner.out), index=1
      ROOT out = (s32[], bf16[8,128], s8[4,128,128]) tuple(mult, fusion.conv.inner, param1)
    }

    %while.cond (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> pred[] {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      %constant.12857 = s32[] constant(20)
      add = s32[] add(%constant.12857, %constant.12857)
      ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, add), direction=LT
    }

    ENTRY main {
      weight = s8[4,128,128] parameter(0)
      p1 = bf16[8,128] parameter(1)
      init = s32[] constant(1)
      while.input = (s32[], bf16[8,128], s8[4,128,128]) tuple(init, p1, weight)
      while.out = (s32[], bf16[8,128], s8[4,128,128]) while(while.input), condition=%while.cond , body=%while.body
      ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 4);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackNestedDSFusionPatternTwoNestedLoops) {
  std::string hlo_string = R"(
    HloModule SimpleLoop
    %fused_computation.slice1 (param_0.51117: s8[4,128,128], p1: s32[]) -> s8[128,128] {
      %param_0.51117 = s8[4,128,128] parameter(0)
      p1 = s32[] parameter(1)
      %constant.85694 = s32[] constant(0)
      %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[4,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
      ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
    }

    %fused_computation.inner1 (param_0.34523: bf16[8,128], param_1.30691: s8[4,128,128], p2: s32[]) -> bf16[8,128] {
      %param_0.34523 = bf16[8,128] parameter(0)
      %param_1.30691 = s8[4,128,128] parameter(1)
      p2 = s32[] parameter(2)
      %fusion.67830 = s8[128,128] fusion(s8[4,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice1
      ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    }

    %while.body.inner1 (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> (s32[], bf16[8,128], s8[4,128,128]) {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      inner_param_0 = bf16[8,128] get-tuple-element(wide_p), index=1
      inner_param_1 = s8[4,128,128] get-tuple-element(wide_p), index=2
      one = s32[] constant(1)
      inc = s32[] add(i, one)
      fusion.conv = bf16[8,128] fusion(inner_param_0, inner_param_1, i), kind=kOutput, calls=%fused_computation.inner1
      ROOT out = (s32[], bf16[8,128], s8[4,128,128]) tuple(inc, fusion.conv, inner_param_1)
    }

    %while.cond.inner1 (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> pred[] {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      %constant.12857 = s32[] constant(4)
      ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
    }

    %while.body1 (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> (s32[], bf16[8,128], s8[4,128,128]) {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      param0 = bf16[8,128] get-tuple-element(wide_p), index=1
      param1 = s8[4,128,128] get-tuple-element(wide_p), index=2
      one = s32[] constant(2)
      zero = s32[] constant(0)
      mult = s32[] multiply(i, one)
      inner.in.1 = (s32[], bf16[8,128], s8[4,128,128]) tuple(zero, param0, param1)
      inner.out.1 = (s32[], bf16[8,128], s8[4,128,128]) while(inner.in.1), condition=%while.cond.inner1, body=%while.body.inner1
      fusion.conv.inner = bf16[8,128] get-tuple-element(inner.out.1), index=1
      ROOT out = (s32[], bf16[8,128], s8[4,128,128]) tuple(mult, fusion.conv.inner, param1)
    }

    %while.cond1 (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> pred[] {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      %constant.12857 = s32[] constant(20)
      add = s32[] add(%constant.12857, %constant.12857)
      ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, add), direction=LT
    }

    %fused_computation.slice2 (param_0.51117: s8[4,128,128], p1: s32[]) -> s8[128,128] {
      %param_0.51117 = s8[4,128,128] parameter(0)
      p1 = s32[] parameter(1)
      %constant.85694 = s32[] constant(0)
      %dynamic-slice.22040 = s8[1,128,128] dynamic-slice(s8[4,128,128] %param_0.51117, p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
      ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.22040)
    }

    %fused_computation.inner2 (param_0.34523: bf16[8,128], param_1.30691: s8[4,128,128], p2: s32[]) -> bf16[8,128] {
      %param_0.34523 = bf16[8,128] parameter(0)
      %param_1.30691 = s8[4,128,128] parameter(1)
      p2 = s32[] parameter(2)
      %fusion.67830 = s8[128,128] fusion(s8[4,128,128] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice2
      ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] %fusion.67830), dim_labels=bf_io->bf
    }

    %while.body.inner2 (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> (s32[], bf16[8,128], s8[4,128,128]) {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      inner_param_0 = bf16[8,128] get-tuple-element(wide_p), index=1
      inner_param_1 = s8[4,128,128] get-tuple-element(wide_p), index=2
      one = s32[] constant(1)
      inc = s32[] add(i, one)
      fusion.conv = bf16[8,128] fusion(inner_param_0, inner_param_1, i), kind=kOutput, calls=%fused_computation.inner2
      ROOT out = (s32[], bf16[8,128], s8[4,128,128]) tuple(inc, fusion.conv, inner_param_1)
    }

    %while.cond.inner2 (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> pred[] {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      %constant.12857 = s32[] constant(4)
      ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
    }

    %while.body2 (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> (s32[], bf16[8,128], s8[4,128,128]) {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      param0 = bf16[8,128] get-tuple-element(wide_p), index=1
      param1 = s8[4,128,128] get-tuple-element(wide_p), index=2
      one = s32[] constant(2)
      zero = s32[] constant(0)
      mult = s32[] multiply(i, one)
      inner.in.2 = (s32[], bf16[8,128], s8[4,128,128]) tuple(zero, param0, param1)
      inner.out.2 = (s32[], bf16[8,128], s8[4,128,128]) while(inner.in.2), condition=%while.cond.inner2, body=%while.body.inner2
      fusion.conv.inner = bf16[8,128] get-tuple-element(inner.out.2), index=1
      ROOT out = (s32[], bf16[8,128], s8[4,128,128]) tuple(mult, fusion.conv.inner, param1)
    }

    %while.cond2 (wide_param: (s32[], bf16[8,128], s8[4,128,128])) -> pred[] {
      wide_p = (s32[], bf16[8,128], s8[4,128,128]) parameter(0)
      i = s32[] get-tuple-element(wide_p), index=0
      %constant.12857 = s32[] constant(20)
      add = s32[] add(%constant.12857, %constant.12857)
      ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, add), direction=LT
    }

    ENTRY main {
      weight = s8[4,128,128] parameter(0)
      p1 = bf16[8,128] parameter(1)
      init = s32[] constant(1)
      while.input = (s32[], bf16[8,128], s8[4,128,128]) tuple(init, p1, weight)
      while.out = (s32[], bf16[8,128], s8[4,128,128]) while(while.input), condition=%while.cond1 , body=%while.body1
      init2 = s32[] get-tuple-element(while.out), index=0
      second.while.input = (s32[], bf16[8,128], s8[4,128,128]) tuple(init2, p1, weight)
      second.while.out = (s32[], bf16[8,128], s8[4,128,128]) while(second.while.input), condition=%while.cond2 , body=%while.body2
      out = bf16[8,128] get-tuple-element(while.out), index=1
      second.out = bf16[8,128] get-tuple-element(second.while.out), index=1
      ROOT result = bf16[8,128] add(out, second.out)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  // Check for the creation of slice instructions. For each loop there is one
  // unstacked input that creates 4 slices, in total 8 slices for two loops.
  EXPECT_EQ(GetInstrCountWithOpcodeInEntry(module.get(), HloOpcode::kSlice), 8);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackDSAndDUSPattern) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s32[4,3], offset: s32[]) -> s32[3] {
    %param_0.51117 = s32[4,3] parameter(0)
    offset = s32[] parameter(1)
    zero = s32[] constant(0)
    %dynamic-slice.22040 = s32[1,3] dynamic-slice(s32[4,3] %param_0.51117, offset, zero), dynamic_slice_sizes={1,3}
    ROOT %bitcast.31250 = s32[3] bitcast(s32[1,3] %dynamic-slice.22040)
  }

  %fused_computation.update.slice (param_0.51117: s32[4,3], p1: s32[], p2: s32[3]) -> s32[4,3] {
    %param_0.51117 = s32[4,3] parameter(0)
    %p1 = s32[] parameter(1)
    %p2 = s32[3] parameter(2)
    %zero = s32[] constant(0)
    %bitcast.31250 = s32[1,3] bitcast(%p2)
    ROOT output_dus = s32[4,3]{1,0} dynamic-update-slice(%param_0.51117, %bitcast.31250, %p1, zero)
  }

  SimpleLoop.body {
    loop_var.1 = (s32[], s32[4,3]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[4,3] get-tuple-element(loop_var.1), index=1
    zero = s32[] constant(0)

    some_const = s32[3] constant({0,1,2})
    constant.1 = s32[] constant(1)
    idx = s32[] add(get-tuple-element.1, constant.1)
    ds = s32[3]{0} fusion(get-tuple-element.2, get-tuple-element.1), kind=kLoop, calls=%fused_computation.slice
    update = s32[3] add(ds, ds)
    dus = s32[3] dynamic-update-slice(ds, update, zero)
    output = s32[4,3] fusion(get-tuple-element.2, get-tuple-element.1, dus), kind=kLoop, calls=%fused_computation.update.slice
    ROOT tuple = (s32[], s32[4,3]) tuple(idx, output)
  }
  SimpleLoop.condition {
    loop_var.1 = (s32[], s32[4,3]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.2 = s32[] constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.1, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    reference = s32[4,3] parameter(0)
    zero = s32[] constant(0)
    zero1 = s32[] constant(0)
    one = s32[] constant(1)
    tuple.1 = (s32[], s32[4,3]) tuple(zero, reference)
    while = (s32[], s32[4,3]) while(tuple.1), condition=SimpleLoop.condition, body=SimpleLoop.body
    ROOT out = s32[] get-tuple-element(while), index=0
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

// Unstacking outer loop at index 1 forces to unstacked inner while at index 1
// as well. This is because the output of the outer loop at index 1 is aliased
// to the output of the inner while at index 1.
TEST_F(UnstackerTest, UnstackDSAndDUSPatternNestedLoop) {
  std::string hlo_string = R"(
  HloModule SimpleLoop

  %fused_computation.slice (param_0.51117: bf16[4,1,8,257,128], offset: s32[]) -> bf16[1,8,257,128] {
    %param_0.51117 = bf16[4,1,8,257,128] parameter(0)
    offset = s32[] parameter(1)
    zero = s32[] constant(0)
    %dynamic-slice.22040 = bf16[1,1,8,257,128]
    dynamic-slice(bf16[4,1,8,257,128] %param_0.51117, offset, zero, zero, zero, zero), dynamic_slice_sizes={1,1,8,257,128}
    ROOT %bitcast.31250 = bf16[1,8,257,128] bitcast(%dynamic-slice.22040)
  }

  %fused_computation.slice.2 (param_0.51117: bf16[4,1,8,257,128], offset: s32[]) -> bf16[1,8,257,128] {
    %param_0.51117 = bf16[4,1,8,257,128] parameter(0)
    offset = s32[] parameter(1)
    zero = s32[] constant(0)
    %dynamic-slice.22040 = bf16[1,1,8,257,128] dynamic-slice(bf16[4,1,8,257,128] %param_0.51117, offset, zero, zero, zero, zero), dynamic_slice_sizes={1,1,8,257,128}
    ROOT %bitcast.31250 = bf16[1,8,257,128] bitcast(%dynamic-slice.22040)
  }

  inner.body {
    loop_var.1 = (s32[], bf16[4,1,8,257,128], bf16[4,1,8,257,128]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = bf16[4,1,8,257,128] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = bf16[4,1,8,257,128] get-tuple-element(loop_var.1), index=2
    sliced = bf16[1,8,257,128] fusion(get-tuple-element.2, get-tuple-element.1), kind=kLoop, calls=%fused_computation.slice
    sliced.2 = bf16[1,8,257,128] fusion(get-tuple-element.3, get-tuple-element.1), kind=kLoop,calls=%fused_computation.slice.2
    temp = bf16[1,8,257,128] add(sliced, sliced.2)
    one = s32[] constant(1) idx = s32[] add(get-tuple-element.1, one)
    ROOT out = tuple(idx, get-tuple-element.2, get-tuple-element.3)
  }
  inner.condition {
    loop_var.1 = (s32[], bf16[4,1,8,257,128], bf16[4,1,8,257,128])
    parameter(0) get-tuple-element.1 = s32[] get-tuple-element(loop_var.1),
    index=0 constant.2 = s32[] constant(4) ROOT less-than = pred[]
    compare(get-tuple-element.1, constant.2), direction=LT
  }

  outer.body {
    loop_var.1 = (s32[], bf16[4,1,8,257,128], bf16[4,1,8,257,128]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = bf16[4,1,8,257,128] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = bf16[4,1,8,257,128] get-tuple-element(loop_var.1), index=2
    zero = s32[] constant(0)
    buffer = bf16[4,1,8,257,128] custom-call(), custom_call_target="AllocateBuffer"
    inner.input = tuple(zero, buffer, get-tuple-element.2)
    inner = while(inner.input), condition=inner.condition, body=inner.body
    out1 = bf16[4,1,8,257,128] get-tuple-element(inner), index=1
    one = s32[] constant(1)
    idx = s32[] add(get-tuple-element.1, one)
    ROOT tuple = (s32[], bf16[4,1,8,257,128], bf16[4,1,8,257,128]) tuple(idx, out1, get-tuple-element.3)
  }
  outer.condition {
    loop_var.1 = (s32[], bf16[4,1,8,257,128], bf16[4,1,8,257,128])
    parameter(0) get-tuple-element.1 = s32[] get-tuple-element(loop_var.1),
    index=0 constant.2 = s32[] constant(4) mul = s32[]
    multiply(get-tuple-element.1, constant.2) ROOT less-than = pred[]
    compare(get-tuple-element.1, mul), direction=LT
  }

  ENTRY SimpleLoop {
    param1 = bf16[4,1,8,257,128] parameter(0)
    param2 = bf16[4,1,8,257,128] parameter(1)
    zero = s32[] constant(0)
    zero1 = s32[] constant(0)
    one = s32[] constant(1)
    tuple.1 = tuple(zero, param1, param2)
    while = while(tuple.1), condition=outer.condition, body=outer.body
    ROOT out = s32[] get-tuple-element(while), index=0
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

// Unstacking the first loop at index 1 forces to unstack the second loop at
// index 1 as well.
TEST_F(UnstackerTest, UnstackDSAndDUSPatternLoopFeedingLoop) {
  std::string hlo_string = R"(
  HloModule SimpleLoop

  %fused_computation.update.slice (param_0.51117: bf16[4,1,8,257,128], p1: s32[], param_0.51118: bf16[1,8,257,128]) -> bf16[4,1,8,257,128] {
    %param_0.51117 = bf16[4,1,8,257,128] parameter(0)
    p1 = s32[] parameter(1)
    %param_0.51118 = bf16[1,8,257,128] parameter(2)
    bitcast = bf16[1,1,8,257,128] bitcast(param_0.51118)
    %constant.85694 = s32[] constant(0)
    ROOT %dynamic-update-slice.22040 = bf16[4,1,8,257,128] dynamic-update-slice(bf16[4,1,8,257,128] %param_0.51117, bitcast, p1, s32[] %constant.85694, s32[] %constant.85694, s32[] %constant.85694, s32[] %constant.85694)
  }

  %fused_computation.slice (param_0.51117: bf16[4,1,8,257,128], offset:s32[]) -> bf16[1,8,257,128] {
    %param_0.51117 = bf16[4,1,8,257,128] parameter(0)
    offset = s32[] parameter(1)
    zero = s32[] constant(0)
    %dynamic-slice.22040 = bf16[1,1,8,257,128] dynamic-slice(bf16[4,1,8,257,128] %param_0.51117, offset, zero, zero, zero, zero), dynamic_slice_sizes={1,1,8,257,128}
    ROOT %bitcast.31250 = bf16[1,8,257,128] bitcast(%dynamic-slice.22040)
  }

  first.body {
    loop_var.1 = (s32[], bf16[4,1,8,257,128]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1),index=0
    get-tuple-element.2 = bf16[4,1,8,257,128] get-tuple-element(loop_var.1), index=1
    constant = bf16[1,8,257,128] constant({...})
    sliced = bf16[1,8,257,128] fusion(get-tuple-element.2, get-tuple-element.1), kind=kLoop, calls=%fused_computation.slice
    tmp = bf16[1,8,257,128] add(sliced, sliced)
    one = s32[] constant(1)
    idx = s32[] add(get-tuple-element.1, one)
    ROOT out = tuple(idx, get-tuple-element.2)
  }
  first.condition {
    loop_var.1 = (s32[], bf16[4,1,8,257,128]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.2 = s32[] constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.1, constant.2), direction=LT
  }

  next.body {
    loop_var.1 = (s32[], bf16[4,1,8,257,128]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1),index=0
    get-tuple-element.2 = bf16[4,1,8,257,128] get-tuple-element(loop_var.1), index=1
    constant = bf16[1,8,257,128] constant({...})
    update.sliced = bf16[4,1,8,257,128] fusion(get-tuple-element.2, get-tuple-element.1, constant), kind=kLoop, calls=%fused_computation.update.slice
    one = s32[] constant(1)
    idx = s32[] add(get-tuple-element.1, one)
    ROOT out = tuple(idx, update.sliced)
  }
  next.condition {
    loop_var.1 = (s32[], bf16[4,1,8,257,128]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.2 = s32[] constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.1, constant.2), direction=LT
  }

  ENTRY SimpleLoop {
    param1 = bf16[4,1,8,257,128] parameter(0)
    param2 = bf16[4,1,8,257,128] parameter(1)
    zero = s32[] constant(0)
    zero1 = s32[] constant(0)
    one = s32[] constant(1)
    tuple.1 = tuple(zero, param1)
    while = while(tuple.1), condition=first.condition, body=first.body
    while.out = bf16[4,1,8,257,128] get-tuple-element(while), index=1
    next.input = tuple(zero, while.out)
    next = while(next.input), condition=next.condition, body=next.body
    ROOT out = s32[] get-tuple-element(next), index=0
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
}

TEST_F(UnstackerTest, UnstackDUSFusionWithPadPatternLoopFeedingLoop) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  fused_computation.75.clone {
    param_0.5713 = bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)} parameter(0)
    param_2.4396 = bf16[1,8,257,128]{3,2,1,0:T(8,128)(2,1)} parameter(2)
    constant.12166 = bf16[]{:T(256)} constant(0)
    pad.496 = bf16[1,8,513,128]{3,2,1,0:T(8,128)(2,1)} pad(param_2.4396, constant.12166), padding=0_0x0_0x0_256x0_0
    bitcast.1262 = bf16[1,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)} bitcast(pad.496)
    param_1.6823 = s32[]{:T(128)} parameter(1)
    constant.12165 = s32[]{:T(128)} constant(0)
    ROOT dynamic-update-slice.193 = bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)} dynamic-update-slice(param_0.5713, bitcast.1262, param_1.6823, constant.12165, constant.12165, /*index=5*/constant.12165, constant.12165)
  } // fused_computation.75.clone

  fused_computation.1 {
    param_0.5712 = bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)}parameter(0)
    param_1.6822 = s32[]{:T(128)} parameter(1)
    constant.12164 = s32[]{:T(128)} constant(0)
    dynamic-slice.1597 = bf16[1,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)} dynamic-slice(param_0.5712, param_1.6822, constant.12164, constant.12164, constant.12164, /*index=5*/constant.12164), dynamic_slice_sizes={1,1,8,513,128}
    ROOT bitcast.1261 = bf16[1,8,513,128]{3,2,1,0:T(8,128)(2,1)} bitcast(dynamic-slice.1597)
  }

  first.body {
    wide.param.29 = (s32[]{:T(128)}, bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)}) parameter(0)
    get-tuple-element.12177 = s32[]{:T(128)} get-tuple-element(wide.param.29), index=0
    constant.12144..sunk.2 = s32[]{:T(128)} constant(1)
    add.4517 = s32[]{:T(128)} add(get-tuple-element.12177, constant.12144..sunk.2)
    get-tuple-element.12178 = bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)} get-tuple-element(wide.param.29), index=1
    fusion.2381 = bf16[1,8,513,128]{3,2,1,0:T(8,128)(2,1)} fusion(get-tuple-element.12178, get-tuple-element.12177), kind=kLoop, calls=fused_computation.1
    tmp = bf16[1,8,513,128]{3,2,1,0:T(8,128)(2,1)} add(fusion.2381, fusion.2381)
    ROOT tuple.949 = (s32[]{:T(128)}, bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)}) tuple(add.4517, get-tuple-element.12178)
  } // wide.region_54.2652.clone_spmd

  first.cond {
    wide.param.28 = (s32[]{:T(128)}, bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)}) parameter(0)
    get-tuple-element.12167 = s32[]{:T(128)} get-tuple-element(wide.param.28), index=0
    constant.12162 = s32[]{:T(128)} constant(2)
    ROOT compare.1815 = pred[]{:T(512)} compare(get-tuple-element.12167, constant.12162), direction=LT
  }

  wide.region_54.2652.clone_spmd {
    wide.param.29 = (s32[]{:T(128)}, bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)}) parameter(0)
    get-tuple-element.12177 = s32[]{:T(128)} get-tuple-element(wide.param.29), index=0
    constant.12144..sunk.2 = s32[]{:T(128)} constant(1)
    add.4517 = s32[]{:T(128)} add(get-tuple-element.12177, constant.12144..sunk.2)
    get-tuple-element.12178 = bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)} get-tuple-element(wide.param.29), index=1
    update = bf16[1,8,257,128]{3,2,1,0:T(8,128)(2,1)} constant({...})
    fusion.2382 = bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)} fusion(get-tuple-element.12178, get-tuple-element.12177, update), kind=kLoop, calls=fused_computation.75.clone
    ROOT tuple.949 = (s32[]{:T(128)}, bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)}) tuple(add.4517, fusion.2382)
  } // wide.region_54.2652.clone_spmd

  wide.region_55.2732.clone_spmd {
    wide.param.28 = (s32[]{:T(128)}, bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)}) parameter(0)
    get-tuple-element.12167 = s32[]{:T(128)} get-tuple-element(wide.param.28), index=0
    constant.12162 = s32[]{:T(128)} constant(2)
    ROOT compare.1815 = pred[]{:T(512)} compare(get-tuple-element.12167, constant.12162), direction=LT
  }
  ENTRY main {
    p0 = bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)} parameter(0)
    init = s32[]{:T(128)} constant(0)
    first.input = tuple(init, p0)
    first.out = while(first.input), condition=first.cond , body=first.body
    o1 = bf16[2,1,8,513,128]{4,3,2,1,0:T(8,128)(2,1)} get-tuple-element(first.out), index=1
    input = tuple(init, o1)
    out = while(input), condition=wide.region_55.2732.clone_spmd , body=wide.region_54.2652.clone_spmd
    ROOT res = s32[]{:T(128)} get-tuple-element(out), index=0
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
}

TEST_F(UnstackerTest, UnstackDUSFusionWithAddPattern) {
  std::string hlo_string = R"(
  HloModule SimpleLoop

  add.2771.reduce_sub_computation {
    lhs.44 = bf16[] parameter(0)
    rhs.44 = bf16[] parameter(1)
    ROOT add.3079 = bf16[] add(lhs.44, rhs.44)
  }

  fused_computation.75.clone {
    param_0.31658 = bf16[2,4096]{1,0:T(8,128)(2,1)} parameter(0)
    param_1.26202 = s32[]{:T(128)} parameter(1)
    constant.47557 = s32[]{:T(128)} constant(0)
    dynamic-slice.12289 = bf16[1,4096]{1,0:T(2,128)(2,1)} dynamic-slice(param_0.31658, param_1.26202, constant.47557), dynamic_slice_sizes={1,4096}
    constant.47559 = bf16[]{:T(256)} constant(1)
    broadcast.39214 = bf16[1,4096]{1,0:T(2,128)(2,1)} broadcast(constant.47559), dimensions={}
    add.13176 = bf16[1,4096]{1,0:T(2,128)(2,1)} add(dynamic-slice.12289, broadcast.39214)
    constant.47558 = bf16[] constant(-0)
    ROOT reduce.8210 = bf16[4096]{0:T(1024)(128)(2,1)} reduce(add.13176, constant.47558), dimensions={0}, to_apply=add.2771.reduce_sub_computation
  } // fused_computation.75.clone

  first.body {
    wide.param.29 = (s32[]{:T(128)}, bf16[2,4096]{1,0:T(8,128)(2,1)}) parameter(0)
    get-tuple-element.12177 = s32[]{:T(128)} get-tuple-element(wide.param.29), index=0
    constant.12144..sunk.2 = s32[]{:T(128)} constant(1)
    add.4517 = s32[]{:T(128)} add(get-tuple-element.12177, constant.12144..sunk.2)
    get-tuple-element.12178 = bf16[2,4096]{1,0:T(8,128)(2,1)} get-tuple-element(wide.param.29), index=1
    fusion.2381 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(get-tuple-element.12178, get-tuple-element.12177), kind=kLoop, calls=fused_computation.75.clone
    tmp = bf16[4096]{0:T(1024)(128)(2,1)} add(fusion.2381, fusion.2381)
    ROOT tuple.949 = (s32[]{:T(128)}, bf16[2,4096]{1,0:T(8,128)(2,1)}) tuple(add.4517, get-tuple-element.12178)
  } // wide.region_54.2652.clone_spmd

  first.cond {
    wide.param.28 = (s32[]{:T(128)}, bf16[2,4096]{1,0:T(8,128)(2,1)}) parameter(0)
    get-tuple-element.12167 = s32[]{:T(128)} get-tuple-element(wide.param.28), index=0
    constant.12162 = s32[]{:T(128)} constant(2)
    ROOT compare.1815 = pred[]{:T(512)} compare(get-tuple-element.12167, constant.12162), direction=LT
  }

  ENTRY main {
    p0 = bf16[2,4096]{1,0:T(8,128)(2,1)} parameter(0)
    init = s32[]{:T(128)} constant(0)
    first.input = tuple(init, p0)
    first.out = while(first.input), condition=first.cond , body=first.body
    ROOT o1 = s32[]{:T(128)} get-tuple-element(first.out), index=0
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto original = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_TRUE(unstacked);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

}  // namespace
}  // namespace xla
