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

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using UnstackerTest = HloTestBase;

TEST_F(UnstackerTest, UnstackLoopSingleNestedFusionUser) {
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
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackLoopSingleNestedFusionUserMultipleIndex) {
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
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, UnstackLoopSingleNestedFusionUserDiffereOperandsOrder) {
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
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

TEST_F(UnstackerTest, NotUnstackLoopMultipleNestedFusionUsers) {
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
  // Currently, we don't unroll if there are multiple nested ds fusions.
  TF_ASSERT_OK_AND_ASSIGN(bool unstacked, HloUnstacker().Run(module.get()));
  EXPECT_FALSE(unstacked);
}

TEST_F(UnstackerTest, UnstackMultipleLoops) {
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
      inner.in = (s32[], bf16[8,128], s8[4,128,128]) tuple(zero, param0, param1)
      inner.out = (s32[], bf16[8,128], s8[4,128,128]) while(inner.in), condition=%while.cond.inner1, body=%while.body.inner1
      fusion.conv.inner = bf16[8,128] get-tuple-element(inner.out), index=1
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
      inner.in = (s32[], bf16[8,128], s8[4,128,128]) tuple(zero, param0, param1)
      inner.out = (s32[], bf16[8,128], s8[4,128,128]) while(inner.in), condition=%while.cond.inner2, body=%while.body.inner2
      fusion.conv.inner = bf16[8,128] get-tuple-element(inner.out), index=1
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
      second.while.input = (s32[], bf16[8,128], s8[4,128,128]) tuple(init, p1, weight)
      second.while.output = (s32[], bf16[8,128], s8[4,128,128]) while(second.while.input), condition=%while.cond2 , body=%while.body2
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

TEST_F(UnstackerTest, UnstackNestedLoopSingleNestedFusionUser) {
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
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(original),
                                      std::nullopt, false));
}

}  // namespace
}  // namespace xla
