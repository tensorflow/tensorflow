/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "tensorflow/compiler/xla/service/cpu/tests/cpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/llvm_irgen_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace cpu {

namespace {

class TreeReductionRewriterTest : public CpuCodegenTest {};

TEST_F(TreeReductionRewriterTest, SimpleRewrite) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[1000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}
  )";

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %main (input: f32[1000]) -> f32[] {
; CHECK-NEXT:    [[INSTR_0:%[^ ]+]] = f32[1000]{0} parameter(0)
; CHECK-NEXT:    [[INSTR_1:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[INSTR_2:%[^ ]+]] = f32[32]{0} reduce-window([[INSTR_0]], [[INSTR_1]]), window={size=32 stride=32 pad=12_12}, to_apply=[[INSTR_3:%[^ ]+]]
; CHECK-NEXT:    ROOT [[INSTR_4:%[^ ]+]] = f32[] reduce([[INSTR_2]], [[INSTR_1]]), dimensions={0}, to_apply=[[INSTR_3]]
      )");
}

TEST_F(TreeReductionRewriterTest, RewriteMultipleDimensions) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[1000,1000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0,1}, to_apply=add
}
  )";

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK:    [[INSTR_0:%[^ ]+]] = f32[32,32]{1,0} reduce-window([[INSTR_1:%[^ ]+]], [[INSTR_2:%[^ ]+]]), window={size=32x32 stride=32x32 pad=12_12x12_12}, to_apply=[[INSTR_3:%[^ ]+]]
; CHECK-NEXT: ROOT [[INSTR_4:%[^ ]+]] = f32[] reduce([[INSTR_0]], [[INSTR_2]]), dimensions={0,1}, to_apply=[[INSTR_3]]
      )");
}

TEST_F(TreeReductionRewriterTest, RewriteMultipleDimensionsSingleSmaller) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[1000,31] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0,1}, to_apply=add
}
  )";

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK:    [[INSTR_0:%[^ ]+]] = f32[32,1]{1,0} reduce-window([[INSTR_1:%[^ ]+]], [[INSTR_2:%[^ ]+]]), window={size=32x31 stride=32x31 pad=12_12x0_0}, to_apply=[[INSTR_3:%[^ ]+]]
; CHECK-NEXT: ROOT [[INSTR_4:%[^ ]+]] = f32[] reduce([[INSTR_0]], [[INSTR_2]]), dimensions={0,1}, to_apply=[[INSTR_3]]
      )");
}

TEST_F(TreeReductionRewriterTest, NoRewriteRequired) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[31,31] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0,1}, to_apply=add
}
  )";

  MatchOptimizedHlo(hlo_text,
                    R"(
// CHECK: ROOT [[INSTR_0:%[^ ]+]] = f32[] reduce([[INSTR_1:%[^ ]+]], [[INSTR_2:%[^ ]+]]), dimensions={0,1}, to_apply=[[INSTR_3:%[^ ]+]]
      )");
}

TEST_F(TreeReductionRewriterTest, NoRewriteRequiredZeroDim) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[3000,0] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0,1}, to_apply=add
}
  )";

  MatchOptimizedHlo(hlo_text,
                    R"(
// CHECK: ROOT {{.*}} = f32[] copy
      )");
}

}  // namespace
}  // namespace cpu
}  // namespace xla
