/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

namespace {

// TODO(b/210165681): The tests in this file are fragile to HLO op names.

class TreeReductionRewriterTest : public GpuCodegenTest {};

TEST_F(TreeReductionRewriterTest, RowReductionSingleDimensionNoBatched) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[50000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[50000]) -> f32[224] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[50000]{0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[] constant(0)
// CHECK:   [[INSTR_2:%[^ ]+]] = f32[50176]{0} pad(f32[50000]{0} [[INSTR_0]], f32[] [[INSTR_1]]), padding=0_176
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[224,224]{1,0} bitcast(f32[50176]{0} [[INSTR_2]])
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[224]{0} reduce(f32[224,224]{1,0} [[INSTR_3]], f32[] [[INSTR_1]]), dimensions={1}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[50000]) -> f32[] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[50000]{0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[224]{0} fusion(f32[50000]{0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[] reduce(f32[224]{0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={0}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, RowReductionWeirdOutputLayout) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[2,4,17000]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[2,4]{0,1} reduce(input, zero), dimensions={2}, to_apply=add
}
)";

  // Check that we preserve the layout.
  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: f32[2,4]{0,1} reduce(
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest,
       RowReductionSingleDimensionNoBatchedDivisible) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[49952] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[49952]) -> f32[223] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[49952]{0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[223,224]{1,0} bitcast(f32[49952]{0} [[INSTR_0]])
// CHECK:   [[INSTR_2:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_3:%[^ ]+]] = f32[223]{0} reduce(f32[223,224]{1,0} [[INSTR_1]], f32[] [[INSTR_2]]), dimensions={1}, to_apply=[[INSTR_4:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[49952]) -> f32[] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[49952]{0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[223]{0} fusion(f32[49952]{0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[] reduce(f32[223]{0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={0}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, RowReductionNoBatched) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[100,10,90000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100,10] reduce(input, zero), dimensions={2}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[100,10,90000]) -> f32[100,10,300] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[100,10,90000]{2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[100,10,300,300]{3,2,1,0} bitcast(f32[100,10,90000]{2,1,0} [[INSTR_0]])
// CHECK:   [[INSTR_2:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_3:%[^ ]+]] = f32[100,10,300]{2,1,0} reduce(f32[100,10,300,300]{3,2,1,0} [[INSTR_1]], f32[] [[INSTR_2]]), dimensions={3}, to_apply=[[INSTR_4:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[100,10,90000]) -> f32[100,10] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[100,10,90000]{2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[100,10,300]{2,1,0} fusion(f32[100,10,90000]{2,1,0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[100,10]{1,0} reduce(f32[100,10,300]{2,1,0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={2}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
      )");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest,
       RowReductionSingleDimensionNoBatchedLargeInput) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1000000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[1000000]) -> f32[1000] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[1000000]{0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[1000,1000]{1,0} bitcast(f32[1000000]{0} [[INSTR_0]])
// CHECK:   [[INSTR_2:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_3:%[^ ]+]] = f32[1000]{0} reduce(f32[1000,1000]{1,0} [[INSTR_1]], f32[] [[INSTR_2]]), dimensions={1}, to_apply=[[INSTR_4:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[1000000]) -> f32[] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[1000000]{0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[1000]{0} fusion(f32[1000000]{0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[] reduce(f32[1000]{0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={0}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, RowReductionBatchedDimensionFits) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[8,100,90000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0,2}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[8,100,90000]) -> f32[100,300] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[8,100,90000]{2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[8,100,300,300]{3,2,1,0} bitcast(f32[8,100,90000]{2,1,0} [[INSTR_0]])
// CHECK:   [[INSTR_2:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_3:%[^ ]+]] = f32[100,300]{1,0} reduce(f32[8,100,300,300]{3,2,1,0} [[INSTR_1]], f32[] [[INSTR_2]]), dimensions={3,0}, to_apply=[[INSTR_4:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[8,100,90000]) -> f32[100] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[8,100,90000]{2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[100,300]{1,0} fusion(f32[8,100,90000]{2,1,0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[100]{0} reduce(f32[100,300]{1,0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={1}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
      )");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, RowReductionBatchedDimensionDoesNotFit) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[32,100,90000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0,2}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[32,100,90000]) -> f32[32,100,300] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[32,100,90000]{2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[32,100,300,300]{3,2,1,0} bitcast(f32[32,100,90000]{2,1,0} [[INSTR_0]])
// CHECK:   [[INSTR_2:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_3:%[^ ]+]] = f32[32,100,300]{2,1,0} reduce(f32[32,100,300,300]{3,2,1,0} [[INSTR_1]], f32[] [[INSTR_2]]), dimensions={3}, to_apply=[[INSTR_4:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[32,100,90000]) -> f32[100] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[32,100,90000]{2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[32,100,300]{2,1,0} fusion(f32[32,100,90000]{2,1,0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   [[INSTR_4:%[^ ]+]] = f32[32,100]{1,0} reduce(f32[32,100,300]{2,1,0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={2}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK:   ROOT [[INSTR_6:%[^ ]+]] = f32[100]{0} reduce(f32[32,100]{1,0} [[INSTR_4]], f32[] [[INSTR_3]]), dimensions={0}, to_apply=[[INSTR_5]]
// CHECK: }
      )");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, ColumnReductionSimple) {
  // TODO(cheshire): reduce duplication for HLO text, factor out the common
  // part.
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[10000,100] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[10000,100]) -> f32[100,100] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[10000,100]{1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[100,100,100]{2,1,0} bitcast(f32[10000,100]{1,0}
// %param_0.2)
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_1:%[^ ]+]] = f32[100,100]{1,0} reduce(f32[100,100,100]{2,1,0} [[INSTR_2:%[^ ]+]], f32[] [[INSTR_0]]), dimensions={0}, to_apply=[[INSTR_3:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[10000,100]) -> f32[100] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[10000,100]{1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[100,100]{1,0} fusion(f32[10000,100]{1,0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[100]{0} reduce(f32[100,100]{1,0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={0}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, ColumnReductionSimpleNoSquareDivisible) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[10302,100] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[10302,100]) -> f32[102,100] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[10302,100]{1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[101,102,100]{2,1,0} bitcast(f32[10302,100]{1,0}
// %param_0.2)
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_1:%[^ ]+]] = f32[102,100]{1,0} reduce(f32[101,102,100]{2,1,0} [[INSTR_2:%[^ ]+]], f32[] [[INSTR_0]]), dimensions={0}, to_apply=[[INSTR_3:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[10302,100]) -> f32[100] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[10302,100]{1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[102,100]{1,0} fusion(f32[10302,100]{1,0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[100]{0} reduce(f32[102,100]{1,0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={0}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, ColumnReductionOtherIndex) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[10000,2,2,2] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[2,2,2] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[10000,2,2,2]) -> f32[100,2,2,2] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[10000,2,2,2]{3,2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[100,100,2,2,2]{4,3,2,1,0} bitcast(f32[10000,2,2,2]{3,2,1,0} [[INSTR_0]])
// CHECK:   [[INSTR_2:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_3:%[^ ]+]] = f32[100,2,2,2]{3,2,1,0} reduce(f32[100,100,2,2,2]{4,3,2,1,0} [[INSTR_1]], f32[] [[INSTR_2]]), dimensions={0}, to_apply=[[INSTR_4:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[10000,2,2,2]) -> f32[2,2,2] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[10000,2,2,2]{3,2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[100,2,2,2]{3,2,1,0} fusion(f32[10000,2,2,2]{3,2,1,0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[2,2,2]{2,1,0} reduce(f32[100,2,2,2]{3,2,1,0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={0}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, ColumnReductionVeryLargeInput) {
  // TODO(cheshire): reduce duplication for HLO text, factor out the common
  // part.
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1000000,5] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[5] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.2: f32[1000000,5]) -> f32[1000,5] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[1000000,5]{1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[1000,1000,5]{2,1,0} bitcast(f32[1000000,5]{1,0}
// %param_0.2)
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_1:%[^ ]+]] = f32[1000,5]{1,0} reduce(f32[1000,1000,5]{2,1,0} [[INSTR_2:%[^ ]+]], f32[] [[INSTR_0]]), dimensions={0}, to_apply=[[INSTR_3:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[1000000,5]) -> f32[5] {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[1000000,5]{1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[1000,5]{1,0} fusion(f32[1000000,5]{1,0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = f32[5]{0} reduce(f32[1000,5]{1,0} [[INSTR_1]], f32[] [[INSTR_3]]), dimensions={0}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, VariadicReductionLargeRow) {
  const char* hlo_text = R"(
HloModule Reduce_R1x2_to_R0x2_argmax

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  input = f32[2,100000] parameter(0)
  idxs = u32[2,100000] iota(), iota_dimension=0
  zero = f32[] constant(0)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[2], u32[2]) reduce(
    input, idxs, zero, zero_idx),
    dimensions={1},
    to_apply=%argmax
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.6: f32[], param_1.7: f32[], param_2.8: u32[], param_3.5: u32[]) -> (f32[], u32[]) {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[] parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[] parameter(1)
// CHECK:   [[INSTR_2:%[^ ]+]] = pred[] compare(f32[] [[INSTR_0]], f32[] [[INSTR_1]]), direction=GT
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] select(pred[] [[INSTR_2]], f32[] [[INSTR_0]], f32[] [[INSTR_1]])
// CHECK:   [[INSTR_4:%[^ ]+]] = u32[] parameter(2)
// CHECK:   [[INSTR_5:%[^ ]+]] = u32[] parameter(3)
// CHECK:   [[INSTR_6:%[^ ]+]].clone.1 = u32[] select(pred[] [[INSTR_2]], u32[] [[INSTR_4]], u32[] [[INSTR_5]])
// CHECK:   ROOT [[INSTR_7:%[^ ]+]] = (f32[], u32[]) tuple(f32[] [[INSTR_3]], u32[] [[INSTR_6]].clone.1)
// CHECK: }
// CHECK: (param_0.2: f32[2,100000]) -> (f32[2,317], u32[2,317]) {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[2,100000]{1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[] constant(0)
// CHECK:   [[INSTR_2:%[^ ]+]] = f32[2,100489]{1,0} pad(f32[2,100000]{1,0} [[INSTR_0]], f32[] [[INSTR_1]]), padding=0_0x0_489
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[2,317,317]{2,1,0} bitcast(f32[2,100489]{1,0} [[INSTR_2]])
// CHECK:   [[INSTR_4:%[^ ]+]] = u32[2,100000]{1,0} iota(), iota_dimension=0
// CHECK:   [[INSTR_5:%[^ ]+]] = u32[] constant(0)
// CHECK:   [[INSTR_6:%[^ ]+]] = u32[2,100489]{1,0} pad(u32[2,100000]{1,0} [[INSTR_4]], u32[] [[INSTR_5]]), padding=0_0x0_489
// CHECK:   [[INSTR_7:%[^ ]+]] = u32[2,317,317]{2,1,0} bitcast(u32[2,100489]{1,0} [[INSTR_6]])
// CHECK:   ROOT [[INSTR_8:%[^ ]+]] = (f32[2,317]{1,0}, u32[2,317]{1,0}) reduce(f32[2,317,317]{2,1,0} [[INSTR_3]], u32[2,317,317]{2,1,0} [[INSTR_7]], f32[] [[INSTR_1]], u32[] [[INSTR_5]]), dimensions={2}, to_apply=[[INSTR_9:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[2,100000]) -> (f32[2], u32[2]) {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[2,100000]{1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = (f32[2,317]{1,0}, u32[2,317]{1,0}) fusion(f32[2,100000]{1,0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[2,317]{1,0} get-tuple-element((f32[2,317]{1,0}, u32[2,317]{1,0}) [[INSTR_1]]), index=0
// CHECK:   [[INSTR_4:%[^ ]+]] = u32[2,317]{1,0} get-tuple-element((f32[2,317]{1,0}, u32[2,317]{1,0}) [[INSTR_1]]), index=1
// CHECK:   [[INSTR_5:%[^ ]+]] = f32[] constant(0)
// CHECK:   [[INSTR_6:%[^ ]+]] = u32[] constant(0)
// CHECK:   ROOT [[INSTR_7:%[^ ]+]] = (f32[2]{0}, u32[2]{0}) reduce(f32[2,317]{1,0} [[INSTR_3]], u32[2,317]{1,0} [[INSTR_4]], f32[] [[INSTR_5]], u32[] [[INSTR_6]]), dimensions={1}, to_apply=[[INSTR_8:%[^ ]+]]
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, VariadicReductionLargeBatchSize) {
  const char* hlo_text = R"(
HloModule Reduce_R1x2_to_R0x2_argmax

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  input = f32[20,2,100] parameter(0)
  idxs = u32[20,2,100] iota(), iota_dimension=0
  zero = f32[] constant(0)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[2], u32[2]) reduce(
    input, idxs, zero, zero_idx),
    dimensions={0,2},
    to_apply=%argmax
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (param_0.4: f32[], param_1.5: f32[], param_2.6: u32[], param_3.3: u32[]) -> (f32[], u32[]) {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[] parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = f32[] parameter(1)
// CHECK:   [[INSTR_2:%[^ ]+]] = pred[] compare(f32[] [[INSTR_0]], f32[] [[INSTR_1]]), direction=GT
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[] select(pred[] [[INSTR_2]], f32[] [[INSTR_0]], f32[] [[INSTR_1]])
// CHECK:   [[INSTR_4:%[^ ]+]] = u32[] parameter(2)
// CHECK:   [[INSTR_5:%[^ ]+]] = u32[] parameter(3)
// CHECK:   [[INSTR_6:%[^ ]+]].clone.1 = u32[] select(pred[] [[INSTR_2]], u32[] [[INSTR_4]], u32[] [[INSTR_5]])
// CHECK:   ROOT [[INSTR_7:%[^ ]+]] = (f32[], u32[]) tuple(f32[] [[INSTR_3]], u32[] [[INSTR_6]].clone.1)
// CHECK: }
// CHECK: (param_0: f32[20,2,100]) -> (f32[20,2], u32[20,2]) {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[20,2,100]{2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = u32[20,2,100]{2,1,0} iota(), iota_dimension=0
// CHECK:   [[INSTR_2:%[^ ]+]] = f32[] constant(0)
// CHECK:   [[INSTR_3:%[^ ]+]] = u32[] constant(0)
// CHECK:   ROOT [[INSTR_4:%[^ ]+]] = (f32[20,2]{1,0}, u32[20,2]{1,0}) reduce(f32[20,2,100]{2,1,0} [[INSTR_0]], u32[20,2,100]{2,1,0} [[INSTR_1]], f32[] [[INSTR_2]], u32[] [[INSTR_3]]), dimensions={2}, to_apply=[[INSTR_5:%[^ ]+]]
// CHECK: }
// CHECK: ENTRY %main (input: f32[20,2,100]) -> (f32[2], u32[2]) {
// CHECK:   [[INSTR_0:%[^ ]+]] = f32[20,2,100]{2,1,0} parameter(0)
// CHECK:   [[INSTR_1:%[^ ]+]] = (f32[20,2]{1,0}, u32[20,2]{1,0}) fusion(f32[20,2,100]{2,1,0} [[INSTR_0]]), kind=kInput, calls=[[INSTR_2:%[^ ]+]]
// CHECK:   [[INSTR_3:%[^ ]+]] = f32[20,2]{1,0} get-tuple-element((f32[20,2]{1,0}, u32[20,2]{1,0}) [[INSTR_1]]), index=0
// CHECK:   [[INSTR_4:%[^ ]+]] = u32[20,2]{1,0} get-tuple-element((f32[20,2]{1,0}, u32[20,2]{1,0}) [[INSTR_1]]), index=1
// CHECK:   [[INSTR_5:%[^ ]+]] = f32[] constant(0)
// CHECK:   [[INSTR_6:%[^ ]+]] = u32[] constant(0)
// CHECK:   ROOT [[INSTR_7:%[^ ]+]] = (f32[2]{0}, u32[2]{0}) reduce(f32[20,2]{1,0} [[INSTR_3]], u32[20,2]{1,0} [[INSTR_4]], f32[] [[INSTR_5]], u32[] [[INSTR_6]]), dimensions={0}, to_apply=[[INSTR_8:%[^ ]+]]
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
