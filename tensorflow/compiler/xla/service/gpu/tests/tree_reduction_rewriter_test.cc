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

class TreeReductionRewriterTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_deterministic_reductions(true);
    return debug_options;
  }

 protected:
  void EnsureDeterminism(absl::string_view hlo_text) {
    std::vector<ExecutionProfile> profiles;
    profiles.emplace_back();
    profiles.emplace_back();
    EXPECT_TRUE(RunMultipleTimes(hlo_text,
                                 /*run_hlo_passes=*/true,
                                 /*profiles=*/&profiles,
                                 /*backend_config=*/"",
                                 /*assert_determinism=*/true));
  }
};

TEST_F(TreeReductionRewriterTest, RowReductionSingleDimensionNoBatched) {
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[10000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}

)";

  // TODO(cheshire): a more generic check, do not hardcode the names.
  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: %param_0.2 = f32[10000]{0} parameter(0)
// CHECK-NEXT: %zero_1 = f32[] constant(0)
// CHECK-NEXT: %pad.1 = f32[10240]{0} pad(f32[10000]{0} %param_0.2, f32[] %zero_1), padding=0_240
// CHECK-NEXT: %bitcast.1 = f32[20,512]{1,0} bitcast(f32[10240]{0} %pad.1)
// CHECK-NEXT: %reduce.3 = f32[512]{0} reduce(f32[20,512]{1,0} %bitcast.1, f32[] %zero_1), dimensions={0}, to_apply=%add
// CHECK-NEXT: ROOT %reduce.2 = f32[] reduce(f32[512]{0} %reduce.3, f32[] %zero_1), dimensions={0}, to_apply=%add
      )");

  EnsureDeterminism(hlo_text);
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
  input = f32[100,100,10000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100,100] reduce(input, zero), dimensions={2}, to_apply=add
}

)";

  EnsureDeterminism(hlo_text);

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: %fused_computation (param_0.2: f32[100,100,10000]) -> f32[100,100,256] {
// CHECK:  %param_0.2 = f32[100,100,10000]{2,1,0} parameter(0)
// CHECK:  %zero_1 = f32[] constant(0)
// CHECK:  %pad.1 = f32[100,100,10240]{2,1,0} pad(f32[100,100,10000]{2,1,0} %param_0.2, f32[] %zero_1), padding=0_0x0_0x0_240
// CHECK:  %bitcast.1 = f32[100,100,40,256]{3,2,1,0} bitcast(f32[100,100,10240]{2,1,0} %pad.1)
// CHECK:  ROOT %reduce.2 = f32[100,100,256]{2,1,0} reduce(f32[100,100,40,256]{3,2,1,0} %bitcast.1, f32[] %zero_1), dimensions={2}, to_apply=%add

// CHECK:  %fusion = f32[100,100,256]{2,1,0} fusion(f32[100,100,10000]{2,1,0} %input), kind=kInput, calls=%fused_computation
// CHECK:  %zero = f32[] constant(0)
// CHECK:  ROOT %reduce.1 = f32[100,100]{1,0} reduce(f32[100,100,256]{2,1,0} %fusion, f32[] %zero), dimensions={2}, to_apply=%add
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
// CHECK: %fused_computation (param_0.2: f32[1000000]) -> f32[512] {
// CHECK:  %param_0.2 = f32[1000000]{0} parameter(0)
// CHECK:  %zero_1 = f32[] constant(0)
// CHECK:  %pad.3 = f32[1000448]{0} pad(f32[1000000]{0} %param_0.2, f32[] %zero_1), padding=0_448
// CHECK:  %bitcast.3 = f32[1954,512]{1,0} bitcast(f32[1000448]{0} %pad.3)
// CHECK:  %pad.2 = f32[2048,512]{1,0} pad(f32[1954,512]{1,0} %bitcast.3, f32[] %zero_1), padding=0_94x0_0
// CHECK:  %bitcast.2 = f32[16,128,512]{2,1,0} bitcast(f32[2048,512]{1,0} %pad.2)
// CHECK:  %reduce.5 = f32[128,512]{1,0} reduce(f32[16,128,512]{2,1,0} %bitcast.2, f32[] %zero_1), dimensions={0}, to_apply=%add
// CHECK:  ROOT %reduce.4 = f32[512]{0} reduce(f32[128,512]{1,0} %reduce.5, f32[] %zero_1), dimensions={0}, to_apply=%add
// CHECK: }
// CHECK: ENTRY %main (input: f32[1000000]) -> f32[] {
// CHECK:  %input = f32[1000000]{0} parameter(0)
// CHECK:  %fusion = f32[512]{0} fusion(f32[1000000]{0} %input), kind=kInput, calls=%fused_computation
// CHECK:  %zero = f32[] constant(0)
// CHECK:  ROOT %reduce.1 = f32[] reduce(f32[512]{0} %fusion, f32[] %zero), dimensions={0}, to_apply=%add
// CHECK: }
      )");

  EnsureDeterminism(hlo_text);
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
  input = f32[8,100,10000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0,2}, to_apply=add
}

)";

  EnsureDeterminism(hlo_text);

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: %fused_computation (param_0.2: f32[8,100,10000]) -> f32[100] {
// CHECK:  %param_0.2 = f32[8,100,10000]{2,1,0} parameter(0)
// CHECK:  %zero_1 = f32[] constant(0)
// CHECK:  %pad.1 = f32[8,100,10240]{2,1,0} pad(f32[8,100,10000]{2,1,0} %param_0.2, f32[] %zero_1), padding=0_0x0_0x0_240
// CHECK:  %bitcast.1 = f32[8,100,40,256]{3,2,1,0} bitcast(f32[8,100,10240]{2,1,0} %pad.1)
// CHECK:  %reduce.3 = f32[100,256]{1,0} reduce(f32[8,100,40,256]{3,2,1,0} %bitcast.1, f32[] %zero_1), dimensions={2,0}, to_apply=%add
// CHECK:  ROOT %reduce.2 = f32[100]{0} reduce(f32[100,256]{1,0} %reduce.3, f32[] %zero_1), dimensions={1}, to_apply=%add
// CHECK: }

// CHECK: ENTRY %main (input: f32[8,100,10000]) -> f32[100] {
// CHECK:  %input = f32[8,100,10000]{2,1,0} parameter(0)
// CHECK:  ROOT %fusion = f32[100]{0} fusion(f32[8,100,10000]{2,1,0} %input), kind=kInput, calls=%fused_computation
// CHECK: }
      )");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TreeReductionRewriterTest, RowReductionBatchedDimensionDoesNotFit) {
  // Note: this could be too slow without shared memory optimization.
  const char* hlo_text = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[32,100,10000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0,2}, to_apply=add
}

)";

  EnsureDeterminism(hlo_text);

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: %fused_computation (param_0.2: f32[32,100,10000]) -> f32[32,100,256] {
// CHECK:  %param_0.2 = f32[32,100,10000]{2,1,0} parameter(0)
// CHECK:  %zero_1 = f32[] constant(0)
// CHECK:  %pad.1 = f32[32,100,10240]{2,1,0} pad(f32[32,100,10000]{2,1,0} %param_0.2, f32[] %zero_1), padding=0_0x0_0x0_240
// CHECK:  %bitcast.1 = f32[32,100,40,256]{3,2,1,0} bitcast(f32[32,100,10240]{2,1,0} %pad.1)
// CHECK:  ROOT %reduce.4 = f32[32,100,256]{2,1,0} reduce(f32[32,100,40,256]{3,2,1,0} %bitcast.1, f32[] %zero_1), dimensions={2}, to_apply=%add
// CHECK: }
// CHECK: ENTRY %main (input: f32[32,100,10000]) -> f32[100] {
// CHECK:  %input = f32[32,100,10000]{2,1,0} parameter(0)
// CHECK:  %fusion = f32[32,100,256]{2,1,0} fusion(f32[32,100,10000]{2,1,0} %input), kind=kInput, calls=%fused_computation
// CHECK:  %zero = f32[] constant(0)
// CHECK:  %reduce.3 = f32[32,100]{1,0} reduce(f32[32,100,256]{2,1,0} %fusion, f32[] %zero), dimensions={2}, to_apply=%add
// CHECK:  ROOT %reduce.1 = f32[100]{0} reduce(f32[32,100]{1,0} %reduce.3, f32[] %zero), dimensions={0}, to_apply=%add
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
// CHECK: %fused_computation (param_0.2: f32[10000,100]) -> f32[128,100] {
// CHECK:  %param_0.2 = f32[10000,100]{1,0} parameter(0)
// CHECK:  %zero_1 = f32[] constant(0)
// CHECK:  %pad.1 = f32[10112,100]{1,0} pad(f32[10000,100]{1,0} %param_0.2, f32[] %zero_1), padding=0_112x0_0
// CHECK:  %bitcast.1 = f32[79,128,100]{2,1,0} bitcast(f32[10112,100]{1,0} %pad.1)
// CHECK:  ROOT %reduce.2 = f32[128,100]{1,0} reduce(f32[79,128,100]{2,1,0} %bitcast.1, f32[] %zero_1), dimensions={0}, to_apply=%add
// CHECK: }

// CHECK: ENTRY %main (input: f32[10000,100]) -> f32[100] {
// CHECK:  %input = f32[10000,100]{1,0} parameter(0)
// CHECK:  %fusion = f32[128,100]{1,0} fusion(f32[10000,100]{1,0} %input), kind=kInput, calls=%fused_computation
// CHECK:  %zero = f32[] constant(0)
// CHECK:  ROOT %reduce.1 = f32[100]{0} reduce(f32[128,100]{1,0} %fusion, f32[] %zero), dimensions={0}, to_apply=%add
// CHECK: }
      )");

  EnsureDeterminism(hlo_text);
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
// CHECK: %fused_computation (param_0.2: f32[10000,2,2,2]) -> f32[128,2,2,2] {
// CHECK:  %param_0.2 = f32[10000,2,2,2]{3,2,1,0} parameter(0)
// CHECK:  %zero_1 = f32[] constant(0)
// CHECK:  %pad.1 = f32[10112,2,2,2]{3,2,1,0} pad(f32[10000,2,2,2]{3,2,1,0} %param_0.2, f32[] %zero_1), padding=0_112x0_0x0_0x0_0
// CHECK:  %bitcast.1 = f32[79,128,2,2,2]{4,3,2,1,0} bitcast(f32[10112,2,2,2]{3,2,1,0} %pad.1)
// CHECK:  ROOT %reduce.2 = f32[128,2,2,2]{3,2,1,0} reduce(f32[79,128,2,2,2]{4,3,2,1,0} %bitcast.1, f32[] %zero_1), dimensions={0}, to_apply=%add
// CHECK: }
// CHECK: ENTRY %main (input: f32[10000,2,2,2]) -> f32[2,2,2] {
// CHECK:  %input = f32[10000,2,2,2]{3,2,1,0} parameter(0)
// CHECK:  %fusion = f32[128,2,2,2]{3,2,1,0} fusion(f32[10000,2,2,2]{3,2,1,0} %input), kind=kInput, calls=%fused_computation
// CHECK:  %zero = f32[] constant(0)
// CHECK:  ROOT %reduce.1 = f32[2,2,2]{2,1,0} reduce(f32[128,2,2,2]{3,2,1,0} %fusion, f32[] %zero), dimensions={0}, to_apply=%add
// CHECK: }
      )");

  EnsureDeterminism(hlo_text);
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
// CHECK: %fused_computation (param_0.2: f32[1000000,5]) -> f32[128,128,5] {
// CHECK:  %param_0.2 = f32[1000000,5]{1,0} parameter(0)
// CHECK:  %zero_1 = f32[] constant(0)
// CHECK:  %pad.3 = f32[1000064,5]{1,0} pad(f32[1000000,5]{1,0} %param_0.2, f32[] %zero_1), padding=0_64x0_0
// CHECK:  %bitcast.3 = f32[7813,128,5]{2,1,0} bitcast(f32[1000064,5]{1,0} %pad.3)
// CHECK:  %pad.2 = f32[7936,128,5]{2,1,0} pad(f32[7813,128,5]{2,1,0} %bitcast.3, f32[] %zero_1), padding=0_123x0_0x0_0
// CHECK:  %bitcast.2 = f32[62,128,128,5]{3,2,1,0} bitcast(f32[7936,128,5]{2,1,0} %pad.2)
// CHECK:  ROOT %reduce.4 = f32[128,128,5]{2,1,0} reduce(f32[62,128,128,5]{3,2,1,0} %bitcast.2, f32[] %zero_1), dimensions={0}, to_apply=%add
// CHECK: }
// CHECK: ENTRY %main (input: f32[1000000,5]) -> f32[5] {
// CHECK:  %input = f32[1000000,5]{1,0} parameter(0)
// CHECK:  %fusion = f32[128,128,5]{2,1,0} fusion(f32[1000000,5]{1,0} %input), kind=kInput, calls=%fused_computation
// CHECK:  %zero = f32[] constant(0)
// CHECK:  %reduce.3 = f32[128,5]{1,0} reduce(f32[128,128,5]{2,1,0} %fusion, f32[] %zero), dimensions={0}, to_apply=%add
// CHECK:  ROOT %reduce.1 = f32[5]{0} reduce(f32[128,5]{1,0} %reduce.3, f32[] %zero), dimensions={0}, to_apply=%add
      )");

  EnsureDeterminism(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
