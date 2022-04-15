/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.h"

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

// TODO(b/210165681): The tests in this file are fragile to HLO op names.

namespace xla {
namespace gpu {
namespace {

class FusionBitcastLiftTest : public HloTestBase {};

// Tests that we lift bitcast outside the fusion.
//
// This test MultiOutputFusion, multiple consecutive lift, bitcast
// with multiple users and bitcast that are used many time by the same
// user. This is a real kernel from Efficient Net, but with smaller
// shape to speed up tests.
//
// Input graph:
// Fusion 4d input, 2 1d output
//
// After optimization, the graph is:
// Bitcast 4d -> 2d
//   |
// Fusion 2d input, 2x1d outputs.
TEST_F(FusionBitcastLiftTest, NoBroadcastTest) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation.4d (param_0: f16[2,14,14,672]) -> (f32[672], f32[672]) {
  %param_0 = f16[2,14,14,672] parameter(0)
  %convert = f32[2,14,14,672] convert(%param_0)
  %bitcast.1 = f32[392,672] bitcast(%convert)
  %constant_0 = f32[] constant(0)
  %reduce.1 = f32[672]{0} reduce(%bitcast.1, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  %multiply = f32[2,14,14,672] multiply(%convert, %convert)
  %bitcast.2 = f32[392,672] bitcast(%multiply)
  %reduce.2 = f32[672]{0} reduce(%bitcast.2, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple = (f32[672]{0}, f32[672]{0}) tuple(%reduce.1, %reduce.2)
}

ENTRY %main {
  %param_0 = f16[2,14,14,672] parameter(0)
  ROOT %fusion.4d = (f32[672]{0}, f32[672]{0}) fusion(%param_0), kind=kInput, calls=%fused_computation.4d
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());
  // Remove the old fusion not used anymore.
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  StatusOr<bool> filecheck_result = RunFileCheck(module->ToString(),
                                                 R"(
; CHECK-LABEL: %fused_computation
; CHECK:         f16[392,672]{1,0} parameter(0)
; CHECK-NOT:     parameter
; CHECK-NOT:     bitcast
; CHECK-LABEL: ENTRY %main
; CHECK-NEXT:    f16[2,14,14,672]{3,2,1,0} parameter(0)
; CHECK-NEXT:    bitcast(
; CHECK-NEXT:    fusion(
      )");
  EXPECT_TRUE(filecheck_result.status().ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

// Tests that we lift bitcast outside the fusion when scalar broadcasting are
// present.
//
// Input graph:
// Fusion 1x4d and 1x0d inputs, 2 1d output
//
// After optimization, the graph is:
// Bitcast 4d -> 2d
//   |
// Fusion 1x2d and 1x0d inputs, 2 1d output
//   Inside the fusion, there is a bitcast left after the broadcast.
TEST_F(FusionBitcastLiftTest, ScalarBroadcastTest) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation.4d (param_0: f16[2,14,14,672], param_1: f32[]) -> (f32[672], f32[672]) {
  %param_0 = f16[2,14,14,672] parameter(0)
  %convert = f32[2,14,14,672] convert(%param_0)
  %bitcast.1 = f32[392,672] bitcast(%convert)
  %constant_0 = f32[] constant(0)
  %reduce.1 = f32[672]{0} reduce(%bitcast.1, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  %param_1 = f32[] parameter(1)
  %broadcast = f32[2,14,14,672] broadcast(%param_1), dimensions={}
  %multiply = f32[2,14,14,672] multiply(%convert, %broadcast)
  %bitcast.2 = f32[392,672] bitcast(%multiply)
  %reduce.2 = f32[672]{0} reduce(%bitcast.2, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple = (f32[672]{0}, f32[672]{0}) tuple(%reduce.1, %reduce.2)
}

ENTRY %main {
  %param_0 = f16[2,14,14,672] parameter(0)
  %param_1 = f32[] parameter(1)
  ROOT %fusion.4d = (f32[672]{0}, f32[672]{0}) fusion(%param_0, %param_1), kind=kInput, calls=%fused_computation.4d
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());
  // Remove the old fusion not used anymore.
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  StatusOr<bool> filecheck_result = RunFileCheck(module->ToString(),
                                                 R"(
; CHECK-LABEL: %fused_computation
; CHECK:         f16[392,672]{1,0} parameter(0)
; CHECK:         f32[] parameter(1)
; CHECK-NOT:     parameter
; CHECK-NOT:     bitcast
; CHECK:         %broadcast.1 =
; CHECK-NEXT:    bitcast(f32[2,14,14,672]{3,2,1,0} %broadcast.1)
; CHECK-NOT:     bitcast(
; CHECK-LABEL: ENTRY %main
; CHECK-NEXT:    f16[2,14,14,672]{3,2,1,0} parameter(0)
; CHECK-NEXT:    bitcast(
; CHECK-NEXT:    %param_1.1 = f32[] parameter(1)
; CHECK-NEXT:    fusion(
      )");
  EXPECT_TRUE(filecheck_result.status().ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

TEST_F(FusionBitcastLiftTest, RowBroadcastTest) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation.4d (param_0: f16[2,14,14,672], param_1: f32[672]) -> (f32[672], f32[672]) {
  %param_0 = f16[2,14,14,672] parameter(0)
  %convert = f32[2,14,14,672] convert(%param_0)
  %bitcast.1 = f32[392,672] bitcast(%convert)
  %constant_0 = f32[] constant(0)
  %reduce.1 = f32[672]{0} reduce(%bitcast.1, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  %param_1 = f32[672] parameter(1)
  %broadcast = f32[2,14,14,672] broadcast(%param_1), dimensions={3}
  %multiply = f32[2,14,14,672] multiply(%convert, %broadcast)
  %bitcast.2 = f32[392,672] bitcast(%multiply)
  %reduce.2 = f32[672]{0} reduce(%bitcast.2, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple = (f32[672]{0}, f32[672]{0}) tuple(%reduce.1, %reduce.2)
}

ENTRY %main {
  %param_0 = f16[2,14,14,672] parameter(0)
  %param_1 = f32[672] parameter(1)
  ROOT %fusion.4d = (f32[672]{0}, f32[672]{0}) fusion(%param_0, %param_1), kind=kInput, calls=%fused_computation.4d
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());
  // Remove the old fusion not used anymore.
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  StatusOr<bool> filecheck_result = RunFileCheck(module->ToString(),
                                                 R"(
; CHECK-LABEL: %fused_computation
; CHECK:         f16[392,672]{1,0} parameter(0)
; CHECK:         f32[672]{0} parameter(1)
; CHECK-NOT:     parameter
; CHECK-NOT:     bitcast
; CHECK:         %broadcast.1
; CHECK:         bitcast(f32[2,14,14,672]{3,2,1,0} %broadcast.1)
; CHECK-NOT:     bitcast(
; CHECK-LABEL: ENTRY %main
; CHECK-NEXT:    f16[2,14,14,672]{3,2,1,0} parameter(0)
; CHECK-NEXT:    bitcast(
; CHECK-NEXT:    %param_1.1 = f32[672]{0} parameter(1)
; CHECK-NEXT:    fusion(
      )");
  EXPECT_TRUE(filecheck_result.status().ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

TEST_F(FusionBitcastLiftTest, ScalarAndRowBroadcastTest) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation.4d (param_0: f16[2,14,14,672], param_1: f32[672], param_2: f32[]) -> (f32[672], f32[672]) {
  %param_0 = f16[2,14,14,672] parameter(0)
  %convert = f32[2,14,14,672] convert(%param_0)
  %bitcast.1 = f32[392,672] bitcast(%convert)
  %constant_0 = f32[] constant(0)
  %reduce.1 = f32[672]{0} reduce(%bitcast.1, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  %param_1 = f32[672] parameter(1)
  %broadcast = f32[2,14,14,672] broadcast(%param_1), dimensions={3}
  %multiply.1 = f32[2,14,14,672] multiply(%convert, %broadcast)
  %param_2 = f32[] parameter(2)
  %broadcast.1 = f32[2,14,14,672] broadcast(%param_2), dimensions={}
  %multiply.2 = f32[2,14,14,672] multiply(%broadcast.1, %multiply.1)
  %bitcast.2 = f32[392,672] bitcast(%multiply.2)
  %reduce.2 = f32[672]{0} reduce(%bitcast.2, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple = (f32[672]{0}, f32[672]{0}) tuple(%reduce.1, %reduce.2)
}

ENTRY %main {
  %param_0 = f16[2,14,14,672] parameter(0)
  %param_1 = f32[672] parameter(1)
  %param_2 = f32[] parameter(2)
  ROOT %fusion.4d = (f32[672]{0}, f32[672]{0}) fusion(%param_0, %param_1, %param_2), kind=kInput, calls=%fused_computation.4d
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());
  // Remove the old fusion not used anymore.
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  StatusOr<bool> filecheck_result = RunFileCheck(module->ToString(),
                                                 R"(
; CHECK-LABEL: %fused_computation
; CHECK-NOT:     bitcast
; CHECK:         f32[2,14,14,672]{3,2,1,0} broadcast(
; CHECK-NEXT:    f32[392,672]{1,0} bitcast(
; CHECK-NOT:     bitcast
; CHECK:         f32[2,14,14,672]{3,2,1,0} broadcast(
; CHECK-NEXT:    f32[392,672]{1,0} bitcast(
; CHECK-NOT:     bitcast(
; CHECK-LABEL: ENTRY %main
; CHECK-NOT:     bitcast(
; CHECK:    bitcast(f16[2,14,14,672]{3,2,1,0} %param_0
; CHECK-NOT:     bitcast(
      )");
  EXPECT_TRUE(filecheck_result.status().ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

// To trigger the bitcast same pattern check.
TEST_F(FusionBitcastLiftTest, StrangeBitcastBroadcastTest) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation.4d (param_0: f16[2,14,14,672], param_1: f32[672], param_2: f32[672]) -> (f32[672], f32[672]) {
  %param_0 = f16[2,14,14,672] parameter(0)
  %convert = f32[2,14,14,672] convert(%param_0)
  %bitcast.1 = f32[392,672] bitcast(%convert)
  %constant_0 = f32[] constant(0)
  %reduce.1 = f32[672]{0} reduce(%bitcast.1, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  %param_1 = f32[672] parameter(1)
  %broadcast = f32[2,14,14,672] broadcast(%param_1), dimensions={3}
  %multiply.1 = f32[2,14,14,672] multiply(%convert, %broadcast)
  %param_2 = f32[672] parameter(2)
  %broadcast.1 = f32[28,14,672] broadcast(%param_2), dimensions={2}
  %bitcast.4 = f32[28,14,672] bitcast(%multiply.1)
  %multiply.2 = f32[28,14,672] multiply(%broadcast.1, %bitcast.4)
  %bitcast.2 = f32[392,672] bitcast(%multiply.2)
  %reduce.2 = f32[672]{0} reduce(%bitcast.2, %constant_0), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple = (f32[672]{0}, f32[672]{0}) tuple(%reduce.1, %reduce.2)
}

ENTRY %main {
  %param_0 = f16[2,14,14,672] parameter(0)
  %param_1 = f32[672] parameter(1)
  %param_2 = f32[672] parameter(2)
  ROOT %fusion.4d = (f32[672]{0}, f32[672]{0}) fusion(%param_0, %param_1, %param_2), kind=kInput, calls=%fused_computation.4d
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_FALSE(FusionBitcastLift().Run(module.get()).ValueOrDie());
}

TEST_F(FusionBitcastLiftTest, ConstantBitcastTest) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation (param_0: f16[392,672], param_1: f32[1]) -> (f32[672], f32[672]) {
  %param_0 = f16[392,672] parameter(0)
  %convert = f32[392,672] convert(%param_0)

  %param_1 = f32[1] parameter(1)
  %constant_0 = f32[1] constant({1.2})
  %add = f32[1] add(%constant_0, %param_1)
  %bitcast.2 = f32[] bitcast(%add)

  %reduce.1 = f32[672]{0} reduce(%convert, %bitcast.2), dimensions={0}, to_apply=%scalar_add_computation
  %multiply = f32[392,672] multiply(%convert, %convert)
  %reduce.2 = f32[672]{0} reduce(%multiply, %bitcast.2), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple = (f32[672]{0}, f32[672]{0}) tuple(%reduce.1, %reduce.2)
}

ENTRY %main {
  %param_0 = f16[392,672] parameter(0)
  %param_1 = f32[1] parameter(1)
  ROOT %fusion = (f32[672]{0}, f32[672]{0}) fusion(%param_0, %param_1), kind=kInput, calls=%fused_computation
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());
  // Remove the old fusion not used anymore.
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  StatusOr<bool> filecheck_result = RunFileCheck(module->ToString(),
                                                 R"(
; CHECK-LABEL: %fused_computation
; CHECK:         f16[392,672]{1,0} parameter(0)
; CHECK-COUNT-1: bitcast(
; CHECK-NOT:     bitcast(
; CHECK-LABEL: ENTRY %main
; CHECK-NEXT:    f16[392,672]{1,0} parameter(0)
; CHECK-NEXT:    f32[1]{0} parameter(1)
; CHECK-NEXT:    bitcast(
; CHECK-NEXT:    fusion(
      )");
  EXPECT_TRUE(filecheck_result.status().ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

TEST_F(FusionBitcastLiftTest, Swish1Test) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}

%fused_computation (param_0.90: f32[672], param_1.127: f16[2,14,14,672], param_2.77: f16[2,14,14,672], param_3.57: f16[2,14,14,672], param_4.57: f32[672], param_5.63: f32[672], param_6.44: f32[672]) -> (f32[672], f32[672]) {
  %param_2.77 = f16[2,14,14,672]{3,2,1,0} parameter(2)
  %param_3.57 = f16[2,14,14,672]{3,2,1,0} parameter(3)
  %constant_153 = f16[] constant(1)
  %broadcast.174 = f16[2,14,14,672]{3,2,1,0} broadcast(f16[] %constant_153), dimensions={}
  %param_1.127 = f16[2,14,14,672]{3,2,1,0} parameter(1)
  %convert.46 = f32[2,14,14,672]{3,2,1,0} convert(f16[2,14,14,672]{3,2,1,0} %param_1.127)
  %param_0.90 = f32[672]{0} parameter(0)
  %constant_77_clone_1 = f32[] constant(9.96492327e-06)
  %broadcast.173 = f32[672]{0} broadcast(f32[] %constant_77_clone_1), dimensions={}
  %multiply.155 = f32[672]{0} multiply(f32[672]{0} %param_0.90, f32[672]{0} %broadcast.173)
  %broadcast.172 = f32[2,14,14,672]{3,2,1,0} broadcast(f32[672]{0} %multiply.155), dimensions={3}
  %subtract.55 = f32[2,14,14,672]{3,2,1,0} subtract(f32[2,14,14,672]{3,2,1,0} %convert.46, f32[2,14,14,672]{3,2,1,0} %broadcast.172)
  %param_6.44 = f32[672]{0} parameter(6)
  %multiply.154 = f32[672]{0} multiply(f32[672]{0} %param_6.44, f32[672]{0} %broadcast.173)
  %multiply.153 = f32[672]{0} multiply(f32[672]{0} %multiply.155, f32[672]{0} %multiply.155)
  %subtract.54 = f32[672]{0} subtract(f32[672]{0} %multiply.154, f32[672]{0} %multiply.153)
  %constant_151 = f32[] constant(0.001)
  %broadcast.171 = f32[672]{0} broadcast(f32[] %constant_151), dimensions={}
  %add.50 = f32[672]{0} add(f32[672]{0} %subtract.54, f32[672]{0} %broadcast.171)
  %rsqrt.23 = f32[672]{0} rsqrt(f32[672]{0} %add.50)
  %broadcast.170 = f32[2,14,14,672]{3,2,1,0} broadcast(f32[672]{0} %rsqrt.23), dimensions={3}
  %multiply.152 = f32[2,14,14,672]{3,2,1,0} multiply(f32[2,14,14,672]{3,2,1,0} %subtract.55, f32[2,14,14,672]{3,2,1,0} %broadcast.170)
  %param_5.63 = f32[672]{0} parameter(5)
  %broadcast.169 = f32[2,14,14,672]{3,2,1,0} broadcast(f32[672]{0} %param_5.63), dimensions={3}
  %multiply.151 = f32[2,14,14,672]{3,2,1,0} multiply(f32[2,14,14,672]{3,2,1,0} %multiply.152, f32[2,14,14,672]{3,2,1,0} %broadcast.169)
  %param_4.57 = f32[672]{0} parameter(4)
  %broadcast.168 = f32[2,14,14,672]{3,2,1,0} broadcast(f32[672]{0} %param_4.57), dimensions={3}
  %add.48 = f32[2,14,14,672]{3,2,1,0} add(f32[2,14,14,672]{3,2,1,0} %multiply.151, f32[2,14,14,672]{3,2,1,0} %broadcast.168)
  %convert.45 = f16[2,14,14,672]{3,2,1,0} convert(f32[2,14,14,672]{3,2,1,0} %add.48)
  %subtract.53 = f16[2,14,14,672]{3,2,1,0} subtract(f16[2,14,14,672]{3,2,1,0} %broadcast.174, f16[2,14,14,672]{3,2,1,0} %param_3.57)
  %multiply.150 = f16[2,14,14,672]{3,2,1,0} multiply(f16[2,14,14,672]{3,2,1,0} %convert.45, f16[2,14,14,672]{3,2,1,0} %subtract.53)
  %add.47 = f16[2,14,14,672]{3,2,1,0} add(f16[2,14,14,672]{3,2,1,0} %broadcast.174, f16[2,14,14,672]{3,2,1,0} %multiply.150)
  %multiply.149 = f16[2,14,14,672]{3,2,1,0} multiply(f16[2,14,14,672]{3,2,1,0} %param_3.57, f16[2,14,14,672]{3,2,1,0} %add.47)
  %multiply.148 = f16[2,14,14,672]{3,2,1,0} multiply(f16[2,14,14,672]{3,2,1,0} %param_2.77, f16[2,14,14,672]{3,2,1,0} %multiply.149)
  %convert.10 = f32[2,14,14,672]{3,2,1,0} convert(f16[2,14,14,672]{3,2,1,0} %multiply.148)
  %bitcast.21 = f32[392,672]{1,0} bitcast(f32[2,14,14,672]{3,2,1,0} %convert.10)
  %constant_57 = f32[] constant(0)
  %reduce.9 = f32[672]{0} reduce(f32[392,672]{1,0} %bitcast.21, f32[] %constant_57), dimensions={0}, to_apply=%scalar_add_computation
  %multiply.30.clone.1 = f32[2,14,14,672]{3,2,1,0} multiply(f32[2,14,14,672]{3,2,1,0} %convert.10, f32[2,14,14,672]{3,2,1,0} %subtract.55)
  %bitcast.20.clone.1 = f32[392,672]{1,0} bitcast(f32[2,14,14,672]{3,2,1,0} %multiply.30.clone.1)
  %reduce.8.clone.1 = f32[672]{0} reduce(f32[392,672]{1,0} %bitcast.20.clone.1, f32[] %constant_57), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple.9 = (f32[672]{0}, f32[672]{0}) tuple(f32[672]{0} %reduce.9, f32[672]{0} %reduce.8.clone.1)
}

ENTRY %main {
  %param_0 = f32[672]{0} parameter(0)
  %param_1 = f16[2,14,14,672]{3,2,1,0} parameter(1)
  %param_2 = f16[2,14,14,672]{3,2,1,0} parameter(2)
  %param_3 = f16[2,14,14,672]{3,2,1,0} parameter(3)
  %param_4 = f32[672]{0} parameter(4)
  %param_5 = f32[672]{0} parameter(5)
  %param_6 = f32[672]{0} parameter(6)

  ROOT %fusion = (f32[672]{0}, f32[672]{0}) fusion(%param_0, %param_1, %param_2, %param_3, %param_4, %param_5, %param_6), kind=kInput, calls=%fused_computation
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());
  // Remove the old fusion not used anymore.
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  StatusOr<bool> filecheck_result = RunFileCheck(module->ToString(),
                                                 R"(
; CHECK-LABEL: %fused_computation
; CHECK-COUNT-6: bitcast(
; CHECK-NOT:     bitcast(
; CHECK-LABEL: ENTRY %main
; CHECK-COUNT-3: bitcast(
; CHECK-NOT:     bitcast(
; CHECK:         fusion(
      )");
  EXPECT_TRUE(filecheck_result.status().ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

TEST_F(FusionBitcastLiftTest, Swish2Test) {
  const char* hlo_text = R"(
HloModule mod

%scalar_add_computation (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[] parameter(0)
  %scalar_rhs.1 = f32[] parameter(1)
  ROOT %add.5 = f32[] add(f32[] %scalar_lhs.1, f32[] %scalar_rhs.1)
}


%fused_computation (param_0.95: f32[672], param_1.128: f16[2,14,14,672], param_2.81: f16[2,14,14,672], param_3.66: f32[672], param_4.61: f32[672], param_5.62: f32[672]) -> (f32[672], f32[672]) {
  %param_2.81 = f16[2,14,14,672]{3,2,1,0} parameter(2)
  %constant_211 = f16[] constant(1)
  %broadcast.288 = f16[2,14,14,672]{3,2,1,0} broadcast(f16[] %constant_211), dimensions={}
  %param_1.128 = f16[2,14,14,672]{3,2,1,0} parameter(1)
  %convert.74 = f32[2,14,14,672]{3,2,1,0} convert(f16[2,14,14,672]{3,2,1,0} %param_1.128)
  %param_0.95 = f32[672]{0} parameter(0)
  %constant_77 = f32[] constant(9.96492327e-06)
  %broadcast.287 = f32[672]{0} broadcast(f32[] %constant_77), dimensions={}
  %multiply.253 = f32[672]{0} multiply(f32[672]{0} %param_0.95, f32[672]{0} %broadcast.287)
  %broadcast.286 = f32[2,14,14,672]{3,2,1,0} broadcast(f32[672]{0} %multiply.253), dimensions={3}
  %subtract.92 = f32[2,14,14,672]{3,2,1,0} subtract(f32[2,14,14,672]{3,2,1,0} %convert.74, f32[2,14,14,672]{3,2,1,0} %broadcast.286)
  %param_5.62 = f32[672]{0} parameter(5)
  %multiply.252 = f32[672]{0} multiply(f32[672]{0} %param_5.62, f32[672]{0} %broadcast.287)
  %multiply.250 = f32[672]{0} multiply(f32[672]{0} %multiply.253, f32[672]{0} %multiply.253)
  %subtract.91 = f32[672]{0} subtract(f32[672]{0} %multiply.252, f32[672]{0} %multiply.250)
  %constant_208 = f32[] constant(0.001)
  %broadcast.284 = f32[672]{0} broadcast(f32[] %constant_208), dimensions={}
  %add.93 = f32[672]{0} add(f32[672]{0} %subtract.91, f32[672]{0} %broadcast.284)
  %rsqrt.37 = f32[672]{0} rsqrt(f32[672]{0} %add.93)
  %broadcast.283 = f32[2,14,14,672]{3,2,1,0} broadcast(f32[672]{0} %rsqrt.37), dimensions={3}
  %multiply.249 = f32[2,14,14,672]{3,2,1,0} multiply(f32[2,14,14,672]{3,2,1,0} %subtract.92, f32[2,14,14,672]{3,2,1,0} %broadcast.283)
  %param_4.61 = f32[672]{0} parameter(4)
  %broadcast.282 = f32[2,14,14,672]{3,2,1,0} broadcast(f32[672]{0} %param_4.61), dimensions={3}
  %multiply.248 = f32[2,14,14,672]{3,2,1,0} multiply(f32[2,14,14,672]{3,2,1,0} %multiply.249, f32[2,14,14,672]{3,2,1,0} %broadcast.282)
  %param_3.66 = f32[672]{0} parameter(3)
  %broadcast.281 = f32[2,14,14,672]{3,2,1,0} broadcast(f32[672]{0} %param_3.66), dimensions={3}
  %add.92 = f32[2,14,14,672]{3,2,1,0} add(f32[2,14,14,672]{3,2,1,0} %multiply.248, f32[2,14,14,672]{3,2,1,0} %broadcast.281)
  %convert.73 = f16[2,14,14,672]{3,2,1,0} convert(f32[2,14,14,672]{3,2,1,0} %add.92)
  %negate.14 = f16[2,14,14,672]{3,2,1,0} negate(f16[2,14,14,672]{3,2,1,0} %convert.73)
  %exponential.12 = f16[2,14,14,672]{3,2,1,0} exponential(f16[2,14,14,672]{3,2,1,0} %negate.14)
  %add.91 = f16[2,14,14,672]{3,2,1,0} add(f16[2,14,14,672]{3,2,1,0} %broadcast.288, f16[2,14,14,672]{3,2,1,0} %exponential.12)
  %divide.22 = f16[2,14,14,672]{3,2,1,0} divide(f16[2,14,14,672]{3,2,1,0} %broadcast.288, f16[2,14,14,672]{3,2,1,0} %add.91)
  %subtract.88 = f16[2,14,14,672]{3,2,1,0} subtract(f16[2,14,14,672]{3,2,1,0} %broadcast.288, f16[2,14,14,672]{3,2,1,0} %divide.22)
  %multiply.241 = f16[2,14,14,672]{3,2,1,0} multiply(f16[2,14,14,672]{3,2,1,0} %convert.73, f16[2,14,14,672]{3,2,1,0} %subtract.88)
  %add.87 = f16[2,14,14,672]{3,2,1,0} add(f16[2,14,14,672]{3,2,1,0} %broadcast.288, f16[2,14,14,672]{3,2,1,0} %multiply.241)
  %multiply.240 = f16[2,14,14,672]{3,2,1,0} multiply(f16[2,14,14,672]{3,2,1,0} %divide.22, f16[2,14,14,672]{3,2,1,0} %add.87)
  %multiply.239 = f16[2,14,14,672]{3,2,1,0} multiply(f16[2,14,14,672]{3,2,1,0} %param_2.81, f16[2,14,14,672]{3,2,1,0} %multiply.240)
  %convert.9 = f32[2,14,14,672]{3,2,1,0} convert(f16[2,14,14,672]{3,2,1,0} %multiply.239)
  %multiply.30 = f32[2,14,14,672]{3,2,1,0} multiply(f32[2,14,14,672]{3,2,1,0} %convert.9, f32[2,14,14,672]{3,2,1,0} %subtract.92)
  %bitcast.20 = f32[392,672]{1,0} bitcast(f32[2,14,14,672]{3,2,1,0} %multiply.30)
  %constant_58 = f32[] constant(0)
  %reduce.8 = f32[672]{0} reduce(f32[392,672]{1,0} %bitcast.20, f32[] %constant_58), dimensions={0}, to_apply=%scalar_add_computation
  %bitcast.21.clone.1 = f32[392,672]{1,0} bitcast(f32[2,14,14,672]{3,2,1,0} %convert.9)
  %reduce.9.clone.1 = f32[672]{0} reduce(f32[392,672]{1,0} %bitcast.21.clone.1, f32[] %constant_58), dimensions={0}, to_apply=%scalar_add_computation
  ROOT %tuple.9 = (f32[672]{0}, f32[672]{0}) tuple(f32[672]{0} %reduce.8, f32[672]{0} %reduce.9.clone.1)
}

ENTRY %main {
  %param_0 = f32[672]{0} parameter(0)
  %param_1 = f16[2,14,14,672]{3,2,1,0} parameter(1)
  %param_2 = f16[2,14,14,672]{3,2,1,0} parameter(2)
  %param_3 = f32[672]{0} parameter(3)
  %param_4 = f32[672]{0} parameter(4)
  %param_5 = f32[672]{0} parameter(5)

  ROOT %fusion = (f32[672]{0}, f32[672]{0}) fusion(%param_0, %param_1, %param_2, %param_3, %param_4, %param_5), kind=kInput, calls=%fused_computation
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_TRUE(FusionBitcastLift().Run(module.get()).ValueOrDie());
  // Remove the old fusion not used anymore.
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  StatusOr<bool> filecheck_result = RunFileCheck(module->ToString(),
                                                 R"(
; CHECK-LABEL: %fused_computation
; CHECK-COUNT-8: bitcast(
; CHECK-NOT:     bitcast(
; CHECK-LABEL: ENTRY %main
; CHECK-COUNT-2: bitcast(
; CHECK-NOT:     bitcast(
; CHECK:         fusion(
      )");
  EXPECT_TRUE(filecheck_result.status().ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

TEST_F(FusionBitcastLiftTest, LayoutChangeNotSupported) {
  const char* hlo_text = R"(
HloModule bla

add {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  ROOT add = f32[] add(param0, param1)
}

fused_computation {
  param_1.11485 = f32[1,1,1536,3072]{3,2,1,0} parameter(1)
  copy.1383 = f32[1,1,1536,3072]{1,0,2,3} copy(param_1.11485)
  param_0.7122 = f32[3072]{0} parameter(0)
  constant.9031 = f32[] constant(0.000651041686)
  broadcast.9040 = f32[3072]{0} broadcast(constant.9031), dimensions={}
  multiply.7225 = f32[3072]{0} multiply(param_0.7122, broadcast.9040)
  broadcast.9039 = f32[1,1,1536,3072]{1,0,2,3} broadcast(multiply.7225), dimensions={3}
  subtract.940 = f32[1,1,1536,3072]{1,0,2,3} subtract(copy.1383, broadcast.9039)
  multiply.7224 = f32[1,1,1536,3072]{1,0,2,3} multiply(subtract.940, subtract.940)
  bitcast.3805 = f32[3072,1536]{1,0} bitcast(multiply.7224)
  constant.25971 = f32[] constant(0)
  ROOT reduce.790 = f32[3072]{0} reduce(bitcast.3805, constant.25971), dimensions={1}, to_apply=add
}

ENTRY entry {
  param_0.0 = f32[3072]{0} parameter(0)
  param_1.0 = f32[1,1,1536,3072]{3,2,1,0} parameter(1)
  ROOT fusion = f32[3072]{0} fusion(param_0.0, param_1.0), kind=kInput, calls=fused_computation
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
  EXPECT_FALSE(FusionBitcastLift().Run(module.get()).ValueOrDie());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
