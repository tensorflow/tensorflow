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
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"

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

  StatusOr<bool> filecheck_result =
      RunFileCheck(module->ToString(),
                    R"(
; CHECK-LABEL: %fused_computation
; CHECK:         f16[392,672]{1,0} parameter(0)
; CHECK-NOT:     parameter
; CHECK-NOT:     bitcast
; CHECK-LABEL: ENTRY %main
; CHECK-NEXT:    f16[2,14,14,672]{3,2,1,0} parameter(0)
; CHECK-NEXT:    bitcast(
; CHECK-NEXT:    ROOT %fusion.4d.bitcast
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

  StatusOr<bool> filecheck_result =
      RunFileCheck(module->ToString(),
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
; CHECK-NEXT:    ROOT %fusion.4d.bitcast
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

  StatusOr<bool> filecheck_result =
      RunFileCheck(module->ToString(),
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
; CHECK-NEXT:    ROOT %fusion.4d.bitcast
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

  StatusOr<bool> filecheck_result =
      RunFileCheck(module->ToString(),
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

}  // namespace
}  // namespace gpu
}  // namespace xla
