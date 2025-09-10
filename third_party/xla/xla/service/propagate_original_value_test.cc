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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/instruction_fusion.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using PropagateOriginalValueTest = HloHardwareIndependentTestBase;
using OriginalValueRecoveryTableTest = HloHardwareIndependentTestBase;

TEST_F(PropagateOriginalValueTest, InstructionFusion) {
  constexpr absl::string_view hlo_string = R"(
HloModule test, entry_computation_layout={(s32[]{:T(256)})->u32[2]{0:T(256)}}

ENTRY test {
  Arg_0 = s32[]{:T(256)} parameter(0), origin={{"Arg_0"}}, metadata={op_name="seed"}
  constant = s32[]{:T(256)} constant(32), origin={{"constant"}}
  shift-right-logical = s32[]{:T(256)} shift-right-logical(Arg_0, constant), origin={{"shift-right-logical"}}
  convert = u32[]{:T(256)} convert(shift-right-logical), origin={{"convert"}}
  bitcast = u32[1]{0:T(256)} bitcast(convert), origin={{"reshape"}}
  constant.1 = u32[]{:T(256)} constant(0)
  pad = u32[2]{0:T(256)} pad(bitcast, constant.1), padding=0_1
  convert.1 = u32[]{:T(256)} convert(Arg_0), origin={{"convert.1"}}
  bitcast.1 = u32[1]{0:T(256)} bitcast(convert.1), origin={{"reshape.1"}}
  pad.1 = u32[2]{0:T(256)} pad(bitcast.1, constant.1), padding=1_0
  ROOT add = u32[2]{0:T(256)} add(pad, pad.1), origin={{"concatenate"}}
}
  )";

  RunAndFilecheckHloRewrite(
      hlo_string,
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true),
      R"(
CHECK: %fused_computation
CHECK:   %[[PARAM:.*]] = s32[]{:T(256)} parameter(0)
CHECK:   %[[CONSTANT:.*]] = s32[]{:T(256)} constant(32), origin={{[{]}}{"constant"}}
CHECK:   %[[SHIFT:.*]] = s32[]{:T(256)} shift-right-logical(%[[PARAM]], %[[CONSTANT]]), origin={{[{]}}{"shift-right-logical"}
CHECK:   %[[CONVERT:.*]] = u32[]{:T(256)} convert(%[[SHIFT]]), origin={{[{]}}{"convert"}
CHECK:   %[[BITCAST:.*]] = u32[1]{0:T(256)} bitcast(%[[CONVERT]]), origin={{[{]}}{"reshape"}
CHECK:   %[[CONSTANT1:.*]] = u32[]{:T(256)} constant(0)
CHECK:   %[[PAD:.*]] = u32[2]{0:T(256)} pad(%[[BITCAST]], %[[CONSTANT1]]), padding=0_1
CHECK:   %[[CONVERT1:.*]] = u32[]{:T(256)} convert(%[[PARAM]]), origin={{[{]}}{"convert.1"}
CHECK:   %[[BITCAST1:.*]] = u32[1]{0:T(256)} bitcast(%[[CONVERT1]]), origin={{[{]}}{"reshape.1"}
CHECK:   %[[PAD1:.*]] = u32[2]{0:T(256)} pad(%[[BITCAST1]], %[[CONSTANT1]]), padding=1_0
CHECK:   ROOT %[[ADD:.*]] = u32[2]{0:T(256)} add(%[[PAD]], %[[PAD1]]), origin={{[{]}}{"concatenate"}

CHECK: ENTRY %test
CHECK:   %Arg_0 = s32[]{:T(256)} parameter(0), origin={{[{]}}{"Arg_0"}
CHECK:   ROOT %pad_add_fusion = u32[2]{0:T(256)} fusion(%Arg_0), kind=kLoop, calls=%fused_computation, origin={{[{]}}{"concatenate"}
)");
}

TEST_F(PropagateOriginalValueTest, CallInlinerMultipleCallSites) {
  const absl::string_view hlo_string = R"(
// CHECK-LABEL:test
// CHECK: %[[LHS:.*]] =
// CHECK:  %[[RHS1:.*]] = f32[] constant(2), origin={{[{]}}{"call.1/rhs"}
// CHECK: %[[ADD1:.*]] = f32[] add(%[[LHS]], %[[RHS1]]), origin={{[{]}}{"call.1/add"}
// CHECK:  %[[RHS2:.*]] = f32[] constant(2), origin={{[{]}}{"call.2/rhs"}
// CHECK: %[[ADD2:.*]] = f32[] add(%[[LHS]], %[[RHS2]]), origin={{[{]}}{"call.2/add"}

  HloModule test

  incr (lhs: f32[]) -> f32[] {
    lhs = f32[] parameter(0)
    rhs = f32[] constant(2), origin={{"rhs"}}
    ROOT add = f32[] add(f32[] lhs, f32[] rhs), origin={{"add"}}
  }

  ENTRY main () -> f32[] {
    lhs = f32[] constant(42)
    call.1 = f32[] call(f32[] lhs), to_apply=incr, origin={{"call.1"}}
    call.2 = f32[] call(f32[] lhs), to_apply=incr, origin={{"call.2"}}
    ROOT add = f32[] add(f32[] call.1, f32[] call.2)
  })";

  RunAndFilecheckHloRewrite(hlo_string,
                            CallInliner(/*single_call_site=*/false));
}

TEST_F(PropagateOriginalValueTest,
       CallInlinerMissingOriginalValueInCallInstruction) {
  const absl::string_view hlo_string = R"(
// CHECK-LABEL:test
// CHECK-NOT:origin
// CHECK-NOT:call(

  HloModule test

  incr (lhs: f32[]) -> f32[] {
    lhs = f32[] parameter(0)
    rhs = f32[] constant(2), origin={{"rhs"}}
    ROOT add = f32[] add(f32[] lhs, f32[] rhs), origin={{"add"}}
  }

  ENTRY main () -> f32[] {
    lhs = f32[] constant(42)
    ROOT call = f32[] call(f32[] lhs), to_apply=incr
  })";

  RunAndFilecheckHloRewrite(hlo_string,
                            CallInliner(/*single_call_site=*/false));
}

TEST_F(PropagateOriginalValueTest, CallInlinerSyntheticCallInstruction) {
  const absl::string_view hlo_string = R"(
// CHECK-LABEL:test
// CHECK: %[[LHS:.*]] =
// CHECK:  %[[RHS:.*]] = f32[] constant(2), origin={{[{]}}{"rhs"}
// CHECK: %[[ADD:.*]] = f32[] add(%[[LHS]], %[[RHS]]), origin={{[{]}}{"add"}
// CHECK-NOT:call(

  HloModule test

  incr (lhs: f32[]) -> f32[] {
    lhs = f32[] parameter(0)
    rhs = f32[] constant(2), origin={{"rhs"}}
    ROOT add = f32[] add(f32[] lhs, f32[] rhs), origin={{"add"}}
  }

  ENTRY main () -> f32[] {
    lhs = f32[] constant(42)
    ROOT call = f32[] call(f32[] lhs), to_apply=incr, origin={[synthetic_call]}
  })";

  RunAndFilecheckHloRewrite(hlo_string,
                            CallInliner(/*single_call_site=*/false));
}

TEST_F(OriginalValueRecoveryTableTest,
       AlgebraicSimplifierReshapeAndBroadcastMerged) {
  constexpr absl::string_view hlo_string = R"(

// CHECK:  HloModule test, entry_computation_layout={(f32[5]{0})->f32[1,2,3,5,1]{4,3,2,1,0}}, origin_recovery_table={
// CHECK:    {"reshape"} : {"param0"},
// CHECK:    "
// CHECK:      ENTRY %recovery_computation (p: f32[5]) -> f32[1,5,1] {
// CHECK:        %p = f32[5]{0} parameter(0)
// CHECK:        ROOT %reshape = f32[1,5,1]{2,1,0} reshape(%p)
// CHECK:      }
// CHECK:    "
// CHECK:  }

HloModule test

ENTRY %ReshapeAndBroadcastMerged (param0: f32[5]) -> f32[1,2,3,5,1] {
  %param0 = f32[5]{0} parameter(0), origin={{"param0"}}
  %reshape = f32[1,5,1]{2,1,0} reshape(%param0), origin={{"reshape"}}
  ROOT %broadcast = f32[1,2,3,5,1]{4,3,2,1,0} broadcast(%reshape), dimensions={0,3,4}
}

  )";

  AlgebraicSimplifierOptions options;
  RunAndFilecheckHloRewrite(hlo_string, AlgebraicSimplifier(options));
}

TEST_F(OriginalValueRecoveryTableTest,
       NullOriginalValueOnTupleGetTupleElementIsNotContagious) {
  constexpr absl::string_view hlo_string = R"(
// CHECK:      HloModule test, entry_computation_layout={((f32[5]{0}, f32[5]{0}))->f32[1,2,3,5,1]{4,3,2,1,0}}, origin_recovery_table={
// CHECK-NEXT:   {"reshape"} : {"reshape__ovp0"},
// CHECK-NEXT:   "
// CHECK-NEXT:     ENTRY %recovery_computation (p: f32[5]) -> f32[1,5,1] {
// CHECK-NEXT:       %p = f32[5]{0} parameter(0)
// CHECK-NEXT:       ROOT %reshape = f32[1,5,1]{2,1,0} reshape(%p)
// CHECK-NEXT:     }
// CHECK-NEXT:   "
// CHECK-NEXT: }
// CHECK:      ENTRY %main (param: (f32[5], f32[5])) -> f32[1,2,3,5,1] {
// CHECK-NEXT:   %param = (f32[5]{0}, f32[5]{0}) parameter(0)
// CHECK-NEXT:   %get-tuple-element = f32[5]{0} get-tuple-element(%param), index=1, origin={{[{]}}{"reshape__ovp0"}}
// CHECK-NEXT:   ROOT %broadcast = f32[1,2,3,5,1]{4,3,2,1,0} broadcast(%get-tuple-element), dimensions={3}
// CHECK-NEXT: }
  
HloModule test

ENTRY %main (param0: (f32[5]{0}, f32[5]{0})) -> f32[1,2,3,5,1] {
  %param = (f32[5]{0}, f32[5]{0}) parameter(0)
  %get-tuple-element = f32[5]{0} get-tuple-element(%param), index=1
  %reshape = f32[1,5,1]{2,1,0} reshape(%get-tuple-element), origin={{"reshape"}}
  ROOT %broadcast = f32[1,2,3,5,1]{4,3,2,1,0} broadcast(%reshape), dimensions={0,3,4}
}

  )";

  AlgebraicSimplifierOptions options;
  RunAndFilecheckHloRewrite(hlo_string, AlgebraicSimplifier(options));
}

}  // namespace
}  // namespace xla
