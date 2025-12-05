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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/literal_util.h"
#include "xla/service/call_inliner.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using PropagateOriginalValueTest = HloHardwareIndependentTestBase;
using OriginalValueRecoveryTableTest = HloHardwareIndependentTestBase;

TEST_F(PropagateOriginalValueTest, Clone) {
  HloComputation::Builder builder(TestName());
  auto* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<float>({
          {1, 2},
          {3, 4},
      })));
  constant->set_original_value(OriginalValue::CreateFromInstruction(constant));
  auto* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant, constant}));
  tuple->set_original_value(OriginalValue::CreateFromInstruction(constant));
  auto clone_shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto tuple_clone_same_shape = tuple->CloneWithNewOperands(
      ShapeUtil::MakeTupleShape({clone_shape, clone_shape}), {});
  clone_shape = ShapeUtil::MakeShape(F32, {3, 3});
  auto tuple_clone_different_shape = tuple->CloneWithNewOperands(
      ShapeUtil::MakeTupleShape({clone_shape, clone_shape}), {});
  // Only the tuple clone with the same shape as the original tuple should
  // preserve the original value.
  EXPECT_TRUE(tuple_clone_same_shape->original_value());
  EXPECT_FALSE(tuple_clone_different_shape->original_value());
}

TEST_F(PropagateOriginalValueTest, ReplaceAllUses) {
  const absl::string_view hlo_string = R"(
HloModule test

ENTRY %main (param: s32[2,8], param.1: s32[8,8]) -> s32[2,8] {
  %param = s32[2,8]{1,0:T(2,128)} parameter(0)
  %param.1 = s32[8,8]{1,0:T(8,128)} parameter(1)
  ROOT %convolution = s32[2,8]{1,0:T(2,128)} convolution(%param, %param.1), dim_labels=bf_io->bf, origin={{"dot_general.1__ovp0"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* entry_computation = module->entry_computation();
  HloInstruction* root = entry_computation->root_instruction();
  HloInstruction* new_root = entry_computation->AddInstruction(root->Clone());
  new_root->set_original_value(nullptr);

  ASSERT_OK(root->ReplaceAllUsesWith(new_root));
  EXPECT_NE(new_root->original_value(), nullptr);
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
// CHECK:          ENTRY %recovery_computation (p: f32[5]) -> f32[1,5,1] {
// CHECK-NEXT:       %p = f32[5]{0} parameter(0)
// CHECK-NEXT:       ROOT %reshape = f32[1,5,1]{2,1,0} reshape(%p)
// CHECK-NEXT:     }
// CHECK:        "
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
