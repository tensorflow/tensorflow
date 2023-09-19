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

#include "xla/service/collective_into_scan_loop_code_motion.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class CollectiveScanLoopCodeMotionTest : public HloTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> ParseModuleAndRunLoopCodeMotion(
      absl::string_view hlo_text, HloOpcode opcode = HloOpcode::kAllReduce) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                        ParseAndReturnVerifiedModule(hlo_text));
    CollectiveIntoScanLoopCodeMotion pass(opcode);
    TF_RETURN_IF_ERROR(pass.Run(hlo_module.get()).status());
    return hlo_module;
  }
};

TEST_F(CollectiveScanLoopCodeMotionTest, OutputDoesNotHaveOtherUsers) {
  const absl::string_view hlo_text = R"(
HloModule test

add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

loop_body {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  inputs = s32[5] get-tuple-element(arg_tuple), index=1
  outputs = s32[5] get-tuple-element(arg_tuple), index=2
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  input = s32[1] dynamic-slice(inputs, i), dynamic_slice_sizes={1}
  updated_outputs = s32[5] dynamic-update-slice(outputs, input, i)
  ROOT tuple = (s32[], s32[5], s32[5]) tuple(next_i, inputs, updated_outputs)
}

loop_condition {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  five = s32[] constant(5)
  ROOT cmp = pred[] compare(i, five), direction=LT
}

ENTRY main {
  inputs_init = s32[5] iota(), iota_dimension=0
  zero = s32[] constant(0)
  outputs_init = s32[5] broadcast(zero), dimensions={}
  loop_init = (s32[], s32[5], s32[5]) tuple(zero, inputs_init, outputs_init)
  while = (s32[], s32[5], s32[5]) while(loop_init), condition=loop_condition, body=loop_body
  outputs = s32[5] get-tuple-element(while), index=2
  ROOT ar = s32[5] all-reduce(outputs), to_apply=add
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> updated_module,
                          ParseModuleAndRunLoopCodeMotion(hlo_text));

  HloPrintOptions options = HloPrintOptions().set_print_operand_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(bool match,
                          RunFileCheck(updated_module->ToString(options), R"(
; CHECK-LABEL: %loop_body
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5])) -> (s32[], s32[5], s32[5]) {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[GTE0:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[GTE1:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=2
; CHECK-DAG:     [[INPUT:%[^ ]+]] = s32[1]{0} dynamic-slice([[GTE1]], [[GTE0]]), dynamic_slice_sizes={1}
; CHECK-DAG:     [[ALL_REDUCE:%[^ ]+]] = s32[1]{0} all-reduce([[INPUT]])
; CHECK-DAG:     [[AR_OUTPUTS:%[^ ]+]] = s32[5]{0} dynamic-update-slice([[GTE2]], [[ALL_REDUCE]], [[GTE0]])
; CHECK-DAG:     [[ADD:%[^ ]+]] = s32[] add
; CHECK:         ROOT [[RET:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple([[ADD]], [[GTE1]], [[AR_OUTPUTS]])

; CHECK-LABEL: %loop_condition
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5])) -> pred[] {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[I:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[FIVE:%[^ ]+]] = s32[] constant(5)
; CHECK:         ROOT [[CMP:%[^ ]+]] = pred[] compare([[I]], [[FIVE]]), direction=LT

; CHECK-LABEL: ENTRY %main () -> s32[5] {
; CHECK:         [[WHILE_INIT:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple
; CHECK:         [[WHILE:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) while([[WHILE_INIT]])
; CHECK-DAG:     ROOT [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element([[WHILE]]), index=2
      )"));
  EXPECT_TRUE(match);
}

TEST_F(CollectiveScanLoopCodeMotionTest, OutputHasOtherUsers) {
  const absl::string_view hlo_text = R"(
HloModule test

add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

loop_body {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  inputs = s32[5] get-tuple-element(arg_tuple), index=1
  outputs = s32[5] get-tuple-element(arg_tuple), index=2
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  input = s32[1] dynamic-slice(inputs, i), dynamic_slice_sizes={1}
  updated_outputs = s32[5] dynamic-update-slice(outputs, input, i)
  ROOT tuple = (s32[], s32[5], s32[5]) tuple(next_i, inputs, updated_outputs)
}

loop_condition {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  five = s32[] constant(5)
  ROOT cmp = pred[] compare(i, five), direction=LT
}

ENTRY main {
  inputs_init = s32[5] iota(), iota_dimension=0
  zero = s32[] constant(0)
  outputs_init = s32[5] broadcast(zero), dimensions={}
  loop_init = (s32[], s32[5], s32[5]) tuple(zero, inputs_init, outputs_init)
  while = (s32[], s32[5], s32[5]) while(loop_init), condition=loop_condition, body=loop_body
  outputs = s32[5] get-tuple-element(while), index=2
  ar = s32[5] all-reduce(outputs), to_apply=add
  ROOT ret = (s32[5], s32[5]) tuple(outputs, ar)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> updated_module,
                          ParseModuleAndRunLoopCodeMotion(hlo_text));

  HloPrintOptions options = HloPrintOptions().set_print_operand_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(bool match,
                          RunFileCheck(updated_module->ToString(options), R"(
; CHECK-LABEL: %loop_body
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5], s32[5])) -> (s32[], s32[5], s32[5], s32[5]) {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[GTE0:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[GTE1:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=2
; CHECK-DAG:     [[GTE3:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=3
; CHECK-DAG:     [[INPUT:%[^ ]+]] = s32[1]{0} dynamic-slice([[GTE1]], [[GTE0]]), dynamic_slice_sizes={1}
; CHECK-DAG:     [[OUTPUTS:%[^ ]+]] = s32[5]{0} dynamic-update-slice([[GTE2]], [[INPUT]], [[GTE0]])
; CHECK-DAG:     [[ALL_REDUCE:%[^ ]+]] = s32[1]{0} all-reduce([[INPUT]])
; CHECK-DAG:     [[AR_OUTPUTS:%[^ ]+]] = s32[5]{0} dynamic-update-slice([[GTE3]], [[ALL_REDUCE]], [[GTE0]])
; CHECK-DAG:     [[ADD:%[^ ]+]] = s32[] add
; CHECK:         ROOT [[RET:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0})
; CHECK-SAME:      tuple([[ADD]], [[GTE1]], [[OUTPUTS]], [[AR_OUTPUTS]])

; CHECK-LABEL: %loop_condition
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5], s32[5])) -> pred[] {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[I:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[FIVE:%[^ ]+]] = s32[] constant(5)
; CHECK:         ROOT [[CMP:%[^ ]+]] = pred[] compare([[I]], [[FIVE]]), direction=LT

; CHECK-LABEL: ENTRY %main () -> (s32[5], s32[5]) {
; CHECK:         [[WHILE_INIT:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0}) tuple
; CHECK:         [[WHILE:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0}) while([[WHILE_INIT]])
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element([[WHILE]]), index=2
; CHECK-DAG:     [[GTE3:%[^ ]+]] = s32[5]{0} get-tuple-element([[WHILE]]), index=3
; CHECK:         ROOT [[OUT:%[^ ]+]] = (s32[5]{0}, s32[5]{0}) tuple([[GTE2]], [[GTE3]])
      )"));
  EXPECT_TRUE(match);
}

TEST_F(CollectiveScanLoopCodeMotionTest, TwoCollectives) {
  const absl::string_view hlo_text = R"(
HloModule test

add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

loop_body {
  arg_tuple = (s32[], s32[5], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  inputs = s32[5] get-tuple-element(arg_tuple), index=1
  outputs0 = s32[5] get-tuple-element(arg_tuple), index=2
  outputs1 = s32[5] get-tuple-element(arg_tuple), index=3
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  input = s32[1] dynamic-slice(inputs, i), dynamic_slice_sizes={1}
  updated_outputs0 = s32[5] dynamic-update-slice(outputs0, input, i)
  updated_outputs1 = s32[5] dynamic-update-slice(outputs1, input, i)
  ROOT tuple = (s32[], s32[5], s32[5], s32[5]) tuple(next_i, inputs, updated_outputs0, updated_outputs1)
}

loop_condition {
  arg_tuple = (s32[], s32[5], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  five = s32[] constant(5)
  ROOT cmp = pred[] compare(i, five), direction=LT
}

ENTRY main {
  inputs_init = s32[5] iota(), iota_dimension=0
  zero = s32[] constant(0)
  outputs_init = s32[5] broadcast(zero), dimensions={}
  loop_init = (s32[], s32[5], s32[5], s32[5]) tuple(zero, inputs_init, outputs_init, outputs_init)
  while = (s32[], s32[5], s32[5], s32[5]) while(loop_init), condition=loop_condition, body=loop_body
  outputs0 = s32[5] get-tuple-element(while), index=2
  outputs1 = s32[5] get-tuple-element(while), index=3
  ar0 = s32[5] all-reduce(outputs0), to_apply=add
  ar1 = s32[5] all-reduce(outputs1), to_apply=add
  ROOT ret = (s32[5], s32[5]) tuple(ar0, ar1)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> updated_module,
                          ParseModuleAndRunLoopCodeMotion(hlo_text));

  HloPrintOptions options = HloPrintOptions().set_print_operand_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(bool match,
                          RunFileCheck(updated_module->ToString(options), R"(
; CHECK-LABEL: %loop_body
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5], s32[5])) -> (s32[], s32[5], s32[5], s32[5]) {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[GTE0:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[GTE1:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=2
; CHECK-DAG:     [[GTE3:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=3
; CHECK-DAG:     [[INPUT:%[^ ]+]] = s32[1]{0} dynamic-slice([[GTE1]], [[GTE0]]), dynamic_slice_sizes={1}
; CHECK-DAG:     [[ALL_REDUCE0:%[^ ]+]] = s32[1]{0} all-reduce([[INPUT]])
; CHECK-DAG:     [[ALL_REDUCE1:%[^ ]+]] = s32[1]{0} all-reduce([[INPUT]])
; CHECK-DAG:     [[AR_OUTPUTS0:%[^ ]+]] = s32[5]{0} dynamic-update-slice([[GTE2]], [[ALL_REDUCE0]], [[GTE0]])
; CHECK-DAG:     [[AR_OUTPUTS1:%[^ ]+]] = s32[5]{0} dynamic-update-slice([[GTE3]], [[ALL_REDUCE1]], [[GTE0]])
; CHECK-DAG:     [[ADD:%[^ ]+]] = s32[] add
; CHECK:         ROOT [[RET:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0})
; CHECK-SAME:      tuple([[ADD]], [[GTE1]], [[AR_OUTPUTS0]], [[AR_OUTPUTS1]])

; CHECK-LABEL: %loop_condition
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5], s32[5])) -> pred[] {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[I:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[FIVE:%[^ ]+]] = s32[] constant(5)
; CHECK:         ROOT [[CMP:%[^ ]+]] = pred[] compare([[I]], [[FIVE]]), direction=LT

; CHECK-LABEL: ENTRY %main () -> (s32[5], s32[5]) {
; CHECK:         [[WHILE_INIT:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0}) tuple
; CHECK:         [[WHILE:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}, s32[5]{0}) while([[WHILE_INIT]])
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element([[WHILE]]), index=2
; CHECK-DAG:     [[GTE3:%[^ ]+]] = s32[5]{0} get-tuple-element([[WHILE]]), index=3
; CHECK;         ROOT [[RET:%[^ ]+]] = (s32[5]{0}, s32[5]{0})) tuple([[GTE2]], [[GTE3]])
      )"));
  EXPECT_TRUE(match);
}

TEST_F(CollectiveScanLoopCodeMotionTest, HigherDimensionality) {
  const absl::string_view hlo_text = R"(
HloModule test

add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

loop_body {
  arg_tuple = (s32[], s32[5,6,7], s32[5,6,7]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  inputs = s32[5,6,7] get-tuple-element(arg_tuple), index=1
  outputs = s32[5,6,7] get-tuple-element(arg_tuple), index=2
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  zero = s32[] constant(0)
  input = s32[1,6,7] dynamic-slice(inputs, i, zero, zero), dynamic_slice_sizes={1,6,7}
  updated_outputs = s32[5,6,7] dynamic-update-slice(outputs, input, i, zero, zero)
  ROOT tuple = (s32[], s32[5,6,7], s32[5,6,7]) tuple(next_i, inputs, updated_outputs)
}

loop_condition {
  arg_tuple = (s32[], s32[5,6,7], s32[5,6,7]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  five = s32[] constant(5)
  ROOT cmp = pred[] compare(i, five), direction=LT
}

ENTRY main {
  inputs_init = s32[5,6,7] iota(), iota_dimension=0
  zero = s32[] constant(0)
  outputs_init = s32[5,6,7] broadcast(zero), dimensions={}
  loop_init = (s32[], s32[5,6,7], s32[5,6,7]) tuple(zero, inputs_init, outputs_init)
  while = (s32[], s32[5,6,7], s32[5,6,7]) while(loop_init), condition=loop_condition, body=loop_body
  outputs = s32[5,6,7] get-tuple-element(while), index=2
  ROOT ar = s32[5,6,7] all-reduce(outputs), to_apply=add
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> updated_module,
                          ParseModuleAndRunLoopCodeMotion(hlo_text));

  HloPrintOptions options = HloPrintOptions().set_print_operand_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(bool match,
                          RunFileCheck(updated_module->ToString(options), R"(
; CHECK-LABEL: %loop_body
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5,6,7], s32[5,6,7])) -> (s32[], s32[5,6,7], s32[5,6,7]) {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5,6,7]{2,1,0}, s32[5,6,7]{2,1,0}) parameter(0)
; CHECK-DAG:     [[GTE0:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[GTE1:%[^ ]+]] = s32[5,6,7]{2,1,0} get-tuple-element(%[[ARG_TUPLE]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5,6,7]{2,1,0} get-tuple-element(%[[ARG_TUPLE]]), index=2
; CHECK-DAG:     [[ZERO:%[^ ]+]] = s32[] constant(0)
; CHECK-DAG:     [[INPUT:%[^ ]+]] = s32[1,6,7]{2,1,0} dynamic-slice([[GTE1]], [[GTE0]], [[ZERO]], [[ZERO]]), dynamic_slice_sizes={1,6,7}
; CHECK-DAG:     [[ALL_REDUCE:%[^ ]+]] = s32[1,6,7]{2,1,0} all-reduce([[INPUT]])
; CHECK-DAG:     [[AR_OUTPUTS:%[^ ]+]] = s32[5,6,7]{2,1,0} dynamic-update-slice([[GTE2]], [[ALL_REDUCE]], [[GTE0]], [[ZERO]], [[ZERO]])
; CHECK-DAG:     [[ADD:%[^ ]+]] = s32[] add
; CHECK:         ROOT [[RET:%[^ ]+]] = (s32[], s32[5,6,7]{2,1,0}, s32[5,6,7]{2,1,0}) tuple([[ADD]], [[GTE1]], [[AR_OUTPUTS]])

; CHECK-LABEL: ENTRY %main () -> s32[5,6,7] {
; CHECK-DAG:     [[WHILE_INIT:%[^ ]+]] = (s32[], s32[5,6,7]{2,1,0}, s32[5,6,7]{2,1,0}) tuple
; CHECK-DAG:     [[WHILE:%[^ ]+]] = (s32[], s32[5,6,7]{2,1,0}, s32[5,6,7]{2,1,0}) while([[WHILE_INIT]])
; CHECK-DAG:     ROOT [[GTE2:%[^ ]+]] = s32[5,6,7]{2,1,0} get-tuple-element([[WHILE]]), index=2
      )"));
  EXPECT_TRUE(match);
}

TEST_F(CollectiveScanLoopCodeMotionTest, CollectivePermute) {
  const absl::string_view hlo_text = R"(
HloModule test

loop_body {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  inputs = s32[5] get-tuple-element(arg_tuple), index=1
  outputs = s32[5] get-tuple-element(arg_tuple), index=2
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  input = s32[1] dynamic-slice(inputs, i), dynamic_slice_sizes={1}
  updated_outputs = s32[5] dynamic-update-slice(outputs, input, i)
  ROOT tuple = (s32[], s32[5], s32[5]) tuple(next_i, inputs, updated_outputs)
}

loop_condition {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  five = s32[] constant(5)
  ROOT cmp = pred[] compare(i, five), direction=LT
}

ENTRY main {
  inputs_init = s32[5] iota(), iota_dimension=0
  zero = s32[] constant(0)
  outputs_init = s32[5] broadcast(zero), dimensions={}
  loop_init = (s32[], s32[5], s32[5]) tuple(zero, inputs_init, outputs_init)
  while = (s32[], s32[5], s32[5]) while(loop_init), condition=loop_condition, body=loop_body
  outputs = s32[5] get-tuple-element(while), index=2
  ROOT permute = s32[5] collective-permute(outputs), source_target_pairs={{0,1}}
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> updated_module,
      ParseModuleAndRunLoopCodeMotion(hlo_text, HloOpcode::kCollectivePermute));

  HloPrintOptions options = HloPrintOptions().set_print_operand_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(bool match,
                          RunFileCheck(updated_module->ToString(options), R"(
; CHECK-LABEL: %loop_body
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5])) -> (s32[], s32[5], s32[5]) {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[GTE0:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[GTE1:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=2
; CHECK-DAG:     [[INPUT:%[^ ]+]] = s32[1]{0} dynamic-slice([[GTE1]], [[GTE0]]), dynamic_slice_sizes={1}
; CHECK-DAG:     [[PERMUTE:%[^ ]+]] = s32[1]{0} collective-permute([[INPUT]]),
; CHECK-SAME{LITERAL}  source_target_pairs={{0,1}}
; CHECK-DAG:     [[PERMUTE_OUTPUTS:%[^ ]+]] = s32[5]{0} dynamic-update-slice([[GTE2]], [[PERMUTE]], [[GTE0]])
; CHECK-DAG:     [[ADD:%[^ ]+]] = s32[] add
; CHECK:         ROOT [[RET:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple([[ADD]], [[GTE1]], [[PERMUTE_OUTPUTS]])

; CHECK-LABEL: ENTRY %main () -> s32[5] {
; CHECK:         [[WHILE_INIT:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple
; CHECK:         [[WHILE:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) while([[WHILE_INIT]])
; CHECK-DAG:     ROOT [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element([[WHILE]]), index=2
      )"));
  EXPECT_TRUE(match);
}

TEST_F(CollectiveScanLoopCodeMotionTest, SelectOnUpdateIdx) {
  const absl::string_view hlo_text = R"(
HloModule test

add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

loop_body {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  inputs = s32[5] get-tuple-element(arg_tuple), index=1
  outputs = s32[5] get-tuple-element(arg_tuple), index=2
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  zero = s32[] constant(0)
  i_lt_zero = pred[] compare(i, zero), direction=LT
  five = s32[] constant(5)
  i_plus_five = s32[] add(i, five)
  idx = s32[] select(i_lt_zero, i_plus_five, i)
  input = s32[1] dynamic-slice(inputs, idx), dynamic_slice_sizes={1}
  updated_outputs = s32[5] dynamic-update-slice(outputs, input, idx)
  ROOT tuple = (s32[], s32[5], s32[5]) tuple(next_i, inputs, updated_outputs)
}

loop_condition {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  five = s32[] constant(5)
  ROOT cmp = pred[] compare(i, five), direction=LT
}

ENTRY main {
  inputs_init = s32[5] iota(), iota_dimension=0
  zero = s32[] constant(0)
  outputs_init = s32[5] broadcast(zero), dimensions={}
  loop_init = (s32[], s32[5], s32[5]) tuple(zero, inputs_init, outputs_init)
  while = (s32[], s32[5], s32[5]) while(loop_init), condition=loop_condition, body=loop_body
  outputs = s32[5] get-tuple-element(while), index=2
  ROOT ar = s32[5] all-reduce(outputs), to_apply=add
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> updated_module,
                          ParseModuleAndRunLoopCodeMotion(hlo_text));

  HloPrintOptions options = HloPrintOptions().set_print_operand_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(bool match,
                          RunFileCheck(updated_module->ToString(options), R"(
; CHECK-LABEL: %loop_body
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5])) -> (s32[], s32[5], s32[5]) {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[GTE0:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[GTE1:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=2
; CHECK-DAG:     [[IDX:%[^ ]+]] = s32[] select
; CHECK-DAG:     [[INPUT:%[^ ]+]] = s32[1]{0} dynamic-slice([[GTE1]], [[IDX]]), dynamic_slice_sizes={1}
; CHECK-DAG:     [[ALL_REDUCE:%[^ ]+]] = s32[1]{0} all-reduce([[INPUT]])
; CHECK-DAG:     [[AR_OUTPUTS:%[^ ]+]] = s32[5]{0} dynamic-update-slice([[GTE2]], [[ALL_REDUCE]], [[IDX]])
; CHECK-DAG:     [[ADD:%[^ ]+]] = s32[] add
; CHECK:         ROOT [[RET:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple([[ADD]], [[GTE1]], [[AR_OUTPUTS]])

; CHECK-LABEL: ENTRY %main () -> s32[5] {
; CHECK:         [[WHILE_INIT:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple
; CHECK:         [[WHILE:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) while([[WHILE_INIT]])
; CHECK-DAG:     ROOT [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element([[WHILE]]), index=2
      )"));
  EXPECT_TRUE(match);
}

TEST_F(CollectiveScanLoopCodeMotionTest, ReversedUpdateIdx) {
  const absl::string_view hlo_text = R"(
HloModule test

add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

loop_body {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  inputs = s32[5] get-tuple-element(arg_tuple), index=1
  outputs = s32[5] get-tuple-element(arg_tuple), index=2
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  four = s32[] constant(4)
  idx = s32[] subtract(four, i)
  input = s32[1] dynamic-slice(inputs, idx), dynamic_slice_sizes={1}
  updated_outputs = s32[5] dynamic-update-slice(outputs, input, idx)
  ROOT tuple = (s32[], s32[5], s32[5]) tuple(next_i, inputs, updated_outputs)
}

loop_condition {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  five = s32[] constant(5)
  ROOT cmp = pred[] compare(i, five), direction=LT
}

ENTRY main {
  inputs_init = s32[5] iota(), iota_dimension=0
  zero = s32[] constant(0)
  outputs_init = s32[5] broadcast(zero), dimensions={}
  loop_init = (s32[], s32[5], s32[5]) tuple(zero, inputs_init, outputs_init)
  while = (s32[], s32[5], s32[5]) while(loop_init), condition=loop_condition, body=loop_body
  outputs = s32[5] get-tuple-element(while), index=2
  ROOT ar = s32[5] all-reduce(outputs), to_apply=add
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> updated_module,
                          ParseModuleAndRunLoopCodeMotion(hlo_text));

  HloPrintOptions options = HloPrintOptions().set_print_operand_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(bool match,
                          RunFileCheck(updated_module->ToString(options), R"(
; CHECK-LABEL: %loop_body
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5])) -> (s32[], s32[5], s32[5]) {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[GTE0:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[GTE1:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=2
; CHECK-DAG:     [[IDX:%[^ ]+]] = s32[] sub
; CHECK-DAG:     [[INPUT:%[^ ]+]] = s32[1]{0} dynamic-slice([[GTE1]], [[IDX]]), dynamic_slice_sizes={1}
; CHECK-DAG:     [[ALL_REDUCE:%[^ ]+]] = s32[1]{0} all-reduce([[INPUT]])
; CHECK-DAG:     [[AR_OUTPUTS:%[^ ]+]] = s32[5]{0} dynamic-update-slice([[GTE2]], [[ALL_REDUCE]], [[IDX]])
; CHECK-DAG:     [[ADD:%[^ ]+]] = s32[] add
; CHECK:         ROOT [[RET:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple([[ADD]], [[GTE1]], [[AR_OUTPUTS]])

; CHECK-LABEL: ENTRY %main () -> s32[5] {
; CHECK:         [[WHILE_INIT:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple
; CHECK:         [[WHILE:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) while([[WHILE_INIT]])
; CHECK-DAG:     ROOT [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element([[WHILE]]), index=2
      )"));
  EXPECT_TRUE(match);
}

TEST_F(CollectiveScanLoopCodeMotionTest, SelectOnReversedUpdateIdx) {
  const absl::string_view hlo_text = R"(
HloModule test

add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

loop_body {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  inputs = s32[5] get-tuple-element(arg_tuple), index=1
  outputs = s32[5] get-tuple-element(arg_tuple), index=2
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  four = s32[] constant(4)
  idx = s32[] subtract(four, i)
  zero = s32[] constant(0)
  idx_lt_zero = pred[] compare(idx, zero), direction=LT
  nine = s32[] constant(9)
  idx_plus_five = s32[] subtract(nine, i)
  idx2 = s32[] select(idx_lt_zero, idx_plus_five, idx)
  input = s32[1] dynamic-slice(inputs, idx2), dynamic_slice_sizes={1}
  updated_outputs = s32[5] dynamic-update-slice(outputs, input, idx2)
  ROOT tuple = (s32[], s32[5], s32[5]) tuple(next_i, inputs, updated_outputs)
}

loop_condition {
  arg_tuple = (s32[], s32[5], s32[5]) parameter(0)
  i = s32[] get-tuple-element(arg_tuple), index=0
  five = s32[] constant(5)
  ROOT cmp = pred[] compare(i, five), direction=LT
}

ENTRY main {
  inputs_init = s32[5] iota(), iota_dimension=0
  zero = s32[] constant(0)
  outputs_init = s32[5] broadcast(zero), dimensions={}
  loop_init = (s32[], s32[5], s32[5]) tuple(zero, inputs_init, outputs_init)
  while = (s32[], s32[5], s32[5]) while(loop_init), condition=loop_condition, body=loop_body
  outputs = s32[5] get-tuple-element(while), index=2
  ROOT ar = s32[5] all-reduce(outputs), to_apply=add
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> updated_module,
                          ParseModuleAndRunLoopCodeMotion(hlo_text));

  HloPrintOptions options = HloPrintOptions().set_print_operand_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(bool match,
                          RunFileCheck(updated_module->ToString(options), R"(
; CHECK-LABEL: %loop_body
; CHECK-SAME:  ([[ARG_TUPLE:[^ ]+]]: (s32[], s32[5], s32[5])) -> (s32[], s32[5], s32[5]) {
; CHECK-DAG:     %[[ARG_TUPLE]] = (s32[], s32[5]{0}, s32[5]{0}) parameter(0)
; CHECK-DAG:     [[GTE0:%[^ ]+]] = s32[] get-tuple-element(%[[ARG_TUPLE]]), index=0
; CHECK-DAG:     [[GTE1:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=1
; CHECK-DAG:     [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element(%[[ARG_TUPLE]]), index=2
; CHECK-DAG:     [[IDX:%[^ ]+]] = s32[] select
; CHECK-DAG:     [[INPUT:%[^ ]+]] = s32[1]{0} dynamic-slice([[GTE1]], [[IDX]]), dynamic_slice_sizes={1}
; CHECK-DAG:     [[ALL_REDUCE:%[^ ]+]] = s32[1]{0} all-reduce([[INPUT]])
; CHECK-DAG:     [[AR_OUTPUTS:%[^ ]+]] = s32[5]{0} dynamic-update-slice([[GTE2]], [[ALL_REDUCE]], [[IDX]])
; CHECK-DAG:     [[ADD:%[^ ]+]] = s32[] add
; CHECK:         ROOT [[RET:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple([[ADD]], [[GTE1]], [[AR_OUTPUTS]])

; CHECK-LABEL: ENTRY %main () -> s32[5] {
; CHECK:         [[WHILE_INIT:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) tuple
; CHECK:         [[WHILE:%[^ ]+]] = (s32[], s32[5]{0}, s32[5]{0}) while([[WHILE_INIT]])
; CHECK-DAG:     ROOT [[GTE2:%[^ ]+]] = s32[5]{0} get-tuple-element([[WHILE]]), index=2
      )"));
  EXPECT_TRUE(match);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
