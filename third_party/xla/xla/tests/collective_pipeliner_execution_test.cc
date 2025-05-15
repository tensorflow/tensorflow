/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/collective_pipeliner_utils.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace {

using CollectivePipelinerExecutionTest = HloPjRtTestBase;

absl::StatusOr<bool> RunOptimizer(
    HloModule* module, bool last_run, int64_t level_to_operate_on = 0,
    HloPredicate should_process = HloPredicateIsOp<HloOpcode::kNegate>,
    collective_pipeliner_utils::PipeliningDirection pipelining_direction =
        collective_pipeliner_utils::PipeliningDirection::kForward,
    bool pipeline_use_tree = false,
    HloPredicate acceptable_formatting = HloPredicateTrue,
    HloPredicate reuse_pipelined_op_buffer = HloPredicateTrue) {
  CollectivePipeliner::Config config = {
      /*level_to_operate_on=*/level_to_operate_on,
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/last_run,
      /*pipeline_use_tree=*/pipeline_use_tree,
      /*process_different_sized_ops=*/true,
      /*direction=*/
      pipelining_direction,
      /*should_process=*/should_process,
      /*acceptable_formatting=*/acceptable_formatting,
      /*reuse_pipelined_op_buffer=*/reuse_pipelined_op_buffer,
  };

  HloPassPipeline pass("optimizer");
  pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/false);
  pass.AddPass<CollectivePipeliner>(config);
  pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/false);
  TF_ASSIGN_OR_RETURN(const bool modified, pass.Run(module));
  HloPassPipeline pass_dce("dce");
  pass_dce.AddPass<HloDCE>(/*remove_cross_partition_collective_ops=*/true);
  return modified;
}

TEST_F(CollectivePipelinerExecutionTest, TransformIncrementIndexByOne) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0).value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, TransformIncrementIndexByOneNoReuse) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, /*level_to_operate_on=*/0,
                   /*should_process=*/HloPredicateIsOp<HloOpcode::kNegate>,
                   /*pipelining_direction=*/
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   /*pipeline_use_tree=*/false,
                   /*acceptable_formatting=*/HloPredicateTrue,
                   /*reuse_pipelined_op_buffer=*/HloPredicateFalse)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, PushAgOver) {
  constexpr absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(bf16[3,8,128]{2,1,0})->bf16[3,8,128]{2,1,0}}

%add (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %add = bf16[] add(bf16[] %lhs, bf16[] %rhs)
}

%while_body.clone (loop_peel_param: (s32[], bf16[3,8,128], s32[])) -> (s32[], bf16[3,8,128], s32[]) {
  %loop_peel_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=0
  %constant.7 = s32[] constant(1)
  %add.4 = s32[] add(s32[] %get-tuple-element.2, s32[] %constant.7)
  %get-tuple-element.3 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=1
  %get-tuple-element.4 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=2
  %constant.12 = s64[] constant(1)
  %custom-call = s32[] custom-call(s32[] %get-tuple-element.4, s64[] %constant.12), custom_call_target="InsertedByPreviousStep"
  %constant.13 = s32[] constant(0)
  %constant.10 = s32[] constant(0)
  %dynamic-slice.2 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13), dynamic_slice_sizes={1,8,128}
  %ar.2 = bf16[1,8,128]{2,1,0} negate(bf16[1,8,128]{2,1,0} %dynamic-slice.2)
  %ag.2 = bf16[1,8,128]{2,1,0} negate(bf16[1,8,128]{2,1,0} %ar.2)
  %dynamic-update-slice.2 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, bf16[1,8,128]{2,1,0} %ag.2, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13)
  %dynamic-slice.1 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %get-tuple-element.2, s32[] %constant.10, s32[] %constant.10), dynamic_slice_sizes={1,8,128}
  %mul.2 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.1, bf16[1,8,128]{2,1,0} %dynamic-slice.1)
  %constant.15 = s32[] constant(0)
  %dynamic-update-slice.4 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %dynamic-update-slice.2, bf16[1,8,128]{2,1,0} %mul.2, s32[] %get-tuple-element.2, s32[] %constant.15, s32[] %constant.15)
  ROOT %tuple.3 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.4, bf16[3,8,128]{2,1,0} %dynamic-update-slice.4, s32[] %get-tuple-element.2)
}

%while_cond.clone (loop_peel_cond_param: (s32[], bf16[3,8,128], s32[])) -> pred[] {
  %loop_peel_cond_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %gte.1 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_cond_param), index=0
  %constant.6 = s32[] constant(0)
  ROOT %cmp.1 = pred[] compare(s32[] %gte.1, s32[] %constant.6), direction=LT
}

ENTRY %entry (p0: bf16[3,8,128]) -> bf16[3,8,128] {
  %c0 = s32[] constant(-3)
  %p0 = bf16[3,8,128]{2,1,0} parameter(0)
  %tuple.1 = (s32[], bf16[3,8,128]{2,1,0}) tuple(s32[] %c0, bf16[3,8,128]{2,1,0} %p0)
  %get-tuple-element.0 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=0
  %constant.0 = s32[] constant(1)
  %constant.4 = s32[] constant(0)
  %add.1 = s32[] add(s32[] %get-tuple-element.0, s32[] %constant.0)
  %get-tuple-element.1 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=1
  %dynamic-slice.0 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4), dynamic_slice_sizes={1,8,128}
  %mul.1 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.0, bf16[1,8,128]{2,1,0} %dynamic-slice.0)
  %dynamic-update-slice.0 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, bf16[1,8,128]{2,1,0} %mul.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4)
  %tuple.4 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.1, bf16[3,8,128]{2,1,0} %dynamic-update-slice.0, s32[] %get-tuple-element.0)
  %while.1 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) while((s32[], bf16[3,8,128]{2,1,0}, s32[]) %tuple.4), condition=%while_cond.clone, body=%while_body.clone
  %get-tuple-element.6 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=1
  %get-tuple-element.5 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=2
  %constant.14 = s32[] constant(0)
  %dynamic-slice.3 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14), dynamic_slice_sizes={1,8,128}
  %ar.3 = bf16[1,8,128]{2,1,0} add(bf16[1,8,128]{2,1,0} %dynamic-slice.3, bf16[1,8,128]{2,1,0} %dynamic-slice.3)
  ROOT %dynamic-update-slice.3 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, bf16[1,8,128]{2,1,0} %ar.3, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14)
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 1).value());
  EXPECT_TRUE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest,
       TransformIncrementIndexByOneNotFirstIdx) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[8,3,128], bf16[8,3,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[8,3,128], bf16[8,3,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[8,3,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[8,3,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[8,1,128] dynamic-slice(get-tuple-element.5, constant.2561, select.1348, constant.2561), dynamic_slice_sizes={8,1,128}
  mul = bf16[8,1,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[8,1,128] negate(mul)
  dynamic-update-slice.35 = bf16[8,3,128] dynamic-update-slice(get-tuple-element.395, ar.1, constant.2561, select.1348, constant.2561)
  ROOT tuple = (s32[], bf16[8,3,128], bf16[8,3,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[8,3,128] parameter(0)
  tuple = (s32[], bf16[8,3,128], bf16[8,3,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[8,3,128], bf16[8,3,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[8,3,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0).value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, TransformIncrementByTwo) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(2)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0).value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, NoTransformCantProveIndexDoesntWrap) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(4)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-1)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_FALSE(RunOptimizer(module.get(), /*last_run=*/true, 0).value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest,
       TransformNegativeIndexIterationToZero) {
  constexpr absl::string_view hlo_string = R"(
 HloModule module

 add {
   lhs = bf16[] parameter(0)
   rhs = bf16[] parameter(1)
   ROOT add = bf16[] add(lhs, rhs)
 }

 while_cond {
   param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
   gte = s32[] get-tuple-element(param), index=0
   constant.1 = s32[] constant(0)
   ROOT cmp = pred[] compare(gte, constant.1), direction=LT
 }

 while_body {
   param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
   get-tuple-element.394 = s32[] get-tuple-element(param), index=0
   get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
   get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
   constant.2557 = s32[] constant(1)
   add.230 = s32[] add(get-tuple-element.394, constant.2557)
   constant.2559 = s32[] constant(3)
   subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
   constant.2560 = s32[] constant(-1)
   add.231 = s32[] add(subtract.139, constant.2560)
   constant.2561 = s32[] constant(0)
   compare.747 = pred[] compare(add.231, constant.2561), direction=LT
   constant.2562 = s32[] constant(2)
   add.232 = s32[] add(subtract.139, constant.2562)
   select.1348 = s32[] select(compare.747, add.232, add.231)
   dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5,
   select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
   mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
   ar.1 = bf16[1,8,128] negate(mul)
   dynamic-update-slice.35 = bf16[3,8,128]
   dynamic-update-slice(get-tuple-element.395, ar.1, select.1348,
   constant.2561, constant.2561) ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128])
   tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
 }

 ENTRY entry {
   c0 = s32[] constant(-3)
   p0 = bf16[3,8,128] parameter(0)
   tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
   while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond,
   body=while_body ROOT gte1 = bf16[3,8,128] get-tuple-element(while),
   index=1
 }
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0).value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, EscapedInputNoTransform) {
  constexpr absl::string_view hlo_string = R"(
 HloModule module

 add {
   lhs = bf16[] parameter(0)
   rhs = bf16[] parameter(1)
   ROOT add = bf16[] add(lhs, rhs)
 }

 while_cond {
   param = (s32[], bf16[3,8,128], bf16[1,8,128]) parameter(0)
   gte = s32[] get-tuple-element(param), index=0
   constant.1 = s32[] constant(0)
   ROOT cmp = pred[] compare(gte, constant.1), direction=LT
 }

 while_body {
   param = (s32[], bf16[3,8,128], bf16[1,8,128]) parameter(0)
   get-tuple-element.394 = s32[] get-tuple-element(param), index=0
   get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
   constant.2557 = s32[] constant(1)
   add.230 = s32[] add(get-tuple-element.394, constant.2557)
   constant.2559 = s32[] constant(3)
   subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
   constant.2560 = s32[] constant(-1)
   add.231 = s32[] add(subtract.139, constant.2560)
   constant.2561 = s32[] constant(0)
   compare.747 = pred[] compare(add.231, constant.2561), direction=LT
   constant.2562 = s32[] constant(2)
   add.232 = s32[] add(subtract.139, constant.2562)
   select.1348 = s32[] select(compare.747, add.232, add.231)
   dynamic-slice.911 = bf16[1,8,128] dynamic-slice(get-tuple-element.395,
   constant.2561, constant.2561, constant.2561),
   dynamic_slice_sizes={1,8,128} dynamic-slice.99 = bf16[1,8,128]
   dynamic-slice(get-tuple-element.395, select.1348, constant.2561,
   constant.2561), dynamic_slice_sizes={1,8,128} mul = bf16[1,8,128]
   multiply(dynamic-slice.99, dynamic-slice.99) ar.1 = bf16[1,8,128]
   negate(mul)
   dynamic-update-slice.35 = bf16[3,8,128]
   dynamic-update-slice(get-tuple-element.395, ar.1, select.1348,
   constant.2561, constant.2561) ROOT tuple = (s32[], bf16[3,8,128],
   bf16[1,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-slice.911)
 }

 ENTRY entry {
   c0 = s32[] constant(-3)
   p0 = bf16[3,8,128] parameter(0)
   cc = bf16[] constant(0)
   c1 = bf16[1,8,128] broadcast(cc), dimensions={}
   tuple = (s32[], bf16[3,8,128], bf16[1,8,128]) tuple(c0, p0, c1)
   while = (s32[], bf16[3,8,128], bf16[1,8,128]) while(tuple),
   condition=while_cond, body=while_body ROOT gte1 = bf16[3,8,128]
   get-tuple-element(while), index=1
 }
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_FALSE(RunOptimizer(module.get(), /*last_run=*/true, 0).value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, TransformWithAg) {
  constexpr absl::string_view hlo_string = R"(
 HloModule module

 add {
   lhs = bf16[] parameter(0)
   rhs = bf16[] parameter(1)
   ROOT add = bf16[] add(lhs, rhs)
 }

 while_cond {
   param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
   gte = s32[] get-tuple-element(param), index=0
   constant.1 = s32[] constant(0)
   ROOT cmp = pred[] compare(gte, constant.1), direction=LT
 }

 while_body {
   param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
   get-tuple-element.394 = s32[] get-tuple-element(param), index=0
   get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
   get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
   constant.2557 = s32[] constant(1)
   add.230 = s32[] add(get-tuple-element.394, constant.2557)
   constant.2559 = s32[] constant(3)
   subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
   constant.2560 = s32[] constant(-1)
   add.231 = s32[] add(subtract.139, constant.2560)
   constant.2561 = s32[] constant(0)
   compare.747 = pred[] compare(add.231, constant.2561), direction=LT
   constant.2562 = s32[] constant(2)
   add.232 = s32[] add(subtract.139, constant.2562)
   select.1348 = s32[] select(compare.747, add.232, add.231)
   dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5,
   select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
   mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
   rs.1 = bf16[1,8,128] negate(mul)
   ag.1 = bf16[1,8,128] negate(rs.1)
   dynamic-update-slice.35 =
   bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ag.1,
   select.1348, constant.2561, constant.2561) ROOT tuple = (s32[],
   bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
 }

 ENTRY entry {
   c0 = s32[] constant(-3)
   p0 = bf16[3,8,128] parameter(0)
   cc = bf16[] constant(0)
   tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
   while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond,
   body=while_body
   ROOT gte1 = bf16[3,8,128] get-tuple-element(while),
   index=1
 }
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0).value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, TransformWithAgWithFormatting) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,9,128], bf16[3,9,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,9,128], bf16[3,9,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,9,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,9,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,9,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,9,128}
  mul = bf16[1,9,128] multiply(dynamic-slice.99, dynamic-slice.99)
  cpd = bf16[] constant(0)
  %pd = bf16[1,16,128] pad(mul, cpd), padding=0_0x0_7x0_0
  rs.1 = bf16[1,16,128] negate(pd)
  ag.1 = bf16[1,16,128] negate(rs.1)
  slc = bf16[1,9,128] slice(ag.1), slice={[0:1], [0:9], [0:128]}
  dynamic-update-slice.35 = bf16[3,9,128] dynamic-update-slice(get-tuple-element.395, slc, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,9,128], bf16[3,9,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,9,128] parameter(0)
  cc = bf16[] constant(0)
  tuple = (s32[], bf16[3,9,128], bf16[3,9,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,9,128], bf16[3,9,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,9,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0).value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, TransformWithAgInsertCustomCall) {
  constexpr absl::string_view hlo_string = R"(
 HloModule module

 add {
   lhs = bf16[] parameter(0)
   rhs = bf16[] parameter(1)
   ROOT add = bf16[] add(lhs, rhs)
 }

 while_cond {
   param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
   gte = s32[] get-tuple-element(param), index=0
   constant.1 = s32[] constant(0)
   ROOT cmp = pred[] compare(gte, constant.1), direction=LT
 }

 while_body {
   param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
   get-tuple-element.394 = s32[] get-tuple-element(param), index=0
   get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
   get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
   constant.2557 = s32[] constant(1)
   constant.2561 = s32[] constant(0)
   add.230 = s32[] add(get-tuple-element.394, constant.2557)
   dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5,
   get-tuple-element.394, constant.2561, constant.2561),
   dynamic_slice_sizes={1,8,128} mul = bf16[1,8,128]
   multiply(dynamic-slice.99, dynamic-slice.99) rs.1 = bf16[1,8,128]
   negate(mul)
   ag.1 = bf16[1,8,128] negate(rs.1)
   dynamic-update-slice.35 = bf16[3,8,128]
   dynamic-update-slice(get-tuple-element.395, ag.1, get-tuple-element.394,
   constant.2561, constant.2561) ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128])
   tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
 }

 ENTRY entry {
   c0 = s32[] constant(-8)
   p0 = bf16[3,8,128] parameter(0)
   cc = bf16[] constant(0)
   tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
   while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond,
   body=while_body ROOT gte1 = bf16[3,8,128] get-tuple-element(while),
   index=1
 }
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0).value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 200).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest,
       TransformIncrementIndexByOneBackwardsPlusForward) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] concatenate(a, a, a, a), dimensions={1}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] negate(mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();

  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0, /*should_process=*/
                   HloPredicateIsOp<HloOpcode::kConcatenate>,
                   collective_pipeliner_utils::PipeliningDirection::kBackward)
          .value());
  EXPECT_FALSE(RunOptimizer(module2.get(), /*last_run=*/true, 0).value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, MultiUsesElementwise) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,8,128] negate(mul)
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.1)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();

  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*should_process=*/HloPredicateIsOp<HloOpcode::kNegate>,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   /*pipeline_use_tree=*/true)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, ElementWiseUser) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  mul2 = bf16[1,8,128] multiply(ar.1, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul2, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();

  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*should_process=*/HloPredicateIsOp<HloOpcode::kNegate>,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   /*pipeline_use_tree=*/true)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest,
       TransformIncrementIndexByOneNotFirstIdxSink) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  %c = bf16[] constant(5.0)
  %b = bf16[1,8,128] broadcast(c), dimensions={}
  %a = bf16[1,8,128] add(ar.1, b)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, a, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();

  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true, 0,
                  /*should_process=*/HloPredicateIsOp<HloOpcode::kNegate>,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink,
                  /*pipeline_use_tree=*/true)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, TransformIncrementByTwoFormat) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  c = bf16[] constant(5.0)
  b = bf16[1,8,128] broadcast(c), dimensions={}
  a = bf16[1,8,128] add(ar.1, b)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, a, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();

  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true, 0,
                  /*should_process=*/HloPredicateIsOp<HloOpcode::kNegate>,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink,
                  /*pipeline_use_tree=*/true)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, MultiUsesElementwiseMerge) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,8,128] sqrt(mul)
  ar.2 = bf16[1,8,128] negate(mul)
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.2)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();

  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*should_process=*/
                   HloPredicateIsOp<HloOpcode::kNegate, HloOpcode::kSqrt>,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   /*pipeline_use_tree=*/true)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, BroadcastAsFormattingOp) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, reduce, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();

  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true, 0,
                  /*should_process=*/HloPredicateIsOp<HloOpcode::kNegate>,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink,
                  /*pipeline_use_tree=*/true)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest,
       ForwardSinkDependentPipelineableCollectives) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  ar.2 = bf16[1,8,128] negate(reduce)
  c1 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c1)
  mul1 = bf16[1,8,128] multiply(ar.2, bc)
  mul3 = bf16[1,8,128] multiply(mul1, ar.2)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul3, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();

  EXPECT_TRUE(
      RunOptimizer(
          module.get(), /*last_run=*/true, 0,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kNegate>,
          collective_pipeliner_utils::PipeliningDirection::kForwardSink,
          /*pipeline_use_tree=*/true,
          /*acceptable_formatting=*/HloPredicateIsNotOp<HloOpcode::kNegate>)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

TEST_F(CollectivePipelinerExecutionTest, MergeTwoCollectivesEachWithTwoDUS) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,8,128] get-tuple-element(param), index=2
  get-tuple-element.397 = bf16[3,8,128] get-tuple-element(param), index=3
  get-tuple-element.398 = bf16[3,8,128] get-tuple-element(param), index=4
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=5
  get-tuple-element.36 = bf16[3,8,128] get-tuple-element(param), index=6
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)

  // ar.1 is used by dynamic-update-slice.35 and dynamic-update-slice.36
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] negate(mul)
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, reduce, select.1348, constant.2561, constant.2561)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.1)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.36 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.396, mul4, select.1348, constant.2561, constant.2561)

  // ar.1 is used by dynamic-update-slice.37 and dynamic-update-slice.38
  // dynamic-update-slice.37 actually uses both ar.1 and ar.2
  dynamic-slice.100 = bf16[1,8,128] dynamic-slice(get-tuple-element.36, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul.1 = bf16[1,8,128] multiply(dynamic-slice.100, dynamic-slice.99)
  ar.2 = bf16[1,8,128] exponential(mul.1)
  divide = bf16[1,8,128] divide(ar.1, ar.2)
  dynamic-update-slice.37 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.397, divide, select.1348, constant.2561, constant.2561)
  mul.2 = bf16[1,8,128] multiply(ar.2, ar.2)
  abs = bf16[1,8,128] abs(mul.2)
  dynamic-update-slice.38 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.398, abs, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-update-slice.36, dynamic-update-slice.37, dynamic-update-slice.38, get-tuple-element.35, get-tuple-element.36)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,8,128] parameter(1)
  p2 = bf16[3,8,128] parameter(2)
  p3 = bf16[3,8,128] parameter(3)
  p4 = bf16[3,8,128] parameter(4)
  p5 = bf16[3,8,128] parameter(5)

  tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p1, p2, p3, p4, p5)
  ROOT while = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string).value();
  auto module2 = ParseAndReturnUnverifiedModule(hlo_string).value();

  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true, 0,
                  /*should_process=*/
                  HloPredicateIsOp<HloOpcode::kNegate, HloOpcode::kExp>,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink,
                  /*pipeline_use_tree=*/true)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  XLA_VLOG_LINES(1, module2->ToString());
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(module), std::move(module2),
                                      ErrorSpec{0.1, 0.1}));
}

}  // namespace
}  // namespace xla
