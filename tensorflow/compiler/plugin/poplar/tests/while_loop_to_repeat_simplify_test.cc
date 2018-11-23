/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <stdlib.h>

namespace xla {
namespace poplarplugin {
namespace {

using WhileLoopToRepeatSimplifyTest = HloTestBase;

/* Note that you should always run WhileLoopConstantSinking and
   WhileLoopConditionSimplify before the WhileLoopToRepeatSimplify pass. */

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] less-than(p_cond.0, const)
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  EXPECT_EQ(annotations.while_loop_num_iterations[root], 10);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_Ge) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  const = s32[] constant(1)
  add = s32[] subtract(p_body.0, const)
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(0)
  ROOT result = pred[] greater-than-or-equal-to(p_cond.0, const)
}

ENTRY entry {
  const_0 = s32[] constant(999)
  const_1 = s32[] constant(0)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  EXPECT_EQ(annotations.while_loop_num_iterations[root], 1000);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_Gt) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  const = s32[] constant(1)
  add = s32[] subtract(p_body.0, const)
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(0)
  ROOT result = pred[] greater-than(p_cond.0, const)
}

ENTRY entry {
  const_0 = s32[] constant(1000)
  const_1 = s32[] constant(0)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  EXPECT_EQ(annotations.while_loop_num_iterations[root], 1000);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_Le) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(999)
  ROOT result = pred[] less-than-or-equal-to(p_cond.0, const)
}

ENTRY entry {
  const_0 = s32[] constant(100)
  const_1 = s32[] constant(999)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  EXPECT_EQ(annotations.while_loop_num_iterations[root], 900);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_NonConstInit) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(999)
  ROOT result = pred[] less-than-or-equal-to(p_cond.0, const)
}

ENTRY entry {
  const_0 = s32[] parameter(0)
  const_1 = s32[] constant(999)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  EXPECT_EQ(annotations.while_loop_num_iterations[root], 0);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_NonConstDelta) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  add = s32[] add(p_body.0, p_body.1)
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(999)
  ROOT result = pred[] less-than-or-equal-to(p_cond.0, const)
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] parameter(0)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  EXPECT_EQ(annotations.while_loop_num_iterations[root], 0);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalF32IncrementBy2) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[],f32[]) parameter(0)
  p_body.0 = f32[] get-tuple-element((f32[],f32[]) p_body), index=0
  const = f32[] constant(2)
  add = f32[] add(p_body.0, const)
  p_body.1 = f32[] get-tuple-element((f32[],f32[]) p_body), index=1
  ROOT root = (f32[],f32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (f32[],f32[]) parameter(0)
  p_cond.0 = f32[] get-tuple-element((f32[],f32[]) p_cond), index=0
  const = f32[] constant(10)
  ROOT result = pred[] less-than(p_cond.0, const)
}

ENTRY entry {
  const_0 = f32[] constant(0)
  const_1 = f32[] constant(10)
  while_init = (f32[],f32[]) tuple(const_0, const_1)
  ROOT while = (f32[],f32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // Get the trip count
  EXPECT_EQ(annotations.while_loop_num_iterations
                [module.get()->entry_computation()->root_instruction()],
            5);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalHoistTheConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] less-than(p_cond.0, const)
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // Get the trip count
  EXPECT_EQ(annotations.while_loop_num_iterations
                [module.get()->entry_computation()->root_instruction()],
            10);
  // Check the constant got hoisted out.
  HloInstruction* while_inst = module->entry_computation()->root_instruction();
  EXPECT_EQ(while_inst->opcode(), HloOpcode::kWhile);
  const HloInstruction* while_init = while_inst->operand(0);
  const HloInstruction* counter = while_init->operand(0);
  VLOG(0) << while_init->parent()->ToString();
  EXPECT_EQ(counter->opcode(), HloOpcode::kConstant);
  int32 loop_counter =
      LiteralScalarInt32toInt32(counter->literal()).ValueOrDie();
  EXPECT_EQ(loop_counter, 10);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalDontHoistTheConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  add2 = s32[] add(p_body.1, add)
  ROOT root = (s32[],s32[]) tuple(add, add2)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] less-than(p_cond.0, const)
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // Get the trip count
  EXPECT_EQ(annotations.while_loop_num_iterations
                [module.get()->entry_computation()->root_instruction()],
            10);
  // Check the constant got hoisted out.
  HloInstruction* while_inst = module->entry_computation()->root_instruction();
  EXPECT_EQ(while_inst->opcode(), HloOpcode::kWhile);
  const HloInstruction* while_init = while_inst->operand(0);
  const HloInstruction* counter = while_init->operand(0);
  VLOG(0) << while_init->parent()->ToString();
  EXPECT_EQ(counter->opcode(), HloOpcode::kConstant);
  int32 loop_start = LiteralScalarInt32toInt32(counter->literal()).ValueOrDie();
  EXPECT_EQ(loop_start, 0);
}

using WhileLoopToRepeatSimplifyTestChangedEnv = HloTestBase;

TEST_F(WhileLoopToRepeatSimplifyTestChangedEnv,
       SingleConditionalF32ChangeBruteForceMaxTripCount) {
  putenv("TF_POPLAR_MAX_WHILE_LOOP_TRIP_COUNT=9");
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[],f32[]) parameter(0)
  p_body.0 = f32[] get-tuple-element((f32[],f32[]) p_body), index=0
  const = f32[] constant(1)
  add = f32[] add(p_body.0, const)
  p_body.1 = f32[] get-tuple-element((f32[],f32[]) p_body), index=1
  ROOT root = (f32[],f32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (f32[],f32[]) parameter(0)
  p_cond.0 = f32[] get-tuple-element((f32[],f32[]) p_cond), index=0
  const = f32[] constant(10)
  ROOT result = pred[] less-than(p_cond.0, const)
}

ENTRY entry {
  const_0 = f32[] constant(0)
  const_1 = f32[] constant(10)
  while_init = (f32[],f32[]) tuple(const_0, const_1)
  ROOT while = (f32[],f32[]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  CompilerAnnotations annotations(module.get());
  WhileLoopToRepeatSimplify wltrs(annotations);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));

  // We didn't get the trip count due to the bound being too low
  EXPECT_EQ(annotations.while_loop_num_iterations.count(
                module.get()->entry_computation()->root_instruction()),
            0);

  // Re run with increased trip count
  putenv("TF_POPLAR_MAX_WHILE_LOOP_TRIP_COUNT=10");
  TF_ASSERT_OK_AND_ASSIGN(changed, wltrs.Run(module.get()));

  EXPECT_EQ(annotations.while_loop_num_iterations
                [module.get()->entry_computation()->root_instruction()],
            10);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
