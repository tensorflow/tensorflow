/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/transforms/embedded_while_loop_unroller.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

using EmbeddedWhileLoopUnrollerTest = HloHardwareIndependentTestBase;

constexpr absl::string_view kLoopComputations = R"(
body {
  state = (s32[], f32[3]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  buffer = f32[3] get-tuple-element(state), index=1
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  update = f32[1] constant({7})
  next_buffer = f32[3] dynamic-update-slice(buffer, update, i)
  ROOT next_state = (s32[], f32[3]) tuple(next_i, next_buffer)
}

condition {
  state = (s32[], f32[3]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  three = s32[] constant(3)
  ROOT continue = pred[] compare(i, three), direction=LT
}
)";

TEST_F(EmbeddedWhileLoopUnrollerTest, UnrollsLoopInSortComparator) {
  constexpr absl::string_view kModule = R"(
compare {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  zero = s32[] constant(0)
  zero_f = f32[] constant(0)
  init = f32[3] broadcast(zero_f), dimensions={}
  init_state = (s32[], f32[3]) tuple(zero, init)
  loop = (s32[], f32[3]) while(init_state), condition=condition, body=body
  result = f32[3] get-tuple-element(loop), index=1
  first = f32[1] slice(result), slice={[0:1]}
  first_scalar = f32[] reshape(first)
  c = f32[] convert(p0)
  ROOT lt = pred[] compare(c, first_scalar), direction=LT
}

ENTRY main {
  x = s32[7] parameter(0)
  ROOT sorted = s32[7] sort(x), dimensions={0}, to_apply=compare
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(absl::StrCat(
                           "HloModule test\n", kLoopComputations, kModule)));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       EmbeddedWhileLoopUnroller().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(hlo_query::ContainsInstrWithOpcode(
      module->GetComputationWithName("compare"), {HloOpcode::kWhile}));
}

TEST_F(EmbeddedWhileLoopUnrollerTest, LeavesEntryLoopAlone) {
  constexpr absl::string_view kModule = R"(
ENTRY main {
  zero = s32[] constant(0)
  init = f32[3] parameter(0)
  init_state = (s32[], f32[3]) tuple(zero, init)
  ROOT loop = (s32[], f32[3]) while(init_state), condition=condition, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(absl::StrCat(
                           "HloModule test\n", kLoopComputations, kModule)));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       EmbeddedWhileLoopUnroller().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_TRUE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                 {HloOpcode::kWhile}));
}

// The entry loop carries a loop-invariant scalar constant, which the
// unroller's module-wide preparation (constant sinking) would have rewritten.
TEST_F(EmbeddedWhileLoopUnrollerTest, LeavesEntryLoopAloneNextToEmbeddedLoop) {
  constexpr absl::string_view kModule = R"(
compare {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  zero = s32[] constant(0)
  zero_f = f32[] constant(0)
  init = f32[3] broadcast(zero_f), dimensions={}
  init_state = (s32[], f32[3]) tuple(zero, init)
  loop = (s32[], f32[3]) while(init_state), condition=condition, body=body
  result = f32[3] get-tuple-element(loop), index=1
  first = f32[1] slice(result), slice={[0:1]}
  first_scalar = f32[] reshape(first)
  c = f32[] convert(p0)
  ROOT lt = pred[] compare(c, first_scalar), direction=LT
}

entry_body {
  state = (s32[], f32[3], f32[]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  buffer = f32[3] get-tuple-element(state), index=1
  step = f32[] get-tuple-element(state), index=2
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  step_b = f32[3] broadcast(step), dimensions={}
  next_buffer = f32[3] add(buffer, step_b)
  ROOT next_state = (s32[], f32[3], f32[]) tuple(next_i, next_buffer, step)
}

entry_condition {
  state = (s32[], f32[3], f32[]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  three = s32[] constant(3)
  ROOT continue = pred[] compare(i, three), direction=LT
}

ENTRY main {
  x = s32[7] parameter(0)
  sorted = s32[7] sort(x), dimensions={0}, to_apply=compare
  zero = s32[] constant(0)
  init = f32[3] parameter(1)
  step = f32[] constant(1)
  init_state = (s32[], f32[3], f32[]) tuple(zero, init, step)
  loop = (s32[], f32[3], f32[]) while(init_state), condition=entry_condition, body=entry_body
  buffer = f32[3] get-tuple-element(loop), index=1
  ROOT out = (s32[7], f32[3]) tuple(sorted, buffer)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(absl::StrCat(
                           "HloModule test\n", kLoopComputations, kModule)));
  const std::string entry_before = module->entry_computation()->ToString();
  const std::string entry_body_before =
      module->GetComputationWithName("entry_body")->ToString();
  const std::string entry_condition_before =
      module->GetComputationWithName("entry_condition")->ToString();

  ASSERT_OK_AND_ASSIGN(bool changed,
                       EmbeddedWhileLoopUnroller().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(hlo_query::ContainsInstrWithOpcode(
      module->GetComputationWithName("compare"), {HloOpcode::kWhile}));
  EXPECT_EQ(module->entry_computation()->ToString(), entry_before);
  EXPECT_EQ(module->GetComputationWithName("entry_body")->ToString(),
            entry_body_before);
  EXPECT_EQ(module->GetComputationWithName("entry_condition")->ToString(),
            entry_condition_before);
}

}  // namespace
}  // namespace xla::cpu
