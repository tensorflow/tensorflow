/* Copyright 2022 The OpenXLA Authors.

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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

// Verifies fix for b/233647273.
TEST_F(CpuCodegenTest, While) {
  const std::string hlo_text = R"(
HloModule module

f1 {
  f1.p0 = s32[] parameter(0)
  ROOT f1.sum = s32[] add(f1.p0, f1.p0)
}

f2 {
  f2.p0 = s32[] parameter(0)
  f2.p1 = s32[] parameter(1)
  ROOT f2.sum = s32[] add(f2.p0, f2.p1)
}

body {
  body.p0 = s32[] parameter(0)
  sum2 = s32[] fusion(body.p0), kind=kLoop, calls=f1
  ROOT sum3 = s32[] fusion(sum2, body.p0), kind=kLoop, calls=f2
}

cond {
  cond.p0 = s32[] parameter(0)
  cond.c1 = s32[] constant(1)
  ROOT cond.root = pred[] compare(cond.p0, cond.c1), direction=EQ
}

ENTRY entry {
  entry.c1 = s32[] constant(1)
  ROOT entry.root = s32[] while(entry.c1), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  // Compile and execute the computation.
  auto result = ExecuteAndTransfer(module->Clone(), {});

  // Check the output correctness.
  LiteralTestUtil::ExpectR0Equal(3, result);
}

// Add a small while loop that calls sort to verify that the small call emitter
// can deal with thread local buffers.
TEST_F(CpuCodegenTest, WhileSort) {
  const std::string hlo_text = R"(
    HloModule sort_loop

    %sort_comparison (x: f32[], y: f32[]) -> pred[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %result = compare(x, y), direction=GT
    }

    %sort_body (input: (f32[4])) -> (f32[4]) {
      %input = f32[4] parameter(0)
      %sorted = sort(%input), dimensions={0}, is_stable=true, to_apply=%sort_comparison
      ROOT %result = tuple(%sorted)
    }

    %while_condition (loop_state: (s32[], f32[4])) -> pred[] {
      %loop_state = (s32[], f32[4]) parameter(0)
      %index = s32[] get-tuple-element(%loop_state), index=0
      %limit = s32[] constant(4)
      ROOT %result = compare(%index, %limit), direction=LT
    }

    %while_body (loop_state: (s32[], f32[4])) -> (s32[], f32[4]) {
      %loop_state = (s32[], f32[4]) parameter(0)
      %index = s32[] get-tuple-element(%loop_state), index=0
      %data = f32[4] get-tuple-element(%loop_state), index=1
      %increment = s32[] constant(1)
      %new_index = add(%index, %increment)
      %sorted_data = sort(%data), dimensions={0}, is_stable=true, to_apply=%sort_comparison
      ROOT %new_state = tuple(%new_index, %sorted_data)
    }

    ENTRY %main (input: f32[4]) -> f32[4] {
      %init_index = s32[] constant(0)
      %input = f32[4] parameter(0)
      %loop_state = (s32[], f32[4]) tuple(%init_index, %input)
      %while_loop = (s32[], f32[4]) while(%loop_state), condition=%while_condition, body=%while_body
      ROOT %final_sorted = f32[4] get-tuple-element(%while_loop), index=1
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal input = LiteralUtil::CreateR1<float>({3, 1, 4, 2});

  // Compile and execute the computation.
  auto result = ExecuteAndTransfer(module->Clone(), {&input});

  // Check the output correctness.
  LiteralTestUtil::ExpectR1Equal(absl::MakeConstSpan({4.0f, 3.0f, 2.0f, 1.0f}),
                                 result);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
