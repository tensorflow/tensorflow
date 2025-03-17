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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
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

// Simply runs a HLO module that was seen to fail before a fix due to an
// attempted deref of a nullptr in eigen runtime dot call. This test is not
// intended to verify the correctness of the result, just that it does not
// result in an error.
TEST_F(CpuCodegenTest, WhileDotDoesNotError) {
  constexpr absl::string_view hlo_text = R"(
    HloModule Complex_1.44, entry_computation_layout={(f32[1,2,2]{2,1,0}, f32[1,2,2]{2,1,0})->f32[]}

    %add_F32.39 (lhs.40: f32[], rhs.41: f32[]) -> f32[] {
      %lhs.40 = f32[] parameter(0)
      %rhs.41 = f32[] parameter(1)
      ROOT %add.42 = f32[] add(%lhs.40, %rhs.41)
    }

    %unblocked_body.14 (parameter.15: (s32[], c64[1,2,2], c64[1,2,2], pred[1,1,1])) -> (s32[], c64[1,2,2], c64[1,2,2], pred[1,1,1]) {
      %parameter.15 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) parameter(0)
      %get-tuple-element.16 = s32[] get-tuple-element(%parameter.15), index=0
      %constant.20 = s32[] constant(1)
      %add.21 = s32[] add(%get-tuple-element.16, %constant.20)
      %get-tuple-element.17 = c64[1,2,2]{2,1,0} get-tuple-element(%parameter.15), index=1
      %iota.23 = s32[1,2,2]{2,1,0} iota(), iota_dimension=1
      %iota.22 = s32[1,2,2]{2,1,0} iota(), iota_dimension=2
      %compare.24 = pred[1,2,2]{2,1,0} compare(%iota.23, %iota.22), direction=GE
      %broadcast.25 = s32[1,2,2]{2,1,0} broadcast(%get-tuple-element.16), dimensions={}
      %compare.26 = pred[1,2,2]{2,1,0} compare(%iota.22, %broadcast.25), direction=EQ
      %and.27 = pred[1,2,2]{2,1,0} and(%compare.24, %compare.26)
      %get-tuple-element.18 = c64[1,2,2]{2,1,0} get-tuple-element(%parameter.15), index=2
      %real.30 = f32[1,2,2]{2,1,0} real(%get-tuple-element.18)
      %imag.31 = f32[1,2,2]{2,1,0} imag(%get-tuple-element.18)
      %negate.32 = f32[1,2,2]{2,1,0} negate(%imag.31)
      %complex.33 = c64[1,2,2]{2,1,0} complex(%real.30, %negate.32)
      %dot.34 = c64[1,2,2]{2,1,0} dot(%get-tuple-element.18, %complex.33), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={2}, operand_precision={highest,highest}, frontend_attributes={grad_x="false",grad_y="false"}
      %transpose.35 = c64[1,2,2]{2,1,0} transpose(%dot.34), dimensions={0,1,2}
      %subtract.36 = c64[1,2,2]{2,1,0} subtract(%get-tuple-element.17, %transpose.35)
      %constant.37 = s32[] constant(0)
      %dynamic-slice.38 = c64[1,1,1]{2,1,0} dynamic-slice(%subtract.36, %constant.37, %get-tuple-element.16, %get-tuple-element.16), dynamic_slice_sizes={1,1,1}
      %real.39 = f32[1,1,1]{2,1,0} real(%dynamic-slice.38)
      %sqrt.40 = f32[1,1,1]{2,1,0} sqrt(%real.39)
      %constant.41 = f32[] constant(0)
      %broadcast.42 = f32[1,1,1]{2,1,0} broadcast(%constant.41), dimensions={}
      %complex.43 = c64[1,1,1]{2,1,0} complex(%sqrt.40, %broadcast.42)
      %reshape.46 = c64[1]{0} reshape(%complex.43)
      %broadcast.47 = c64[1,2,2]{2,1,0} broadcast(%reshape.46), dimensions={0}
      %divide.48 = c64[1,2,2]{2,1,0} divide(%subtract.36, %broadcast.47)
      %constant.28 = c64[] constant((0, 0))
      %broadcast.29 = c64[1,2,2]{2,1,0} broadcast(%constant.28), dimensions={}
      %select.49 = c64[1,2,2]{2,1,0} select(%and.27, %divide.48, %broadcast.29)
      %add.50 = c64[1,2,2]{2,1,0} add(%select.49, %get-tuple-element.18)
      %get-tuple-element.19 = pred[1,1,1]{2,1,0} get-tuple-element(%parameter.15), index=3
      %compare.44 = pred[1,1,1]{2,1,0} compare(%sqrt.40, %sqrt.40), direction=NE
      %or.45 = pred[1,1,1]{2,1,0} or(%get-tuple-element.19, %compare.44)
      ROOT %tuple.51 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) tuple(%add.21, %get-tuple-element.17, %add.50, %or.45)
    }

    %unblocked_condition.52 (parameter.53: (s32[], c64[1,2,2], c64[1,2,2], pred[1,1,1])) -> pred[] {
      %parameter.53 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) parameter(0)
      %get-tuple-element.55 = c64[1,2,2]{2,1,0} get-tuple-element(%parameter.53), index=1
      %get-tuple-element.56 = c64[1,2,2]{2,1,0} get-tuple-element(%parameter.53), index=2
      %get-tuple-element.57 = pred[1,1,1]{2,1,0} get-tuple-element(%parameter.53), index=3
      %get-tuple-element.54 = s32[] get-tuple-element(%parameter.53), index=0
      %constant.58 = s32[] constant(2)
      ROOT %compare.59 = pred[] compare(%get-tuple-element.54, %constant.58), direction=LT
    }

    %xla.cholesky_c64_1_2_2__lower.76 (a.1: c64[1,2,2]) -> c64[1,2,2] {
      %constant.12 = s32[] constant(0)
      %a.1 = c64[1,2,2]{2,1,0} parameter(0)
      %slice.6 = c64[1,2,2]{2,1,0} slice(%a.1), slice={[0:1], [0:2], [0:2]}
      %slice.7 = c64[1,2,2]{2,1,0} slice(%slice.6), slice={[0:1], [0:2], [0:2]}
      %constant.8 = c64[] constant((0, 0))
      %broadcast.9 = c64[1,2,2]{2,1,0} broadcast(%constant.8), dimensions={}
      %constant.10 = pred[] constant(false)
      %broadcast.11 = pred[1,1,1]{2,1,0} broadcast(%constant.10), dimensions={}
      %tuple.13 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) tuple(%constant.12, %slice.7, %broadcast.9, %broadcast.11)
      %while.60 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) while(%tuple.13), condition=%unblocked_condition.52, body=%unblocked_body.14
      %get-tuple-element.61 = s32[] get-tuple-element(%while.60), index=0
      %get-tuple-element.62 = c64[1,2,2]{2,1,0} get-tuple-element(%while.60), index=1
      %constant.4 = pred[] constant(false)
      %broadcast.5 = pred[1,1,1]{2,1,0} broadcast(%constant.4), dimensions={}
      %get-tuple-element.64 = pred[1,1,1]{2,1,0} get-tuple-element(%while.60), index=3
      %or.65 = pred[1,1,1]{2,1,0} or(%broadcast.5, %get-tuple-element.64)
      %broadcast.70 = pred[1,1,1]{2,1,0} broadcast(%or.65), dimensions={0,1,2}
      %reshape.71 = pred[1]{0} reshape(%broadcast.70)
      %broadcast.72 = pred[1,2,2]{2,1,0} broadcast(%reshape.71), dimensions={0}
      %constant.73 = c64[] constant((nan, 0))
      %broadcast.74 = c64[1,2,2]{2,1,0} broadcast(%constant.73), dimensions={}
      %constant.2 = c64[] constant((0, 0))
      %broadcast.3 = c64[1,2,2]{2,1,0} broadcast(%constant.2), dimensions={}
      %get-tuple-element.63 = c64[1,2,2]{2,1,0} get-tuple-element(%while.60), index=2
      %constant.66 = s32[] constant(0)
      %constant.67 = s32[] constant(0)
      %constant.68 = s32[] constant(0)
      %dynamic-update-slice.69 = c64[1,2,2]{2,1,0} dynamic-update-slice(%broadcast.3, %get-tuple-element.63, %constant.66, %constant.67, %constant.68)
      ROOT %select.75 = c64[1,2,2]{2,1,0} select(%broadcast.72, %broadcast.74, %dynamic-update-slice.69)
    }

    ENTRY %Complex_1.44 (input_real.1: f32[1,2,2], input_imag.2: f32[1,2,2]) -> f32[] {
      %input_real.1 = f32[1,2,2]{2,1,0} parameter(0)
      %input_imag.2 = f32[1,2,2]{2,1,0} parameter(1)
      %complex.3 = c64[1,2,2]{2,1,0} complex(%input_real.1, %input_imag.2)
      %real.4 = f32[1,2,2]{2,1,0} real(%complex.3)
      %imag.5 = f32[1,2,2]{2,1,0} imag(%complex.3)
      %negate.6 = f32[1,2,2]{2,1,0} negate(%imag.5)
      %complex.7 = c64[1,2,2]{2,1,0} complex(%real.4, %negate.6)
      %transpose.8 = c64[1,2,2]{1,2,0} transpose(%complex.7), dimensions={0,2,1}
      %dot.9 = c64[1,2,2]{2,1,0} dot(%complex.3, %transpose.8), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}, operand_precision={highest,highest}, frontend_attributes={grad_x="false",grad_y="false"}
      %transpose.10 = c64[1,2,2]{2,1,0} transpose(%dot.9), dimensions={0,1,2}
      %iota.13 = s32[2]{0} iota(), iota_dimension=0
      %constant.14 = s32[] constant(0)
      %broadcast.15 = s32[2]{0} broadcast(%constant.14), dimensions={}
      %add.16 = s32[2]{0} add(%iota.13, %broadcast.15)
      %broadcast.18 = s32[2,2]{1,0} broadcast(%add.16), dimensions={0}
      %iota.12 = s32[2]{0} iota(), iota_dimension=0
      %broadcast.17 = s32[2,2]{1,0} broadcast(%iota.12), dimensions={1}
      %compare.19 = pred[2,2]{1,0} compare(%broadcast.18, %broadcast.17), direction=GE
      %broadcast.20 = pred[1,2,2]{2,1,0} broadcast(%compare.19), dimensions={1,2}
      %call = c64[1,2,2]{2,1,0} call(%transpose.10), to_apply=%xla.cholesky_c64_1_2_2__lower.76
      %constant.21 = c64[] constant((0, 0))
      %broadcast.22 = c64[1,2,2]{2,1,0} broadcast(%constant.21), dimensions={}
      %select.23 = c64[1,2,2]{2,1,0} select(%broadcast.20, %call, %broadcast.22)
      %real.24 = f32[1,2,2]{2,1,0} real(%select.23)
      %imag.25 = f32[1,2,2]{2,1,0} imag(%select.23)
      %negate.26 = f32[1,2,2]{2,1,0} negate(%imag.25)
      %complex.27 = c64[1,2,2]{2,1,0} complex(%real.24, %negate.26)
      %transpose.28 = c64[1,2,2]{1,2,0} transpose(%complex.27), dimensions={0,2,1}
      %dot.29 = c64[1,2,2]{2,1,0} dot(%select.23, %transpose.28), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}, operand_precision={highest,highest}, frontend_attributes={grad_x="false",grad_y="false"}
      %transpose.30 = c64[1,2,2]{2,1,0} transpose(%dot.29), dimensions={0,1,2}
      %subtract.31 = c64[1,2,2]{2,1,0} subtract(%transpose.10, %transpose.30)
      %real.32 = f32[1,2,2]{2,1,0} real(%subtract.31)
      %imag.33 = f32[1,2,2]{2,1,0} imag(%subtract.31)
      %negate.34 = f32[1,2,2]{2,1,0} negate(%imag.33)
      %complex.35 = c64[1,2,2]{2,1,0} complex(%real.32, %negate.34)
      %multiply.36 = c64[1,2,2]{2,1,0} multiply(%subtract.31, %complex.35)
      %abs.37 = f32[1,2,2]{2,1,0} abs(%multiply.36)
      %constant.38 = f32[] constant(0)
      ROOT %reduce.43 = f32[] reduce(%abs.37, %constant.38), dimensions={0,1,2}, to_apply=%add_F32.39
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  Literal input_real =
      LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
  Literal input_imag =
      LiteralUtil::CreateR3<float>({{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}});

  // No need to check the values of the Literal, just that it returns a valid
  // result.
  TF_EXPECT_OK(Execute(module->Clone(), {&input_real, &input_imag}));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
