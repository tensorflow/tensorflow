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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

class CpuWhileTest : public HloTestBase {};

// Verifies fix for b/233647273.
TEST_F(CpuWhileTest, While) {
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

  // Compile and execute the computation.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

  // Check the output correctness.
  LiteralTestUtil::ExpectR0Equal(3, result);
}

// Add a small while loop that calls sort to verify that the small call emitter
// can deal with thread local buffers.
TEST_F(CpuWhileTest, WhileSort) {
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

  // Compile and execute the computation.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal input = LiteralUtil::CreateR1<float>({3, 1, 4, 2});

  ASSERT_OK_AND_ASSIGN(const Literal result,
                       Execute(std::move(module), {&input}));

  // Check the output correctness.
  LiteralTestUtil::ExpectR1Equal(absl::MakeConstSpan({4.0f, 3.0f, 2.0f, 1.0f}),
                                 result);
}

// Simply runs a HLO module that was seen to fail before a fix due to an
// attempted deref of a nullptr in eigen runtime dot call. This test is not
// intended to verify the correctness of the result, just that it does not
// result in an error.
TEST_F(CpuWhileTest, WhileDotDoesNotError) {
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

  // No need to check the values of the Literal, just that it returns a valid
  // result.
  Literal input_real = LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}});
  Literal input_imag = LiteralUtil::CreateR3<float>({{{9, 10}, {11, 12}}});

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  EXPECT_OK(Execute(std::move(module), {&input_real, &input_imag}));
}

TEST_F(CpuWhileTest, FusedDynamicUpdateSliceInSmallWhileLoop) {
  const std::string hlo_text = R"(
    HloModule FusedDynamicUpdateSliceInSmallWhileLoop

    fused_dus {
      param_data = f32[8] parameter(0)
      param_update = f32[2] parameter(1)
      param_idx = s32[] parameter(2)
      ROOT dus = f32[8] dynamic-update-slice(param_data, param_update, param_idx)
    }

    body {
      loop_state = (s32[], f32[8]) parameter(0)
      idx = s32[] get-tuple-element(loop_state), index=0
      data = f32[8] get-tuple-element(loop_state), index=1
      one = s32[] constant(1)
      next_idx = s32[] add(idx, one)
      update = f32[2] constant({10.0, 20.0})
      fused = f32[8] fusion(data, update, idx), kind=kLoop, calls=fused_dus
      ROOT new_state = (s32[], f32[8]) tuple(next_idx, fused)
    }

    cond {
      loop_state = (s32[], f32[8]) parameter(0)
      idx = s32[] get-tuple-element(loop_state), index=0
      limit = s32[] constant(3)
      ROOT cmp = pred[] compare(idx, limit), direction=LT
    }

    ENTRY main {
      init_idx = s32[] constant(0)
      zero = f32[] constant(0.0)
      init_data = f32[8] broadcast(zero), dimensions={}
      init_state = (s32[], f32[8]) tuple(init_idx, init_data)
      while_loop = (s32[], f32[8]) while(init_state), condition=cond, body=body
      ROOT result = f32[8] get-tuple-element(while_loop), index=1
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

  LiteralTestUtil::ExpectR1Equal(
      absl::MakeConstSpan({10.0f, 10.0f, 10.0f, 20.0f, 0.0f, 0.0f, 0.0f, 0.0f}),
      result);
}

// The two loops share one kernel, which must use each call's own buffers.
TEST_F(CpuWhileTest, IdenticalSmallWhileLoopsSharedKernelCorrect) {
  constexpr absl::string_view hlo_text = R"(
    HloModule IdenticalSmallWhileLoops

    body {
      state = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      x = f32[4] get-tuple-element(state), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      c = f32[] constant(2)
      cb = f32[4] broadcast(c), dimensions={}
      sum = f32[4] add(x, cb)
      next_x = f32[4] multiply(sum, cb)
      ROOT next_state = (s32[], f32[4]) tuple(next_i, next_x)
    }

    cond {
      state = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(5)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      zero = s32[] constant(0)
      init0 = (s32[], f32[4]) tuple(zero, p0)
      init1 = (s32[], f32[4]) tuple(zero, p1)
      while0 = (s32[], f32[4]) while(init0), condition=cond, body=body
      while1 = (s32[], f32[4]) while(init1), condition=cond, body=body
      r0 = f32[4] get-tuple-element(while0), index=1
      r1 = f32[4] get-tuple-element(while1), index=1
      ROOT r = (f32[4], f32[4]) tuple(r0, r1)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  Literal p0 = LiteralUtil::CreateR1<float>({0, 1, 2, 3});
  Literal p1 = LiteralUtil::CreateR1<float>({10, 20, 30, 40});
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&p0, &p1}));

  // Five iterations of x = (x + 2) * 2 give 32 * x + 124.
  std::vector<Literal> results = result.DecomposeTuple();
  ASSERT_EQ(results.size(), 2);
  LiteralTestUtil::ExpectR1Equal(
      absl::MakeConstSpan({124.0f, 156.0f, 188.0f, 220.0f}), results[0]);
  LiteralTestUtil::ExpectR1Equal(
      absl::MakeConstSpan({444.0f, 764.0f, 1084.0f, 1404.0f}), results[1]);
}

// Small loops become one kernel running the thunks of the loop. These bodies
// exercise each kind of thunk inside that kernel.
class CpuSmallWhileLoopTest : public HloInterpreterReferenceMixin<HloTestBase> {
 protected:
  // Checks that the loop compiled to one kernel, since a fallback to a call
  // thunk would still produce correct results.
  void ExpectOneKernel(absl::string_view hlo_text) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    CpuCompiler compiler;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> optimized,
        compiler.RunHloPasses(std::move(module), /*stream_exec=*/nullptr,
                              /*options=*/{}));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        compiler.RunBackend(std::move(optimized), /*stream_exec=*/nullptr,
                            /*options=*/{}));
    int64_t num_kernels = 0;
    for (const auto& thunk : absl::down_cast<CpuExecutable*>(executable.get())
                                 ->thunks()
                                 .thunk_sequence()) {
      EXPECT_NE(thunk->kind(), Thunk::Kind::kWhile);
      EXPECT_NE(thunk->kind(), Thunk::Kind::kCall);
      num_kernels += thunk->kind() == Thunk::Kind::kKernel;
    }
    EXPECT_EQ(num_kernels, 1);
  }

  void ExpectBackendNoCallThunk(absl::string_view hlo_text,
                                int64_t expected_kernels) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    CpuCompiler compiler;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        compiler.RunBackend(std::move(module), /*stream_exec=*/nullptr,
                            /*options=*/{}));
    const ThunkSequence& thunks =
        absl::down_cast<CpuExecutable*>(executable.get())
            ->thunks()
            .thunk_sequence();
    int64_t num_kernels = 0;
    for (const auto& thunk : thunks) {
      num_kernels += thunk->kind() == Thunk::Kind::kKernel;
    }
    EXPECT_EQ(num_kernels, expected_kernels);
    thunks.ForEach([](const Thunk& thunk) {
      EXPECT_NE(thunk.kind(), Thunk::Kind::kCall);
    });
  }
};

TEST_F(CpuSmallWhileLoopTest, NestedWhileAndConditional) {
  constexpr absl::string_view hlo_text = R"(
    HloModule NestedWhileAndConditional

    inner_body {
      s = (s32[], f32[4]) parameter(0)
      j = s32[] get-tuple-element(s), index=0
      x = f32[4] get-tuple-element(s), index=1
      one = s32[] constant(1)
      c = f32[] constant(0.5)
      cb = f32[4] broadcast(c), dimensions={}
      ROOT r = (s32[], f32[4]) tuple(add(j, one), add(x, cb))
    }

    inner_cond {
      s = (s32[], f32[4]) parameter(0)
      j = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(j, s32[] constant(3)), direction=LT
    }

    double {
      x = f32[4] parameter(0)
      ROOT r = f32[4] add(x, x)
    }

    negate {
      x = f32[4] parameter(0)
      ROOT r = f32[4] negate(x)
    }

    body {
      s = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      x = f32[4] get-tuple-element(s), index=1
      zero = s32[] constant(0)
      inner = (s32[], f32[4]) while((s32[], f32[4]) tuple(zero, x)),
          condition=inner_cond, body=inner_body
      y = f32[4] get-tuple-element(inner), index=1
      two = s32[] constant(2)
      branch = s32[] remainder(i, two)
      z = f32[4] conditional(branch, y, y), branch_computations={double, negate}
      one = s32[] constant(1)
      ROOT r = (s32[], f32[4]) tuple(add(i, one), z)
    }

    cond {
      s = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(4)), direction=LT
    }

    ENTRY main {
      p = f32[4] parameter(0)
      zero = s32[] constant(0)
      w = (s32[], f32[4]) while((s32[], f32[4]) tuple(zero, p)),
          condition=cond, body=body
      ROOT r = f32[4] get-tuple-element(w), index=1
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5}));
}

TEST_F(CpuSmallWhileLoopTest, DotsAndScatter) {
  constexpr absl::string_view hlo_text = R"(
    HloModule DotsAndScatter

    add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }

    body {
      s = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      m = bf16[16,16] get-tuple-element(s), index=1
      v = bf16[16] get-tuple-element(s), index=2
      a = f32[16,16] get-tuple-element(s), index=3
      acc = f32[16] get-tuple-element(s), index=4
      mv = bf16[16] dot(m, v), lhs_contracting_dims={1},
          rhs_contracting_dims={0}
      av = f32[16] dot(a, acc), lhs_contracting_dims={1},
          rhs_contracting_dims={0}
      sum = f32[16] add(av, acc)
      c = s32[] constant(5)
      idx = s32[3,1] broadcast(remainder(i, c)), dimensions={}
      offsets = s32[3,1] constant({{0}, {5}, {10}})
      indices = s32[3,1] add(idx, offsets)
      updates = f32[3] constant({1, 2, 3})
      scattered = f32[16] scatter(sum, indices, updates), to_apply=add,
          update_window_dims={}, inserted_window_dims={0},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1
      one = s32[] constant(1)
      mn = bf16[16,16] negate(m)
      ROOT r = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16])
          tuple(add(i, one), mn, mv, a, scattered)
    }

    cond {
      s = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      m = bf16[16,16] parameter(0)
      v = bf16[16] parameter(1)
      a = f32[16,16] parameter(2)
      acc = f32[16] parameter(3)
      zero = s32[] constant(0)
      init = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16])
          tuple(zero, m, v, a, acc)
      w = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16]) while(init),
          condition=cond, body=body
      mv = bf16[16] get-tuple-element(w), index=2
      out = f32[16] get-tuple-element(w), index=4
      ROOT r = (bf16[16], f32[16]) tuple(mv, out)
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

// An output fusion of a dot and an addend, which no HLO pass is left to
// create from the matrix vector product below.
TEST_F(CpuSmallWhileLoopTest, DotOutputFusion) {
  constexpr absl::string_view hlo_text = R"(
    HloModule DotOutputFusion

    fused_dot_add {
      a = f32[16,16] parameter(0)
      x = f32[16] parameter(1)
      addend = f32[16] parameter(2)
      d = f32[16] dot(a, x), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT r = f32[16] add(d, addend)
    }

    body {
      s = (s32[], f32[16,16], f32[16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[16,16] get-tuple-element(s), index=1
      x = f32[16] get-tuple-element(s), index=2
      y = f32[16] fusion(a, x, x), kind=kOutput, calls=fused_dot_add
      one = s32[] constant(1)
      ROOT r = (s32[], f32[16,16], f32[16]) tuple(add(i, one), a, y)
    }

    cond {
      s = (s32[], f32[16,16], f32[16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      a = f32[16,16] parameter(0)
      x = f32[16] parameter(1)
      zero = s32[] constant(0)
      w = (s32[], f32[16,16], f32[16]) while((s32[], f32[16,16], f32[16])
          tuple(zero, a, x)), condition=cond, body=body
      ROOT r = f32[16] get-tuple-element(w), index=2
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// The body fusion is large enough to be split into several workgroups, which
// the loop kernel launches in sequence on the calling thread.
TEST_F(CpuSmallWhileLoopTest, MultipleWorkgroups) {
  constexpr absl::string_view hlo_text = R"(
    HloModule MultipleWorkgroups

    body {
      s = (s32[], f32[256,1024]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      x = f32[256,1024] get-tuple-element(s), index=1
      fi = f32[] convert(i)
      b = f32[256,1024] broadcast(fi), dimensions={}
      iota = f32[256,1024] iota(), iota_dimension=0
      y = f32[256,1024] add(x, multiply(iota, b))
      one = s32[] constant(1)
      ROOT r = (s32[], f32[256,1024]) tuple(add(i, one), y)
    }

    cond {
      s = (s32[], f32[256,1024]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(4)), direction=LT
    }

    ENTRY main {
      p = f32[256,1024] parameter(0)
      zero = s32[] constant(0)
      w = (s32[], f32[256,1024]) while((s32[], f32[256,1024]) tuple(zero, p)),
          condition=cond, body=body
      ROOT r = f32[256,1024] get-tuple-element(w), index=1
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(CpuSmallWhileLoopTest, InPlaceDynamicUpdateSlice) {
  constexpr absl::string_view hlo_text = R"(
    HloModule InPlaceDynamicUpdateSlice

    body {
      s = (s32[], s32[], f32[32]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      acc = s32[] get-tuple-element(s), index=1
      arr = f32[32] get-tuple-element(s), index=2
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      next_acc = s32[] add(acc, next_i)
      slice = f32[1] dynamic-slice(arr, i), dynamic_slice_sizes={1}
      delta = f32[1] reshape(f32[] convert(next_acc))
      upd = f32[1] add(slice, delta)
      next_arr = f32[32] dynamic-update-slice(arr, upd, i)
      ROOT r = (s32[], s32[], f32[32]) tuple(next_i, next_acc, next_arr)
    }

    cond {
      s = (s32[], s32[], f32[32]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(16)), direction=LT
    }

    ENTRY main {
      p = f32[32] parameter(0)
      z = s32[] constant(0)
      w = (s32[], s32[], f32[32]) while((s32[], s32[], f32[32]) tuple(z, z, p)),
          condition=cond, body=body
      ROOT r = f32[32] get-tuple-element(w), index=2
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5}));
}

TEST_F(CpuSmallWhileLoopTest, ReusedTempBuffers) {
  constexpr absl::string_view hlo_text = R"(
    HloModule ReusedTempBuffers

    inner_body {
      s = (s32[], u32[2], u32[4]) parameter(0)
      j = s32[] get-tuple-element(s), index=0
      v = u32[2] get-tuple-element(s), index=1
      k = u32[4] get-tuple-element(s), index=2
      k0 = u32[2] slice(k), slice={[0:2]}
      k1 = u32[2] slice(k), slice={[2:4]}
      v1 = u32[2] xor(add(v, k0), k1)
      k_next = u32[4] concatenate(k1, k0), dimensions={0}
      ROOT r = (s32[], u32[2], u32[4]) tuple(add(j, s32[] constant(1)), v1, k_next)
    }

    inner_cond {
      s = (s32[], u32[2], u32[4]) parameter(0)
      j = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(j, s32[] constant(4)), direction=LT
    }

    body {
      s = (s32[], u32[2]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      v = u32[2] get-tuple-element(s), index=1
      k = u32[4] constant({13, 15, 26, 6})
      inner = (s32[], u32[2], u32[4]) while(
          (s32[], u32[2], u32[4]) tuple(s32[] constant(0), v, k)),
          condition=inner_cond, body=inner_body
      v_next = u32[2] get-tuple-element(inner), index=1
      ROOT r = (s32[], u32[2]) tuple(add(i, s32[] constant(1)), v_next)
    }

    cond {
      s = (s32[], u32[2]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(5)), direction=LT
    }

    ENTRY main {
      p = u32[2] parameter(0)
      w = (s32[], u32[2]) while((s32[], u32[2]) tuple(s32[] constant(0), p)),
          condition=cond, body=body
      ROOT r = u32[2] get-tuple-element(w), index=1
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0}));
}

TEST_F(CpuSmallWhileLoopTest, NestedFusionKernelNameCollision) {
  // Fusion and concatenate emitted before enclosing small call.
  constexpr absl::string_view kBackward = R"(
    HloModule NestedFusionKernelNameCollision

    f_add {
      x = s32[4] parameter(0)
      y = s32[4] parameter(1)
      ROOT s = s32[4] add(x, y)
    }

    f_mul {
      x = s32[4] parameter(0)
      ROOT y = s32[4] multiply(x, s32[4] constant({2, 2, 2, 2}))
    }

    small_fn {
      v = s32[4] parameter(0)
      cat = s32[4] concatenate(s32[2] constant({1, 2}),
                               s32[2] constant({3, 4})), dimensions={0}
      call_kernel = s32[4] fusion(v, cat), kind=kLoop, calls=f_add
      ROOT call_kernel_body = s32[4] fusion(call_kernel), kind=kLoop, calls=f_mul
    }

    ENTRY main {
      p = s32[4] parameter(0)
      ROOT call = s32[4] call(p), to_apply=small_fn,
          frontend_attributes={xla_cpu_small_call="true"}
    }
  )";
  ExpectBackendNoCallThunk(kBackward, /*expected_kernels=*/1);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kBackward, ErrorSpec{0}));

  // Unfused concatenate before small call `c`, `c` before top level fusion
  // `c_kernel`, and nested small call `n` before sibling fusion `n_kernel`.
  constexpr absl::string_view kForward = R"(
    HloModule ForwardKernelNameCollision

    f_add {
      x = s32[4] parameter(0)
      ROOT y = s32[4] add(x, s32[4] constant({1, 2, 3, 4}))
    }

    f_mul_outer {
      x = s32[4] parameter(0)
      ROOT y = s32[4] multiply(x, s32[4] constant({3, 3, 3, 3}))
    }

    f_mul_entry {
      x = s32[4] parameter(0)
      y = s32[4] parameter(1)
      ROOT z = s32[4] add(x, y)
    }

    inner_fn {
      v = s32[4] parameter(0)
      ROOT a = s32[4] fusion(v), kind=kLoop, calls=f_add
    }

    outer_fn {
      v = s32[4] parameter(0)
      n = s32[4] call(v), to_apply=inner_fn,
          frontend_attributes={xla_cpu_small_call="true"}
      ROOT n_kernel = s32[4] fusion(n), kind=kLoop, calls=f_mul_outer
    }

    ENTRY main {
      p = s32[4] parameter(0)
      pre = s32[4] concatenate(s32[2] constant({1, 2}),
                               s32[2] constant({3, 4})), dimensions={0}
      c = s32[4] call(p), to_apply=outer_fn,
          frontend_attributes={xla_cpu_small_call="true"}
      ROOT c_kernel = s32[4] fusion(c, pre), kind=kLoop, calls=f_mul_entry
    }
  )";
  ExpectBackendNoCallThunk(kForward, /*expected_kernels=*/3);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kForward, ErrorSpec{0}));

  // Sort comparator computation named like small call `c`'s kernel.
  constexpr absl::string_view kComparator = R"(
    HloModule ComparatorKernelNameCollision

    c_kernel {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT lt = pred[] compare(a, b), direction=LT
    }

    f_add {
      x = s32[4] parameter(0)
      ROOT y = s32[4] add(x, s32[4] constant({1, 2, 3, 4}))
    }

    small_fn {
      v = s32[4] parameter(0)
      ROOT a = s32[4] fusion(v), kind=kLoop, calls=f_add
    }

    ENTRY main {
      p = s32[4] parameter(0)
      s = s32[4] sort(p), dimensions={0}, to_apply=c_kernel
      ROOT c = s32[4] call(s), to_apply=small_fn,
          frontend_attributes={xla_cpu_small_call="true"}
    }
  )";
  ExpectBackendNoCallThunk(kComparator, /*expected_kernels=*/1);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kComparator, ErrorSpec{0}));
}

TEST_F(CpuSmallWhileLoopTest, BatchedMatrixVectorDot) {
  constexpr absl::string_view hlo_text = R"(
    HloModule BatchedMatrixVectorDot

    body {
      s = (s32[], f32[2,16,16], f32[2,16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[2,16,16] get-tuple-element(s), index=1
      x = f32[2,16] get-tuple-element(s), index=2
      y = f32[2,16] dot(a, x), lhs_batch_dims={0}, rhs_batch_dims={0},
          lhs_contracting_dims={2}, rhs_contracting_dims={1}
      next_i = s32[] add(i, s32[] constant(1))
      ROOT r = (s32[], f32[2,16,16], f32[2,16]) tuple(next_i, a, y)
    }

    cond {
      s = (s32[], f32[2,16,16], f32[2,16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      a = f32[2,16,16] parameter(0)
      x = f32[2,16] parameter(1)
      w = (s32[], f32[2,16,16], f32[2,16]) while(
          (s32[], f32[2,16,16], f32[2,16]) tuple(s32[] constant(0), a, x)),
          condition=cond, body=body
      ROOT out = f32[2,16] get-tuple-element(w), index=2
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4}));
}

TEST_F(CpuSmallWhileLoopTest, ZeroSizedArrayInLoopState) {
  constexpr absl::string_view hlo_text = R"(
    HloModule ZeroSizedArrayInLoopState

    body {
      s = (s32[], f32[0], f32[4]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      e = f32[0] get-tuple-element(s), index=1
      x = f32[4] get-tuple-element(s), index=2
      next_i = s32[] add(i, s32[] constant(1))
      next_e = f32[0] negate(e)
      next_x = f32[4] add(x, x)
      ROOT r = (s32[], f32[0], f32[4]) tuple(next_i, next_e, next_x)
    }

    cond {
      s = (s32[], f32[0], f32[4]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      e = f32[0] parameter(0)
      p = f32[4] parameter(1)
      w = (s32[], f32[0], f32[4]) while((s32[], f32[0], f32[4]) tuple(
          s32[] constant(0), e, p)), condition=cond, body=body
      e_out = f32[0] get-tuple-element(w), index=1
      x_out = f32[4] get-tuple-element(w), index=2
      ROOT r = (f32[0], f32[4]) tuple(e_out, x_out)
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0}));
}

TEST_F(CpuSmallWhileLoopTest, ReadOnlyWhileConditionEmitsNoCallThunk) {
  constexpr absl::string_view kHlo = R"(
    HloModule ReadOnlyWhileCondition

    f_double {
      x = s32[4] parameter(0)
      ROOT y = s32[4] add(x, x)
    }

    cond {
      s = (pred[], s32[4]) parameter(0)
      ROOT c = pred[] get-tuple-element(s), index=0
    }

    body {
      s = (pred[], s32[4]) parameter(0)
      c = pred[] get-tuple-element(s), index=0
      x = s32[4] get-tuple-element(s), index=1
      nx = s32[4] fusion(x), kind=kLoop, calls=f_double
      ROOT r = (pred[], s32[4]) tuple(c, nx)
    }

    small_fn {
      c = pred[] parameter(0)
      x = s32[4] parameter(1)
      x_copy = s32[4] copy(x)
      init = (pred[], s32[4]) tuple(c, x_copy)
      w = (pred[], s32[4]) while(init), condition=cond, body=body
      ROOT out = s32[4] get-tuple-element(w), index=1
    }

    ENTRY main {
      p = s32[4] parameter(0)
      f = pred[] constant(false)
      ROOT call = s32[4] call(f, p), to_apply=small_fn,
          frontend_attributes={xla_cpu_small_call="true"}
    }
  )";
  ExpectBackendNoCallThunk(kHlo, /*expected_kernels=*/1);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{0}));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
