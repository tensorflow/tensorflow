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

#include "xla/hlo/transforms/simplifiers/reduce_window_rewriter.h"

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ReduceWindowRewriterTest : public HloHardwareIndependentTestBase {
 public:
  void CheckReduceWindowRewrite(absl::string_view hlo,
                                std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, ReduceWindowRewriter{128}, expected);
  }
};

TEST_F(ReduceWindowRewriterTest, EliminateR1) {
  const char* hlo = R"(
%binary_add {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

ENTRY %EliminateR1 (input: f32[10]) -> f32[10] {
  %input = f32[10]{0} parameter(0)
  %constant = f32[] constant(0)
  ROOT %reduce-window = f32[10]{0} reduce-window(f32[10]{0} %input, f32[] %constant), window={size=5 pad=2_2}, to_apply=%binary_add
}
)";

  CheckReduceWindowRewrite(hlo, R"(
// CHECK: [[reduce_window_1_0:%[^ ]+]] = f32[10,1]{0,1} reduce-window([[reshape_1:%[^ ]+]], [[constant_2:%[^ ]+]]), window={size=5x1 pad=2_2x0_0}, to_apply=[[binary_add_3:%[^ ]+]]
// CHECK-NEXT: ROOT [[reshape_1_4:%[^ ]+]] = f32[10]{0} reshape([[reduce_window_1_0]])
)");
}

TEST_F(ReduceWindowRewriterTest, EliminateR1Variadic) {
  const char* hlo = R"(
HloModule reduce-window

add_float {
  lhs.0 = f32[] parameter(0)
  lhs.1 = f32[] parameter(1)
  rhs.0 = f32[] parameter(2)
  rhs.1 = f32[] parameter(3)
  sum.0 = f32[] add(lhs.0, rhs.0)
  sum.1 = f32[] add(lhs.1, rhs.1)
  ROOT root = (f32[], f32[]) tuple(sum.0, sum.1)
}

ENTRY entry (arg: f32[10]) -> (f32[10], f32[10]) {
  arg = f32[10]{0} parameter(0)
  constant = f32[] constant(0)
  ROOT reduce-window = (f32[10]{0}, f32[10]{0}) reduce-window(f32[10]{0} %arg, f32[10]{0} %arg, f32[] %constant, f32[] %constant), window={size=5 pad=2_2}, to_apply=%add_float
})";

  CheckReduceWindowRewrite(hlo, R"(
// CHECK: ENTRY %entry (arg: f32[10]) -> (f32[10], f32[10]) {
// CHECK-NEXT:  [[arg_0:%[^ ]+]] = f32[10]{0} parameter(0)
// CHECK-NEXT:  [[reshape_1:%[^ ]+]] = f32[10,1]{0,1} reshape([[arg_0]])
// CHECK-NEXT:  [[reshape_1_2:%[^ ]+]] = f32[10,1]{0,1} reshape([[arg_0]])
// CHECK-NEXT:  [[constant_3:%[^ ]+]] = f32[] constant(0)
// CHECK-NEXT:  [[reduce_window_1_4:%[^ ]+]] = (f32[10,1]{0,1}, f32[10,1]{0,1}) reduce-window([[reshape_1]], [[reshape_1_2]], [[constant_3]], [[constant_3]]), window={size=5x1 pad=2_2x0_0}, to_apply=[[add_float_5:%[^ ]+]]
// CHECK-NEXT:  [[get_tuple_element_6:%[^ ]+]] = f32[10,1]{0,1} get-tuple-element([[reduce_window_1_4]]), index=0
// CHECK-NEXT:  [[reshape_2_7:%[^ ]+]] = f32[10]{0} reshape([[get_tuple_element_6]])
// CHECK-NEXT:  [[get_tuple_element_1_8:%[^ ]+]] = f32[10,1]{0,1} get-tuple-element([[reduce_window_1_4]]), index=1
// CHECK-NEXT:  [[reshape_3_9:%[^ ]+]] = f32[10]{0} reshape([[get_tuple_element_1_8]])
// CHECK-NEXT:  ROOT [[tuple_10:%[^ ]+]] = (f32[10]{0}, f32[10]{0}) tuple([[reshape_2_7]], [[reshape_3_9]])
// CHECK-NEXT:}
)");
}

TEST_F(ReduceWindowRewriterTest, OptimizeR1InclusiveScan) {
  const char* hlo = R"(
HloModule reduce-window

add_float {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}

ENTRY entry (arg: f32[46592]) -> f32[46592] {
  arg = f32[46592]{0} parameter(0)
  constant = f32[] constant(0)
  ROOT reduce-window = f32[46592]{0} reduce-window(f32[46592]{0} %arg, f32[] %constant), window={size=46592 pad=46591_0}, to_apply=%add_float
})";

  CheckReduceWindowRewrite(hlo, R"(
// CHECK: ENTRY %entry (arg: f32[46592]) -> f32[46592] {
// CHECK-NEXT:  [[arg_0:%[^ ]+]] = f32[46592]{0} parameter(0)
// CHECK-NEXT:  [[reshape_1:%[^ ]+]] = f32[364,128]{0,1} reshape([[arg_0]])
// CHECK-NEXT:  [[constant_2:%[^ ]+]] = f32[] constant(0)
// CHECK-NEXT:  [[reduce_window_1_3:%[^ ]+]] = f32[364,128]{0,1} reduce-window([[reshape_1]], [[constant_2]]), window={size=1x128 pad=0_0x127_0}, to_apply=[[add_float_4:%[^ ]+]]
// CHECK-NEXT:  [[slice_5:%[^ ]+]] = f32[364,1]{0,1} slice([[reduce_window_1_3]]), slice={[0:364], [127:128]}
// CHECK-NEXT:  [[reshape_1_6:%[^ ]+]] = f32[364]{0} reshape([[slice_5]])
// CHECK-NEXT:  [[reduce_window_2_7:%[^ ]+]] = f32[365]{0} reduce-window([[reshape_1_6]], [[constant_2]]), window={size=364 pad=364_0}, to_apply=[[add_float_4]]
// CHECK-NEXT:  [[slice_1_8:%[^ ]+]] = f32[364]{0} slice([[reduce_window_2_7]]), slice={[0:364]}
// CHECK-NEXT:  [[broadcast_9:%[^ ]+]] = f32[364,128]{0,1} broadcast([[slice_1_8]]), dimensions={0}
// CHECK-NEXT:  [[map_10:%[^ ]+]] = f32[364,128]{0,1} map([[reduce_window_1_3]], [[broadcast_9]]), dimensions={0,1}, to_apply=[[add_float_4]]
// CHECK-NEXT:  ROOT [[reshape_2_11:%[^ ]+]] = f32[46592]{0} reshape([[map_10]])
// CHECK-NEXT:}
)");
}

TEST_F(ReduceWindowRewriterTest, OptimizeR1InclusiveScanVariadic) {
  const std::string hlo_string = R"(
HloModule reduce-window

MaxMin {
  l.max = f32[] parameter(0)
  l.min = f32[] parameter(1)
  r.max = f32[] parameter(2)
  r.min = f32[] parameter(3)
  max = f32[] maximum(l.max, r.max)
  min = f32[] minimum(l.min, r.min)
  ROOT root = (f32[], f32[]) tuple(max, min)
}

ENTRY entry (arg_0: f32[46592], arg_1: f32[46592]) -> (f32[46592], f32[46592]) {
  arg.0 = f32[46592]{0} parameter(0)
  arg.1 = f32[46592]{0} parameter(1)
  init_ninf = f32[] constant(-inf)
  init_inf = f32[] constant(inf)
  ROOT reduce-window = (f32[46592]{0}, f32[46592]{0}) reduce-window(f32[46592]{0} %arg.0, f32[46592]{0} %arg.1, f32[] %init_ninf, f32[] %init_inf), window={size=46592 pad=46591_0}, to_apply=%MaxMin
}
)";

  CheckReduceWindowRewrite(hlo_string, R"(
// CHECK: ENTRY %entry (arg.0: f32[46592], arg.1: f32[46592]) -> (f32[46592], f32[46592]) {
// CHECK-NEXT:   [[arg_0_0:%[^ ]+]] = f32[46592]{0} parameter(0)
// CHECK-NEXT:   [[reshape_1:%[^ ]+]] = f32[364,128]{0,1} reshape([[arg_0_0]])
// CHECK-NEXT:   [[arg_1_2:%[^ ]+]] = f32[46592]{0} parameter(1)
// CHECK-NEXT:   [[reshape_1_3:%[^ ]+]] = f32[364,128]{0,1} reshape([[arg_1_2]])
// CHECK-NEXT:   [[init_ninf_4:%[^ ]+]] = f32[] constant(-inf)
// CHECK-NEXT:   [[init_inf_5:%[^ ]+]] = f32[] constant(inf)
// CHECK-NEXT:   [[reduce_window_1_6:%[^ ]+]] = (f32[364,128]{0,1}, f32[364,128]{0,1}) reduce-window([[reshape_1]], [[reshape_1_3]], [[init_ninf_4]], [[init_inf_5]]), window={size=1x128 pad=0_0x127_0}, to_apply=[[MaxMin_7:%[^ ]+]]
// CHECK-NEXT:   [[get_tuple_element_4_8:%[^ ]+]] = f32[364,128]{0,1} get-tuple-element([[reduce_window_1_6]]), index=0
// CHECK-NEXT:   [[get_tuple_element_5_9:%[^ ]+]] = f32[364,128]{0,1} get-tuple-element([[reduce_window_1_6]]), index=1
// CHECK-NEXT:   [[get_tuple_element_10:%[^ ]+]] = f32[364,128]{0,1} get-tuple-element([[reduce_window_1_6]]), index=0
// CHECK-NEXT:   [[slice_11:%[^ ]+]] = f32[364,1]{0,1} slice([[get_tuple_element_10]]), slice={[0:364], [127:128]}
// CHECK-NEXT:   [[reshape_2_12:%[^ ]+]] = f32[364]{0} reshape([[slice_11]])
// CHECK-NEXT:   [[get_tuple_element_1_13:%[^ ]+]] = f32[364,128]{0,1} get-tuple-element([[reduce_window_1_6]]), index=1
// CHECK-NEXT:   [[slice_1_14:%[^ ]+]] = f32[364,1]{0,1} slice([[get_tuple_element_1_13]]), slice={[0:364], [127:128]}
// CHECK-NEXT:   [[reshape_3_15:%[^ ]+]] = f32[364]{0} reshape([[slice_1_14]])
// CHECK-NEXT:   [[reduce_window_2_16:%[^ ]+]] = (f32[365]{0}, f32[365]{0}) reduce-window([[reshape_2_12]], [[reshape_3_15]], [[init_ninf_4]], [[init_inf_5]]), window={size=364 pad=364_0}, to_apply=[[MaxMin_7]]
// CHECK-NEXT:   [[get_tuple_element_2_17:%[^ ]+]] = f32[365]{0} get-tuple-element([[reduce_window_2_16]]), index=0
// CHECK-NEXT:   [[slice_2_18:%[^ ]+]] = f32[364]{0} slice([[get_tuple_element_2_17]]), slice={[0:364]}
// CHECK-NEXT:   [[broadcast_19:%[^ ]+]] = f32[364,128]{0,1} broadcast([[slice_2_18]]), dimensions={0}
// CHECK-NEXT:   [[get_tuple_element_3_20:%[^ ]+]] = f32[365]{0} get-tuple-element([[reduce_window_2_16]]), index=1
// CHECK-NEXT:   [[slice_3_21:%[^ ]+]] = f32[364]{0} slice([[get_tuple_element_3_20]]), slice={[0:364]}
// CHECK-NEXT:   [[broadcast_1_22:%[^ ]+]] = f32[364,128]{0,1} broadcast([[slice_3_21]]), dimensions={0}
// CHECK-NEXT:   [[map_23:%[^ ]+]] = f32[364,128]{0,1} map([[get_tuple_element_4_8]], [[get_tuple_element_5_9]], [[broadcast_19]], [[broadcast_1_22]]), dimensions={0,1}, to_apply=[[MaxMin_7]].clone
// CHECK-NEXT:   [[reshape_4_24:%[^ ]+]] = f32[46592]{0} reshape([[map_23]])
// CHECK-NEXT:   [[map_1_25:%[^ ]+]] = f32[364,128]{0,1} map([[get_tuple_element_4_8]], [[get_tuple_element_5_9]], [[broadcast_19]], [[broadcast_1_22]]), dimensions={0,1}, to_apply=[[MaxMin_7]].clone.1
// CHECK-NEXT:   [[reshape_5_26:%[^ ]+]] = f32[46592]{0} reshape([[map_1_25]])
// CHECK-NEXT:   ROOT [[tuple_27:%[^ ]+]] = (f32[46592]{0}, f32[46592]{0}) tuple([[reshape_4_24]], [[reshape_5_26]])
// CHECK-NEXT: }
  )");
}

}  // namespace
}  // namespace xla
