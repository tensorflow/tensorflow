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
#include "xla/hlo/transforms/expanders/reduce_decomposer.h"

#include <optional>

#include <gtest/gtest.h>
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/test.h"
#include "xla/test_helpers.h"

namespace xla {
namespace {

class ReduceDecomposerTest : public HloHardwareIndependentTestBase {};

TEST_F(ReduceDecomposerTest, ReducePerformsTransposition) {
  // Reshape is already a bitcast, nothing should be changed.
  const char* hlo = R"(
HloModule module

add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT out = add(a, b)
}

ENTRY c {
    p = f32[5,3,4]{2,1,0} parameter(0)
    z = f32[] constant(0)
    ROOT r = f32[5,4]{0,1} reduce(p, z), dimensions={1}, to_apply=add
}
)";

  RunAndFilecheckHloRewrite(
      hlo,
      ReduceDecomposer{/*custom_layout_allowed=*/[&](const HloInstruction*) {
        return true;
      }},
      std::nullopt);

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{},
                            R"(
// CHECK: [[reduce_0:%[^ ]+]] = f32[5,4]{1,0} reduce([[p_1:%[^ ]+]], [[z_2:%[^ ]+]]), dimensions={1}, to_apply=[[add_3:%[^ ]+]]
// CHECK-NEXT: ROOT [[copy_4:%[^ ]+]] = f32[5,4]{0,1} copy([[reduce_0]])
      )");
}

TEST_F(ReduceDecomposerTest, ReduceNaturalLayout) {
  // Reshape is already a bitcast, nothing should be changed.
  const char* hlo = R"(
HloModule module

add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT out = add(a, b)
}

ENTRY c {
    p = f32[5,3,4]{2,1,0} parameter(0)
    z = f32[] constant(0)
    ROOT r = reduce(p, z), dimensions={1}, to_apply=add
}
)";

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{}, std::nullopt);
}

TEST_F(ReduceDecomposerTest, VariadicReductionWithTranspose) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChangeVariadicDifferent

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  arg0 = f32[2,3,4,1024]{3,2,1,0}  parameter(0)
  idxs = u32[2,3,4,1024]{3,2,1,0}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[2,3,4]{0,1,2},
      u32[2,3,4]{0,1,2}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}
  )";

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{},
                            R"(
// CHECK:  [[_reduce_0:%[^ ]+]] = (f32[2,3,4]{2,1,0}, u32[2,3,4]{2,1,0}) reduce([[_arg0_1:%[^ ]+]], [[_idxs_2:%[^ ]+]], [[_constant0_3:%[^ ]+]], [[_constant1_4:%[^ ]+]]), dimensions={3}, to_apply=[[_argmax_5:%[^ ]+]]
// CHECK-NEXT:  [[_get_tuple_element_6:%[^ ]+]] = f32[2,3,4]{2,1,0} get-tuple-element([[_reduce_0]]), index=0
// CHECK-NEXT:  [[_copy_7:%[^ ]+]] = f32[2,3,4]{0,1,2} copy([[_get_tuple_element_6]])
// CHECK-NEXT:  [[_get_tuple_element_1_8:%[^ ]+]] = u32[2,3,4]{2,1,0} get-tuple-element([[_reduce_0]]), index=1
// CHECK-NEXT:  [[_copy_1_9:%[^ ]+]] = u32[2,3,4]{0,1,2} copy([[_get_tuple_element_1_8]])
// CHECK-NEXT:  ROOT [[_tuple_10:%[^ ]+]] = (f32[2,3,4]{0,1,2}, u32[2,3,4]{0,1,2}) tuple([[_copy_7]], [[_copy_1_9]])
                        )");
}

TEST_F(ReduceDecomposerTest, VariadicReductionDescendingLayout) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChangeVariadicDifferent

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  arg0 = f32[2,3,4,1024]{3,2,1,0}  parameter(0)
  idxs = u32[2,3,4,1024]{3,2,1,0}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[2,3,4]{2,1,0},
      u32[2,3,4]{2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}
  )";

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{}, std::nullopt);
}

TEST_F(ReduceDecomposerTest, VariadicReductionInputsDifferentLayout) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChangeVariadicDifferent

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  arg0 = f32[2,3,4,1024]{3,2,1,0}  parameter(0)
  idxs = u32[2,3,4,1024]{2,1,3,0}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[2,3,4]{2,1,0},
      u32[2,3,4]{2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}
  )";

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{}, R"(
// CHECK: [[arg0_0:%[^ ]+]] = f32[2,3,4,1024]{3,2,1,0} parameter(0)
// CHECK: [[idxs_1:%[^ ]+]] = u32[2,3,4,1024]{2,1,3,0} parameter(1)
// CHECK: [[copy_2:%[^ ]+]] = u32[2,3,4,1024]{3,2,1,0} copy([[idxs_1]])
// CHECK: [[constant0_3:%[^ ]+]] = f32[] constant(0)
// CHECK: [[constant1_4:%[^ ]+]] = u32[] constant(0)
// CHECK: ROOT [[reduce0_5:%[^ ]+]] = (f32[2,3,4]{2,1,0}, u32[2,3,4]{2,1,0}) reduce([[arg0_0]], [[copy_2]], [[constant0_3]], [[constant1_4]]), dimensions={3}, to_apply=[[argmax_6:%[^ ]+]]
)");
}

}  // namespace
}  // namespace xla
