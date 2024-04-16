/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/reduction_layout_normalizer.h"

#include <optional>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {

namespace {

class ReductionLayoutNormalizerTest : public HloTestBase {
 public:
  void CheckReductionLayoutNormalizer(
      absl::string_view hlo, std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, gpu::ReductionLayoutNormalizer{}, expected);
  }
};

TEST_F(ReductionLayoutNormalizerTest, LayoutCanonicalizerTest) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChange

add {
  x0 = f32[] parameter(0)
  y0 = f32[] parameter(1)
  ROOT add0 = f32[] add(x0, y0)
}

ENTRY main {
  arg0 = f32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(0)
  constant0 = f32[] constant(0)
  ROOT reduce0 = f32[4,5,16,12,12]{4,3,2,1,0} reduce(arg0, constant0),
    dimensions={1,6,7}, to_apply=add
}

)";

  CheckReductionLayoutNormalizer(hlo,
                                 R"(
// CHECK:  [[bitcast_0:%[^ ]+]] = f32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} bitcast([[arg0_1:%[^ ]+]])
// CHECK:  [[reduce_2:%[^ ]+]] = f32[4,12,12,16,5]{2,1,3,4,0} reduce([[bitcast_0]], [[constant0_3:%[^ ]+]]), dimensions={0,1,2}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[bitcast_1_5:%[^ ]+]] = f32[4,5,16,12,12]{4,3,2,1,0} bitcast([[reduce_2]])
      )");
}

TEST_F(ReductionLayoutNormalizerTest, LayoutCanonicalizerTestVariadic) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChangeVariadic


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
  arg0 = f32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(0)
  idxs = u32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[4,5,16,12,12]{4,3,2,1,0},
      u32[4,5,16,12,12]{4,3,2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={1,6,7}, to_apply=argmax
}


)";

  CheckReductionLayoutNormalizer(hlo,
                                 R"(
// CHECK:  [[arg0_0:%[^ ]+]] = f32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} bitcast([[arg0_0]])
// CHECK:  [[idxs_2:%[^ ]+]] = u32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1} parameter(1)
// CHECK:  [[bitcast_1_3:%[^ ]+]] = u32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} bitcast([[idxs_2]])
// CHECK:  [[reduce_4:%[^ ]+]] = (f32[4,12,12,16,5]{2,1,3,4,0}, u32[4,12,12,16,5]{2,1,3,4,0}) reduce([[bitcast_1]], [[bitcast_1_3]], [[constant0_5:%[^ ]+]], [[constant1_6:%[^ ]+]]), dimensions={0,1,2}, to_apply=[[argmax_7:%[^ ]+]]
// CHECK:  [[get_tuple_element_8:%[^ ]+]] = f32[4,12,12,16,5]{2,1,3,4,0} get-tuple-element([[reduce_4]]), index=0
// CHECK:  [[bitcast_2_9:%[^ ]+]] = f32[4,5,16,12,12]{4,3,2,1,0} bitcast([[get_tuple_element_8]])
// CHECK:  [[get_tuple_element_1_10:%[^ ]+]] = u32[4,12,12,16,5]{2,1,3,4,0} get-tuple-element([[reduce_4]]), index=1
// CHECK:  [[bitcast_3_11:%[^ ]+]] = u32[4,5,16,12,12]{4,3,2,1,0} bitcast([[get_tuple_element_1_10]])
// CHECK:  ROOT [[tuple_12:%[^ ]+]] = (f32[4,5,16,12,12]{4,3,2,1,0}, u32[4,5,16,12,12]{4,3,2,1,0}) tuple([[bitcast_2_9]], [[bitcast_3_11]])
      )");
}

TEST_F(ReductionLayoutNormalizerTest,
       LayoutCanonicalizerTestVariadicDifferentLayouts) {
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
  arg0 = f32[2,3,4,7]{2,1,0,3}  parameter(0)
  idxs = u32[2,3,4,7]{3,2,1,0}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[2,3,4]{2,1,0},
      u32[2,3,4]{2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}


)";

  CheckReductionLayoutNormalizer(hlo,
                                 R"(
// CHECK:  [[arg0_0:%[^ ]+]] = f32[2,3,4,7]{2,1,0,3} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[7,2,3,4]{3,2,1,0} bitcast([[arg0_0]])
// CHECK:  [[idxs_2:%[^ ]+]] = u32[2,3,4,7]{3,2,1,0} parameter(1)
// CHECK:  [[copy_3:%[^ ]+]] = u32[2,3,4,7]{2,1,0,3} copy([[idxs_2]])
// CHECK:  [[bitcast_1_4:%[^ ]+]] = u32[7,2,3,4]{3,2,1,0} bitcast([[copy_3]])
// CHECK:  ROOT [[reduce0_5:%[^ ]+]] = (f32[2,3,4]{2,1,0}, u32[2,3,4]{2,1,0}) reduce([[bitcast_1]], [[bitcast_1_4]], [[constant0_6:%[^ ]+]], [[constant1_7:%[^ ]+]]), dimensions={0}, to_apply=[[argmax_8:%[^ ]+]]
      )");
  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla
