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

#include "xla/service/gpu/reduction_dimension_grouper.h"

#include <optional>

#include "absl/strings/string_view.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {

namespace {

class ReductionDimensionGrouperTest : public HloTestBase {
 public:
  void CheckDimensionGrouper(absl::string_view hlo,
                             std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, gpu::ReductionDimensionGrouper{}, expected);
  }
};

TEST_F(ReductionDimensionGrouperTest, ReductionWithGrouping) {
  const char* hlo = R"(
HloModule ReductionWithGrouping

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[100,10,32,3]{3,2,1,0} parameter(0)
  zero = f32[] constant(0)

  ROOT out = f32[100,10]{0,1} reduce(input, zero), dimensions={2,3}, to_apply=add
}
)";

  CheckDimensionGrouper(hlo,
                        R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[100,10,32,3]{3,2,1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[100,10,96]{2,1,0} bitcast([[input_0]])
// CHECK:  ROOT [[out_1_2:%[^ ]+]] = f32[100,10]{0,1} reduce([[bitcast_1]], [[zero_3:%[^ ]+]]), dimensions={2}, to_apply=[[add_4:%[^ ]+]]
      )");
}

TEST_F(ReductionDimensionGrouperTest, ReductionWithGroupingVariadic) {
  const char* hlo = R"(
HloModule ReductionWithGrouping

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
  input = f32[100,10,32,3]{3,2,1,0} parameter(0)
  idxs = u32[100,10,32,3]{3,2,1,0} parameter(1)
  zero = f32[] constant(0)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[100,10]{1,0}, u32[100,10]{1,0}) reduce(input, idxs, zero, zero_idx), dimensions={2,3}, to_apply=argmax
}
)";

  CheckDimensionGrouper(hlo, R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[100,10,32,3]{3,2,1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[100,10,96]{2,1,0} bitcast([[input_0]])
// CHECK:  [[idxs_2:%[^ ]+]] = u32[100,10,32,3]{3,2,1,0} parameter(1)
// CHECK:  [[bitcast_1_3:%[^ ]+]] = u32[100,10,96]{2,1,0} bitcast([[idxs_2]])
// CHECK:  ROOT [[out_1_4:%[^ ]+]] = (f32[100,10]{1,0}, u32[100,10]{1,0}) reduce([[bitcast_1]], [[bitcast_1_3]], [[zero_5:%[^ ]+]], [[zero_idx_6:%[^ ]+]]), dimensions={2}, to_apply=[[argmax_7:%[^ ]+]]
)");
}

}  // namespace
}  // namespace xla
