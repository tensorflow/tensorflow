/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.h"

#include <optional>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

namespace {

class ReductionDegenerateDimRemoverTest : public HloTestBase {
 public:
  void CheckDegenerateDimRemover(absl::string_view hlo,
                                 std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, gpu::ReductionDegenerateDimRemover{},
                              expected);
  }
};

TEST_F(ReductionDegenerateDimRemoverTest, ReductionWithDegenerateDimensions) {
  const char* hlo = R"(
HloModule ReduceWithDegenerateDimensions

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1,3,1,4,1,5,1] parameter(0)
  zero = f32[] constant(0)

  ROOT out = f32[1,1,1,1] reduce(input, zero), dimensions={1,3,5}, to_apply=add
}

)";

  CheckDegenerateDimRemover(hlo, R"(
// CHECK: [[bitcast_0:%[^ ]+]] = f32[3,4,5]{2,1,0} bitcast([[input_1:%[^ ]+]])
// CHECK: [[reduce_2:%[^ ]+]] = f32[] reduce([[bitcast_0]], [[zero_3:%[^ ]+]]), dimensions={0,1,2}, to_apply=[[add_4:%[^ ]+]]
// CHECK: ROOT [[bitcast_1_5:%[^ ]+]] = f32[1,1,1,1]{3,2,1,0} bitcast([[reduce_2]])
  )");
}

TEST_F(ReductionDegenerateDimRemoverTest,
       ReductionWithDegenerateDimensionsVariadic) {
  const char* hlo = R"(
HloModule ReduceWithDegenerateDimensions

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
  input = f32[1,3,1,4,1,5,1] parameter(0)
  idxs = u32[1,3,1,4,1,5,1] parameter(1)
  zero = f32[] constant(0)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[1,1,1,1], u32[1,1,1,1]) reduce(input, idxs, zero, zero_idx), dimensions={1,3,5}, to_apply=argmax
}

)";

  CheckDegenerateDimRemover(hlo, R"(
// CHECK:  [[bitcast_0:%[^ ]+]] = f32[3,4,5]{2,1,0} bitcast([[input_1:%[^ ]+]])
// CHECK:  [[bitcast_1_2:%[^ ]+]] = u32[3,4,5]{2,1,0} bitcast([[idxs_3:%[^ ]+]])
// CHECK:  [[reduce_4:%[^ ]+]] = (f32[], u32[]) reduce([[bitcast_0]], [[bitcast_1_2]], [[zero_5:%[^ ]+]], [[zero_idx_6:%[^ ]+]]), dimensions={0,1,2}, to_apply=[[argmax_7:%[^ ]+]]
// CHECK-NEXT:  [[get_tuple_element_8:%[^ ]+]] = f32[] get-tuple-element([[reduce_4]]), index=0
// CHECK-NEXT:  [[bitcast_2_9:%[^ ]+]] = f32[1,1,1,1]{3,2,1,0} bitcast([[get_tuple_element_8]])
// CHECK-NEXT:  [[get_tuple_element_1_10:%[^ ]+]] = u32[] get-tuple-element([[reduce_4]]), index=1
// CHECK-NEXT:  [[bitcast_3_11:%[^ ]+]] = u32[1,1,1,1]{3,2,1,0} bitcast([[get_tuple_element_1_10]])
// CHECK-NEXT:  ROOT [[tuple_12:%[^ ]+]] = (f32[1,1,1,1]{3,2,1,0}, u32[1,1,1,1]{3,2,1,0}) tuple([[bitcast_2_9]], [[bitcast_3_11]])
)");
}

TEST_F(ReductionDegenerateDimRemoverTest, DegenerateWithEmptyDimension) {
  const char* hlo = R"(
HloModule ReduceWithDegenerateDimensions

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1,3,1,4,1,5,1] parameter(0)
  zero = f32[] constant(0)

  ROOT out = f32[3,4,5,1] reduce(input, zero), dimensions={0,2,4}, to_apply=add
}
)";

  CheckDegenerateDimRemover(hlo,
                            R"(
// CHECK: ROOT [[bitcast_0:%[^ ]+]] = f32[3,4,5,1]{3,2,1,0} bitcast([[input_1:%[^ ]+]])
      )");
}

}  // namespace
}  // namespace xla
