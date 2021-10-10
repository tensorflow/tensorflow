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

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {

namespace {

class ReductionDimensionGrouperTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.add_xla_disable_hlo_passes("reduction-layout-normalizer");
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    return debug_options;
  }
};

TEST_F(ReductionDimensionGrouperTest, ReductionWithGrouping) {
  const char* hlo_text = R"(
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

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: f32[100,10]{0,1} reduce(f32[100,10,96]{2,1,0} {{.+}}, f32[] {{.+}}), dimensions={2}, to_apply=%add
      )");
}

TEST_F(ReductionDimensionGrouperTest, ReductionWithGroupingVariadic) {
  const char* hlo_text = R"(
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

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (f32[100,10]{1,0}, u32[100,10]{1,0}) reduce(f32[100,10,96]{2,1,0} %bitcast.3, u32[100,10,96]{2,1,0} %bitcast.2, f32[] %zero_1, u32[] %zero_idx_1), dimensions={2}
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
