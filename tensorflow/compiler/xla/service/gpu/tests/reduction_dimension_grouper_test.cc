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

}  // namespace
}  // namespace gpu
}  // namespace xla
