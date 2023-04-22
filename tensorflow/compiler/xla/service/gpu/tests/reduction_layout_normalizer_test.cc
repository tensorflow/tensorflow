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

class ReductionLayoutNormalizerTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.add_xla_disable_hlo_passes("reduction-dimension-grouper");
    debug_options.add_xla_disable_hlo_passes("reduction-splitter");
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    debug_options.add_xla_disable_hlo_passes("gpu-tree-reduction-rewriter");
    return debug_options;
  }
};

TEST_F(ReductionLayoutNormalizerTest, LayoutCanonicalizerTest) {
  const char* hlo_text = R"(
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

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: f32[4,12,12,16,5]{2,1,3,4,0} reduce(f32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} {{.+}}, f32[] {{.+}}), dimensions={0,1,2}, to_apply=%add
      )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
