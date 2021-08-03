/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

class ReductionJoinerTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.add_xla_disable_hlo_passes("reduction-dimension-grouper");
    debug_options.add_xla_disable_hlo_passes("reduction-splitter");
    debug_options.add_xla_disable_hlo_passes("reduction-layout-normalizer");
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    debug_options.add_xla_disable_hlo_passes("gpu-tree-reduction-rewriter");
    return debug_options;
  }
};

TEST_F(ReductionJoinerTest, RowSimple) {
  const char* hlo_text = R"(
HloModule Reduce

add {
  x0 = f32[] parameter(0)
  y0 = f32[] parameter(1)
  ROOT add0 = f32[] add(x0, y0)
}

ENTRY main {
  p0 = f32[100]  parameter(0)
  p1 = f32[100]  parameter(1)
  zero = f32[] constant(0)
  r0 = f32[] reduce(p0, zero), dimensions={0}, to_apply=add
  r1 = f32[] reduce(p1, zero), dimensions={0}, to_apply=add
  ROOT out = f32[] add(r0, r1)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
// CHECK: f32[200]{0} concatenate
// CHECK: f32[] reduce(%concatenate.1, %zero_1), dimensions={0}, to_apply=%add
      )");
}

TEST_F(ReductionJoinerTest, ColumnSimple) {
  const char* hlo_text = R"(
HloModule Reduce

add {
  x0 = f32[] parameter(0)
  y0 = f32[] parameter(1)
  ROOT add0 = f32[] add(x0, y0)
}

ENTRY main {
  p0 = f32[100,100]  parameter(0)
  p1 = f32[100,100]  parameter(1)
  zero = f32[] constant(0)
  r0 = f32[100] reduce(p0, zero), dimensions={0}, to_apply=add
  r1 = f32[100] reduce(p1, zero), dimensions={0}, to_apply=add
  ROOT out = f32[100] add(r0, r1)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
// CHECK: f32[200,100]{1,0} concatenate
// CHECK: f32[100]{0} reduce(%concatenate.1, %zero_1), dimensions={0}, to_apply=%add
      )");
}

TEST_F(ReductionJoinerTest, DimensionMismatch) {
  const char* hlo_text = R"(
HloModule Reduce

add {
  x0 = f32[] parameter(0)
  y0 = f32[] parameter(1)
  ROOT add0 = f32[] add(x0, y0)
}

ENTRY main {
  p0 = f32[100,100]  parameter(0)
  p1 = f32[100,100]  parameter(1)
  zero = f32[] constant(0)
  r0 = f32[100] reduce(p0, zero), dimensions={0}, to_apply=add
  r1 = f32[100] reduce(p1, zero), dimensions={1}, to_apply=add
  ROOT out = f32[100] add(r0, r1)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
// CHECK-NOT: concat
      )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
