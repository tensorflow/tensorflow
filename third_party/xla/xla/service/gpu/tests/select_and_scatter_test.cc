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

#include "xla/error_spec.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"

namespace xla {
namespace gpu {
namespace {

using SelectAndScatterTest = GpuCodegenTest;

TEST_F(SelectAndScatterTest, RegressionOOBWrites) {
  const char* hlo_text = R"(
HloModule TestModule

%select_op (a: f32[], b: f32[]) -> pred[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %compare = pred[] compare(f32[] %a, f32[] %b), direction=GE
}

%scatter_op (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

ENTRY %select_and_scatter (operand: f32[5,5], source: f32[3,3]) -> f32[3,3] {
  %source = f32[3,3]{1,0} parameter(1)
  %operand = f32[5,5]{1,0} parameter(0)
  %initial = f32[] constant(0)
  ROOT %result = f32[3,3]{1,0} select-and-scatter(f32[3,3]{1,0} %source, f32[5,5]{1,0} %operand, f32[] %initial), window={size=1x1 pad=1_1x1_1}, select=%select_op, scatter=%scatter_op
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
