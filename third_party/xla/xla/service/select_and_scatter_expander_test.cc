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

#include "xla/service/select_and_scatter_expander.h"

#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

constexpr absl::string_view kModuleStr =
    R"(HloModule R4F32OverlapSmall_module, entry_computation_layout={()->f32[4,5,1,1]{3,2,1,0}}
  %ge_F32.v3 (lhs: f32[], rhs: f32[]) -> pred[] {
    %lhs = f32[] parameter(0)
    %rhs = f32[] parameter(1)
    ROOT %greater-than-or-equal-to = pred[] compare(f32[] %lhs, f32[] %rhs), direction=GE, type=TOTALORDER
  }

  %add_F32.v3 (lhs.1: f32[], rhs.1: f32[]) -> f32[] {
    %lhs.1 = f32[] parameter(0)
    %rhs.1 = f32[] parameter(1)
    ROOT %add = f32[] add(f32[] %lhs.1, f32[] %rhs.1)
  }

  ENTRY %R4F32OverlapSmall.v4 () -> f32[4,5,1,1] {
    %constant = f32[4,5,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {7} }, { /*i1=1*/ {2} }, { /*i1=2*/ {5} }, { /*i1=3*/ {3} }, { /*i1=4*/ {8} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {8} }, { /*i1=2*/ {9} }, { /*i1=3*/ {3} }, { /*i1=4*/ {4} } }, { /*i0=2*/ { /*i1=0*/ {1} }, { /*i1=1*/ {5} }, { /*i1=2*/ {7} }, { /*i1=3*/ {5} }, { /*i1=4*/ {6} } }, { /*i0=3*/ { /*i1=0*/ {0} }, { /*i1=1*/ {6} }, { /*i1=2*/ {2} }, { /*i1=3*/ {10} }, { /*i1=4*/ {2} } } })
    %constant.1 = f32[2,2,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {2} }, { /*i1=1*/ {6} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {1} } } })
    %constant.2 = f32[] constant(0)
    ROOT %select-and-scatter = f32[4,5,1,1]{3,2,1,0} select-and-scatter(f32[4,5,1,1]{3,2,1,0} %constant, f32[2,2,1,1]{3,2,1,0} %constant.1, f32[] %constant.2), window={size=2x3x1x1 stride=2x2x1x1}, select=%ge_F32.v3, scatter=%add_F32.v3
  })";

class SelectAndScatterExpanderTest : public HloHardwareIndependentTestBase {
 protected:
  // The HLO parser changes all no layout shapes from the input to have a
  // default layout. Clear the layout of the scatter operand for testing.
  void ClearInstructionLayout(HloModule* module, absl::string_view inst_name) {
    HloInstruction* inst = FindInstruction(module, inst_name);
    inst->mutable_shape()->clear_layout();
  }
};

// Test for the expected primary composite ops after this transformation and
// leave correctness to runtime tests instead of golden IR checks.
TEST_F(SelectAndScatterExpanderTest, ReplacesSelectAndScatter) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RunAndFilecheckHloRewrite(kModuleStr, SelectAndScatterExpander(), R"(
    CHECK-NOT: select-and-scatter
  )");
}

TEST_F(SelectAndScatterExpanderTest, CreatesReduceAndScatter) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RunAndFilecheckHloRewrite(kModuleStr, SelectAndScatterExpander(), R"(
    CHECK: reduce
    CHECK: scatter
  )");
}

}  // namespace
}  // namespace xla
