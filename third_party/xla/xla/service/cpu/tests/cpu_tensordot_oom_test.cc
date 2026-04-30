/* Copyright 2024 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace cpu {
namespace {

using ::tsl::testing::IsOk;

class CpuTensordotOomTest : public HloPjRtTestBase {};

// Regression test for OOM caused by tensordots.
// Reference: https://github.com/openxla/xla/pull/41174
TEST_F(CpuTensordotOomTest, TensordotDoesNotOom) {
  const std::string hlo_text = R"(
HloModule jit_loss

ENTRY %main.1 (p_0: f32[2,2,2], p_1: f32[2,2,2], p_2: f32[2,2,2], p_3: f32[2,2,2], p_4: f32[2,2,2], p_5: f32[2,2,2], p_6: f32[2,2,2], p_7: f32[2,2,2], p_8: f32[2,2,2], p_9: f32[2,2,2], h_0: f32[2,2,2,2], h_1: f32[2,2,2,2], h_2: f32[2,2,2,2], h_3: f32[2,2,2,2], h_4: f32[2,2,2,2], h_5: f32[2,2,2,2], h_6: f32[2,2,2,2], h_7: f32[2,2,2,2], h_8: f32[2,2,2,2], h_9: f32[2,2,2,2]) -> f32[] {
  %p_1.1 = f32[2,2,2]{2,1,0} parameter(1)
  %p_2.1 = f32[2,2,2]{2,1,0} parameter(2)
  %dot.33 = f32[2,2,2,2]{3,2,1,0} dot(%p_1.1, %p_2.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %p_0.1 = f32[2,2,2]{2,1,0} parameter(0)
  %dot.34 = f32[2,2,2,2,2]{4,3,2,1,0} dot(%dot.33, %p_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  %p_3.1 = f32[2,2,2]{2,1,0} parameter(3)
  %p_4.1 = f32[2,2,2]{2,1,0} parameter(4)
  %dot.35 = f32[2,2,2,2]{3,2,1,0} dot(%p_3.1, %p_4.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.36 = f32[2,2,2,2,2,2,2]{6,5,4,3,2,1,0} dot(%dot.34, %dot.35), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %p_7.1 = f32[2,2,2]{2,1,0} parameter(7)
  %p_8.1 = f32[2,2,2]{2,1,0} parameter(8)
  %dot.29 = f32[2,2,2,2]{3,2,1,0} dot(%p_7.1, %p_8.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %p_9.1 = f32[2,2,2]{2,1,0} parameter(9)
  %dot.30 = f32[2,2,2,2,2]{4,3,2,1,0} dot(%dot.29, %p_9.1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  %p_5.1 = f32[2,2,2]{2,1,0} parameter(5)
  %p_6.1 = f32[2,2,2]{2,1,0} parameter(6)
  %dot.31 = f32[2,2,2,2]{3,2,1,0} dot(%p_5.1, %p_6.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.32 = f32[2,2,2,2,2,2,2]{6,5,4,3,2,1,0} dot(%dot.30, %dot.31), lhs_contracting_dims={0}, rhs_contracting_dims={2}
  %dot.37 = f32[2,2,2,2,2,2,2,2,2,2]{9,8,7,6,5,4,3,2,1,0} dot(%dot.36, %dot.32), lhs_contracting_dims={2,5}, rhs_contracting_dims={2,4}
  %h_0.1 = f32[2,2,2,2]{3,2,1,0} parameter(10)
  %h_9.1 = f32[2,2,2,2]{3,2,1,0} parameter(19)
  %dot.39 = f32[2,2,2,2,2,2]{5,4,3,2,1,0} dot(%h_0.1, %h_9.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  %h_1.1 = f32[2,2,2,2]{3,2,1,0} parameter(11)
  %h_2.1 = f32[2,2,2,2]{3,2,1,0} parameter(12)
  %dot.38 = f32[2,2,2,2,2,2]{5,4,3,2,1,0} dot(%h_1.1, %h_2.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.40 = f32[2,2,2,2,2,2,2,2,2,2]{9,8,7,6,5,4,3,2,1,0} dot(%dot.39, %dot.38), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  %dot.41 = f32[2,2,2,2,2,2,2,2,2,2,2,2]{11,10,9,8,7,6,5,4,3,2,1,0} dot(%dot.37, %dot.40), lhs_contracting_dims={0,1,2,7}, rhs_contracting_dims={5,8,0,3}
  %h_7.1 = f32[2,2,2,2]{3,2,1,0} parameter(17)
  %h_8.1 = f32[2,2,2,2]{3,2,1,0} parameter(18)
  %dot.42 = f32[2,2,2,2,2,2]{5,4,3,2,1,0} dot(%h_7.1, %h_8.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.43 = f32[2,2,2,2,2,2,2,2,2,2,2,2]{11,10,9,8,7,6,5,4,3,2,1,0} dot(%dot.41, %dot.42), lhs_contracting_dims={2,3,7}, rhs_contracting_dims={1,4,3}
  %h_3.1 = f32[2,2,2,2]{3,2,1,0} parameter(13)
  %h_4.1 = f32[2,2,2,2]{3,2,1,0} parameter(14)
  %dot.45 = f32[2,2,2,2,2,2]{5,4,3,2,1,0} dot(%h_3.1, %h_4.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %h_5.1 = f32[2,2,2,2]{3,2,1,0} parameter(15)
  %h_6.1 = f32[2,2,2,2]{3,2,1,0} parameter(16)
  %dot.44 = f32[2,2,2,2,2,2]{5,4,3,2,1,0} dot(%h_5.1, %h_6.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.46 = f32[2,2,2,2,2,2,2,2,2,2]{9,8,7,6,5,4,3,2,1,0} dot(%dot.45, %dot.44), lhs_contracting_dims={3}, rhs_contracting_dims={0}
  %dot.47 = f32[2,2,2,2,2,2,2,2,2,2]{9,8,7,6,5,4,3,2,1,0} dot(%dot.43, %dot.46), lhs_contracting_dims={0,1,2,3,7,9}, rhs_contracting_dims={1,3,5,8,0,7}
  %dot.48 = f32[2,2,2,2]{3,2,1,0} dot(%p_5.1, %p_6.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.49 = f32[2,2,2,2,2]{4,3,2,1,0} dot(%dot.48, %p_4.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  %dot.50 = f32[2,2,2,2]{3,2,1,0} dot(%p_7.1, %p_8.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.51 = f32[2,2,2,2,2,2,2]{6,5,4,3,2,1,0} dot(%dot.49, %dot.50), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.52 = f32[2,2,2,2,2,2,2]{6,5,4,3,2,1,0} dot(%dot.47, %dot.51), lhs_contracting_dims={4,5,7,8,9}, rhs_contracting_dims={4,6,3,0,1}
  %dot.53 = f32[2,2,2,2,2,2]{5,4,3,2,1,0} dot(%dot.52, %p_9.1), lhs_contracting_dims={1,6}, rhs_contracting_dims={2,0}
  %dot.54 = f32[2,2,2,2,2]{4,3,2,1,0} dot(%dot.53, %p_0.1), lhs_contracting_dims={0,5}, rhs_contracting_dims={2,0}
  %dot.55 = f32[2,2,2,2]{3,2,1,0} dot(%dot.54, %p_1.1), lhs_contracting_dims={0,4}, rhs_contracting_dims={2,0}
  %dot.56 = f32[2,2,2]{2,1,0} dot(%dot.55, %p_2.1), lhs_contracting_dims={0,3}, rhs_contracting_dims={2,0}
  ROOT %dot.57 = f32[] dot(%dot.56, %p_3.1), lhs_contracting_dims={0,1,2}, rhs_contracting_dims={2,1,0}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  // Ensure that compiling the module does not crash or run out of memory.
  EXPECT_THAT(
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true).status(),
      IsOk());
}

}  // namespace
}  // namespace cpu
}  // namespace xla
