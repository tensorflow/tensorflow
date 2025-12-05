/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/composite_rewriter.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

TEST(CompositeRewriterTest, ScaledDotCompositeRewrite) {
  const std::string hlo_string = R"(
    HloModule jit_my_dot

    %xla.scaled_dot.1 {
      %lhs = f8e4m3fn[3,128,256]{2,1,0} parameter(0)
      %lhs_bf16 = bf16[3,128,256]{2,1,0} convert(%lhs)
      %lhs_scales = f8e8m0fnu[3,128,8]{2,1,0} parameter(2)
      %lhs_scales_bf16 = bf16[3,128,8]{2,1,0} convert(%lhs_scales)
      %lhs_scales_bf16_broadcasted = bf16[3,128,8,32]{3,2,1,0} broadcast(%lhs_scales_bf16), dimensions={0,1,2}
      %lhs_scales_broadcasted = bf16[3,128,256]{2,1,0} reshape(%lhs_scales_bf16_broadcasted)
      %lhs_scaled = bf16[3,128,256]{2,1,0} multiply(%lhs_bf16, %lhs_scales_broadcasted)
      %rhs = f8e4m3fn[3,128,256]{2,1,0} parameter(1)
      %rhs_bf16 = bf16[3,128,256]{2,1,0} convert(%rhs)
      %rhs_scales = f8e8m0fnu[3,128,8]{2,1,0} parameter(3)
      %rhs_scales_bf16 = bf16[3,128,8]{2,1,0} convert(%rhs_scales)
      %rhs_scales_bf16_broadcasted = bf16[3,128,8,32]{3,2,1,0} broadcast(%rhs_scales_bf16), dimensions={0,1,2}
      %rhs_scales_broadcasted = bf16[3,128,256]{2,1,0} reshape(%rhs_scales_bf16_broadcasted)
      %rhs_scaled = bf16[3,128,256]{2,1,0} multiply(%rhs_bf16, %rhs_scales_broadcasted)
      %rhs_scaled_transposed = bf16[3,256,128]{1,2,0} transpose(%rhs_scaled), dimensions={0,2,1}
      ROOT %dot_general.1 = bf16[3,128,128]{2,1,0} dot(%lhs_scaled, %rhs_scaled_transposed),
          lhs_batch_dims={0},
          lhs_contracting_dims={2},
          rhs_batch_dims={0},
          rhs_contracting_dims={1}
    }

    ENTRY %main {
      %lhs = f8e4m3fn[3,128,256]{2,1,0} parameter(0)
      %rhs = f8e4m3fn[3,256,128]{2,1,0} parameter(1)
      %lhs_scales = f8e8m0fnu[3,128,8]{2,1,0} parameter(2)
      %rhs_scales = f8e8m0fnu[3,8,128]{2,1,0} parameter(3)
      ROOT %call.1 = bf16[3,128,128]{2,1,0} call(%lhs, %rhs, %lhs_scales, %rhs_scales),
          to_apply=%xla.scaled_dot.1,
          is_composite=true,
          frontend_attributes={
            composite.attributes="{dimension_numbers=[[[2],[1]],[[0],[0]]]}",
            composite.name="xla.scaled_dot",
            composite.version="1"
          }
    })";
  CompositeRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  EXPECT_THAT(rewriter.Run(module.get()), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kScaledDot);
}

}  // namespace
}  // namespace xla::gpu
