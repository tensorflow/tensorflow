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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/transforms/block_scaling_rewriter.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;
using BlockScalingRewriterCudnnTest =
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(BlockScalingRewriterCudnnTest, Mxfp8) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[256,256] parameter(0)
  %rhs = f8e4m3fn[256,256] parameter(1)
  %lhs_scale = f8e8m0fnu[256,8] parameter(2)
  %rhs_scale = f8e8m0fnu[256,8] parameter(3)
  ROOT %result = f32[256,256] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(/*allow_cudnn=*/false);
        EXPECT_THAT(RunHloPass(&pass, reference_module), IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(/*allow_cudnn=*/true);
        EXPECT_THAT(RunHloPass(&pass, test_module), IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(false),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(true),
                            "CHECK: __cudnn$blockScaledDot");
}

TEST_F(BlockScalingRewriterCudnnTest, Mxfp8_MixedTypes) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,128,224] parameter(0)
  %rhs = f8e5m2[4,120,224] parameter(1)
  %lhs_scale = f8e8m0fnu[4,128,7] parameter(2)
  %rhs_scale = f8e8m0fnu[4,120,7] parameter(3)
  ROOT %result = f32[4,128,120] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(/*allow_cudnn=*/false);
        EXPECT_THAT(RunHloPass(&pass, reference_module), IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(/*allow_cudnn=*/true);
        EXPECT_THAT(RunHloPass(&pass, test_module), IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(false),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(true),
                            "CHECK: __cudnn$blockScaledDot");
}

// Scale E2M1FN inputs, as otherwise they become all zeros for the random
// distribution produced by the test due to low type precision.
// Use positive block scale values, as Blackwell MMA discards the sign bit on
// the scale tensor.
TEST_F(BlockScalingRewriterCudnnTest, Nvfp4) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %mult_scalar = f16[] constant(6)
  %mult = f16[256,256] broadcast(%mult_scalar), dimensions={}
  %p0 = f16[256,256] parameter(0)
  %p1 = f16[256,256] parameter(1)
  %lhs = f4e2m1fn[256,256] convert(f16[256,256] multiply(%p0, %mult))
  %rhs = f4e2m1fn[256,256] convert(f16[256,256] multiply(%p1, %mult))
  %p2 = f8e4m3fn[256,16] parameter(2)
  %p3 = f8e4m3fn[256,16] parameter(3)
  %lhs_scale = f8e4m3fn[256,16] abs(%p2)
  %rhs_scale = f8e4m3fn[256,16] abs(%p3)
  ROOT %result = f32[256,256] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(/*allow_cudnn=*/false);
        EXPECT_THAT(RunHloPass(&pass, reference_module), IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(/*allow_cudnn=*/true);
        EXPECT_THAT(RunHloPass(&pass, test_module), IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(false),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(true),
                            "CHECK: __cudnn$blockScaledDot");
}

}  // namespace
}  // namespace xla::gpu
