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

#include "xla/service/gpu/transforms/block_scaling_rewriter.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using BlockScalingRewriterTest = HloTestBase;

TEST_F(BlockScalingRewriterTest, ExpandQuantizeCustomCall) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %input = f32[10,256] parameter(0)
  ROOT %result = (f8e4m3fn[10,256], f8e5m2[10,8]) custom-call(%input),
      custom_call_target="__op$quantize"
})";

  BlockScalingRewriter pass;
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[input:%.+]] = f32[10,256]{1,0} parameter(0)
  CHECK: [[blocks:%.+]] = f32[10,8,32]{2,1,0} reshape([[input]])
  CHECK: [[zero:%.+]] = f32[] constant(0)
  CHECK: [[amax:%.+]] = f32[10,8]{1,0} reduce([[blocks]], [[zero]]), dimensions={2}, to_apply=%amax
  CHECK: [[emax:%.+]] = f32[] constant(256)
  CHECK: [[emax_bc:%.+]] = f32[10,8]{1,0} broadcast([[emax]]), dimensions={}
  CHECK: [[amax_norm:%.+]] = f32[10,8]{1,0} divide([[amax]], [[emax_bc]])
  CHECK: [[scale:%.+]] = f8e5m2[10,8]{1,0} convert([[amax_norm]])
  CHECK: [[scale_cvt:%.+]] = f32[10,8]{1,0} convert([[scale]])
  CHECK: [[scale_bc:%.+]] = f32[10,8,32]{2,1,0} broadcast([[scale_cvt]]), dimensions={0,1}
  CHECK: [[scale_rs:%.+]] = f32[10,256]{1,0} reshape([[scale_bc]])
  CHECK: [[result:%.+]] = f32[10,256]{1,0} divide([[input]], [[scale_rs]])
  CHECK: [[quantized:%.+]] = f8e4m3fn[10,256]{1,0} convert([[result]])
  CHECK: ROOT {{.+}} = (f8e4m3fn[10,256]{1,0}, f8e5m2[10,8]{1,0}) tuple([[quantized]], [[scale]])
})");
}

TEST_F(BlockScalingRewriterTest, ExpandDequantizeCustomCall) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %input = f8e4m3fn[10,256] parameter(0)
  %scale = f8e5m2[10,8] parameter(1)
  ROOT %result = f32[10,256] custom-call(%input, %scale),
      custom_call_target="__op$dequantize"
})";

  BlockScalingRewriter pass;
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[input:%.+]] = f8e4m3fn[10,256]{1,0} parameter(0)
  CHECK: [[input_cvt:%.+]] = f32[10,256]{1,0} convert([[input]])
  CHECK: [[scale:%.+]] = f8e5m2[10,8]{1,0} parameter(1)
  CHECK: [[scale_cvt:%.+]] = f32[10,8]{1,0} convert([[scale]])
  CHECK: [[broadcast:%.+]] = f32[10,8,32]{2,1,0} broadcast([[scale_cvt]]), dimensions={0,1}
  CHECK: [[reshape:%.+]] = f32[10,256]{1,0} reshape([[broadcast]])
  CHECK: ROOT {{.+}} = f32[10,256]{1,0} multiply([[input_cvt]], [[reshape]])
})");
}

TEST_F(BlockScalingRewriterTest, ExpandBlockScaledDotCustomCall) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,16,256] parameter(0)
  %rhs = f8e4m3fn[4,32,256] parameter(1)
  %lhs_scale = f8e5m2[4,16,8] parameter(2)
  %rhs_scale = f8e5m2[4,32,8] parameter(3)
  ROOT %result = f32[4,16,32] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass;
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[lhs_quant:%.+]] = f8e4m3fn[4,16,256]{2,1,0} parameter(0)
  CHECK: [[lhs_quant_cvt:%.+]] = f32[4,16,256]{2,1,0} convert([[lhs_quant]])
  CHECK: [[lhs_scale:%.+]] = f8e5m2[4,16,8]{2,1,0} parameter(2)
  CHECK: [[lhs_scale_cvt:%.+]] = f32[4,16,8]{2,1,0} convert([[lhs_scale]])
  CHECK: [[lhs_scale_bc:%.+]] = f32[4,16,8,32]{3,2,1,0} broadcast([[lhs_scale_cvt]])
  CHECK: [[lhs_scale_rs:%.+]] = f32[4,16,256]{2,1,0} reshape([[lhs_scale_bc]])
  CHECK: [[lhs:%.+]] = f32[4,16,256]{2,1,0} multiply([[lhs_quant_cvt]], [[lhs_scale_rs]])
  CHECK: [[rhs_quant:%.+]] = f8e4m3fn[4,32,256]{2,1,0} parameter(1)
  CHECK: [[rhs_quant_cvt:%.+]] = f32[4,32,256]{2,1,0} convert([[rhs_quant]])
  CHECK: [[rhs_scale:%.+]] = f8e5m2[4,32,8]{2,1,0} parameter(3)
  CHECK: [[rhs_scale_cvt:%.+]] = f32[4,32,8]{2,1,0} convert([[rhs_scale]])
  CHECK: [[rhs_scale_bc:%.+]] = f32[4,32,8,32]{3,2,1,0} broadcast([[rhs_scale_cvt]])
  CHECK: [[rhs_scale_rs:%.+]] = f32[4,32,256]{2,1,0} reshape([[rhs_scale_bc]])
  CHECK: [[rhs:%.+]] = f32[4,32,256]{2,1,0} multiply([[rhs_quant_cvt]], [[rhs_scale_rs]])
  CHECK: ROOT {{.+}} = f32[4,16,32]{2,1,0} dot([[lhs]], [[rhs]])
  CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={2}
  CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={2}
})");
}

TEST_F(BlockScalingRewriterTest, ExpandBlockScaledDotQuantizedLhs) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[16,256] parameter(0)
  %rhs = f16[32,256] parameter(1)
  %lhs_scale = f8e5m2[16,8] parameter(2)
  ROOT %result = f16[16,32] custom-call(%lhs, %rhs, %lhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass;
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[lhs_quant:%.+]] = f8e4m3fn[16,256]{1,0} parameter(0)
  CHECK: [[lhs_quant_cvt:%.+]] = f16[16,256]{1,0} convert([[lhs_quant]])
  CHECK: [[lhs_scale:%.+]] = f8e5m2[16,8]{1,0} parameter(2)
  CHECK: [[lhs_scale_cvt:%.+]] = f16[16,8]{1,0} convert([[lhs_scale]])
  CHECK: [[lhs_scale_bc:%.+]] = f16[16,8,32]{2,1,0} broadcast([[lhs_scale_cvt]])
  CHECK: [[lhs_scale_rs:%.+]] = f16[16,256]{1,0} reshape([[lhs_scale_bc]])
  CHECK: [[lhs:%.+]] = f16[16,256]{1,0} multiply([[lhs_quant_cvt]], [[lhs_scale_rs]])
  CHECK: [[rhs:%.+]] = f16[32,256]{1,0} parameter(1)
  CHECK: ROOT {{.+}} = f16[16,32]{1,0} dot([[lhs]], [[rhs]])
  CHECK-SAME: lhs_contracting_dims={1}, rhs_contracting_dims={1}
})");
}

TEST_F(BlockScalingRewriterTest, QuantizeDequantizeCompare) {
  constexpr absl::string_view hlo_test = R"(
HloModule test
ENTRY main {
  %input = f32[256,256] parameter(0)
  %quantized = (f8e4m3fn[256,256], f8e5m2[256,16]) custom-call(%input),
      custom_call_target="__op$quantize"
  %values = f8e4m3fn[256,256] get-tuple-element(%quantized), index=0
  %scales = f8e5m2[256,16] get-tuple-element(%quantized), index=1
  ROOT %dequantized = f32[256,256] custom-call(%values, %scales),
      custom_call_target="__op$dequantize"
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnUnverifiedModule(hlo_test));

  BlockScalingRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(
      auto changed, pass.Run(test_module.get(), /*execution_threads=*/{}));
  EXPECT_TRUE(changed);

  constexpr absl::string_view hlo_reference = R"(
HloModule reference
ENTRY main {
  ROOT %input = f32[256,256] parameter(0)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto reference_module,
                          ParseAndReturnUnverifiedModule(hlo_reference));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(test_module),
                                      std::move(reference_module),
                                      ErrorSpec(/*aabs=*/0.01, /*arel=*/0.07),
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace xla::gpu
