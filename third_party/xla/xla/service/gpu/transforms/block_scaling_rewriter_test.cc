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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using BlockScalingRewriterTest = HloHardwareIndependentTestBase;

TEST_F(BlockScalingRewriterTest, ExpandQuantizeCustomCall) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %input = f32[10,256] parameter(0)
  ROOT %result = (f8e4m3fn[10,256], f8e5m2[10,8]) custom-call(%input),
      custom_call_target="__op$quantize"
})";

  BlockScalingRewriter pass(se::dnn::VersionInfo{});
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

  BlockScalingRewriter pass(se::dnn::VersionInfo{});
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

  BlockScalingRewriter pass(se::dnn::VersionInfo{});
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

TEST_F(BlockScalingRewriterTest, ExpandBlockScaledDotGlobalScale) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,16,256] parameter(0)
  %rhs = f8e4m3fn[4,32,256] parameter(1)
  %lhs_scale = f8e5m2[4,16,8] parameter(2)
  %rhs_scale = f8e5m2[4,32,8] parameter(3)
  %global_scale = f32[] parameter(4)
  ROOT %result = f32[4,16,32] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale, %global_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass(se::dnn::VersionInfo{});
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
  CHECK: [[dot:%.+]] = f32[4,16,32]{2,1,0} dot([[lhs]], [[rhs]])
  CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={2}
  CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={2}
  CHECK: [[global_scale:%.+]] = f32[] parameter(4)
  CHECK: [[global_scale_bc:%.+]] = f32[4,16,32]{2,1,0} broadcast([[global_scale]]), dimensions={}
  CHECK: ROOT {{.+}} = f32[4,16,32]{2,1,0} multiply([[dot]], [[global_scale_bc]])
})");
}

TEST_F(BlockScalingRewriterTest, ExpandBlockScaledDotNonDefaultLayout) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,16,256]{2,0,1} parameter(0)
  %rhs = f8e4m3fn[4,32,256]{0,1,2} parameter(1)
  %lhs_scale = f8e5m2[4,16,8]{2,0,1} parameter(2)
  %rhs_scale = f8e5m2[4,32,8]{0,1,2} parameter(3)
  ROOT %result = f32[4,16,32] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass(se::dnn::VersionInfo{});
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[lhs_quant:%.+]] = f8e4m3fn[4,16,256]{2,0,1} parameter(0)
  CHECK: [[lhs_quant_cvt:%.+]] = f32[4,16,256]{2,0,1} convert([[lhs_quant]])
  CHECK: [[lhs_scale:%.+]] = f8e5m2[4,16,8]{2,0,1} parameter(2)
  CHECK: [[lhs_scale_cvt:%.+]] = f32[4,16,8]{2,0,1} convert([[lhs_scale]])
  CHECK: [[lhs_scale_bc:%.+]] = f32[4,16,8,32]{3,0,2,1} broadcast([[lhs_scale_cvt]])
  CHECK: [[lhs_scale_rs:%.+]] = f32[4,16,256]{2,0,1} reshape([[lhs_scale_bc]])
  CHECK: [[lhs:%.+]] = f32[4,16,256]{2,0,1} multiply([[lhs_quant_cvt]], [[lhs_scale_rs]])
  CHECK: [[rhs_quant:%.+]] = f8e4m3fn[4,32,256]{0,1,2} parameter(1)
  CHECK: [[rhs_quant_cvt:%.+]] = f32[4,32,256]{0,1,2} convert([[rhs_quant]])
  CHECK: [[rhs_scale:%.+]] = f8e5m2[4,32,8]{0,1,2} parameter(3)
  CHECK: [[rhs_scale_cvt:%.+]] = f32[4,32,8]{0,1,2} convert([[rhs_scale]])
  CHECK: [[rhs_scale_bc:%.+]] = f32[4,32,8,32]{0,1,3,2} broadcast([[rhs_scale_cvt]])
  CHECK: [[rhs_scale_rs:%.+]] = f32[4,32,256]{0,1,2} reshape([[rhs_scale_bc]])
  CHECK: [[rhs:%.+]] = f32[4,32,256]{0,1,2} multiply([[rhs_quant_cvt]], [[rhs_scale_rs]])
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

  BlockScalingRewriter pass(se::dnn::VersionInfo{});
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

TEST_F(BlockScalingRewriterTest, ExpandBlockScaledDotExplicitBlockSize) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,16,224] parameter(0)
  %rhs = f8e4m3fn[4,32,224] parameter(1)
  %lhs_scale = f8e5m2[4,16,8] parameter(2)
  %rhs_scale = f8e5m2[4,32,8] parameter(3)
  ROOT %result = f32[4,16,32] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot",
      backend_config={"block_scaled_dot_backend_config":{block_size:32}}
})";

  BlockScalingRewriter pass(se::dnn::VersionInfo{});
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[lhs_quant:%.+]] = f8e4m3fn[4,16,224]{2,1,0} parameter(0)
  CHECK: [[lhs_quant_cvt:%.+]] = f32[4,16,224]{2,1,0} convert([[lhs_quant]])
  CHECK: [[lhs_scale:%.+]] = f8e5m2[4,16,8]{2,1,0} parameter(2)
  CHECK: [[lhs_scale_sl:%.+]] = f8e5m2[4,16,7]{2,1,0} slice([[lhs_scale]])
  CHECK: [[lhs_scale_cvt:%.+]] = f32[4,16,7]{2,1,0} convert([[lhs_scale_sl]])
  CHECK: [[lhs_scale_bc:%.+]] = f32[4,16,7,32]{3,2,1,0} broadcast([[lhs_scale_cvt]])
  CHECK: [[lhs_scale_rs:%.+]] = f32[4,16,224]{2,1,0} reshape([[lhs_scale_bc]])
  CHECK: [[lhs:%.+]] = f32[4,16,224]{2,1,0} multiply([[lhs_quant_cvt]], [[lhs_scale_rs]])
  CHECK: [[rhs_quant:%.+]] = f8e4m3fn[4,32,224]{2,1,0} parameter(1)
  CHECK: [[rhs_quant_cvt:%.+]] = f32[4,32,224]{2,1,0} convert([[rhs_quant]])
  CHECK: [[rhs_scale:%.+]] = f8e5m2[4,32,8]{2,1,0} parameter(3)
  CHECK: [[rhs_scale_sl:%.+]] = f8e5m2[4,32,7]{2,1,0} slice([[rhs_scale]])
  CHECK: [[rhs_scale_cvt:%.+]] = f32[4,32,7]{2,1,0} convert([[rhs_scale_sl]])
  CHECK: [[rhs_scale_bc:%.+]] = f32[4,32,7,32]{3,2,1,0} broadcast([[rhs_scale_cvt]])
  CHECK: [[rhs_scale_rs:%.+]] = f32[4,32,224]{2,1,0} reshape([[rhs_scale_bc]])
  CHECK: [[rhs:%.+]] = f32[4,32,224]{2,1,0} multiply([[rhs_quant_cvt]], [[rhs_scale_rs]])
  CHECK: ROOT {{.+}} = f32[4,16,32]{2,1,0} dot([[lhs]], [[rhs]])
  CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={2}
  CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={2}
})");
}

TEST_F(BlockScalingRewriterTest, CudnnScaledDotSimple) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,128,128] parameter(0)
  %rhs = f8e4m3fn[4,128,128] parameter(1)
  %lhs_scale = f8e8m0fnu[4,128,4] parameter(2)
  %rhs_scale = f8e8m0fnu[4,128,4] parameter(3)
  ROOT %result = f16[4,128,128] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass(kCudnnSupportsBlockScaledDot);
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[lhs:%.+]] = f8e4m3fn[4,128,128]{2,1,0} parameter(0)
  CHECK: [[rhs:%.+]] = f8e4m3fn[4,128,128]{2,1,0} parameter(1)
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[4,128,4]{2,1,0} parameter(2)
  CHECK: [[lhs_scale_rs:%.+]] = f8e8m0fnu[4,1,4,32,1,4]{5,4,3,2,1,0} reshape([[lhs_scale]])
  CHECK: [[lhs_scale_tr:%.+]] = f8e8m0fnu[4,1,1,32,4,4]{5,2,3,4,1,0} transpose([[lhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[lhs_scale_swizzle:%.+]] = f8e8m0fnu[4,128,4]{2,1,0} reshape([[lhs_scale_tr]])
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[4,128,4]{2,1,0} parameter(3)
  CHECK: [[rhs_scale_rs:%.+]] = f8e8m0fnu[4,1,4,32,1,4]{5,4,3,2,1,0} reshape([[rhs_scale]])
  CHECK: [[rhs_scale_tr:%.+]] = f8e8m0fnu[4,1,1,32,4,4]{5,2,3,4,1,0} transpose([[rhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[rhs_scale_swizzle:%.+]] = f8e8m0fnu[4,128,4]{2,1,0} reshape([[rhs_scale_tr]])
  CHECK: [[call:%.+]] = ({{.+}}) custom-call([[lhs]], [[rhs]], [[lhs_scale_swizzle]], [[rhs_scale_swizzle]])
  CHECK-SAME: custom_call_target="__cudnn$blockScaledDot"
  CHECK: ROOT {{.+}} = f16[4,128,128]{2,1,0} get-tuple-element([[call]]), index=0
})");
}

TEST_F(BlockScalingRewriterTest, CudnnScaledDotTransforms) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[128,96] parameter(0)
  %rhs = f8e4m3fn[120,96] parameter(1)
  %lhs_scale = f8e8m0fnu[128,3] parameter(2)
  %rhs_scale = f8e8m0fnu[120,3] parameter(3)
  ROOT %result = f16[128,120] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass(kCudnnSupportsBlockScaledDot);
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[lhs:%.+]] = f8e4m3fn[128,96]{1,0} parameter(0)
  CHECK: [[rhs:%.+]] = f8e4m3fn[120,96]{1,0} parameter(1)
  CHECK: [[rhs_pad:%.+]] = f8e4m3fn[128,96]{1,0} pad([[rhs]], {{.+}}), padding=0_8x0_0
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[128,3]{1,0} parameter(2)
  CHECK: [[lhs_scale_pad:%.+]] = f8e8m0fnu[128,4]{1,0} pad([[lhs_scale]], {{.+}}), padding=0_0x0_1
  CHECK: [[lhs_scale_rs2:%.+]] = f8e8m0fnu[1,1,4,32,1,4]{5,4,3,2,1,0} reshape([[lhs_scale_pad]])
  CHECK: [[lhs_scale_tr2:%.+]] = f8e8m0fnu[1,1,1,32,4,4]{5,2,3,4,1,0} transpose([[lhs_scale_rs2]]), dimensions={0,1,4,3,2,5}
  CHECK: [[lhs_scale_swizzle:%.+]] = f8e8m0fnu[128,4]{1,0} reshape([[lhs_scale_tr2]])
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[120,3]{1,0} parameter(3)
  CHECK: [[rhs_scale_pad:%.+]] = f8e8m0fnu[128,4]{1,0} pad([[rhs_scale]], {{.+}}), padding=0_8x0_1
  CHECK: [[rhs_scale_rs2:%.+]] = f8e8m0fnu[1,1,4,32,1,4]{5,4,3,2,1,0} reshape([[rhs_scale_pad]])
  CHECK: [[rhs_scale_tr2:%.+]] = f8e8m0fnu[1,1,1,32,4,4]{5,2,3,4,1,0} transpose([[rhs_scale_rs2]]), dimensions={0,1,4,3,2,5}
  CHECK: [[rhs_scale_swizzle:%.+]] = f8e8m0fnu[128,4]{1,0} reshape([[rhs_scale_tr2]])
  CHECK: [[call:%.+]] = ({{.+}}) custom-call([[lhs]], [[rhs_pad]], [[lhs_scale_swizzle]], [[rhs_scale_swizzle]])
  CHECK: [[gte:%.+]] = f16[128,128]{1,0} get-tuple-element([[call]]), index=0
  CHECK: ROOT {{.+}} = f16[128,120]{1,0} slice([[gte]]), slice={[0:128], [0:120]}
})");
}

TEST_F(BlockScalingRewriterTest, CudnnScaledDotPaddedScales) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,120,96] parameter(0)
  %rhs = f8e4m3fn[4,120,96] parameter(1)
  %lhs_scale = f8e8m0fnu[4,128,4] parameter(2)
  %rhs_scale = f8e8m0fnu[4,128,4] parameter(3)
  ROOT %result = f16[4,120,120] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot",
      backend_config={"block_scaled_dot_backend_config":{block_size:32}}
})";

  BlockScalingRewriter pass(kCudnnSupportsBlockScaledDot);
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[lhs:%.+]] = f8e4m3fn[4,120,96]{2,1,0} parameter(0)
  CHECK: [[lhs_pad:%.+]] = f8e4m3fn[4,128,96]{2,1,0} pad([[lhs]], {{.+}}), padding=0_0x0_8x0_0
  CHECK: [[rhs:%.+]] = f8e4m3fn[4,120,96]{2,1,0} parameter(1)
  CHECK: [[rhs_pad:%.+]] = f8e4m3fn[4,128,96]{2,1,0} pad([[rhs]], {{.+}}), padding=0_0x0_8x0_0
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[4,128,4]{2,1,0} parameter(2)
  CHECK: [[lhs_scale_rs:%.+]] = f8e8m0fnu[4,1,4,32,1,4]{5,4,3,2,1,0} reshape([[lhs_scale]])
  CHECK: [[lhs_scale_tr:%.+]] = f8e8m0fnu[4,1,1,32,4,4]{5,2,3,4,1,0} transpose([[lhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[lhs_scale_swizzle:%.+]] = f8e8m0fnu[4,128,4]{2,1,0} reshape([[lhs_scale_tr]])
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[4,128,4]{2,1,0} parameter(3)
  CHECK: [[rhs_scale_rs:%.+]] = f8e8m0fnu[4,1,4,32,1,4]{5,4,3,2,1,0} reshape([[rhs_scale]])
  CHECK: [[rhs_scale_tr:%.+]] = f8e8m0fnu[4,1,1,32,4,4]{5,2,3,4,1,0} transpose([[rhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[rhs_scale_swizzle:%.+]] = f8e8m0fnu[4,128,4]{2,1,0} reshape([[rhs_scale_tr]])
  CHECK: [[call:%.+]] = ({{.+}}) custom-call([[lhs_pad]], [[rhs_pad]], [[lhs_scale_swizzle]], [[rhs_scale_swizzle]])
  CHECK-SAME: custom_call_target="__cudnn$blockScaledDot"
  CHECK: [[gte:%.+]] = f16[4,128,128]{2,1,0} get-tuple-element([[call]]), index=0
  CHECK: ROOT {{.+}} = f16[4,120,120]{2,1,0} slice([[gte]]), slice={[0:4], [0:120], [0:120]}
})");
}

TEST_F(BlockScalingRewriterTest, CudnnFusionSupportedE4M3) {
  constexpr absl::string_view hlo_string = R"(
ENTRY main {
  %lhs = f8e4m3fn[128,256] parameter(0)
  %rhs = f8e4m3fn[128,256] parameter(1)
  %lhs_scale = f8e8m0fnu[128,8] parameter(2)
  %rhs_scale = f8e8m0fnu[128,8] parameter(3)
  ROOT %result = f16[128,128] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CudnnScaledDotHelper::IsSupported(Cast<HloScaledDotInstruction>(
      test_module->entry_computation()->root_instruction())));
}

TEST_F(BlockScalingRewriterTest, CudnnFusionUnsupportedE5M2) {
  constexpr absl::string_view hlo_string = R"(
ENTRY main {
  %lhs = f8e5m2[128,256] parameter(0)
  %rhs = f8e5m2[128,256] parameter(1)
  %lhs_scale = f8e8m0fnu[128,8] parameter(2)
  %rhs_scale = f8e8m0fnu[128,8] parameter(3)
  ROOT %result = f16[128,128] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(CudnnScaledDotHelper::IsSupported(Cast<HloScaledDotInstruction>(
      test_module->entry_computation()->root_instruction())));
}

TEST_F(BlockScalingRewriterTest, CudnnFusionUnsupportedDimensions) {
  constexpr absl::string_view hlo_string = R"(
ENTRY main {
  %lhs = f8e4m3fn[128,256] parameter(0)
  %rhs = f8e4m3fn[256,128] parameter(1)
  %lhs_scale = f8e8m0fnu[128,8] parameter(2)
  %rhs_scale = f8e8m0fnu[8,128] parameter(3)
  ROOT %result = f16[128,128] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(CudnnScaledDotHelper::IsSupported(Cast<HloScaledDotInstruction>(
      test_module->entry_computation()->root_instruction())));
}

TEST_F(BlockScalingRewriterTest, CudnnFusionUnupportedLayout) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[16,128]{0,1} parameter(0)
  %rhs = f8e4m3fn[32,128]{1,0} parameter(1)
  %lhs_scale = f8e8m0fnu[16,4]{1,0} parameter(2)
  %rhs_scale = f8e8m0fnu[32,4]{1,0} parameter(3)
  ROOT %result = f16[16,32]{1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(CudnnScaledDotHelper::IsSupported(Cast<HloScaledDotInstruction>(
      test_module->entry_computation()->root_instruction())));
}

TEST_F(BlockScalingRewriterTest, CudnnFusionUnsupportedInputs) {
  constexpr absl::string_view hlo_string = R"(
ENTRY main {
  %lhs = f8e4m3fn[128] parameter(0)
  %lhs_bc = f8e4m3fn[128,256] broadcast(%lhs), dimensions={0}
  %rhs = f8e4m3fn[128,256] parameter(1)
  %lhs_scale = f8e8m0fnu[128,8] parameter(2)
  %rhs_scale = f8e8m0fnu[128,8] parameter(3)
  ROOT %result = f16[128,128] scaled-dot(%lhs_bc, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(CudnnScaledDotHelper::IsSupported(Cast<HloScaledDotInstruction>(
      test_module->entry_computation()->root_instruction())));
}

TEST_F(BlockScalingRewriterTest, CudnnFusionSwizzleSimple) {
  constexpr absl::string_view hlo_string = R"(
fusion {
  %lhs = f8e4m3fn[4,384,256] parameter(0)
  %rhs = f8e4m3fn[4,512,256] parameter(1)
  %lhs_scale = f8e8m0fnu[4,384,8] parameter(2)
  %rhs_scale = f8e8m0fnu[4,512,8] parameter(3)
  ROOT %result = f32[4,384,512] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  %lhs = f8e4m3fn[4,384,256] parameter(0)
  %rhs = f8e4m3fn[4,512,256] parameter(1)
  %lhs_scale = f8e8m0fnu[4,384,8] parameter(2)
  %rhs_scale = f8e8m0fnu[4,512,8] parameter(3)
  ROOT %result = f32[4,384,512] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=fusion,
      backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(CudnnScaledDotHelper::AddScaleSwizzle(Cast<HloFusionInstruction>(
      test_module->entry_computation()->root_instruction())));

  constexpr absl::string_view expected = R"(
  // %lhs_scale.1_swizzle.1
  CHECK: [[lhs_bc:%.+]] = f8e8m0fnu[4,3,4,32,2,4]{5,4,3,2,1,0} bitcast({{.+}})
  CHECK: [[lhs_tr:%.+]] = f8e8m0fnu[4,3,2,32,4,4]{5,4,3,2,1,0} transpose([[lhs_bc]]), dimensions={0,1,4,3,2,5}
  CHECK: {{.+}} = f8e8m0fnu[4,384,8]{2,1,0} bitcast([[lhs_tr]])
  // %rhs_scale.1_swizzle.1
  CHECK: [[rhs_bc:%.+]] = f8e8m0fnu[4,4,4,32,2,4]{5,4,3,2,1,0} bitcast({{.+}})
  CHECK: [[rhs_tr:%.+]] = f8e8m0fnu[4,4,2,32,4,4]{5,4,3,2,1,0} transpose([[rhs_bc]]), dimensions={0,1,4,3,2,5}
  CHECK: {{.+}} = f8e8m0fnu[4,512,8]{2,1,0} bitcast([[rhs_tr]])
  // %fusion
  CHECK: [[lhs:%.+]] = f8e4m3fn[4,384,256]{2,1,0} parameter(0)
  CHECK: [[rhs:%.+]] = f8e4m3fn[4,512,256]{2,1,0} parameter(1)
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[4,384,8]{2,1,0} parameter(2)
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[4,512,8]{2,1,0} parameter(3)
  CHECK: {{.+}} = f32[4,384,512]{2,1,0} scaled-dot([[lhs]], [[rhs]], [[lhs_scale]], [[rhs_scale]])
)";
  EXPECT_THAT(RunFileCheck(test_module->ToString(), expected),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(BlockScalingRewriterTest, CudnnFusionSwizzlePadContracting) {
  constexpr absl::string_view hlo_string = R"(
fusion {
  %lhs = f8e4m3fn[4,384,224] parameter(0)
  %rhs = f8e4m3fn[4,512,224] parameter(1)
  %lhs_scale = f8e8m0fnu[4,384,7] parameter(2)
  %rhs_scale = f8e8m0fnu[4,512,7] parameter(3)
  ROOT %result = f32[4,384,512] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  %lhs = f8e4m3fn[4,384,224] parameter(0)
  %rhs = f8e4m3fn[4,512,224] parameter(1)
  %lhs_scale = f8e8m0fnu[4,384,7] parameter(2)
  %rhs_scale = f8e8m0fnu[4,512,7] parameter(3)
  ROOT %result = f32[4,384,512] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=fusion,
      backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(CudnnScaledDotHelper::AddScaleSwizzle(Cast<HloFusionInstruction>(
      test_module->entry_computation()->root_instruction())));

  constexpr absl::string_view expected = R"(
  // %lhs_scale.1_swizzle.1
  CHECK: [[lhs_pad:%.+]] = f8e8m0fnu[4,384,8]{2,1,0} pad({{.+}}, {{.+}}), padding=0_0x0_0x0_1
  CHECK: [[lhs_bc:%.+]] = f8e8m0fnu[4,3,4,32,2,4]{5,4,3,2,1,0} bitcast([[lhs_pad]])
  CHECK: [[lhs_tr:%.+]] = f8e8m0fnu[4,3,2,32,4,4]{5,4,3,2,1,0} transpose([[lhs_bc]]), dimensions={0,1,4,3,2,5}
  CHECK: {{.+}} = f8e8m0fnu[4,384,8]{2,1,0} bitcast([[lhs_tr]])
  // %rhs_scale.1_swizzle.1
  CHECK: [[rhs_pad:%.+]] = f8e8m0fnu[4,512,8]{2,1,0} pad({{.+}}, {{.+}}), padding=0_0x0_0x0_1
  CHECK: [[rhs_bc:%.+]] = f8e8m0fnu[4,4,4,32,2,4]{5,4,3,2,1,0} bitcast([[rhs_pad]])
  CHECK: [[rhs_tr:%.+]] = f8e8m0fnu[4,4,2,32,4,4]{5,4,3,2,1,0} transpose([[rhs_bc]]), dimensions={0,1,4,3,2,5}
  CHECK: {{.+}} = f8e8m0fnu[4,512,8]{2,1,0} bitcast([[rhs_tr]])
  // %fusion
  CHECK: [[lhs:%.+]] = f8e4m3fn[4,384,224]{2,1,0} parameter(0)
  CHECK: [[rhs:%.+]] = f8e4m3fn[4,512,224]{2,1,0} parameter(1)
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[4,384,8]{2,1,0} parameter(2)
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[4,512,8]{2,1,0} parameter(3)
  CHECK: {{.+}} = f32[4,384,512]{2,1,0} scaled-dot([[lhs]], [[rhs]], [[lhs_scale]], [[rhs_scale]])
)";
  EXPECT_THAT(RunFileCheck(test_module->ToString(), expected),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(BlockScalingRewriterTest, CudnnFusionSwizzlePadNoncontracting) {
  constexpr absl::string_view hlo_string = R"(
fusion {
  %lhs = f8e4m3fn[4,320,256] parameter(0)
  %rhs = f8e4m3fn[4,448,256] parameter(1)
  %lhs_scale = f8e8m0fnu[4,320,8] parameter(2)
  %rhs_scale = f8e8m0fnu[4,448,8] parameter(3)
  ROOT %result = f32[4,320,448] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  %lhs = f8e4m3fn[4,320,256] parameter(0)
  %rhs = f8e4m3fn[4,448,256] parameter(1)
  %lhs_scale = f8e8m0fnu[4,320,8] parameter(2)
  %rhs_scale = f8e8m0fnu[4,448,8] parameter(3)
  ROOT %result = f32[4,320,448] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=fusion,
      backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(CudnnScaledDotHelper::AddScaleSwizzle(Cast<HloFusionInstruction>(
      test_module->entry_computation()->root_instruction())));

  constexpr absl::string_view expected = R"(
  // %lhs_scale.1_swizzle.1
  CHECK: [[lhs_pad:%.+]] = f8e8m0fnu[4,384,8]{2,1,0} pad({{.+}}, {{.+}}), padding=0_0x0_64x0_0
  CHECK: [[lhs_bc:%.+]] = f8e8m0fnu[4,3,4,32,2,4]{5,4,3,2,1,0} bitcast([[lhs_pad]])
  CHECK: [[lhs_tr:%.+]] = f8e8m0fnu[4,3,2,32,4,4]{5,4,3,2,1,0} transpose([[lhs_bc]]), dimensions={0,1,4,3,2,5}
  CHECK: {{.+}} = f8e8m0fnu[4,384,8]{2,1,0} bitcast([[lhs_tr]])
  // %rhs_scale.1_swizzle.1
  CHECK: [[rhs_pad:%.+]] = f8e8m0fnu[4,512,8]{2,1,0} pad({{.+}}, {{.+}}), padding=0_0x0_64x0_0
  CHECK: [[rhs_bc:%.+]] = f8e8m0fnu[4,4,4,32,2,4]{5,4,3,2,1,0} bitcast([[rhs_pad]])
  CHECK: [[rhs_tr:%.+]] = f8e8m0fnu[4,4,2,32,4,4]{5,4,3,2,1,0} transpose([[rhs_bc]]), dimensions={0,1,4,3,2,5}
  CHECK: {{.+}} = f8e8m0fnu[4,512,8]{2,1,0} bitcast([[rhs_tr]])
  // %fusion
  CHECK: [[lhs:%.+]] = f8e4m3fn[4,320,256]{2,1,0} parameter(0)
  CHECK: [[rhs:%.+]] = f8e4m3fn[4,448,256]{2,1,0} parameter(1)
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[4,384,8]{2,1,0} parameter(2)
  CHECK: [[lhs_slice:%.+]] = f8e8m0fnu[4,320,8]{2,1,0} slice([[lhs_scale]]), slice={[0:4], [0:320], [0:8]}
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[4,512,8]{2,1,0} parameter(3)
  CHECK: [[rhs_slice:%.+]] = f8e8m0fnu[4,448,8]{2,1,0} slice([[rhs_scale]]), slice={[0:4], [0:448], [0:8]}
  CHECK: {{.+}} = f32[4,320,448]{2,1,0} scaled-dot([[lhs]], [[rhs]], [[lhs_slice]], [[rhs_slice]])
)";
  EXPECT_THAT(RunFileCheck(test_module->ToString(), expected),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla::gpu
