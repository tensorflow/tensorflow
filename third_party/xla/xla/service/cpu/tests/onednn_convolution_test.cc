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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <utility>

#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/service/cpu/onednn_contraction_rewriter.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/cpu_info.h"

namespace xla {
namespace cpu {

class ConvolutionTest : public HloTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_use_thunk_runtime(false);
    return debug_options;
  }

  const char* conv_rewrite_str_ = R"(
    ; CHECK:     custom_call_target="__onednn$convolution",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "onednn_conv_config":{
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";

  const char* conv_rewrite_bias_str_ = R"(
    ; CHECK:     custom_call_target="__onednn$convolution",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "onednn_conv_config":{
    ; CHECK-DAG:       "fusions":{
    ; CHECK-DAG:         "ops":["BIAS"]
    ; CHECK-DAG:     }
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";

  const char* fused_convolution_binary_add_ = R"(
    ; CHECK:     custom_call_target="__onednn$convolution",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "onednn_conv_config":{
    ; CHECK-DAG:       "fusions":{
    ; CHECK-DAG:         "ops":["BINARY_ADD"]
    ; CHECK-DAG:     }
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";
};

TEST_F(ConvolutionTest, Simple2DTestF32) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.f32

  ENTRY convolution.test.f32 {
    arg.0 = f32[1,22,22,1] parameter(0)
    reshape.0 = f32[1,22,22,1] reshape(arg.0)
    arg.1 = f32[8,8,1,1] parameter(1)
    reshape.1 = f32[8,8,1,1] reshape(arg.1)
    convolution.0 = f32[1,11,11,1] convolution(reshape.0, reshape.1), window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    reshape.2 = f32[1,11,11,1] reshape(convolution.0)
    tuple.0 = (f32[1,11,11,1]) tuple(reshape.2)
    ROOT get-tuple-element.0 = f32[1,11,11,1] get-tuple-element(tuple.0), index=0
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_);
}

TEST_F(ConvolutionTest, Simple3DTestBF16) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* convolution_module_str = R"(
  HloModule convolution.test.bf16

  ENTRY convolution.test.bf16 {
    p0 = bf16[8,4,5,5,1] parameter(0)
    p1 = bf16[3,3,3,1,32] parameter(1)
    ROOT conv = bf16[8,4,5,5,32] convolution(p0, p1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
})";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_);
}

TEST_F(ConvolutionTest, Simple2DTestF16) {
  if (!IsSupportedType(PrimitiveType::F16)) {
    GTEST_SKIP() << "CPU does not support F16.";
  }

  const char* convolution_module_str = R"(
  HloModule convolution.test.f16

  ENTRY convolution.test.bf16 {
    p0 = f16[8,4,5,5,1] parameter(0)
    p1 = f16[3,3,3,1,32] parameter(1)
    ROOT conv = f16[8,4,5,5,32] convolution(p0, p1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
})";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_);
}

TEST_F(ConvolutionTest, Conv3DWithBiasBF16) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.with.bias.relu.bf16.3D

  ENTRY TestComputation {
    arg.0 = bf16[15,4,5,5,28] parameter(0)
    arg.1 = bf16[3,3,3,28,64] parameter(1)
    conv = bf16[15,4,5,5,64] convolution(arg.0, arg.1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    bias = bf16[64] parameter(2)
    broadcasted_bias = bf16[15,4,5,5,64] broadcast(bias), dimensions={4}
    ROOT add = bf16[15,4,5,5,64] add(conv, broadcasted_bias)
})";
  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{0.01, 0.01}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_str_);
}

TEST_F(ConvolutionTest, SimpleTestF32WithBinaryAddFusion1) {
  const char* convolution_module_str = R"(
  HloModule conv.binaryadd.test.f32

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[1,22,22,1] parameter(0)
    constant.3 = f32[] constant(1)
    broadcast.4 = f32[8,8,1,1] broadcast(constant.3), dimensions={}
    convolution.0 = f32[1,11,11,1] convolution(arg0.1, broadcast.4), window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[1] broadcast(constant.5), dimensions={}
    broadcast.9 = f32[1,11,11,1] broadcast(broadcast.6), dimensions={3}
    ROOT add.10 = f32[1,11,11,1] add(convolution.0, broadcast.9)
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, fused_convolution_binary_add_);
}

// This test should match BIAS + Residual Add when the residual add fusion is
// re-enabled.
TEST_F(ConvolutionTest, SimpleTestBF16WithBiasAndAddFusion) {
  const char* convolution_module_str = R"(
  HloModule convolution.add.test.bf16

  ENTRY convolution.add.test.bf16 {
    arg0.1 = bf16[1,22,22,1] parameter(0)
    arg0.2 = bf16[8,8,1,10] parameter(1)
    convolution.0 = bf16[1,11,11,10] convolution(arg0.1, arg0.2), window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    const.0 = bf16[10] constant(15)
    bcast.1 = bf16[1,11,11,10] broadcast(const.0), dimensions={3}
    add.0 = bf16[1,11,11,10] add(convolution.0, bcast.1)
    const.1 = bf16[1,11,11,10] constant({...})
    ROOT add.1 = bf16[1,11,11,10] add(add.0, const.1)
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_str_);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
