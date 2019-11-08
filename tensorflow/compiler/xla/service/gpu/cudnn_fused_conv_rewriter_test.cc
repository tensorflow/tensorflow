/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_conv_rewriter.h"

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

class CudnnFusedConvRewriterTest : public GpuCodegenTest {
 protected:
  string GetOptimizedHlo(absl::string_view hlo_string) {
    return backend()
        .compiler()
        ->RunHloPasses(
            ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest())
                .ConsumeValueOrDie(),
            backend().default_stream_executor(), backend().memory_allocator())
        .ConsumeValueOrDie()
        ->ToString();
  }

  void TestMatchWithAllTypes(absl::string_view hlo_string) {
    for (absl::string_view type : {"f16", "f32", "f64"}) {
      const string hlo_with_new_type =
          absl::StrReplaceAll(hlo_string, {{"TYPE", type}});
      string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
      EXPECT_THAT(optimized_hlo_string,
                  Not(HasSubstr(kCudnnConvForwardCallTarget)));
      EXPECT_THAT(optimized_hlo_string,
                  HasSubstr(kCudnnConvBiasActivationForwardCallTarget));
      EXPECT_TRUE(RunAndCompare(hlo_with_new_type, ErrorSpec{0.01}))
          << optimized_hlo_string;
    }
  }

  void TestClamp(absl::string_view pre_hlo_string,
                 absl::string_view post_hlo_string) {
    string alpha_conv_scalar, alpha_side_input_scalar;
    string elementwise_type;

    string optimized_hlo_string = GetOptimizedHlo(pre_hlo_string);
    EXPECT_THAT(optimized_hlo_string, Not(HasSubstr("Convert")));
    EXPECT_THAT(optimized_hlo_string, HasSubstr("__cudnn$conv"));
    EXPECT_TRUE(RunAndCompare(pre_hlo_string, ErrorSpec{0.01}))
        << pre_hlo_string;
    MatchOptimizedHlo(pre_hlo_string, post_hlo_string);
  }

  void TestNotMatchWithAllTypes(absl::string_view hlo_string) {
    for (absl::string_view type : {"f16", "f32", "f64"}) {
      const string hlo_with_new_type =
          absl::StrReplaceAll(hlo_string, {{"TYPE", type}});
      string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
      EXPECT_THAT(optimized_hlo_string, HasSubstr(kCudnnConvForwardCallTarget));
      EXPECT_THAT(optimized_hlo_string,
                  Not(HasSubstr(kCudnnConvBiasActivationForwardCallTarget)));
    }
  }
};

TEST_F(CudnnFusedConvRewriterTest, TestConvOnly) {
  // max(0, conv(x, w));
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,32,9,9] broadcast(zero), dimensions={}

      input = TYPE[1,17,9,9] parameter(0)
      filter = TYPE[3,3,17,32] parameter(1)

      conv = TYPE[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      ROOT relu = TYPE[1,32,9,9] maximum(zeros, conv)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestBias) {
  // max(0, conv(x, w) + bias);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      bias = TYPE[64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = TYPE[1,3,3,64] add(conv, broadcasted_bias)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestSideInputOnly) {
  // max(0, conv(x, w) + side_input);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      add1 = TYPE[1,3,3,64] add(conv, side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestBiasAndSideInput) {
  // max(0, conv(x, w) + side_input + bias);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)
      bias = TYPE[64] parameter(3)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = TYPE[1,3,3,64] add(conv, broadcasted_bias)
      add2 = TYPE[1,3,3,64] add(add1, side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add2)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestScaledConv) {
  // max(0, 0.999994934 * conv(x, w));
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,32,9,9] broadcast(zero), dimensions={}
      alpha_conv_scalar = TYPE[] constant(0.999994934)

      input = TYPE[1,17,9,9] parameter(0)
      filter = TYPE[3,3,17,32] parameter(1)

      conv = TYPE[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      alpha_conv = TYPE[1,32,9,9] broadcast(alpha_conv_scalar), dimensions={}
      scaled_conv = TYPE[1,32,9,9] multiply(conv, alpha_conv)
      ROOT relu = TYPE[1,32,9,9] maximum(zeros, scaled_conv)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestNoCrashOnInf) {
  EXPECT_TRUE(RunAndCompare(R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(inf)
      zeros = f32[1,32,9,9] broadcast(zero), dimensions={}
      alpha_conv_scalar = f32[] constant(0.999994934)

      input = f32[1,17,9,9] parameter(0)
      filter = f32[3,3,17,32] parameter(1)

      conv = f32[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      alpha_conv = f32[1,32,9,9] broadcast(alpha_conv_scalar), dimensions={}
      scaled_conv = f32[1,32,9,9] multiply(conv, alpha_conv)
      ROOT relu = f32[1,32,9,9] maximum(zeros, scaled_conv)
    })",
                            ErrorSpec{0.01}));
}

TEST_F(CudnnFusedConvRewriterTest, TestScaledConvAndSideInput) {
  // max(0, conv(x, w) + 0.899994934 * side_input);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_side_input_scalar = TYPE[] constant(0.899994934)
      alpha_side_input = TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      scaled_side_input = TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      add1 = TYPE[1,3,3,64] add(conv, scaled_side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestScaledConvAndScaledSideInput) {
  // max(0, 0.999994934 * conv(x, w) + 0.899994934 * side_input);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = TYPE[] constant(0.999994934)
      alpha_conv = TYPE[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = TYPE[] constant(0.899994934)
      alpha_side_input = TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      scaled_conv = TYPE[1,3,3,64] multiply(conv, alpha_conv)
      scaled_side_input = TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      add1 = TYPE[1,3,3,64] add(scaled_conv, scaled_side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestScaledConvAndScaledSideInputWithBias) {
  // max(0, 0.999994934 * conv(x, w) + 0.899994934 * side_input + bias);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = TYPE[] constant(0.999994934)
      alpha_conv = TYPE[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = TYPE[] constant(0.899994934)
      alpha_side_input = TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)
      bias = TYPE[64] parameter(3)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      scaled_conv = TYPE[1,3,3,64] multiply(conv, alpha_conv)
      scaled_side_input = TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = TYPE[1,3,3,64] add(scaled_conv, broadcasted_bias)
      add2 = TYPE[1,3,3,64] add(add1, scaled_side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add2)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestMatchMaxZeroOnly) {
  // max(0.1, conv(x, w)) shouldn't match.
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      point_one = TYPE[] constant(0.1)
      point_ones = TYPE[1,32,9,9] broadcast(point_one), dimensions={}

      input = TYPE[1,17,9,9] parameter(0)
      filter = TYPE[3,3,17,32] parameter(1)

      conv = TYPE[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      ROOT relu = TYPE[1,32,9,9] maximum(point_ones, conv)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestMatchBroadcastedBiasOnly) {
  // max(0, conv(x, w) + side_input1 + side_input2) shouldn't match.
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input1 = TYPE[1,3,3,64] parameter(2)
      side_input2 = TYPE[1,3,3,64] parameter(3)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      add1 = TYPE[1,3,3,64] add(conv, side_input2)
      add2 = TYPE[1,3,3,64] add(add1, side_input1)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add2)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, PreservesMetadata) {
  const char* kHloString = R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,32,9,9] broadcast(zero), dimensions={}

      input = f32[1,17,9,9] parameter(0)
      filter = f32[3,3,17,32] parameter(1)

      conv = f32[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1, metadata={op_type="foo"}
      ROOT relu = f32[1,32,9,9] maximum(zeros, conv)
    })";

  const string optimized_hlo_string =
      backend()
          .compiler()
          ->RunHloPasses(
              ParseAndReturnVerifiedModule(kHloString, GetModuleConfigForTest())
                  .ConsumeValueOrDie(),
              backend().default_stream_executor(), backend().memory_allocator())
          .ConsumeValueOrDie()
          ->ToString();
  EXPECT_THAT(
      optimized_hlo_string,
      ::testing::ContainsRegex(R"(custom-call.*metadata=\{op_type="foo"\})"));
}

TEST_F(CudnnFusedConvRewriterTest, TestPreservesFeatureGroupCount) {
  // The convolution below would crash if feature_count is not preserved.
  const char* kHloString = R"(
    HloModule jaxpr_computation__6.19

    primitive_computation__1.4 {
      parameter.5 = f32[] parameter(0)
      parameter.6 = f32[] parameter(1)
      ROOT add.7 = f32[] add(parameter.5, parameter.6)
    }

    ENTRY jaxpr_computation__7.8 {
      parameter.11 = f32[2,64,64,53]{3,2,1,0} parameter(1)
      parameter.10 = f32[3,3,1,53]{3,2,1,0} parameter(0)
      convolution.12 = f32[2,64,64,53]{3,2,1,0} convolution(parameter.11, parameter.10), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=53
      constant.13 = f32[] constant(0)
      broadcast.14 = f32[2,64,64,53]{3,2,1,0} broadcast(constant.13), dimensions={}
      maximum.15 = f32[2,64,64,53]{3,2,1,0} maximum(convolution.12, broadcast.14)
      ROOT reduce.17 = f32[] reduce(maximum.15, constant.13), dimensions={0,1,2,3}, to_apply=primitive_computation__1.4
    }
  )";
  EXPECT_TRUE(RunAndCompare(kHloString, ErrorSpec{0.01}));
}

TEST_F(CudnnFusedConvRewriterTest, TestConvInt8ToInt8) {
  // max(0, clamp(conv(x, w)))); for int8
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = s8[] constant(0)
      zeros = s8[1,32,9,9] broadcast(zero), dimensions={}

      input = s8[1,17,9,9] parameter(0)
      filter = s8[3,3,17,32] parameter(1)

      inputs32 = s32[1,17,9,9] convert(input)
      filters32 = s32[3,3,17,32] convert(filter)

      conv = s32[1,32,9,9] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1

      lower = s32[] constant(-128)
      lowers = s32[1,32,9,9] broadcast(lower), dimensions={}
      upper = s32[] constant(127)
      uppers = s32[1,32,9,9] broadcast(upper), dimensions={}

      clamp = s32[1,32,9,9] clamp(lowers, conv, uppers)

      convert = s8[1,32,9,9] convert(clamp)
      ROOT relu = s8[1,32,9,9] maximum(zeros, convert)
    })",
      // post_hlo
      R"(
      ; CHECK-LABEL: ENTRY %Test (input: s8[1,17,9,9], filter: s8[3,3,17,32]) -> s8[1,32,9,9] {
      ; CHECK:  %custom-call{{(\.[0-9])?}} = (s8[1,32,9,9]{1,3,2,0}, u8[{{[0-9]*}}]{0}) custom-call(%fusion{{(\.[0-9])?}}, %fusion{{(\.[0-9])?}}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convForward", backend_config=
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvInt8ToFloat) {
  // convert<float>(conv<int32>(convert<int32>(int8_x),
  // convert<int32>(int8_w)));
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      input = s8[1,17,9,9] parameter(0)
      filter = s8[3,3,17,32] parameter(1)

      inputs32 = s32[1,17,9,9] convert(input)
      filters32 = s32[3,3,17,32] convert(filter)

      conv = s32[1,32,9,9] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1

      ROOT convert = f32[1,32,9,9] convert(conv)
    })",
      // post_hlo
      R"(
      ; CHECK-LABEL: ENTRY %Test (input: s8[1,17,9,9], filter: s8[3,3,17,32]) -> f32[1,32,9,9] {
      ; CHECK:  %custom-call{{(\.[0-9])?}} = (f32[1,32,9,9]{1,3,2,0}, u8[{{[0-9]+}}]{0}) custom-call(%fusion{{(\.[0-9])?}}, %fusion{{(\.[0-9])?}}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convForward", backend_config=
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestFusedConvInt8ToInt8) {
  // clamp(max(0, conv(x, w)+bias)); for int8
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,3,3,64] broadcast(zero), dimensions={}

      input = s8[1,3,3,64] parameter(0)
      filter = s8[3,3,64,64] parameter(1)
      bias = f32[64] parameter(2)

      inputs32 = s32[1,3,3,64] convert(input)
      filters32 = s32[3,3,64,64] convert(filter)

      conv = s32[1,3,3,64] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1

      convfloat = f32[1,3,3,64] convert(conv)
      broadcasted_bias = f32[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = f32[1,3,3,64] add(convfloat, broadcasted_bias)
      relu = f32[1,3,3,64] maximum(zeros, add1)

      lower = f32[] constant(-128)
      lowers = f32[1,3,3,64] broadcast(lower), dimensions={}
      upper = f32[] constant(127)
      uppers = f32[1,3,3,64] broadcast(upper), dimensions={}

      clamp = f32[1,3,3,64] clamp(lowers, relu, uppers)

      ROOT convert = s8[1,3,3,64] convert(clamp)      
    })",
      // post_hlo
      R"(
      ; CHECK-LABEL: ENTRY %Test (input: s8[1,3,3,64], filter: s8[3,3,64,64], bias: f32[64]) -> s8[1,3,3,64]
      ; CHECK:  %custom-call{{(\.[0-9])?}} = (s8[1,3,3,64]{3,2,1,0}, u8[{{[0-9]+}}]{0}) custom-call(%input, %copy{{(\.[0-9])?}}, %bias), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convBiasActivationForward", backend_config=
      ; CHECK-NEXT:  ROOT %get-tuple-element{{(\.[0-9])?}} = s8[1,3,3,64]{3,2,1,0} get-tuple-element(%custom-call{{(\.[0-9])?}}), index=0
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestFusedConvInt8ToFloat) {
  // max(0, convert<float>(conv<int32>(int8_x),
  // conv<int32>(int8_w))+float_bias)); int8 to float via bias.
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,3,3,64] broadcast(zero), dimensions={}

      input = s8[1,3,3,64] parameter(0)
      filter = s8[3,3,64,64] parameter(1)
      bias = f32[64] parameter(2)

      inputs32 = s32[1,3,3,64] convert(input)
      filters32 = s32[3,3,64,64] convert(filter)

      conv = s32[1,3,3,64] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1

      convfloat = f32[1,3,3,64] convert(conv)
      broadcasted_bias = f32[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = f32[1,3,3,64] add(convfloat, broadcasted_bias)
      ROOT relu = f32[1,3,3,64] maximum(zeros, add1)     
    })",
      // post_hlo
      R"(
      ; CHECK-LABEL: ENTRY %Test (input: s8[1,3,3,64], filter: s8[3,3,64,64], bias: f32[64]) -> f32[1,3,3,64] {
      ; CHECK:  %custom-call{{(\.[0-9])?}} = (f32[1,3,3,64]{3,2,1,0}, u8[{{[0-9]*}}]{0}) custom-call(%input, %copy{{(\.[0-9])?}}, %bias), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convBiasActivationForward", backend_config=
      ; CHECK-NEXT:  ROOT %get-tuple-element{{(\.[0-9])?}} = f32[1,3,3,64]{3,2,1,0} get-tuple-element(%custom-call{{(\.[0-9])?}}), index=0
      )");
}

TEST_F(CudnnFusedConvRewriterTest,
       TestFusedConvWithScaledInt8SideInputBiasInt8ToInt8) {
  // clamp(max(0, alpha_conv * conv(x, w) + alpha_side *
  // convert<int32>(int8_side_input) + bias)); for int8
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = f32[] constant(0.999994934)
      alpha_conv = f32[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = f32[] constant(0.899994934)
      alpha_side_input = f32[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = s8[1,3,3,64] parameter(0)
      filter = s8[3,3,64,64] parameter(1)
      side_input = s8[1,3,3,64] parameter(2)
      bias = f32[64] parameter(3)

      inputs32 = s32[1,3,3,64] convert(input)
      filters32 = s32[3,3,64,64] convert(filter)
      side_input_f32 = f32[1,3,3,64] convert(side_input)

      conv = s32[1,3,3,64] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1

      convfloat = f32[1,3,3,64] convert(conv)
      scaled_conv = f32[1,3,3,64] multiply(convfloat, alpha_conv)
      scaled_side_input = f32[1,3,3,64] multiply(side_input_f32, alpha_side_input)
      broadcasted_bias = f32[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = f32[1,3,3,64] add(scaled_conv, broadcasted_bias)
      add2 = f32[1,3,3,64] add(add1, scaled_side_input)
      relu = f32[1,3,3,64] maximum(zeros, add2)

      lower = f32[] constant(-128)
      lowers = f32[1,3,3,64] broadcast(lower), dimensions={}
      upper = f32[] constant(127)
      uppers = f32[1,3,3,64] broadcast(upper), dimensions={}

      clamp = f32[1,3,3,64] clamp(lowers, relu, uppers)

      ROOT convert = s8[1,3,3,64] convert(clamp) 
    })",
      // post_hlo
      R"(
      ; CHECK-LABEL: ENTRY %Test (input: s8[1,3,3,64], filter: s8[3,3,64,64], side_input: s8[1,3,3,64], bias: f32[64]) -> s8[1,3,3,64] {
      ; CHECK:  %custom-call{{(\.[0-9])?}} = (s8[1,3,3,64]{3,2,1,0}, u8[{{[0-9]+}}]{0}) custom-call(%input, %copy{{(\.[0-9])?}}, %bias, %side_input), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convBiasActivationForward", backend_config=
      ; CHECK-NEXT:  ROOT %get-tuple-element{{(\.[0-9])?}} = s8[1,3,3,64]{3,2,1,0} get-tuple-element(%custom-call{{(\.[0-9])?}}), index=0
      )");
}

TEST_F(CudnnFusedConvRewriterTest,
       TestFusedConvWithScaledFloatSideInputBiasInt8ToInt8) {
  // From:
  // convert<int8>(clamp(max(0, alpha_conv * conv(x, w) + alpha_side *
  // float_side_input + bias))); To: convert<int8>(clamp(conv(int8_x, int8_w,
  // float_alpha_side, float_side_input, float_bias)));
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = f32[] constant(0.999994934)
      alpha_conv = f32[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = f32[] constant(0.899994934)
      alpha_side_input = f32[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = s8[1,3,3,64] parameter(0)
      filter = s8[3,3,64,64] parameter(1)
      side_input = f32[1,3,3,64] parameter(2)
      bias = f32[64] parameter(3)

      inputs32 = s32[1,3,3,64] convert(input)
      filters32 = s32[3,3,64,64] convert(filter)

      conv = s32[1,3,3,64] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1

      convfloat = f32[1,3,3,64] convert(conv)
      scaled_conv = f32[1,3,3,64] multiply(convfloat, alpha_conv)
      scaled_side_input = f32[1,3,3,64] multiply(side_input, alpha_side_input)
      broadcasted_bias = f32[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = f32[1,3,3,64] add(scaled_conv, broadcasted_bias)
      add2 = f32[1,3,3,64] add(add1, scaled_side_input)
      relu = f32[1,3,3,64] maximum(zeros, add2)

      lower = f32[] constant(-128)
      lowers = f32[1,3,3,64] broadcast(lower), dimensions={}
      upper = f32[] constant(127)
      uppers = f32[1,3,3,64] broadcast(upper), dimensions={}

      clamp = f32[1,3,3,64] clamp(lowers, relu, uppers)

      ROOT convert = s8[1,3,3,64] convert(clamp) 
    })",
      //  post_hlo
      R"(
      ; CHECK-LABEL: ENTRY %Test (input: s8[1,3,3,64], filter: s8[3,3,64,64], side_input: f32[1,3,3,64], bias: f32[64]) -> s8[1,3,3,64] {
      ; CHECK:  %custom-call{{(\.[0-9])?}} = (f32[1,3,3,64]{3,2,1,0}, u8[{{[0-9]+}}]{0}) custom-call(%input, %copy{{(\.[0-9])?}}, %bias, %side_input), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convBiasActivationForward", backend_config=
      ; CHECK:  ROOT %fusion = s8[1,3,3,64]{3,2,1,0} fusion(%get-tuple-element{{(\.[0-9])?}}), kind=kLoop, calls=%fused_computation
      )");
}

TEST_F(CudnnFusedConvRewriterTest,
       TestFusedConvWithScaledInt8SideInputBiasInt8ToFloat) {
  // From:
  // clamp(max(0, alpha_conv * conv(x, w) + alpha_side *
  // convert<float>(int8_side_input) + bias)); To: clamp(conv(int8_x, int8_w,
  // float_alpha_side, convert<float>(int8_side_input), float_bias));
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = f32[] constant(0.999994934)
      alpha_conv = f32[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = f32[] constant(0.899994934)
      alpha_side_input = f32[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = s8[1,3,3,64] parameter(0)
      filter = s8[3,3,64,64] parameter(1)
      side_input = s8[1,3,3,64] parameter(2)
      bias = f32[64] parameter(3)

      inputs32 = s32[1,3,3,64] convert(input)
      filters32 = s32[3,3,64,64] convert(filter)
      side_input_f32 = f32[1,3,3,64] convert(side_input)

      conv = s32[1,3,3,64] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1

      convfloat = f32[1,3,3,64] convert(conv)
      scaled_conv = f32[1,3,3,64] multiply(convfloat, alpha_conv)
      scaled_side_input = f32[1,3,3,64] multiply(side_input_f32, alpha_side_input)
      broadcasted_bias = f32[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = f32[1,3,3,64] add(scaled_conv, broadcasted_bias)
      add2 = f32[1,3,3,64] add(add1, scaled_side_input)
      relu = f32[1,3,3,64] maximum(zeros, add2)

      lower = f32[] constant(-128)
      lowers = f32[1,3,3,64] broadcast(lower), dimensions={}
      upper = f32[] constant(127)
      uppers = f32[1,3,3,64] broadcast(upper), dimensions={}

      ROOT clamp = f32[1,3,3,64] clamp(lowers, relu, uppers)
    })",
      // post_hlo
      R"(
      ; CHECK-LABEL: ENTRY %Test (input: s8[1,3,3,64], filter: s8[3,3,64,64], side_input: s8[1,3,3,64], bias: f32[64]) -> f32[1,3,3,64] {
      ; CHECK:  %side_input_f32 = f32[1,3,3,64]{3,2,1,0} convert(%side_input)
      ; CHECK:  %custom-call{{(\.[0-9])?}} = (f32[1,3,3,64]{3,2,1,0}, u8[{{[0-9]*}}]{0}) custom-call(%input, %copy{{(\.[0-9])?}}, %bias, %side_input_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convBiasActivationForward", backend_config=
      ; CHECK:  ROOT %fusion = f32[1,3,3,64]{3,2,1,0} fusion(%get-tuple-element{{(\.[0-9])?}}), kind=kLoop, calls=%fused_computation
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvInt8ToInt8NoClamp) {
  // Check that integer convolution without clamp to int8 is not allowed.
  // convert<int8>(custom_call<int32>(int32_x, int32_w,
  // cudnnConvolutionForward))
  const string module_str = absl::StrFormat(R"(
    HloModule Test

    ENTRY Test (input: s8[1,17,9,9], filter: s8[3,3,17,32]) -> s8[1,32,9,9] {
      zero = s8[] constant(0)
      zeros = s8[1,32,9,9]{3,2,1,0} broadcast(s8[] zero), dimensions={}
      input = s8[1,17,9,9]{3,2,1,0} parameter(0)
      filter = s8[3,3,17,32]{3,2,1,0} parameter(1)
      custom-call = (s32[1,32,9,9]{3,2,1,0}, u8[0]{0}) custom-call(s8[1,17,9,9]{3,2,1,0} input, s8[3,3,17,32]{3,2,1,0} filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convForward", backend_config="{\"convResultScale\":1}"
      get-tuple-element = s32[1,32,9,9]{3,2,1,0} get-tuple-element((s32[1,32,9,9]{3,2,1,0}, u8[0]{0}) custom-call), index=0
      convert = s8[1,32,9,9]{3,2,1,0} convert(s32[1,32,9,9]{3,2,1,0} get-tuple-element)
      ROOT relu = s8[1,32,9,9]{3,2,1,0} maximum(s8[1,32,9,9]{3,2,1,0} zeros, s8[1,32,9,9]{3,2,1,0} convert)
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ASSERT_FALSE(CudnnFusedConvRewriter().Run(m.get()).ok());
}

TEST_F(CudnnFusedConvRewriterTest, TestFusedConvInt8ToInt8NoClamp) {
  // Although bias and so on are fused with forward convolution,
  // it is still not allowed if the output is not clampped/converted to int8
  // max(0, alpha_conv * conv(x, w) + alpha_side * side_input + bias); for int8

  const string module_str = absl::StrFormat(R"(
    HloModule Test

    ENTRY Test (input: s8[1,17,9,9], filter: s8[3,3,17,32]) -> s8[1,32,9,9] {
      zero = s8[] constant(0)
      zeros = s8[1,32,9,9]{3,2,1,0} broadcast(s8[] zero), dimensions={}
      input = s8[1,17,9,9]{3,2,1,0} parameter(0)
      filter = s8[3,3,17,32]{3,2,1,0} parameter(1)
      custom-call = (s32[1,32,9,9]{3,2,1,0}, u8[0]{0}) custom-call(s8[1,17,9,9]{3,2,1,0} input, s8[3,3,17,32]{3,2,1,0} filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convForward", backend_config="{\"convResultScale\":1}"
      get-tuple-element = s32[1,32,9,9]{3,2,1,0} get-tuple-element((s32[1,32,9,9]{3,2,1,0}, u8[0]{0}) custom-call), index=0
      convert = s8[1,32,9,9]{3,2,1,0} convert(s32[1,32,9,9]{3,2,1,0} get-tuple-element)
      ROOT relu = s8[1,32,9,9]{3,2,1,0} maximum(s8[1,32,9,9]{3,2,1,0} zeros, s8[1,32,9,9]{3,2,1,0} convert)
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ASSERT_FALSE(CudnnFusedConvRewriter().Run(m.get()).ok());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
