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

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

class CudnnFusedConvRewriterTest : public HloTestBase {
 protected:
  string GetOptimizedHlo(absl::string_view hlo_string) {
    return backend()
        .compiler()
        ->RunHloPasses(ParseHloString(hlo_string, GetModuleConfigForTest())
                           .ConsumeValueOrDie(),
                       backend().default_stream_executor(),
                       backend().memory_allocator())
        .ConsumeValueOrDie()
        ->ToString();
  }

  void TestMatchWithAllTypes(absl::string_view hlo_string) {
    string alpha_conv_scalar, alpha_side_input_scalar;
    string elementwise_type;
    for (absl::string_view type : {"f16", "f32", "f64", "s8"}) {
      if(type == "s8") {
        alpha_conv_scalar = "2";
        alpha_side_input_scalar = "-3";
        elementwise_type = "f32"; 
      }
      else {
        alpha_conv_scalar = "0.999994934";
        alpha_side_input_scalar = "0.899994934";
        elementwise_type = string(type);
      }
      string hlo_resolved_string =
          absl::StrReplaceAll(hlo_string, {{"INPUT_TYPE", type}, {"ELEMENTWISE_TYPE", elementwise_type}, {"ALPHA_CONV_SCALAR", alpha_conv_scalar}, {"ALPHA_SIDE_INPUT_SCALAR", alpha_side_input_scalar}});
      string optimized_hlo_string = GetOptimizedHlo(hlo_resolved_string);
      EXPECT_THAT(optimized_hlo_string,
                  Not(HasSubstr(kCudnnConvForwardCallTarget)));
      EXPECT_THAT(optimized_hlo_string,
                  HasSubstr(kCudnnConvBiasActivationForwardCallTarget));
      EXPECT_TRUE(RunAndCompare(hlo_resolved_string, ErrorSpec{0.01}))
          << hlo_resolved_string;
    }
  }

  void TestNotMatchWithAllTypes(absl::string_view hlo_string) {
    for (absl::string_view type : {"f16", "f32", "f64"}) {
      string hlo_resolved_string =
          absl::StrReplaceAll(hlo_string, {{"INPUT_TYPE", type}, {"ELEMENTWISE_TYPE", type}, });
      string optimized_hlo_string = GetOptimizedHlo(hlo_resolved_string);
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
      zero = ELEMENTWISE_TYPE[] constant(0)
      zeros = ELEMENTWISE_TYPE[1,32,9,9] broadcast(zero), dimensions={}

      input = INPUT_TYPE[1,17,9,9] parameter(0)
      filter = INPUT_TYPE[3,3,17,32] parameter(1)

      conv = ELEMENTWISE_TYPE[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      ROOT relu = ELEMENTWISE_TYPE[1,32,9,9] maximum(zeros, conv)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestBias) {
  // max(0, conv(x, w) + bias);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = ELEMENTWISE_TYPE[] constant(0)
      zeros = ELEMENTWISE_TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = INPUT_TYPE[1,3,3,64] parameter(0)
      filter = INPUT_TYPE[3,3,64,64] parameter(1)
      bias = ELEMENTWISE_TYPE[64] parameter(2)

      conv = ELEMENTWISE_TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = ELEMENTWISE_TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = ELEMENTWISE_TYPE[1,3,3,64] add(conv, broadcasted_bias)
      ROOT relu = ELEMENTWISE_TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestSideInputOnly) {
  // max(0, conv(x, w) + side_input);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = ELEMENTWISE_TYPE[] constant(0)
      zeros = ELEMENTWISE_TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = INPUT_TYPE[1,3,3,64] parameter(0)
      filter = INPUT_TYPE[3,3,64,64] parameter(1)
      side_input = ELEMENTWISE_TYPE[1,3,3,64] parameter(2)

      conv = ELEMENTWISE_TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      add1 = ELEMENTWISE_TYPE[1,3,3,64] add(conv, side_input)
      ROOT relu = ELEMENTWISE_TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestBiasAndSideInput) {
  // max(0, conv(x, w) + side_input + bias);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = ELEMENTWISE_TYPE[] constant(0)
      zeros = ELEMENTWISE_TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = INPUT_TYPE[1,3,3,64] parameter(0)
      filter = INPUT_TYPE[3,3,64,64] parameter(1)
      side_input = ELEMENTWISE_TYPE[1,3,3,64] parameter(2)
      bias = ELEMENTWISE_TYPE[64] parameter(3)

      conv = ELEMENTWISE_TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = ELEMENTWISE_TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = ELEMENTWISE_TYPE[1,3,3,64] add(conv, broadcasted_bias)
      add2 = ELEMENTWISE_TYPE[1,3,3,64] add(add1, side_input)
      ROOT relu = ELEMENTWISE_TYPE[1,3,3,64] maximum(zeros, add2)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestScaledConv) {
  // max(0, ALPHA_CONV_SCALAR * conv(x, w));
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = ELEMENTWISE_TYPE[] constant(0)
      zeros = ELEMENTWISE_TYPE[1,32,9,9] broadcast(zero), dimensions={}
      alpha_conv_scalar = ELEMENTWISE_TYPE[] constant(ALPHA_CONV_SCALAR)

      input = INPUT_TYPE[1,17,9,9] parameter(0)
      filter = INPUT_TYPE[3,3,17,32] parameter(1)

      conv = ELEMENTWISE_TYPE[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      alpha_conv = ELEMENTWISE_TYPE[1,32,9,9] broadcast(alpha_conv_scalar), dimensions={}
      scaled_conv = ELEMENTWISE_TYPE[1,32,9,9] multiply(conv, alpha_conv)
      ROOT relu = ELEMENTWISE_TYPE[1,32,9,9] maximum(zeros, scaled_conv)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestScaledConvAndSideInput) {
  // max(0, conv(x, w) + ALPHA_SIDE_INPUT_SCALAR * side_input);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = ELEMENTWISE_TYPE[] constant(0)
      zeros = ELEMENTWISE_TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_side_input_scalar = ELEMENTWISE_TYPE[] constant(ALPHA_SIDE_INPUT_SCALAR)
      alpha_side_input = ELEMENTWISE_TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = INPUT_TYPE[1,3,3,64] parameter(0)
      filter = INPUT_TYPE[3,3,64,64] parameter(1)
      side_input = ELEMENTWISE_TYPE[1,3,3,64] parameter(2)

      conv = ELEMENTWISE_TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      scaled_side_input = ELEMENTWISE_TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      add1 = ELEMENTWISE_TYPE[1,3,3,64] add(conv, scaled_side_input)
      ROOT relu = ELEMENTWISE_TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestScaledConvAndScaledSideInput) {
  // max(0, ALPHA_CONV_SCALAR * conv(x, w) + ALPHA_SIDE_INPUT_SCALAR * side_input);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = ELEMENTWISE_TYPE[] constant(0)
      zeros = ELEMENTWISE_TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = ELEMENTWISE_TYPE[] constant(ALPHA_CONV_SCALAR)
      alpha_conv = ELEMENTWISE_TYPE[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = ELEMENTWISE_TYPE[] constant(ALPHA_SIDE_INPUT_SCALAR)
      alpha_side_input = ELEMENTWISE_TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = INPUT_TYPE[1,3,3,64] parameter(0)
      filter = INPUT_TYPE[3,3,64,64] parameter(1)
      side_input = ELEMENTWISE_TYPE[1,3,3,64] parameter(2)

      conv = ELEMENTWISE_TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      scaled_conv = ELEMENTWISE_TYPE[1,3,3,64] multiply(conv, alpha_conv)
      scaled_side_input = ELEMENTWISE_TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      add1 = ELEMENTWISE_TYPE[1,3,3,64] add(scaled_conv, scaled_side_input)
      ROOT relu = ELEMENTWISE_TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestScaledConvAndScaledSideInputWithBias) {
  // max(0, ALPHA_CONV_SCALAR * conv(x, w) + ALPHA_SIDE_INPUT_SCALAR * side_input + bias);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = ELEMENTWISE_TYPE[] constant(0)
      zeros = ELEMENTWISE_TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = ELEMENTWISE_TYPE[] constant(ALPHA_CONV_SCALAR)
      alpha_conv = ELEMENTWISE_TYPE[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = ELEMENTWISE_TYPE[] constant(ALPHA_SIDE_INPUT_SCALAR)
      alpha_side_input = ELEMENTWISE_TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = INPUT_TYPE[1,3,3,64] parameter(0)
      filter = INPUT_TYPE[3,3,64,64] parameter(1)
      side_input = ELEMENTWISE_TYPE[1,3,3,64] parameter(2)
      bias = ELEMENTWISE_TYPE[64] parameter(3)

      conv = ELEMENTWISE_TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      scaled_conv = ELEMENTWISE_TYPE[1,3,3,64] multiply(conv, alpha_conv)
      scaled_side_input = ELEMENTWISE_TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      broadcasted_bias = ELEMENTWISE_TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = ELEMENTWISE_TYPE[1,3,3,64] add(scaled_conv, broadcasted_bias)
      add2 = ELEMENTWISE_TYPE[1,3,3,64] add(add1, scaled_side_input)
      ROOT relu = ELEMENTWISE_TYPE[1,3,3,64] maximum(zeros, add2)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestMatchMaxZeroOnly) {
  // max(1, conv(x, w)) shouldn't match.
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      one = ELEMENTWISE_TYPE[] constant(1)
      ones = ELEMENTWISE_TYPE[1,32,9,9] broadcast(one), dimensions={}

      input = INPUT_TYPE[1,17,9,9] parameter(0)
      filter = INPUT_TYPE[3,3,17,32] parameter(1)

      conv = ELEMENTWISE_TYPE[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      ROOT relu = ELEMENTWISE_TYPE[1,32,9,9] maximum(ones, conv)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestMatchBroadcastedBiasOnly) {
  // max(0, conv(x, w) + side_input1 + side_input2) shouldn't match.
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = ELEMENTWISE_TYPE[] constant(0)
      zeros = ELEMENTWISE_TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = INPUT_TYPE[1,3,3,64] parameter(0)
      filter = INPUT_TYPE[3,3,64,64] parameter(1)
      side_input1 = ELEMENTWISE_TYPE[1,3,3,64] parameter(2)
      side_input2 = ELEMENTWISE_TYPE[1,3,3,64] parameter(3)

      conv = ELEMENTWISE_TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      add1 = ELEMENTWISE_TYPE[1,3,3,64] add(conv, side_input2)
      add2 = ELEMENTWISE_TYPE[1,3,3,64] add(add1, side_input1)
      ROOT relu = ELEMENTWISE_TYPE[1,3,3,64] maximum(zeros, add2)
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
          ->RunHloPasses(ParseHloString(kHloString, GetModuleConfigForTest())
                             .ConsumeValueOrDie(),
                         backend().default_stream_executor(),
                         backend().memory_allocator())
          .ConsumeValueOrDie()
          ->ToString();
  EXPECT_THAT(
      optimized_hlo_string,
      ::testing::ContainsRegex(R"(custom-call.*metadata=\{op_type="foo"\})"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
