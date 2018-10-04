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
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class CudnnFusedConvolutionRewriterTest : public HloTestBase {
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
    for (absl::string_view type : {"f16", "f32", "f64"}) {
      const string hlo_with_new_type =
          absl::StrReplaceAll(hlo_string, {{"TYPE", type}});
      const string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
      EXPECT_EQ(absl::string_view::npos,
                optimized_hlo_string.find("__cudnn$convForward"))
          << optimized_hlo_string;
      EXPECT_NE(absl::string_view::npos,
                optimized_hlo_string.find("__cudnn$convBiasActivationForward"))
          << optimized_hlo_string;
      EXPECT_TRUE(RunAndCompare(hlo_with_new_type, ErrorSpec{0.01}))
          << optimized_hlo_string;
    }
  }

  void TestNotMatchWithAllTypes(absl::string_view hlo_string) {
    for (absl::string_view type : {"f16", "f32", "f64"}) {
      const string hlo_with_new_type =
          absl::StrReplaceAll(hlo_string, {{"TYPE", type}});
      string optimized_hlo = GetOptimizedHlo(hlo_with_new_type);
      EXPECT_NE(absl::string_view::npos,
                optimized_hlo.find("__cudnn$convForward"))
          << optimized_hlo;
      EXPECT_EQ(absl::string_view::npos,
                optimized_hlo.find("__cudnn$convBiasActivationForward"))
          << optimized_hlo;
    }
  }
};

TEST_F(CudnnFusedConvolutionRewriterTest, TestConvOnly) {
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

TEST_F(CudnnFusedConvolutionRewriterTest, TestBias) {
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

TEST_F(CudnnFusedConvolutionRewriterTest, TestSideInputOnly) {
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

TEST_F(CudnnFusedConvolutionRewriterTest, TestBiasAndSideInput) {
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

TEST_F(CudnnFusedConvolutionRewriterTest, TestScaledConv) {
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

TEST_F(CudnnFusedConvolutionRewriterTest, TestScaledConvAndSideInput) {
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

TEST_F(CudnnFusedConvolutionRewriterTest, TestScaledConvAndScaledSideInput) {
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

TEST_F(CudnnFusedConvolutionRewriterTest,
       TestScaledConvAndScaledSideInputWithBias) {
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

TEST_F(CudnnFusedConvolutionRewriterTest, TestMatchMaxZeroOnly) {
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

TEST_F(CudnnFusedConvolutionRewriterTest, TestMatchBroadcastedBiasOnly) {
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

}  // namespace
}  // namespace gpu
}  // namespace xla
