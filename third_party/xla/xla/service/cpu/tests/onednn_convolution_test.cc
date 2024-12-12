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

#include "absl/strings/str_replace.h"
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

class ConvolutionTest : public HloTestBase,
                        public ::testing::WithParamInterface<PrimitiveType> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_use_thunk_runtime(false);
    return debug_options;
  }

  PrimitiveType dtype_;
  std::string dtypeString_;
  bool user_scratchpad_;
  bool weights_prepacked_;
  float atol_;
  float rtol_;

  constexpr static const char* kConvRewriteStr = R"(
    ; CHECK:     custom_call_target="__onednn$convolution",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "onednn_conv_config":{$fusions_str,$opt_config
    ; CHECK-DAG:    }
    ; CHECK:      }
    )";

  constexpr static const char* kConvRewriteFusionsStr = R"(
    ; CHECK-DAG:          "fusions":{
    ; CHECK-DAG:            "ops":[$fused_ops]
    ; CHECK-DAG:      },)";

  constexpr static const char* kConvRewriteOptimizationsStr = R"(
    ; CHECK-DAG:          "optimization_config":{
    ; CHECK-DAG:            "weights_prepacked":$weights_prepacked,
    ; CHECK-DAG:            "user_scratchpad":$user_scratchpad,
    ; CHECK-DAG:      })";

  ConvolutionTest() {
    dtype_ = GetParam();
    atol_ = rtol_ = (dtype_ == F32) ? 1e-4 : 1e-2;
    // TODO(intel-tf): Set default value of user_scratchpad to true after
    // enabling feature
    user_scratchpad_ = false;
    weights_prepacked_ = false;
    dtypeString_ = primitive_util::LowercasePrimitiveTypeName(dtype_);
  }

  void SetUp() override {
    if (!IsSupportedType(dtype_)) {
      GTEST_SKIP() << "CPU does not support " << dtypeString_;
    }
  }

  void SetWeightsPrepacked(bool value) { weights_prepacked_ = value; }

  void SetUserScratchpad(bool value) { user_scratchpad_ = value; }

  std::string GetOptimizationsString() {
    return (user_scratchpad_ || weights_prepacked_)
               ? absl::StrReplaceAll(kConvRewriteOptimizationsStr,
                                     {{"$weights_prepacked",
                                       weights_prepacked_ ? "true" : "false"},
                                      {"$user_scratchpad",
                                       user_scratchpad_ ? "true" : "false"}})
               : "";
  }

  std::string ConvStringWithOptimizations(
      const std::vector<absl::string_view> fused_ops) {
    std::ostringstream stream;
    std::for_each(
        fused_ops.begin(), fused_ops.end(),
        [&](const absl::string_view& arg) { stream << "\"" << arg << "\","; });
    std::string fusions = stream.str();
    if (fused_ops.size() > 0) {
      fusions.pop_back();
      return absl::StrReplaceAll(
          kConvRewriteStr,
          {{"$fusions_str,", absl::StrReplaceAll(kConvRewriteFusionsStr,
                                                 {{"$fused_ops", fusions}})},
           {"$opt_config", GetOptimizationsString()}});
    }
    return absl::StrReplaceAll(
        kConvRewriteStr,
        {{"$fusions_str,", ""}, {"$opt_config", GetOptimizationsString()}});
  }

  // TODO(intel-tf): Remove this and simplify patterns when Elemental BF16 is
  // fully supported.
  PrimitiveType PromotedDtype() {
    // BF16 is promoted to F32 because not all HLO Instructions currently
    // support BF16 computations. Meanwhile, FP32 and FP16 elementwise
    // instructions are not promoted and remain unchanged.
    return (dtype_ == BF16) ? F32 : dtype_;
  }

  void AdjustToleranceForDtype(PrimitiveType for_type, float atol, float rtol) {
    if (dtype_ == for_type) {
      atol_ = atol;
      rtol_ = rtol;
    }
  }

  std::string PromotedDtypeToString() {
    return primitive_util::LowercasePrimitiveTypeName(PromotedDtype());
  }

  void RunCompareAndMatchOptimizedHlo(
      const absl::string_view outline,
      const std::vector<absl::string_view> fused_ops) {
    const std::string convolution_module_str = absl::StrReplaceAll(
        outline,
        {{"$dtype", dtypeString_}, {"$pdtype", PromotedDtypeToString()}});
    EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{atol_, rtol_}));
    MatchOptimizedHlo(convolution_module_str,
                      ConvStringWithOptimizations(fused_ops));
  }
};

TEST_P(ConvolutionTest, Simple2DTest1) {
  const absl::string_view outline = R"(
  HloModule convolution.test

  ENTRY convolution.test {
    arg.0 = $dtype[1,22,22,1] parameter(0)
    reshape.0 = $dtype[1,22,22,1] reshape(arg.0)
    arg.1 = $dtype[8,8,1,1] parameter(1)
    reshape.1 = $dtype[8,8,1,1] reshape(arg.1)
    convolution.0 = $dtype[1,11,11,1] convolution(reshape.0, reshape.1),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    reshape.2 = $dtype[1,11,11,1] reshape(convolution.0)
    tuple.0 = ($dtype[1,11,11,1]) tuple(reshape.2)
    ROOT gte.0 = $dtype[1,11,11,1] get-tuple-element(tuple.0), index=0
  })";

  RunCompareAndMatchOptimizedHlo(outline, {});
}

TEST_P(ConvolutionTest, Simple3DTest1) {
  const absl::string_view outline = R"(
  HloModule convolution.test

  ENTRY convolution.test {
    p0 = $dtype[8,4,5,5,1] parameter(0)
    p1 = $dtype[3,3,3,1,32] parameter(1)
    ROOT conv = $dtype[8,4,5,5,32] convolution(p0, p1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
})";

  RunCompareAndMatchOptimizedHlo(outline, {});
}

TEST_P(ConvolutionTest, Conv3DWithBiasTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test.with.bias

  ENTRY convolution.test.with.bias {
    arg.0 = $dtype[15,4,5,5,28] parameter(0)
    arg.1 = $dtype[3,3,3,28,64] parameter(1)
    conv = $dtype[15,4,5,5,64] convolution(arg.0, arg.1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    bias = $dtype[64] parameter(2)
    broadcasted_bias = $dtype[15,4,5,5,64] broadcast(bias), dimensions={4}
    ROOT add = $dtype[15,4,5,5,64] add(conv, broadcasted_bias)
})";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS"});
}

TEST_P(ConvolutionTest, Conv3DReluTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test.with.relu

  ENTRY convolution.test.with.relu {
    arg.0 = $dtype[15,4,5,5,28] parameter(0)
    arg.1 = $dtype[3,3,3,28,64] parameter(1)
    conv = $dtype[15,4,5,5,64] convolution(arg.0, arg.1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    const.1 = $pdtype[] constant(0)
    convert.0 = $dtype[] convert(const.1)
    bcast.2 = $dtype[15,4,5,5,64] broadcast(convert.0), dimensions={}
    ROOT maximum.1 = $dtype[15,4,5,5,64] maximum(conv, bcast.2)
})";

  RunCompareAndMatchOptimizedHlo(outline, {"RELU"});
}

TEST_P(ConvolutionTest, Conv2DWithBiasAndReluTest) {
  const absl::string_view outline = R"(
  HloModule convolution.bias.relu.test

  ENTRY convolution.bias.relu.test {
    arg0.1 = $dtype[1,22,22,1] parameter(0)
    arg0.2 = $dtype[8,8,1,10] parameter(1)
    convolution.0 = $dtype[1,11,11,10] convolution(arg0.1, arg0.2),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    const.0 = $dtype[10] constant(15)
    bcast.1 = $dtype[1,11,11,10] broadcast(const.0), dimensions={3}
    add.0 = $dtype[1,11,11,10] add(convolution.0, bcast.1)
    const.1 = $pdtype[] constant(0)
    convert.0 = $dtype[] convert(const.1)
    bcast.2 = $dtype[1,11,11,10] broadcast(convert.0), dimensions={}
    ROOT maximum.1 = $dtype[1,11,11,10] maximum(add.0, bcast.2)
  })";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS", "RELU"});
}

TEST_P(ConvolutionTest, Conv2DWithBinaryAddTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test.with.binaryadd

  ENTRY convolution.test.with.binaryadd {
    arg0.1 = $dtype[1,22,22,1] parameter(0)
    constant.3 = $dtype[] constant(1)
    broadcast.4 = $dtype[8,8,1,1] broadcast(constant.3), dimensions={}
    convolution.0 = $dtype[1,11,11,1] convolution(arg0.1, broadcast.4),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    constant.5 = $dtype[] constant(15)
    broadcast.6 = $dtype[1] broadcast(constant.5), dimensions={}
    broadcast.9 = $dtype[1,11,11,1] broadcast(broadcast.6), dimensions={3}
    ROOT add.10 = $dtype[1,11,11,1] add(convolution.0, broadcast.9)
  })";

  RunCompareAndMatchOptimizedHlo(outline, {"BINARY_ADD"});
}

// This test should match BIAS + RESIDUAL ADD when the residual add fusion is
// re-enabled.
TEST_P(ConvolutionTest, Conv2DWithBiasAndBinaryAddTest) {
  const absl::string_view outline = R"(
  HloModule convolution.add.test

  ENTRY convolution.add.test {
    arg0.1 = $dtype[1,22,22,1] parameter(0)
    arg0.2 = $dtype[8,8,1,10] parameter(1)
    convolution.0 = $dtype[1,11,11,10] convolution(arg0.1, arg0.2),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    const.0 = $dtype[10] constant(15)
    bcast.1 = $dtype[1,11,11,10] broadcast(const.0), dimensions={3}
    add.0 = $dtype[1,11,11,10] add(convolution.0, bcast.1)
    const.1 = $dtype[1,11,11,10] constant({...})
    ROOT add.1 = $dtype[1,11,11,10] add(add.0, const.1)
  })";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS"});
}

TEST_P(ConvolutionTest, ToeplitzConstrcutionTest) {
  if (dtype_ == BF16 || dtype_ == F16) {
    GTEST_SKIP() << "Skipping test for " << dtypeString_
                 << ". HLO Binary Complex instruction expects F32 inputs and "
                    "Unary Real and Imag instructions output F32 shapes only.";
  }

  const absl::string_view outline = R"(
  HloModule toeplitz.construction.test

  ENTRY toeplitz.construction.test {
    Arg_0.1 = c64[1,23,1] parameter(0)
    real.3 = $dtype[1,23,1] real(Arg_0.1)
    imag.4 = $dtype[1,23,1] imag(Arg_0.1)
    add.7 = $dtype[1,23,1] add(real.3, imag.4)
    Arg_1.2 = c64[1,3,3] parameter(1)
    real.5 = $dtype[1,3,3] real(Arg_1.2)
    convolution.8 = $dtype[1,21,3] convolution(add.7, real.5),
          window={size=3}, dim_labels=b0f_io0->b0f
    imag.6 = $dtype[1,3,3] imag(Arg_1.2)
    add.11 = $dtype[1,3,3] add(real.5, imag.6)
    convolution.12 = $dtype[1,21,3] convolution(imag.4, add.11),
          window={size=3}, dim_labels=b0f_io0->b0f
    subtract.13 = $dtype[1,21,3] subtract(convolution.8, convolution.12)
    subtract.9 = $dtype[1,3,3] subtract(imag.6, real.5)
    convolution.10 = $dtype[1,21,3] convolution(real.3, subtract.9),
          window={size=3}, dim_labels=b0f_io0->b0f
    add.14 = $dtype[1,21,3] add(convolution.8, convolution.10)
    ROOT complex.15 = c64[1,21,3] complex(subtract.13, add.14)
  })";

  RunCompareAndMatchOptimizedHlo(outline, {"BINARY_ADD"});
}

TEST_P(ConvolutionTest, Conv2DWithBiasAndTanhTest) {
  const absl::string_view outline = R"(
  HloModule convolution.bias.tanh.test

  ENTRY convolution.bias.tanh.test {
    arg0.1 = $dtype[1,22,22,1] parameter(0)
    arg0.2 = $dtype[8,8,1,10] parameter(1)
    convolution.0 = $dtype[1,11,11,10] convolution(arg0.1, arg0.2),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    const.0 = $dtype[10] constant(15)
    bcast.1 = $dtype[1,11,11,10] broadcast(const.0), dimensions={3}
    add.0 = $dtype[1,11,11,10] add(convolution.0, bcast.1)
    tanh.0 = $dtype[1,11,11,10] tanh(add.0)
    tuple.0 = ($dtype[1,11,11,10]) tuple(tanh.0)
    ROOT gte.0 = $dtype[1,11,11,10] get-tuple-element(tuple.0), index=0
  })";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS", "TANH"});
}

TEST_P(ConvolutionTest, Conv2DWithLinearAndBinaryAddTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test.linear.binaryadd

  ENTRY convolution.test.linear.binaryadd {
    arg0.1 = $dtype[1,22,22,1] parameter(0)
    constant.3 = $dtype[] constant(1)
    broadcast.4 = $dtype[8,8,1,1] broadcast(constant.3), dimensions={}
    convolution.0 = $dtype[1,11,11,1] convolution(arg0.1, broadcast.4),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    constant.4 = $pdtype[] constant(0.044715)
    convert.0 = $dtype[] convert(constant.4)
    broadcast.5 = $dtype[1,11,11,1] broadcast(convert.0), dimensions={}
    multiply.0 = $dtype[1,11,11,1] multiply(convolution.0,broadcast.5)
    constant.5 = $dtype[] constant(15)
    broadcast.6 = $dtype[1] broadcast(constant.5), dimensions={}
    broadcast.9 = $dtype[1,11,11,1] broadcast(broadcast.6), dimensions={3}
    ROOT add.10 = $dtype[1,11,11,1] add(multiply.0, broadcast.9)
  })";

  RunCompareAndMatchOptimizedHlo(outline, {"LINEAR", "BINARY_ADD"});
}

TEST_P(ConvolutionTest, Conv3DWithBiasAndRelu6Test) {
  const absl::string_view outline = R"(
  HloModule convolution.test.bias.relu6

  ENTRY convolution.test.bias.relu6 {
    arg.0 = $dtype[15,4,5,5,28] parameter(0)
    arg.1 = $dtype[3,3,3,28,64] parameter(1)
    conv = $dtype[15,4,5,5,64] convolution(arg.0, arg.1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    bias = $dtype[64] parameter(2)
    broadcasted_bias = $dtype[15,4,5,5,64] broadcast(bias), dimensions={4}
    add = $dtype[15,4,5,5,64] add(conv, broadcasted_bias)
    const.0 = $pdtype[] constant(0)
    convert.0 = $dtype[] convert(const.0)
    broadcast.0 = $dtype[15,4,5,5,64] broadcast(convert.0), dimensions={}
    const.1 = $pdtype[] constant(6)
    convert.1 = $dtype[] convert(const.1)
    broadcast.1 = $dtype[15,4,5,5,64] broadcast(convert.1), dimensions={}
    ROOT clamp.0 = $dtype[15,4,5,5,64] clamp(broadcast.0, add, broadcast.1)
})";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS", "RELU6"});
}

TEST_P(ConvolutionTest, Conv2DWithBiasAndSigmoidTest) {
  const absl::string_view outline = R"(
  HloModule convolution.bias.sigmoid.test

  ENTRY convolution.bias.sigmoid.test {
    arg0.1 = $dtype[1,22,22,1] parameter(0)
    arg0.2 = $dtype[8,8,1,10] parameter(1)
    convolution.0 = $dtype[1,11,11,10] convolution(arg0.1, arg0.2),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    const.0 = $dtype[10] constant(15)
    bcast.1 = $dtype[1,11,11,10] broadcast(const.0), dimensions={3}
    add.0 = $dtype[1,11,11,10] add(convolution.0, bcast.1)
    const.1 = $pdtype[] constant(1)
    convert.0 = $dtype[] convert(const.1)
    bcast.2 = $dtype[1,11,11,10] broadcast(convert.0), dimensions={}
    negate.0 = $dtype[1,11,11,10] negate(add.0)
    exponential.0 = $dtype[1,11,11,10] exponential(negate.0)
    add.1 = $dtype[1,11,11,10] add(bcast.2, exponential.0)
    divide.0 = $dtype[1,11,11,10] divide(bcast.2, add.1)
    tuple.0 =($dtype[1,11,11,10]) tuple(divide.0)
    ROOT gte.0 = $dtype[1,11,11,10] get-tuple-element(tuple.0), index=0
  })";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS", "SIGMOID"});
}

TEST_P(ConvolutionTest, Conv3DWithBiasAndEluTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test.bias.elu

  ENTRY convolution.test.bias.elu {
    arg.0 = $dtype[15,4,5,5,28] parameter(0)
    arg.1 = $dtype[3,3,3,28,64] parameter(1)
    conv = $dtype[15,4,5,5,64] convolution(arg.0, arg.1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    bias = $dtype[64] parameter(2)
    broadcasted_bias = $dtype[15,4,5,5,64] broadcast(bias), dimensions={4}
    add = $dtype[15,4,5,5,64] add(conv, broadcasted_bias)
    const.0 = $pdtype[] constant(0)
    convert.0 = $dtype[] convert(const.0)
    broadcast.0 = $dtype[15,4,5,5,64] broadcast(convert.0), dimensions={}
    compare.0 = pred[15,4,5,5,64] compare(add, broadcast.0), direction=GT
    exp-min-one.0 = $dtype[15,4,5,5,64] exponential-minus-one(add)
    ROOT select.0 = $dtype[15,4,5,5,64] select(compare.0, add, exp-min-one.0)
})";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS", "ELU"});
}

TEST_P(ConvolutionTest, Conv2DWithGeluApproxTest) {
  const absl::string_view outline = R"(
  HloModule convolution.gelu.approx.test

  ENTRY convolution.gelu.approx.test {
    arg0.1 = $dtype[1,22,22,1] parameter(0)
    arg0.2 = $dtype[8,8,1,10] parameter(1)
    convolution.0 = $dtype[1,11,11,10] convolution(arg0.1, arg0.2),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    mul.0 = $dtype[1,11,11,10] multiply(convolution.0, convolution.0)
    mul.1 = $dtype[1,11,11,10] multiply(convolution.0, mul.0)
    const.0 = $pdtype[] constant(0.044715)
    convert.0 = $dtype[] convert(const.0)
    bcast.0 = $dtype[1,11,11,10] broadcast(convert.0), dimensions={}
    mul.2 = $dtype[1,11,11,10] multiply(mul.1, bcast.0)
    add.0 = $dtype[1,11,11,10] add(convolution.0, mul.2)
    const.1 = $pdtype[] constant(0.797884583)
    convert.1 = $dtype[] convert(const.1)
    bcast.1 = $dtype[1,11,11,10] broadcast(convert.1), dimensions={}
    mul.3 = $dtype[1,11,11,10] multiply(add.0, bcast.1)
    tanh = $dtype[1,11,11,10] tanh(mul.3)
    const.2 = $pdtype[] constant(1)
    convert.2 = $dtype[] convert(const.2)
    bcast.2 = $dtype[1,11,11,10] broadcast(convert.2), dimensions={}
    add.2 = $dtype[1,11,11,10] add(tanh, bcast.2)
    const.3 = $pdtype[] constant(0.5)
    convert.3 = $dtype[] convert(const.3)
    bcast.3 = $dtype[1,11,11,10] broadcast(convert.3), dimensions={}
    mul.4 = $dtype[1,11,11,10] multiply(add.2, bcast.3)
    ROOT out = $dtype[1,11,11,10] multiply(convolution.0, mul.4)
  })";

  RunCompareAndMatchOptimizedHlo(outline, {"GELU_TANH"});
}

TEST_P(ConvolutionTest, Conv2DWithBiasAndGeluApproxTest) {
  const absl::string_view outline = R"(
  HloModule convolution.bias.gelu.approx.test

  ENTRY convolution.bias.gelu.approx.test {
    arg0.1 = $dtype[1,22,22,1] parameter(0)
    arg0.2 = $dtype[8,8,1,10] parameter(1)
    convolution.0 = $dtype[1,11,11,10] convolution(arg0.1, arg0.2),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    constant.0 = $dtype[10] constant(15)
    bcast.1 = $dtype[1,11,11,10] broadcast(constant.0), dimensions={3}
    add.0 = $dtype[1,11,11,10] add(convolution.0, bcast.1)
    constant.12 = $pdtype[] constant(0.044715)
    convert.0 = $dtype[] convert(constant.12)
    broadcast.13 = $dtype[1,11,11,10] broadcast(convert.0), dimensions={}
    multiply.14 = $dtype[1,11,11,10] multiply(broadcast.13, add.0)
    multiply.11 = $dtype[1,11,11,10] multiply(add.0, add.0)
    multiply.15 = $dtype[1,11,11,10] multiply(multiply.14, multiply.11)
    add.16 = $dtype[1,11,11,10] add(add.0, multiply.15)
    constant.17 = $pdtype[] constant(0.797884583)
    convert.1 = $dtype[] convert(constant.17)
    broadcast.18 = $dtype[1,11,11,10] broadcast(convert.1), dimensions={}
    multiply.19 = $dtype[1,11,11,10] multiply(add.16, broadcast.18)
    tanh.20 = $dtype[1,11,11,10] tanh(multiply.19)
    constant.21 = $pdtype[] constant(1)
    convert.2 = $dtype[] convert(constant.21)
    broadcast.22 = $dtype[1,11,11,10] broadcast(convert.2), dimensions={}
    add.23 = $dtype[1,11,11,10] add(tanh.20, broadcast.22)
    constant.24 = $pdtype[] constant(0.5)
    convert.3 = $dtype[] convert(constant.24)
    broadcast.25 = $dtype[1,11,11,10] broadcast(convert.3), dimensions={}
    multiply.26 = $dtype[1,11,11,10] multiply(add.23, broadcast.25)
    ROOT multiply.27 = $dtype[1,11,11,10] multiply(add.0, multiply.26)
  })";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS", "GELU_TANH"});
}

TEST_P(ConvolutionTest, Conv3DWithGeluExactTest) {
  const absl::string_view outline = R"(
  HloModule convolution.gelu.exact.test

  ENTRY convolution.gelu.exact.test {
    arg.0 = $dtype[15,4,5,5,28] parameter(0)
    arg.1 = $dtype[3,3,3,28,64] parameter(1)
    conv = $dtype[15,4,5,5,64] convolution(arg.0, arg.1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    const.0 = $pdtype[] constant(0.707106769)
    convert.0 = $dtype[] convert(const.0)
    bcast.0 = $dtype[15,4,5,5,64] broadcast(convert.0), dimensions={}
    mul.0 = $dtype[15,4,5,5,64] multiply(conv, bcast.0)
    erf.0 = $dtype[15,4,5,5,64] erf(mul.0)
    const.1 = $pdtype[] constant(1)
    convert.1 = $dtype[] convert(const.1)
    bcast.1 = $dtype[15,4,5,5,64] broadcast(convert.1), dimensions={}
    add.0 = $dtype[15,4,5,5,64] add(erf.0, bcast.1)
    const.2 = $pdtype[] constant(0.5)
    convert.2 = $dtype[] convert(const.2)
    bcast.2 = $dtype[15,4,5,5,64] broadcast(convert.2), dimensions={}
    mul.1 = $dtype[15,4,5,5,64] multiply(add.0, bcast.2)
    ROOT out = $dtype[15,4,5,5,64] multiply(conv, mul.1)
})";

  RunCompareAndMatchOptimizedHlo(outline, {"GELU_ERF"});
}

TEST_P(ConvolutionTest, Conv2DWithBiasAndGeluExactPattern1Test) {
  const absl::string_view outline = R"(
  HloModule convolution.test.with.bias.gelu.exact

  ENTRY convolution.test.with.bias.gelu.exact {
    arg.0 = $dtype[1,22,22,1] parameter(0)
    arg.1 = $dtype[8,8,1,10] parameter(1)
    conv = $dtype[1,11,11,10] convolution(arg.0, arg.1),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    bias = $dtype[10] parameter(2)
    broadcasted_bias = $dtype[1,11,11,10] broadcast(bias), dimensions={3}
    add = $dtype[1,11,11,10] add(conv, broadcasted_bias)
    const.0 = $pdtype[] constant(0.70703125)
    convert.0 = $dtype[] convert(const.0)
    bcast.0 = $dtype[1,11,11,10] broadcast(convert.0), dimensions={}
    mul.0 = $dtype[1,11,11,10] multiply(add, bcast.0)
    erf.0 = $dtype[1,11,11,10] erf(mul.0)
    const.1 = $pdtype[] constant(1)
    convert.1 = $dtype[] convert(const.1)
    bcast.1 = $dtype[1,11,11,10] broadcast(convert.1), dimensions={}
    add.0 = $dtype[1,11,11,10] add(erf.0, bcast.1)
    const.2 = $pdtype[] constant(0.5)
    convert.2 = $dtype[] convert(const.2)
    bcast.2 = $dtype[1,11,11,10] broadcast(convert.2), dimensions={}
    mul.1 = $dtype[1,11,11,10] multiply(add.0, bcast.2)
    ROOT out = $dtype[1,11,11,10] multiply(add, mul.1)
})";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS", "GELU_ERF"});
}

TEST_P(ConvolutionTest, Conv2DWithBiasAndGeluExactPattern2Test) {
  const absl::string_view outline = R"(
  HloModule convolution.test.with.bias.gelu.exact

  ENTRY convolution.test.with.bias.gelu.exact {
    arg.0 = $dtype[1,22,22,1] parameter(0)
    arg.1 = $dtype[8,8,1,10] parameter(1)
    conv = $dtype[1,11,11,10] convolution(arg.0, arg.1),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    bias = $dtype[10] parameter(2)
    broadcasted_bias = $dtype[1,11,11,10] broadcast(bias), dimensions={3}
    add = $dtype[1,11,11,10] add(conv, broadcasted_bias)
    constant.384 = $pdtype[] constant(0.707182348)
    convert.0 = $dtype[] convert(constant.384)
    broadcast.385 = $dtype[1,11,11,10] broadcast(convert.0), dimensions={}
    multiply.386 = $dtype[1,11,11,10] multiply(broadcast.385, add)
    erf.387 = $dtype[1,11,11,10] erf(multiply.386)
    constant.388 = $pdtype[] constant(1)
    convert.1 = $dtype[] convert(constant.388)
    broadcast.389 = $dtype[1,11,11,10] broadcast(convert.1), dimensions={}
    add.390 = $dtype[1,11,11,10] add(erf.387, broadcast.389)
    multiply.393 = $dtype[1,11,11,10] multiply(add.390, add)
    constant.391 = $pdtype[] constant(0.5)
    convert.2 = $dtype[] convert(constant.391)
    broadcast.392 = $dtype[1,11,11,10] broadcast(convert.2)
    ROOT mul.394 = $dtype[1,11,11,10] multiply(multiply.393, broadcast.392)
})";

  RunCompareAndMatchOptimizedHlo(outline, {"BIAS", "GELU_ERF"});
}

INSTANTIATE_TEST_SUITE_P(
    OneDnnConvolutionTestSuite, ConvolutionTest,
    ::testing::Values(F32, BF16, F16),
    [](const ::testing::TestParamInfo<ConvolutionTest::ParamType>& info) {
      auto test_name = primitive_util::LowercasePrimitiveTypeName(info.param);
      std::transform(test_name.begin(), test_name.end(), test_name.begin(),
                     [](auto c) { return std::toupper(c); });
      return test_name;
    });

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
