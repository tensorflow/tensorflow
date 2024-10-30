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
