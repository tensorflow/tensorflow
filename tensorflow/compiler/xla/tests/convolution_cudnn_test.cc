/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <tuple>
#include <utility>

#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

class ConvolutionHloTest : public HloTestBase {};

XLA_TEST_F(ConvolutionHloTest, TestCudnnConvInt8x32) {
  // This convolution should be transformed to "cudnn-conv" and vectorized as
  // INT8x32_CONFIG on GPUs.
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY TestComputation {
  %input = s8[4,48,48,64] parameter(0)
  %filter = s8[64,3,3,64] parameter(1)
  ROOT %conv = s8[4,48,48,64] convolution(%input, %filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0, 0}));
}

XLA_TEST_F(ConvolutionHloTest, TestCudnnConvInt8x32Bias) {
  // cudnnConvolutionBiasActivationForward() for int8 is only supported on GPUs
  // with compute capability 6.1 or later.
  if (!backend()
          .default_stream_executor()
          ->GetDeviceDescription()
          .cuda_compute_capability()
          .IsAtLeast(6, 1) ) {
    return;
  }

  // This convolution with the following add/relu ops should be transformed to
  // "cudnn-conv-bias-activation" and vectorized as INT8x32_CONFIG on GPUs.
  // In order to verify this with non-zero bias and without adding test-specific
  // code to HLO evaluator, the overflow is then cleared by taking a remainder.
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY TestComputation {
  %input = s8[4,48,48,64] parameter(0)
  %filter = s8[64,3,3,64] parameter(1)
  %zero = s8[] constant(0)
  %zeros = s8[4,48,48,64] broadcast(%zero), dimensions={}
  %one = s8[] constant(1)
  %ones = s8[4,48,48,64] broadcast(%one), dimensions={}
  %conv = s8[4,48,48,64] convolution(%input, %filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f
  %result = add(%conv, %ones)
  %relu = maximum(%result, %zeros)
  %ceil = s8[] constant(127)
  %ceil_broadcast = s8[4,48,48,64] broadcast(%ceil), dimensions={}
  ROOT %output = remainder(%relu, %ceil_broadcast)
})";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0, 0}));
}

XLA_TEST_F(ConvolutionHloTest, TestCudnnConvInt8x32BiasNonConst) {
  // Test two GPU compiled HLOs, first version with vectorization disabled,
  // second with vectorization enabled. The reference implementation
  // (Interpreter) does not support the fused conv-add-relu-clamp operation,
  // thus cannot be used.
  if (backend()
          .default_stream_executor()
          ->GetDeviceDescription()
          .cuda_compute_capability()
          .major != 8) {
    return;
  }
  constexpr char kHloBase[] = R"(
HloModule TestModule, entry_computation_layout={(s8[4,48,48,64]{3,2,1,0},s8[64,3,3,64]{3,2,1,0},s8[64]{0})->s8[4,48,48,64]{3,2,1,0}}

ENTRY TestComputation {
  input = s8[4,48,48,64]{3,2,1,0} parameter(0)
  filter = s8[64,3,3,64]{3,2,1,0} parameter(1)
  bias = s8[64]{0} parameter(2)
  convert.1 = f32[64]{0} convert(bias)
  cudnn-conv-bias-activation.3 = (s8[4,48,48,64]{3,2,1,0}, u8[0]{0}) custom-call(input, filter, convert.1),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convBiasActivationForward",
      backend_config="{\"activation_mode\":\"2\",\"conv_result_scale\":1,\"side_input_scale\":0,\"algorithm\":{
        \"algo_id\":\"38\",\"math_type\":\"DEFAULT_MATH\",\"tuning_knobs\":{\"14\":\"5\",\"13\":\"1\",\"23\":\"0\",\"2\":\"1\"},
        \"is_cudnn_frontend\":true,\"workspace_size\":\"0\"}}"
  ROOT get-tuple-element.1 = s8[4,48,48,64]{3,2,1,0} get-tuple-element(cudnn-conv-bias-activation.3), index=0
})";
  constexpr char kHloVectorized[] = R"(
HloModule TestModule, entry_computation_layout={(s8[4,48,48,64]{3,2,1,0},s8[64,3,3,64]{3,2,1,0},s8[64]{0})->s8[4,48,48,64]{3,2,1,0}}

ENTRY TestComputation {
  input = s8[4,48,48,64]{3,2,1,0} parameter(0)
  bitcast.36 = s8[4,48,48,2,32]{4,3,2,1,0} bitcast(input)
  transpose = s8[4,2,48,48,32]{4,3,2,1,0} transpose(bitcast.36), dimensions={0,3,1,2,4}
  filter = s8[64,3,3,64]{3,2,1,0} parameter(1)
  bitcast.19 = s8[8,4,2,3,3,2,8,4]{7,6,5,4,3,2,1,0} bitcast(filter)
  transpose.3 = s8[2,3,3,8,2,8,4,4]{7,6,5,4,3,2,1,0} transpose(bitcast.19), dimensions={5,3,4,0,2,6,1,7}
  bitcast.28 = s8[64,2,3,3,32]{4,3,2,1,0} bitcast(transpose.3)
  bias = s8[64]{0} parameter(2)
  convert.2 = f32[64]{0} convert(bias)
  bitcast.33 = f32[2,4,2,4]{3,2,1,0} bitcast(convert.2)
  transpose.6 = f32[2,2,4,4]{3,2,1,0} transpose(bitcast.33), dimensions={0,2,1,3}
  bitcast.37 = f32[64]{0} bitcast(transpose.6)
  cudnn-conv-bias-activation.4 = (s8[4,2,48,48,32]{4,3,2,1,0}, u8[51328]{0}) custom-call(transpose, bitcast.28, bitcast.37),
      window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward",
      backend_config="{\"activation_mode\":\"2\",\"conv_result_scale\":1,\"side_input_scale\":0,\"algorithm\":{
        \"algo_id\":\"7\",\"math_type\":\"DEFAULT_MATH\",\"tuning_knobs\":{\"7\":\"3\",\"2\":\"0\",\"5\":\"4\",\"6\":\"4\",\"4\":\"2\",\"21\":\"0\"},
        \"is_cudnn_frontend\":true,\"workspace_size\":\"51328\"},\"reordered_int8_nchw_vect\":true}"
  get-tuple-element.6 = s8[4,2,48,48,32]{4,3,2,1,0} get-tuple-element(cudnn-conv-bias-activation.4), index=0
  transpose.1 = s8[4,48,48,2,32]{4,3,2,1,0} transpose(get-tuple-element.6), dimensions={0,2,3,1,4}
  ROOT bitcast.1 = s8[4,48,48,64]{3,2,1,0} bitcast(transpose.1)
})";
  EXPECT_TRUE(RunAndCompareTwoModules(kHloBase, kHloVectorized, ErrorSpec{0, 0},
                                      /*run_hlo_passes=*/false));
}

class HloCompareModulesTest : public HloTestBase {
 public:
  HloCompareModulesTest(std::string hlo_base, std::string hlo_test,
                        bool run_hlo_passes)
      : hlo_base_(std::move(hlo_base)),
        hlo_test_(std::move(hlo_test)),
        run_hlo_passes_(run_hlo_passes) {}

  void TestBody() override {
    EXPECT_TRUE(RunAndCompareTwoModules(hlo_base_, hlo_test_, ErrorSpec{0, 0},
                                        run_hlo_passes_));
  }

 private:
  std::string hlo_base_, hlo_test_;
  bool run_hlo_passes_;
};

XLA_TEST_F(ConvolutionHloTest, TestCudnnConvInt8x32Revectorize) {
  // Compare re-vectorized custom call vs the default version.
  constexpr char kHlo[] = R"(
HloModule TestModule

ENTRY TestComputation {
  %input = s8[4,48,48,64] parameter(0)
  %filter = s8[64,3,3,64] parameter(1)
  %conv = (s8[4,48,48,64], u8[0]) custom-call(%input, %filter),
        window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f,
        custom_call_target="__cudnn$convForward",
        backend_config="{\"activation_mode\":\"0\",\"conv_result_scale\":1,\"side_input_scale\":0}"
  ROOT %gte = s8[4,48,48,64] get-tuple-element(%conv), index=0
})";
  constexpr char kHloVectorized[] = R"(
HloModule TestModule

ENTRY TestComputation {
  %input = s8[4,48,48,64] parameter(0)
  %input.1 = s8[4,48,48,16,4] reshape(%input)
  %filter = s8[64,3,3,64] parameter(1)
  %filter.1 = s8[64,3,3,16,4] reshape(%filter)
  %conv = (s8[4,48,48,16,4], u8[0]) custom-call(%input.1, %filter.1),
        window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f,
        custom_call_target="__cudnn$convForward",
        backend_config="{\"activation_mode\":\"0\",\"conv_result_scale\":1,\"side_input_scale\":0}"
  %gte = s8[4,48,48,16,4] get-tuple-element(%conv), index=0
  ROOT reshape.3 = s8[4,48,48,64] reshape(%gte)
})";
  HloCompareModulesTest(kHlo, kHloVectorized, /*run_hlo_passes=*/true)
      .TestBody();
}

class ReorderFilterHloTest : public ::testing::TestWithParam<std::tuple<
                                 /*input_features=*/int64_t,
                                 /*output_features=*/int64_t,
                                 /*spatial_size=*/int64_t>> {};

XLA_TEST_P(ReorderFilterHloTest, TestCudnnReorderFilter) {
  // Test that the bitcast-transpose-bitcast with reverse-engineered reordering
  // works the same way as cudnnReorderFilterAndBias custom call. If at any
  // point in the future the black-box CUDA implementation changes, this test
  // should fail. This test only verifies the filter reordering (no bias).
  auto [input_features, output_features, size] = GetParam();

  // Filter shape is [O, I/32, H, W, 32]
  std::string shape = absl::Substitute("$0,$1,$2,$2,32", output_features,
                                       input_features / 32, size);

  std::string hloTranspose = absl::Substitute(
      R"(
HloModule TestModule, entry_computation_layout={(s8[$0]{4,3,2,1,0})->s8[$0]{4,3,2,1,0}}

ENTRY TestComputation {
  %filter = s8[$0]{4,3,2,1,0} parameter(0)
  %bitcast.1 = s8[$3,4,2,$2,$1,$1,8,4]{7,6,5,4,3,2,1,0} bitcast(%filter)
  %transpose = s8[$2,$1,$1,$3,2,8,4,4]{7,6,5,4,3,2,1,0} transpose(%bitcast.1), dimensions={3,4,5,0,2,6,1,7}
  ROOT %bitcast.2 = s8[$0]{4,3,2,1,0} bitcast(%transpose)
})",
      shape, size, input_features / 32, output_features / 8);

  std::string hloCustomCall = absl::Substitute(
      R"(
HloModule TestModule, entry_computation_layout={(s8[$0]{4,3,2,1,0})->s8[$0]{4,3,2,1,0}}

ENTRY TestComputation {
  %filter = s8[$0]{4,3,2,1,0} parameter(0)
  ROOT %custom-call = s8[$0]{4,3,2,1,0} custom-call(%filter), custom_call_target="__cudnn$$convReorderFilter"
})",
      shape);

  HloCompareModulesTest(hloTranspose, hloCustomCall, /*run_hlo_passes=*/false)
      .TestBody();
}

INSTANTIATE_TEST_SUITE_P(CudnnReorderSuite, ReorderFilterHloTest,
                         ::testing::Combine(
                             /*input_features=*/::testing::Values(32, 64),
                             /*output_features=*/::testing::Values(32, 64),
                             /*spatial_size=*/::testing::Values(1, 3, 5)));

class ReorderFilterAndBiasHloTest : public ::testing::TestWithParam<int64_t> {};

XLA_TEST_P(ReorderFilterAndBiasHloTest, TestCudnnReorderFilterAndBias) {
  // This test verifies that the bias reordering works correctly; the filter
  // reordering is verified by the previous test.
  const int64_t bias_size = GetParam();

  std::string hloTranspose = absl::Substitute(
      R"(
HloModule TestModule, entry_computation_layout={(s8[32,1,3,3,32]{4,3,2,1,0},f32[$0]{0})->(s8[32,1,3,3,32]{4,3,2,1,0},f32[$0]{0})}

ENTRY TestComputation {
  %filter = s8[32,1,3,3,32]{4,3,2,1,0} parameter(0)
  %bias = f32[$0]{0} parameter(1)
  %bitcast.1   = s8[4,4,2,1,3,3,8,4]{7,6,5,4,3,2,1,0} bitcast(%filter)
  %transpose.1 = s8[1,3,3,4,2,8,4,4]{7,6,5,4,3,2,1,0} transpose(%bitcast.1), dimensions={3,4,5,0,2,6,1,7}
  %reverse.1   = s8[32,1,3,3,32]{4,3,2,1,0} bitcast(%transpose.1)
  %bitcast.2   = f32[$1,4,2,4]{3,2,1,0} bitcast(%bias)
  %transpose.2 = f32[$1,2,4,4]{3,2,1,0} transpose(%bitcast.2), dimensions={0,2,1,3}
  %reverse.2   = f32[$0]{0} bitcast(%transpose.2)
  ROOT %result = (s8[32,1,3,3,32]{4,3,2,1,0}, f32[$0]{0}) tuple(%reverse.1, %reverse.2)
})",
      bias_size, bias_size / 32);

  std::string hloCustomCall = absl::Substitute(
      R"(
HloModule TestModule, entry_computation_layout={(s8[32,1,3,3,32]{4,3,2,1,0},f32[$0]{0})->(s8[32,1,3,3,32]{4,3,2,1,0},f32[$0]{0})}

ENTRY TestComputation {
  %filter = s8[32,1,3,3,32]{4,3,2,1,0} parameter(0)
  %bias = f32[$0]{0} parameter(1)
  ROOT %result = (s8[32,1,3,3,32]{4,3,2,1,0}, f32[$0]{0}) custom-call(%filter, %bias), custom_call_target="__cudnn$$convReorderFilterAndBias"
})",
      bias_size);

  HloCompareModulesTest(hloTranspose, hloCustomCall, /*run_hlo_passes=*/false)
      .TestBody();
}

INSTANTIATE_TEST_SUITE_P(CudnnReorderSuite, ReorderFilterAndBiasHloTest,
                         /*bias_size=*/::testing::Values(32, 64, 96));

}  // namespace
}  // namespace xla
