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
  if (!backend()
           .default_stream_executor()
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeast(8)) {
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
  bitcast.18 = s8[64,3,3,2,32]{4,3,2,1,0} bitcast(filter)
  transpose.3 = s8[64,2,3,3,32]{4,3,2,1,0} transpose(bitcast.18), dimensions={0,3,1,2,4}
  bias = s8[64]{0} parameter(2)
  convert.2 = f32[64]{0} convert(bias)
  custom-call.3 = (s8[64,2,3,3,32]{4,3,2,1,0}, f32[64]{0}) custom-call(transpose.3, convert.2), custom_call_target="__cudnn$convReorderFilterAndBias"
  get-tuple-element.2 = s8[64,2,3,3,32]{4,3,2,1,0} get-tuple-element(custom-call.3), index=0
  get-tuple-element.3 = f32[64]{0} get-tuple-element(custom-call.3), index=1
  cudnn-conv-bias-activation.4 = (s8[4,2,48,48,32]{4,3,2,1,0}, u8[51328]{0}) custom-call(transpose, get-tuple-element.2, get-tuple-element.3),
      window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward",
      backend_config="{\"activation_mode\":\"2\",\"conv_result_scale\":1,\"side_input_scale\":0,\"algorithm\":{
        \"algo_id\":\"7\",\"math_type\":\"DEFAULT_MATH\",\"tuning_knobs\":{\"7\":\"3\",\"2\":\"0\",\"5\":\"4\",\"6\":\"4\",\"4\":\"2\",\"21\":\"0\"},
        \"is_cudnn_frontend\":true,\"workspace_size\":\"51328\"},\"reordered_int8_nchw_vect\":true}"
  get-tuple-element.6 = s8[4,2,48,48,32]{4,3,2,1,0} get-tuple-element(cudnn-conv-bias-activation.4), index=0
  transpose.4 = s8[4,48,48,2,32]{4,3,2,1,0} transpose(get-tuple-element.6), dimensions={0,2,3,1,4}
  ROOT bitcast.1 = s8[4,48,48,64]{3,2,1,0} bitcast(transpose.4)
})";
  EXPECT_TRUE(RunAndCompareTwoModules(kHloBase, kHloVectorized, ErrorSpec{0, 0},
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace xla
