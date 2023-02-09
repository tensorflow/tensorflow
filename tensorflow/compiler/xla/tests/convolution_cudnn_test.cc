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

}  // namespace
}  // namespace xla
