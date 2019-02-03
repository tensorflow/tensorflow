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

#include "tensorflow/compiler/plugin/poplar/driver/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ConvolutionClassifierTest = HloTestBase;

// Check basic parameter matching
TEST_F(ConvolutionClassifierTest, Training1) {
  std::string hlo_string = R"(
HloModule top

add14 {
  x.14.0 = f32[] parameter(0)
  y.14.1 = f32[] parameter(1)
  ROOT add.14.2 = f32[] add(x.14.0, y.14.1)
}

max13 {
  x.13.0 = f32[] parameter(0)
  y.13.1 = f32[] parameter(1)
  ROOT maximum.13.2 = f32[] maximum(x.13.0, y.13.1)
}

_pop_op_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  broadcast.19.18.clone = f16[1,16,16,4] broadcast(f16[4] arg_1), dimensions={3}
  ROOT add.19.19.clone = f16[1,16,16,4] add(f16[1,16,16,4] arg_0, f16[1,16,16,4] broadcast.19.18.clone)
}

_pop_op_biasadd.1 {
  arg_0.1 = f16[1,16,16,64] parameter(0)
  arg_1.1 = f16[64] parameter(1)
  broadcast.19.15.clone = f16[1,16,16,64] broadcast(f16[64] arg_1.1), dimensions={3}
  ROOT add.19.16.clone = f16[1,16,16,64] add(f16[1,16,16,64] arg_0.1, f16[1,16,16,64] broadcast.19.15.clone)
}

_pop_op_wide_const {
  constant.19.10.clone = f16[] constant(0.0100021362)
  ROOT broadcast.19.63.clone = f16[4] broadcast(f16[] constant.19.10.clone), dimensions={}
}

_pop_op_wide_const.1  {
  constant.19.10.clone.1 = f16[] constant(0.0100021362)
  ROOT broadcast.19.69.clone = f16[5,5,64,4] broadcast(f16[] constant.19.10.clone.1), dimensions={}
}

_pop_op_wide_const.2  {
  constant.19.10.clone.2 = f16[] constant(0.0100021362)
  ROOT broadcast.19.83.clone = f16[64] broadcast(f16[] constant.19.10.clone.2), dimensions={}
}

_pop_op_wide_const.3 {
  constant.19.10.clone.3 = f16[] constant(0.0100021362)
  ROOT broadcast.19.87.clone = f16[7,7,4,64] broadcast(f16[] constant.19.10.clone.3), dimensions={}
}

_pop_op_conv_with_reverse {
  arg_0.2 = f16[1,16,16,4] parameter(0)
  arg_1.2 = f16[5,5,64,4] parameter(1)
  reverse.19.67.clone = f16[5,5,64,4] reverse(f16[5,5,64,4] arg_1.2), dimensions={0,1}
  ROOT convolution.19.68.clone = f16[1,16,16,64] convolution(f16[1,16,16,4] arg_0.2, f16[5,5,64,4] reverse.19.67.clone), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f
}

pop_backprop_conv {
  arg_0.3 = f16[1,16,16,4] parameter(0)
  arg_1.3 = f16[5,5,64,4] parameter(1)
  ROOT call.2.clone = f16[1,16,16,64] fusion(f16[1,16,16,4] arg_0.3, f16[5,5,64,4] arg_1.3), kind=kCustom, calls=_pop_op_conv_with_reverse
}

pop_convolution {
  arg_0.4 = f16[1,16,16,64] parameter(0)
  arg_1.4 = f16[5,5,64,4] parameter(1)
  ROOT convolution.19.17.clone = f16[1,16,16,4] convolution(f16[1,16,16,64] arg_0.4, f16[5,5,64,4] arg_1.4), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01io->b01f
}

pop_convolution.1 {
  arg_0.5 = f16[1,16,16,4] parameter(0)
  arg_1.5 = f16[7,7,4,64] parameter(1)
  ROOT convolution.19.14.clone = f16[1,16,16,64] convolution(f16[1,16,16,4] arg_0.5, f16[7,7,4,64] arg_1.5), window={size=7x7 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
}

pop_convolution.2 {
  arg_0.6 = f16[1,16,16,64] parameter(0)
  arg_1.6 = f16[1,16,16,4] parameter(1)
  ROOT convolution.19.66.clone = f16[5,5,64,4] convolution(f16[1,16,16,64] arg_0.6, f16[1,16,16,4] arg_1.6), window={size=16x16 pad=2_2x2_2}, dim_labels=f01b_i01o->01bf
}

pop_convolution.3 {
  arg_0.7 = f16[1,16,16,4] parameter(0)
  arg_1.7 = f16[1,16,16,64] parameter(1)
  ROOT convolution.19.86.clone = f16[7,7,4,64] convolution(f16[1,16,16,4] arg_0.7, f16[1,16,16,64] arg_1.7), window={size=16x16 pad=3_3x3_3}, dim_labels=f01b_i01o->01bf
}

__arithmetic_expression {
  arg_1.8 = f32[1,1024] parameter(1)
  arg_2 = f32[1,1024] parameter(2)
  divide.19.48.clone = f32[1,1024] divide(f32[1,1024] arg_1.8, f32[1,1024] arg_2)
  arg_0.8 = f32[1,1024] parameter(0)
  ROOT subtract.19.49.clone = f32[1,1024] subtract(f32[1,1024] divide.19.48.clone, f32[1,1024] arg_0.8)
}

cluster_1  {
  arg2.19.2 = f16[4] parameter(2)
  call.3 = f16[4] fusion(), kind=kCustom, calls=_pop_op_wide_const
  arg1.19.1 = f16[1,1024] parameter(1)
  convert.19.11 = f32[1,1024] convert(f16[1,1024] arg1.19.1)
  arg0.19.0 = f16[1,16,16,4] parameter(0)
  arg5.19.5 = f16[7,7,4,64] parameter(5)
  call.9 = f16[1,16,16,64] fusion(f16[1,16,16,4] arg0.19.0, f16[7,7,4,64] arg5.19.5), kind=kCustom, calls=pop_convolution.1
  arg4.19.4 = f16[64] parameter(4)
  call.1 = f16[1,16,16,64] fusion(f16[1,16,16,64] call.9, f16[64] arg4.19.4), kind=kCustom, calls=_pop_op_biasadd.1
  arg3.19.3 = f16[5,5,64,4] parameter(3)
  call.8 = f16[1,16,16,4] fusion(f16[1,16,16,64] call.1, f16[5,5,64,4] arg3.19.3), kind=kCustom, calls=pop_convolution
  call = f16[1,16,16,4] fusion(f16[1,16,16,4] call.8, f16[4] arg2.19.2), kind=kCustom, calls=_pop_op_biasadd
  convert = f32[1,16,16,4] convert(f16[1,16,16,4] call)
  reshape = f32[1,1024] reshape(f32[1,16,16,4] convert)
  constant.19.29 = f32[] constant(-inf)
  reduce = f32[1] reduce(f32[1,16,16,4] convert, constant.19.29), dimensions={1,2,3}, to_apply=max13
  broadcast.19.31 = f32[1,1024] broadcast(reduce), dimensions={0}
  subtract.19.32 = f32[1,1024] subtract(f32[1,1024] reshape, f32[1,1024] broadcast.19.31)
  exponential.19.33 = f32[1,1024] exponential(f32[1,1024] subtract.19.32)
  constant.19.35 = f32[] constant(0)
  reduce.19.36 = f32[1] reduce(f32[1,1024] exponential.19.33, constant.19.35), dimensions={1}, to_apply=add14
  broadcast.19.47 = f32[1,1024] broadcast(reduce.19.36), dimensions={0}
  call.12 = f32[1,1024] call(f32[1,1024] convert.19.11, f32[1,1024] exponential.19.33, f32[1,1024] broadcast.19.47), to_apply=__arithmetic_expression
  convert.19.50 = f16[1,1024] convert(f32[1,1024] call.12)
  convert.1 = f32[1,1024] convert(f16[1,1024] convert.19.50)
  reshape.1 = f32[1,16,16,4] reshape(f32[1,1024] convert.1)
  reduce.19.61 = f32[4] reduce(f32[1,16,16,4] reshape.1, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.62 = f16[4] convert(f32[4] reduce.19.61)
  multiply.19.64 = f16[4] multiply(f16[4] call.3, f16[4] convert.19.62)
  subtract.19.65 = f16[4] subtract(f16[4] arg2.19.2, f16[4] multiply.19.64)
  call.4 = f16[5,5,64,4] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  convert.3 = f16[1,1024] convert(f32[1,1024] convert.1)
  reshape.10 = f16[1,16,16,4] reshape(f16[1,1024] convert.3)
  call.10 = f16[5,5,64,4] fusion(f16[1,16,16,64] call.1, f16[1,16,16,4] reshape.10), kind=kCustom, calls=pop_convolution.2
  multiply.19.70 = f16[5,5,64,4] multiply(f16[5,5,64,4] call.4, f16[5,5,64,4] call.10)
  subtract.19.71 = f16[5,5,64,4] subtract(f16[5,5,64,4] arg3.19.3, f16[5,5,64,4] multiply.19.70)
  call.5 = f16[64] fusion(), kind=kCustom, calls=_pop_op_wide_const.2
  call.7 = f16[1,16,16,64] fusion(f16[1,16,16,4] reshape.10, f16[5,5,64,4] arg3.19.3), kind=kCustom, calls=pop_backprop_conv
  convert.19.78 = f32[1,16,16,64] convert(f16[1,16,16,64] call.7)
  reduce.19.81 = f32[64] reduce(f32[1,16,16,64] convert.19.78, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.82 = f16[64] convert(f32[64] reduce.19.81)
  multiply.19.84 = f16[64] multiply(f16[64] call.5, f16[64] convert.19.82)
  subtract.19.85 = f16[64] subtract(f16[64] arg4.19.4, f16[64] multiply.19.84)
  call.6 = f16[7,7,4,64] fusion(), kind=kCustom, calls=_pop_op_wide_const.3
  convert.2 = f16[1,16,16,64] convert(f32[1,16,16,64] convert.19.78)
  call.11 = f16[7,7,4,64] fusion(f16[1,16,16,4] arg0.19.0, f16[1,16,16,64] convert.2), kind=kCustom, calls=pop_convolution.3
  multiply.19.88 = f16[7,7,4,64] multiply(f16[7,7,4,64] call.6, f16[7,7,4,64] call.11)
  subtract.19.89 = f16[7,7,4,64] subtract(f16[7,7,4,64] arg5.19.5, f16[7,7,4,64] multiply.19.88)
  ROOT tuple.19.98 = (f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(f16[4] subtract.19.65, f16[5,5,64,4] subtract.19.71, f16[64] subtract.19.85, f16[7,7,4,64] subtract.19.89)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3, 4, 5});
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ConvolutionClassifier classifier(annotations);

  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  EXPECT_EQ(annotations.classification_map.size(), 5);

  for (auto it : annotations.classification_map) {
    if (it.first->name() == "call.2.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_INPUT);
    } else if (it.first->name() == "convolution.19.17.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "convolution.19.14.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "convolution.19.66.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else if (it.first->name() == "convolution.19.86.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else {
      // Should not have missing convolutions
      EXPECT_EQ(1, 0);
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingInRepeat) {
  std::string hlo_string = R"(
HloModule top

add14 {
  x.14.0 = f32[] parameter(0)
  y.14.1 = f32[] parameter(1)
  ROOT add.14.2 = f32[] add(x.14.0, y.14.1)
}

max13 {
  x.13.0 = f32[] parameter(0)
  y.13.1 = f32[] parameter(1)
  ROOT maximum.13.2 = f32[] maximum(x.13.0, y.13.1)
}

_pop_op_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  broadcast.19.18.clone = f16[1,16,16,4] broadcast(f16[4] arg_1), dimensions={3}
  ROOT add.19.19.clone = f16[1,16,16,4] add(f16[1,16,16,4] arg_0, f16[1,16,16,4] broadcast.19.18.clone)
}

_pop_op_biasadd.1 {
  arg_0.1 = f16[1,16,16,64] parameter(0)
  arg_1.1 = f16[64] parameter(1)
  broadcast.19.15.clone = f16[1,16,16,64] broadcast(f16[64] arg_1.1), dimensions={3}
  ROOT add.19.16.clone = f16[1,16,16,64] add(f16[1,16,16,64] arg_0.1, f16[1,16,16,64] broadcast.19.15.clone)
}

_pop_op_wide_const {
  constant.19.10.clone = f16[] constant(0.0100021362)
  ROOT broadcast.19.63.clone = f16[4] broadcast(f16[] constant.19.10.clone), dimensions={}
}

_pop_op_wide_const.1  {
  constant.19.10.clone.1 = f16[] constant(0.0100021362)
  ROOT broadcast.19.69.clone = f16[5,5,64,4] broadcast(f16[] constant.19.10.clone.1), dimensions={}
}

_pop_op_wide_const.2  {
  constant.19.10.clone.2 = f16[] constant(0.0100021362)
  ROOT broadcast.19.83.clone = f16[64] broadcast(f16[] constant.19.10.clone.2), dimensions={}
}

_pop_op_wide_const.3 {
  constant.19.10.clone.3 = f16[] constant(0.0100021362)
  ROOT broadcast.19.87.clone = f16[7,7,4,64] broadcast(f16[] constant.19.10.clone.3), dimensions={}
}

_pop_op_conv_with_reverse {
  arg_0.2 = f16[1,16,16,4] parameter(0)
  arg_1.2 = f16[5,5,64,4] parameter(1)
  reverse.19.67.clone = f16[5,5,64,4] reverse(f16[5,5,64,4] arg_1.2), dimensions={0,1}
  ROOT convolution.19.68.clone = f16[1,16,16,64] convolution(f16[1,16,16,4] arg_0.2, f16[5,5,64,4] reverse.19.67.clone), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f
}

pop_backprop_conv {
  arg_0.3 = f16[1,16,16,4] parameter(0)
  arg_1.3 = f16[5,5,64,4] parameter(1)
  ROOT call.2.clone = f16[1,16,16,64] fusion(f16[1,16,16,4] arg_0.3, f16[5,5,64,4] arg_1.3), kind=kCustom, calls=_pop_op_conv_with_reverse
}

pop_convolution {
  arg_0.4 = f16[1,16,16,64] parameter(0)
  arg_1.4 = f16[5,5,64,4] parameter(1)
  ROOT convolution.19.17.clone = f16[1,16,16,4] convolution(f16[1,16,16,64] arg_0.4, f16[5,5,64,4] arg_1.4), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01io->b01f
}

pop_convolution.1 {
  arg_0.5 = f16[1,16,16,4] parameter(0)
  arg_1.5 = f16[7,7,4,64] parameter(1)
  ROOT convolution.19.14.clone = f16[1,16,16,64] convolution(f16[1,16,16,4] arg_0.5, f16[7,7,4,64] arg_1.5), window={size=7x7 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
}

pop_convolution.2 {
  arg_0.6 = f16[1,16,16,64] parameter(0)
  arg_1.6 = f16[1,16,16,4] parameter(1)
  ROOT convolution.19.66.clone = f16[5,5,64,4] convolution(f16[1,16,16,64] arg_0.6, f16[1,16,16,4] arg_1.6), window={size=16x16 pad=2_2x2_2}, dim_labels=f01b_i01o->01bf
}

pop_convolution.3 {
  arg_0.7 = f16[1,16,16,4] parameter(0)
  arg_1.7 = f16[1,16,16,64] parameter(1)
  ROOT convolution.19.86.clone = f16[7,7,4,64] convolution(f16[1,16,16,4] arg_0.7, f16[1,16,16,64] arg_1.7), window={size=16x16 pad=3_3x3_3}, dim_labels=f01b_i01o->01bf
}

__arithmetic_expression {
  arg_1.8 = f32[1,1024] parameter(1)
  arg_2 = f32[1,1024] parameter(2)
  divide.19.48.clone = f32[1,1024] divide(f32[1,1024] arg_1.8, f32[1,1024] arg_2)
  arg_0.8 = f32[1,1024] parameter(0)
  ROOT subtract.19.49.clone = f32[1,1024] subtract(f32[1,1024] divide.19.48.clone, f32[1,1024] arg_0.8)
}

loop_body {
  p = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) parameter(0)
  counter = s32[] get-tuple-element(p), index=0
  arg0.19.0 = f16[1,16,16,4] get-tuple-element(p), index=1
  arg1.19.1 = f16[1,1024] get-tuple-element(p), index=2
  arg2.19.2 = f16[4] get-tuple-element(p), index=3
  arg3.19.3 = f16[5,5,64,4] get-tuple-element(p), index=4
  arg4.19.4 = f16[64] get-tuple-element(p), index=5
  arg5.19.5 = f16[7,7,4,64] get-tuple-element(p), index=6
  call.3 = f16[4] fusion(), kind=kCustom, calls=_pop_op_wide_const
  convert.19.11 = f32[1,1024] convert(f16[1,1024] arg1.19.1)
  call.9 = f16[1,16,16,64] fusion(f16[1,16,16,4] arg0.19.0, f16[7,7,4,64] arg5.19.5), kind=kCustom, calls=pop_convolution.1
  call.1 = f16[1,16,16,64] fusion(f16[1,16,16,64] call.9, f16[64] arg4.19.4), kind=kCustom, calls=_pop_op_biasadd.1
  call.8 = f16[1,16,16,4] fusion(f16[1,16,16,64] call.1, f16[5,5,64,4] arg3.19.3), kind=kCustom, calls=pop_convolution
  call = f16[1,16,16,4] fusion(f16[1,16,16,4] call.8, f16[4] arg2.19.2), kind=kCustom, calls=_pop_op_biasadd
  convert = f32[1,16,16,4] convert(f16[1,16,16,4] call)
  reshape = f32[1,1024] reshape(f32[1,16,16,4] convert)
  constant.19.29 = f32[] constant(-inf)
  reduce = f32[1] reduce(f32[1,16,16,4] convert, constant.19.29), dimensions={1,2,3}, to_apply=max13
  broadcast.19.31 = f32[1,1024] broadcast(reduce), dimensions={0}
  subtract.19.32 = f32[1,1024] subtract(f32[1,1024] reshape, f32[1,1024] broadcast.19.31)
  exponential.19.33 = f32[1,1024] exponential(f32[1,1024] subtract.19.32)
  constant.19.35 = f32[] constant(0)
  reduce.19.36 = f32[1] reduce(f32[1,1024] exponential.19.33, constant.19.35), dimensions={1}, to_apply=add14
  broadcast.19.47 = f32[1,1024] broadcast(reduce.19.36), dimensions={0}
  call.12 = f32[1,1024] call(f32[1,1024] convert.19.11, f32[1,1024] exponential.19.33, f32[1,1024] broadcast.19.47), to_apply=__arithmetic_expression
  convert.19.50 = f16[1,1024] convert(f32[1,1024] call.12)
  convert.1 = f32[1,1024] convert(f16[1,1024] convert.19.50)
  reshape.1 = f32[1,16,16,4] reshape(f32[1,1024] convert.1)
  reduce.19.61 = f32[4] reduce(f32[1,16,16,4] reshape.1, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.62 = f16[4] convert(f32[4] reduce.19.61)
  multiply.19.64 = f16[4] multiply(f16[4] call.3, f16[4] convert.19.62)
  subtract.19.65 = f16[4] subtract(f16[4] arg2.19.2, f16[4] multiply.19.64)
  call.4 = f16[5,5,64,4] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  convert.3 = f16[1,1024] convert(f32[1,1024] convert.1)
  reshape.10 = f16[1,16,16,4] reshape(f16[1,1024] convert.3)
  call.10 = f16[5,5,64,4] fusion(f16[1,16,16,64] call.1, f16[1,16,16,4] reshape.10), kind=kCustom, calls=pop_convolution.2
  multiply.19.70 = f16[5,5,64,4] multiply(f16[5,5,64,4] call.4, f16[5,5,64,4] call.10)
  subtract.19.71 = f16[5,5,64,4] subtract(f16[5,5,64,4] arg3.19.3, f16[5,5,64,4] multiply.19.70)
  call.5 = f16[64] fusion(), kind=kCustom, calls=_pop_op_wide_const.2
  call.7 = f16[1,16,16,64] fusion(f16[1,16,16,4] reshape.10, f16[5,5,64,4] arg3.19.3), kind=kCustom, calls=pop_backprop_conv
  convert.19.78 = f32[1,16,16,64] convert(f16[1,16,16,64] call.7)
  reduce.19.81 = f32[64] reduce(f32[1,16,16,64] convert.19.78, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.82 = f16[64] convert(f32[64] reduce.19.81)
  multiply.19.84 = f16[64] multiply(f16[64] call.5, f16[64] convert.19.82)
  subtract.19.85 = f16[64] subtract(f16[64] arg4.19.4, f16[64] multiply.19.84)
  call.6 = f16[7,7,4,64] fusion(), kind=kCustom, calls=_pop_op_wide_const.3
  convert.2 = f16[1,16,16,64] convert(f32[1,16,16,64] convert.19.78)
  call.11 = f16[7,7,4,64] fusion(f16[1,16,16,4] arg0.19.0, f16[1,16,16,64] convert.2), kind=kCustom, calls=pop_convolution.3
  multiply.19.88 = f16[7,7,4,64] multiply(f16[7,7,4,64] call.6, f16[7,7,4,64] call.11)
  subtract.19.89 = f16[7,7,4,64] subtract(f16[7,7,4,64] arg5.19.5, f16[7,7,4,64] multiply.19.88)
  ROOT tuple.19.98 = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(counter, arg0.19.0, arg1.19.1, subtract.19.65, subtract.19.71, subtract.19.85, subtract.19.89)
}

__repeat {
  repeat_count = s32[] parameter(0)
  input_tuple = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) parameter(1)
  ROOT call = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) call(input_tuple), to_apply=loop_body
}

ENTRY in {
  c = s32[] constant(10)
  p0 = f16[1,16,16,4] parameter(0)
  p1 = f16[1,1024] parameter(1)
  p2 = f16[4] parameter(2)
  p3 = f16[5,5,64,4] parameter(3)
  p4 = f16[64] parameter(4)
  p5 = f16[7,7,4,64] parameter(5)
  in = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(c, p0, p1, p2, p3, p4, p5)
  ROOT r0 = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) call(c, in), to_apply=__repeat
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3, 4, 5});
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ConvolutionClassifier classifier(annotations);

  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  EXPECT_EQ(annotations.classification_map.size(), 5);

  for (auto it : annotations.classification_map) {
    if (it.first->name() == "call.2.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_INPUT);
    } else if (it.first->name() == "convolution.19.17.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "convolution.19.14.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "convolution.19.66.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else if (it.first->name() == "convolution.19.86.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else {
      // Should not have missing convolutions
      EXPECT_EQ(1, 0);
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingInfeedInRepeat) {
  std::string hlo_string = R"(
HloModule top

add14 {
  x.14.0 = f32[] parameter(0)
  y.14.1 = f32[] parameter(1)
  ROOT add.14.2 = f32[] add(x.14.0, y.14.1)
}

max13 {
  x.13.0 = f32[] parameter(0)
  y.13.1 = f32[] parameter(1)
  ROOT maximum.13.2 = f32[] maximum(x.13.0, y.13.1)
}

_pop_op_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  broadcast.19.18.clone = f16[1,16,16,4] broadcast(f16[4] arg_1), dimensions={3}
  ROOT add.19.19.clone = f16[1,16,16,4] add(f16[1,16,16,4] arg_0, f16[1,16,16,4] broadcast.19.18.clone)
}

_pop_op_biasadd.1 {
  arg_0.1 = f16[1,16,16,64] parameter(0)
  arg_1.1 = f16[64] parameter(1)
  broadcast.19.15.clone = f16[1,16,16,64] broadcast(f16[64] arg_1.1), dimensions={3}
  ROOT add.19.16.clone = f16[1,16,16,64] add(f16[1,16,16,64] arg_0.1, f16[1,16,16,64] broadcast.19.15.clone)
}

_pop_op_wide_const {
  constant.19.10.clone = f16[] constant(0.0100021362)
  ROOT broadcast.19.63.clone = f16[4] broadcast(f16[] constant.19.10.clone), dimensions={}
}

_pop_op_wide_const.1  {
  constant.19.10.clone.1 = f16[] constant(0.0100021362)
  ROOT broadcast.19.69.clone = f16[5,5,64,4] broadcast(f16[] constant.19.10.clone.1), dimensions={}
}

_pop_op_wide_const.2  {
  constant.19.10.clone.2 = f16[] constant(0.0100021362)
  ROOT broadcast.19.83.clone = f16[64] broadcast(f16[] constant.19.10.clone.2), dimensions={}
}

_pop_op_wide_const.3 {
  constant.19.10.clone.3 = f16[] constant(0.0100021362)
  ROOT broadcast.19.87.clone = f16[7,7,4,64] broadcast(f16[] constant.19.10.clone.3), dimensions={}
}

_pop_op_conv_with_reverse {
  arg_0.2 = f16[1,16,16,4] parameter(0)
  arg_1.2 = f16[5,5,64,4] parameter(1)
  reverse.19.67.clone = f16[5,5,64,4] reverse(f16[5,5,64,4] arg_1.2), dimensions={0,1}
  ROOT convolution.19.68.clone = f16[1,16,16,64] convolution(f16[1,16,16,4] arg_0.2, f16[5,5,64,4] reverse.19.67.clone), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f
}

pop_backprop_conv {
  arg_0.3 = f16[1,16,16,4] parameter(0)
  arg_1.3 = f16[5,5,64,4] parameter(1)
  ROOT call.2.clone = f16[1,16,16,64] fusion(f16[1,16,16,4] arg_0.3, f16[5,5,64,4] arg_1.3), kind=kCustom, calls=_pop_op_conv_with_reverse
}

pop_convolution {
  arg_0.4 = f16[1,16,16,64] parameter(0)
  arg_1.4 = f16[5,5,64,4] parameter(1)
  ROOT convolution.19.17.clone = f16[1,16,16,4] convolution(f16[1,16,16,64] arg_0.4, f16[5,5,64,4] arg_1.4), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01io->b01f
}

pop_convolution.1 {
  arg_0.5 = f16[1,16,16,4] parameter(0)
  arg_1.5 = f16[7,7,4,64] parameter(1)
  ROOT convolution.19.14.clone = f16[1,16,16,64] convolution(f16[1,16,16,4] arg_0.5, f16[7,7,4,64] arg_1.5), window={size=7x7 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
}

pop_convolution.2 {
  arg_0.6 = f16[1,16,16,64] parameter(0)
  arg_1.6 = f16[1,16,16,4] parameter(1)
  ROOT convolution.19.66.clone = f16[5,5,64,4] convolution(f16[1,16,16,64] arg_0.6, f16[1,16,16,4] arg_1.6), window={size=16x16 pad=2_2x2_2}, dim_labels=f01b_i01o->01bf
}

pop_convolution.3 {
  arg_0.7 = f16[1,16,16,4] parameter(0)
  arg_1.7 = f16[1,16,16,64] parameter(1)
  ROOT convolution.19.86.clone = f16[7,7,4,64] convolution(f16[1,16,16,4] arg_0.7, f16[1,16,16,64] arg_1.7), window={size=16x16 pad=3_3x3_3}, dim_labels=f01b_i01o->01bf
}

__arithmetic_expression {
  arg_1.8 = f32[1,1024] parameter(1)
  arg_2 = f32[1,1024] parameter(2)
  divide.19.48.clone = f32[1,1024] divide(f32[1,1024] arg_1.8, f32[1,1024] arg_2)
  arg_0.8 = f32[1,1024] parameter(0)
  ROOT subtract.19.49.clone = f32[1,1024] subtract(f32[1,1024] divide.19.48.clone, f32[1,1024] arg_0.8)
}

loop_body {
  p = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) parameter(0)
  counter = s32[] get-tuple-element(p), index=0
  arg2.19.2 = f16[4] get-tuple-element(p), index=1
  arg3.19.3 = f16[5,5,64,4] get-tuple-element(p), index=2
  arg4.19.4 = f16[64] get-tuple-element(p), index=3
  arg5.19.5 = f16[7,7,4,64] get-tuple-element(p), index=4
  after-all = token[] after-all()
  infeed = ((f16[1,16,16,4], f16[1,1024]), token[]) infeed(after-all), infeed_config="01234567"
  infeed_tuple = (f16[1,16,16,4], f16[1,1024]) get-tuple-element(infeed), index=0
  arg0.19.0 = f16[1,16,16,4] get-tuple-element(infeed_tuple), index=0
  arg1.19.1 = f16[1,1024] get-tuple-element(infeed_tuple), index=0
  call.3 = f16[4] fusion(), kind=kCustom, calls=_pop_op_wide_const
  convert.19.11 = f32[1,1024] convert(f16[1,1024] arg1.19.1)
  call.9 = f16[1,16,16,64] fusion(f16[1,16,16,4] arg0.19.0, f16[7,7,4,64] arg5.19.5), kind=kCustom, calls=pop_convolution.1
  call.1 = f16[1,16,16,64] fusion(f16[1,16,16,64] call.9, f16[64] arg4.19.4), kind=kCustom, calls=_pop_op_biasadd.1
  call.8 = f16[1,16,16,4] fusion(f16[1,16,16,64] call.1, f16[5,5,64,4] arg3.19.3), kind=kCustom, calls=pop_convolution
  call = f16[1,16,16,4] fusion(f16[1,16,16,4] call.8, f16[4] arg2.19.2), kind=kCustom, calls=_pop_op_biasadd
  convert = f32[1,16,16,4] convert(f16[1,16,16,4] call)
  reshape = f32[1,1024] reshape(f32[1,16,16,4] convert)
  constant.19.29 = f32[] constant(-inf)
  reduce = f32[1] reduce(f32[1,16,16,4] convert, constant.19.29), dimensions={1,2,3}, to_apply=max13
  broadcast.19.31 = f32[1,1024] broadcast(reduce), dimensions={0}
  subtract.19.32 = f32[1,1024] subtract(f32[1,1024] reshape, f32[1,1024] broadcast.19.31)
  exponential.19.33 = f32[1,1024] exponential(f32[1,1024] subtract.19.32)
  constant.19.35 = f32[] constant(0)
  reduce.19.36 = f32[1] reduce(f32[1,1024] exponential.19.33, constant.19.35), dimensions={1}, to_apply=add14
  broadcast.19.47 = f32[1,1024] broadcast(reduce.19.36), dimensions={0}
  call.12 = f32[1,1024] call(f32[1,1024] convert.19.11, f32[1,1024] exponential.19.33, f32[1,1024] broadcast.19.47), to_apply=__arithmetic_expression
  convert.19.50 = f16[1,1024] convert(f32[1,1024] call.12)
  convert.1 = f32[1,1024] convert(f16[1,1024] convert.19.50)
  reshape.1 = f32[1,16,16,4] reshape(f32[1,1024] convert.1)
  reduce.19.61 = f32[4] reduce(f32[1,16,16,4] reshape.1, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.62 = f16[4] convert(f32[4] reduce.19.61)
  multiply.19.64 = f16[4] multiply(f16[4] call.3, f16[4] convert.19.62)
  subtract.19.65 = f16[4] subtract(f16[4] arg2.19.2, f16[4] multiply.19.64)
  call.4 = f16[5,5,64,4] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  convert.3 = f16[1,1024] convert(f32[1,1024] convert.1)
  reshape.10 = f16[1,16,16,4] reshape(f16[1,1024] convert.3)
  call.10 = f16[5,5,64,4] fusion(f16[1,16,16,64] call.1, f16[1,16,16,4] reshape.10), kind=kCustom, calls=pop_convolution.2
  multiply.19.70 = f16[5,5,64,4] multiply(f16[5,5,64,4] call.4, f16[5,5,64,4] call.10)
  subtract.19.71 = f16[5,5,64,4] subtract(f16[5,5,64,4] arg3.19.3, f16[5,5,64,4] multiply.19.70)
  call.5 = f16[64] fusion(), kind=kCustom, calls=_pop_op_wide_const.2
  call.7 = f16[1,16,16,64] fusion(f16[1,16,16,4] reshape.10, f16[5,5,64,4] arg3.19.3), kind=kCustom, calls=pop_backprop_conv
  convert.19.78 = f32[1,16,16,64] convert(f16[1,16,16,64] call.7)
  reduce.19.81 = f32[64] reduce(f32[1,16,16,64] convert.19.78, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.82 = f16[64] convert(f32[64] reduce.19.81)
  multiply.19.84 = f16[64] multiply(f16[64] call.5, f16[64] convert.19.82)
  subtract.19.85 = f16[64] subtract(f16[64] arg4.19.4, f16[64] multiply.19.84)
  call.6 = f16[7,7,4,64] fusion(), kind=kCustom, calls=_pop_op_wide_const.3
  convert.2 = f16[1,16,16,64] convert(f32[1,16,16,64] convert.19.78)
  call.11 = f16[7,7,4,64] fusion(f16[1,16,16,4] arg0.19.0, f16[1,16,16,64] convert.2), kind=kCustom, calls=pop_convolution.3
  multiply.19.88 = f16[7,7,4,64] multiply(f16[7,7,4,64] call.6, f16[7,7,4,64] call.11)
  subtract.19.89 = f16[7,7,4,64] subtract(f16[7,7,4,64] arg5.19.5, f16[7,7,4,64] multiply.19.88)
  ROOT tuple.19.98 = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(counter, subtract.19.65, subtract.19.71, subtract.19.85, subtract.19.89)
}

__repeat {
  repeat_count = s32[] parameter(0)
  input_tuple = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) parameter(1)
  ROOT call = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) call(input_tuple), to_apply=loop_body
}

ENTRY in {
  c = s32[] constant(10)
  p0 = f16[4] parameter(0)
  p1 = f16[5,5,64,4] parameter(1)
  p2 = f16[64] parameter(2)
  p3 = f16[7,7,4,64] parameter(3)
  in = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(c, p0, p1, p2, p3)
  ROOT r0 = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) call(c, in), to_apply=__repeat
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3, 4, 5});
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ConvolutionClassifier classifier(annotations);

  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  EXPECT_EQ(annotations.classification_map.size(), 5);

  for (auto it : annotations.classification_map) {
    if (it.first->name() == "call.2.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_INPUT);
    } else if (it.first->name() == "convolution.19.17.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "convolution.19.14.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "convolution.19.66.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else if (it.first->name() == "convolution.19.86.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else {
      // Should not have missing convolutions
      EXPECT_EQ(1, 0);
    }
  }
}

TEST_F(ConvolutionClassifierTest, SingleConvTraining) {
  std::string hlo_string = R"(
  HloModule top

  Mean-reduction4 {
    x.4.0 = f32[] parameter(0)
    y.4.1 = f32[] parameter(1)
    ROOT add.4.2 = f32[] add(x.4.0, y.4.1)
  }

  max5 {
    x.5.0 = f32[] parameter(0)
    y.5.1 = f32[] parameter(1)
    ROOT maximum.5.2 = f32[] maximum(x.5.0, y.5.1)
  }

  _pop_op_wide_const {
    constant.7.6.clone = f32[] constant(0.01)
    ROOT broadcast.7.52.clone = f32[3,3,4,12] broadcast(constant.7.6.clone), dimensions={}
  }

  _pop_op_wide_const.1 {
    constant.7.19.clone = f32[] constant(576)
    ROOT broadcast.7.20.clone = f32[1,12] broadcast(constant.7.19.clone), dimensions={}
  }

  _pop_op_wide_const.2 {
    constant.7.5.clone = f32[] constant(0.00173611112)
    ROOT broadcast.7.49.clone = f32[1,24,24,12] broadcast(constant.7.5.clone), dimensions={}
  }

  pop_convolution {
    arg_0 = f32[1,24,24,4] parameter(0)
    arg_1 = f32[1,24,24,12] parameter(1)
    ROOT convolution.7.51.clone = f32[3,3,4,12] convolution(f32[1,24,24,4] arg_0, f32[1,24,24,12] arg_1), window={size=24x24 pad=1_1x1_1}, dim_labels=f01b_i01o->01bf
  }

  pop_convolution.1 {
    arg_0.1 = f32[1,24,24,4] parameter(0)
    arg_1.1 = f32[3,3,4,12] parameter(1)
    ROOT convolution.7.13.clone = f32[1,24,24,12] convolution(f32[1,24,24,4] arg_0.1, f32[3,3,4,12] arg_1.1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  }

  __arithmetic_expression {
    arg_1.2 = f32[1,12] parameter(1)
    arg_0.2 = f32[1,12] parameter(0)
    divide.7.41.clone = f32[1,12] divide(arg_1.2, arg_0.2)
    arg_2 = f32[1,12] parameter(2)
    ROOT subtract.7.42.clone = f32[1,12] subtract(divide.7.41.clone, arg_2)
  }

  ENTRY cluster_1 {
    arg2.7.2 = f32[3,3,4,12] parameter(2)
    call = f32[3,3,4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
    arg0.7.0 = f32[1,24,24,4] parameter(0)
    call.4 = f32[1,24,24,12] fusion(f32[1,24,24,4] arg0.7.0, f32[3,3,4,12] arg2.7.2), kind=kCustom, calls=pop_convolution.1
    constant.7.15 = f32[] constant(0)
    reduce.7.17 = f32[1,12] reduce(f32[1,24,24,12] call.4, constant.7.15), dimensions={1,2}, to_apply=Mean-reduction4
    call.1 = f32[1,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
    divide.7.21 = f32[1,12] divide(reduce.7.17, call.1)
    constant.7.22 = f32[] constant(-inf)
    reduce.7.23 = f32[1] reduce(divide.7.21, constant.7.22), dimensions={1}, to_apply=max5
    broadcast.7.24 = f32[1,12] broadcast(reduce.7.23), dimensions={0}
    subtract.7.25 = f32[1,12] subtract(divide.7.21, broadcast.7.24)
    exponential.7.26 = f32[1,12] exponential(subtract.7.25)
    reduce.7.29 = f32[1] reduce(exponential.7.26, constant.7.15), dimensions={1}, to_apply=Mean-reduction4
    broadcast.7.40 = f32[1,12] broadcast(reduce.7.29), dimensions={0}
    arg1.7.1 = f32[1,12] parameter(1)
    call.5 = f32[1,12] call(broadcast.7.40, exponential.7.26, arg1.7.1), to_apply=__arithmetic_expression
    broadcast = f32[1,24,24,12] broadcast(call.5), dimensions={0,3}
    call.2 = f32[1,24,24,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.2
    multiply.7.50 = f32[1,24,24,12] multiply(f32[1,24,24,12] broadcast, f32[1,24,24,12] call.2)
    call.3 = f32[3,3,4,12] fusion(f32[1,24,24,4] arg0.7.0, f32[1,24,24,12] multiply.7.50), kind=kCustom, calls=pop_convolution
    multiply.7.53 = f32[3,3,4,12] multiply(f32[3,3,4,12] call, f32[3,3,4,12] call.3)
    subtract.7.54 = f32[3,3,4,12] subtract(f32[3,3,4,12] arg2.7.2, f32[3,3,4,12] multiply.7.53)
    ROOT tuple.7.57 = (f32[3,3,4,12]) tuple(f32[3,3,4,12] subtract.7.54)
  }
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2});
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ConvolutionClassifier classifier(annotations);

  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  EXPECT_EQ(annotations.classification_map.size(), 2);

  for (auto it : annotations.classification_map) {
    if (it.first->name() == "convolution.7.51.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else if (it.first->name() == "convolution.7.13.clone") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else {
      // Should not have missing convolutions
      EXPECT_EQ(1, 0);
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingMatMul) {
  std::string hlo_string = R"(
HloModule top

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0)
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0)
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0)
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}
  greater-than.9.44.clone = pred[1,12] greater-than(arg_0.1, broadcast.9.11.clone.1)
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(pred[1,12] greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1)
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01)
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01)
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}
}

ENTRY cluster_1 {
  arg2.9.2 = f32[12,12] parameter(2)
  call.2 = f32[12,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
  arg0.9.0 = f32[1,4] parameter(0)
  arg3.9.3 = f32[4,12] parameter(3)
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call = f32[1,12] fusion(dot.9.9), kind=kCustom, calls=_pop_op_relu
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.9.14 = f32[] constant(-inf)
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16)
  exponential.9.18 = f32[1,12] exponential(subtract.9.17)
  constant.9.10 = f32[] constant(0)
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32)
  arg1.9.1 = f32[1,12] parameter(1)
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1)
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36)
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40)
  call.3 = f32[4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call.1 = f32[1,12] fusion(call, dot.9.38), kind=kCustom, calls=_pop_op_relugrad
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47)
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49)
  ROOT tuple.9.55 = (f32[12,12], f32[4,12]) tuple(subtract.9.41, subtract.9.50)
}

)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3});
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ConvolutionClassifier classifier(annotations);

  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  EXPECT_EQ(annotations.classification_map.size(), 5);

  for (auto it : annotations.classification_map) {
    if (it.first->name() == "dot.9.9") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "dot.9.13") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "dot.9.36") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else if (it.first->name() == "dot.9.38") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_INPUT);
    } else if (it.first->name() == "dot.9.47") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else {
      // Should not have missing matmuls
      EXPECT_EQ(1, 0);
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingMatMulInRepeat) {
  std::string hlo_string = R"(
HloModule top

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0)
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0)
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0)
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}
  greater-than.9.44.clone = pred[1,12] greater-than(arg_0.1, broadcast.9.11.clone.1)
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(pred[1,12] greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1)
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01)
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01)
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}
}

loop_body {
  p0 = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) parameter(0)
  counter = s32[] get-tuple-element(p0), index=0
  arg0.9.0 = f32[1,4] get-tuple-element(p0), index=1
  arg1.9.1 = f32[1,12] get-tuple-element(p0), index=2
  arg2.9.2 = f32[12,12] get-tuple-element(p0), index=3
  arg3.9.3 = f32[4,12] get-tuple-element(p0), index=4
  call.2 = f32[12,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call = f32[1,12] fusion(dot.9.9), kind=kCustom, calls=_pop_op_relu
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.9.14 = f32[] constant(-inf)
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16)
  exponential.9.18 = f32[1,12] exponential(subtract.9.17)
  constant.9.10 = f32[] constant(0)
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32)
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1)
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36)
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40)
  call.3 = f32[4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call.1 = f32[1,12] fusion(call, dot.9.38), kind=kCustom, calls=_pop_op_relugrad
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47)
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49)
  ROOT tuple.9.55 = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) tuple(counter, arg0.9.0, arg1.9.1, subtract.9.41, subtract.9.50)
}

__repeat {
  repeat_count = s32[] parameter(0)
  input_tuple = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) parameter(1)
  ROOT call = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) call(input_tuple), to_apply=loop_body
}

ENTRY in {
  c = s32[] constant(10)
  p0 = f32[1,4] parameter(0)
  p1 = f32[1,12] parameter(1)
  p2 = f32[12,12] parameter(2)
  p3 = f32[4,12] parameter(3)
  in = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) tuple(c, p0, p1, p2, p3)
  ROOT r0 = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) call(c, in), to_apply=__repeat
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3});
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ConvolutionClassifier classifier(annotations);

  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  EXPECT_EQ(annotations.classification_map.size(), 5);

  for (auto it : annotations.classification_map) {
    if (it.first->name() == "dot.9.9") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "dot.9.13") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "dot.9.36") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else if (it.first->name() == "dot.9.38") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_INPUT);
    } else if (it.first->name() == "dot.9.47") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else {
      // Should not have missing matmuls
      EXPECT_EQ(1, 0);
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingInfeedMatMulInRepeat) {
  std::string hlo_string = R"(
HloModule top

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0)
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0)
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0)
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}
  greater-than.9.44.clone = pred[1,12] greater-than(arg_0.1, broadcast.9.11.clone.1)
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(pred[1,12] greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1)
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01)
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01)
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}
}

loop_body {
  p0 = (s32[], f32[12,12], f32[4,12]) parameter(0)
  counter = s32[] get-tuple-element(p0), index=0
  arg2.9.2 = f32[12,12] get-tuple-element(p0), index=1
  arg3.9.3 = f32[4,12] get-tuple-element(p0), index=2
  after-all = token[] after-all()
  infeed = ((f32[1,4], f32[1,12]), token[]) infeed(after-all), infeed_config="01234567"
  infeed_tuple = (f32[1,4], f32[1,12]) get-tuple-element(infeed), index=0
  arg0.9.0 = f16[1,4] get-tuple-element(infeed_tuple), index=0
  arg1.9.1 = f16[1,12] get-tuple-element(infeed_tuple), index=0
  call.2 = f32[12,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call = f32[1,12] fusion(dot.9.9), kind=kCustom, calls=_pop_op_relu
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.9.14 = f32[] constant(-inf)
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16)
  exponential.9.18 = f32[1,12] exponential(subtract.9.17)
  constant.9.10 = f32[] constant(0)
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32)
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1)
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36)
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40)
  call.3 = f32[4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call.1 = f32[1,12] fusion(call, dot.9.38), kind=kCustom, calls=_pop_op_relugrad
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47)
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49)
  ROOT tuple.9.55 = (s32[], f32[12,12], f32[4,12]) tuple(counter, subtract.9.41, subtract.9.50)
}

__repeat {
  repeat_count = s32[] parameter(0)
  input_tuple = (s32[], f32[12,12], f32[4,12]) parameter(1)
  ROOT call = (s32[], f32[12,12], f32[4,12]) call(input_tuple), to_apply=loop_body
}

ENTRY in {
  c = s32[] constant(10)
  p0 = f32[12,12] parameter(0)
  p1 = f32[4,12] parameter(1)
  in = (s32[], f32[12,12], f32[4,12]) tuple(c, p0, p1)
  ROOT r0 = (s32[], f32[12,12], f32[4,12]) call(c, in), to_apply=__repeat
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3});
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ConvolutionClassifier classifier(annotations);

  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  EXPECT_EQ(annotations.classification_map.size(), 5);

  for (auto it : annotations.classification_map) {
    if (it.first->name() == "dot.9.9") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "dot.9.13") {
      EXPECT_EQ(it.second, ConvClassificationType::FORWARD);
    } else if (it.first->name() == "dot.9.36") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else if (it.first->name() == "dot.9.38") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_INPUT);
    } else if (it.first->name() == "dot.9.47") {
      EXPECT_EQ(it.second, ConvClassificationType::BACKPROP_FILTER);
    } else {
      // Should not have missing matmuls
      EXPECT_EQ(1, 0);
    }
  }
}

TEST_F(ConvolutionClassifierTest, InferenceMatMul) {
  std::string hlo_string = R"(
HloModule top

_pop_op_relu {
  constant.17.9.clone = f32[] constant(0)
  broadcast.17.10.clone = f32[32,32] broadcast(constant.17.9.clone), dimensions={}
  arg_0 = f32[32,32] parameter(0)
  ROOT maximum.17.11.clone = f32[32,32] maximum(broadcast.17.10.clone, arg_0)
}

_pop_op_sigmoid {
  constant.17.15.clone = f32[] constant(0.5)
  broadcast.17.21.clone = f32[32,1] broadcast(constant.17.15.clone), dimensions={}
  arg_0.1 = f32[32,1] parameter(0)
  multiply.17.17.clone = f32[32,1] multiply(broadcast.17.21.clone, arg_0.1)
  tanh.17.18.clone = f32[32,1] tanh(multiply.17.17.clone)
  multiply.17.20.clone = f32[32,1] multiply(broadcast.17.21.clone, tanh.17.18.clone)
  ROOT add.17.22.clone = f32[32,1] add(broadcast.17.21.clone, multiply.17.20.clone)
}

ENTRY cluster_9 {
  arg0.17.0 = f32[32,100] parameter(0)
  arg4.17.4 = f32[100,32] parameter(4)
  dot.17.6 = f32[32,32] dot(arg0.17.0, arg4.17.4), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  arg3.17.3 = f32[32] parameter(3)
  broadcast.17.7 = f32[32,32] broadcast(arg3.17.3), dimensions={1}
  add.17.8 = f32[32,32] add(dot.17.6, broadcast.17.7)
  call = f32[32,32] fusion(add.17.8), kind=kCustom, calls=_pop_op_relu
  arg2.17.2 = f32[32,1] parameter(2)
  dot.17.12 = f32[32,1] dot(call, arg2.17.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  arg1.17.1 = f32[1] parameter(1)
  broadcast.17.13 = f32[32,1] broadcast(arg1.17.1), dimensions={1}
  add.17.14 = f32[32,1] add(dot.17.12, broadcast.17.13)
  call.1 = f32[32,1] fusion(add.17.14), kind=kCustom, calls=_pop_op_sigmoid
  ROOT tuple.17.24 = (f32[32,1]) tuple(call.1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({});
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ConvolutionClassifier classifier(annotations);

  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  EXPECT_EQ(annotations.classification_map.size(), 2);

  for (auto it : annotations.classification_map) {
    if (it.first->name() == "dot.17.12") {
      EXPECT_EQ(it.second, ConvClassificationType::INFERENCE);
    } else if (it.first->name() == "dot.17.6") {
      EXPECT_EQ(it.second, ConvClassificationType::INFERENCE);
    } else {
      // Should not have missing matmuls
      EXPECT_EQ(1, 0);
    }
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
