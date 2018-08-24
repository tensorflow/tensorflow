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

%add14 {
  %x.14.0 = f32[] parameter(0)
  %y.14.1 = f32[] parameter(1)
  ROOT %add.14.2 = f32[] add(f32[] %x.14.0, f32[] %y.14.1)
}

%max13 {
  %x.13.0 = f32[] parameter(0)
  %y.13.1 = f32[] parameter(1)
  ROOT %maximum.13.2 = f32[] maximum(f32[] %x.13.0, f32[] %y.13.1)
}

%_pop_op_biasadd {
  %arg_0 = f16[1,16,16,4]{3,2,1,0} parameter(0)
  %arg_1 = f16[4]{0} parameter(1)
  %broadcast.19.18.clone = f16[1,16,16,4]{3,2,1,0} broadcast(f16[4]{0} %arg_1), dimensions={3}, metadata={op_type="Add" op_name="all/add_1"}
  ROOT %add.19.19.clone = f16[1,16,16,4]{3,2,1,0} add(f16[1,16,16,4]{3,2,1,0} %arg_0, f16[1,16,16,4]{3,2,1,0} %broadcast.19.18.clone), metadata={op_type="Add" op_name="all/add_1"}
}

%_pop_op_biasadd.1 {
  %arg_0.1 = f16[1,16,16,64]{3,2,1,0} parameter(0)
  %arg_1.1 = f16[64]{0} parameter(1)
  %broadcast.19.15.clone = f16[1,16,16,64]{3,2,1,0} broadcast(f16[64]{0} %arg_1.1), dimensions={3}, metadata={op_type="Add" op_name="all/add"}
  ROOT %add.19.16.clone = f16[1,16,16,64]{3,2,1,0} add(f16[1,16,16,64]{3,2,1,0} %arg_0.1, f16[1,16,16,64]{3,2,1,0} %broadcast.19.15.clone), metadata={op_type="Add" op_name="all/add"}
}

%_pop_op_wide_const {
  %constant.19.10.clone = f16[] constant(0.0100021362), metadata={op_type="Const" op_name="GradientDescent/update_all/1bias/Cast"}
  ROOT %broadcast.19.63.clone = f16[4]{0} broadcast(f16[] %constant.19.10.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/2bias/ResourceApplyGradientDescent"}
}

%_pop_op_wide_const.1  {
  %constant.19.10.clone.1 = f16[] constant(0.0100021362), metadata={op_type="Const" op_name="GradientDescent/update_all/1bias/Cast"}
  ROOT %broadcast.19.69.clone = f16[5,5,64,4]{3,2,1,0} broadcast(f16[] %constant.19.10.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/2weights/ResourceApplyGradientDescent"}
}

%_pop_op_wide_const.2  {
  %constant.19.10.clone.2 = f16[] constant(0.0100021362), metadata={op_type="Const" op_name="GradientDescent/update_all/1bias/Cast"}
  ROOT %broadcast.19.83.clone = f16[64]{0} broadcast(f16[] %constant.19.10.clone.2), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/1bias/ResourceApplyGradientDescent"}
}

%_pop_op_wide_const.3 {
  %constant.19.10.clone.3 = f16[] constant(0.0100021362), metadata={op_type="Const" op_name="GradientDescent/update_all/1bias/Cast"}
  ROOT %broadcast.19.87.clone = f16[7,7,4,64]{3,2,1,0} broadcast(f16[] %constant.19.10.clone.3), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/1weights/ResourceApplyGradientDescent"}
}

%_pop_op_conv_with_reverse {
  %arg_0.2 = f16[1,16,16,4]{3,2,1,0} parameter(0)
  %arg_1.2 = f16[5,5,64,4]{3,2,1,0} parameter(1)
  %reverse.19.67.clone = f16[5,5,64,4]{3,2,1,0} reverse(f16[5,5,64,4]{3,2,1,0} %arg_1.2), dimensions={0,1}, metadata={op_type="Conv2DBackpropInput" op_name="gradients/all/Conv2D_1_grad/Conv2DBackpropInput"}
  ROOT %convolution.19.68.clone = f16[1,16,16,64]{3,2,1,0} convolution(f16[1,16,16,4]{3,2,1,0} %arg_0.2, f16[5,5,64,4]{3,2,1,0} %reverse.19.67.clone), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f, metadata={op_type="Conv2DBackpropInput" op_name="gradients/all/Conv2D_1_grad/Conv2DBackpropInput"}
}

%pop_backprop_conv {
  %arg_0.3 = f16[1,16,16,4]{3,2,1,0} parameter(0)
  %arg_1.3 = f16[5,5,64,4]{3,2,1,0} parameter(1)
  ROOT %call.2.clone = f16[1,16,16,64]{3,2,1,0} call(f16[1,16,16,4]{3,2,1,0} %arg_0.3, f16[5,5,64,4]{3,2,1,0} %arg_1.3), to_apply=%_pop_op_conv_with_reverse, metadata={op_type="Conv2DBackpropInput" op_name="gradients/all/Conv2D_1_grad/Conv2DBackpropInput"}
}

%pop_convolution {
  %arg_0.4 = f16[1,16,16,64]{3,2,1,0} parameter(0)
  %arg_1.4 = f16[5,5,64,4]{3,2,1,0} parameter(1)
  ROOT %convolution.19.17.clone = f16[1,16,16,4]{3,2,1,0} convolution(f16[1,16,16,64]{3,2,1,0} %arg_0.4, f16[5,5,64,4]{3,2,1,0} %arg_1.4), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="all/Conv2D_1"}
}

%pop_convolution.1 {
  %arg_0.5 = f16[1,16,16,4]{3,2,1,0} parameter(0)
  %arg_1.5 = f16[7,7,4,64]{3,2,1,0} parameter(1)
  ROOT %convolution.19.14.clone = f16[1,16,16,64]{3,2,1,0} convolution(f16[1,16,16,4]{3,2,1,0} %arg_0.5, f16[7,7,4,64]{3,2,1,0} %arg_1.5), window={size=7x7 pad=3_3x3_3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="all/Conv2D"}
}

%pop_convolution.2 {
  %arg_0.6 = f16[1,16,16,64]{3,2,1,0} parameter(0)
  %arg_1.6 = f16[1,16,16,4]{3,2,1,0} parameter(1)
  ROOT %convolution.19.66.clone = f16[5,5,64,4]{3,2,1,0} convolution(f16[1,16,16,64]{3,2,1,0} %arg_0.6, f16[1,16,16,4]{3,2,1,0} %arg_1.6), window={size=16x16 pad=2_2x2_2}, dim_labels=f01b_i01o->01bf, metadata={op_type="Conv2DBackpropFilter" op_name="gradients/all/Conv2D_1_grad/Conv2DBackpropFilter"}
}

%pop_convolution.3 {
  %arg_0.7 = f16[1,16,16,4]{3,2,1,0} parameter(0)
  %arg_1.7 = f16[1,16,16,64]{3,2,1,0} parameter(1)
  ROOT %convolution.19.86.clone = f16[7,7,4,64]{3,2,1,0} convolution(f16[1,16,16,4]{3,2,1,0} %arg_0.7, f16[1,16,16,64]{3,2,1,0} %arg_1.7), window={size=16x16 pad=3_3x3_3}, dim_labels=f01b_i01o->01bf, metadata={op_type="Conv2DBackpropFilter" op_name="gradients/all/Conv2D_grad/Conv2DBackpropFilter"}
}

%__arithmetic_expression {
  %arg_1.8 = f32[1,1024]{1,0} parameter(1)
  %arg_2 = f32[1,1024]{1,0} parameter(2)
  %divide.19.48.clone = f32[1,1024]{1,0} divide(f32[1,1024]{1,0} %arg_1.8, f32[1,1024]{1,0} %arg_2), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %arg_0.8 = f32[1,1024]{1,0} parameter(0)
  ROOT %subtract.19.49.clone = f32[1,1024]{1,0} subtract(f32[1,1024]{1,0} %divide.19.48.clone, f32[1,1024]{1,0} %arg_0.8), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
}

%cluster_1  {
  %arg2.19.2 = f16[4]{0} parameter(2), metadata={op_name="XLA_Args"}
  %call.3 = f16[4]{0} call(), to_apply=%_pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/update_all/1bias/Cast"}
  %arg1.19.1 = f16[1,1024]{1,0} parameter(1), metadata={op_name="XLA_Args"}
  %convert.19.11 = f32[1,1024]{1,0} convert(f16[1,1024]{1,0} %arg1.19.1), metadata={op_type="Cast" op_name="softmax_cross_entropy_with_logits_sg/Cast_1"}
  %arg0.19.0 = f16[1,16,16,4]{3,2,1,0} parameter(0), metadata={op_name="XLA_Args"}
  %arg5.19.5 = f16[7,7,4,64]{3,2,1,0} parameter(5), metadata={op_name="XLA_Args"}
  %call.9 = f16[1,16,16,64]{3,2,1,0} call(f16[1,16,16,4]{3,2,1,0} %arg0.19.0, f16[7,7,4,64]{3,2,1,0} %arg5.19.5), to_apply=%pop_convolution.1, metadata={op_type="Conv2D" op_name="all/Conv2D"}
  %arg4.19.4 = f16[64]{0} parameter(4), metadata={op_name="XLA_Args"}
  %call.1 = f16[1,16,16,64]{3,2,1,0} call(f16[1,16,16,64]{3,2,1,0} %call.9, f16[64]{0} %arg4.19.4), to_apply=%_pop_op_biasadd.1, metadata={op_type="Add" op_name="all/add"}
  %arg3.19.3 = f16[5,5,64,4]{3,2,1,0} parameter(3), metadata={op_name="XLA_Args"}
  %call.8 = f16[1,16,16,4]{3,2,1,0} call(f16[1,16,16,64]{3,2,1,0} %call.1, f16[5,5,64,4]{3,2,1,0} %arg3.19.3), to_apply=%pop_convolution, metadata={op_type="Conv2D" op_name="all/Conv2D_1"}
  %call = f16[1,16,16,4]{3,2,1,0} call(f16[1,16,16,4]{3,2,1,0} %call.8, f16[4]{0} %arg2.19.2), to_apply=%_pop_op_biasadd, metadata={op_type="Add" op_name="all/add_1"}
  %convert = f32[1,16,16,4]{3,2,1,0} convert(f16[1,16,16,4]{3,2,1,0} %call), metadata={op_type="Cast" op_name="softmax_cross_entropy_with_logits_sg/Cast"}
  %reshape = f32[1,1024]{1,0} reshape(f32[1,16,16,4]{3,2,1,0} %convert)
  %constant.19.29 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %reduce = f32[1]{0} reduce(f32[1,16,16,4]{3,2,1,0} %convert, f32[] %constant.19.29), dimensions={1,2,3}, to_apply=%max13, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.19.31 = f32[1,1024]{1,0} broadcast(f32[1]{0} %reduce), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %subtract.19.32 = f32[1,1024]{1,0} subtract(f32[1,1024]{1,0} %reshape, f32[1,1024]{1,0} %broadcast.19.31), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %exponential.19.33 = f32[1,1024]{1,0} exponential(f32[1,1024]{1,0} %subtract.19.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %constant.19.35 = f32[] constant(0), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %reduce.19.36 = f32[1]{0} reduce(f32[1,1024]{1,0} %exponential.19.33, f32[] %constant.19.35), dimensions={1}, to_apply=%add14, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.19.47 = f32[1,1024]{1,0} broadcast(f32[1]{0} %reduce.19.36), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %call.12 = f32[1,1024]{1,0} call(f32[1,1024]{1,0} %convert.19.11, f32[1,1024]{1,0} %exponential.19.33, f32[1,1024]{1,0} %broadcast.19.47), to_apply=%__arithmetic_expression, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %convert.19.50 = f16[1,1024]{1,0} convert(f32[1,1024]{1,0} %call.12), metadata={op_type="Cast" op_name="gradients/softmax_cross_entropy_with_logits_sg/Cast_grad/Cast"}
  %convert.1 = f32[1,1024]{1,0} convert(f16[1,1024]{1,0} %convert.19.50), metadata={op_type="Sum" op_name="gradients/all/add_1_grad/Sum_1"}
  %reshape.1 = f32[1,16,16,4]{3,2,1,0} reshape(f32[1,1024]{1,0} %convert.1)
  %reduce.19.61 = f32[4]{0} reduce(f32[1,16,16,4]{3,2,1,0} %reshape.1, f32[] %constant.19.35), dimensions={0,1,2}, to_apply=%add14, metadata={op_type="Sum" op_name="gradients/all/add_1_grad/Sum_1"}
  %convert.19.62 = f16[4]{0} convert(f32[4]{0} %reduce.19.61), metadata={op_type="Sum" op_name="gradients/all/add_1_grad/Sum_1"}
  %multiply.19.64 = f16[4]{0} multiply(f16[4]{0} %call.3, f16[4]{0} %convert.19.62), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/2bias/ResourceApplyGradientDescent"}
  %subtract.19.65 = f16[4]{0} subtract(f16[4]{0} %arg2.19.2, f16[4]{0} %multiply.19.64), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/2bias/ResourceApplyGradientDescent"}
  %call.4 = f16[5,5,64,4]{3,2,1,0} call(), to_apply=%_pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/update_all/1bias/Cast"}
  %convert.3 = f16[1,1024]{1,0} convert(f32[1,1024]{1,0} %convert.1), metadata={op_type="Sum" op_name="gradients/all/add_1_grad/Sum"}
  %reshape.10 = f16[1,16,16,4]{3,2,1,0} reshape(f16[1,1024]{1,0} %convert.3), metadata={op_type="Reshape" op_name="gradients/all/add_1_grad/Reshape"}
  %call.10 = f16[5,5,64,4]{3,2,1,0} call(f16[1,16,16,64]{3,2,1,0} %call.1, f16[1,16,16,4]{3,2,1,0} %reshape.10), to_apply=%pop_convolution.2, metadata={op_type="Conv2DBackpropFilter" op_name="gradients/all/Conv2D_1_grad/Conv2DBackpropFilter"}
  %multiply.19.70 = f16[5,5,64,4]{3,2,1,0} multiply(f16[5,5,64,4]{3,2,1,0} %call.4, f16[5,5,64,4]{3,2,1,0} %call.10), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/2weights/ResourceApplyGradientDescent"}
  %subtract.19.71 = f16[5,5,64,4]{3,2,1,0} subtract(f16[5,5,64,4]{3,2,1,0} %arg3.19.3, f16[5,5,64,4]{3,2,1,0} %multiply.19.70), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/2weights/ResourceApplyGradientDescent"}
  %call.5 = f16[64]{0} call(), to_apply=%_pop_op_wide_const.2, metadata={op_type="Const" op_name="GradientDescent/update_all/1bias/Cast"}
  %call.7 = f16[1,16,16,64]{3,2,1,0} call(f16[1,16,16,4]{3,2,1,0} %reshape.10, f16[5,5,64,4]{3,2,1,0} %arg3.19.3), to_apply=%pop_backprop_conv, metadata={op_type="Conv2DBackpropInput" op_name="gradients/all/Conv2D_1_grad/Conv2DBackpropInput"}
  %convert.19.78 = f32[1,16,16,64]{3,2,1,0} convert(f16[1,16,16,64]{3,2,1,0} %call.7), metadata={op_type="Sum" op_name="gradients/all/add_grad/Sum_1"}
  %reduce.19.81 = f32[64]{0} reduce(f32[1,16,16,64]{3,2,1,0} %convert.19.78, f32[] %constant.19.35), dimensions={0,1,2}, to_apply=%add14, metadata={op_type="Sum" op_name="gradients/all/add_grad/Sum_1"}
  %convert.19.82 = f16[64]{0} convert(f32[64]{0} %reduce.19.81), metadata={op_type="Sum" op_name="gradients/all/add_grad/Sum_1"}
  %multiply.19.84 = f16[64]{0} multiply(f16[64]{0} %call.5, f16[64]{0} %convert.19.82), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/1bias/ResourceApplyGradientDescent"}
  %subtract.19.85 = f16[64]{0} subtract(f16[64]{0} %arg4.19.4, f16[64]{0} %multiply.19.84), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/1bias/ResourceApplyGradientDescent"}
  %call.6 = f16[7,7,4,64]{3,2,1,0} call(), to_apply=%_pop_op_wide_const.3, metadata={op_type="Const" op_name="GradientDescent/update_all/1bias/Cast"}
  %convert.2 = f16[1,16,16,64]{3,2,1,0} convert(f32[1,16,16,64]{3,2,1,0} %convert.19.78), metadata={op_type="Sum" op_name="gradients/all/add_grad/Sum"}
  %call.11 = f16[7,7,4,64]{3,2,1,0} call(f16[1,16,16,4]{3,2,1,0} %arg0.19.0, f16[1,16,16,64]{3,2,1,0} %convert.2), to_apply=%pop_convolution.3, metadata={op_type="Conv2DBackpropFilter" op_name="gradients/all/Conv2D_grad/Conv2DBackpropFilter"}
  %multiply.19.88 = f16[7,7,4,64]{3,2,1,0} multiply(f16[7,7,4,64]{3,2,1,0} %call.6, f16[7,7,4,64]{3,2,1,0} %call.11), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/1weights/ResourceApplyGradientDescent"}
  %subtract.19.89 = f16[7,7,4,64]{3,2,1,0} subtract(f16[7,7,4,64]{3,2,1,0} %arg5.19.5, f16[7,7,4,64]{3,2,1,0} %multiply.19.88), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_all/1weights/ResourceApplyGradientDescent"}
  ROOT %tuple.19.98 = (f16[4]{0}, f16[5,5,64,4]{3,2,1,0}, f16[64]{0}, f16[7,7,4,64]{3,2,1,0}) tuple(f16[4]{0} %subtract.19.65, f16[5,5,64,4]{3,2,1,0} %subtract.19.71, f16[64]{0} %subtract.19.85, f16[7,7,4,64]{3,2,1,0} %subtract.19.89), metadata={op_name="XLA_Retvals"}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_count(4);
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  CompilerAnnotations annotations;
  ConvolutionClassifier classifier(annotations);

  auto* module = module_or_status.ValueOrDie().get();
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

  %Mean-reduction4 {
    %x.4.0 = f32[] parameter(0)
    %y.4.1 = f32[] parameter(1)
    ROOT %add.4.2 = f32[] add(f32[] %x.4.0, f32[] %y.4.1)
  }

  %max5 {
    %x.5.0 = f32[] parameter(0)
    %y.5.1 = f32[] parameter(1)
    ROOT %maximum.5.2 = f32[] maximum(f32[] %x.5.0, f32[] %y.5.1)
  }

  %_pop_op_wide_const {
    %constant.7.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
    ROOT %broadcast.7.52.clone = f32[3,3,4,12]{3,2,1,0} broadcast(f32[] %constant.7.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_weights/ResourceApplyGradientDescent"}
  }

  %_pop_op_wide_const.1 {
    %constant.7.19.clone = f32[] constant(576), metadata={op_type="Mean" op_name="Mean"}
    ROOT %broadcast.7.20.clone = f32[1,12]{1,0} broadcast(f32[] %constant.7.19.clone), dimensions={}, metadata={op_type="Mean" op_name="Mean"}
  }

  %_pop_op_wide_const.2 {
    %constant.7.5.clone = f32[] constant(0.00173611112), metadata={op_type="Const" op_name="ConstantFolding/gradients/Mean_grad/truediv_recip"}
    ROOT %broadcast.7.49.clone = f32[1,24,24,12]{3,2,1,0} broadcast(f32[] %constant.7.5.clone), dimensions={}, metadata={op_type="Mul" op_name="gradients/Mean_grad/truediv"}
  }

  %pop_convolution {
    %arg_0 = f32[1,24,24,4]{3,2,1,0} parameter(0)
    %arg_1 = f32[1,24,24,12]{3,2,1,0} parameter(1)
    ROOT %convolution.7.51.clone = f32[3,3,4,12]{3,2,1,0} convolution(f32[1,24,24,4]{3,2,1,0} %arg_0, f32[1,24,24,12]{3,2,1,0} %arg_1), window={size=24x24 pad=1_1x1_1}, dim_labels=f01b_i01o->01bf, metadata={op_type="Conv2DBackpropFilter" op_name="gradients/Conv2D_grad/Conv2DBackpropFilter"}
  }

  %pop_convolution.1 {
    %arg_0.1 = f32[1,24,24,4]{3,2,1,0} parameter(0)
    %arg_1.1 = f32[3,3,4,12]{3,2,1,0} parameter(1)
    ROOT %convolution.7.13.clone = f32[1,24,24,12]{3,2,1,0} convolution(f32[1,24,24,4]{3,2,1,0} %arg_0.1, f32[3,3,4,12]{3,2,1,0} %arg_1.1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="Conv2D"}
  }

  %__arithmetic_expression {
    %arg_1.2 = f32[1,12]{1,0} parameter(1)
    %arg_0.2 = f32[1,12]{1,0} parameter(0)
    %divide.7.41.clone = f32[1,12]{1,0} divide(f32[1,12]{1,0} %arg_1.2, f32[1,12]{1,0} %arg_0.2), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
    %arg_2 = f32[1,12]{1,0} parameter(2)
    ROOT %subtract.7.42.clone = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %divide.7.41.clone, f32[1,12]{1,0} %arg_2), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  }

  ENTRY %cluster_1 {
    %arg2.7.2 = f32[3,3,4,12]{3,2,1,0} parameter(2), metadata={op_name="XLA_Args"}
    %call = f32[3,3,4,12]{3,2,1,0} call(), to_apply=%_pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
    %arg0.7.0 = f32[1,24,24,4]{3,2,1,0} parameter(0), metadata={op_name="XLA_Args"}
    %call.4 = f32[1,24,24,12]{3,2,1,0} call(f32[1,24,24,4]{3,2,1,0} %arg0.7.0, f32[3,3,4,12]{3,2,1,0} %arg2.7.2), to_apply=%pop_convolution.1, metadata={op_type="Conv2D" op_name="Conv2D"}
    %constant.7.15 = f32[] constant(0), metadata={op_type="Mean" op_name="Mean"}
    %reduce.7.17 = f32[1,12]{1,0} reduce(f32[1,24,24,12]{3,2,1,0} %call.4, f32[] %constant.7.15), dimensions={1,2}, to_apply=%Mean-reduction4, metadata={op_type="Mean" op_name="Mean"}
    %call.1 = f32[1,12]{1,0} call(), to_apply=%_pop_op_wide_const.1, metadata={op_type="Mean" op_name="Mean"}
    %divide.7.21 = f32[1,12]{1,0} divide(f32[1,12]{1,0} %reduce.7.17, f32[1,12]{1,0} %call.1), metadata={op_type="Mean" op_name="Mean"}
    %constant.7.22 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
    %reduce.7.23 = f32[1]{0} reduce(f32[1,12]{1,0} %divide.7.21, f32[] %constant.7.22), dimensions={1}, to_apply=%max5, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
    %broadcast.7.24 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.7.23), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
    %subtract.7.25 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %divide.7.21, f32[1,12]{1,0} %broadcast.7.24), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
    %exponential.7.26 = f32[1,12]{1,0} exponential(f32[1,12]{1,0} %subtract.7.25), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
    %reduce.7.29 = f32[1]{0} reduce(f32[1,12]{1,0} %exponential.7.26, f32[] %constant.7.15), dimensions={1}, to_apply=%Mean-reduction4, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
    %broadcast.7.40 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.7.29), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
    %arg1.7.1 = f32[1,12]{1,0} parameter(1), metadata={op_name="XLA_Args"}
    %call.5 = f32[1,12]{1,0} call(f32[1,12]{1,0} %broadcast.7.40, f32[1,12]{1,0} %exponential.7.26, f32[1,12]{1,0} %arg1.7.1), to_apply=%__arithmetic_expression, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
    %broadcast = f32[1,24,24,12]{3,2,1,0} broadcast(f32[1,12]{1,0} %call.5), dimensions={0,3}, metadata={op_type="Tile" op_name="gradients/Mean_grad/Tile"}
    %call.2 = f32[1,24,24,12]{3,2,1,0} call(), to_apply=%_pop_op_wide_const.2, metadata={op_type="Const" op_name="ConstantFolding/gradients/Mean_grad/truediv_recip"}
    %multiply.7.50 = f32[1,24,24,12]{3,2,1,0} multiply(f32[1,24,24,12]{3,2,1,0} %broadcast, f32[1,24,24,12]{3,2,1,0} %call.2), metadata={op_type="Mul" op_name="gradients/Mean_grad/truediv"}
    %call.3 = f32[3,3,4,12]{3,2,1,0} call(f32[1,24,24,4]{3,2,1,0} %arg0.7.0, f32[1,24,24,12]{3,2,1,0} %multiply.7.50), to_apply=%pop_convolution, metadata={op_type="Conv2DBackpropFilter" op_name="gradients/Conv2D_grad/Conv2DBackpropFilter"}
    %multiply.7.53 = f32[3,3,4,12]{3,2,1,0} multiply(f32[3,3,4,12]{3,2,1,0} %call, f32[3,3,4,12]{3,2,1,0} %call.3), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_weights/ResourceApplyGradientDescent"}
    %subtract.7.54 = f32[3,3,4,12]{3,2,1,0} subtract(f32[3,3,4,12]{3,2,1,0} %arg2.7.2, f32[3,3,4,12]{3,2,1,0} %multiply.7.53), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_weights/ResourceApplyGradientDescent"}
    ROOT %tuple.7.57 = (f32[3,3,4,12]{3,2,1,0}) tuple(f32[3,3,4,12]{3,2,1,0} %subtract.7.54), metadata={op_name="XLA_Retvals"}
  }
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_count(1);
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  CompilerAnnotations annotations;
  ConvolutionClassifier classifier(annotations);

  auto* module = module_or_status.ValueOrDie().get();
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

%max7 {
  %x.7.0 = f32[] parameter(0)
  %y.7.1 = f32[] parameter(1)
  ROOT %maximum.7.2 = f32[] maximum(f32[] %x.7.0, f32[] %y.7.1)
}

%add8 {
  %x.8.0 = f32[] parameter(0)
  %y.8.1 = f32[] parameter(1)
  ROOT %add.8.2 = f32[] add(f32[] %x.8.0, f32[] %y.8.1)
}

%_pop_op_relu {
  %constant.9.10.clone = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.9.11.clone = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %arg_0 = f32[1,12]{1,0} parameter(0)
  ROOT %maximum.9.12.clone = f32[1,12]{1,0} maximum(f32[1,12]{1,0} %broadcast.9.11.clone, f32[1,12]{1,0} %arg_0), metadata={op_type="Relu" op_name="Relu"}
}

%_pop_op_relugrad {
  %arg_0.1 = f32[1,12]{1,0} parameter(0)
  %constant.9.10.clone.1 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.9.11.clone.1 = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone.1), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %greater-than.9.44.clone = pred[1,12]{1,0} greater-than(f32[1,12]{1,0} %arg_0.1, f32[1,12]{1,0} %broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %arg_1 = f32[1,12]{1,0} parameter(1)
  ROOT %select.9.45.clone = f32[1,12]{1,0} select(pred[1,12]{1,0} %greater-than.9.44.clone, f32[1,12]{1,0} %arg_1, f32[1,12]{1,0} %broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
}

%_pop_op_wide_const {
  %constant.9.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %broadcast.9.39.clone = f32[12,12]{1,0} broadcast(f32[] %constant.9.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
}

%_pop_op_wide_const.1 {
  %constant.9.6.clone.1 = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %broadcast.9.48.clone = f32[4,12]{1,0} broadcast(f32[] %constant.9.6.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
}

ENTRY %cluster_1 {
  %arg2.9.2 = f32[12,12]{1,0} parameter(2), metadata={op_name="XLA_Args"}
  %call.2 = f32[12,12]{1,0} call(), to_apply=%_pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %arg0.9.0 = f32[1,4]{1,0} parameter(0), metadata={op_name="XLA_Args"}
  %arg3.9.3 = f32[4,12]{1,0} parameter(3), metadata={op_name="XLA_Args"}
  %dot.9.9 = f32[1,12]{1,0} dot(f32[1,4]{1,0} %arg0.9.0, f32[4,12]{1,0} %arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  %call = f32[1,12]{1,0} call(f32[1,12]{1,0} %dot.9.9), to_apply=%_pop_op_relu, metadata={op_type="Relu" op_name="Relu"}
  %transpose.9.35 = f32[12,1]{0,1} transpose(f32[1,12]{1,0} %call), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %dot.9.13 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %call, f32[12,12]{1,0} %arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  %constant.9.14 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %reduce.9.15 = f32[1]{0} reduce(f32[1,12]{1,0} %dot.9.13, f32[] %constant.9.14), dimensions={1}, to_apply=%max7, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.9.16 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.15), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %subtract.9.17 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %dot.9.13, f32[1,12]{1,0} %broadcast.9.16), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %exponential.9.18 = f32[1,12]{1,0} exponential(f32[1,12]{1,0} %subtract.9.17), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %constant.9.10 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %reduce.9.21 = f32[1]{0} reduce(f32[1,12]{1,0} %exponential.9.18, f32[] %constant.9.10), dimensions={1}, to_apply=%add8, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.9.32 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.21), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %divide.9.33 = f32[1,12]{1,0} divide(f32[1,12]{1,0} %exponential.9.18, f32[1,12]{1,0} %broadcast.9.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %arg1.9.1 = f32[1,12]{1,0} parameter(1), metadata={op_name="XLA_Args"}
  %subtract.9.34 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %divide.9.33, f32[1,12]{1,0} %arg1.9.1), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %dot.9.36 = f32[12,12]{1,0} dot(f32[12,1]{0,1} %transpose.9.35, f32[1,12]{1,0} %subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %multiply.9.40 = f32[12,12]{1,0} multiply(f32[12,12]{1,0} %call.2, f32[12,12]{1,0} %dot.9.36), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %subtract.9.41 = f32[12,12]{1,0} subtract(f32[12,12]{1,0} %arg2.9.2, f32[12,12]{1,0} %multiply.9.40), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %call.3 = f32[4,12]{1,0} call(), to_apply=%_pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %transpose.9.46 = f32[4,1]{0,1} transpose(f32[1,4]{1,0} %arg0.9.0), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %transpose.9.37 = f32[12,12]{0,1} transpose(f32[12,12]{1,0} %arg2.9.2), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %dot.9.38 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %subtract.9.34, f32[12,12]{0,1} %transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %call.1 = f32[1,12]{1,0} call(f32[1,12]{1,0} %call, f32[1,12]{1,0} %dot.9.38), to_apply=%_pop_op_relugrad, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %dot.9.47 = f32[4,12]{1,0} dot(f32[4,1]{0,1} %transpose.9.46, f32[1,12]{1,0} %call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %multiply.9.49 = f32[4,12]{1,0} multiply(f32[4,12]{1,0} %call.3, f32[4,12]{1,0} %dot.9.47), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  %subtract.9.50 = f32[4,12]{1,0} subtract(f32[4,12]{1,0} %arg3.9.3, f32[4,12]{1,0} %multiply.9.49), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  ROOT %tuple.9.55 = (f32[12,12]{1,0}, f32[4,12]{1,0}) tuple(f32[12,12]{1,0} %subtract.9.41, f32[4,12]{1,0} %subtract.9.50), metadata={op_name="XLA_Retvals"}
}

)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_count(2);
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  CompilerAnnotations annotations;
  ConvolutionClassifier classifier(annotations);

  auto* module = module_or_status.ValueOrDie().get();
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
      // Should not have missing convolutions
      EXPECT_EQ(1, 0);
    }
  }
}

TEST_F(ConvolutionClassifierTest, InferenceMatMul) {
  std::string hlo_string = R"(
HloModule top

%_pop_op_relu {
  %constant.17.9.clone = f32[] constant(0), metadata={op_type="Relu" op_name="dense/Relu"}
  %broadcast.17.10.clone = f32[32,32]{1,0} broadcast(f32[] %constant.17.9.clone), dimensions={}, metadata={op_type="Relu" op_name="dense/Relu"}
  %arg_0 = f32[32,32]{1,0} parameter(0)
  ROOT %maximum.17.11.clone = f32[32,32]{1,0} maximum(f32[32,32]{1,0} %broadcast.17.10.clone, f32[32,32]{1,0} %arg_0), metadata={op_type="Relu" op_name="dense/Relu"}
}

%_pop_op_sigmoid {
  %constant.17.15.clone = f32[] constant(0.5), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  %broadcast.17.21.clone = f32[32,1]{1,0} broadcast(f32[] %constant.17.15.clone), dimensions={}, metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  %arg_0.1 = f32[32,1]{1,0} parameter(0)
  %multiply.17.17.clone = f32[32,1]{1,0} multiply(f32[32,1]{1,0} %broadcast.17.21.clone, f32[32,1]{1,0} %arg_0.1), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  %tanh.17.18.clone = f32[32,1]{1,0} tanh(f32[32,1]{1,0} %multiply.17.17.clone), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  %multiply.17.20.clone = f32[32,1]{1,0} multiply(f32[32,1]{1,0} %broadcast.17.21.clone, f32[32,1]{1,0} %tanh.17.18.clone), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  ROOT %add.17.22.clone = f32[32,1]{1,0} add(f32[32,1]{1,0} %broadcast.17.21.clone, f32[32,1]{1,0} %multiply.17.20.clone), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
}

ENTRY %cluster_9 {
  %arg0.17.0 = f32[32,100]{1,0} parameter(0), metadata={op_name="XLA_Args"}
  %arg4.17.4 = f32[100,32]{1,0} parameter(4), metadata={op_name="XLA_Args"}
  %dot.17.6 = f32[32,32]{1,0} dot(f32[32,100]{1,0} %arg0.17.0, f32[100,32]{1,0} %arg4.17.4), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="dense/MatMul"}
  %arg3.17.3 = f32[32]{0} parameter(3), metadata={op_name="XLA_Args"}
  %broadcast.17.7 = f32[32,32]{1,0} broadcast(f32[32]{0} %arg3.17.3), dimensions={1}, metadata={op_type="BiasAdd" op_name="dense/BiasAdd"}
  %add.17.8 = f32[32,32]{1,0} add(f32[32,32]{1,0} %dot.17.6, f32[32,32]{1,0} %broadcast.17.7), metadata={op_type="BiasAdd" op_name="dense/BiasAdd"}
  %call = f32[32,32]{1,0} call(f32[32,32]{1,0} %add.17.8), to_apply=%_pop_op_relu, metadata={op_type="Relu" op_name="dense/Relu"}
  %arg2.17.2 = f32[32,1]{1,0} parameter(2), metadata={op_name="XLA_Args"}
  %dot.17.12 = f32[32,1]{1,0} dot(f32[32,32]{1,0} %call, f32[32,1]{1,0} %arg2.17.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="dense_1/MatMul"}
  %arg1.17.1 = f32[1]{0} parameter(1), metadata={op_name="XLA_Args"}
  %broadcast.17.13 = f32[32,1]{1,0} broadcast(f32[1]{0} %arg1.17.1), dimensions={1}, metadata={op_type="BiasAdd" op_name="dense_1/BiasAdd"}
  %add.17.14 = f32[32,1]{1,0} add(f32[32,1]{1,0} %dot.17.12, f32[32,1]{1,0} %broadcast.17.13), metadata={op_type="BiasAdd" op_name="dense_1/BiasAdd"}
  %call.1 = f32[32,1]{1,0} call(f32[32,1]{1,0} %add.17.14), to_apply=%_pop_op_sigmoid, metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  ROOT %tuple.17.24 = (f32[32,1]{1,0}) tuple(f32[32,1]{1,0} %call.1), metadata={op_name="XLA_Retvals"}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_count(0);
  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  CompilerAnnotations annotations;
  ConvolutionClassifier classifier(annotations);

  auto* module = module_or_status.ValueOrDie().get();
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
      // Should not have missing convolutions
      EXPECT_EQ(1, 0);
    }
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
