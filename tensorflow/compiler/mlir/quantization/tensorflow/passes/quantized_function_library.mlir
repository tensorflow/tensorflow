// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Quantization as a function library.

module {
  // TODO(b/220993213): factor out common logic.
  func private @quantized_conv2d_fn(%input : tensor<*xi8>,
                         %filter : tensor<*xi8>, %bias : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %fused_scale = "tf.Mul"(%input_scale, %filter_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %2 = "tf.Cast"(%filter) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %filter_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // TODO(b/215633216): Optimize this function with the XLA convolution op.
    %5 = "tf.Conv2D"(%1, %3) {
      padding = "VALID", strides = [1, 1, 1, 1],
      attr_map = "strides:0,use_cudnn_on_gpu:1,padding:2,explicit_paddings:3,dilations:4"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %6 = "tf.AddV2"(%5, %bias) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %7 = "tf.Div"(%fused_scale, %out_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %8 = "tf.Cast"(%6) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %9 = "tf.Mul"(%8, %7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %10 = "tf.Round"(%9) : (tensor<*xf32>) -> tensor<*xf32>
    %11 = "tf.Cast"(%10) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi32>
    %12 = "tf.AddV2"(%11, %out_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %13 = "tf.ClipByValue"(%12, %i8_min, %i8_max) : (tensor<*xi32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
    %14 = "tf.Cast"(%13) {Truncate = false} : (tensor<*xi32>) -> tensor<*xi8>
    return %14 : tensor<*xi8>
  }

  func private @quantized_conv2d_relu_fn(%input : tensor<*xi8>,
                         %filter : tensor<*xi8>, %bias : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %fused_scale = "tf.Mul"(%input_scale, %filter_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %2 = "tf.Cast"(%filter) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %filter_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // TODO(b/215633216): Optimize this function with the XLA convolution op.
    %5 = "tf.Conv2D"(%1, %3) {
      padding = "VALID", strides = [1, 1, 1, 1],
      attr_map = "strides:0,use_cudnn_on_gpu:1,padding:2,explicit_paddings:3,dilations:4"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %6 = "tf.AddV2"(%5, %bias) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %7 = "tf.Div"(%fused_scale, %out_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %8 = "tf.Cast"(%6) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %9 = "tf.Mul"(%8, %7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %10 = "tf.Round"(%9) : (tensor<*xf32>) -> tensor<*xf32>
    %11 = "tf.Cast"(%10) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi32>
    %12 = "tf.AddV2"(%11, %out_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %clip_min = "tf.Maximum"(%i8_min, %out_zp) : (tensor<i32>, tensor<*xi32>) -> tensor<i32>
    %13 = "tf.ClipByValue"(%12, %clip_min, %i8_max) : (tensor<*xi32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
    %14 = "tf.Cast"(%13) {Truncate = false} : (tensor<*xi32>) -> tensor<*xi8>
    return %14 : tensor<*xi8>
  }

  func private @quantized_conv2d_relu6_fn(%input : tensor<*xi8>,
                         %filter : tensor<*xi8>, %bias : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %fused_scale = "tf.Mul"(%input_scale, %filter_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %2 = "tf.Cast"(%filter) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %filter_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // TODO(b/215633216): Optimize this function with the XLA convolution op.
    %5 = "tf.Conv2D"(%1, %3) {
      padding = "VALID", strides = [1, 1, 1, 1],
      attr_map = "strides:0,use_cudnn_on_gpu:1,padding:2,explicit_paddings:3,dilations:4"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %6 = "tf.AddV2"(%5, %bias) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %7 = "tf.Div"(%fused_scale, %out_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %8 = "tf.Cast"(%6) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %9 = "tf.Mul"(%8, %7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %10 = "tf.Round"(%9) : (tensor<*xf32>) -> tensor<*xf32>
    %11 = "tf.Cast"(%10) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi32>
    %12 = "tf.AddV2"(%11, %out_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %act_max =  "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
    %i8_act_max_0 = "tf.PartitionedCall"(%act_max, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@quantize_i8
      } : (tensor<f32>, tensor<*xf32>, tensor<*xi32>) -> tensor<i8>
    %i8_act_max_1 = "tf.Cast"(%i8_act_max_0) {Truncate = false} : (tensor<i8>) -> tensor<i32>
    %clip_min = "tf.Maximum"(%i8_min, %out_zp) : (tensor<i32>, tensor<*xi32>) -> tensor<i32>
    %clip_max = "tf.Minimum"(%i8_max, %i8_act_max_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %13 = "tf.ClipByValue"(%12, %clip_min, %clip_max) : (tensor<*xi32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
    %14 = "tf.Cast"(%13) {Truncate = false} : (tensor<*xi32>) -> tensor<*xi8>
    return %14 : tensor<*xi8>
  }

  // TODO(b/215620570): remove after all biases are handled as i32
  func private @quantized_conv2d_relu6_f32_bias_fn(%input : tensor<*xi8>,
    %filter : tensor<*xi8>, %bias : tensor<*xf32>,
    %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
    %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
    %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %bias_scale = "tf.Mul"(%input_scale, %filter_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %q_bias = "tf.Div"(%bias, %bias_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %4 = "tf.Cast"(%q_bias) : (tensor<*xf32>) -> tensor<*xi32>
    %conv = "tf.PartitionedCall"(%input, %filter, %4,
      %input_scale, %input_zp, %filter_scale, %filter_zp,
      %bias_scale, %filter_zp, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@quantized_conv2d_relu6_fn
      } : (tensor<*xi8>, tensor<*xi8>, tensor<*xi32>,
        tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>,
        tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xi8>
    return %conv: tensor<*xi8>
  }

  // TODO(b/220993213): factor out common logic.
  func private @quantized_matmul_fn(%input : tensor<*xi8>,
                         %filter : tensor<*xi8>, %bias : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>,
                         %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %fused_scale = "tf.Mul"(%input_scale, %weight_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %2 = "tf.Cast"(%filter) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %weight_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // TODO(b/215633216): Optimize this function with the XLA Dot op.
    %5 = "tf.MatMul"(%1, %3) {
      attr_map = "transpose_a:0,transpose_b:1"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %6 = "tf.AddV2"(%5, %bias) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %7 = "tf.Div"(%fused_scale, %out_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %8 = "tf.Cast"(%6) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %9 = "tf.Mul"(%8, %7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %10 = "tf.Round"(%9) : (tensor<*xf32>) -> tensor<*xf32>
    %11 = "tf.Cast"(%10) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi32>
    %12 = "tf.AddV2"(%11, %out_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %13 = "tf.ClipByValue"(%12, %i8_min, %i8_max) : (tensor<*xi32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
    %14 = "tf.Cast"(%13) {Truncate = false} : (tensor<*xi32>) -> tensor<*xi8>
    return %14 : tensor<*xi8>
  }

  func private @quantized_matmul_relu_fn(%input : tensor<*xi8>,
                         %filter : tensor<*xi8>, %bias : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>,
                         %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %fused_scale = "tf.Mul"(%input_scale, %weight_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %2 = "tf.Cast"(%filter) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %weight_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // TODO(b/215633216): Optimize this function with the XLA Dot op.
    %5 = "tf.MatMul"(%1, %3) {
      attr_map = "transpose_a:0,transpose_b:1"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %6 = "tf.AddV2"(%5, %bias) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %7 = "tf.Div"(%fused_scale, %out_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %8 = "tf.Cast"(%6) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %9 = "tf.Mul"(%8, %7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %10 = "tf.Round"(%9) : (tensor<*xf32>) -> tensor<*xf32>
    %11 = "tf.Cast"(%10) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi32>
    %12 = "tf.AddV2"(%11, %out_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %clip_min = "tf.Maximum"(%i8_min, %out_zp) : (tensor<i32>, tensor<*xi32>) -> tensor<i32>
    %13 = "tf.ClipByValue"(%12, %clip_min, %i8_max) : (tensor<*xi32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
    %14 = "tf.Cast"(%13) {Truncate = false} : (tensor<*xi32>) -> tensor<*xi8>
    return %14 : tensor<*xi8>
  }

  func private @quantized_matmul_relu6_fn(%input : tensor<*xi8>,
                         %weight : tensor<*xi8>, %bias : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>,
                         %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %fused_scale = "tf.Mul"(%input_scale, %weight_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %2 = "tf.Cast"(%weight) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %weight_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // TODO(b/215633216): Optimize this function with the XLA Dot op.
    %5 = "tf.MatMul"(%1, %3) {
      attr_map = "transpose_a:0,transpose_b:1"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %6 = "tf.AddV2"(%5, %bias) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %7 = "tf.Div"(%fused_scale, %out_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %8 = "tf.Cast"(%6) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %9 = "tf.Mul"(%8, %7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %10 = "tf.Round"(%9) : (tensor<*xf32>) -> tensor<*xf32>
    %11 = "tf.Cast"(%10) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi32>
    %12 = "tf.AddV2"(%11, %out_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %act_max =  "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
    %i8_act_max_0 = "tf.PartitionedCall"(%act_max, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@quantize_i8
      } : (tensor<f32>, tensor<*xf32>, tensor<*xi32>) -> tensor<i8>
    %i8_act_max_1 = "tf.Cast"(%i8_act_max_0) {Truncate = false} : (tensor<i8>) -> tensor<i32>
    %clip_min = "tf.Maximum"(%i8_min, %out_zp) : (tensor<i32>, tensor<*xi32>) -> tensor<i32>
    %clip_max = "tf.Minimum"(%i8_max, %i8_act_max_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %13 = "tf.ClipByValue"(%12, %clip_min, %clip_max) : (tensor<*xi32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
    %14 = "tf.Cast"(%13) {Truncate = false} : (tensor<*xi32>) -> tensor<*xi8>
    return %14 : tensor<*xi8>
  }

  // Note: following functions won't handle per-channel quantization for now.
  func private @quantize_i8(%input : tensor<*xf32>, %scale : tensor<*xf32>, %zp : tensor<*xi32>) -> tensor<*xi8> {
    %div = "tf.Div"(%input, %scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %round = "tf.Round"(%div) : (tensor<*xf32>) -> tensor<*xf32>
    %cast = "tf.Cast"(%round) : (tensor<*xf32>) -> tensor<*xi32>
    %add = "tf.AddV2"(%cast, %zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %i8 = "tf.Cast"(%add) : (tensor<*xi32>) -> tensor<*xi8>
    return %i8 : tensor<*xi8>
  }

  func private @dequantize_i8(%input : tensor<*xi8>, %scale : tensor<*xf32>, %zp : tensor<*xi32>) -> tensor<*xf32> {
    %input_i32 = "tf.Cast"(%input) : (tensor<*xi8>) -> tensor<*xi32>
    %output = "tf.Sub"(%input_i32, %zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %cast = "tf.Cast"(%output) : (tensor<*xi32>) -> tensor<*xf32>
    %mul = "tf.Mul"(%cast, %scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    return %mul : tensor<*xf32>
  }
}
