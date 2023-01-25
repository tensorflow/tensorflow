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

// Quantization as a function library with Uniform Quantized Ops for Static
// PTQ
//
// Internal functions should be marked as private. They will be inlined and
// deleted in `InsertQuantizedFunctionsPass`.
//
// Function template can generate functions with different parameters. Ex:
// ```
// parameters[
//   {"key1": "value11", "key2": "value21"},
//   {"key1": "value12", "key2": "value22"},
// ]
// func.func func_name_${key1}_fn (...) {
//   ...${key2}...
// }
// ```
// The above template with generate two functions by substituting `key1` and
// `key2` with given values.

module {

  for main_op in ["Conv2D", "DepthwiseConv2D"] {
    parameters[
      {"quantized_ops": ["${main_op}", "BiasAdd"], "act_func": "internal_requantize_no_activation_fn", "output_type": "!tf_type.qint8"},
      {"quantized_ops": ["${main_op}", "BiasAdd", "Relu"], "act_func": "internal_requantize_and_relu_fn", "output_type": "!tf_type.qint8"},
      {"quantized_ops": ["${main_op}", "BiasAdd", "Relu6"], "act_func": "internal_requantize_and_relu6_fn", "output_type": "!tf_type.qint8"},
    ]
    func.func @GenerateQuantizedFunctionName(${quantized_ops})(%input : tensor<*x!tf_type.qint8>,
                          %filter : tensor<*x!tf_type.qint8>, %bias : tensor<*x!tf_type.qint32>,
                          %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                          %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                          %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>,
                          %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x${output_type}>
        attributes {tf_quant.quantized_ops = ${quantized_ops}} {
      // TODO(b/258729559): Revisit scale/zp after e2e path for SRQ on UQ is ready.
      %main_out = "tf.PartitionedCall"(%input, %filter, %input_scale, %input_zp,
                                  %filter_scale, %filter_zp, %out_scale, %out_zp) {
          config = "", config_proto = "", executor_type = "", f=@GenerateImplFunctionName(${main_op})
        } : (tensor<*x!tf_type.qint8>, tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint32>
      %add = "tf.UniformQuantizedAdd"(%main_out, %bias, %input_scale, %input_zp, %bias_scale, %bias_zp, %out_scale, %out_zp) {
        lhs_quantization_axis = -1,
        lhs_quantization_min_val = -128,
        lhs_quantization_max_val = 127,
        rhs_quantization_axis = -1,
        rhs_quantization_min_val = -128,
        rhs_quantization_max_val = 127,
        output_quantization_axis = -1,
        output_quantization_min_val = -128,
        output_quantization_max_val = 127,
        T = "tfdtype$DT_QINT32",
        attr_map = ""
      } : (tensor<*x!tf_type.qint32>, tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint32>
      %act = "tf.PartitionedCall"(%add, %input_scale, %input_zp, %out_scale, %out_zp) {
          config = "", config_proto = "", executor_type = "", f=@${act_func}
        } : (tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x${output_type}>
      func.return %act : tensor<*x${output_type}>
    }

    parameters[
      {"quantized_ops": ["${main_op}"], "act_func": "internal_requantize_no_activation_fn", "output_type": "!tf_type.qint8"},
      {"quantized_ops": ["${main_op}", "Relu"], "act_func": "internal_requantize_and_relu_fn", "output_type": "!tf_type.qint8"},
      {"quantized_ops": ["${main_op}", "Relu6"], "act_func": "internal_requantize_and_relu6_fn", "output_type": "!tf_type.qint8"},
    ]
    func.func @GenerateQuantizedFunctionName(${quantized_ops})(%input : tensor<*x!tf_type.qint8>, %filter : tensor<*x!tf_type.qint8>,
                          %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                          %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                          %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x${output_type}>
        attributes {tf_quant.quantized_ops = ${quantized_ops}} {
      // TODO(b/258729559): Revisit scale/zp after e2e path for SRQ on UQ is ready.
      %main_out = "tf.PartitionedCall"(%input, %filter, %input_scale, %input_zp,
                                  %filter_scale, %filter_zp, %out_scale, %out_zp) {
          config = "", config_proto = "", executor_type = "", f=@GenerateImplFunctionName(${main_op})
        } : (tensor<*x!tf_type.qint8>, tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint32>
      %act = "tf.PartitionedCall"(%main_out, %input_scale, %input_zp, %out_scale, %out_zp) {
          config = "", config_proto = "", executor_type = "", f=@${act_func}
        } : (tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x${output_type}>
      func.return %act : tensor<*x${output_type}>
    }
  } // end for

  // Conv2d Convolution.
  func.func private @internal_conv2d_fn(
                         %input : tensor<*x!tf_type.qint8>, %filter : tensor<*x!tf_type.qint8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>, %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x!tf_type.qint32> {
    %conv_out = "tf.UniformQuantizedConvolution"(%input, %filter,
                                %input_scale, %input_zp, %filter_scale, %filter_zp, %out_scale, %out_zp) {
        Tin = "tfdtype$DT_QINT8",
        Tout = "tfdtype$DT_QINT32",
        window_strides = [1, 1],
        padding = "SAME",
        explicit_padding = [],
        lhs_dilation = [],
        rhs_dilation = [],
        batch_group_count = 1,
        feature_group_count = 1,
        dimension_numbers = "",
        lhs_quantization_axis = -1,
        lhs_quantization_min_val = -128,
        lhs_quantization_max_val = 127,
        rhs_quantization_axis = -1,
        rhs_quantization_min_val = -128,
        rhs_quantization_max_val = 127,
        output_quantization_axis = -1,
        output_quantization_min_val = -128,
        output_quantization_max_val = 127,
        attr_map = ""
      } : (tensor<*x!tf_type.qint8>, tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint32>
    func.return %conv_out : tensor<*x!tf_type.qint32>
  }

  // Depthwise convolution. feature_group_count is set to 3rd dim of input shape.
  func.func private @internal_depthwise_conv2d_fn(
                         %input : tensor<*x!tf_type.qint8>, %filter : tensor<*x!tf_type.qint8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>, %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x!tf_type.qint32> {
    %conv_out = "tf.UniformQuantizedConvolution"(%input, %filter,
                                %input_scale, %input_zp, %filter_scale, %filter_zp, %out_scale, %out_zp) {
        Tin = "tfdtype$DT_QINT8",
        Tout = "tfdtype$DT_QINT32",
        window_strides = [1, 1],
        padding = "SAME",
        explicit_padding = [],
        lhs_dilation = [],
        rhs_dilation = [],
        batch_group_count = 1,
        feature_group_count = 1,
        dimension_numbers = "",
        lhs_quantization_axis = -1,
        lhs_quantization_min_val = -128,
        lhs_quantization_max_val = 127,
        rhs_quantization_axis = -1,
        rhs_quantization_min_val = -128,
        rhs_quantization_max_val = 127,
        output_quantization_axis = -1,
        output_quantization_min_val = -128,
        output_quantization_max_val = 127,
        attr_map = ""
      } : (tensor<*x!tf_type.qint8>, tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint32>
    func.return %conv_out : tensor<*x!tf_type.qint32>
  }

  // Quantize initial input at the start of the graph. Output is qint8.
  func.func @quantize_i8(%input : tensor<*xf32>, %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>) -> tensor<*x!tf_type.qint8> {
    %quantize = "tf.UniformQuantize"(%input, %input_scale, %input_zp) {
      Tin = "tfdtype$DT_FLOAT",
      Tout = "tfdtype$DT_QINT8",
      quantization_axis = -1,
      quantization_min_val = -128,
      quantization_max_val = 127,
      attr_map = ""
    } : (tensor<*xf32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint8>
    func.return %quantize : tensor<*x!tf_type.qint8>
  }

  // Requantize a qint32 tensor to qint8 tensor for the next input.
  func.func private @internal_requantize_qi8_fn(%input : tensor<*x!tf_type.qint32>, %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>, %out_scale: tensor<*xf32>, %out_zp: tensor<*xi32>) -> tensor<*x!tf_type.qint8> {
    %requantize = "tf.UniformRequantize"(%input, %input_scale, %input_zp, %out_scale, %out_zp) {
            Tin = "tfdtype$DT_QINT32",
            Tout = "tfdtype$DT_QINT8",
            input_quantization_axis = -1,
            input_quantization_min_val = -128,
            input_quantization_max_val = 127,
            output_quantization_axis = -1,
            output_quantization_min_val = -128,
            output_quantization_max_val = 127,
            attr_map = ""
          } : (tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint8>
    func.return %requantize : tensor<*x!tf_type.qint8>
  }

  // Dequantize final graph output back to f32. Input is qint8.
  func.func @dequantize_i8(%input : tensor<*x!tf_type.qint8>, %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>) -> tensor<*xf32> {
    %dequantize = "tf.UniformDequantize"(%input, %input_scale, %input_zp) {
      Tin = "tfdtype$DT_QINT8",
      Tout = "tfdtype$DT_FLOAT",
      quantization_axis = -1,
      quantization_min_val = -128,
      quantization_max_val = 127,
      attr_map = ""
    } : (tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    func.return %dequantize : tensor<*xf32>
  }

  // Requantizes and applies quantized Relu by clipping.
  func.func private @internal_requantize_no_activation_fn(%input : tensor<*x!tf_type.qint32>, %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x!tf_type.qint8> {
    %q_out = "tf.PartitionedCall"(%input, %input_scale, %input_zp, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_requantize_qi8_fn
      } : (tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint8>
    func.return %q_out : tensor<*x!tf_type.qint8>
  }

  // Requantizes and applies quantized Relu6 by clipping.
  func.func private @internal_requantize_and_relu_fn(%input : tensor<*x!tf_type.qint32>, %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x!tf_type.qint8> {
    %i8_min = "tf.Const"() {value = dense<-128.0> : tensor<f32>} : () -> tensor<f32>
    %i8_max = "tf.Const"() {value = dense<127.0> : tensor<f32>} : () -> tensor<f32>
    %float_out_zp = "tf.Cast"(%out_zp) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %clip_min = "tf.Maximum"(%i8_min, %float_out_zp) : (tensor<f32>, tensor<*xf32>) -> tensor<f32>
    %qclip_min = "tf.Cast"(%i8_min) {Truncate = false} : (tensor<f32>) -> tensor<!tf_type.qint32>
    %qi8_max = "tf.Cast"(%i8_max) {Truncate = false} : (tensor<f32>) -> tensor<!tf_type.qint32>
    %relu = "tf.UniformQuantizedClipByValue"(%input, %qclip_min, %qi8_max, %out_scale, %out_zp) {
      T = "tfdtype$DT_QINT32",
      quantization_axis = -1,
      quantization_min_val = -128,
      quantization_max_val = 127,
      attr_map = ""
    } : (tensor<*x!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint32>
    %requantize = "tf.PartitionedCall"(%relu, %input_scale, %input_zp, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_requantize_qi8_fn
      } : (tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint8>
    func.return %requantize : tensor<*x!tf_type.qint8>
  }

   // Apply requantization and relu6.
  func.func private @internal_requantize_and_relu6_fn(%input : tensor<*x!tf_type.qint32>, %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x!tf_type.qint8> {
    %i8_min = "tf.Const"() {value = dense<-128.0> : tensor<f32>} : () -> tensor<f32>
    %i8_max = "tf.Const"() {value = dense<127.0> : tensor<f32>} : () -> tensor<f32>
    %act_max =  "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
    %i8_act_max_0 = "tf.PartitionedCall"(%act_max, %input_scale, %input_zp) {
        config = "", config_proto = "", executor_type = "", f=@quantize_i8
      } : (tensor<f32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint8>
    %i8_act_max_1 = "tf.Cast"(%i8_act_max_0) {Truncate = false} : (tensor<*x!tf_type.qint8>) -> tensor<f32>
    %float_out_zp = "tf.Cast"(%out_zp) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %clip_min = "tf.Maximum"(%i8_min, %float_out_zp) : (tensor<f32>, tensor<*xf32>) -> tensor<f32>
    %clip_max = "tf.Minimum"(%i8_max, %i8_act_max_1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %qclip_min = "tf.Cast"(%i8_min) {Truncate = false} : (tensor<f32>) -> tensor<!tf_type.qint32>
    %qclip_max = "tf.Cast"(%i8_max) {Truncate = false} : (tensor<f32>) -> tensor<!tf_type.qint32>
    %relu = "tf.UniformQuantizedClipByValue"(%input, %qclip_min, %qclip_max, %out_scale, %out_zp) {
      T = "tfdtype$DT_QINT32",
      quantization_axis = -1,
      quantization_min_val = -128,
      quantization_max_val = 127,
      attr_map = ""
    } : (tensor<*x!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint32>
    %requantize = "tf.PartitionedCall"(%relu, %input_scale, %input_zp, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_requantize_qi8_fn
      } : (tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint8>
    func.return %requantize : tensor<*x!tf_type.qint8>
  }
}

