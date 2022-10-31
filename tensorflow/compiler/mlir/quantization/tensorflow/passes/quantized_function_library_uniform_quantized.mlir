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

  // TODO(b/240931497) Replace with core tf ops once uniform quantization is submitted.
  // Ref bugs for op: b/230804708, b/230805744
  func.func @quantized_conv2d_with_bias_fn(
                         %input : tensor<*x!tf_type.qint8>, %filter : tensor<*x!tf_type.qint8>, %bias : tensor<*x!tf_type.qint32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>, %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>, %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>, %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x!tf_type.qint8> {
   %conv_out = "tf.ExperimentalUniformQuantizedConvolution"(%input, %filter,
                                %input_scale, %input_zp, %filter_scale, %filter_zp, %out_scale, %out_zp) {
        // TODO(b/238600711): Populate attributes for quantized_function_library_uniform_quantized
        Tlhs = "tfdtype$DT_QINT8",
        Trhs = "tfdtype$DT_QINT8",
        Tout = "tfdtype$DT_QINT32",
        window_strides = [1, 1],
        padding = "",
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
        attr_map = "0:Tlhs,1:Trhs,2:Tout,3:lhs_quantization_min_val,4:lhs_quantization_max_val,5:rhs_quantization_min_val,6:rhs_quantization_max_val,7:output_quantization_min_val,8:output_quantization_max_val"
    } : (tensor<*x!tf_type.qint8>, tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint32>
    %add_bias = "tf.ExperimentalUniformQuantizedAdd"(%conv_out, %bias, %input_scale, %input_zp, %bias_scale, %bias_zp, %out_scale, %out_zp) {
      // TODO(b/238600711): Populate attributes for quantized_function_library_uniform_quantized
      lhs_quantization_axis = -1,
      lhs_quantization_min_val = -128,
      lhs_quantization_max_val = 127,
      rhs_quantization_axis = -1,
      rhs_quantization_min_val = -128,
      rhs_quantization_max_val = 127,
      output_quantization_axis = -1,
      output_quantization_min_val = -128,
      output_quantization_max_val = 127,
      T = 1,
      attr_map = "0:Tlhs,1:Trhs,2:Tout,3:lhs_quantization_min_val,4:lhs_quantization_max_val,5:rhs_quantization_min_val,6:rhs_quantization_max_val,7:output_quantization_min_val,8:output_quantization_max_val"
    } : (tensor<*x!tf_type.qint32>, tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint32>
    %requantized_out = "tf.PartitionedCall"(%add_bias, %input_scale, %input_zp, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@requantize_qi8
    } : (tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint8>
    func.return %requantized_out : tensor<*x!tf_type.qint8>
  }

  // Quantize initial input at the start of the graph.
  func.func @quantize_qi8(%input : tensor<*xf32>, %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>, %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x!tf_type.qint8> {
    %quantized_out = "tf.ExperimentalUniformQuantize"(%input, %input_scale, %input_zp) {
      // TODO(b/238600711): Populate attributes for quantized_function_library_uniform_quantized
      quantization_axis = -1,
      quantization_min_val = -128,
      quantization_max_val = 127,
      T = 1,
      attr_map = "0:Tin,1:Tout,2:quantization_axis,3:quantization_min_val,4:quantization_max_val"
    } : (tensor<*xf32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint8>
    func.return %quantized_out : tensor<*x!tf_type.qint8>
  }

  // Requantize a qint32 tensor to qint8 tensor for the next input.
  func.func @requantize_qi8(%input : tensor<*x!tf_type.qint32>, %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>, %out_scale: tensor<*xf32>, %out_zp: tensor<*xi32>) -> tensor<*x!tf_type.qint8> {
    %requantized_out = "tf.ExperimentalUniformRequantize"(%input, %input_scale, %input_zp, %out_scale, %out_zp) {
      // TODO(b/238600711): Populate attributes for quantized_function_library_uniform_quantized
      input_quantization_axis = -1,
      input_quantization_min_val = -128,
      input_quantization_max_val = 127,
      output_quantization_axis = -1,
      output_quantization_min_val = -128,
      output_quantization_max_val = 127,
      attr_map = "0:Tin,1:Tout,2:input_quantization_axis,3:input_quantization_min_val,4:input_quantization_max_val,5:output_quantization_axis,6:output_quantization_min_val,7:output_quantization_min_val"
    } : (tensor<*x!tf_type.qint32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf_type.qint8>
    func.return %requantized_out : tensor<*x!tf_type.qint8>
  }

  // Dequantize final graph output back to f32.
  func.func @dequantize_qi8(%input : tensor<*x!tf_type.qint8>, %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>, %out_scale : tensor<*xf32>, %out_zp : tensor<*xi8>) -> tensor<*xf32> {
    %dequantized_out = "tf.ExperimentalUniformDequantize"(%input, %input_scale, %input_zp) {
      // TODO(b/238600711): Populate attributes for quantized_function_library_uniform_quantized
      quantization_axis = -1,
      quantization_min_val = -128,
      quantization_max_val = 127,
      attr_map = "0:Tin,1:Tout,2:quantization_axis,3:quantization_min_val,4:quantization_max_val"
    } : (tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    func.return %dequantized_out : tensor<*xf32>
  }
}

