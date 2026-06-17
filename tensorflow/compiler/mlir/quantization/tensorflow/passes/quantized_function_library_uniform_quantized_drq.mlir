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

// Quantization as a function library with Uniform Quantized Ops for Dynamic
// PTQ
//
// Internal functions should be marked as private. They will be inlined and
// deleted in `InsertQuantizedFunctionsPass`.
//
// For Uniform Quantized op case, attributes are generated during quantize
// composite pass. Therefore, attr_map is set to an empty string.

module {

  // Currently only 4-d case is supported
  func.func @quantized_conv2d_fn(
                         %input : tensor<*xf32>, %weight : tensor<*x!tf_type.qint8>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xf32>
      attributes {tf_quant.quantized_ops = ["Conv2D"]} {

    %out = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight,
                           %weight_scale, %weight_zp) {
        Tlhs = "tfdtype$DT_FLOAT",
        Trhs = "tfdtype$DT_QINT8",
        Tout = "tfdtype$DT_FLOAT",
        window_strides = [1, 1],
        padding = "",
        explicit_padding = [],
        lhs_dilation = [],
        rhs_dilation = [],
        dimension_numbers = "",
        batch_group_count = 1,
        feature_group_count = 1,
        rhs_quantization_axis = -1,
        rhs_quantization_min_val = -128,
        rhs_quantization_max_val = 127,
        attr_map = ""
      } : (tensor<*xf32>, tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>


    func.return %out : tensor<*xf32>
  }

  // Currently only 4-d case is supported
  func.func @quantized_depthwise_conv2d_fn(
                         %input : tensor<*xf32>, %weight : tensor<*x!tf_type.qint8>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xf32>
      attributes {tf_quant.quantized_ops = ["DepthwiseConv2D"]} {

    %out = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight,
                           %weight_scale, %weight_zp) {
        Tlhs = "tfdtype$DT_FLOAT",
        Trhs = "tfdtype$DT_QINT8",
        Tout = "tfdtype$DT_FLOAT",
        window_strides = [1, 1],
        padding = "",
        explicit_padding = [],
        lhs_dilation = [],
        rhs_dilation = [],
        dimension_numbers = "",
        batch_group_count = 1,
        feature_group_count = 1,
        rhs_quantization_axis = -1,
        rhs_quantization_min_val = -128,
        rhs_quantization_max_val = 127,
        attr_map = ""
      } : (tensor<*xf32>, tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>

    func.return %out : tensor<*xf32>
  }

  // Currently only 4-d case is supported
  func.func @quantized_matmul_fn(
                         %input : tensor<*xf32>, %weight : tensor<*x!tf_type.qint8>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xf32>
      attributes {tf_quant.quantized_ops = ["MatMul"]} {

    %out = "tf.UniformQuantizedDotHybrid"(%input, %weight,
                                %weight_scale, %weight_zp) {
        Tlhs = "tfdtype$DT_FLOAT",
        Trhs = "tfdtype$DT_QINT8",
        Tout = "tfdtype$DT_FLOAT",
        rhs_quantization_axis = -1,
        rhs_quantization_min_val = -128,
        rhs_quantization_max_val = 127,
        attr_map = ""
      } : (tensor<*xf32>, tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>

    func.return %out : tensor<*xf32>
  }
}
