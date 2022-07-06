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

// RUN: tf-quant-opt %s -split-input-file -quant-insert-quantized-functions -quant-quantize-composite-functions -symbol-dce | FileCheck %s

module {
  func.func @conv(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
    %cst = "tf.Const"() {value = dense<[[[[1.600000e-01, 1.000000e-01], [5.100000e-01, 5.400000e-01], [-5.000000e-01, 4.100000e-01]], [[-3.500000e-01, 5.000000e-02], [-0.00999999977, 1.600000e-01], [-4.800000e-01, -2.400000e-01]]], [[[-3.500000e-01, -2.100000e-01], [-1.400000e-01, -2.000000e-02], [4.800000e-01, 3.500000e-01]], [[-1.900000e-01, 3.200000e-01], [0.00999999977, -7.000000e-02], [2.000000e-01, -4.000000e-02]]]]> : tensor<2x2x3x2xf32>} : () -> tensor<2x2x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<[-2.000000e+00, 3.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "quant.qcast"(%cst) : (tensor<2x2x3x2xf32>) -> tensor<2x2x3x2x!quant.uniform<i8<-127:127>:f32:3, {4.000000e-03,5.000000e-03}>>
    %1 = "quant.dcast"(%0) : (tensor<2x2x3x2x!quant.uniform<i8<-127:127>:f32:3, {4.000000e-03,5.000000e-03}>>) -> tensor<*xf32>
    %2 = "quant.qcast"(%arg0) : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 8.000000e-03>>
    %3 = "quant.dcast"(%2) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 8.000000e-03>>) -> tensor<*xf32>
    %4 = "tf.PartitionedCall"(%3, %1, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2} : (tensor<*xf32>, tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %5 = "quant.qcast"(%4) : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-1>>
    %6 = "quant.dcast"(%5) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-1>>) -> tensor<*xf32>
    %7 = "tf.PartitionedCall"(%arg0, %cst, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1} : (tensor<1x2x2x3xf32>, tensor<2x2x3x2xf32>, tensor<2xf32>) -> tensor<*xf32>
    func.return %6, %7 : tensor<*xf32>, tensor<*xf32>
  }
  func.func private @composite_conv2d_with_bias_and_relu6_fn_2(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }
  func.func private @composite_conv2d_with_bias_and_relu6_fn_1(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x2x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x2x2x3xf32>, tensor<2x2x3x2xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }

// CHECK-LABEL: func @conv
// CHECK-DAG: %[[w_float:.*]] = "tf.Const"() {value = dense<{{\[\[\[\[}}1.600000e-01
// CHECK-DAG: %[[b_float:.*]] = "tf.Const"() {value = dense<[-2.000000e+00, 3.000000e+00]> : tensor<2xf32>
// CHECK-DAG: %[[in_scale:.*]] = "tf.Const"() {value = dense<8.000000e-03> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[in_zp:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>}
// CHECK-DAG: %[[w_scale:.*]] = "tf.Const"() {value = dense<[4.000000e-03
// CHECK-DAG: %[[w_zp:.*]] = "tf.Const"() {value = dense<0> : tensor<2xi32>}
// CHECK-DAG: %[[b_scale:.*]] = "tf.Const"() {value = dense<[3.200000e-05, 4.000000e-05]> : tensor<2xf32>}
// CHECK-DAG: %[[out_scale:.*]] = "tf.Const"() {value = dense<5.000000e-02> : tensor<f32>}
// CHECK-DAG: %[[out_zp:.*]] = "tf.Const"() {value = dense<-1> : tensor<i32>}
// CHECK-DAG: %[[b_quant:.*]] = "tf.Const"() {value = dense<[-62500, 75000]> : tensor<2xi32>}
// CHECK-DAG: %[[w_quant:.*]] = "tf.Const"() {value = dense<{{\[\[\[\[}}40, 20]
// CHECK-DAG: {{\[\[\[}}-87, -42]

// CHECK: %[[quantize:.*]] = "tf.PartitionedCall"(%arg0, %[[in_scale]], %[[in_zp]])
// CHECK-SAME: f = @quantize_i8
// CHECK: %[[conv_quant:.*]] = "tf.PartitionedCall"(%[[quantize]], %[[w_quant]], %[[b_quant]],
// CHECK-SAME: %[[in_scale]], %[[in_zp]], %[[w_scale]], %[[w_zp]],
// CHECK-SAME: %[[b_scale]], %[[w_zp]], %[[out_scale]], %[[out_zp]])
// CHECK-SAME: f = @quantized_conv2d_with_bias_and_relu6_fn_0
// CHECK-SAME: (tensor<1x2x2x3xi8>, tensor<2x2x3x2xi8>, tensor<2xi32>, tensor<f32>, tensor<i32>, tensor<2xf32>, tensor<2xi32>, tensor<2xf32>, tensor<2xi32>, tensor<f32>, tensor<i32>) -> tensor<*xi8>
// CHECK: %[[dequantize:.*]] = "tf.PartitionedCall"(%[[conv_quant]], %[[out_scale]], %[[out_zp]])
// CHECK-SAME: f = @dequantize_i8

// CHECK: %[[conv_float:.*]] = "tf.PartitionedCall"(%arg0, %[[w_float]], %[[b_float]])
// CHECK-SAME: f = @composite_conv2d_with_bias_and_relu6_fn_1

// CHECK: return %[[dequantize]], %[[conv_float]]

// CHECK-LABEL: func private @composite_conv2d_with_bias_and_relu6_fn_1
// CHECK:      %[[CONV2D_0:.*]] = "tf.Conv2D"
// CHECK-SAME: data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
// CHECK:      %[[BIASADD_0:.*]] = "tf.BiasAdd"
// CHECK:      %[[RELU6_0:.*]] = "tf.Relu6"

// CHECK-LABEL: func private @quantized_conv2d_with_bias_and_relu6_fn_0
// CHECK-SAME: (%arg0: tensor<1x2x2x3xi8>, %arg1: tensor<2x2x3x2xi8>, %arg2: tensor<2xi32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<2xf32>, %arg6: tensor<2xi32>, %arg7: tensor<2xf32>, %arg8: tensor<2xi32>, %arg9: tensor<f32>, %arg10: tensor<i32>) -> tensor<*xi8>
// CHECK:      %[[CONV2D_0:.*]] = "tf.Conv2D"
// CHECK-SAME: {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}
}

// -----

module {
  func.func @conv_with_default_attributes(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>) {
    %cst = "tf.Const"() {value = dense<[[[[1.600000e-01, 1.000000e-01], [5.100000e-01, 5.400000e-01], [-5.000000e-01, 4.100000e-01]], [[-3.500000e-01, 5.000000e-02], [-0.00999999977, 1.600000e-01], [-4.800000e-01, -2.400000e-01]]], [[[-3.500000e-01, -2.100000e-01], [-1.400000e-01, -2.000000e-02], [4.800000e-01, 3.500000e-01]], [[-1.900000e-01, 3.200000e-01], [0.00999999977, -7.000000e-02], [2.000000e-01, -4.000000e-02]]]]> : tensor<2x2x3x2xf32>} : () -> tensor<2x2x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<[-2.000000e+00, 3.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "quant.qcast"(%cst) : (tensor<2x2x3x2xf32>) -> tensor<2x2x3x2x!quant.uniform<i8<-127:127>:f32:3, {4.000000e-03,5.000000e-03}>>
    %1 = "quant.dcast"(%0) : (tensor<2x2x3x2x!quant.uniform<i8<-127:127>:f32:3, {4.000000e-03,5.000000e-03}>>) -> tensor<*xf32>
    %2 = "quant.qcast"(%arg0) : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 8.000000e-03>>
    %3 = "quant.dcast"(%2) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 8.000000e-03>>) -> tensor<*xf32>
    %4 = "tf.PartitionedCall"(%3, %1, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1} : (tensor<*xf32>, tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %5 = "quant.qcast"(%4) : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-1>>
    %6 = "quant.dcast"(%5) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-1>>) -> tensor<*xf32>
    func.return %6 : tensor<*xf32>
  }
  func.func private @composite_conv2d_with_bias_and_relu6_fn_1(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }

// CHECK-LABEL: func @conv_with_default_attributes

// CHECK: %[[quantize:.*]] = "tf.PartitionedCall"(%arg0
// CHECK-SAME: f = @quantize_i8
// CHECK: %[[conv_quant:.*]] = "tf.PartitionedCall"(%[[quantize]]
// CHECK-SAME: f = @quantized_conv2d_with_bias_and_relu6_fn_0
// CHECK-SAME: (tensor<1x2x2x3xi8>, tensor<2x2x3x2xi8>, tensor<2xi32>, tensor<f32>, tensor<i32>, tensor<2xf32>, tensor<2xi32>, tensor<2xf32>, tensor<2xi32>, tensor<f32>, tensor<i32>) -> tensor<*xi8>
// CHECK: %[[dequantize:.*]] = "tf.PartitionedCall"(%[[conv_quant]]
// CHECK-SAME: f = @dequantize_i8
// CHECK: return %[[dequantize]]

// CHECK-LABEL: func private @quantized_conv2d_with_bias_and_relu6_fn_0
// CHECK-SAME: (%arg0: tensor<1x2x2x3xi8>, %arg1: tensor<2x2x3x2xi8>, %arg2: tensor<2xi32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<2xf32>, %arg6: tensor<2xi32>, %arg7: tensor<2xf32>, %arg8: tensor<2xi32>, %arg9: tensor<f32>, %arg10: tensor<i32>) -> tensor<*xi8>
// CHECK:      %[[CONV2D_0:.*]] = "tf.Conv2D"
// CHECK-SAME: {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}
}
