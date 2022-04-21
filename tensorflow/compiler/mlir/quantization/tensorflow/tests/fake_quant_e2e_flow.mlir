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

// RUN: tf-quant-opt %s -quant-convert-fake-quant-to-qdq -quant-lift-quantizable-spots-as-functions -quant-insert-quantized-functions -quant-quantize-composite-functions -symbol-dce | FileCheck %s

func.func @fake_quant_conv(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<*xf32> {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg1) {device = "", max = 2.000000e+00 : f32, min = -1.000000e+00 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %1 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {device = "", max = 2.000000e-01 : f32, min = -1.000000e-01 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<1x3x4x3xf32>) -> tensor<*xf32>
  %2 = "tf.Conv2D"(%1, %0) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.BiasAdd"(%2, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %4 = "tf.FakeQuantWithMinMaxArgs"(%3) {device = "", max = 4.000000e-01 : f32, min = -3.000000e-01 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %4 : tensor<*xf32>
}

// CHECK-LABEL: @fake_quant_conv
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x3
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x3
// CHECK-DAG: %[[CST:.*]] = "tf.Const"() {value = dense<0.00117647066> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() {value = dense<-43> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"() {value = dense<0.0117647061> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[CST_2:.*]] = "tf.Const"() {value = dense<1.38408304E-5> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[CST_3:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: %[[CST_4:.*]] = "tf.Const"() {value = dense<0.0027450982> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[CST_5:.*]] = "tf.Const"() {value = dense<-19> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: %[[CST_6:.*]] = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK-NEXT: %[[V0:.*]] = "tf.PartitionedCall"(%[[ARG1]], %[[CST_1]], %[[CST_0]]) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<2x3x3x2xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3x3x2xi8>
// CHECK-NEXT: %[[V1:.*]] = "tf.PartitionedCall"(%[[ARG0]], %[[CST]], %[[CST_0]]) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<1x3x4x3xf32>, tensor<f32>, tensor<i32>) -> tensor<1x3x4x3xi8>
// CHECK-NEXT: %[[V2:.*]] = "tf.PartitionedCall"(%[[V1]], %[[V0]], %[[CST_6]], %[[CST]], %[[CST_0]], %[[CST_1]], %[[CST_0]], %[[CST_2]], %[[CST_3]], %[[CST_4]], %[[CST_5]]) {config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_with_bias_fn_0} : (tensor<1x3x4x3xi8>, tensor<2x3x3x2xi8>, tensor<2xi32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<*xi8>
// CHECK-NEXT: %[[V3:.*]] = "tf.PartitionedCall"(%[[V2]], %[[CST_4]], %[[CST_5]]) {config = "", config_proto = "", executor_type = "", f = @dequantize_i8} : (tensor<*xi8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK-NEXT: return %[[V3]] : tensor<*xf32>

// CHECK: func private @quantize_i8(
// CHECK: func private @dequantize_i8(
// CHECK: func private @quantized_conv2d_with_bias_fn_0(
