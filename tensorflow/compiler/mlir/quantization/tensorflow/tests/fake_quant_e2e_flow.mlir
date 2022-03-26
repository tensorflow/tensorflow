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
  %2 = "tf._FusedConv2D"(%1, %0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd"], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %3 = "tf.FakeQuantWithMinMaxArgs"(%2) {device = "", max = 4.000000e-01 : f32, min = -3.000000e-01 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// CHECK: func @fake_quant_conv(
// CHECK-NEXT: %cst = "tf.Const"() {value = dense<0.00117647066> : tensor<f32>} : () -> tensor<f32>
// CHECK-NEXT: %cst_0 = "tf.Const"() {value = dense<-43> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT: %cst_1 = "tf.Const"() {value = dense<0.0117647061> : tensor<f32>} : () -> tensor<f32>
// CHECK-NEXT: %cst_2 = "tf.Const"() {value = dense<1.38408304E-5> : tensor<f32>} : () -> tensor<f32>
// CHECK-NEXT: %cst_3 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT: %cst_4 = "tf.Const"() {value = dense<0.0027450982> : tensor<f32>} : () -> tensor<f32>
// CHECK-NEXT: %cst_5 = "tf.Const"() {value = dense<-19> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT: %cst_6 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK-NEXT: %0 = "tf.PartitionedCall"(%arg1, %cst_1, %cst_0) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<2x3x3x2xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3x3x2xi8>
// CHECK-NEXT: %1 = "tf.PartitionedCall"(%arg0, %cst, %cst_0) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<1x3x4x3xf32>, tensor<f32>, tensor<i32>) -> tensor<1x3x4x3xi8>
// CHECK-NEXT: %2 = "tf.PartitionedCall"(%1, %0, %cst_6, %cst, %cst_0, %cst_1, %cst_0, %cst_2, %cst_3, %cst_4, %cst_5) {config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_fn_0} : (tensor<1x3x4x3xi8>, tensor<2x3x3x2xi8>, tensor<2xi32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<*xi8>
// CHECK-NEXT: %3 = "tf.PartitionedCall"(%2, %cst_4, %cst_5) {config = "", config_proto = "", executor_type = "", f = @dequantize_i8} : (tensor<*xi8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK-NEXT: return %3 : tensor<*xf32>

// CHECK: func private @quantize_i8(
// CHECK: func private @dequantize_i8(
// CHECK: func private @quantized_conv2d_fn_0(
