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

// RUN: tf-quant-opt %s -quant-prepare-lifting | FileCheck %s

func.func @decompose_batch_norm(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %add, %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%arg0, %cst, %cst_0, %cst_0, %cst) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return %add : tensor<*xf32>
}
// CHECK: func @decompose_batch_norm
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.49743462E-5> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.999950051> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[mul:.*]] = "tf.Mul"(%arg0, %[[CONST_0]]) : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
// CHECK: %[[add:.*]] = "tf.Add"(%[[mul]], %[[CONST]]) : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
// CHECK-NEXT: return %[[add]] : tensor<*xf32>

func.func @not_decompose_batch_norm(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %bn, %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%arg0, %cst, %cst_0, %cst_0, %cst) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = true} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return %bn : tensor<*xf32>
}
// CHECK: func @not_decompose_batch_norm
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[bn:.*]], %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%arg0, %[[CONST]], %[[CONST_0]], %[[CONST_0]], %[[CONST]]) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = true} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
// CHECK-NEXT: return %[[bn]] : tensor<*xf32>

func.func @convert_add_to_biasadd(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.Add"(%0, %cst_0) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %1 : tensor<1x3x2x2xf32>
}
// CHECK: func @convert_add_to_biasadd
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<1x3x2x2xf32>

func.func @not_convert_add_to_biasadd(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x3xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x3xf32>} : () -> tensor<2x3x3x3xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<1x3x2x3xf32>} : () -> tensor<1x3x2x3xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x3xf32>) -> tensor<1x3x2x3xf32>
  %1 = "tf.Add"(%0, %cst_0) : (tensor<1x3x2x3xf32>, tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32>
  func.return %1 : tensor<1x3x2x3xf32>
}
// CHECK: func @not_convert_add_to_biasadd
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x3xf32>} : () -> tensor<2x3x3x3xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<1x3x2x3xf32>} : () -> tensor<1x3x2x3xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x3xf32>) -> tensor<1x3x2x3xf32>
// CHECK-NEXT: %[[ADD:.*]] = "tf.Add"(%[[CONV2D]], %[[CONST_0]]) : (tensor<1x3x2x3xf32>, tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32>
// CHECK-NEXT: return %[[ADD]] : tensor<1x3x2x3xf32>

func.func @fuse_conv2d_and_mul(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.Mul"(%0, %cst_0) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %1 : tensor<1x3x2x2xf32>
}
// CHECK: func @fuse_conv2d_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<8.000000e-01> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[CONV2D]] : tensor<1x3x2x2xf32>

func.func @not_fuse_conv2d_and_mul(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.Mul"(%0, %cst_0) : (tensor<1x3x2x2xf32>, tensor<2x2xf32>) -> tensor<1x3x2x2xf32>
  func.return %1 : tensor<1x3x2x2xf32>
}
// CHECK: func @not_fuse_conv2d_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<4.000000e-01> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[ADD:.*]] = "tf.Mul"(%[[CONV2D]], %[[CONST_0]]) : (tensor<1x3x2x2xf32>, tensor<2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[ADD]] : tensor<1x3x2x2xf32>

func.func @fuse_conv2d_with_bias_and_mul(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  %2 = "tf.Mul"(%1, %cst_1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %2 : tensor<1x3x2x2xf32>
}
// CHECK: func @fuse_conv2d_with_bias_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<2.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<1x3x2x2xf32>

func.func @not_fuse_conv2d_with_bias_and_mul(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>, tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {value = dense<0.800000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  %2 = "tf.Mul"(%0, %cst_1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %1, %2 : tensor<1x3x2x2xf32>, tensor<1x3x2x2xf32>
}
// CHECK: func @not_fuse_conv2d_with_bias_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<4.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<8.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[MUL:.*]] = "tf.Mul"(%[[CONV2D]], %[[CONST_1]]) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[BIASADD]], %[[MUL]] : tensor<1x3x2x2xf32>, tensor<1x3x2x2xf32>

func.func @fuse_conv2d_with_bias_and_add(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  %2 = "tf.Add"(%1, %cst_1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %2 : tensor<1x3x2x2xf32>
}
// CHECK: func @fuse_conv2d_with_bias_and_add
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<1x3x2x2xf32>

func.func @not_fuse_conv2d_with_bias_and_add(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  %2 = "tf.Add"(%1, %arg1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %2 : tensor<1x3x2x2xf32>
}
// CHECK: func @not_fuse_conv2d_with_bias_and_add
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<4.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[ADD:.*]] = "tf.Add"(%[[BIASADD]], %arg1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[ADD]] : tensor<1x3x2x2xf32>

func.func @match_depthwise_conv2d_and_add(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %1 = "tf.Add"(%0, %cst_0) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}
// CHECK: func @match_depthwise_conv2d_and_add
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<4.000000e-01> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-NEXT: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[DEPTHWISE_CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<*xf32>

func.func @match_depthwise_conv2d_and_mul(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %1 = "tf.Mul"(%0, %cst_0) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}
// CHECK: func @match_depthwise_conv2d_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<8.000000e-01> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK-NEXT: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
// CHECK-NEXT: return %[[DEPTHWISE_CONV2D]] : tensor<*xf32>

func.func @match_depthwise_conv2d_with_bias_and_add(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  %2 = "tf.Add"(%1, %cst_1) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}
// CHECK: func @match_depthwise_conv2d_with_bias_and_add
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<8.000000e-01> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-NEXT: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[DEPTHWISE_CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<*xf32>

func.func @match_depthwise_conv2d_with_bias_and_mul(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  %2 = "tf.Mul"(%1, %cst_1) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}
// CHECK: func @match_depthwise_conv2d_with_bias_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<2.000000e-01> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-NEXT: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[DEPTHWISE_CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<*xf32>
