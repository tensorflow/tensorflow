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

// RUN: tf-quant-opt %s -split-input-file -quant-prepare-quantize -quant-test-post-training-quantize | FileCheck %s

// -----

module {
  func.func @same_scale_ptq_test(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    %cst = arith.constant dense<[-1, 144]> : tensor<2xi32>
    %cst_1 = arith.constant dense<1.0> : tensor<144x10xf32>
    %cst_2 = arith.constant dense<0.1> : tensor<10xf32>
    %0 = "quant.stats"(%arg0) {
      layerStats = dense<[-1.28, 1.28]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.MaxPool"(%0) {
      data_format = "NHWC", device = "", explicit_paddings = [],
      ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 2, 2, 1]
    } : (tensor<*xf32>) -> tensor<*xf32>
    %2 = "quant.stats"(%1) {
      layerStats = dense<[-0.9, 1.0]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    %3 = "tf.Reshape"(%2, %cst) {device = ""} : (tensor<*xf32>, tensor<2xi32>) -> tensor<*xf32>
    %4 = "quant.stats"(%3) {
      layerStats = dense<[-1.0, 0.8]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    %5 = "tf.PartitionedCall"(%4, %cst_1, %cst_2) {
      _tfl_quant_trait = "fully_quantizable", config = "", config_proto = "",
      executor_type = "", f = @composite_matmul_with_bias_fn_1
    } : (tensor<*xf32>, tensor<144x10xf32>, tensor<10xf32>) -> tensor<*xf32>
    %6 = "quant.stats"(%5) {
      layerStats = dense<[-2.0, 2.0]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    %7 = "tf.Identity"(%6) : (tensor<*xf32>) -> tensor<*xf32>
    %8 = "quant.stats"(%7) {
      layerStats = dense<[-2.0, 2.0]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    func.return %8 : tensor<*xf32>
  }

  func.func private @composite_matmul_with_bias_fn_1(%a: tensor<*xf32>, %b: tensor<*xf32>, %c: tensor<*xf32>) -> tensor<*xf32> {
    func.return %a: tensor<*xf32>
  }
}

// CHECK-LABEL: same_scale_ptq_test

// CHECK: %[[q0:.*]] = "quant.qcast"(%arg0)
// CHECK: %[[dq0:.*]] = "quant.dcast"(%[[q0]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[maxpool:.*]] = "tf.MaxPool"(%[[dq0]])
// CHECK: %[[q1:.*]] = "quant.qcast"(%[[maxpool]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[dq1:.*]] = "quant.dcast"(%[[q1]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[reshape:.*]] = "tf.Reshape"(%[[dq1]]
// CHECK: %[[q2:.*]] = "quant.qcast"(%[[reshape]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[dq2:.*]] = "quant.dcast"(%[[q2]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[call:.*]] = "tf.PartitionedCall"(%[[dq2]]
// CHECK-SAME: f = @composite_matmul_with_bias_fn_1
// CHECK: %[[q3:.*]] = "quant.qcast"(%[[call]])
// CHECK-SAME: quant.uniform<i8:f32, 0.015686274509803921:-1>
// CHECK: %[[dq3:.*]] = "quant.dcast"(%[[q3]])
// CHECK-SAME: quant.uniform<i8:f32, 0.015686274509803921:-1>
// CHECK: %[[identity:.*]] = "tf.Identity"(%[[dq3]])
// CHECK: "quant.qcast"(%[[identity]])
// CHECK-SAME: quant.uniform<i8:f32, 0.015686274509803921:-1>

// -----

module {
  func.func private @conv_with_bias_and_relu(%arg0: tensor<1x3x4x3xf32>) -> tensor<*xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[7.11401462, 7.05456924]> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[[[-6.30731344, 5.4962182], [1.80364347, -7.64542675], [-2.11145878, -7.08605719]], [[-9.54062747, -6.14013147], [6.12640238, -4.18223286], [5.05738974, 8.99269962]], [[3.3535192, 0.84816426], [-6.64676809, -7.95477629], [5.81315517, 9.21566581]]], [[[1.38622558, 4.63866329], [9.54742622, -1.43770897], [-7.96835279, 8.99996852]], [[0.989735424, -4.83384752], [-7.27702999, 1.17216611], [9.33735656, 0.728900194]], [[5.1286211, 8.98645591], [1.55008793, -3.85491467], [3.7003777, 9.26594448]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "quant.stats"(%arg0) {layerStats = dense<[1.27501142, 149.824783]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.PartitionedCall"(%0, %cst_0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", device = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_10} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "quant.stats"(%1) {layerStats = dense<[0.000000e+00, 6.000000e+00]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    return %2 : tensor<*xf32>
  }

  func.func private @composite_conv2d_with_bias_and_relu6_fn_10(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf.tf_quant.composite_function} {
    %0 = "quant.stats"(%arg1) {layerStats = dense<[-9.54062747, 9.54742622]> : tensor<2xf32>} : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2xf32>
    %1 = "quant.stats"(%arg0) {layerStats = dense<[1.27501142, 149.824783]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %2 = "tf.Conv2D"(%1, %0) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
    %3 = "quant.stats"(%arg2) {layerStats = dense<[7.05456924, 7.11401462]> : tensor<2xf32>} : (tensor<2xf32>) -> tensor<2xf32>
    %4 = "quant.stats"(%2) {layerStats = dense<[-2795.36523, 4609.57373]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %5 = "tf.BiasAdd"(%4, %3) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %6 = "quant.stats"(%5) {layerStats = dense<[-2788.31055, 4616.62842]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %7 = "tf.Relu6"(%6) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %8 = "quant.stats"(%7) {layerStats = dense<[0.000000e+00, 6.000000e+00]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    return %8 : tensor<*xf32>
  }

// CHECK-LABEL: conv_with_bias_and_relu
// CHECK: %[[cst:.*]] = arith.constant dense<[7.11401462, 7.05456924]> : tensor<2xf32>
// CHECK: %[[q0:.*]] = "quant.qcast"(%[[cst]]) {volatile}
// CHECK-SAME: tensor<2x!quant.uniform<i32:f32:0, {0.044169864606680966,0.042867627733627671}>>
// CHECK: %[[dq0:.*]] = "quant.dcast"(%[[q0]])

// CHECK: %[[cst_1:.*]] = arith.constant
// CHECK: %[[q1:.*]] = "quant.qcast"(%[[cst_1]]) {volatile}
// CHECK-SAME: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.075176584439014829,0.072960192762960605}
// CHECK: %[[dq1:.*]] = "quant.dcast"(%[[q1]])

// CHECK: %[[q2:.*]] = "quant.qcast"(%arg0)
// CHECK-SAME: tensor<1x3x4x3x!quant.uniform<i8:f32, 0.58754816990272674:-128>>
// CHECK: %[[dq2:.*]] = "quant.dcast"(%[[q2]])

// CHECK: %[[call:.*]] = "tf.PartitionedCall"(%[[dq2]], %[[dq1]], %[[dq0]])
// CHECK-SAME: f = @composite_conv2d_with_bias_and_relu6_fn_10
// CHECK: %[[q3:.*]] = "quant.qcast"(%[[call]]) {volatile}
// CHECK-SAME: tensor<*x!quant.uniform<i8:f32, 0.023529411764705882:-128>>
// CHECK: %[[dq3:.*]] = "quant.dcast"(%[[q3]])
}
