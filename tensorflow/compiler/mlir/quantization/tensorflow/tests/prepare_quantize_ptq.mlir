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
      executor_type = "", f = @fused_matmul_with_bias_fn_1
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

  func.func private @fused_matmul_with_bias_fn_1(%a: tensor<*xf32>, %b: tensor<*xf32>, %c: tensor<*xf32>) -> tensor<*xf32> {
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
// CHECK-SAME: f = @fused_matmul_with_bias_fn_1
// CHECK: %[[q3:.*]] = "quant.qcast"(%[[call]])
// CHECK-SAME: quant.uniform<i8:f32, 0.015686274509803921:-1>
// CHECK: %[[dq3:.*]] = "quant.dcast"(%[[q3]])
// CHECK-SAME: quant.uniform<i8:f32, 0.015686274509803921:-1>
// CHECK: %[[identity:.*]] = "tf.Identity"(%[[dq3]])
// CHECK: "quant.qcast"(%[[identity]])
// CHECK-SAME: quant.uniform<i8:f32, 0.015686274509803921:-1>
