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

func.func @fold_batch_norm(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %add, %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%arg0, %cst, %cst_0, %cst_0, %cst) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return %add : tensor<*xf32>
}
// CHECK: func @fold_batch_norm
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.49743462E-5> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.999950051> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[mul:.*]] = "tf.Mul"(%arg0, %[[CONST_0]]) : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
// CHECK: %[[add:.*]] = "tf.Add"(%[[mul]], %[[CONST]]) : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
// CHECK-NEXT: return %[[add]] : tensor<*xf32>

func.func @not_fold_batch_norm(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %bn, %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%arg0, %cst, %cst_0, %cst_0, %cst) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = true} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return %bn : tensor<*xf32>
}
// CHECK: func @not_fold_batch_norm
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[bn:.*]], %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%arg0, %[[CONST]], %[[CONST_0]], %[[CONST_0]], %[[CONST]]) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = true} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
// CHECK-NEXT: return %[[bn]] : tensor<*xf32>
