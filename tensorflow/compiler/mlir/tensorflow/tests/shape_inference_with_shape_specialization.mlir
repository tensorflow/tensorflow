// Copyright 2026 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tf-opt %s -tf-shape-inference=input-arg-shapes=1 -verify-diagnostics -split-input-file | FileCheck %s
// RUN: not tf-opt %s -tf-shape-inference=input-arg-shapes=* 2>&1 | FileCheck --check-prefix=INPUT_ARG_SHAPES_ERROR %s
// INPUT_ARG_SHAPES_ERROR: Missing input argument shapes

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    // CHECK-LABEL: func.func @main
    // CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    // CHECK-DAG: %[[CST_1:.*]] = "tf.Const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
    // CHECK-NEXT: %[[UQ:.*]] = "tf.UniformQuantize"(%arg0, %cst, %cst_0) <{quantization_axis = -1 : i64, quantization_max_val = 127 : i64, quantization_min_val = -128 : i64}> : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>
    // CHECK-NEXT: %[[UDQ:.*]] = "tf.UniformDequantize"(%[[UQ]], %[[CST_0]], %[[CST_1]]) <{quantization_axis = -1 : i64, quantization_max_val = 127 : i64, quantization_min_val = -128 : i64}> : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1xf32>
    // CHECK-NEXT: return %[[UDQ]] : tensor<1xf32>
    func.func @main(%arg0 : tensor<?xf32>) -> tensor<?xf32> {
      %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
      %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

      %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x!tf_type.qint8>
      %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<?x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<?xf32>
      func.return %1 : tensor<?xf32>
    }
}

// -----

// expected-error@+1 {{Input shapes provided but no `main` function found.}}
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @non_main(%arg0 : tensor<?xf32>) -> tensor<?xf32> {
      %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
      %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

      %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x!tf_type.qint8>
      %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<?x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<?xf32>
      func.return %1 : tensor<?xf32>
    }
}
