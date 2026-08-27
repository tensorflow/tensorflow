// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
// RUN: litert-opt %s -tfl-optimize-op-order | FileCheck %s

// CHECK-LABEL: dequantize_pushdown
func.func @dequantize_pushdown(%arg0: tensor<1000x1000x!quant.uniform<i8:f32, 7.812500e-03>>, %arg1: tensor<1x1xi32>) -> tensor<1x1x1000xf32> {
  %0 = "tfl.dequantize"(%arg0) : (tensor<1000x1000x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1000x1000xf32>
  %1 = "tfl.gather"(%0, %arg1) {axis = 0 : i32, batch_dims = 0 : i32}: (tensor<1000x1000xf32>, tensor<1x1xi32>) -> tensor<1x1x1000xf32>
  func.return %1 : tensor<1x1x1000xf32>

// CHECK-NEXT: tfl.gather
// CHECK-NEXT: tfl.dequantize
}

// CHECK-LABEL: dequantize_pushdown_gather_with_reduction
func.func @dequantize_pushdown_gather_with_reduction(%arg0: tensor<2xi32>) -> tensor<2x2xf32> {
  %w = "tfl.pseudo_qconst"() {qtype = tensor<12x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, value = dense<127> : tensor<12x2xi8>} : () -> tensor<12x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
  %dq_w = "tfl.dequantize"(%w) : (tensor<12x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<12x2xf32>
  %emb = "tfl.gather"(%dq_w, %arg0) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<12x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  func.return %emb : tensor<2x2xf32>

// CHECK-NEXT: tfl.pseudo_qconst
// CHECK-NEXT: tfl.gather
// CHECK-NEXT: tfl.dequantize
}

// CHECK-LABEL: no_pushdown_multiple_inputs
func.func @no_pushdown_multiple_inputs(%arg0: tensor<1000x1000x!quant.uniform<i8:f32, 7.812500e-03>>, %arg1: tensor<1000x1000xf32>) -> tensor<2000x1000xf32> {
  %0 = "tfl.dequantize"(%arg0) : (tensor<1000x1000x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1000x1000xf32>
  %1 = "tfl.concatenation"(%0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1000x1000xf32>, tensor<1000x1000xf32>) -> tensor<2000x1000xf32>
  func.return %1 : tensor<2000x1000xf32>

// CHECK-NEXT: tfl.dequantize
// CHECK-NEXT: tfl.concatenation
}

// CHECK-LABEL: no_pushdown_multiple_outputs
func.func @no_pushdown_multiple_outputs(%arg0: tensor<1000x2x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1000xf32> {
  %0 = "tfl.dequantize"(%arg0) : (tensor<1000x2x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1000x2xf32>
  %1:2 = "tfl.unpack"(%0) {axis = 1 : i32, num = 2 : i32} : (tensor<1000x2xf32>) -> (tensor<1000xf32>, tensor<1000xf32>)
  func.return %1#0 : tensor<1000xf32>

// CHECK-NEXT: tfl.dequantize
// CHECK-NEXT: tfl.unpack
}

// CHECK-LABEL: pushdown_dynamic_shape
func.func @pushdown_dynamic_shape(%arg0: tensor<?x1000x1000x!quant.uniform<i8:f32, 7.812500e-03>>, %arg1: tensor<1x1xi32>) -> tensor<?x1x1000xf32> {
  %0 = "tfl.dequantize"(%arg0) : (tensor<?x1000x1000x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<?x1000x1000xf32>
  %1 = "tfl.gather"(%0, %arg1) {axis = 0 : i32, batch_dims = 0 : i32}: (tensor<?x1000x1000xf32>, tensor<1x1xi32>) -> tensor<?x1x1000xf32>
  func.return %1 : tensor<?x1x1000xf32>

// CHECK-NEXT: tfl.gather
// CHECK-NEXT: tfl.dequantize
}

// CHECK-LABEL: no_pushdown_gather_with_no_reduction
func.func @no_pushdown_gather_with_no_reduction(%arg0: tensor<2xi32>) -> tensor<2x2xf32> {
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, value = dense<127> : tensor<2x2xi8>} : () -> tensor<2x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
  %dq_w = "tfl.dequantize"(%w) : (tensor<2x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<2x2xf32>
  %emb = "tfl.gather"(%dq_w, %arg0) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  func.return %emb : tensor<2x2xf32>

// CHECK-NEXT: tfl.pseudo_qconst
// CHECK-NEXT: tfl.dequantize
// CHECK-NEXT: tfl.gather
}

