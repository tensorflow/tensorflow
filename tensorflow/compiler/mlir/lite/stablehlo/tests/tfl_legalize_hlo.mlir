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

// COM: This file is there to check that the `tfl-legalize-hlo` pass exists in `odml-to-stablehlo-opt`.

// RUN: odml-to-stablehlo-opt %s -tfl-legalize-hlo -split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: main
func.func @main(%arg0: tensor<5x7xf32>) -> tensor<5x7xf32> {
  func.return %arg0: tensor<5x7xf32>
}

// CHECK: return %arg0 : tensor<5x7xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.transpose
//===----------------------------------------------------------------------===//

// CHECK-LABEL: transpose_2d
func.func @transpose_2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "stablehlo.transpose"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<2x3xf32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK: %[[OUT:.*]] = "tfl.transpose"(%arg0, %[[CST]]) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK: return %[[OUT]] : tensor<3x2xf32>

// -----

// CHECK-LABEL: transpose_3d
func.func @transpose_3d(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = "stablehlo.transpose"(%arg0) <{permutation = array<i64: 2, 1, 0>}> : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
// CHECK: %[[OUT:.*]] = "tfl.transpose"(%arg0, %[[CST]]) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x2x1xf32>
// CHECK: return %[[OUT]] : tensor<3x2x1xf32>

// -----

// CHECK-LABEL: transpose_dynamic_2d
func.func @transpose_dynamic_2d(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %0 = "stablehlo.transpose"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<?x4xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK: %[[OUT:.*]] = "tfl.transpose"(%arg0, %[[CST]]) : (tensor<?x4xf32>, tensor<2xi32>) -> tensor<4x?xf32>
// CHECK: return %[[OUT]] : tensor<4x?xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.dot_general
//===----------------------------------------------------------------------===//

// CHECK-LABEL: dot_general_transposed_contracting_dims_3d
func.func @dot_general_transposed_contracting_dims_3d(%arg0: tensor<2x8x256x1792xf32>, %arg1: tensor<1x2x128x8x256xf32>) -> (tensor<1x128x1792xf32>){
  %4 = "stablehlo.dot_general"(%arg1, %arg0)
    <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [4, 3, 1],
    rhs_contracting_dimensions = [2, 1, 0]>}>: (tensor<1x2x128x8x256xf32>,
    tensor<2x8x256x1792xf32>) -> tensor<1x128x1792xf32>
  return %4 : tensor<1x128x1792xf32>
  // CHECK  %0 = "tfl.pseudo_const"() <{value = dense<[0, 2, 1, 3, 4]> : tensor<5xi64>}> : () -> tensor<5xi64>
  // CHECK  %1 = "tfl.cast"(%0) : (tensor<5xi64>) -> tensor<5xi32>
  // CHECK  %2 = "tfl.transpose"(%arg1, %1) : (tensor<1x2x128x8x256xf32>, tensor<5xi32>) -> tensor<1x128x2x8x256xf32>
  // CHECK  %cst = arith.constant dense<[128, 4096]> : tensor<2xi64>
  // CHECK  %3 = "tfl.cast"(%cst) : (tensor<2xi64>) -> tensor<2xi32>
  // CHECK  %4 = "tfl.reshape"(%2, %3) : (tensor<1x128x2x8x256xf32>, tensor<2xi32>) -> tensor<128x4096xf32>
  // CHECK  %cst_0 = arith.constant dense<[4096, 1792]> : tensor<2xi64>
  // CHECK  %5 = "tfl.cast"(%cst_0) : (tensor<2xi64>) -> tensor<2xi32>
  // CHECK  %6 = "tfl.reshape"(%arg0, %5) : (tensor<2x8x256x1792xf32>, tensor<2xi32>) -> tensor<4096x1792xf32>
  // CHECK  %7 = "tfl.batch_matmul"(%4, %6) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<128x4096xf32>, tensor<4096x1792xf32>) -> tensor<128x1792xf32>
  // CHECK  %cst_1 = arith.constant dense<[1, 128, 1792]> : tensor<3xi64>
  // CHECK  %8 = "tfl.cast"(%cst_1) : (tensor<3xi64>) -> tensor<3xi32>
  // CHECK  %9 = "tfl.reshape"(%7, %8) : (tensor<128x1792xf32>, tensor<3xi32>) -> tensor<1x128x1792xf32>
  // CHECK  return %9 : tensor<1x128x1792xf32>
}

// -----

// CHECK-LABEL: dot_general_transposed_contracting_dims
func.func @dot_general_transposed_contracting_dims(%arg0: tensor<64x8x8xf32>, %arg1: tensor<1x10x8x8xf32>) -> (tensor<1x10x64xf32>){
  %4 = "stablehlo.dot_general"(%arg1, %arg0)
    <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [3, 2],
    rhs_contracting_dimensions = [2, 1]>}>
    : (tensor<1x10x8x8xf32>, tensor<64x8x8xf32>) -> tensor<1x10x64xf32>
  return %4 : tensor<1x10x64xf32>
  // CHECK  %0 = "tfl.pseudo_const"() <{value = dense<[1, 2, 0]> : tensor<3xi64>}> : () -> tensor<3xi64>
  // CHECK  %1 = "tfl.cast"(%0) : (tensor<3xi64>) -> tensor<3xi32>
  // CHECK  %2 = "tfl.transpose"(%arg0, %1) : (tensor<64x8x8xf32>, tensor<3xi32>) -> tensor<8x8x64xf32>
  // CHECK  %cst = arith.constant dense<[10, 64]> : tensor<2xi64>
  // CHECK  %3 = "tfl.cast"(%cst) : (tensor<2xi64>) -> tensor<2xi32>
  // CHECK  %4 = "tfl.reshape"(%arg1, %3) : (tensor<1x10x8x8xf32>, tensor<2xi32>) -> tensor<10x64xf32>
  // CHECK  %cst_0 = arith.constant dense<64> : tensor<2xi64>
  // CHECK  %5 = "tfl.cast"(%cst_0) : (tensor<2xi64>) -> tensor<2xi32>
  // CHECK  %6 = "tfl.reshape"(%2, %5) : (tensor<8x8x64xf32>, tensor<2xi32>) -> tensor<64x64xf32>
  // CHECK  %7 = "tfl.batch_matmul"(%4, %6) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<10x64xf32>, tensor<64x64xf32>) -> tensor<10x64xf32>
  // CHECK  %cst_1 = arith.constant dense<[1, 10, 64]> : tensor<3xi64>
  // CHECK  %8 = "tfl.cast"(%cst_1) : (tensor<3xi64>) -> tensor<3xi32>
  // CHECK  %9 = "tfl.reshape"(%7, %8) : (tensor<10x64xf32>, tensor<3xi32>) -> tensor<1x10x64xf32>
  // CHECK  return %9 : tensor<1x10x64xf32>
}

// -----

// CHECK-LABEL: dot_general_non_transposed_contracting_dims
func.func @dot_general_non_transposed_contracting_dims(%arg0: tensor<8x256x1792xf32>, %arg1: tensor<1x128x8x256xf32>) -> (tensor<1x128x1792xf32>){
  %4 = "stablehlo.dot_general"(%arg1, %arg0)
    <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [3, 2],
    rhs_contracting_dimensions = [1, 0]>}>
    : (tensor<1x128x8x256xf32>, tensor<8x256x1792xf32>) -> tensor<1x128x1792xf32>
  return %4 : tensor<1x128x1792xf32>
  // CHECK  %cst = arith.constant dense<[128, 2048]> : tensor<2xi64>
  // CHECK  %0 = "tfl.cast"(%cst) : (tensor<2xi64>) -> tensor<2xi32>
  // CHECK  %1 = "tfl.reshape"(%arg1, %0) : (tensor<1x128x8x256xf32>, tensor<2xi32>) -> tensor<128x2048xf32>
  // CHECK  %cst_0 = arith.constant dense<[2048, 1792]> : tensor<2xi64>
  // CHECK  %2 = "tfl.cast"(%cst_0) : (tensor<2xi64>) -> tensor<2xi32>
  // CHECK  %3 = "tfl.reshape"(%arg0, %2) : (tensor<8x256x1792xf32>, tensor<2xi32>) -> tensor<2048x1792xf32>
  // CHECK  %4 = "tfl.batch_matmul"(%1, %3) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<128x2048xf32>, tensor<2048x1792xf32>) -> tensor<128x1792xf32>
  // CHECK  %cst_1 = arith.constant dense<[1, 128, 1792]> : tensor<3xi64>
  // CHECK  %5 = "tfl.cast"(%cst_1) : (tensor<3xi64>) -> tensor<3xi32>
  // CHECK  %6 = "tfl.reshape"(%4, %5) : (tensor<128x1792xf32>, tensor<3xi32>) -> tensor<1x128x1792xf32>
  // CHECK  return %6 : tensor<1x128x1792xf32>
}

// -----

// CHECK-LABEL: dot_general
func.func @dot_general(%arg0: tensor<3x2x6x5x1xf32>, %arg1: tensor<3x2x4x6xf32>) -> tensor<3x5x1x4xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1, 2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1, 3]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<3x2x6x5x1xf32>, tensor<3x2x4x6xf32>) -> tensor<3x5x1x4xf32>
  func.return %0 : tensor<3x5x1x4xf32>
}

// CHECK: %[[TRANSPOSED_0:.*]] = "tfl.transpose"
// CHECK: %[[TRANSPOSED_1:.*]] = "tfl.transpose"
// CHECK: %[[RESHAPED_0:.*]] = "tfl.reshape"(%[[TRANSPOSED_0]]
// CHECK: %[[RESHAPED_1:.*]] = "tfl.reshape"(%[[TRANSPOSED_1]]
// CHECK: %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %[[RESHAPED_1]]) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<3x5x12xf32>, tensor<3x12x4xf32>) -> tensor<3x5x4xf32>
// CHECK: %[[RESHAPED_BMM:.*]] = "tfl.reshape"(%[[BMM_0]]
// CHECK: return %[[RESHAPED_BMM]] : tensor<3x5x1x4xf32>

// -----

// CHECK-LABEL: dot_general_repeated
func.func @dot_general_repeated(%arg0: tensor<1x1x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1x1x1024xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x1x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x1x1024xf32>
  func.return %0 : tensor<1x1x1024xf32>
}

// CHECK: "tfl.batch_matmul"
// CHECK: return {{.*}} : tensor<1x1x1024xf32>

// -----

// CHECK-LABEL: dot_general_int8
func.func @dot_general_int8(%arg0: tensor<256xi8>, %arg1: tensor<256x8xi8>) -> tensor<8xi32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<256xi8>, tensor<256x8xi8>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// CHECK: "tfl.batch_matmul"
// CHECK: return {{.*}} : tensor<8xi32>

// -----

// CHECK-LABEL: dot_general_dynamic_rhs_out_dim
func.func @dot_general_dynamic_rhs_out_dim(%arg0: tensor<4x4x256xf32>, %arg1: tensor<4x?x256xf32>) -> tensor<4x4x?xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]
    >} : (tensor<4x4x256xf32>, tensor<4x?x256xf32>) -> tensor<4x4x?xf32>
  func.return %0 : tensor<4x4x?xf32>
}

// CHECK:     %[[MATMUL_R:.*]] = "tfl.batch_matmul"
// CHECK:     %[[RESHAPE_OUT_R:.*]] = "tfl.reshape"(%[[MATMUL_R]], %{{.*}}) : (tensor<4x4x?xf32>, tensor<3xi32>) -> tensor<4x4x?xf32>
// CHECK:     return %[[RESHAPE_OUT_R]] : tensor<4x4x?xf32>

// -----

// CHECK-LABEL: dot_general_dynamic_batch_dim
func.func @dot_general_dynamic_batch_dim(%arg0: tensor<2x?x2x3xf32>, %arg1: tensor<2x?x4x3xf32>) -> tensor<2x?x2x4xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [3]
    >} : (tensor<2x?x2x3xf32>, tensor<2x?x4x3xf32>) -> tensor<2x?x2x4xf32>
  func.return %0 : tensor<2x?x2x4xf32>
}

// CHECK:     %[[MATMUL_0:.*]] = "tfl.batch_matmul"
// CHECK:     %[[RESHAPE_OUT:.*]] = "tfl.reshape"(%[[MATMUL_0]], %{{.*}}) : (tensor<2x?x2x4xf32>, tensor<4xi32>) -> tensor<2x?x2x4xf32>
// CHECK:     return %[[RESHAPE_OUT]] : tensor<2x?x2x4xf32>

// -----

// CHECK-LABEL: dot_general_dynamic_lhs_rhs_out_dims
func.func @dot_general_dynamic_lhs_rhs_out_dims(%arg0: tensor<2x2x?x3xf32>, %arg1: tensor<2x4x?x3xf32>) -> tensor<2x2x?x4x?xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [3]
    >} : (tensor<2x2x?x3xf32>, tensor<2x4x?x3xf32>) -> tensor<2x2x?x4x?xf32>
  func.return %0 : tensor<2x2x?x4x?xf32>
}

// CHECK:     %[[VAL_12:.*]] = "tfl.reshape"
// CHECK:     %[[VAL_22:.*]] = "tfl.reshape"
// CHECK:     %[[VAL_23:.*]] = "tfl.batch_matmul"(%[[VAL_12]], %[[VAL_22]]) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<2x?x3xf32>, tensor<2x3x?xf32>) -> tensor<2x?x?xf32>
// CHECK:     %[[VAL_32:.*]] = "tfl.reshape"(%[[VAL_23]], %{{.*}}) : (tensor<2x?x?xf32>, tensor<5xi32>) -> tensor<2x2x?x4x?xf32>
// CHECK:     return %[[VAL_32]] : tensor<2x2x?x4x?xf32>

// -----

// CHECK-LABEL: dot_general_dynamic_contracting_dim
func.func @dot_general_dynamic_contracting_dim(%arg0: tensor<4x4x?xf32>, %arg1: tensor<4x?x256xf32>) -> tensor<4x4x256xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<4x4x?xf32>, tensor<4x?x256xf32>) -> tensor<4x4x256xf32>
  func.return %0 : tensor<4x4x256xf32>
}

// CHECK:     %[[MATMUL:.*]] = "tfl.batch_matmul"
// CHECK:     %[[FINAL_RESHAPE:.*]] = "tfl.reshape"(%[[MATMUL]]
// CHECK:     return %[[FINAL_RESHAPE]] : tensor<4x4x256xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.reduce
//===----------------------------------------------------------------------===//

// CHECK-LABEL:   convert_reduce_maximum_to_reduce_any
func.func @convert_reduce_maximum_to_reduce_any(%arg0: tensor<128x1x1024xi1>) -> tensor<128x1xi1> {
    %0 = stablehlo.constant dense<false> : tensor<i1>
    %1271 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.maximum across dimensions = [2] : (tensor<128x1x1024xi1>, tensor<i1>) -> tensor<128x1xi1>
    func.return %1271 : tensor<128x1xi1>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2> : tensor<1xi32>
// CHECK: "tfl.reduce_any"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<128x1x1024xi1>, tensor<1xi32>) -> tensor<128x1xi1>

// -----

// CHECK-LABEL: argmax
func.func @argmax(%arg0: tensor<4x32x256xf32>) -> (tensor<4x32xf32>, tensor<4x32xi32>) {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %2 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<256xi32>
  %3 = "stablehlo.broadcast_in_dim"(%2) <{broadcast_dimensions = array<i64: 2>}> : (tensor<256xi32>) -> tensor<4x32x256xi32>
  %4:2 = "stablehlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "stablehlo.compare"(%arg1, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.or %7, %8 : tensor<i1>
    %10 = "stablehlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = stablehlo.and %11, %12 : tensor<i1>
    %14 = stablehlo.or %9, %13 : tensor<i1>
    %15 = "stablehlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = array<i64: 2>} : (tensor<4x32x256xf32>, tensor<4x32x256xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x32xf32>, tensor<4x32xi32>)
  func.return %4#0, %4#1 : tensor<4x32xf32>, tensor<4x32xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2> : tensor<1xi32>
// CHECK: %[[REDUCE:.*]] = "tfl.reduce_max"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xf32>
// CHECK: %[[ARG:.*]] = "tfl.arg_max"(%arg0, %[[CST]]) : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xi32>
// CHECK: return %[[REDUCE]], %[[ARG]] : tensor<4x32xf32>, tensor<4x32xi32>

// -----

// CHECK-LABEL: argmax_constant
func.func @argmax_constant(%arg0: tensor<2x2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %3 = stablehlo.constant dense<[[[0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x2x4xi32>
  %4:2 = "stablehlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "stablehlo.compare"(%arg1, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.or %7, %8 : tensor<i1>
    %10 = "stablehlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = stablehlo.and %11, %12 : tensor<i1>
    %14 = stablehlo.or %9, %13 : tensor<i1>
    %15 = "stablehlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = array<i64: 2>} : (tensor<2x2x4xf32>, tensor<2x2x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %4#0, %4#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2> : tensor<1xi32>
// CHECK: %[[REDUCE:.*]] = "tfl.reduce_max"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xf32>
// CHECK: %[[ARG_MAX:.*]] = "tfl.arg_max"(%arg0, %[[CST]]) : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xi32>
// CHECK: return %[[REDUCE:.*]], %[[ARG_MAX]] : tensor<2x2xf32>, tensor<2x2xi32>

// -----

// CHECK-LABEL: argmax_constant_non_z_axis
func.func @argmax_constant_non_z_axis(%arg0: tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %3 = stablehlo.constant dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]> : tensor<4x4xi32>
  %4:2 = "stablehlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "stablehlo.compare"(%arg1, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.or %7, %8 : tensor<i1>
    %10 = "stablehlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = stablehlo.and %11, %12 : tensor<i1>
    %14 = stablehlo.or %9, %13 : tensor<i1>
    %15 = "stablehlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<4x4xf32>, tensor<4x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<4xf32>, tensor<4xi32>)
  func.return %4#0, %4#1 : tensor<4xf32>, tensor<4xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<1xi32>
// CHECK: %[[REDUCE:.*]] = "tfl.reduce_max"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<4x4xf32>, tensor<1xi32>) -> tensor<4xf32>
// CHECK: %[[ARG_MAX:.*]] = "tfl.arg_max"(%arg0, %[[CST]]) : (tensor<4x4xf32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK: return %[[REDUCE]], %[[ARG_MAX]] : tensor<4xf32>, tensor<4xi32>

// -----

// CHECK-LABEL: argmax_bool
func.func @argmax_bool(%arg0: tensor<2xi1>) -> tensor<i32> {
  %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xi32>
  %1 = stablehlo.constant dense<false> : tensor<i1>
  %2 = stablehlo.constant dense<0> : tensor<i32>
  %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<2xi1>, tensor<2xi32>, tensor<i1>, tensor<i32>) -> (tensor<i1>, tensor<i32>)
    reducer(%arg1: tensor<i1>, %arg3: tensor<i1>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %4 = stablehlo.compare  GT, %arg1, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<i1>
    %6 = stablehlo.compare  EQ, %arg1, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = stablehlo.compare  LT, %arg2, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %8 = stablehlo.and %6, %7 : tensor<i1>
    %9 = stablehlo.or %4, %8 : tensor<i1>
    %10 = stablehlo.select %9, %arg2, %arg4 : tensor<i1>, tensor<i32>
    stablehlo.return %5, %10 : tensor<i1>, tensor<i32>
  }
  return %3#1 : tensor<i32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<1xi32>
// CHECK: "tfl.reduce_any"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<2xi1>, tensor<1xi32>) -> tensor<i1>

// -----

// CHECK-LABEL: argmin
func.func @argmin(%arg0: tensor<4x32x256xf32>) -> (tensor<4x32xf32>, tensor<4x32xi32>) {
  %0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %2 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<256xi32>
  %3 = "stablehlo.broadcast_in_dim"(%2) <{broadcast_dimensions = array<i64: 2>}> : (tensor<256xi32>) -> tensor<4x32x256xi32>
  %4:2 = "stablehlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "stablehlo.compare"(%arg1, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.or %7, %8 : tensor<i1>
    %10 = "stablehlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = stablehlo.and %11, %12 : tensor<i1>
    %14 = stablehlo.or %9, %13 : tensor<i1>
    %15 = "stablehlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = array<i64: 2>} : (tensor<4x32x256xf32>, tensor<4x32x256xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x32xf32>, tensor<4x32xi32>)
  func.return %4#0, %4#1 : tensor<4x32xf32>, tensor<4x32xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2> : tensor<1xi32>
// CHECK: %[[REDUCE:.*]] = "tfl.reduce_min"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xf32>
// CHECK: %[[ARG:.*]] = "tfl.arg_min"(%arg0, %[[CST]]) : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xi32>
// CHECK: return %[[REDUCE]], %[[ARG]] : tensor<4x32xf32>, tensor<4x32xi32>

// -----

// CHECK-LABEL: argmin_i16
func.func @argmin_i16(%arg0: tensor<2xi16>) -> (tensor<i16>, tensor<i32>) {
  %0 = stablehlo.constant dense<false> : tensor<i1>
  %1 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xi32>
  %2 = stablehlo.constant dense<32767> : tensor<i16>
  %3 = stablehlo.constant dense<0> : tensor<i32>
  %4:2 = "stablehlo.reduce"(%arg0, %1, %2, %3) ({
  ^bb0(%arg1: tensor<i16>, %arg2: tensor<i32>, %arg3: tensor<i16>, %arg4: tensor<i32>):
    %11 = stablehlo.constant dense<false> : tensor<i1>
    %12 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i16>, tensor<i16>) -> tensor<i1>
    %13 = "stablehlo.select"(%12, %arg1, %arg3) : (tensor<i1>, tensor<i16>, tensor<i16>) -> tensor<i16>
    %14 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<i16>, tensor<i16>) -> tensor<i1>
    %15 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %16 = stablehlo.and %14, %15 : tensor<i1>
    %17 = stablehlo.or %12, %16 : tensor<i1>
    %18 = "stablehlo.select"(%17, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%13, %18) : (tensor<i16>, tensor<i32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<2xi16>, tensor<2xi32>, tensor<i16>, tensor<i32>) -> (tensor<i16>, tensor<i32>)
  func.return %4#0, %4#1 : tensor<i16>, tensor<i32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<1xi32>
// CHECK: %[[REDUCE:.*]] = "tfl.reduce_min"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<2xi16>, tensor<1xi32>) -> tensor<i16>
// CHECK: %[[ARG:.*]] = "tfl.arg_min"(%arg0, %[[CST]]) : (tensor<2xi16>, tensor<1xi32>) -> tensor<i32>
// CHECK: return %[[REDUCE]], %[[ARG]] : tensor<i16>, tensor<i32>

// -----

// CHECK-LABEL: argmin_constant
func.func @argmin_constant(%arg0: tensor<2x2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
  %0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %3 = stablehlo.constant dense<[[[0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x2x4xi32>
  %4:2 = "stablehlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "stablehlo.compare"(%arg1, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.or %7, %8 : tensor<i1>
    %10 = "stablehlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = stablehlo.and %11, %12 : tensor<i1>
    %14 = stablehlo.or %9, %13 : tensor<i1>
    %15 = "stablehlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = array<i64: 2>} : (tensor<2x2x4xf32>, tensor<2x2x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %4#0, %4#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2> : tensor<1xi32>
// CHECK: %[[REDUCE:.*]] = "tfl.reduce_min"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xf32>
// CHECK: %[[ARG_MIN:.*]] = "tfl.arg_min"(%arg0, %[[CST]]) : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xi32>
// CHECK: return %[[REDUCE]], %[[ARG_MIN]] : tensor<2x2xf32>, tensor<2x2xi32>

// -----

// CHECK-LABEL: argmin_bool
func.func @argmin_bool(%arg0: tensor<2xi1>) -> tensor<i32> {
  %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xi32>
  %1 = stablehlo.constant dense<false> : tensor<i1>
  %2 = stablehlo.constant dense<0> : tensor<i32>
  %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<2xi1>, tensor<2xi32>, tensor<i1>, tensor<i32>) -> (tensor<i1>, tensor<i32>)
    reducer(%arg1: tensor<i1>, %arg3: tensor<i1>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %4 = stablehlo.compare  LT, %arg1, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<i1>
    %6 = stablehlo.compare  EQ, %arg1, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = stablehlo.compare  LT, %arg2, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %8 = stablehlo.and %6, %7 : tensor<i1>
    %9 = stablehlo.or %4, %8 : tensor<i1>
    %10 = stablehlo.select %9, %arg2, %arg4 : tensor<i1>, tensor<i32>
    stablehlo.return %5, %10 : tensor<i1>, tensor<i32>
  }
  return %3#1 : tensor<i32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<1xi32>
// CHECK: %[[ARG_MIN:.*]] = "tfl.arg_min"(%arg0, %[[CST]]) : (tensor<2xi1>, tensor<1xi32>) -> tensor<i32>
// CHECK: return %[[ARG_MIN]] : tensor<i32>

// -----

// CHECK-LABEL: argmax_with_reshaped_iota
func.func @argmax_with_reshaped_iota(%arg0: tensor<1x32x1xf32>) -> (tensor<1x1xf32>, tensor<1x1xi32>) {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %2 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<32xi32>
  %3 = "stablehlo.reshape"(%2) : (tensor<32xi32>) -> tensor<1x32x1xi32>
  %4:2 = stablehlo.reduce(%arg0 init: %0), (%3 init: %1) across dimensions = [1] : (tensor<1x32x1xf32>, tensor<1x32x1xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x1xf32>, tensor<1x1xi32>)
   reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %5 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = "stablehlo.compare"(%arg1, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = stablehlo.or %5, %6 : tensor<i1>
    %8 = "stablehlo.select"(%7, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %9 = "stablehlo.compare"(%arg1, %arg3) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %10 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %11 = stablehlo.and %9, %10 : tensor<i1>
    %12 = stablehlo.or %7, %11 : tensor<i1>
    %13 = "stablehlo.select"(%12, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%8, %13) : (tensor<f32>, tensor<i32>) -> ()
  }
  func.return %4#0, %4#1 : tensor<1x1xf32>, tensor<1x1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %[[REDUCE:.*]] = "tfl.reduce_max"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x32x1xf32>, tensor<1xi32>) -> tensor<1x1xf32>
// CHECK: %[[ARG:.*]] = "tfl.arg_max"(%arg0, %[[CST]]) : (tensor<1x32x1xf32>, tensor<1xi32>) -> tensor<1x1xi32>
// CHECK: return %[[REDUCE]], %[[ARG]] : tensor<1x1xf32>, tensor<1x1xi32>

// -----

// CHECK-LABEL: pytorch_argmax
func.func @pytorch_argmax(%arg0: tensor<1x9xi32>) -> tensor<1xi32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<-2147483648> : tensor<i32>
  %2 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<9xi32>
  %3 = stablehlo.reshape %2 : (tensor<9xi32>) -> tensor<1x9xi32>
  %4:2 = stablehlo.reduce(%arg0 init: %1), (%3 init: %0) across dimensions = [1] : (tensor<1x9xi32>, tensor<1x9xi32>, tensor<i32>, tensor<i32>) -> (tensor<1xi32>, tensor<1xi32>)
    reducer(%arg1: tensor<i32>, %arg3: tensor<i32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %6 = stablehlo.compare  GE, %arg1, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<i32>
    %8 = stablehlo.compare  EQ, %arg1, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.minimum %arg2, %arg4 : tensor<i32>
    %10 = stablehlo.select %6, %arg2, %arg4 : tensor<i1>, tensor<i32>
    %11 = stablehlo.select %8, %9, %10 : tensor<i1>, tensor<i32>
    stablehlo.return %7, %11 : tensor<i32>, tensor<i32>
  }
  func.return %4#1 : tensor<1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %[[ARG:.*]] = "tfl.arg_max"(%arg0, %[[CST]]) : (tensor<1x9xi32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK: return %[[ARG]] : tensor<1xi32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.cbrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: cbrt_f32
func.func @cbrt_f32(%arg0: tensor<1x32x1xf32>) -> tensor<1x32x1xf32> {
  %0 = "stablehlo.cbrt"(%arg0) : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  func.return %0 : tensor<1x32x1xf32>
}

// CHECK-DAG: %cst = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %cst_0 = arith.constant dense<3.000000e+00> : tensor<f32>
// CHECK:     %0 = tfl.div %cst, %cst_0 {fused_activation_function = "NONE"} : tensor<f32>
// CHECK:     %1 = tfl.pow(%arg0, %0) : (tensor<1x32x1xf32>, tensor<f32>) -> tensor<1x32x1xf32>
// CHECK:     return %1 : tensor<1x32x1xf32>

// -----

// CHECK-LABEL: cbrt_f64
func.func @cbrt_f64(%arg0: tensor<1x32x1xf64>) -> tensor<1x32x1xf64> {
  %0 = "stablehlo.cbrt"(%arg0) : (tensor<1x32x1xf64>) -> tensor<1x32x1xf64>
  func.return %0 : tensor<1x32x1xf64>
}

// CHECK-NOT: tfl

// -----

//===----------------------------------------------------------------------===//
// stablehlo.(dynamic)reshape
//===----------------------------------------------------------------------===//

// CHECK-LABEL: reshape
func.func @reshape(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<2x3xf32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// CHECK: %cst = arith.constant dense<[3, 2]> : tensor<2xi64>
// CHECK: %0 = "tfl.cast"(%cst) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %1 = "tfl.reshape"(%arg0, %0) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>

// -----

// CHECK-LABEL: dynamic_reshape_i32
func.func @dynamic_reshape_i32(%arg0: tensor<2x3xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK: %0 = "tfl.cast"(%arg1) : (tensor<2xi32>) -> tensor<2xi32>
// CHECK: %1 = "tfl.reshape"(%arg0, %0) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>

// -----

// CHECK-LABEL: dynamic_reshape_i64
func.func @dynamic_reshape_i64(%arg0: tensor<2x3xf32>, %arg1: tensor<2xi64>) -> tensor<?x?xf32> {
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK: %0 = "tfl.cast"(%arg1) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %1 = "tfl.reshape"(%arg0, %0) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>

// -----

//===----------------------------------------------------------------------===//
// mhlo binary bit-wise ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: logical_and
func.func @logical_and(%arg0: tensor<2xi1>, %arg1: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.and %arg0, %arg1 : tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.logical_and
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: bitwise_and
func.func @bitwise_and(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.and %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK: stablehlo.and
// CHECK-NOT: tfl

// -----

// CHECK-LABEL: logical_or
func.func @logical_or(%arg0: tensor<2xi1>, %arg1: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.or %arg0, %arg1 : tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.logical_or
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: bitwise_or
func.func @bitwise_or(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.or %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK: stablehlo.or
// CHECK-NOT: tfl

// -----

// CHECK-LABEL: logical_xor
func.func @logical_xor(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// stablehlo.convolution
//===----------------------------------------------------------------------===//

//
// 2D
//=---

// CHECK-LABEL: transpose_conv2d_valid_padding_odd
func.func @transpose_conv2d_valid_padding_odd(%arg0: tensor<1x200x198x4xf32>, %arg1: tensor<4x4x4x4xf32>) -> tensor<1x402x398x4xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {pad = [[3, 3], [3, 3]],lhs_dilate = [2, 2]}
    {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x200x198x4xf32>, tensor<4x4x4x4xf32>) -> tensor<1x402x398x4xf32>
  func.return %0 : tensor<1x402x398x4xf32>
  // CHECK  %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
  // CHECK  %cst_0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  // CHECK  %0 = "tfl.reverse_v2"(%arg1, %cst_0) : (tensor<4x4x4x4xf32>, tensor<2xi32>) -> tensor<4x4x4x4xf32>
  // CHECK  %cst_1 = arith.constant dense<[1, 402, 398, 4]> : tensor<4xi32>
  // CHECK  %1 = "tfl.transpose_conv"(%cst_1, %0, %arg0, %cst) <{fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4xi32>, tensor<4x4x4x4xf32>, tensor<1x200x198x4xf32>, tensor<4xf32>) -> tensor<1x402x398x4xf32>
  // CHECK  return %1 : tensor<1x402x398x4xf32>
}

// CHECK-LABEL: transpose_conv2d_same_padding
func.func @transpose_conv2d_same_padding(%input: tensor<1x256x256x2xf32>, %filter:tensor<2x4x4x2xf32>) -> tensor<1x512x512x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2]}
    {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x256x256x2xf32>, tensor<2x4x4x2xf32>) -> tensor<1x512x512x2xf32>
    func.return %0 : tensor<1x512x512x2xf32>
  // CHECK  %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
  // CHECK  %cst_0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  // CHECK  %0 = "tfl.reverse_v2"(%arg1, %cst_0) : (tensor<2x4x4x2xf32>, tensor<2xi32>) -> tensor<2x4x4x2xf32>
  // CHECK  %1 = "tfl.pseudo_const"() <{value = dense<[1, 512, 512, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK  %2 = "tfl.transpose_conv"(%1, %0, %arg0, %cst) <{fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4xi32>, tensor<2x4x4x2xf32>, tensor<1x256x256x2xf32>, tensor<2xf32>) -> tensor<1x512x512x2xf32>
  // CHECK  return %2 : tensor<1x512x512x2xf32>
}

// -----

// CHECK-LABEL: transpose_conv2d_valid_padding
func.func @transpose_conv2d_valid_padding(%input: tensor<1x256x256x2xf32>, %filter:tensor<2x4x4x2xf32>) -> tensor<1x514x514x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {pad = [[3, 3], [3, 3]], lhs_dilate = [2, 2]}
    {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x256x256x2xf32>, tensor<2x4x4x2xf32>) -> tensor<1x514x514x2xf32>
  func.return %0 : tensor<1x514x514x2xf32>
  // CHECK  %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
  // CHECK  %cst_0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  // CHECK  %0 = "tfl.reverse_v2"(%arg1, %cst_0) : (tensor<2x4x4x2xf32>, tensor<2xi32>) -> tensor<2x4x4x2xf32>
  // CHECK  %1 = "tfl.pseudo_const"() <{value = dense<[1, 514, 514, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK  %2 = "tfl.transpose_conv"(%1, %0, %arg0, %cst) <{fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4xi32>, tensor<2x4x4x2xf32>, tensor<1x256x256x2xf32>, tensor<2xf32>) -> tensor<1x514x514x2xf32>
  // CHECK  return %2 : tensor<1x514x514x2xf32>
}

// -----

// CHECK-LABEL: transpose_conv2d_valid_padding_equal_strides
func.func @transpose_conv2d_valid_padding_equal_strides(%arg0: tensor<1x200x198x3xf32>, %arg1: tensor<3x3x3x3xf32>) -> tensor<1x401x397x3xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2]}
    {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x200x198x3xf32>, tensor<3x3x3x3xf32>) -> tensor<1x401x397x3xf32>
  func.return %0 : tensor<1x401x397x3xf32>
  // CHECK  %cst = arith.constant dense<0.000000e+00> : tensor<3xf32>
  // CHECK  %cst_0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  // CHECK  %0 = "tfl.reverse_v2"(%arg1, %cst_0) : (tensor<3x3x3x3xf32>, tensor<2xi32>) -> tensor<3x3x3x3xf32>
  // CHECK  %cst_1 = arith.constant dense<[1, 401, 397, 3]> : tensor<4xi32>
  // CHECK  %1 = "tfl.transpose_conv"(%cst_1, %0, %arg0, %cst) <{fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4xi32>, tensor<3x3x3x3xf32>, tensor<1x200x198x3xf32>, tensor<3xf32>) -> tensor<1x401x397x3xf32>
  // CHECK  return %1 : tensor<1x401x397x3xf32>
}

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc
func.func @conv2d_nhwc_ohwi_nhwc(%input: tensor<1x256x256x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>
  } : (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>, tensor<2xf32>) -> tensor<1x256x256x2xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_strides
func.func @conv2d_nhwc_ohwi_nhwc_no_strides(%arg0: tensor<1x64x64x4xf32>, %arg1: tensor<320x3x3x4xf32>) -> tensor<1x62x62x320xf32> {
	%0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>
  } : (tensor<1x64x64x4xf32>, tensor<320x3x3x4xf32>) -> tensor<1x62x62x320xf32>
  func.return %0 : tensor<1x62x62x320xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x64x64x4xf32>, tensor<320x3x3x4xf32>, tensor<320xf32>) -> tensor<1x62x62x320xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_padding
func.func @conv2d_nhwc_ohwi_nhwc_no_padding(%arg0: tensor<1x6x6x207xf32>, %arg1: tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    lhs_dilation = array<i64: 1, 1>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32>
  func.return %0 : tensor<1x4x4x16xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>, tensor<16xf32>) -> tensor<1x4x4x16xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_lhs_dilation
func.func @conv2d_nhwc_ohwi_nhwc_no_lhs_dilation(%arg0: tensor<1x6x6x207xf32>, %arg1: tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32>
  func.return %0 : tensor<1x4x4x16xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>, tensor<16xf32>) -> tensor<1x4x4x16xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_rhs_dilation
func.func @conv2d_nhwc_ohwi_nhwc_no_rhs_dilation(%arg0: tensor<1x6x6x207xf32>, %arg1: tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    lhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32>
  func.return %0 : tensor<1x4x4x16xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>, tensor<16xf32>) -> tensor<1x4x4x16xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_strides
func.func @conv2d_nhwc_ohwi_nhwc_no_strides(%arg0: tensor<1x6x6x207xf32>, %arg1: tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>
  } : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32>
  func.return %0 : tensor<1x4x4x16xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>, tensor<16xf32>) -> tensor<1x4x4x16xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_rhs_dilated
func.func @conv2d_nhwc_ohwi_nhwc_rhs_dilated(%input: tensor<1x256x256x3xf32>, %filter: tensor<2x5x5x3xf32>) -> tensor<1x248x248x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 2, 2>,
    lhs_dilation = array<i64: 1, 1>
  } : (tensor<1x256x256x3xf32>, tensor<2x5x5x3xf32>) -> tensor<1x248x248x2xf32>
  func.return %0 : tensor<1x248x248x2xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x3xf32>, tensor<2x5x5x3xf32>, tensor<2xf32>) -> tensor<1x248x248x2xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_with_strides
func.func @conv2d_nhwc_ohwi_nhwc_with_strides(%input: tensor<1x115x115x96xf32>, %filter: tensor<96x7x7x96xf32>) -> tensor<1x55x55x96xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 2, 2>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>
  } : (tensor<1x115x115x96xf32>, tensor<96x7x7x96xf32>) -> tensor<1x55x55x96xf32>
  func.return %0 : tensor<1x55x55x96xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x115x115x96xf32>, tensor<96x7x7x96xf32>, tensor<96xf32>) -> tensor<1x55x55x96xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_grouped_features
func.func @conv2d_nhwc_ohwi_nhwc_grouped_features(%input: tensor<1x256x256x6xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 2 : i64,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>
  } : (tensor<1x256x256x6xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x6xf32>, tensor<2x1x1x3xf32>, tensor<2xf32>) -> tensor<1x256x256x2xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_dynamic_batch
func.func @conv2d_nhwc_ohwi_nhwc_dynamic_batch(%input: tensor<?x256x256x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32> {
  %0 = stablehlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<?x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32>
  func.return %0 : tensor<?x256x256x2xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<?x256x256x3xf32>, tensor<2x1x1x3xf32>, tensor<2xf32>) -> tensor<?x256x256x2xf32>

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ihwo_nhwc
func.func @depthwise_conv2d_nhwc_ihwo_nhwc(%arg0: tensor<1x10x10x207xf32>, %arg1: tensor<1x3x3x207xf32>) -> tensor<1x8x8x207xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>) -> tensor<1x8x8x207xf32>
  func.return %0 : tensor<1x8x8x207xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>, tensor<207xf32>) -> tensor<1x8x8x207xf32>
// CHECK: return %0

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ihwo_nhwc_dynamic
func.func @depthwise_conv2d_nhwc_ihwo_nhwc_dynamic(%arg0: tensor<?x10x10x207xf32>, %arg1: tensor<1x3x3x207xf32>) -> tensor<?x8x8x207xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<?x10x10x207xf32>, tensor<1x3x3x207xf32>) -> tensor<?x8x8x207xf32>
  func.return %0 : tensor<?x8x8x207xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<?x10x10x207xf32>, tensor<1x3x3x207xf32>, tensor<207xf32>) -> tensor<?x8x8x207xf32>
// CHECK: return %0

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ihwo_nhwc_strided
func.func @depthwise_conv2d_nhwc_ihwo_nhwc_strided(%arg0: tensor<1x10x10x207xf32>, %arg1: tensor<1x3x3x207xf32>) -> tensor<1x4x4x207xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 2, 2>
  } : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>) -> tensor<1x4x4x207xf32>
  func.return %0 : tensor<1x4x4x207xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>, tensor<207xf32>) -> tensor<1x4x4x207xf32>
// CHECK: return %0

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ihwo_nhwc_dilated
func.func @depthwise_conv2d_nhwc_ihwo_nhwc_dilated(%arg0: tensor<1x10x10x207xf32>, %arg1: tensor<1x3x3x207xf32>) -> tensor<1x6x6x207xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 2, 2>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>) -> tensor<1x6x6x207xf32>
  func.return %0 : tensor<1x6x6x207xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 1 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>, tensor<207xf32>) -> tensor<1x6x6x207xf32>
// CHECK: return %0

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ihwo_nhwc_non_trivial_depth_multiplier
func.func @depthwise_conv2d_nhwc_ihwo_nhwc_non_trivial_depth_multiplier(%arg0: tensor<1x10x10x207xf32>, %arg1: tensor<1x3x3x3519xf32>) -> tensor<1x6x6x3519xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 2, 2>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x10x10x207xf32>, tensor<1x3x3x3519xf32>) -> tensor<1x6x6x3519xf32>
  func.return %0 : tensor<1x6x6x3519xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 17 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x10x10x207xf32>, tensor<1x3x3x3519xf32>, tensor<3519xf32>) -> tensor<1x6x6x3519xf32>
// CHECK: return %0

// -----

// CHECK-LABEL: conv2d_resize_perferred_nhwc_hwoi_nhwc
func.func @conv2d_resize_perferred_nhwc_hwoi_nhwc(%arg0: tensor<1x56x1248x16xf32>, %arg1: tensor<16x3x1x1xf32>) -> tensor<1x111x1248x16xf32> {
	%0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 16 : i64,
    lhs_dilation = array<i64: 2, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x56x1248x16xf32>, tensor<16x3x1x1xf32>) -> tensor<1x111x1248x16xf32>
  func.return %0 : tensor<1x111x1248x16xf32>
  // CHECK  %0 = "tfl.pseudo_const"() <{value = dense<[111, 1248]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK  %1 = "tfl.resize_bilinear"(%arg0, %0) <{align_corners = false, half_pixel_centers = false}> : (tensor<1x56x1248x16xf32>, tensor<2xi32>) -> tensor<1x111x1248x16xf32>
  // CHECK  return %1 : tensor<1x111x1248x16xf32>
}

// -----

// CHECK-LABEL: conv2d_to_resize_nhwc_hwoi_nhwc
func.func @conv2d_to_resize_nhwc_hwoi_nhwc(%arg0: tensor<1x56x624x16xf32>, %arg1: tensor<16x1x257x1xf32>) -> tensor<1x56x904x16xf32> {
	%0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 16 : i64,
    lhs_dilation = array<i64: 1, 129>,
    padding = dense<[[0, 0], [128, 128]]> : tensor<2x2xi64>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 89>
  } : (tensor<1x56x624x16xf32>, tensor<16x1x257x1xf32>) -> tensor<1x56x904x16xf32>
  func.return %0 : tensor<1x56x904x16xf32>
  // CHECK  %0 = "tfl.pseudo_const"() <{value = dense<[56, 904]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK  %1 = "tfl.resize_bilinear"(%arg0, %0) <{align_corners = true, half_pixel_centers = false}> : (tensor<1x56x624x16xf32>, tensor<2xi32>) -> tensor<1x56x904x16xf32>
  // CHECK  return %1 : tensor<1x56x904x16xf32>
}

// -----

//
// 3D
//=---

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc
func.func @conv3d_ndhwc_dhwio_ndhwc(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1, 1>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = array<i64: 1, 1, 1>,
    lhs_dilation = array<i64: 1, 1, 1>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_no_strides
func.func @conv3d_ndhwc_dhwio_ndhwc_no_strides(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = array<i64: 1, 1, 1>,
    lhs_dilation = array<i64: 1, 1, 1>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_no_pad
func.func @conv3d_ndhwc_dhwio_ndhwc_no_pad(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1, 1>,
    rhs_dilation = array<i64: 1, 1, 1>,
    lhs_dilation = array<i64: 1, 1, 1>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_no_lhs_dilation
func.func @conv3d_ndhwc_dhwio_ndhwc_no_lhs_dilation(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1, 1>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = array<i64: 1, 1, 1>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_no_rhs_dilation
func.func @conv3d_ndhwc_dhwio_ndhwc_no_rhs_dilation(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1, 1>,
    padding = dense<0> : tensor<3x2xi64>,
    lhs_dilation = array<i64: 1, 1, 1>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_rhs_dilated
func.func @conv3d_ndhwc_dhwio_ndhwc_rhs_dilated(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<5x5x5x3x2xf32>) -> tensor<1x248x248x248x2xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 1, 1, 1>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = array<i64: 2, 2, 2>,
    lhs_dilation = array<i64: 1, 1, 1>
  } : (tensor<1x256x256x256x3xf32>, tensor<5x5x5x3x2xf32>) -> tensor<1x248x248x248x2xf32>
  func.return %0 : tensor<1x248x248x248x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 2 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<5x5x5x3x2xf32>, tensor<2xf32>) -> tensor<1x248x248x248x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_strided
func.func @conv3d_ndhwc_dhwio_ndhwc_strided(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x52x52x52x2xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = array<i64: 5, 5, 5>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = array<i64: 1, 1, 1>,
    lhs_dilation = array<i64: 1, 1, 1>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x52x52x52x2xf32>
  func.return %0 : tensor<1x52x52x52x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 5 : i32, stride_h = 5 : i32, stride_w = 5 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x52x52x52x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_grouped_features
func.func @conv3d_ndhwc_dhwio_ndhwc_grouped_features(%input: tensor<1x256x256x256x6xf32>, %filter: tensor<1x1x1x2x6xf32>) -> tensor<1x256x256x256x6xf32> {
  %0 = "stablehlo.convolution"(%input, %filter) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 3 : i64,
    window_strides = array<i64: 1, 1, 1>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = array<i64: 1, 1, 1>,
    lhs_dilation = array<i64: 1, 1, 1>
  } : (tensor<1x256x256x256x6xf32>, tensor<1x1x1x2x6xf32>) -> tensor<1x256x256x256x6xf32>
  func.return %0 : tensor<1x256x256x256x6xf32>
}


// Unlike TF, tfl.conv_3d requires kernel_dims[3] == input_dims[4]

// CHECK-NOT: tfl
// CHECK: stablehlo.convolution

// -----

//===----------------------------------------------------------------------===//
// stablehlo.pad
//===----------------------------------------------------------------------===//

func.func @pad_cst_zero_vals(%arg0: tensor<8x128xf32>) -> tensor<11x131xf32> {
  %pad_val = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = "stablehlo.pad"(%arg0, %pad_val) {
    edge_padding_low = array<i64: 1, 0>,
    edge_padding_high = array<i64: 2, 3>,
    interior_padding = array<i64: 0, 0>
  } : (tensor<8x128xf32>, tensor<f32>) -> tensor<11x131xf32>
  func.return %0 : tensor<11x131xf32>
}

// CHECK: %[[PADDINGS:.*]] = arith.constant{{.*}}[1, 2], [0, 3]
// CHECK: "tfl.pad"(%arg0, %[[PADDINGS]]) : (tensor<8x128xf32>, tensor<2x2xi64>) -> tensor<11x131xf32>

// -----

func.func @pad(%arg0: tensor<8x128xf32>, %arg1: tensor<f32>) -> tensor<11x131xf32> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 1, 0>,
    edge_padding_high = array<i64: 2, 3>,
    interior_padding = array<i64: 0, 0>
  } : (tensor<8x128xf32>, tensor<f32>) -> tensor<11x131xf32>
  func.return %0 : tensor<11x131xf32>
}

// CHECK:      %[[PADDINGS:.*]] = arith.constant
// CHECK-SAME: [1, 2], [0, 3]
// CHECK-SAME: tensor<2x2xi64>
// CHECK:      "tfl.padv2"(%arg0, %[[PADDINGS]], %arg1) : (tensor<8x128xf32>, tensor<2x2xi64>, tensor<f32>) -> tensor<11x131xf32>

//===------------------------------------------------------------------------===
// stablehlo.reduce { stablehlo.multiply }
//===------------------------------------------------------------------------===

// CHECK-LABEL: int_reduce_to_prod(
func.func @int_reduce_to_prod(%arg0: tensor<1x256xi32>) -> tensor<1xi32> {
  %0 = stablehlo.constant dense<1> : tensor<i32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %2 = stablehlo.multiply %arg1, %arg2 : tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x256xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.reduce_prod"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xi32>, tensor<1xi32>) -> tensor<1xi32>

// -----

// CHECK-LABEL: reduce_to_prod
func.func @reduce_to_prod(%arg0: tensor<1x256xf32>) -> tensor<1xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.multiply %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.reduce_prod"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: reduce_to_prod_non_constant_init
func.func @reduce_to_prod_non_constant_init(%arg0: tensor<1x256xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  %1 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = stablehlo.multiply %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %0 = "tfl.reduce_prod"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>
// CHECK: %1 = tfl.mul(%0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>

// -----

//===------------------------------------------------------------------------===
// stablehlo.reduce { stablehlo.add }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_sum
func.func @reduce_to_sum(%arg0: tensor<1x256xf32>) -> tensor<1xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.sum"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: reduce_to_sum_non_constant_init
func.func @reduce_to_sum_non_constant_init(%arg0: tensor<1x256xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  %1 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %0 = "tfl.sum"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>
// CHECK: %1 = tfl.add(%0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: int_reduce_to_sum
func.func @int_reduce_to_sum(%arg0: tensor<1x256xi32>) -> tensor<1xi32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %2 = stablehlo.add %arg1, %arg2 : tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x256xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.sum"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xi32>, tensor<1xi32>) -> tensor<1xi32>

// -----

//===------------------------------------------------------------------------===
// stablehlo.reduce { stablehlo.max }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_max
func.func @reduce_to_max(%arg0: tensor<1x256xf32>) -> tensor<1xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.reduce_max"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: reduce_to_max_int
func.func @reduce_to_max_int(%arg0: tensor<1x4xi32>) -> tensor<1xi32> {
  // -2147483648 is MIN for INT32
  %0 = stablehlo.constant dense<-2147483648> : tensor<i32>
  %1 = stablehlo.reduce(%arg0 init: %0) across dimensions = [1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
   reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
    %892 = stablehlo.maximum %arg2, %arg3 : tensor<i32>
    "stablehlo.return"(%892) : (tensor<i32>) -> ()
  }
  func.return %1 : tensor<1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.reduce_max"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x4xi32>, tensor<1xi32>) -> tensor<1xi32>

// -----

//===------------------------------------------------------------------------===
// stablehlo.reduce { stablehlo.min }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_min
func.func @reduce_to_min(%arg0: tensor<1x256xf32>) -> tensor<1xf32> {
  // "0x7F800000" represents INF for f32.
  %0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.minimum %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.reduce_min"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: reduce_to_min_int
func.func @reduce_to_min_int(%arg0: tensor<1x4xi32>) -> tensor<1xi32> {
  // 2147483647 is MAX for INT32
  %0 = stablehlo.constant dense<2147483647> : tensor<i32>
  %1 = stablehlo.reduce(%arg0 init: %0) across dimensions = [1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
   reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
    %892 = stablehlo.minimum %arg2, %arg3 : tensor<i32>
    "stablehlo.return"(%892) : (tensor<i32>) -> ()
  }
  func.return %1 : tensor<1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.reduce_min"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x4xi32>, tensor<1xi32>) -> tensor<1xi32>

// -----

//===------------------------------------------------------------------------===
// stablehlo.reduce { stablehlo.and }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_all
func.func @reduce_to_all(%arg0: tensor<1x2x3x4x5xi1>, %arg1: tensor<2xi64>) -> tensor<2x4x5xi1> {
  %0 = stablehlo.constant dense<true> : tensor<i1>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
        %2 = stablehlo.and %arg2, %arg3 : tensor<i1>
        "stablehlo.return"(%2) : (tensor<i1>) -> ()
    }) {dimensions = array<i64: 0, 2>} : (tensor<1x2x3x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>
  func.return %1: tensor<2x4x5xi1>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 2]> : tensor<2xi32>
// CHECK: "tfl.reduce_all"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x2x3x4x5xi1>, tensor<2xi32>) -> tensor<2x4x5xi1>

// -----

// CHECK-LABEL: reduce_to_all_non_constant_init
func.func @reduce_to_all_non_constant_init(%arg0: tensor<i1>, %arg1: tensor<1x2x3x4x5xi1>, %arg2: tensor<2xi64>) -> tensor<2x4x5xi1> {
  %0 = "stablehlo.reduce"(%arg1, %arg0) ({
    ^bb0(%arg3: tensor<i1>, %arg4: tensor<i1>):
        %1 = stablehlo.and %arg3, %arg4 : tensor<i1>
        "stablehlo.return"(%1) : (tensor<i1>) -> ()
    }) {dimensions = array<i64: 0, 2>} : (tensor<1x2x3x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>
  func.return %0: tensor<2x4x5xi1>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 2]> : tensor<2xi32>
// CHECK: %[[REDUCE:.*]] = "tfl.reduce_all"(%arg1, %[[CST]]) <{keep_dims = false}> : (tensor<1x2x3x4x5xi1>, tensor<2xi32>) -> tensor<2x4x5xi1>
// CHECK: tfl.logical_and(%[[REDUCE]], %arg0) : (tensor<2x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>

// -----

//===------------------------------------------------------------------------===
// stablehlo.reduce { stablehlo.or }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_any
func.func @reduce_to_any(%arg0: tensor<1x2x3x4x5xi1>, %arg1: tensor<2xi64>) -> tensor<2x4x5xi1> {
  %0 = stablehlo.constant dense<false> : tensor<i1>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
        %2 = stablehlo.or %arg2, %arg3 : tensor<i1>
        "stablehlo.return"(%2) : (tensor<i1>) -> ()
    }) {dimensions = array<i64: 0, 2>} : (tensor<1x2x3x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>
  func.return %1: tensor<2x4x5xi1>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 2]> : tensor<2xi32>
// CHECK: "tfl.reduce_any"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x2x3x4x5xi1>, tensor<2xi32>) -> tensor<2x4x5xi1>

// -----

// CHECK-LABEL: reduce_to_any_non_constant_init
func.func @reduce_to_any_non_constant_init(%arg0: tensor<i1>, %arg1: tensor<1x2x3x4x5xi1>, %arg2: tensor<2xi64>) -> tensor<2x4x5xi1> {
  %0 = "stablehlo.reduce"(%arg1, %arg0) ({
    ^bb0(%arg3: tensor<i1>, %arg4: tensor<i1>):
        %1 = stablehlo.or %arg3, %arg4 : tensor<i1>
        "stablehlo.return"(%1) : (tensor<i1>) -> ()
    }) {dimensions = array<i64: 0, 2>} : (tensor<1x2x3x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>
  func.return %0: tensor<2x4x5xi1>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 2]> : tensor<2xi32>
// CHECK: %[[REDUCE:.*]] = "tfl.reduce_any"(%arg1, %[[CST]]) <{keep_dims = false}> : (tensor<1x2x3x4x5xi1>, tensor<2xi32>) -> tensor<2x4x5xi1>
// CHECK: tfl.logical_or(%[[REDUCE]], %arg0) : (tensor<2x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>

// -----

//===------------------------------------------------------------------------===
// stablehlo.gather
//===------------------------------------------------------------------------===

// CHECK-LABEL: gather_to_slice
func.func @gather_to_slice(%arg0: tensor<3x2944xi32>, %arg1: tensor<3x2xi32>) -> tensor<3x1504xi32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1,
    >,
    indices_are_sorted = true,
    slice_sizes = array<i64: 1, 1504>
  } : (tensor<3x2944xi32>, tensor<3x2xi32>) -> tensor<3x1504xi32>
  func.return %0 : tensor<3x1504xi32>
}

// CHECK-DAG: %[[CST:.*]] = arith.constant dense<[2, 1440]> : tensor<2xi32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK:     %[[VAL_0:.*]] = "tfl.maximum"(%arg1, %[[CST_0]]) : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<3x2xi32>
// CHECK:     %[[VAL_1:.*]] = "tfl.minimum"(%[[VAL_0]], %[[CST]]) : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<3x2xi32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<[1, 1504]> : tensor<2xi32>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<[1, 2]> : tensor<2xi32>
// CHECK:     %[[VAL_2:.*]] = "tfl.slice"(%[[VAL_1]], %[[CST_2]], %[[CST_3]]) : (tensor<3x2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK:     %[[VAL_3:.*]] = "tfl.squeeze"(%[[VAL_2]]) <{squeeze_dims = [0]}> : (tensor<1x2xi32>) -> tensor<2xi32>
// CHECK:     %[[VAL_4:.*]] = "tfl.slice"(%arg0, %[[VAL_3]], %[[CST_1]]) : (tensor<3x2944xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x1504xi32>
// CHECK-DAG: %[[CST_4:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK-DAG: %[[CST_5:.*]] = arith.constant dense<[1, 2]> : tensor<2xi32>
// CHECK:     %[[VAL_5:.*]] = "tfl.slice"(%[[VAL_1]], %[[CST_4]], %[[CST_5]]) : (tensor<3x2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK:     %[[VAL_6:.*]] = "tfl.squeeze"(%[[VAL_5]]) <{squeeze_dims = [0]}> : (tensor<1x2xi32>) -> tensor<2xi32>
// CHECK:     %[[VAL_7:.*]] = "tfl.slice"(%arg0, %[[VAL_6]], %[[CST_1]]) : (tensor<3x2944xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x1504xi32>
// CHECK-DAG: %[[CST_6:.*]] = arith.constant dense<[2, 0]> : tensor<2xi32>
// CHECK-DAG: %[[CST_7:.*]] = arith.constant dense<[1, 2]> : tensor<2xi32>
// CHECK:     %[[VAL_8:.*]] = "tfl.slice"(%[[VAL_1]], %[[CST_6]], %[[CST_7]]) : (tensor<3x2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK:     %[[VAL_9:.*]] = "tfl.squeeze"(%[[VAL_8]]) <{squeeze_dims = [0]}> : (tensor<1x2xi32>) -> tensor<2xi32>
// CHECK:     %[[VAL_10:.*]] = "tfl.slice"(%arg0, %[[VAL_9]], %[[CST_1]]) : (tensor<3x2944xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x1504xi32>
// CHECK:     %[[VAL_11:.*]] = "tfl.concatenation"(%[[VAL_4]], %[[VAL_7]], %[[VAL_10]]) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x1504xi32>, tensor<1x1504xi32>, tensor<1x1504xi32>) -> tensor<3x1504xi32>

// -----

// CHECK-LABEL: gather
func.func @gather(%arg0: tensor<147456xf16>, %arg1: tensor<192x256x1xi32>) -> tensor<192x256xf16> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
			start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1>
  } : (tensor<147456xf16>, tensor<192x256x1xi32>) -> tensor<192x256xf16>
  func.return %0 : tensor<192x256xf16>
}

// -----

// CHECK-LABEL: gather_with_ui32indices
func.func @gather_with_ui32indices(%arg0: tensor<147456xf16>, %arg1: tensor<192x256x1xui32>) -> tensor<192x256xf16> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
			start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1>
  } : (tensor<147456xf16>, tensor<192x256x1xui32>) -> tensor<192x256xf16>
  func.return %0 : tensor<192x256xf16>
}

// CHECK: %{{.*}} = "tfl.cast"(%{{.*}}) : (tensor<192x256x1xui32>) -> tensor<192x256x1xi64>

// -----

// CHECK-LABEL: gather_nd
func.func @gather_nd(%arg0: tensor<98x128xf32>, %arg1: tensor<4x64xi32>) -> tensor<4x64x128xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
			offset_dims = [2],
			start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 128>
  } : (tensor<98x128xf32>, tensor<4x64xi32>) -> tensor<4x64x128xf32>
  func.return %0 : tensor<4x64x128xf32>
}

// CHECK: %[[VAL_1:.*]] = "tfl.gather_nd"

// -----

// CHECK-LABEL: gather_transpose
func.func @gather_transpose(%arg0: tensor<128x256xf32>, %arg1: tensor<4x1xi32>) -> tensor<4x128xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [1],
      index_vector_dim = 1,
			offset_dims = [1],
			start_index_map = [1],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 128, 1>
  } : (tensor<128x256xf32>, tensor<4x1xi32>) -> tensor<4x128xf32>
  func.return %0 : tensor<4x128xf32>
}

// CHECK: %[[VAL_3:.*]] = "tfl.gather_nd"

// -----

// CHECK-LABEL: gather_offset
func.func @gather_offset(%arg0: tensor<1x20xi32>, %arg1: tensor<1x1xi32>) -> tensor<1x1xi32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [1],
      index_vector_dim = 1,
			offset_dims = [0],
			start_index_map = [1],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 1>
  } : (tensor<1x20xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
  func.return %0 : tensor<1x1xi32>
}

// CHECK: %[[VAL_3:.*]] = "tfl.gather_nd"
// CHECK: %[[VAL_6:.*]] = "tfl.transpose"

// -----


// CHECK-LABEL: gather_batching_dims
func.func @gather_batching_dims(%arg0: tensor<2x3x128xf32>, %arg1: tensor<3x128x2x1xi32>) -> tensor<3x128x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      index_vector_dim = 3,
      start_index_map = [2],
      operand_batching_dims = [0, 1],
      start_indices_batching_dims = [2, 0],
      collapsed_slice_dims = [2],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 1, 1>
  } : (tensor<2x3x128xf32>, tensor<3x128x2x1xi32>) -> tensor<3x128x2xf32>
  func.return %0 : tensor<3x128x2xf32>
}

// CHECK-DAG: %[[CST:.*]] = arith.constant dense<[6, 128]> : tensor<2xi64>
// CHECK:     %[[VAL_0:.*]] = "tfl.cast"(%[[CST]]) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK:     %[[VAL_1:.*]] = "tfl.reshape"(%{{.*}}, %[[VAL_0]]) : (tensor<2x3x128xf32>, tensor<2xi32>) -> tensor<6x128xf32>
// CHECK-DAG: %[[VAL_2:.*]] = arith.constant dense<[2, 0, 1, 3]> : tensor<4xi32>
// CHECK:     %[[VAL_4:.*]] = "tfl.transpose"(%arg1, %[[VAL_2]]) : (tensor<3x128x2x1xi32>, tensor<4xi32>) -> tensor<2x3x128x1xi32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<[6, 128, 1]> : tensor<3xi64>
// CHECK:     %[[VAL_5:.*]] = "tfl.cast"(%[[CST_0]]) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK:     %[[VAL_6:.*]] = "tfl.reshape"(%[[VAL_4]], %[[VAL_5]]) : (tensor<2x3x128x1xi32>, tensor<3xi32>) -> tensor<6x128x1xi32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<6> : tensor<i32>
// CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:     %[[VAL_7:.*]] = "tfl.range"(%[[CST_1]], %[[CST_2]], %[[CST_3]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6xi32>
// CHECK-DAG: %[[CST_4:.*]] = arith.constant dense<[6, 1, 1]> : tensor<3xi64>
// CHECK:     %[[VAL_8:.*]] = "tfl.cast"(%[[CST_4]]) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK:     %[[VAL_9:.*]] = "tfl.reshape"(%[[VAL_7]], %[[VAL_8]]) : (tensor<6xi32>, tensor<3xi32>) -> tensor<6x1x1xi32>
// CHECK-DAG: %[[CST_5:.*]] = arith.constant dense<[6, 128, 1]> : tensor<3xi64>
// CHECK:     %[[VAL_10:.*]] = "tfl.broadcast_to"(%[[VAL_9]], %[[CST_5]]) : (tensor<6x1x1xi32>, tensor<3xi64>) -> tensor<6x128x1xi32>
// CHECK:     %[[VAL_11:.*]] = "tfl.concatenation"(%[[VAL_10]], %[[VAL_6]]) <{axis = 2 : i32, fused_activation_function = "NONE"}> : (tensor<6x128x1xi32>, tensor<6x128x1xi32>) -> tensor<6x128x2xi32>
// CHECK:     %[[VAL_12:.*]] = "tfl.gather_nd"(%[[VAL_1]], %[[VAL_11]]) : (tensor<6x128xf32>, tensor<6x128x2xi32>) -> tensor<6x128xf32>
// CHECK-DAG: %[[CST_6:.*]] = arith.constant dense<[2, 3, 128]> : tensor<3xi64>
// CHECK:     %[[VAL_13:.*]] = "tfl.cast"(%[[CST_6]]) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK:     %[[VAL_14:.*]] = "tfl.reshape"(%[[VAL_12]], %[[VAL_13]]) : (tensor<6x128xf32>, tensor<3xi32>) -> tensor<2x3x128xf32>
// CHECK-DAG: %[[VAL_16:.*]] = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
// CHECK:     %[[VAL_17:.*]] = "tfl.transpose"(%[[VAL_14]], %[[VAL_16]]) : (tensor<2x3x128xf32>, tensor<3xi32>) -> tensor<3x128x2xf32>

// -----

// CHECK-LABEL: convert_gather_non_collapsed_index_dim
func.func @convert_gather_non_collapsed_index_dim(%arg0: tensor<10x5xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x1x5xi32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 5>
  } : (tensor<10x5xi32>, tensor<2x1xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// CHECK:     %[[VAL_0:.*]] = "tfl.gather_nd"
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<[2, 1, 5]> : tensor<3xi64>
// CHECK:     %[[VAL_1:.*]] = "tfl.cast"(%[[CST]]) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK:     %[[VAL_2:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_1]]) : (tensor<2x5xi32>, tensor<3xi32>) -> tensor<2x1x5xi32>

// -----

// CHECK-LABEL: convert_gather_indexed_dimension_slice
func.func @convert_gather_indexed_dimension_slice(%arg0: tensor<4x5x6xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x1x5x6xi32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      index_vector_dim = 1,
      offset_dims = [1, 2, 3],
      start_index_map = [0, 2],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 5, 6>
  } : (tensor<4x5x6xi32>, tensor<2x2xi32>) -> tensor<2x1x5x6xi32>
  func.return %0 : tensor<2x1x5x6xi32>
}

// CHECK:     %[[CST_TRANS:.*]] = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
// CHECK:     %[[VAL_2:.*]] = "tfl.transpose"(%arg0, %[[CST_TRANS]]) : (tensor<4x5x6xi32>, tensor<3xi32>) -> tensor<4x6x5xi32>
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<[2, 1, 2]> : tensor<3xi64>
// CHECK:     %[[VAL_3:.*]] = "tfl.cast"(%[[CST]]) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK:     %[[VAL_4:.*]] = "tfl.reshape"(%{{.*}}, %[[VAL_3]]) : (tensor<2x2xi32>, tensor<3xi32>) -> tensor<2x1x2xi32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<6> : tensor<i32>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:     %[[VAL_5:.*]] = "tfl.range"(%[[CST_0]], %[[CST_1]], %[[CST_2]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6xi32>
// CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<[1, 6, 1]> : tensor<3xi64>
// CHECK:     %[[VAL_6:.*]] = "tfl.cast"(%[[CST_3]]) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK:     %[[VAL_7:.*]] = "tfl.reshape"(%[[VAL_5]], %[[VAL_6]]) : (tensor<6xi32>, tensor<3xi32>) -> tensor<1x6x1xi32>
// CHECK-DAG: %[[CST_4:.*]] = arith.constant dense<[1, 6, 1]> : tensor<3xi64>
// CHECK:     %[[VAL_8:.*]] = "tfl.broadcast_to"(%[[VAL_7]], %[[CST_4]]) : (tensor<1x6x1xi32>, tensor<3xi64>) -> tensor<1x6x1xi32>
// CHECK-DAG: %[[CST_5:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_6:.*]] = arith.constant
// CHECK-SAME{LITERAL}: dense<[[0, 0], [0, 0], [1, 0]]> : tensor<3x2xi64>
// CHECK:     %[[VAL_9:.*]] = "tfl.pad"(%[[VAL_8]], %[[CST_6]]) : (tensor<1x6x1xi32>, tensor<3x2xi64>) -> tensor<1x6x2xi32>
// CHECK:     %[[VAL_10:.*]] = tfl.add(%[[VAL_4]], %[[VAL_9]]) <{fused_activation_function = "NONE"}> : (tensor<2x1x2xi32>, tensor<1x6x2xi32>) -> tensor<2x6x2xi32>
// CHECK:     %[[VAL_11:.*]] = "tfl.gather_nd"(%{{.*}}, %[[VAL_10]]) : (tensor<4x6x5xi32>, tensor<2x6x2xi32>) -> tensor<2x6x5xi32>
// CHECK-DAG: %[[CST_7:.*]] = arith.constant dense<[2, 1, 6, 5]> : tensor<4xi64>
// CHECK:     %[[VAL_12:.*]] = "tfl.cast"(%[[CST_7]]) : (tensor<4xi64>) -> tensor<4xi32>
// CHECK:     %[[VAL_13:.*]] = "tfl.reshape"(%[[VAL_11]], %[[VAL_12]]) : (tensor<2x6x5xi32>, tensor<4xi32>) -> tensor<2x1x6x5xi32>
// CHECK-DAG: %[[CST_OUT_TRANS:.*]] = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
// CHECK:     %[[VAL_16:.*]] = "tfl.transpose"(%[[VAL_13]], %[[CST_OUT_TRANS]]) : (tensor<2x1x6x5xi32>, tensor<4xi32>) -> tensor<2x1x5x6xi32>

// -----

// CHECK-LABEL: gather_to_slice_batch_size_1
func.func @gather_to_slice_batch_size_1(%arg0: tensor<1x2944xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x1504xi32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1,
    >,
    indices_are_sorted = true,
    slice_sizes = array<i64: 1, 1504>
  } : (tensor<1x2944xi32>, tensor<1x2xi32>) -> tensor<1x1504xi32>
  func.return %0 : tensor<1x1504xi32>
}

// CHECK-DAG: %[[CST:.*]] = arith.constant dense<[0, 1440]> : tensor<2xi32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK:     %[[VAL_0:.*]] = "tfl.maximum"(%arg1, %[[CST_0]]) : (tensor<1x2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK:     %[[VAL_1:.*]] = "tfl.minimum"(%[[VAL_0]], %[[CST]]) : (tensor<1x2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<[1, 1504]> : tensor<2xi32>
// CHECK:     %[[VAL_2:.*]] = "tfl.squeeze"(%[[VAL_1]]) <{squeeze_dims = [0]}> : (tensor<1x2xi32>) -> tensor<2xi32>
// CHECK:     %[[VAL_3:.*]] = "tfl.slice"(%arg0, %[[VAL_2]], %[[CST_1]]) : (tensor<1x2944xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x1504xi32>

// -----

// CHECK-LABEL: gather_slice_dynamic_indices
func.func @gather_slice_dynamic_indices(%arg0: tensor<256000x1024xi8>, %arg1: tensor<?x?x1xi32>) -> tensor<?x?x1024xi8> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1024>
  } : (tensor<256000x1024xi8>, tensor<?x?x1xi32>) -> tensor<?x?x1024xi8>
  func.return %0 : tensor<?x?x1024xi8>
}

// CHECK: %[[GATHER:.*]] = "tfl.gather_nd"

// -----

// CHECK-LABEL: gather_scalar_dynamic_indices
func.func @gather_scalar_dynamic_indices(%arg0: tensor<256000xf32>, %arg1: tensor<?x?x1xi32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1>
  } : (tensor<256000xf32>, tensor<?x?x1xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK: %[[GATHER:.*]] = "tfl.gather_nd"

// -----

//===------------------------------------------------------------------------===
// stablehlo.reduce_window -> avg pool
//===------------------------------------------------------------------------===

// CHECK-LABEL: avgpool_same_channel_first
func.func @avgpool_same_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x8x8xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<4x16x16x3xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "stablehlo.transpose"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
  %3 = "stablehlo.reduce_window"(%2, %1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %8 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %8 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x8x8x3xf32>
  %4 = "stablehlo.transpose"(%3) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<4x8x8x3xf32>) -> tensor<4x3x8x8xf32>
  %5 = "stablehlo.reduce_window"(%0, %1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %8 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %8 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x8x8x3xf32>
  %6 = "stablehlo.transpose"(%5) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<4x8x8x3xf32>) -> tensor<4x3x8x8xf32>
  %7 = stablehlo.divide %4, %6 : tensor<4x3x8x8xf32>
  return %7 : tensor<4x3x8x8xf32>
}

// CHECK:      %[[TPOSED_ARG0:.*]] = "tfl.transpose"(%arg0
// CHECK-SAME: (tensor<4x3x16x16xf32>, tensor<4xi32>) -> tensor<4x16x16x3xf32>
// CHECK:      %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%[[TPOSED_ARG0]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x3xf32>) -> tensor<4x8x8x3xf32>
// CHECK:      %[[TPOSED_OUT:.*]] = "tfl.transpose"(%[[POOL_OUT]]
// CHECK-SAME: (tensor<4x8x8x3xf32>, tensor<4xi32>) -> tensor<4x3x8x8xf32>
// CHECK:      return %[[TPOSED_OUT]]

// -----

// CHECK-LABEL: avgpool_valid_channel_first
func.func @avgpool_valid_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x7x7xf32> {
  %0 = stablehlo.constant dense<9.000000e+00> : tensor<4x3x7x7xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "stablehlo.transpose"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
  %3 = "stablehlo.reduce_window"(%2, %1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %6 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x7x7x3xf32>
  %4 = "stablehlo.transpose"(%3) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<4x7x7x3xf32>) -> tensor<4x3x7x7xf32>
  %5 = stablehlo.divide %4, %0 : tensor<4x3x7x7xf32>
  return %5 : tensor<4x3x7x7xf32>
}

// CHECK:      %[[TPOSED_ARG0:.*]] = "tfl.transpose"(%arg0
// CHECK-SAME: (tensor<4x3x16x16xf32>, tensor<4xi32>) -> tensor<4x16x16x3xf32>
// CHECK:      %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%[[TPOSED_ARG0]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x3xf32>) -> tensor<4x7x7x3xf32>
// CHECK:      %[[TPOSED_OUT:.*]] = "tfl.transpose"(%[[POOL_OUT]]
// CHECK-SAME: (tensor<4x7x7x3xf32>, tensor<4xi32>) -> tensor<4x3x7x7xf32>
// CHECK:      return %[[TPOSED_OUT]]

// -----

func.func @avgpool_valid(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.constant dense<9.0> : tensor<4x7x7x8xf32>
  %2 = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %5 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%5) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<0> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %3 = stablehlo.divide %2, %1 : tensor<4x7x7x8xf32>
  func.return %3 : tensor<4x7x7x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_valid_broadcasted_divisor
func.func @avgpool_valid_broadcasted_divisor(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.constant dense<9.0> : tensor<f32>
  %2 = "stablehlo.broadcast_in_dim"(%1) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<4x7x7x8xf32>
  %3 = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %5 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%5) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<0> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %4 = stablehlo.divide %3, %2 : tensor<4x7x7x8xf32>
  func.return %4 : tensor<4x7x7x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_valid_rw
func.func @avgpool_valid_rw(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  %0 = stablehlo.constant dense<1.0> : tensor<4x16x16x8xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<f32>
  %2 = "stablehlo.reduce_window"(%arg0, %1) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %3 = "stablehlo.reduce_window"(%0, %1) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %4 = stablehlo.divide %2, %3 : tensor<4x7x7x8xf32>
  func.return %4 : tensor<4x7x7x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_valid_rw_broadcasted_const_lhs
func.func @avgpool_valid_rw_broadcasted_const_lhs(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  %0 = stablehlo.constant dense<1.0> : tensor<f32>
  %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<4x16x16x8xf32>
  %2 = stablehlo.constant dense<0.0> : tensor<f32>
  %3 = "stablehlo.reduce_window"(%arg0, %2) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %4 = "stablehlo.reduce_window"(%1, %2) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %5 = stablehlo.divide %3, %4 : tensor<4x7x7x8xf32>
  func.return %5 : tensor<4x7x7x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_same
func.func @avgpool_same(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32> {
  %0 = stablehlo.constant dense<1.0> : tensor<4x16x16x8xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<f32>
  %2 = "stablehlo.reduce_window"(%arg0, %1) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x8x8x8xf32>
  %3 = "stablehlo.reduce_window"(%0, %1) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x8x8x8xf32>
  %4 = stablehlo.divide %2, %3 : tensor<4x8x8x8xf32>
  func.return %4 : tensor<4x8x8x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_reshape_broadcast
func.func @avgpool_reshape_broadcast(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<1x16x16x1xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "stablehlo.reduce_window"(%arg0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %7 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %7 : tensor<f32>
  }) {base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x8x8x8xf32>
  %3 = "stablehlo.reduce_window"(%0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %7 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %7 : tensor<f32>
  }) {base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>} : (tensor<1x16x16x1xf32>, tensor<f32>) -> tensor<1x8x8x1xf32>
  %4 = stablehlo.reshape %3 : (tensor<1x8x8x1xf32>) -> tensor<8x8xf32>
  %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64: 1, 2>}> : (tensor<8x8xf32>) -> tensor<4x8x8x8xf32>
  %6 = stablehlo.divide %2, %5 : tensor<4x8x8x8xf32>
  return %6 : tensor<4x8x8x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

//===------------------------------------------------------------------------===
// stablehlo.reduce_window -> max pool
//===------------------------------------------------------------------------===

// CHECK-LABEL: maxpool_same
func.func @maxpool_same(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x8x8x8xf32>
  func.return %1 : tensor<4x8x8x8xf32>
}

// CHECK: "tfl.max_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32>

// -----

// CHECK-LABEL: maxpool_explicit
func.func @maxpool_explicit(%arg0: tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0)({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %2 = stablehlo.maximum %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) {
    padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1> } : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<1x56x56x64xf32>
  func.return %1 : tensor<1x56x56x64xf32>
}

// CHECK: %[[PADDING_CONST:.*]] = arith.constant dense<0xFF800000> : tensor<f32>
// CHECK: %[[PADDING:.*]] = arith.constant dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
// CHECK: %[[PAD_V2_OUT:.*]] = "tfl.padv2"(%arg0, %[[PADDING]], %[[PADDING_CONST]]) : (tensor<1x112x112x64xf32>, tensor<4x2xi64>, tensor<f32>) -> tensor<1x114x114x64xf32>
// CHECK: "tfl.max_pool_2d"(%[[PAD_V2_OUT]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x114x114x64xf32>) -> tensor<1x56x56x64xf32>


// -----

// CHECK-LABEL: maxpool_valid
func.func @maxpool_valid(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = array<i64: 1, 1, 1, 1>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  func.return %1 : tensor<4x7x7x8xf32>
}

// CHECK: "tfl.max_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>

// -----

// CHECK-LABEL: maxpool_valid_channel_first
func.func @maxpool_valid_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x7x7xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.transpose"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
  %2 = "stablehlo.reduce_window"(%1, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    stablehlo.return %4 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x7x7x3xf32>
  %3 = "stablehlo.transpose"(%2) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<4x7x7x3xf32>) -> tensor<4x3x7x7xf32>
  return %3 : tensor<4x3x7x7xf32>
}

// CHECK:      %[[TPOSED_ARG0:.*]] = "tfl.transpose"(%arg0
// CHECK:      "tfl.max_pool_2d"(%[[TPOSED_ARG0]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x3xf32>) -> tensor<4x7x7x3xf32>
// CHECK:      return
// CHECK-SAME: tensor<4x3x7x7xf32>

// -----

// CHECK-LABEL: maxpool_same_channel_first
func.func @maxpool_same_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x8x8xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.transpose"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
  %2 = "stablehlo.reduce_window"(%1, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    stablehlo.return %4 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x8x8x3xf32>
  %3 = "stablehlo.transpose"(%2) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<4x8x8x3xf32>) -> tensor<4x3x8x8xf32>
  return %3 : tensor<4x3x8x8xf32>
}

// CHECK:      %[[TPOSED_ARG0:.*]] = "tfl.transpose"(%arg0
// CHECK:      "tfl.max_pool_2d"(%[[TPOSED_ARG0]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x3xf32>) -> tensor<4x8x8x3xf32>
// CHECK:      return
// CHECK-SAME: tensor<4x3x8x8xf32>

// -----

//===------------------------------------------------------------------------===
// stablehlo.reduce_window -> tfl.cumsum
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_window_sum
func.func @reduce_window_sum(%arg0: tensor<4x12xf32>) -> tensor<4x12xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {base_dilations = array<i64: 1, 1>, padding = dense<[[3, 0], [0, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 4, 1>, window_strides = array<i64: 1, 1>} : (tensor<4x12xf32>, tensor<f32>) -> tensor<4x12xf32>
  func.return %1 : tensor<4x12xf32>
}

// CHECK: %[[AXIS:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK: "tfl.cumsum"(%arg0, %[[AXIS]]) <{exclusive = false, reverse = false}> : (tensor<4x12xf32>, tensor<i32>) -> tensor<4x12xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: slice
func.func @slice(%arg0: tensor<1x4672xf32>) -> tensor<1x519xf32> {
  %0 = "stablehlo.slice"(%arg0) <{limit_indices = array<i64: 1, 4672>, start_indices = array<i64: 0, 4153>, strides = array<i64: 1, 1>}> : (tensor<1x4672xf32>) -> tensor<1x519xf32>
  func.return %0 : tensor<1x519xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 4153]> : tensor<2xi32>
// CHECK: %[[CST_0:.*]] = arith.constant dense<[1, 4672]> : tensor<2xi32>
// CHECK: %[[CST_1:.*]] = arith.constant dense<1> : tensor<2xi32>
// CHECK: %[[VAL_3:.*]] = "tfl.strided_slice"(%arg0, %[[CST]], %[[CST_0]], %[[CST_1]]) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<1x4672xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x519xf32>

// -----

// CHECK-LABEL: rank5_slice
func.func @rank5_slice(%arg0: tensor<16x192x48x1x2xf32>) -> tensor<16x192x48x1x1xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<5xi32>
  // CHECK: %[[CST_0:.*]] = arith.constant dense<[16, 192, 48, 1, 1]> : tensor<5xi32>
  // CHECK: %[[CST_1:.*]] = arith.constant dense<1> : tensor<5xi32>
  // CHECK: "tfl.strided_slice"(%arg0, %[[CST]], %[[CST_0]], %[[CST_1]])
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 0, 0, 0, 0, 0>,
    limit_indices = array<i64: 16, 192, 48, 1, 1>,
    strides = array<i64: 1, 1, 1, 1, 1>
  } : (tensor<16x192x48x1x2xf32>) -> tensor<16x192x48x1x1xf32>
  return %0 : tensor<16x192x48x1x1xf32>
}

// -----

// CHECK-LABEL: dynamic_slice_mhlo
func.func @dynamic_slice_mhlo(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<2xi32>
  // CHECK: %[[CST_0:.*]] = arith.constant dense<1> : tensor<2xi32>
  // CHECK: %[[CST_1:.*]] = arith.constant dense<1> : tensor<2xi32>
  // CHECK: "tfl.strided_slice"(%arg0, %[[CST]], %[[CST_0]], %[[CST_1]])
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 1, 1>,
    strides = array<i64: 1, 1>
  } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: rank6_slice_collapse
func.func @rank6_slice_collapse(%arg0: tensor<1x16x192x48x1x2xf32>) -> tensor<1x16x192x48x1x1xf32> {
  // CHECK: %[[RESHAPE_IN:.*]] = "tfl.reshape"(%arg0, %{{.*}}) : (tensor<1x16x192x48x1x2xf32>, tensor<5xi32>) -> tensor<16x192x48x1x2xf32>
  // CHECK: %[[SLICE:.*]] = "tfl.strided_slice"(%[[RESHAPE_IN]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} (tensor<16x192x48x1x2xf32>, tensor<5xi32>, tensor<5xi32>, tensor<5xi32>) -> tensor<16x192x48x1x1xf32>
  // CHECK: %[[RESHAPE_OUT:.*]] = "tfl.reshape"(%[[SLICE]], %{{.*}}) : (tensor<16x192x48x1x1xf32>, tensor<6xi32>) -> tensor<1x16x192x48x1x1xf32>
  %cst = arith.constant dense<0> : tensor<6xi32>
  %cst_0 = arith.constant dense<[1, 16, 192, 48, 1, 1]> : tensor<6xi32>
  %cst_1 = arith.constant dense<1> : tensor<6xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<1x16x192x48x1x2xf32>, tensor<6xi32>, tensor<6xi32>, tensor<6xi32>) -> tensor<1x16x192x48x1x1xf32>
  return %0 : tensor<1x16x192x48x1x1xf32>
}

// -----

//===----------------------------------------------------------------------===//
// stablehlo.real_dynamic_slice
//===----------------------------------------------------------------------===//

func.func @real_dynamic_slice(%arg0: tensor<1x?x4x256xf32>, %arg1: tensor<4xi32>, %arg2: tensor<4xi32>) -> tensor<1x?x4x128xf32> {
  %cst = stablehlo.constant dense<1> : tensor<4xi32>
  %0 = stablehlo.real_dynamic_slice %arg0, %arg1, %arg2, %cst : (tensor<1x?x4x256xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x?x4x128xf32>
  func.return %0 : tensor<1x?x4x128xf32>
}

// CHECK: "tfl.strided_slice"(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<1x?x4x256xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x?x4x128xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.sort
//===----------------------------------------------------------------------===//

// CHECK-LABEL: sort_to_topk_iota_broadcast
func.func @sort_to_topk_iota_broadcast(%arg0: tensor<3x6xf32>) -> (tensor<3x6xf32>, tensor<3x6xi32>) {
  %0 = "stablehlo.iota"() <{ iota_dimension = 0 : i64 }> : () -> tensor<6xi32>
  %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64: 1>, name = "broadcast.0"}> : (tensor<6xi32>) -> tensor<3x6xi32>
  %2:2 = "stablehlo.sort"(%arg0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %3 = "stablehlo.compare"(%arg1, %arg2) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%3) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<3x6xf32>, tensor<3x6xi32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)
  func.return %2#0, %2#1 : tensor<3x6xf32>, tensor<3x6xi32>
}

// CHECK: arith.constant dense<6> : tensor<i32>
// CHECK: %[[CST:.*]] = arith.constant dense<6> : tensor<i32>
// CHECK: %values, %indices = "tfl.topk_v2"(%arg0, %[[CST]]) : (tensor<3x6xf32>, tensor<i32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)

// -----

// CHECK-LABEL: sort_to_topk_iota_cst_broadcast
func.func @sort_to_topk_iota_cst_broadcast(%arg0: tensor<3x6xf32>) -> (tensor<3x6xf32>, tensor<3x6xi32>) {
  %0 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5]> : tensor<6xi32>
  %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64: 1>, name = "broadcast.0"}> : (tensor<6xi32>) -> tensor<3x6xi32>
  %2:2 = "stablehlo.sort"(%arg0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %3 = "stablehlo.compare"(%arg1, %arg2) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%3) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<3x6xf32>, tensor<3x6xi32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)
  func.return %2#0, %2#1 : tensor<3x6xf32>, tensor<3x6xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<6> : tensor<i32>
// CHECK: %values, %indices = "tfl.topk_v2"(%arg0, %[[CST]]) : (tensor<3x6xf32>, tensor<i32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)

// -----

// CHECK-LABEL: sort_to_topk_const
func.func @sort_to_topk_const(%arg0: tensor<3x6xf32>) -> (tensor<3x6xf32>, tensor<3x6xi32>) {
  %0 = stablehlo.constant dense<[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]> : tensor<3x6xi32>
  %1:2 = "stablehlo.sort"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %3 = "stablehlo.compare"(%arg1, %arg2) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%3) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<3x6xf32>, tensor<3x6xi32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)
  func.return %1#0, %1#1 : tensor<3x6xf32>, tensor<3x6xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<6> : tensor<i32>
// CHECK: %values, %indices = "tfl.topk_v2"(%arg0, %[[CST]]) : (tensor<3x6xf32>, tensor<i32>) -> (tensor<3x6xf32>, tensor<3x6xi32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.iota
//===----------------------------------------------------------------------===//

// CHECK-LABEL: iota_1d
func.func @iota_1d() -> tensor<123xf32> {
  %0 = "stablehlo.iota"() <{ iota_dimension = 0 : i64 }> : () -> tensor<123xf32>
  func.return %0 : tensor<123xf32>
}

// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<1.230000e+02> : tensor<f32>
// CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK:     "tfl.range"(%[[CST_1]], %[[CST_2]], %[[CST_3]]) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<123xf32>

// -----

// CHECK-LABEL: iota_i64
func.func @iota_i64() -> tensor<123xi64> {
  %0 = "stablehlo.iota"() <{ iota_dimension = 0 : i64 }> : () -> tensor<123xi64>
  func.return %0 : tensor<123xi64>
}

// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<123> : tensor<i64>
// CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<1> : tensor<i64>
// CHECK:     "tfl.range"(%[[CST_1]], %[[CST_2]], %[[CST_3]]) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<123xi64>

// -----

// CHECK-LABEL: iota_i32
func.func @iota_i32() -> tensor<123xi32> {
  %0 = "stablehlo.iota"() <{ iota_dimension = 0 : i64 }> : () -> tensor<123xi32>
  func.return %0 : tensor<123xi32>
}

// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<123> : tensor<i32>
// CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:     "tfl.range"(%[[CST_1]], %[[CST_2]], %[[CST_3]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<123xi32>

// -----

// CHECK-LABEL: iota_3d
func.func @iota_3d() -> tensor<5x7x9xi32> {
  %0 = "stablehlo.iota"() <{ iota_dimension = 1 : i64 }> : () -> tensor<5x7x9xi32>
  func.return %0 : tensor<5x7x9xi32>
}

// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<7> : tensor<i32>
// CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:     %[[RANGE:.*]] = "tfl.range"(%[[CST_1]], %[[CST_2]], %[[CST_3]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<7xi32>
// CHECK:     %[[CST_4:.*]] = arith.constant dense<[1, 7, 1]> : tensor<3xi64>
// CHECK:     %[[CAST:.*]] = "tfl.cast"(%[[CST_4]]) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK:     %[[RESHAPE:.*]] = "tfl.reshape"(%[[RANGE]], %[[CAST]]) : (tensor<7xi32>, tensor<3xi32>) -> tensor<1x7x1xi32>
// CHECK:     %[[CST_5:.*]] = arith.constant dense<[5, 7, 9]> : tensor<3xi64>
// CHECK:     "tfl.broadcast_to"(%[[RESHAPE]], %[[CST_5]]) : (tensor<1x7x1xi32>, tensor<3xi64>) -> tensor<5x7x9xi32>

// -----

// CHECK-LABEL: dynamic_iota_i32_1d
func.func @dynamic_iota_i32_1d(%arg0: tensor<1xi32>) -> tensor<?xi32> {
  %0 = "stablehlo.dynamic_iota"(%arg0) <{iota_dimension = 0 : i64}> : (tensor<1xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<> : tensor<0xi32>
// CHECK:     %0 = "tfl.reshape"(%arg0, %[[CST_1]]) : (tensor<1xi32>, tensor<0xi32>) -> tensor<i32>
// CHECK:     %1 = "tfl.range"(%[[CST]], %0, %[[CST_0]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>

// -----

// CHECK-LABEL: dynamic_iota_f32_1d
func.func @dynamic_iota_f32_1d(%arg0: tensor<1xi32>) -> tensor<?xf32> {
  %0 = "stablehlo.dynamic_iota"(%arg0) <{iota_dimension = 0 : i64}> : (tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK:     %0 = "tfl.cast"(%arg0) : (tensor<1xi32>) -> tensor<1xf32>
// CHECK:     %[[CST_1:.*]] = arith.constant dense<> : tensor<0xi32>
// CHECK:     %1 = "tfl.reshape"(%0, %[[CST_1]]) : (tensor<1xf32>, tensor<0xi32>) -> tensor<f32>
// CHECK:     %2 = "tfl.range"(%[[CST]], %1, %[[CST_0]]) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.dynamic_slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: dynamic_slice
func.func @dynamic_slice(%arg0: tensor<7x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = array<i64: 4, 2>}> : (tensor<7x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}

// CHECK-DAG: %[[CST_IS_0:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_IS_3:.*]] = arith.constant dense<3> : tensor<i32>
// CHECK:     %[[MAX_1:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[MIN_1:.*]] = "tfl.minimum"(%[[CST_IS_3]], %[[MAX_1]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[CST_IS_1:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:     %[[MAX_2:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[MIN_2:.*]] = "tfl.minimum"(%[[CST_IS_1]], %[[MAX_2]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[PACK:.*]] = "tfl.pack"(%[[MIN_1]], %[[MIN_2]]) <{axis = 0 : i32, values_count = 2 : i32}> : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
// CHECK:     %[[SLICE_SIZE:.*]] = arith.constant dense<[4, 2]> : tensor<2xi64>
// CHECK:     "tfl.slice"(%arg0, %[[PACK]], %[[SLICE_SIZE]]) : (tensor<7x3xf32>, tensor<2xi32>, tensor<2xi64>) -> tensor<4x2xf32>

// -----

// CHECK-LABEL: dynamic_slice_i64
func.func @dynamic_slice_i64(%arg0: tensor<7x3xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<4x2xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = array<i64: 4, 2>}> : (tensor<7x3xf32>, tensor<i64>, tensor<i64>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}

// CHECK-DAG: %[[CST_IS_0:.*]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[CST_IS_3:.*]] = arith.constant dense<3> : tensor<i64>
// CHECK:     %[[MAX_1:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:     %[[MIN_1:.*]] = "tfl.minimum"(%[[CST_IS_3]], %[[MAX_1]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:     %[[CST_IS_1:.*]] = arith.constant dense<1> : tensor<i64>
// CHECK:     %[[MAX_2:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:     %[[MIN_2:.*]] = "tfl.minimum"(%[[CST_IS_1]], %[[MAX_2]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:     %[[PACK:.*]] = "tfl.pack"(%[[MIN_1]], %[[MIN_2]]) <{axis = 0 : i32, values_count = 2 : i32}> : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
// CHECK:     %[[SLICE_SIZE:.*]] = arith.constant dense<[4, 2]> : tensor<2xi64>
// CHECK:     "tfl.slice"(%arg0, %[[PACK]], %[[SLICE_SIZE]]) : (tensor<7x3xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x2xf32>

// -----

// CHECK-LABEL: dynamic_slice_splat_sizes
func.func @dynamic_slice_splat_sizes(%arg0: tensor<7x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = array<i64: 2, 2>}> : (tensor<7x3xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-DAG: %[[CST_IS_0:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_IS_5:.*]] = arith.constant dense<5> : tensor<i32>
// CHECK:     %[[MAX_1:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[MIN_1:.*]] = "tfl.minimum"(%[[CST_IS_5]], %[[MAX_1]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[CST_IS_1:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:     %[[MAX_2:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[MIN_2:.*]] = "tfl.minimum"(%[[CST_IS_1]], %[[MAX_2]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[PACK:.*]] = "tfl.pack"(%[[MIN_1]], %[[MIN_2]]) <{axis = 0 : i32, values_count = 2 : i32}> : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
// CHECK:     %[[SLICE_SIZE:.*]] = arith.constant dense<2> : tensor<2xi64>
// CHECK:     "tfl.slice"(%arg0, %[[PACK]], %[[SLICE_SIZE]]) : (tensor<7x3xf32>, tensor<2xi32>, tensor<2xi64>) -> tensor<2x2xf32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.dynamic_update_slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: dynamic_update_slice
func.func @dynamic_update_slice(%arg0: tensor<28x1x100xf32>, %arg1: tensor<1x1x100xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> tensor<28x1x100xf32> {
  %0 = "stablehlo.dynamic_update_slice"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<28x1x100xf32>, tensor<1x1x100xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<28x1x100xf32>
  func.return %0 : tensor<28x1x100xf32>
}

// CHECK: %0 = "tfl.pack"(%arg2, %arg3, %arg4) <{axis = 0 : i32, values_count = 3 : i32}> : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
// CHECK: %1 = "tfl.dynamic_update_slice"(%arg0, %arg1, %0) : (tensor<28x1x100xf32>, tensor<1x1x100xf32>, tensor<3xi32>) -> tensor<28x1x100xf32>

// -----

// CHECK-LABEL: dynamic_update_slice_inputs_have_dynamic_dim
func.func @dynamic_update_slice_inputs_have_dynamic_dim(%arg0: tensor<?x4xi32>, %arg1: tensor<?x2xi32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<?x4xi32> {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<?x4xi32>, tensor<?x2xi32>, tensor<i32>, tensor<i32>) -> tensor<?x4xi32>
  func.return %0 : tensor<?x4xi32>
}

// CHECK: %0 = "tfl.pack"(%arg2, %arg3) <{axis = 0 : i32, values_count = 2 : i32}> : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
// CHECK: %1 = "tfl.dynamic_update_slice"(%arg0, %arg1, %0) : (tensor<?x4xi32>, tensor<?x2xi32>, tensor<2xi32>) -> tensor<?x4xi32>

// -----

// CHECK-LABEL: dynamic_update_slice_operand_has_dynamic_dim
func.func @dynamic_update_slice_operand_has_dynamic_dim(%arg0: tensor<1x?x256xf32>, %arg1: tensor<1x1x256xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> tensor<1x?x256xf32> {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 : (tensor<1x?x256xf32>, tensor<1x1x256xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x?x256xf32>
  func.return %0 : tensor<1x?x256xf32>
}

// CHECK: %0 = "tfl.pack"(%arg2, %arg3, %arg4) <{axis = 0 : i32, values_count = 3 : i32}> : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
// CHECK: %1 = "tfl.dynamic_update_slice"(%arg0, %arg1, %0) : (tensor<1x?x256xf32>, tensor<1x1x256xf32>, tensor<3xi32>) -> tensor<1x?x256xf32>

// -----

//===----------------------------------------------------------------------===//
// rounding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @nearest_even
func.func @nearest_even(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.round_nearest_even"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK: %0 = "tfl.round"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>

// -----

// CHECK-LABEL: @round
func.func @round(%arg0: tensor<8x128xf32>) -> tensor<8x128xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<8x128xf32>
  %1 = stablehlo.constant dense<5.000000e-01> : tensor<8x128xf32>
  %2 = stablehlo.constant dense<1.000000e+00> : tensor<8x128xf32>
  %3 = "stablehlo.floor"(%arg0) : (tensor<8x128xf32>) -> tensor<8x128xf32>
  %4 = stablehlo.subtract %arg0, %3 : tensor<8x128xf32>
  %5 = "stablehlo.compare"(%4, %1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xi1>
  %6 = "stablehlo.compare"(%4, %1) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xi1>
  %7 = stablehlo.multiply %arg0, %1 : tensor<8x128xf32>
  %8 = "stablehlo.floor"(%7) : (tensor<8x128xf32>) -> tensor<8x128xf32>
  %9 = stablehlo.multiply %8, %0 : tensor<8x128xf32>
  %10 = stablehlo.subtract %3, %9 : tensor<8x128xf32>
  %11 = "stablehlo.compare"(%10, %2) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xi1>
  %12 = stablehlo.and %6, %11 : tensor<8x128xi1>
  %13 = stablehlo.or %5, %12 : tensor<8x128xi1>
  %14 = stablehlo.add %3, %2 : tensor<8x128xf32>
  %15 = "stablehlo.select"(%13, %14, %3) : (tensor<8x128xi1>, tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>
  func.return %15 : tensor<8x128xf32>
}

// CHECK: "tfl.round"(%arg0) : (tensor<8x128xf32>) -> tensor<8x128xf32>

// -----

// CHECK-LABEL: floor_mod_float
func.func @floor_mod_float(%arg0: tensor<192x8xf32>, %arg1: tensor<192x8xf32>) -> tensor<192x8xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<192x8xf32>
  %1 = stablehlo.remainder %arg0, %arg1 : tensor<192x8xf32>
  %2 = "stablehlo.compare"(%1, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<192x8xf32>, tensor<192x8xf32>) -> tensor<192x8xi1>
  %3 = "stablehlo.compare"(%arg1, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<192x8xf32>, tensor<192x8xf32>) -> tensor<192x8xi1>
  %4 = "stablehlo.compare"(%2, %3) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<192x8xi1>, tensor<192x8xi1>) -> tensor<192x8xi1>
  %5 = "stablehlo.compare"(%1, %0) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<192x8xf32>, tensor<192x8xf32>) -> tensor<192x8xi1>
  %6 = stablehlo.and %4, %5 : tensor<192x8xi1>
  %7 = stablehlo.add %1, %arg1 : tensor<192x8xf32>
  %8 = "stablehlo.select"(%6, %7, %1) : (tensor<192x8xi1>, tensor<192x8xf32>, tensor<192x8xf32>) -> tensor<192x8xf32>
  func.return %8 : tensor<192x8xf32>
}

// CHECK: "tfl.floor_mod"(%arg0, %arg1) : (tensor<192x8xf32>, tensor<192x8xf32>) -> tensor<192x8xf32>

// -----

// CHECK-LABEL: floor_mod_int
func.func @floor_mod_int(%arg0: tensor<192x8xi32>, %arg1: tensor<192x8xi32>) -> tensor<192x8xi32> {
  %0 = stablehlo.constant dense<0> : tensor<192x8xi32>
  %1 = stablehlo.remainder %arg0, %arg1 : tensor<192x8xi32>
  %2 = "stablehlo.compare"(%1, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<192x8xi32>, tensor<192x8xi32>) -> tensor<192x8xi1>
  %3 = "stablehlo.compare"(%arg1, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<192x8xi32>, tensor<192x8xi32>) -> tensor<192x8xi1>
  %4 = "stablehlo.compare"(%2, %3) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<192x8xi1>, tensor<192x8xi1>) -> tensor<192x8xi1>
  %5 = "stablehlo.compare"(%1, %0) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<192x8xi32>, tensor<192x8xi32>) -> tensor<192x8xi1>
  %6 = stablehlo.and %4, %5 : tensor<192x8xi1>
  %7 = stablehlo.add %1, %arg1 : tensor<192x8xi32>
  %8 = "stablehlo.select"(%6, %7, %1) : (tensor<192x8xi1>, tensor<192x8xi32>, tensor<192x8xi32>) -> tensor<192x8xi32>
  func.return %8 : tensor<192x8xi32>
}

// CHECK: "tfl.floor_mod"(%arg0, %arg1) : (tensor<192x8xi32>, tensor<192x8xi32>) -> tensor<192x8xi32>

// -----

// CHECK-LABEL: floor_mod_float_cst
func.func @floor_mod_float_cst(%arg0: tensor<192x8xf32>) -> tensor<192x8xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<192x8xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<192x8xf32>
  %2 = stablehlo.remainder %arg0, %1 : tensor<192x8xf32>
  %3 = "stablehlo.compare"(%2, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<192x8xf32>, tensor<192x8xf32>) -> tensor<192x8xi1>
  %4 = "stablehlo.compare"(%2, %0) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<192x8xf32>, tensor<192x8xf32>) -> tensor<192x8xi1>
  %5 = stablehlo.and %3, %4 : tensor<192x8xi1>
  %6 = stablehlo.add %2, %1 : tensor<192x8xf32>
  %7 = "stablehlo.select"(%5, %6, %2) : (tensor<192x8xi1>, tensor<192x8xf32>, tensor<192x8xf32>) -> tensor<192x8xf32>
  func.return %7 : tensor<192x8xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<192x8xf32>
// CHECK: "tfl.floor_mod"(%arg0, %[[CST]]) : (tensor<192x8xf32>, tensor<192x8xf32>) -> tensor<192x8xf32>

// -----

// CHECK-LABEL: floor_mod_int_cst
func.func @floor_mod_int_cst(%arg0: tensor<192x8xi32>) -> tensor<192x8xi32> {
  %0 = stablehlo.constant dense<0> : tensor<192x8xi32>
  %1 = stablehlo.constant dense<2> : tensor<192x8xi32>
  %2 = stablehlo.remainder %arg0, %1 : tensor<192x8xi32>
  %3 = "stablehlo.compare"(%2, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<192x8xi32>, tensor<192x8xi32>) -> tensor<192x8xi1>
  %4 = "stablehlo.compare"(%2, %0) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<192x8xi32>, tensor<192x8xi32>) -> tensor<192x8xi1>
  %5 = stablehlo.and %3, %4 : tensor<192x8xi1>
  %6 = stablehlo.add %2, %1 : tensor<192x8xi32>
  %7 = "stablehlo.select"(%5, %6, %2) : (tensor<192x8xi1>, tensor<192x8xi32>, tensor<192x8xi32>) -> tensor<192x8xi32>
  func.return %7 : tensor<192x8xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2> : tensor<192x8xi32>
// CHECK: "tfl.floor_mod"(%arg0, %[[CST]]) : (tensor<192x8xi32>, tensor<192x8xi32>) -> tensor<192x8xi32>

// -----

// CHECK-LABEL: floor_div
func.func @floor_div(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<10x10xf32>
  %1 = stablehlo.constant dense<-1.000000e+00> : tensor<10x10xf32>
  %2 = stablehlo.remainder %arg0, %arg1 : tensor<10x10xf32>
  %3 = "stablehlo.compare"(%2, %0) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xi1>
  %4 = "stablehlo.sign"(%arg1) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %5 = "stablehlo.sign"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %6 = "stablehlo.compare"(%4, %5) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xi1>
  %7 = stablehlo.and %3, %6 : tensor<10x10xi1>
  %8 = stablehlo.subtract %arg0, %2 : tensor<10x10xf32>
  %9 = stablehlo.divide %8, %arg1 : tensor<10x10xf32>
  %10 = stablehlo.add %9, %1 : tensor<10x10xf32>
  %11 = "stablehlo.select"(%7, %10, %9) : (tensor<10x10xi1>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %12 = "stablehlo.round_nearest_afz"(%11) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %13 = "stablehlo.tuple"(%12) : (tensor<10x10xf32>) -> tuple<tensor<10x10xf32>>
  func.return %12 : tensor<10x10xf32>
}

// CHECK: tfl.floor_div %arg0, %arg1 : tensor<10x10xf32

// -----

// CHECK-LABEL: floor_div_cst
func.func @floor_div_cst(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<10x10xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<10x10xf32>
  %2 = stablehlo.constant dense<1.000000e+00> : tensor<10x10xf32>
  %3 = stablehlo.constant dense<5.000000e-01> : tensor<10x10xf32>
  %4 = stablehlo.constant dense<-1.000000e+00> : tensor<10x10xf32>
  %5 = stablehlo.remainder %arg0, %0 : tensor<10x10xf32>
  %6 = "stablehlo.compare"(%5, %1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xi1>
  %7 = "stablehlo.sign"(%5) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %8 = "stablehlo.compare"(%2, %7) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xi1>
  %9 = stablehlo.and %6, %8 : tensor<10x10xi1>
  %10 = stablehlo.subtract %arg0, %5 : tensor<10x10xf32>
  %11 = stablehlo.multiply %10, %3 : tensor<10x10xf32>
  %12 = stablehlo.add %11, %4 : tensor<10x10xf32>
  %13 = "stablehlo.select"(%9, %12, %11) : (tensor<10x10xi1>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %14 = "stablehlo.round_nearest_afz"(%13) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %15 = "stablehlo.tuple"(%14) : (tensor<10x10xf32>) -> tuple<tensor<10x10xf32>>
  func.return %14 : tensor<10x10xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[BCAST:.*]] = "tfl.broadcast_to"(%[[CST]]
// CHECK: tfl.floor_div %arg0, %[[BCAST]] : tensor<10x10xf32>

// -----

// CHECK-LABEL: floor_div_cst2
func.func @floor_div_cst2(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<10x10xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<10x10xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<10x10xf32>
  %3 = stablehlo.constant dense<-1.000000e+00> : tensor<10x10xf32>
  %4 = stablehlo.remainder %arg0, %1 : tensor<10x10xf32>
  %5 = "stablehlo.compare"(%4, %2) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xi1>
  %6 = "stablehlo.sign"(%4) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %7 = "stablehlo.compare"(%0, %6) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xi1>
  %8 = stablehlo.and %5, %7 : tensor<10x10xi1>
  %9 = stablehlo.subtract %arg0, %4 : tensor<10x10xf32>
  %10 = stablehlo.divide %9, %1 : tensor<10x10xf32>
  %11 = stablehlo.add %10, %3 : tensor<10x10xf32>
  %12 = "stablehlo.select"(%8, %11, %10) : (tensor<10x10xi1>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %13 = "stablehlo.round_nearest_afz"(%12) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %14 = "stablehlo.tuple"(%13) : (tensor<10x10xf32>) -> tuple<tensor<10x10xf32>>
  func.return %13 : tensor<10x10xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[BCAST:.*]] = "tfl.broadcast_to"(%[[CST]]
// CHECK: tfl.floor_div %arg0, %[[BCAST]] : tensor<10x10xf32>

// -----

// CHECK-LABEL: floor_div_broadcast_cst
func.func @floor_div_broadcast_cst(%arg0: tensor<10x8xf32>) -> tensor<10x8xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<10x8xf32>
  %1 = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 4.000000e+00, 8.000000e+00, 1.600000e+01, 3.200000e+01, 6.400000e+01, 1.280000e+02]> : tensor<8xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<10x8xf32>
  %3 = stablehlo.constant dense<-1.000000e+00> : tensor<10x8xf32>
  %5 = "stablehlo.broadcast_in_dim"(%1) <{broadcast_dimensions = array<i64: 1>}> : (tensor<8xf32>) -> tensor<10x8xf32>
  %6 = stablehlo.remainder %arg0, %5 : tensor<10x8xf32>
  %7 = "stablehlo.compare"(%6, %2) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10x8xf32>, tensor<10x8xf32>) -> tensor<10x8xi1>
  %8 = "stablehlo.sign"(%6) : (tensor<10x8xf32>) -> tensor<10x8xf32>
  %9 = "stablehlo.compare"(%0, %8) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10x8xf32>, tensor<10x8xf32>) -> tensor<10x8xi1>
  %10 = stablehlo.and %7, %9 : tensor<10x8xi1>
  %11 = stablehlo.subtract %arg0, %6 : tensor<10x8xf32>
  %12 = stablehlo.divide %11, %5 : tensor<10x8xf32>
  %13 = stablehlo.add %12, %3 : tensor<10x8xf32>
  %14 = "stablehlo.select"(%10, %13, %12) : (tensor<10x8xi1>, tensor<10x8xf32>, tensor<10x8xf32>) -> tensor<10x8xf32>
  %15 = "stablehlo.round_nearest_afz"(%14) : (tensor<10x8xf32>) -> tensor<10x8xf32>
  %16 = "stablehlo.tuple"(%15) : (tensor<10x8xf32>) -> tuple<tensor<10x8xf32>>
  func.return %15 : tensor<10x8xf32>
}

// CHECK: %[[BCAST:.*]] = "tfl.broadcast_to"
// CHECK: tfl.floor_div %arg0, %[[BCAST]] : tensor<10x8xf32>

// -----

// CHECK-LABEL: floor_div_cst3
func.func @floor_div_cst3(%arg0: tensor<12xf32>) -> tensor<12xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<12xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "stablehlo.broadcast_in_dim"(%2) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<12xf32>
  %4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<12xf32>
  %8 = stablehlo.remainder %arg0, %1 : tensor<12xf32>
  %9 = stablehlo.subtract %arg0, %8 : tensor<12xf32>
  %10 = stablehlo.divide %9, %1 : tensor<12xf32>
  %11 = stablehlo.compare  NE, %8, %3,  FLOAT : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xi1>
  %12 = stablehlo.sign %8 : tensor<12xf32>
  %13 = stablehlo.compare  NE, %5, %12,  FLOAT : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xi1>
  %14 = stablehlo.and %11, %13 : tensor<12xi1>
  %15 = stablehlo.subtract %10, %5 : tensor<12xf32>
  %16 = stablehlo.select %14, %15, %10 : tensor<12xi1>, tensor<12xf32>
  %17 = stablehlo.round_nearest_afz %16 : tensor<12xf32>
  return %17 : tensor<12xf32>
}

// CHECK: %[[BCAST:.*]] = "tfl.broadcast_to"
// CHECK: %[[FLOOR_DIV:.*]] = tfl.floor_div %arg0, %[[BCAST]] : tensor<12xf32>
// CHECK: return %[[FLOOR_DIV]] : tensor<12xf32>

// -----

//===----------------------------------------------------------------------===//
// unary elementwise
//===----------------------------------------------------------------------===//

// CHECK-LABEL: convert_i32_f32
func.func @convert_i32_f32(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.cast

// -----

// CHECK-LABEL: abs
func.func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.abs

// -----

// CHECK-LABEL: abs_dynamic
func.func @abs_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.abs

// -----

// CHECK-LABEL: ceil
func.func @ceil(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.ceil

// -----

// CHECK-LABEL: ceil_dynamic
func.func @ceil_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.ceil"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.ceil

// -----

// CHECK-LABEL: complex_abs
func.func @complex_abs(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  %0 = "stablehlo.abs"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: %0 = "tfl.complex_abs"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK: return %0 : tensor<2xf32>
// -----

func.func @is_finite(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %0 = "stablehlo.is_finite"(%arg0) : (tensor<2xf32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: %0 = tfl.sub %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2xf32>
// CHECK: %cst = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %1 = "tfl.equal"(%0, %cst) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
// CHECK: return %1 : tensor<2xi1>

// -----

func.func @is_finite_dynamic(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  %0 = "stablehlo.is_finite"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// CHECK: %0 = tfl.sub %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<?xf32>
// CHECK: %cst = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %1 = "tfl.equal"(%0, %cst) : (tensor<?xf32>, tensor<f32>) -> tensor<?xi1>

// -----

// CHECK-LABEL: cos
func.func @cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.cosine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.cos

// -----

// CHECK-LABEL: cos_dynamic
func.func @cos_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.cosine"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.cos

// -----

// CHECK-LABEL: logistic
func.func @logistic(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.logistic"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.logistic

// -----

// CHECK-LABEL: exp
func.func @exp(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.exp

// -----

// CHECK-LABEL: exp_dynamic
func.func @exp_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.exponential"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.exp

// -----

// CHECK-LABEL: expm1
func.func @expm1(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: %0 = "tfl.exp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %1 = tfl.sub(%0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>

// -----

// CHECK-LABEL: floor
func.func @floor(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.floor

// -----

// CHECK-LABEL: floor_dynamic
func.func @floor_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.floor

// -----

// CHECK-LABEL: log
func.func @log(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.log

// -----

// CHECK-LABEL: log_dynamic
func.func @log_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.log"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.log

// -----

// CHECK-LABEL: log1p
func.func @log1p(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
// CHECK: %1 = "tfl.log"(%0) : (tensor<2xf32>) -> tensor<2xf32>

// -----

// CHECK-LABEL: log1p_dynamic
func.func @log1p_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK: %1 = "tfl.log"(%0) : (tensor<?xf32>) -> tensor<?xf32>

// -----

// CHECK-LABEL: neg
func.func @neg(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.negate"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.neg

// -----

// CHECK-LABEL: neg_dynamic
func.func @neg_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.negate"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.neg

// -----

// CHECK-LABEL: sin
func.func @sin(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.sine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.sin

// -----

// CHECK-LABEL: sin_dynamic
func.func @sin_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.sine"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.sin

// -----

// CHECK-LABEL: rsqrt
func.func @rsqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.rsqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.rsqrt

// -----

// CHECK-LABEL: rsqrt_dynamic
func.func @rsqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.rsqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.rsqrt

// -----

// CHECK-LABEL: @sqrt
func.func @sqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.sqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.sqrt

// -----

// CHECK-LABEL: sqrt_dynamic
func.func @sqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.sqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.sqrt

// -----

// CHECK-LABEL: tanh
func.func @tanh(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.tanh

// -----

// CHECK-LABEL: tanh_dynamic
func.func @tanh_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.tanh

// -----

// CHECK-LABEL: bitcast
func.func @bitcast(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: tfl.bitcast

// -----

// CHECK-LABEL: bitcast_dynamic
func.func @bitcast_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.bitcast_convert"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK: tfl.bitcast

// -----

// CHECK-LABEL: bitcast_same_widths
func.func @bitcast_same_widths(%arg0: tensor<2xf32>) -> tensor<2xi32> {
  %0 = "stablehlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// CHECK: tfl.bitcast

// -----

//===----------------------------------------------------------------------===//
// logical and bitwise ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: not
func.func @not(%arg0: tensor<5x3x1xi1>) -> tensor<5x3x1xi1> {
  %0 = "stablehlo.not"(%arg0): (tensor<5x3x1xi1>) -> (tensor<5x3x1xi1>)
  func.return %0 : tensor<5x3x1xi1>
}

// CHECK: %0 = "tfl.logical_not"(%arg0) : (tensor<5x3x1xi1>) -> tensor<5x3x1xi1>

// -----

// CHECK-LABEL: not_i8
func.func @not_i8(%arg0: tensor<7x9x11xi8>) -> tensor<7x9x11xi8> {
  %0 = "stablehlo.not"(%arg0): (tensor<7x9x11xi8>) -> (tensor<7x9x11xi8>)
  func.return %0 : tensor<7x9x11xi8>
}

// CHECK: %cst = arith.constant dense<-1> : tensor<i8>
// CHECK: %0 = "tfl.bitwise_xor"(%arg0, %cst) : (tensor<7x9x11xi8>, tensor<i8>) -> tensor<7x9x11xi8>

// -----

// CHECK-LABEL: not_i16
func.func @not_i16(%arg0: tensor<7x9x11xi16>) -> tensor<7x9x11xi16> {
  %0 = "stablehlo.not"(%arg0): (tensor<7x9x11xi16>) -> (tensor<7x9x11xi16>)
  func.return %0 : tensor<7x9x11xi16>
}

// CHECK: %cst = arith.constant dense<-1> : tensor<i16>
// CHECK: %0 = "tfl.bitwise_xor"(%arg0, %cst) : (tensor<7x9x11xi16>, tensor<i16>) -> tensor<7x9x11xi16>

// -----

// CHECK-LABEL: not_i32
func.func @not_i32(%arg0: tensor<7x9x11xi32>) -> tensor<7x9x11xi32> {
  %0 = "stablehlo.not"(%arg0): (tensor<7x9x11xi32>) -> (tensor<7x9x11xi32>)
  func.return %0 : tensor<7x9x11xi32>
}

// CHECK: %cst = arith.constant dense<-1> : tensor<i32>
// CHECK: %0 = "tfl.bitwise_xor"(%arg0, %cst) : (tensor<7x9x11xi32>, tensor<i32>) -> tensor<7x9x11xi32>

// -----

// CHECK-LABEL: not_ui8
func.func @not_ui8(%arg0: tensor<7x9x11xui8>) -> tensor<7x9x11xui8> {
  %0 = "stablehlo.not"(%arg0): (tensor<7x9x11xui8>) -> (tensor<7x9x11xui8>)
  func.return %0 : tensor<7x9x11xui8>
}

// CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<255> : tensor<ui8>}> : () -> tensor<ui8>
// CHECK: %{{.*}} = "tfl.bitwise_xor"(%arg0, %[[CST]]) : (tensor<7x9x11xui8>, tensor<ui8>) -> tensor<7x9x11xui8>

// -----

// CHECK-LABEL: not_ui16
func.func @not_ui16(%arg0: tensor<7x9x11xui16>) -> tensor<7x9x11xui16> {
  %0 = "stablehlo.not"(%arg0): (tensor<7x9x11xui16>) -> (tensor<7x9x11xui16>)
  func.return %0 : tensor<7x9x11xui16>
}

// CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<65535> : tensor<ui16>}> : () -> tensor<ui16>
// CHECK: %{{.*}} = "tfl.bitwise_xor"(%arg0, %[[CST]]) : (tensor<7x9x11xui16>, tensor<ui16>) -> tensor<7x9x11xui16>

// -----

// CHECK-LABEL: not_ui32
func.func @not_ui32(%arg0: tensor<7x9x11xui32>) -> tensor<7x9x11xui32> {
  %0 = "stablehlo.not"(%arg0): (tensor<7x9x11xui32>) -> (tensor<7x9x11xui32>)
  func.return %0 : tensor<7x9x11xui32>
}

// CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
// CHECK: %{{.*}} = "tfl.bitwise_xor"(%arg0, %[[CST]]) : (tensor<7x9x11xui32>, tensor<ui32>) -> tensor<7x9x11xui32>

// -----

//===----------------------------------------------------------------------===//
// binary ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: remainder
func.func @remainder(%arg0: tensor<10x8xi32>, %arg1: tensor<10x8xi32>) -> tensor<10x8xi32> {
  %0 = stablehlo.remainder %arg0, %arg1 : tensor<10x8xi32>
  func.return %0 : tensor<10x8xi32>
}

// CHECK: %0 = "tfl.floor_mod"(%arg0, %arg1) : (tensor<10x8xi32>, tensor<10x8xi32>) -> tensor<10x8xi32>

// -----

// CHECK-LABEL: shift_right_arith
func.func @shift_right_arith(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK: %0 = "tfl.right_shift"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

// -----

// CHECK-LABEL: shift_right_logical
func.func @shift_right_logical(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK: %0 = "tfl.right_shift"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.compare
//===----------------------------------------------------------------------===//

// CHECK-LABEL: greater_unsupported_compare_type
func.func @greater_unsupported_compare_type(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK-NOT: tfl
// CHECK:     stablehlo.compare

// -----

// CHECK-LABEL: equal
func.func @equal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.equal

// -----

// CHECK-LABEL: notequal
func.func @notequal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.not_equal

// -----

// CHECK-LABEL: greater
func.func @greater(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.greater

// -----

// CHECK-LABEL: greater_equal
func.func @greater_equal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.greater_equal

// -----

// CHECK-LABEL: less
func.func @less(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.less

// -----

// CHECK-LABEL: less_equal
func.func @less_equal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction LE>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.less_equal

// -----

//===----------------------------------------------------------------------===//
// mhlo binary element-wise ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: add
func.func @add(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// CHECK: tfl.add
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: maximum
func.func @maximum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK: "tfl.maximum"(%arg0, %arg1)
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: minimum
func.func @minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK: "tfl.minimum"(%arg0, %arg1)
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: mul
func.func @mul(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// CHECK: tfl.mul %arg0, %arg0
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: pow
func.func @pow(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.power %arg0, %arg0 : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK: tfl.pow
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: sub
func.func @sub(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.subtract %arg0, %arg0 : tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// CHECK: tfl.sub %arg0, %arg0
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: div
func.func @div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.divide %arg0, %arg0 : tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// CHECK: tfl.div %arg0, %arg0
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: atan2
func.func @atan2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.atan2 %arg0, %arg0 : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK: "tfl.atan2"(%arg0, %arg0)
// CHECK-NOT: mhlo

// -----

// CHECK-LABEL: broadcast_add
func.func @broadcast_add(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1000xf32>) -> (tensor<1x1000xf32>, tensor<1x1000xf32>) {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf32>) -> tensor<1x1000xf32>
  %1 = stablehlo.add %0, %arg1 : tensor<1x1000xf32>
  %2 = stablehlo.add %arg1, %0 : tensor<1x1000xf32>
  func.return %1, %2 : tensor<1x1000xf32>, tensor<1x1000xf32>
}

// CHECK: tfl.add(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x1xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
// CHECK: tfl.add(%arg1, %arg0) <{fused_activation_function = "NONE"}> : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>

// -----

// CHECK-LABEL: broadcast_add_cst
func.func @broadcast_add_cst(%arg1: tensor<1x1000xf32>) -> (tensor<1x1000xf32>, tensor<1x1000xf32>) {
  %cst = stablehlo.constant dense<1.200000e+01> : tensor<f32>
  %0 = "stablehlo.broadcast_in_dim"(%cst) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1000xf32>
  %1 = stablehlo.add %0, %arg1 : tensor<1x1000xf32>
  %2 = stablehlo.add %arg1, %0 : tensor<1x1000xf32>
  func.return %1, %2 : tensor<1x1000xf32>, tensor<1x1000xf32>
}

// CHECK: tfl.add{{.*}}(tensor<1x1000xf32>, tensor<f32>) -> tensor<1x1000xf32>

// -----

// CHECK-LABEL: broadcast_div
func.func @broadcast_div(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1000xf32>) -> (tensor<1x1000xf32>, tensor<1x1000xf32>) {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf32>) -> tensor<1x1000xf32>
  %1 = stablehlo.divide %0, %arg1 : tensor<1x1000xf32>
  %2 = stablehlo.divide %arg1, %0 : tensor<1x1000xf32>
  func.return %1, %2 : tensor<1x1000xf32>, tensor<1x1000xf32>
}

// CHECK: tfl.div(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x1xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
// CHECK: tfl.div(%arg1, %arg0) <{fused_activation_function = "NONE"}> : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>

// -----

// CHECK-LABEL: broadcast_maximum
func.func @broadcast_maximum(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1000xf32>) -> (tensor<1x1000xf32>, tensor<1x1000xf32>) {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf32>) -> tensor<1x1000xf32>
  %1 = stablehlo.maximum %0, %arg1 : tensor<1x1000xf32>
  %2 = stablehlo.maximum %arg1, %0 : tensor<1x1000xf32>
  func.return %1, %2 : tensor<1x1000xf32>, tensor<1x1000xf32>
}

// CHECK: "tfl.maximum"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
// CHECK: "tfl.maximum"(%arg1, %arg0) : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>

// -----

// CHECK-LABEL: broadcast_minimum
func.func @broadcast_minimum(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1000xf32>) -> (tensor<1x1000xf32>, tensor<1x1000xf32>) {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf32>) -> tensor<1x1000xf32>
  %1 = stablehlo.minimum %0, %arg1 : tensor<1x1000xf32>
  %2 = stablehlo.minimum %arg1, %0 : tensor<1x1000xf32>
  func.return %1, %2 : tensor<1x1000xf32>, tensor<1x1000xf32>
}

// CHECK: "tfl.minimum"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
// CHECK: "tfl.minimum"(%arg1, %arg0) : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>

// -----

// CHECK-LABEL: broadcast_mul
func.func @broadcast_mul(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1000xf32>) -> (tensor<1x1000xf32>, tensor<1x1000xf32>) {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf32>) -> tensor<1x1000xf32>
  %1 = stablehlo.multiply %0, %arg1 : tensor<1x1000xf32>
  %2 = stablehlo.multiply %arg1, %0 : tensor<1x1000xf32>
  func.return %1, %2 : tensor<1x1000xf32>, tensor<1x1000xf32>
}

// CHECK: tfl.mul(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x1xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
// CHECK: tfl.mul(%arg1, %arg0) <{fused_activation_function = "NONE"}> : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>

// -----

// CHECK-LABEL: broadcast_sub
func.func @broadcast_sub(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1000xf32>) -> (tensor<1x1000xf32>, tensor<1x1000xf32>) {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf32>) -> tensor<1x1000xf32>
  %1 = stablehlo.subtract %0, %arg1 : tensor<1x1000xf32>
  %2 = stablehlo.subtract %arg1, %0 : tensor<1x1000xf32>
  func.return %1, %2 : tensor<1x1000xf32>, tensor<1x1000xf32>
}

// CHECK: tfl.sub(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x1xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
// CHECK: tfl.sub(%arg1, %arg0) <{fused_activation_function = "NONE"}> : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>

// -----

// CHECK-LABEL: broadcast_pow
func.func @broadcast_pow(%arg0: tensor<1xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xi32>) -> tensor<4xi32>
  %1 = stablehlo.power %0, %arg1 : tensor<4xi32>
  %2 = stablehlo.power %arg1, %0 : tensor<4xi32>
  func.return %1, %2 : tensor<4xi32>, tensor<4xi32>
}

// CHECK: tfl.pow(%arg0, %arg1) : (tensor<1xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK: tfl.pow(%arg1, %arg0) : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>

// -----

//===----------------------------------------------------------------------===//
// mhlo ternary ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: clamp
func.func @clamp(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.clamp"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-NEXT: %0 = "tfl.minimum"(%arg1, %arg2)
// CHECK-NEXT: %1 = "tfl.maximum"(%0, %arg0)
// CHECK-NEXT: return %1 : tensor<f32>

// -----

// CHECK-LABEL: select
func.func @select(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK:     "tfl.select"(%arg0, %arg1, %arg2)
// CHECK-NOT: mhlo

// -----

//===----------------------------------------------------------------------===//
// stablehlo.while
//===----------------------------------------------------------------------===//

// CHECK-LABEL: while_with_variadic
func.func @while_with_variadic() -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %cst = arith.constant dense<1> : tensor<i32>
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %cst_1 = arith.constant dense<1000> : tensor<i32>
  %0:3 = "stablehlo.while"(%cst_0, %cst, %cst_1) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>):
    %1 = "stablehlo.compare"(%arg0, %arg2) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>):
    %1 = stablehlo.add %arg0, %arg1 : tensor<i32>
    "stablehlo.return"(%1, %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  func.return %0#0, %0#1, %0#2 : tensor<i32>, tensor<i32>, tensor<i32>
}


// CHECK-DAG: %cst = arith.constant dense<1> : tensor<i32>
// CHECK-DAG: %cst_0 = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %cst_1 = arith.constant dense<1000> : tensor<i32>
// CHECK:     %0:3 = "tfl.while"(%cst_0, %cst, %cst_1) <{is_stateless = false}> ({
// CHECK:     ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>):
// CHECK:     %1 = tfl.less(%arg0, %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:     "tfl.yield"(%1) : (tensor<i1>) -> ()
// CHECK:     }, {
// CHECK:     ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>):
// CHECK:     %1 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<i32>
// CHECK:     "tfl.yield"(%1, %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
// CHECK:     }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)

// -----

// CHECK-LABEL: while_with_reduce
func.func @while_with_reduce(%arg0: tensor<1x256xf32>, %arg1: tensor<1xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1xf32>) {
  %cst = arith.constant dense<1> : tensor<i32>
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %cst_1 = arith.constant dense<1000> : tensor<i32>
  %0:5 = "stablehlo.while"(%cst_0, %cst, %cst_1, %arg0 , %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<1x256xf32>, %arg6: tensor<1xf32>):
    %1 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<1x256xf32>, %arg6: tensor<1xf32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "stablehlo.reduce"(%arg5, %2) ({
      ^bb0(%arg7: tensor<f32>, %arg8: tensor<f32>):
        %4 = stablehlo.add %arg7, %arg8 : tensor<f32>
        "stablehlo.return"(%4) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
    %5  = stablehlo.add %3, %arg6 : tensor<1xf32>
    "stablehlo.return"(%1, %arg3, %arg4, %arg5, %5) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x256xf32>, tensor<1xf32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x256xf32>, tensor<1xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x256xf32>, tensor<1xf32>)
  func.return %0#0, %0#1, %0#2, %0#4: tensor<i32>, tensor<i32>, tensor<i32>, tensor<1xf32>
}

// CHECK-DAG: %cst = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %cst_0 = arith.constant dense<1> : tensor<i32>
// CHECK-DAG: %cst_1 = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %cst_2 = arith.constant dense<1000> : tensor<i32>
// CHECK:     %0:5 = "tfl.while"(%cst_1, %cst_0, %cst_2, %arg0, %arg1) <{is_stateless = false}> ({
// CHECK:     ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<1x256xf32>, %arg6: tensor<1xf32>):
// CHECK:       %1 = tfl.less(%arg2, %arg4) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:       "tfl.yield"(%1) : (tensor<i1>) -> ()
// CHECK:     ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<1x256xf32>, %arg6: tensor<1xf32>):
// CHECK:       %1 = tfl.add %arg2, %arg3 {fused_activation_function = "NONE"} : tensor<i32>
// CHECK:       %cst_3 = arith.constant dense<1> : tensor<1xi32>
// CHECK:       %2 = "tfl.sum"(%arg5, %cst_3) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>
// CHECK:       %3 = tfl.add %2, %arg6 {fused_activation_function = "NONE"} : tensor<1xf32>
// CHECK:       "tfl.yield"(%1, %arg3, %arg4, %arg5, %3) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x256xf32>, tensor<1xf32>) -> ()
// CHECK:     }) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x256xf32>, tensor<1xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x256xf32>, tensor<1xf32>)

// -----

//===----------------------------------------------------------------------===//
// data movement and shaping
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @get_dimension_size(
func.func @get_dimension_size(%arg0: tensor<4x256x?xf32>) -> tensor<i32> {
  %0 = "stablehlo.get_dimension_size"(%arg0) <{dimension = 1 : i64}> : (tensor<4x256x?xf32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK: %[[SHAPE:.*]] = "tfl.shape"(%arg0)
// CHECK: %[[SLICE:.*]] = "tfl.slice"(%[[SHAPE]], %{{.*}}, %{{.*}})
// CHECK: %[[SQUEEZE:.*]] = "tfl.squeeze"(%{{.*}})
// CHECK: return %[[SQUEEZE]] : tensor<i32>

// -----

// CHECK-LABEL: reverse
func.func @reverse(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "stablehlo.reverse"(%arg0) <{dimensions = array<i64: 0>}> : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<1xi32>
// CHECK: %[[REV:.*]] = "tfl.reverse_v2"(%arg0, %[[CST]]) : (tensor<3x2xi32>, tensor<1xi32>) -> tensor<3x2xi32>
// CHECK: return %[[REV]] : tensor<3x2xi32>

// -----

// CHECK-LABEL: concat
func.func @concat(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) <{dimension = 0 : i64}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  func.return %0 : tensor<6x3xf32>
}

// CHECK: %0 = "tfl.concatenation"(%arg0, %arg1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>

// -----

// CHECK-LABEL: broadcast_in_dim_tfl_style
func.func @broadcast_in_dim_tfl_style(%arg0: tensor<8x1x16xf32>) -> tensor<3x8x8x16xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 1, 2, 3>, name = "broadcast.0"}> : (tensor<8x1x16xf32>) -> tensor<3x8x8x16xf32>
  func.return %0 : tensor<3x8x8x16xf32>
}

// CHECK: %cst = arith.constant dense<[3, 8, 8, 16]> : tensor<4xi64>
// CHECK: %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<8x1x16xf32>, tensor<4xi64>) -> tensor<3x8x8x16xf32>

// -----

// CHECK-LABEL: broadcast_in_dim_general_case
func.func @broadcast_in_dim_general_case(%arg0: tensor<3x1x16xf32>) -> tensor<3x8x8x16xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 2, 3>, name = "broadcast.0"}> : (tensor<3x1x16xf32>) -> tensor<3x8x8x16xf32>
  func.return %0 : tensor<3x8x8x16xf32>
}

// CHECK: %cst = arith.constant dense<[3, 1, 1, 16]> : tensor<4xi32>
// CHECK: %0 = "tfl.reshape"(%arg0, %cst) : (tensor<3x1x16xf32>, tensor<4xi32>) -> tensor<3x1x1x16xf32>
// CHECK: %cst_0 = arith.constant dense<[3, 8, 8, 16]> : tensor<4xi64>
// CHECK: %1 = "tfl.broadcast_to"(%0, %cst_0) : (tensor<3x1x1x16xf32>, tensor<4xi64>) -> tensor<3x8x8x16xf32>

// -----

// CHECK-LABEL: dynamic_broadcast_in_dim_tf_style
func.func @dynamic_broadcast_in_dim_tf_style(%arg0: tensor<?x1x1x2x1xf32>, %arg1: tensor<5xi32>) -> tensor<?x750x1x2x384xf32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) <{broadcast_dimensions = array<i64: 0, 1, 2, 3, 4>}> : (tensor<?x1x1x2x1xf32>, tensor<5xi32>) -> tensor<?x750x1x2x384xf32>
  func.return %0 : tensor<?x750x1x2x384xf32>
}

// CHECK: %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<?x1x1x2x1xf32>, tensor<5xi32>) -> tensor<?x750x1x2x384xf32>

// -----

// CHECK-LABEL: dynamic_broadcast_in_dim_general_case_expand_back_dims
func.func @dynamic_broadcast_in_dim_general_case_expand_back_dims(%arg0: tensor<?x3000xf32>, %arg1: tensor<4xi32>) -> tensor<?x3000x2x4xf32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<?x3000xf32>, tensor<4xi32>) -> tensor<?x3000x2x4xf32>
  func.return %0 : tensor<?x3000x2x4xf32>
}

// CHECK: %cst = arith.constant dense<2> : tensor<i64>
// CHECK: %0 = "tfl.expand_dims"(%arg0, %cst) : (tensor<?x3000xf32>, tensor<i64>) -> tensor<?x3000x1xf32>
// CHECK: %cst_0 = arith.constant dense<3> : tensor<i64>
// CHECK: %1 = "tfl.expand_dims"(%0, %cst_0) : (tensor<?x3000x1xf32>, tensor<i64>) -> tensor<?x3000x1x1xf32>
// CHECK: %2 = "tfl.broadcast_to"(%1, %arg1) : (tensor<?x3000x1x1xf32>, tensor<4xi32>) -> tensor<?x3000x2x4xf32>


// -----

//===----------------------------------------------------------------------===//
// stablehlo.case
//===----------------------------------------------------------------------===//

// CHECK-LABEL: case_func
func.func @case_func(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<i32>) {
  %0 = "stablehlo.case"(%arg0) ({
    %2 = stablehlo.add %arg1, %arg2 : tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  }, {
    %2 = stablehlo.multiply %arg1, %arg1 : tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>
  func.return %0: tensor<i32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK: %[[PRED:.*]] = tfl.not_equal(%arg0, %[[CST]]) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK: %[[IF:.*]] = "tfl.if"(%[[PRED]]) ({
// CHECK:   %[[MUL:.*]] = tfl.mul %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<i32>
// CHECK:   "tfl.yield"(%[[MUL]]) : (tensor<i32>) -> ()
// CHECK: }, {
// CHECK:   %[[ADD:.*]] = tfl.add %arg1, %arg2 {fused_activation_function = "NONE"} : tensor<i32>
// CHECK:   "tfl.yield"(%[[ADD]]) : (tensor<i32>) -> ()
// CHECK: }) : (tensor<i1>) -> tensor<i32>
// CHECK: return %[[IF]] : tensor<i32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.if
//===----------------------------------------------------------------------===//

// CHECK-LABEL: if_label
func.func @if_label(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<i32>) {
  %0 = stablehlo.add %arg1, %arg2 : tensor<i32>
  %1 = "stablehlo.if"(%arg0) ({
    "stablehlo.return"(%0) : (tensor<i32>) -> ()
  }, {
    %2 = stablehlo.multiply %arg1, %arg1 : tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  }) : (tensor<i1>) -> tensor<i32>
  func.return %1: tensor<i32>
}

// CHECK: %0 = tfl.add %arg1, %arg2 {fused_activation_function = "NONE"} : tensor<i32>
// CHECK: %1 = "tfl.if"(%arg0) ({
// CHECK:   "tfl.yield"(%0) : (tensor<i32>) -> ()
// CHECK: }, {
// CHECK:   %2 = tfl.mul %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<i32>
// CHECK:   "tfl.yield"(%2) : (tensor<i32>) -> ()
// CHECK: }) : (tensor<i1>) -> tensor<i32>
// CHECK: return %1 : tensor<i32>

// -----

//===----------------------------------------------------------------------===//
// stablehlo.fft
//===----------------------------------------------------------------------===//

// CHECK-LABEL: legalize_fft_mhlo_fft_type_rfft
func.func @legalize_fft_mhlo_fft_type_rfft(%arg0 : tensor<1x1x512xf32>) -> tensor<1x1x257xcomplex<f32>> {
  %270 = "stablehlo.fft"(%arg0) <{fft_length = array<i64: 1, 512>, fft_type = #stablehlo<fft_type RFFT>}> : (tensor<1x1x512xf32>) -> tensor<1x1x257xcomplex<f32>>
  return %270: tensor<1x1x257xcomplex<f32>>
}

// CHECK:  %cst = arith.constant dense<[1, 512]> : tensor<2xi32>
// CHECK:  %0 = "tfl.rfft2d"(%arg0, %cst) : (tensor<1x1x512xf32>, tensor<2xi32>) -> tensor<1x1x257xcomplex<f32>>
// CHECK:  return %0 : tensor<1x1x257xcomplex<f32>>

// -----

// CHECK-LABEL: @fft
func.func @fft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) <{ fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type FFT> }> : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// CHECK: stablehlo.fft
// CHECK: return %{{.*}} : tensor<3x9xcomplex<f32>>

// -----

// CHECK-LABEL: @mhlo_nd_fft
func.func @mhlo_nd_fft(%arg0: tensor<2x3x345x1x256xf32>) -> tensor<2x3x345x1x129xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) <{fft_length = array<i64: 1, 256>, fft_type = #stablehlo<fft_type RFFT>}> : (tensor<2x3x345x1x256xf32>) -> tensor<2x3x345x1x129xcomplex<f32>>
  func.return %0 : tensor<2x3x345x1x129xcomplex<f32>>
}

// CHECK: %cst = arith.constant dense<[1, 256]> : tensor<2xi32>
// CHECK: %0 = "tfl.rfft2d"(%arg0, %cst) : (tensor<2x3x345x1x256xf32>, tensor<2xi32>) -> tensor<2x3x345x1x129xcomplex<f32>>
// CHECK: return %0 : tensor<2x3x345x1x129xcomplex<f32>>

// -----

// CHECK-LABEL: @mhlo_nd_fft_1
func.func @mhlo_nd_fft_1(%arg0: tensor<2x3x345x4x256xf32>) -> tensor<2x3x345x4x129xcomplex<f32>> {
  %43 = "stablehlo.fft"(%arg0) <{fft_length = array<i64: 4, 256>, fft_type = #stablehlo<fft_type RFFT>}> : (tensor<2x3x345x4x256xf32>) -> tensor<2x3x345x4x129xcomplex<f32>>
  return %43 : tensor<2x3x345x4x129xcomplex<f32>>
}

// CHECK: %cst = arith.constant dense<[4, 256]> : tensor<2xi32>
// CHECK: %0 = "tfl.rfft2d"(%arg0, %cst) : (tensor<2x3x345x4x256xf32>, tensor<2xi32>) -> tensor<2x3x345x4x129xcomplex<f32>>
// CHECK: return %0 : tensor<2x3x345x4x129xcomplex<f32>>

// -----

// CHECK-LABEL: @mhlo_dynamic_fft_1
func.func @mhlo_dynamic_fft_1(%arg0: tensor<?x9x1x2560xf32>) -> tensor<?x9x1x1281xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) <{fft_length = array<i64: 1, 2560>, fft_type = #stablehlo<fft_type RFFT>}> : (tensor<?x9x1x2560xf32>) -> tensor<?x9x1x1281xcomplex<f32>>
  return %0 : tensor<?x9x1x1281xcomplex<f32>>
  // CHECK: %cst = arith.constant dense<[1, 2560]> : tensor<2xi32>
  // CHECK: %0 = "tfl.rfft2d"(%arg0, %cst) : (tensor<?x9x1x2560xf32>, tensor<2xi32>) -> tensor<?x9x1x1281xcomplex<f32>>
  // CHECK: return %0 : tensor<?x9x1x1281xcomplex<f32>>
}

// -----

// CHECK-LABEL: @mhlo_dynamic_fft_2
func.func @mhlo_dynamic_fft_2(%arg0: tensor<?x?x1x2560xf32>) -> tensor<?x?x1x1281xcomplex<f32>> {
  %9 = "stablehlo.fft"(%arg0) <{fft_length = array<i64: 1, 2560>, fft_type = #stablehlo<fft_type RFFT>}> : (tensor<?x?x1x2560xf32>) -> tensor<?x?x1x1281xcomplex<f32>>
  return %9 : tensor<?x?x1x1281xcomplex<f32>>
  // CHECK: %cst = arith.constant dense<[1, 2560]> : tensor<2xi32>
  // CHECK: %0 = "tfl.rfft2d"(%arg0, %cst) : (tensor<?x?x1x2560xf32>, tensor<2xi32>) -> tensor<?x?x1x1281xcomplex<f32>>
  // CHECK: return %0 : tensor<?x?x1x1281xcomplex<f32>>
}

//===----------------------------------------------------------------------===//
// stablehlo.imag
//===----------------------------------------------------------------------===//

func.func @mhlo_imag_test1(%arg0: tensor<1x2049xcomplex<f32>>) -> tensor<1x2049xf32> {
  %339 = stablehlo.imag %arg0 : (tensor<1x2049xcomplex<f32>>) -> tensor<1x2049xf32>
  return %339 : tensor<1x2049xf32>
  // CHECK:  %0 = "tfl.imag"(%arg0) : (tensor<1x2049xcomplex<f32>>) -> tensor<1x2049xf32>
  // CHECK:  return %0 : tensor<1x2049xf32>
}

func.func @mhlo_imag_test_neg(%arg0: tensor<1x2049xf32>) -> tensor<1x2049xf32> {
  %339 = stablehlo.imag %arg0 : (tensor<1x2049xf32>) -> tensor<1x2049xf32>
  return %339 : tensor<1x2049xf32>
  // CHECK:  %0 = stablehlo.imag %arg0 : tensor<1x2049xf32>
  // CHECK:  return %0 : tensor<1x2049xf32>
}

// -----

//===----------------------------------------------------------------------===//
// stablehlo.real
//===----------------------------------------------------------------------===//

func.func @mhlo_real_test1(%arg0: tensor<1x2049xcomplex<f32>>) -> tensor<1x2049xf32> {
  %339 = stablehlo.real %arg0 : (tensor<1x2049xcomplex<f32>>) -> tensor<1x2049xf32>
  return %339 : tensor<1x2049xf32>
  // CHECK:  %0 = "tfl.real"(%arg0) : (tensor<1x2049xcomplex<f32>>) -> tensor<1x2049xf32>
  // CHECK:  return %0 : tensor<1x2049xf32>
}

func.func @mhlo_real_test_neg(%arg0: tensor<1x2049xf32>) -> tensor<1x2049xf32> {
  %339 = stablehlo.real %arg0 : (tensor<1x2049xf32>) -> tensor<1x2049xf32>
  return %339 : tensor<1x2049xf32>
  // CHECK:  %0 = stablehlo.real %arg0 : tensor<1x2049xf32>
  // CHECK:  return %0 : tensor<1x2049xf32>
}
