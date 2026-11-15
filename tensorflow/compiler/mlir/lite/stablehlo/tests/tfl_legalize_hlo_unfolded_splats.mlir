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
// RUN: odml-to-stablehlo-opt %s -unfold-splat-constant-pass -tfl-legalize-hlo -split-input-file | FileCheck %s --dump-input=fail

// The converter pipelines run -unfold-splat-constant-pass right before
// -tfl-legalize-hlo, so the fused rounding patterns see splat constants as
// broadcasts of scalars. They must still match.

// CHECK-LABEL: round
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

// CHECK:     %[[ROUND:.*]] = "tfl.round"(%arg0) : (tensor<8x128xf32>) -> tensor<8x128xf32>
// CHECK-NOT: tfl.select
// CHECK:     return %[[ROUND]]

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

// CHECK:     %[[FLOOR_MOD:.*]] = "tfl.floor_mod"(%arg0, %{{.*}})
// CHECK-NOT: tfl.select
// CHECK:     return %[[FLOOR_MOD]]

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
  func.return %12 : tensor<10x10xf32>
}

// CHECK: %[[FLOOR_DIV:.*]] = tfl.floor_div %arg0, %arg1 : tensor<10x10xf32>
// CHECK: return %[[FLOOR_DIV]]

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
  func.return %14 : tensor<10x10xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<10x10xf32>
// CHECK: %[[FLOOR_DIV:.*]] = tfl.floor_div %arg0, %[[CST]] : tensor<10x10xf32>
// CHECK: return %[[FLOOR_DIV]]

// -----

// Splat constant operands that can't take an implicit broadcast stay
// constants, e.g. a composite's scale.

// CHECK-LABEL: composite_splat_operand
func.func @composite_splat_operand(%arg0: tensor<1x8x16xf32>) -> tensor<1x8x16xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<16xf32>
  %0 = stablehlo.composite "odml.rms_norm" %arg0, %cst {composite_attributes = {epsilon = 9.99999997E-7 : f32}, decomposition = @composite_splat_operand_impl} : (tensor<1x8x16xf32>, tensor<16xf32>) -> tensor<1x8x16xf32>
  func.return %0 : tensor<1x8x16xf32>
}
func.func private @composite_splat_operand_impl(%arg0: tensor<1x8x16xf32>, %arg1: tensor<16xf32>) -> tensor<1x8x16xf32> {
  %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<16xf32>) -> tensor<1x8x16xf32>
  %1 = stablehlo.multiply %arg0, %0 : tensor<1x8x16xf32>
  func.return %1 : tensor<1x8x16xf32>
}

// CHECK:     %[[SCALE:.*]] = arith.constant dense<1.000000e+00> : tensor<16xf32>
// CHECK-NOT: tfl.broadcast_to
// CHECK:     stablehlo.composite "odml.rms_norm" %arg0, %[[SCALE]]
