// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
// RUN: odml-to-stablehlo-opt %s -tfl-chlo-recompose-ops -symbol-dce -split-input-file -verify-diagnostics | FileCheck %s

/////
// Composite Recomposition

// CHECK-LABEL: func @erf_recompose_composite
func.func @erf_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.erf
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.erf" %arg0 {decomposition = @chlo.erf.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.erf.impl
func.func private @chlo.erf.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.erf %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @erfc_recompose_composite
func.func @erfc_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.erfc
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.erfc" %arg0 {decomposition = @chlo.erfc.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.erfc.impl
func.func private @chlo.erfc.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.erfc %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @acosh_recompose_composite
func.func @acosh_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.acosh
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.acosh" %arg0 {decomposition = @chlo.acosh.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.acosh.impl
func.func private @chlo.acosh.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.acosh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @atanh_recompose_composite
func.func @atanh_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.atanh
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.atanh" %arg0 {decomposition = @chlo.atanh.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.atanh.impl
func.func private @chlo.atanh.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.atanh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @acos_recompose_composite
func.func @acos_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.acos
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.acos" %arg0 {decomposition = @chlo.acos.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.acos.impl
func.func private @chlo.acos.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.acos %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @cosh_recompose_composite
func.func @cosh_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.cosh
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.cosh" %arg0 {decomposition = @chlo.cosh.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.cosh.impl
func.func private @chlo.cosh.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.cosh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @sinh_recompose_composite
func.func @sinh_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.sinh
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.sinh" %arg0 {decomposition = @chlo.sinh.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.sinh.impl
func.func private @chlo.sinh.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.sinh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @asin_recompose_composite
func.func @asin_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.asin
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.asin" %arg0 {decomposition = @chlo.asin.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.asin.impl
func.func private @chlo.asin.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.asin %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @asinh_recompose_composite
func.func @asinh_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.asinh
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.asinh" %arg0 {decomposition = @chlo.asinh.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.asinh.impl
func.func private @chlo.asinh.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.asinh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @mulhi_recompose_composite
func.func @mulhi_recompose_composite(%arg0: tensor<3x20x20xi32>, %arg1: tensor<3x20x20xi32>) -> tensor<?x20x20xi32> {
  // CHECK-NEXT: chlo.mulhi
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.mulhi" %arg0, %arg1 {decomposition = @chlo.mulhi.impl, version = 1 : i32} : (tensor<3x20x20xi32>, tensor<3x20x20xi32>) -> tensor<?x20x20xi32>
  return %0 : tensor<?x20x20xi32>
}
// CHECK-NOT: @chlo.mulhi.impl
func.func private @chlo.mulhi.impl(%arg0: tensor<3x20x20xi32>, %arg1: tensor<3x20x20xi32>) -> tensor<?x20x20xi32> {
  %0 = chlo.mulhi %arg0, %arg1 : tensor<3x20x20xi32>, tensor<3x20x20xi32> -> tensor<?x20x20xi32>
  return %0 : tensor<?x20x20xi32>
}

// -----

// CHECK-LABEL: func @ragged_dot_recompose_composite
func.func @ragged_dot_recompose_composite(%arg0: tensor<2x11x5xf32>, %arg1: tensor<3x2x5x7xf32>, %arg2: tensor<3xi64>) -> tensor<2x11x7xf32> {
  // CHECK: "chlo.ragged_dot"(%arg0, %arg1, %arg2) <{precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], ragged_dot_dimension_numbers = #chlo.ragged_dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2], lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}> : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.ragged_dot" %arg0, %arg1, %arg2 {composite_attributes = {precision_config = ["DEFAULT", "DEFAULT"], ragged_dot_dimension_numbers = [dense<0> : tensor<1xi64>, dense<1> : tensor<1xi64>, dense<2> : tensor<1xi64>, dense<2> : tensor<1xi64>, dense<1> : tensor<1xi64>, dense<0> : tensor<1xi64>]}, decomposition = @chlo.ragged_dot.impl, version = 1 : i32} : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  return %0 : tensor<2x11x7xf32>
}
// CHECK-NOT: @chlo.ragged_dot.impl
func.func private @chlo.ragged_dot.impl(%arg0: tensor<2x11x5xf32>, %arg1: tensor<3x2x5x7xf32>, %arg2: tensor<3xi64>) -> tensor<2x11x7xf32> {
  %0 = "chlo.ragged_dot"(%arg0, %arg1, %arg2) <{precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], ragged_dot_dimension_numbers = #chlo.ragged_dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2], lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}> : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  return %0 : tensor<2x11x7xf32>
}

// -----

// CHECK-LABEL: func @topk_recompose_composite
func.func @topk_recompose_composite(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // CHECK-NEXT: chlo.top_k
  // CHECK-NOT: stablehlo.composite
  %0:2 = stablehlo.composite "chlo.top_k" %arg0 {composite_attributes = {k = 4 : i64, largest = true}, decomposition = @chlo.top_k.impl, version = 1 : i32} : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}
// CHECK-NOT: @chlo.top_k.impl
func.func private @chlo.top_k.impl(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  %values, %indices = chlo.top_k(%arg0, k = 4) {largest = true} : tensor<5x16xf32> -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %values, %indices : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @topk_recompose_composite_unstable
func.func @topk_recompose_composite_unstable(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // CHECK: %{{.*}}, %{{.*}} = chlo.top_k(%arg0, k = 4, is_stable = false) {largest = true}
  // CHECK-NOT: stablehlo.composite
  %0:2 = stablehlo.composite "chlo.top_k" %arg0 {composite_attributes = {is_stable = false, k = 4 : i64, largest = true}, decomposition = @chlo.top_k.unstable.impl, version = 1 : i32} : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}
// CHECK-NOT: @chlo.top_k.unstable.impl
func.func private @chlo.top_k.unstable.impl(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  %values, %indices = chlo.top_k(%arg0, k = 4, is_stable = false) {largest = true} : tensor<5x16xf32> -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %values, %indices : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @scan_recompose_composite
func.func public @scan_recompose_composite(%arg0: tensor<2xf64>, %arg1: tensor<4xf64>, %arg2: tensor<5x3xf64>) -> (tensor<4xf64>, tensor<5xf64>) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2xf64>) -> tensor<5x2xf64>
  // CHECK: chlo.scan(%0, %arg2) inits (%arg1) dimension=0
  // CHECK-NOT: stablehlo.composite
  %1:2 = stablehlo.composite "chlo.scan" %0, %arg2, %arg1 ({
  ^bb0(%b0: tensor<2xf64>, %b1: tensor<3xf64>, %b2: tensor<4xf64>):
    %2 = stablehlo.add %b2, %b2 : tensor<4xf64>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    stablehlo.return %3, %2 : tensor<f64>, tensor<4xf64>
  }) {
    composite_attributes = {
      dimension = 0 : i64,
      operandSegmentSizes = array<i32: 2, 1>,
      resultSegmentSizes = array<i32: 1, 1>
    },
    decomposition = @chlo.scan.impl,
    version = 1 : i32
  } : (tensor<5x2xf64>, tensor<5x3xf64>, tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>)
  return %1#1, %1#0 : tensor<4xf64>, tensor<5xf64>
}

// CHECK-NOT: @chlo.scan.impl
func.func private @chlo.scan.impl(%arg0: tensor<5x2xf64>, %arg1: tensor<5x3xf64>, %arg2: tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>) {
  %0:2 = chlo.scan(%arg0, %arg1) inits(%arg2) dimension=0 {
  ^bb0(%b0: tensor<2xf64>, %b1: tensor<3xf64>, %b2: tensor<4xf64>):
    %1 = stablehlo.add %b2, %b2 : tensor<4xf64>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    stablehlo.return %2, %1 : tensor<f64>, tensor<4xf64>
  } : (tensor<5x2xf64>, tensor<5x3xf64>, tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>)
  return %0#0, %0#1 : tensor<5xf64>, tensor<4xf64>
}

// -----

// A composite whose number of regions differs from that of the CHLO op is not
// recomposed.
// CHECK-LABEL: func @erf_composite_unexpected_region
func.func @erf_composite_unexpected_region(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK: stablehlo.composite "chlo.erf"
  %0 = stablehlo.composite "chlo.erf" %arg0 ({
  ^bb0(%b0: tensor<3xf32>):
    stablehlo.return %b0 : tensor<3xf32>
  }) {decomposition = @chlo.erf.impl, version = 1 : i32} : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
func.func private @chlo.erf.impl(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = chlo.erf %arg0 : tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

// CHECK-LABEL: func @scan_composite_missing_region
func.func @scan_composite_missing_region(%arg0: tensor<5x2xf64>, %arg1: tensor<5x3xf64>, %arg2: tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>) {
  // CHECK: stablehlo.composite "chlo.scan"
  %0:2 = stablehlo.composite "chlo.scan" %arg0, %arg1, %arg2 {
    composite_attributes = {
      dimension = 0 : i64,
      operandSegmentSizes = array<i32: 2, 1>,
      resultSegmentSizes = array<i32: 1, 1>
    },
    decomposition = @chlo.scan.impl,
    version = 1 : i32
  } : (tensor<5x2xf64>, tensor<5x3xf64>, tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>)
  return %0#0, %0#1 : tensor<5xf64>, tensor<4xf64>
}
func.func private @chlo.scan.impl(%arg0: tensor<5x2xf64>, %arg1: tensor<5x3xf64>, %arg2: tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>) {
  %0:2 = chlo.scan(%arg0, %arg1) inits(%arg2) dimension=0 {
  ^bb0(%b0: tensor<2xf64>, %b1: tensor<3xf64>, %b2: tensor<4xf64>):
    %1 = stablehlo.add %b2, %b2 : tensor<4xf64>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    stablehlo.return %2, %1 : tensor<f64>, tensor<4xf64>
  } : (tensor<5x2xf64>, tensor<5x3xf64>, tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>)
  return %0#0, %0#1 : tensor<5xf64>, tensor<4xf64>
}

// -----

// Composites with a version other than 1 are left alone.
// CHECK-LABEL: func @erf_composite_other_version
func.func @erf_composite_other_version(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK: stablehlo.composite "chlo.erf"
  %0 = stablehlo.composite "chlo.erf" %arg0 {decomposition = @chlo.erf.impl, version = 2 : i32} : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
func.func private @chlo.erf.impl(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = chlo.erf %arg0 : tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

// Other composites, including CHLO ops that are not preserved as composites,
// are left alone.
// CHECK-LABEL: func @other_composites
func.func @other_composites(%arg0: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
  // CHECK: stablehlo.composite "chlo.tan"
  // CHECK: stablehlo.composite "odml.erf"
  %0 = stablehlo.composite "chlo.tan" %arg0 {decomposition = @tan_impl, version = 1 : i32} : (tensor<3xf32>) -> tensor<3xf32>
  %1 = stablehlo.composite "odml.erf" %arg0 {decomposition = @erf_impl, version = 1 : i32} : (tensor<3xf32>) -> tensor<3xf32>
  return %0, %1 : tensor<3xf32>, tensor<3xf32>
}
func.func private @tan_impl(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = chlo.tan %arg0 : tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
func.func private @erf_impl(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = chlo.erf %arg0 : tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

// Malformed attributes are reported and the composite is left alone.
// CHECK-LABEL: func @ragged_dot_invalid_precision
func.func @ragged_dot_invalid_precision(%arg0: tensor<2x11x5xf32>, %arg1: tensor<3x2x5x7xf32>, %arg2: tensor<3xi64>) -> tensor<2x11x7xf32> {
  // CHECK: stablehlo.composite "chlo.ragged_dot"
  // expected-error @+1 {{invalid precision string}}
  %0 = stablehlo.composite "chlo.ragged_dot" %arg0, %arg1, %arg2 {composite_attributes = {precision_config = ["BOGUS", "DEFAULT"], ragged_dot_dimension_numbers = [dense<0> : tensor<1xi64>, dense<1> : tensor<1xi64>, dense<2> : tensor<1xi64>, dense<2> : tensor<1xi64>, dense<1> : tensor<1xi64>, dense<0> : tensor<1xi64>]}, decomposition = @ragged_dot_impl, version = 1 : i32} : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  return %0 : tensor<2x11x7xf32>
}
func.func private @ragged_dot_impl(%arg0: tensor<2x11x5xf32>, %arg1: tensor<3x2x5x7xf32>, %arg2: tensor<3xi64>) -> tensor<2x11x7xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x11x7xf32>
  return %0 : tensor<2x11x7xf32>
}

// -----

// CHECK-LABEL: func @ragged_dot_invalid_dimension_numbers
func.func @ragged_dot_invalid_dimension_numbers(%arg0: tensor<2x11x5xf32>, %arg1: tensor<3x2x5x7xf32>, %arg2: tensor<3xi64>) -> tensor<2x11x7xf32> {
  // CHECK: stablehlo.composite "chlo.ragged_dot"
  // expected-error @+1 {{ragged_dot_dimension_numbers is not an ArrayAttr with 6 elements}}
  %0 = stablehlo.composite "chlo.ragged_dot" %arg0, %arg1, %arg2 {composite_attributes = {precision_config = ["DEFAULT", "DEFAULT"], ragged_dot_dimension_numbers = [dense<0> : tensor<1xi64>, dense<1> : tensor<1xi64>, dense<2> : tensor<1xi64>, dense<2> : tensor<1xi64>, dense<1> : tensor<1xi64>]}, decomposition = @ragged_dot_impl, version = 1 : i32} : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  return %0 : tensor<2x11x7xf32>
}
func.func private @ragged_dot_impl(%arg0: tensor<2x11x5xf32>, %arg1: tensor<3x2x5x7xf32>, %arg2: tensor<3xi64>) -> tensor<2x11x7xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<2x11x7xf32>
  return %0 : tensor<2x11x7xf32>
}

// -----

// CHECK-LABEL: func @top_k_non_builtin_attribute
func.func @top_k_non_builtin_attribute(%arg0: tensor<16xf32>) -> (tensor<4xf32>, tensor<4xi32>) {
  // CHECK: stablehlo.composite "chlo.top_k"
  // expected-error @+1 {{unsupported attribute for chlo recompose:}}
  %0:2 = stablehlo.composite "chlo.top_k" %arg0 {composite_attributes = {k = 4 : i64, largest = true, direction = #stablehlo<comparison_direction GT>}, decomposition = @top_k_impl, version = 1 : i32} : (tensor<16xf32>) -> (tensor<4xf32>, tensor<4xi32>)
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xi32>
}
func.func private @top_k_impl(%arg0: tensor<16xf32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %0 = stablehlo.slice %arg0 [0:4] : (tensor<16xf32>) -> tensor<4xf32>
  %1 = stablehlo.iota dim = 0 : tensor<4xi32>
  return %0, %1 : tensor<4xf32>, tensor<4xi32>
}
