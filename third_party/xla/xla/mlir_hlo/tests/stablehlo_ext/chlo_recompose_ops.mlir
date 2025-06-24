// RUN: mlir-hlo-opt --stablehlo-ext-chlo-recompose-ops --symbol-dce --split-input-file --verify-diagnostics %s | FileCheck %s

/////
// Composite Recomposition

// CHECK-LABEL: func @erf_recompose_composite
func.func @erf_recompose_composite(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-NEXT: chlo.erf
  // CHECK-NOT: stablehlo.composite
  %0 = stablehlo.composite "chlo.erf" %arg0 {decomposition = @chlo.erf.impl, version = 1 : i32} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}
// CHECK-NOT: @chlo.erf.imp
func.func private @chlo.erf.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.erf %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
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

/////
// (Deprecated) CustomCall Recomposition

// CHECK-LABEL: @erf_recompose_cc
func.func @erf_recompose_cc(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: %0 = chlo.erf %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  %0 = "stablehlo.custom_call"(%arg0) {
    backend_config = "",
    call_target_name = "mhlo.erf",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  func.return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @ragged_dot_recompose_cc
func.func @ragged_dot_recompose_cc(%arg0: tensor<2x11x5xf32>, %arg1: tensor<3x2x5x7xf32>, %arg2: tensor<3xi64>) -> tensor<2x11x7xf32> {
  // CHECK: "chlo.ragged_dot"(%arg0, %arg1, %arg2) <{precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], ragged_dot_dimension_numbers = #chlo.ragged_dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2], lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>}> : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  // CHECK-NOT: stablehlo.custom_call
  %0 = stablehlo.custom_call @chlo.ragged_dot(%arg0, %arg1, %arg2) {mhlo.attributes = {precision_config = ["DEFAULT", "DEFAULT"], ragged_dot_dimension_numbers = [dense<0> : tensor<1xi64>, dense<1> : tensor<1xi64>, dense<2> : tensor<1xi64>, dense<2> : tensor<1xi64>, dense<1> : tensor<1xi64>, dense<0> : tensor<1xi64>]}, mhlo.version = 1 : i64} : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32> 
  return %0 : tensor<2x11x7xf32>
}

// -----

// CHECK-LABEL: @tan_recompose_cc
func.func @tan_recompose_cc(%arg0: tensor<16xf32>) -> tensor<?xf32> {
  // CHECK: %0 = chlo.tan %arg0 : tensor<16xf32> -> tensor<?xf32>
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "mhlo.tan",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<16xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @topk_recompose_cc
func.func @topk_recompose_cc(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // CHECK: %values, %indices = chlo.top_k(%arg0, k = 4) {largest = true} : tensor<5x16xf32> -> (tensor<?x?xf32>, tensor<?x?xi32>)
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @topk_no_recompose_invalid_attr
func.func @topk_no_recompose_invalid_attr(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // CHECK: stablehlo.custom_call @mhlo.topk
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = false}
  } : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}
