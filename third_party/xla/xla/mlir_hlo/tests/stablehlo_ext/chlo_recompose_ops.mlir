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
// CHECK-NOT: @chlo.erf.impl
func.func private @chlo.erf.impl(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  %0 = chlo.erf %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
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
// CHECK-NOT: @chlo.atanh.imp
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

// CHECK-LABEL: @scan_recompose_composite
func.func public @scan_recompose_composite(%arg0: tensor<2xf64>, %arg1: tensor<4xf64>, %arg2: tensor<5x3xf64>) -> (tensor<4xf64>, tensor<5xf64>) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2xf64>) -> tensor<5x2xf64>
  // CHECK: chlo.scan(%0, %arg2) inits (%arg1) dimension=0
  // CHECK-NOT: stablehlo.composite
  %1:2 = stablehlo.composite "chlo.scan" %0, %arg2, %arg1 {
    composite_attributes = {
      dimension = 0 : i64
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

// CHECK-LABEL: @acosh_recompose_cc
func.func @acosh_recompose_cc(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: %0 = chlo.acosh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  %0 = "stablehlo.custom_call"(%arg0) {
    backend_config = "",
    call_target_name = "mhlo.acosh",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  func.return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: @acos_recompose_cc
func.func @acos_recompose_cc(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: %0 = chlo.acos %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  %0 = "stablehlo.custom_call"(%arg0) {
    backend_config = "",
    call_target_name = "mhlo.acos",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  func.return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: @atanh_recompose_cc
func.func @atanh_recompose_cc(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: %0 = chlo.atanh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  %0 = "stablehlo.custom_call"(%arg0) {
    backend_config = "",
    call_target_name = "mhlo.atanh",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  func.return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: @cosh_recompose_cc
func.func @cosh_recompose_cc(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: %0 = chlo.cosh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  %0 = "stablehlo.custom_call"(%arg0) {
    backend_config = "",
    call_target_name = "mhlo.cosh",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  func.return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: @asin_recompose_cc
func.func @asin_recompose_cc(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: %0 = chlo.asin %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  %0 = "stablehlo.custom_call"(%arg0) {
    backend_config = "",
    call_target_name = "mhlo.asin",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  func.return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: @asinh_recompose_cc
func.func @asinh_recompose_cc(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: %0 = chlo.asinh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  %0 = "stablehlo.custom_call"(%arg0) {
    backend_config = "",
    call_target_name = "mhlo.asinh",
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

// -----

// CHECK-LABEL: @scan_recompose_cc
func.func @scan_recompose_cc(%arg0: tensor<5x2xf64>, %arg1: tensor<5x3xf64>, %arg2: tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>) {
  // CHECK: chlo.scan(%arg0, %arg1) inits (%arg2) dimension=0
  %0:2 = stablehlo.custom_call @chlo.scan(%arg0, %arg1, %arg2) {
    called_computations = [@chlo.scan.impl],
    mhlo.attributes = {
      dimension = 0 : i64,
      operandSegmentSizes = array<i32: 2, 1>,
      resultSegmentSizes = array<i32: 1, 1>
    },
    mhlo.version = 1 : i64
  } : (tensor<5x2xf64>, tensor<5x3xf64>, tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>)
  return %0#0, %0#1 : tensor<5xf64>, tensor<4xf64>
}
func.func private @chlo.scan.impl(%arg0: tensor<2xf64>, %arg1: tensor<3xf64>, %arg2: tensor<4xf64>) -> (tensor<f64>, tensor<4xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.add %arg2, %arg2 : tensor<4xf64>
  func.return %cst, %0 : tensor<f64>, tensor<4xf64>
}
