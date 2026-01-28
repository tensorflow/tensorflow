// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-preserve-high-level-ops --split-input-file | FileCheck %s
// RUN: mlir-hlo-opt %s > %t.0
// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-preserve-high-level-ops --stablehlo-ext-chlo-recompose-ops --symbol-dce > %t.1
// RUN: diff %t.0 %t.1

// CHECK-LABEL: func @ragged_dot_to_composite
func.func @ragged_dot_to_composite(%lhs : tensor<2x11x5xf32>, %rhs : tensor<3x2x5x7xf32>, %group_sizes : tensor<3xi64>) -> (tensor<2x11x7xf32>) {
  // CHECK: stablehlo.composite "chlo.ragged_dot"
  // CHECK-SAME{LITERAL}: {composite_attributes = {precision_config = ["DEFAULT", "DEFAULT"]
  // CHECK-SAME: ragged_dot_dimension_numbers = [dense<0>{{.*}}, dense<1>{{.*}}, dense<2>{{.*}}, dense<2>{{.*}}, dense<1>{{.*}}, dense<0>{{.*}}]
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  func.return %0 : tensor<2x11x7xf32>
}
// CHECK-LABEL: func.func private @chlo.ragged_dot.impl

// -----

// CHECK-LABEL: func @multiple_ragged_dots_name_conflict
// CHECK: stablehlo.composite "chlo.ragged_dot"
// CHECK-LABEL: func.func private @chlo.ragged_dot.impl
// CHECK: chlo.ragged_dot
// CHECK-LABEL: func.func private @chlo.ragged_dot.impl_0
// CHECK: chlo.ragged_dot
func.func @multiple_ragged_dots_name_conflict(%lhs : tensor<2x11x5xf32>, %rhs : tensor<3x2x5x7xf32>, %group_sizes : tensor<3xi64>) -> (tensor<2x11x7xf32>, tensor<2x11x7xf32>) {
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  %1 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0], rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [1], rhs_group_dimensions = [0]>,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  func.return %0, %1 : tensor<2x11x7xf32>, tensor<2x11x7xf32>
}

// -----

// CHECK-LABEL: func @topk_preserve
func.func @topk_preserve(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // CHECK: stablehlo.composite "chlo.top_k" %arg0 {composite_attributes = {k = 4 : i64, largest = true}, decomposition = @chlo.top_k.impl, version = 1 : i32}
  %values, %indices = chlo.top_k(%arg0, k = 4) {largest = true} : tensor<5x16xf32> -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %values, %indices : tensor<?x?xf32>, tensor<?x?xi32>
}


// -----

// CHECK-LABEL: func @erf_preserve
func.func @erf_preserve(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: stablehlo.composite "chlo.erf" %arg0 {decomposition = @chlo.erf.impl, version = 1 : i32}
  %0 = chlo.erf %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @acosh_preserve
func.func @acosh_preserve(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: stablehlo.composite "chlo.acosh" %arg0 {decomposition = @chlo.acosh.impl, version = 1 : i32}
  %0 = chlo.acosh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @acos_preserve
func.func @acos_preserve(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: stablehlo.composite "chlo.acos" %arg0 {decomposition = @chlo.acos.impl, version = 1 : i32}
  %0 = chlo.acos %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @atanh_preserve
func.func @atanh_preserve(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: stablehlo.composite "chlo.atanh" %arg0 {decomposition = @chlo.atanh.impl, version = 1 : i32}
  %0 = chlo.atanh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @cosh_preserve
func.func @cosh_preserve(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: stablehlo.composite "chlo.cosh" %arg0 {decomposition = @chlo.cosh.impl, version = 1 : i32}
  %0 = chlo.cosh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @sinh_preserve
func.func @sinh_preserve(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: stablehlo.composite "chlo.sinh" %arg0 {decomposition = @chlo.sinh.impl, version = 1 : i32}
  %0 = chlo.sinh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @asin_preserve
func.func @asin_preserve(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: stablehlo.composite "chlo.asin" %arg0 {decomposition = @chlo.asin.impl, version = 1 : i32}
  %0 = chlo.asin %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @asinh_preserve
func.func @asinh_preserve(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK: stablehlo.composite "chlo.asinh" %arg0 {decomposition = @chlo.asinh.impl, version = 1 : i32}
  %0 = chlo.asinh %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
  return %0 : tensor<?x20x20xbf16>
}

// -----

// CHECK-LABEL: func @tan_no_preserve
func.func @tan_no_preserve(%arg0: tensor<16xf32>) -> tensor<?xf32> {
  // CHECK: chlo.tan
  %0 = chlo.tan %arg0 : tensor<16xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @unregistered_attrs_preserve
func.func @unregistered_attrs_preserve(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // CHECK: stablehlo.composite "chlo.top_k" %arg0 {composite_attributes = {k = 4 : i64, largest = true, mhlo.frontend_attributes = {foo = "true"}}, decomposition = @chlo.top_k.impl, version = 1 : i32}
  %values, %indices = chlo.top_k(%arg0, k = 4) {largest = true, mhlo.frontend_attributes = {foo = "true"}} : tensor<5x16xf32> -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %values, %indices : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @scan_preserve
func.func @scan_preserve(%arg0: tensor<5x2xf64>, %arg1: tensor<5x3xf64>, %arg2: tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>) {
  // CHECK: stablehlo.composite "chlo.scan" %arg0, %arg1, %arg2 {composite_attributes = {dimension = 0 : i64, operandSegmentSizes = array<i32: 2, 1>, resultSegmentSizes = array<i32: 1, 1>}, decomposition = @chlo.scan.impl, version = 1 : i32}
  %0:2 = chlo.scan(%arg0, %arg1) inits(%arg2) dimension=0 {
  ^bb0(%b0: tensor<2xf64>, %b1: tensor<3xf64>, %b2: tensor<4xf64>):
    %2 = stablehlo.constant dense<0.0> : tensor<f64>
    %1 = stablehlo.add %b2, %b2 : tensor<4xf64>
    stablehlo.return %2, %1 : tensor<f64>, tensor<4xf64>
  } : (tensor<5x2xf64>, tensor<5x3xf64>, tensor<4xf64>) -> (tensor<5xf64>, tensor<4xf64>)
  return %0#0, %0#1 : tensor<5xf64>, tensor<4xf64>
}
// CHECK: func.func private @chlo.scan.impl
