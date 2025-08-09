// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-preserve-high-level-ops=use-custom-call-encoding=false -split-input-file | FileCheck %s
// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-preserve-high-level-ops=use-custom-call-encoding=true -split-input-file | FileCheck %s --check-prefixes=CHECK-CC
// RUN: mlir-hlo-opt %s > %t.0
// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-preserve-high-level-ops=use-custom-call-encoding=false | mlir-hlo-opt --stablehlo-ext-chlo-recompose-ops -symbol-dce > %t.1
// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-preserve-high-level-ops=use-custom-call-encoding=true | mlir-hlo-opt --stablehlo-ext-chlo-recompose-ops -symbol-dce > %t.2
// RUN: diff %t.0 %t.1
// RUN: diff %t.0 %t.2


// CHECK-LABEL: func @ragged_dot_to_composite
func.func @ragged_dot_to_composite(%lhs : tensor<2x11x5xf32>, %rhs : tensor<3x2x5x7xf32>, %group_sizes : tensor<3xi64>) -> (tensor<2x11x7xf32>) {
  // CHECK: stablehlo.composite "chlo.ragged_dot"
  // CHECK-SAME{LITERAL}: {composite_attributes = {precision_config = ["DEFAULT", "DEFAULT"]
  // CHECK-SAME: ragged_dot_dimension_numbers = [dense<0>{{.*}}, dense<1>{{.*}}, dense<2>{{.*}}, dense<2>{{.*}}, dense<1>{{.*}}, dense<0>{{.*}}]
  // CHECK-CC{LITERAL}: stablehlo.custom_call @chlo.ragged_dot(%arg0, %arg1, %arg2) {mhlo.attributes = {precision_config = ["DEFAULT", "DEFAULT"], ragged_dot_dimension_numbers = [dense<0>
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
  // CHECK-CC: stablehlo.custom_call @mhlo.topk(%arg0) {mhlo.attributes = {k = 4 : i64, largest = true}, mhlo.version = 1 : i64} : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  // CHECK: stablehlo.composite "chlo.top_k" %arg0 {composite_attributes = {k = 4 : i64, largest = true}, decomposition = @chlo.top_k.impl, version = 1 : i32}
  %values, %indices = chlo.top_k(%arg0, k = 4) {largest = true} : tensor<5x16xf32> -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %values, %indices : tensor<?x?xf32>, tensor<?x?xi32>
}


// -----

// CHECK-LABEL: func @erf_preserve
func.func @erf_preserve(%arg0: tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16> {
  // CHECK-CC: stablehlo.custom_call @mhlo.erf(%arg0) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<3x20x20xbf16>) -> tensor<?x20x20xbf16>
  // CHECK: stablehlo.composite "chlo.erf" %arg0 {decomposition = @chlo.erf.impl, version = 1 : i32}
  %0 = chlo.erf %arg0 : tensor<3x20x20xbf16> -> tensor<?x20x20xbf16>
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
  // CHECK-CC: stablehlo.custom_call @mhlo.topk(%arg0) {mhlo.attributes = {k = 4 : i64, largest = true, mhlo.frontend_attributes = {foo = "true"}}, mhlo.version = 1 : i64} : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  // CHECK: stablehlo.composite "chlo.top_k" %arg0 {composite_attributes = {k = 4 : i64, largest = true, mhlo.frontend_attributes = {foo = "true"}}, decomposition = @chlo.top_k.impl, version = 1 : i32}
  %values, %indices = chlo.top_k(%arg0, k = 4) {largest = true, mhlo.frontend_attributes = {foo = "true"}} : tensor<5x16xf32> -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %values, %indices : tensor<?x?xf32>, tensor<?x?xi32>
}
