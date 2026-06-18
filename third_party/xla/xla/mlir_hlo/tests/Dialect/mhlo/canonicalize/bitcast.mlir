// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// CHECK-LABEL:@no_layout
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
// CHECK-NOT: bitcast
// CHECK: return %[[ARG0]]
func.func @no_layout(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.bitcast"(%arg) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL:@same_layout
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
// CHECK-NOT: bitcast
// CHECK: return %[[ARG0]]
func.func @same_layout(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.bitcast"(%arg) {
    source_layout = dense<[1, 0]> : tensor<2xindex>,
    result_layout = dense<[1, 0]> : tensor<2xindex>
  }: (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL:@different_layout
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
// CHECK: bitcast
func.func @different_layout(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.bitcast"(%arg) {
    source_layout = dense<[0, 1]> : tensor<2xindex>,
    result_layout = dense<[1, 0]> : tensor<2xindex>
  }: (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL:@source_layout_only
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
// CHECK: bitcast
func.func @source_layout_only(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.bitcast"(%arg) {
    source_layout = dense<[0, 1]> : tensor<2xindex>
  }: (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL:@result_layout_only
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
// CHECK: bitcast
func.func @result_layout_only(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.bitcast"(%arg) {
    result_layout = dense<[1, 0]> : tensor<2xindex>
  }: (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL:@type_cast
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
// CHECK: bitcast
func.func @type_cast(%arg: tensor<2x4xf32>) -> tensor<2x4xi32> {
  %0 = "mhlo.bitcast"(%arg): (tensor<2x4xf32>) -> tensor<2x4xi32>
  func.return %0 : tensor<2x4xi32>
}
