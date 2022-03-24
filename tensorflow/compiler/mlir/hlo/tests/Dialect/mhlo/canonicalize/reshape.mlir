// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @const_fold_collapse_to_scalar
func.func @const_fold_collapse_to_scalar() -> tensor<i32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<i32>
  %cst = mhlo.constant dense<42> : tensor<1x1xi32>
  %0 = "mhlo.reshape"(%cst) : (tensor<1x1xi32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_fold_collapse_to_tensor
func.func @const_fold_collapse_to_tensor() -> tensor<2xi32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<2xi32>
  %cst = mhlo.constant dense<42> : tensor<1x2xi32>
  %0 = "mhlo.reshape"(%cst) : (tensor<1x2xi32>) -> tensor<2xi32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @const_fold_expand
func.func @const_fold_expand() -> tensor<1xi32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<1xi32>
  %cst = mhlo.constant dense<42> : tensor<i32>
  %0 = "mhlo.reshape"(%cst) : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<1xi32>
}

// -----

// CHECK-LABEL: func @const_fold_nontrivial
func.func @const_fold_nontrivial() -> tensor<16xi64> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<16xi64>
  %cst = mhlo.constant dense<42> : tensor<4x4xi64>
  %0 = "mhlo.reshape"(%cst) : (tensor<4x4xi64>) -> tensor<16xi64>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<16xi64>
}

// -----

// CHECK-LABEL: func @const_fold_flatten
func.func @const_fold_flatten() -> tensor<16xi64> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<16xi64>
  %cst = mhlo.constant dense<42> : tensor<4x4xi64>
  %0 = "mhlo.reshape"(%cst) : (tensor<4x4xi64>) -> tensor<16xi64>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<16xi64>
}

// -----

// CHECK-LABEL: func @const_fold_6
func.func @const_fold_6() -> tensor<6xi32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %cst = mhlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %0 = "mhlo.reshape"(%cst) : (tensor<3x2xi32>) -> tensor<6xi32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<6xi32>
}

// -----

// CHECK-LABEL: func @const_fold_same_shape
func.func @const_fold_same_shape() -> tensor<2x3xi32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<[
  // CHECK-SAME:   [1, 2, 3], [4, 5, 6]
  // CHECK-SAME: ]> : tensor<2x3xi32>
  %cst = mhlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %0 = "mhlo.reshape"(%cst) : (tensor<6xi32>) -> tensor<2x3xi32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @const_fold_float
func.func @const_fold_float() -> tensor<16xf64> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<4.2{{0*}}e+00> : tensor<16xf64>
  %cst = mhlo.constant dense<4.2> : tensor<4x4xf64>
  %0 = "mhlo.reshape"(%cst) : (tensor<4x4xf64>) -> tensor<16xf64>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<16xf64>
}

// -----

// CHECK-LABEL: func @non_const_same_shape
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @non_const_same_shape(%arg : tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-NEXT: return [[ARG]]
  %0 = "mhlo.reshape"(%arg) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @non_const_chained_reshape
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @non_const_chained_reshape(%arg : tensor<2x3xi32>) -> (tensor<3x2xi32>, tensor<6xi32>) {
  // CHECK-NEXT: "mhlo.reshape"([[ARG]]) : (tensor<2x3xi32>) -> tensor<3x2xi32>
  // CHECK-NEXT: "mhlo.reshape"([[ARG]]) : (tensor<2x3xi32>) -> tensor<6xi32>
  %0 = "mhlo.reshape"(%arg) : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %1 = "mhlo.reshape"(%0) : (tensor<3x2xi32>) -> tensor<6xi32>
  func.return %0, %1 : tensor<3x2xi32>, tensor<6xi32> // return both so nothing is removed
}

// -----

// CHECK-LABEL: func @non_const_chained_reshape_unused_parent
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @non_const_chained_reshape_unused_parent(%arg : tensor<2x3xi32>) -> tensor<6xi32> {
  // CHECK-NEXT: [[RES:%.+]] = "mhlo.reshape"([[ARG]]) : (tensor<2x3xi32>) -> tensor<6xi32>
  %0 = "mhlo.reshape"(%arg) : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %1 = "mhlo.reshape"(%0) : (tensor<3x2xi32>) -> tensor<6xi32>
  // CHECK-NEXT: return [[RES]]
  func.return %1 : tensor<6xi32>
}

// -----

// CHECK-LABEL: func @non_const_chained_reshape_becomes_noop
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @non_const_chained_reshape_becomes_noop(%arg : tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "mhlo.reshape"(%arg) : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %1 = "mhlo.reshape"(%0) : (tensor<3x2xi32>) -> tensor<2x3xi32>
  // CHECK-NEXT: return [[ARG]]
  func.return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @non_const_many_chained_reshapes
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @non_const_many_chained_reshapes(%arg : tensor<2x3x4xi32>) -> tensor<1x2x4x3xi32> {
  // CHECK-NEXT: [[RES:%.+]] = "mhlo.reshape"([[ARG]]) : (tensor<2x3x4xi32>) -> tensor<1x2x4x3xi32>
  %0 = "mhlo.reshape"(%arg) : (tensor<2x3x4xi32>) -> tensor<4x3x2xi32>
  %1 = "mhlo.reshape"(%0) : (tensor<4x3x2xi32>) -> tensor<12x2xi32>
  %2 = "mhlo.reshape"(%1) : (tensor<12x2xi32>) -> tensor<2x12xi32>
  %3 = "mhlo.reshape"(%2) : (tensor<2x12xi32>) -> tensor<24xi32>
  %4 = "mhlo.reshape"(%3) : (tensor<24xi32>) -> tensor<1x2x4x3xi32>
  // CHECK-NEXT: return [[RES]]
  func.return %4 : tensor<1x2x4x3xi32>
}
