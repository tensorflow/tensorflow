// RUN: tf-opt %s -split-input-file -xla-legalize-to-std -canonicalize | FileCheck %s

// CHECK-LABEL: func @const_fold_collapse_to_scalar
func @const_fold_collapse_to_scalar() -> tensor<i32> {
  // CHECK-NEXT: [[CST:%.+]] = constant dense<42> : tensor<i32>
  %cst = constant dense<42> : tensor<1x1xi32>
  %0 = "xla_hlo.reshape"(%cst) : (tensor<1x1xi32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_fold_collapse_to_tensor
func @const_fold_collapse_to_tensor() -> tensor<2xi32> {
  // CHECK-NEXT: [[CST:%.+]] = constant dense<42> : tensor<2xi32>
  %cst = constant dense<42> : tensor<1x2xi32>
  %0 = "xla_hlo.reshape"(%cst) : (tensor<1x2xi32>) -> tensor<2xi32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @const_fold_expand
func @const_fold_expand() -> tensor<1xi32> {
  // CHECK-NEXT: [[CST:%.+]] = constant dense<42> : tensor<1xi32>
  %cst = constant dense<42> : tensor<i32>
  %0 = "xla_hlo.reshape"(%cst) : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<1xi32>
}

// -----

// CHECK-LABEL: func @const_fold_nontrivial
func @const_fold_nontrivial() -> tensor<16xi64> {
  // CHECK-NEXT: [[CST:%.+]] = constant dense<42> : tensor<16xi64>
  %cst = constant dense<42> : tensor<4x4xi64>
  %0 = "xla_hlo.reshape"(%cst) : (tensor<4x4xi64>) -> tensor<16xi64>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<16xi64>
}

// -----

// CHECK-LABEL: func @const_fold_flatten
func @const_fold_flatten() -> tensor<16xi64> {
  // CHECK-NEXT: [[CST:%.+]] = constant dense<42> : tensor<16xi64>
  %cst = constant dense<42> : tensor<4x4xi64>
  %0 = "xla_hlo.reshape"(%cst) : (tensor<4x4xi64>) -> tensor<16xi64>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<16xi64>
}

// -----

// CHECK-LABEL: func @const_fold_6
func @const_fold_6() -> tensor<6xi32> {
  // CHECK-NEXT: [[CST:%.+]] = constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %cst = constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %0 = "xla_hlo.reshape"(%cst) : (tensor<3x2xi32>) -> tensor<6xi32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<6xi32>
}

// -----

// CHECK-LABEL: func @const_fold_same_shape
func @const_fold_same_shape() -> tensor<2x3xi32> {
  // CHECK-NEXT: [[CST:%.+]] = constant dense<[
  // CHECK-SAME:   [1, 2, 3], [4, 5, 6]
  // CHECK-SAME: ]> : tensor<2x3xi32>
  %cst = constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %0 = "xla_hlo.reshape"(%cst) : (tensor<6xi32>) -> tensor<2x3xi32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @const_fold_float
func @const_fold_float() -> tensor<16xf64> {
  // CHECK-NEXT: [[CST:%.+]] = constant dense<4.2{{0*}}e+00> : tensor<16xf64>
  %cst = constant dense<4.2> : tensor<4x4xf64>
  %0 = "xla_hlo.reshape"(%cst) : (tensor<4x4xf64>) -> tensor<16xf64>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<16xf64>
}

// -----

// CHECK-LABEL: func @non_const_same_shape
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @non_const_same_shape(%arg : tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-NEXT: return [[ARG]]
  %0 = "xla_hlo.reshape"(%arg) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}