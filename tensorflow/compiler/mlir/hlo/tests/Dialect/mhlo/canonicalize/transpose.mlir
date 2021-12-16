// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='builtin.func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @remove_noop
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @remove_noop(%arg : tensor<2x3x9x5xi32>) -> tensor<2x3x9x5xi32> {
  %0 = "mhlo.transpose"(%arg) {permutation = dense<[0, 1, 2, 3]> : tensor<4xi64>}: (tensor<2x3x9x5xi32>) -> tensor<2x3x9x5xi32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : tensor<2x3x9x5xi32>
}

// -----

// CHECK-LABEL: func @keep_real_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @keep_real_transpose(%arg : tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32> {
  // CHECK-NEXT: "mhlo.transpose"([[ARG]])
  %0 = "mhlo.transpose"(%arg) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}: (tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32>
  return %0 : tensor<3x2x5x9xi32>
}

// -----

// CHECK-LABEL: func @keep_same_shape_real_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @keep_same_shape_real_transpose(%arg : tensor<4x4xi32>) -> tensor<4x4xi32> {
  // CHECK-NEXT: "mhlo.transpose"([[ARG]])
  %0 = "mhlo.transpose"(%arg) {permutation = dense<[1, 0]> : tensor<2xi64>}: (tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: @eliminate_redundant_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @eliminate_redundant_transpose(%arg : tensor<3x4x16x2xf32>) -> tensor<3x2x16x4xf32> {
  %0 = "mhlo.transpose"(%arg) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}: (tensor<3x4x16x2xf32>) -> tensor<3x2x4x16xf32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>}: (tensor<3x2x4x16xf32>) -> tensor<3x2x16x4xf32>
  // CHECK: [[RET:%[a-zA-Z0-9]+]] = "mhlo.transpose"([[ARG]])
  // CHECK-SAME: dense<[0, 3, 2, 1]
  // CHECK-NEXT: return [[RET]]
  return %1 : tensor<3x2x16x4xf32>
}
