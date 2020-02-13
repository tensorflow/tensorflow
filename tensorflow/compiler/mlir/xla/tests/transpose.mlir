// RUN: tf-opt %s -split-input-file -pass-pipeline='func(canonicalize)' | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @remove_noop
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @remove_noop(%arg : tensor<2x3x9x5xi32>) -> tensor<2x3x9x5xi32> {
  %0 = "xla_hlo.transpose"(%arg) {permutation = dense<[0, 1, 2, 3]> : tensor<4xi64>}: (tensor<2x3x9x5xi32>) -> tensor<2x3x9x5xi32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : tensor<2x3x9x5xi32>
}

// -----

// CHECK-LABEL: func @keep_real_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @keep_real_transpose(%arg : tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32> {
  // CHECK-NEXT: "xla_hlo.transpose"([[ARG]])
  %0 = "xla_hlo.transpose"(%arg) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}: (tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32>
  return %0 : tensor<3x2x5x9xi32>
}

// -----

// CHECK-LABEL: func @keep_same_shape_real_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @keep_same_shape_real_transpose(%arg : tensor<4x4xi32>) -> tensor<4x4xi32> {
  // CHECK-NEXT: "xla_hlo.transpose"([[ARG]])
  %0 = "xla_hlo.transpose"(%arg) {permutation = dense<[1, 0]> : tensor<2xi64>}: (tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
